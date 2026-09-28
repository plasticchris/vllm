# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE with MXFP4 weights and BF16/FP16 activations, in Triton.

Expert weights stay packed (two E2M1 values per byte along K, one E8M0 scale
per 32 K elements) and are decoded tile by tile inside the GEMM, so no
full-precision copy of the experts is materialised. Targets GPUs without
native MX support, such as RDNA3 (gfx1100).

With few rows per expert the GEMMs are GEMVs bound by weight bandwidth, so a
dot-free path decodes int32 weight words with a v_perm_b32 byte LUT and
multiplies with v_dot2_f32_bf16. Larger batches decode to bf16 tiles for
tl.dot.
"""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MXFP4_BLOCK = 32


@triton.jit
def _e8m0_bf16_bias(scale):
    # bf16 exponent field of 2^(scale - 127) for an E2M1 value with e=1.
    # Clamped to 2^[-125, 125] so every scaled E2M1 value is a normal bf16.
    return (tl.minimum(tl.maximum(scale, 2), 252) - 1) << 7


@triton.jit
def _e2m1_to_bf16(nib, sbias):
    em = nib & 7
    mag = tl.where(em >= 2, (em << 6) + sbias, tl.where(em == 1, sbias, 0))
    bits = mag | ((nib & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


# One int32 word (bytes b0..b3) -> unscaled bf16 pairs. LO gives
# $0 = [lo(b0), lo(b1)], $1 = [lo(b2), lo(b3)]; HI the same for high nibbles.
# The bf16 high/low bytes of E2M1 magnitudes 0..7 are 00 3F 3F 3F 40 40 40 40
# and 00 00 80 C0 00 40 80 C0; $5/$6 carry the lower table halves because a
# VOP3 instruction takes one literal. $2/$3 are scratch.
_LUT_PAIRS = "v_perm_b32 $1, $3, $0, 0x07030602\nv_perm_b32 $0, $3, $0, 0x05010400"
_LUT_LO = tl.constexpr(
    "v_and_b32 $2, 0x07070707, $4\n"
    "v_perm_b32 $3, 0x40404040, $5, $2\n"
    "v_perm_b32 $0, 0xC0804000, $6, $2\n"
    "v_lshlrev_b32 $2, 4, $4\n"
    "v_and_or_b32 $3, $2, 0x80808080, $3\n" + _LUT_PAIRS
)
_LUT_HI = tl.constexpr(
    "v_lshrrev_b32 $2, 4, $4\n"
    "v_and_b32 $2, 0x07070707, $2\n"
    "v_perm_b32 $3, 0x40404040, $5, $2\n"
    "v_perm_b32 $0, 0xC0804000, $6, $2\n"
    "v_and_or_b32 $3, $4, 0x80808080, $3\n" + _LUT_PAIRS
)
_LUT_CONSTRAINTS = tl.constexpr("=&v,=&v,=&v,=&v,v,s,s")


@triton.jit
def _e2m1_lut_words(w, t_hi, t_lo):
    """int32 words of 8 E2M1 -> four int32 tensors of unscaled bf16 pairs:
    elements (0, 2), (4, 6), (1, 3), (5, 7) of each word."""
    i32 = tl.int32
    lo0, lo1, _, _ = tl.inline_asm_elementwise(
        _LUT_LO,
        _LUT_CONSTRAINTS,
        [w, t_hi, t_lo],
        dtype=(i32, i32, i32, i32),
        is_pure=True,
        pack=1,
    )
    hi0, hi1, _, _ = tl.inline_asm_elementwise(
        _LUT_HI,
        _LUT_CONSTRAINTS,
        [w, t_hi, t_lo],
        dtype=(i32, i32, i32, i32),
        is_pure=True,
        pack=1,
    )
    return lo0, lo1, hi0, hi1


@triton.jit
def _dot2_bf16(w, a, acc):
    return tl.inline_asm_elementwise(
        "v_dot2_f32_bf16 $0, $1, $2, $3",
        "=v,v,v,v",
        [w, a, acc],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _activation_pairs(p0, p1, p2, p3):
    # Words p0..p3 hold activation elements (0,1) (2,3) (4,5) (6,7); regroup
    # them to match the pairs of _e2m1_lut_words.
    p0 = p0.to(tl.uint32, bitcast=True)
    p1 = p1.to(tl.uint32, bitcast=True)
    p2 = p2.to(tl.uint32, bitcast=True)
    p3 = p3.to(tl.uint32, bitcast=True)
    a0 = (p0 & 0xFFFF) | (p1 << 16)
    a1 = (p2 & 0xFFFF) | (p3 << 16)
    a2 = (p0 >> 16) | (p1 & 0xFFFF0000)
    a3 = (p2 >> 16) | (p3 & 0xFFFF0000)
    i32 = tl.int32
    return (
        a0.to(i32, bitcast=True),
        a1.to(i32, bitcast=True),
        a2.to(i32, bitcast=True),
        a3.to(i32, bitcast=True),
    )


@triton.jit
def fused_moe_kernel_mxfp4_w4a16(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    EVEN_K: tl.constexpr,
    GEMV: tl.constexpr,
):
    """C[t, :] = A[t // top_k, :] @ dequant(B[e]).T for each routed row t.

    A is [M, K]; B is [E, N, K // 2] uint8 with element 2i in the low nibble
    of byte i; B_scale is [E, N, K // 32] uint8 E8M0. Rows are visited in the
    order produced by moe_align_block_size, as in fused_moe_kernel_gptq_awq.
    GEMV requires bf16 activations with even, 4-byte aligned rows.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Expert lives on another EP rank.
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    s_base = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn

    if GEMV:
        # Each int32 word holds 8 consecutive K elements; its 4 words of a
        # 32-group share one scale, applied in fp32 after v_dot2.
        BKW: tl.constexpr = BLOCK_SIZE_K // 8
        offs_kw = tl.arange(0, BKW)
        w_ptrs = (
            b_ptr.to(tl.pointer_type(tl.int32))
            + off_experts * (stride_be // 4)
            + offs_bn[None, :] * (stride_bn // 4)
            + offs_kw[:, None]
        )
        s_ptrs = s_base + (offs_kw[:, None] // 4) * stride_bsk
        aw_ptr = a_ptr.to(tl.pointer_type(tl.int32))
        t_hi = tl.full((BKW, BLOCK_SIZE_N), 0x3F3F3F00, tl.int32)
        t_lo = tl.full((BKW, BLOCK_SIZE_N), -1065353216, tl.int32)  # 0xC0800000
        if BLOCK_SIZE_M == 1:
            tok = tl.load(sorted_token_ids_ptr + pid_m)
            row = tl.where(tok < num_valid_tokens, tok // top_k, 0).to(tl.int64)
            a_ptrs = aw_ptr + row * (stride_am // 2) + 4 * offs_kw
            acc = tl.zeros((BKW, BLOCK_SIZE_N), dtype=tl.float32)
        else:
            rows = tl.where(token_mask, offs_token // top_k, 0)
            a_ptrs = aw_ptr + rows[:, None] * (stride_am // 2) + 4 * offs_kw[None, :]
            acc = tl.zeros((BLOCK_SIZE_M, BKW, BLOCK_SIZE_N), dtype=tl.float32)
        zero = tl.zeros(acc.shape, dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                w = tl.load(w_ptrs)
                s = tl.load(s_ptrs)
                p0 = tl.load(a_ptrs)
                p1 = tl.load(a_ptrs + 1)
                p2 = tl.load(a_ptrs + 2)
                p3 = tl.load(a_ptrs + 3)
            else:
                kw_mask = offs_kw < (K - k * BLOCK_SIZE_K) // 8
                a_mask = kw_mask if BLOCK_SIZE_M == 1 else kw_mask[None, :]
                w = tl.load(w_ptrs, mask=kw_mask[:, None], other=0)
                s = tl.load(s_ptrs, mask=kw_mask[:, None], other=0)
                p0 = tl.load(a_ptrs, mask=a_mask, other=0)
                p1 = tl.load(a_ptrs + 1, mask=a_mask, other=0)
                p2 = tl.load(a_ptrs + 2, mask=a_mask, other=0)
                p3 = tl.load(a_ptrs + 3, mask=a_mask, other=0)
            s = tl.minimum(tl.maximum(s.to(tl.int32), 1), 254)
            scale = (s << 23).to(tl.float32, bitcast=True)
            w0, w1, w2, w3 = _e2m1_lut_words(w, t_hi, t_lo)
            a0, a1, a2, a3 = _activation_pairs(p0, p1, p2, p3)
            if BLOCK_SIZE_M == 1:
                t = _dot2_bf16(w0, a0[:, None], zero)
                t = _dot2_bf16(w1, a1[:, None], t)
                t = _dot2_bf16(w2, a2[:, None], t)
                t = _dot2_bf16(w3, a3[:, None], t)
                acc += t * scale
            else:
                t = _dot2_bf16(w0[None, :, :], a0[:, :, None], zero)
                t = _dot2_bf16(w1[None, :, :], a1[:, :, None], t)
                t = _dot2_bf16(w2[None, :, :], a2[:, :, None], t)
                t = _dot2_bf16(w3[None, :, :], a3[:, :, None], t)
                acc += t * scale[None, :, :]
            w_ptrs += BKW
            s_ptrs += (BLOCK_SIZE_K // 32) * stride_bsk
            a_ptrs += BLOCK_SIZE_K // 2
        if BLOCK_SIZE_M == 1:
            accumulator = tl.sum(acc, axis=0)[None, :]
        else:
            accumulator = tl.sum(acc, axis=1)
    else:
        BKP: tl.constexpr = BLOCK_SIZE_K // 2
        NG: tl.constexpr = BLOCK_SIZE_K // 32
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_kp = tl.arange(0, BKP)
        offs_g = tl.arange(0, NG)
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[None, :] * stride_bn
            + offs_kp[:, None] * stride_bk
        )
        s_ptrs = s_base + offs_g[:, None] * stride_bsk
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
                s = tl.load(s_ptrs)
            else:
                k_rem = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_rem),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_kp[:, None] < k_rem // 2, other=0)
                s = tl.load(s_ptrs, mask=offs_g[:, None] < k_rem // 32, other=0)
            # One scale per 32-group, broadcast over its 16 packed rows.
            sbias = _e8m0_bf16_bias(s.to(tl.int32))
            sbias = tl.reshape(
                tl.broadcast_to(sbias[:, None, :], (NG, 16, BLOCK_SIZE_N)),
                (BKP, BLOCK_SIZE_N),
            )
            b = b.to(tl.int32)
            lo = _e2m1_to_bf16(b & 0xF, sbias)
            hi = _e2m1_to_bf16(b >> 4, sbias)
            w = tl.reshape(
                tl.permute(tl.join(lo, hi), (0, 2, 1)), (BLOCK_SIZE_K, BLOCK_SIZE_N)
            )
            accumulator = tl.dot(a, w.to(compute_type), acc=accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BKP * stride_bk
            s_ptrs += NG * stride_bsk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)


def get_mxfp4_w4a16_moe_configs(
    M: int, num_experts: int, K1: int, K2: int, top_k: int, dtype: torch.dtype
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Launch configs for the two GEMMs (K1 = hidden, K2 = intermediate).

    Both share BLOCK_SIZE_M so one moe_align_block_size serves both. Tuned
    on a 7900 XTX at MiMo-V2.6 TP4/EP4 expert shapes.
    """
    rows_per_expert = M * top_k / num_experts
    if dtype == torch.bfloat16 and rows_per_expert <= 4:
        block_m = 1 if rows_per_expert <= 1 else 2

        def gemv(K: int) -> dict[str, Any]:
            block_k = 512 if K >= 2048 else 256 if block_m == 1 else 128
            return {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": 1,
                "GEMV": True,
                "num_warps": 4 if block_m > 1 and block_k == 512 else 1,
                "num_stages": 2,
            }

        return gemv(K1), gemv(K2)

    if rows_per_expert >= 64:
        block_m = 64
    elif rows_per_expert > 4:
        block_m = 32
    else:
        block_m = 16

    def dot(K: int) -> dict[str, Any]:
        if block_m == 64:
            block_n, block_k, warps = 64, 64, 4
        elif K >= 2048:
            block_n, block_k, warps = 64, 128, 8
        else:
            block_n, block_k, warps = 128, 64, 4
        return {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 1,
            "GEMV": False,
            "num_warps": warps,
            "num_stages": 2,
        }

    return dot(K1), dot(K2)


def invoke_fused_moe_mxfp4_w4a16_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    assert B.dtype == torch.uint8 and B_scale.dtype == torch.uint8
    assert A.size(1) == 2 * B.size(2) == MXFP4_BLOCK * B_scale.size(2)
    assert config["BLOCK_SIZE_K"] % MXFP4_BLOCK == 0
    if config["GEMV"]:
        assert A.dtype == torch.bfloat16 and A.stride(1) == 1 and B.stride(2) == 1
        assert A.stride(0) % 2 == 0 and A.data_ptr() % 4 == 0
        assert B.stride(0) % 4 == 0 and B.stride(1) % 4 == 0 and B.data_ptr() % 4 == 0

    num_tokens = A.size(0)
    EM = sorted_token_ids.size(0)
    if num_tokens < config["BLOCK_SIZE_M"]:
        # Each token's experts are distinct, so at most M * top_k blocks are live.
        EM = min(EM, num_tokens * top_k * config["BLOCK_SIZE_M"])
    N = B.size(1)
    K = A.size(1)
    grid = (
        triton.cdiv(EM, config["BLOCK_SIZE_M"])
        * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    fused_moe_kernel_mxfp4_w4a16[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(-2),
        C.stride(-1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        EVEN_K=K % config["BLOCK_SIZE_K"] == 0,
        **config,
    )


class TritonMxfp4W4A16Experts(TritonExperts):
    """MXFP4-weight experts that decode the weights inside the Triton GEMM."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        if quant_config.w1_bias is not None or quant_config.w2_bias is not None:
            raise NotImplementedError(
                "TritonMxfp4W4A16Experts does not support expert biases."
            )

    @staticmethod
    def supports_lora() -> bool:
        return False

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx11

        # The GEMV path's inline asm is RDNA3 ISA.
        return on_gfx11()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key is None

    @staticmethod
    def _supports_batch_invariance():
        return False

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert hidden_states.dim() == 2 and hidden_states.is_contiguous()
        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        else:
            raise ValueError(f"Unsupported activation dtype {hidden_states.dtype}")

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        num_align_experts = w1.shape[0] if expert_map is None else global_num_experts
        config1, config2 = get_mxfp4_w4a16_moe_configs(
            num_tokens,
            num_align_experts,
            K,
            activation_out_dim,
            top_k_num,
            hidden_states.dtype,
        )

        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config1["BLOCK_SIZE_M"], num_align_experts, expert_map
        )

        invoke_fused_moe_mxfp4_w4a16_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            top_k_num,
            config1,
            compute_type,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        invoke_fused_moe_mxfp4_w4a16_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config2,
            compute_type,
        )

        self.moe_sum(intermediate_cache3, output)
