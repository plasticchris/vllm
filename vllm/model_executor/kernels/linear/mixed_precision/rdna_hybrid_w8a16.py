# SPDX-License-Identifier: Apache-2-0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W8A16 kernel: Triton for prefill, HIP skinny for decode.

The W8A16 rung of the quality/speed tradeoff on gfx1100 (RDNA3): INT8
weights (one signed byte per value), BF16/FP16 activations, group-wise
scales. Decode is 1.3-1.9x faster than the bf16 skinny path (measured
2026-10-25, /tmp/int8_bench3.log) at ~5x lower weight KLD vs BF16 than the
official FP8 quantization (lued/Qwen3.8-27B-INT8-W8A16-MTP, KLD 0.000894
vs 0.004396 nats/token).

Routes based on batch size M:
  M <= MAX_SKINNY_BATCH_SIZE: HIP skinny GEMM (wvSplitK_int8_g)
  M >  MAX_SKINNY_BATCH_SIZE: Triton W8A16 fused dequant GEMM

Stores the weights ONCE as int8 [N, K] (compressed-tensors pack-quantized:
4 int8 per int32, viewed back to int8). The dequant is exact (an 8-bit
integer is exactly representable in bf16/fp16), so there is no LUT and no
subnormal/NaN edge cases — only value * scale in the GEMM.
"""

import torch
from contextlib import nullcontext

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.parameter import permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

SUPPORTED_GROUP_SIZES = [32, 64, 128]
MAX_SKINNY_BATCH_SIZE = 8
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


# ---------------------------------------------------------------------------
# Triton kernel for the prefill path: weights int8 [N, K], scales f32 [N, K/G]
# ---------------------------------------------------------------------------


@triton.jit
def _triton_w8a16_gemm_kernel(
    a_ptr,  # [M, K] bf16/fp16 activations
    b_ptr,  # [N, K] int8 weights
    scales_ptr,  # [N, num_groups] fp32
    c_ptr,  # [M, N] output
    M,
    N,
    K,
    num_groups,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
        mask_b = (offs_n[:, None] < N) & mask_k[None, :]
        # exact: int8 -> f32 carries no error
        b = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.float32)

        g_idx = (k_start * BLOCK_K) // group_size
        scale_ptrs = scales_ptr + offs_n * num_groups + g_idx
        scale_mask = offs_n < N
        scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

        b_fp = (b * scales[:, None]).to(a_ptr.type.element_ty)
        b_fp_t = tl.trans(b_fp)
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w8a16_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Prefill W8A16 GEMM: x @ (dequant(W)*scales).T. Weights int8 [N, K]."""
    assert a.is_contiguous(), "Activation must be contiguous"
    assert b_q.is_contiguous(), "Weights must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[0]
    num_groups = K // group_size

    assert b_q.shape == (N, K), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K})"
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # gfx1100 prefill heuristic, tuned 2026-10-25 (/tmp/pf_tune.py, GPU3):
    #   M<=128: legacy W4A16-style config (0.81-1.00x bf16 GEMM, unchanged)
    #   128<M<=1024: BM64/BN128/BK32 w8 s3 -> 0.84-1.14x bf16 (old config was
    #                1.2-1.5x; this is the prod prefill-chunk range)
    #   M>1024: BM128/BN128/BK32 w4 s2 (tuned config regresses at M~2832:
    #           1.44x vs 1.28x bf16; huge chunks are rare)
    if M <= 128:
        num_warps = 4
        if K >= N:  # tall K (down_proj)
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
        num_stages = 2
    elif M <= 1024:
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 64, 128, 32, 8, 3
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 128, 128, 32, 4, 2

    BLOCK_K = min(BLOCK_K, group_size)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _triton_w8a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        c,
        M,
        N,
        K,
        num_groups,
        group_size=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# ---------------------------------------------------------------------------
# Hybrid dispatch (custom op for torch.compile compatibility)
# ---------------------------------------------------------------------------


def _rdna_hybrid_w8a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M."""
    import vllm._custom_ops as ops

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = w_q.shape[0]

    # The HIP skinny op is registered by the _rocm_C build (skinny_gemms_int8.cu).
    # If this copy of vllm was built without it, fall back to the numerically
    # identical Triton path (slower decode, same math).
    has_skinny = getattr(
        getattr(torch.ops, "_rocm_C", None), "wvSplitK_int8_g", None
    ) is not None

    if has_skinny and M <= MAX_SKINNY_BATCH_SIZE and K * M <= 3 * LDS_CAPACITY_ELEMENTS:
        cctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"wvsplitk_int8 {M}x{N}x{K}")
        )
        with cctx:
            return ops.wvSplitK_int8_g(w_q, x_2d, w_s, cu_count, group_size, bias)

    cctx = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch.profiler.record_function(f"hybrid_triton_w8a16 {M}x{N}x{K}")
    )
    with cctx:
        output = triton_w8a16_gemm(a=x_2d, b_q=w_q, scales=w_s, group_size=group_size)
        if bias is not None:
            output.add_(bias)
    return output

def _rdna_hybrid_w8a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    M = x_2d.size(0)
    N = w_q.size(0)
    return torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)


direct_register_custom_op(
    op_name="rdna_hybrid_w8a16_apply",
    op_func=_rdna_hybrid_w8a16_apply_impl,
    mutates_args=[],
    fake_impl=_rdna_hybrid_w8a16_apply_fake,
)


# ---------------------------------------------------------------------------
# MPLinearKernel integration (compressed-tensors W8A16 / pack-quantized)
# ---------------------------------------------------------------------------


class RDNAHybridW8A16LinearKernel(MPLinearKernel):
    """Hybrid W8A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores the weights ONCE as int8 [N, K] (compressed-tensors pack-quantized:
    4 int8 per int32, viewed back to int8). Both paths read this single
    buffer. The int8 -> bf16/fp16 dequant is exact, so no LUT is needed.
    """

    SUPPORTED_QUANT_TYPES = [scalar_types.uint8b128]  # symmetric int8

    @classmethod
    def get_min_capability(cls) -> int:
        # Arch filtering is handled by can_implement (_on_gfx1x check)
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNAHybridW8A16LinearKernel only targets ROCm"

        from vllm.platforms.rocm import on_gfx1x

        if not on_gfx1x():
            return False, "RDNAHybridW8A16LinearKernel only targets gfx11/gfx12"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if getattr(c, "has_g_idx", False):
            return False, "does not support g_idx reordering"

        gs = c.group_size
        if gs not in SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size {gs} not supported; supported: {SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        if K % gs != 0:
            return (
                False,
                f"K={K} must be divisible by group_size={gs}",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        w_q_raw = getattr(layer, self.w_q_name)
        w_s_raw = getattr(layer, self.w_s_name)

        # compressed-tensors pack-quantized int8: 4 int8 per int32. For 8-bit
        # (pack_factor=4) vLLM's unpack == a plain .view(int8) == natural K
        # order (verified: unpack(arange)==arange), so no de-interleave — the
        # kernel reads k, k+1, k+2, ... sequentially, exactly as stored.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            w_q_skinny = w_q_raw.data.t().contiguous().view(torch.int8)
        else:
            w_q_skinny = w_q_raw.data.view(torch.int8).contiguous()

        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        # The HIP skinny kernel takes fp32 scales; the Triton path is f32 too.
        w_s_skinny = w_s_raw.data.contiguous().to(torch.float32)

        self._transform_param(layer, self.w_q_name, lambda x: w_q_skinny)
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q, w_s, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        N = w_q.shape[0]
        out_shape = x.shape[:-1] + (N,)

        cu_count = num_compute_units()
        output = torch.ops.vllm.rdna_hybrid_w8a16_apply(
            x_2d,
            w_q,
            w_s,
            bias,
            cu_count,
            c.group_size,
        )
        return output.reshape(out_shape)
