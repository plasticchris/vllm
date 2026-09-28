# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton MXFP4 W4A16 fused MoE experts.

Run `pytest tests/kernels/moe/test_triton_mxfp4_moe.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.triton_mxfp4_moe import (
    TritonMxfp4W4A16Experts,
    _e2m1_lut_words,
    _e2m1_to_bf16,
    _e8m0_bf16_bias,
    invoke_fused_moe_mxfp4_w4a16_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="TritonMxfp4W4A16Experts is ROCm only."
)

device = "cuda"

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _dequant_ref(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[N, K // 2] uint8 + [N, K // 32] E8M0 -> [N, K] float32."""
    lut = E2M1.to(w.device)
    vals = torch.stack((lut[(w & 0xF).long()], lut[(w >> 4).long()]), dim=-1)
    scale = torch.exp2(s.float() - 127).repeat_interleave(32, dim=-1)
    return vals.flatten(-2) * scale


@triton.jit
def _decode_kernel(nib_ptr, scale_ptr, out_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    sbias = _e8m0_bf16_bias(tl.load(scale_ptr + offs))
    tl.store(out_ptr + offs, _e2m1_to_bf16(tl.load(nib_ptr + offs), sbias))


@triton.jit
def _lut_kernel(w_ptr, out_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    w = tl.load(w_ptr + offs)
    t_hi = tl.full((n,), 0x3F3F3F00, tl.int32)
    t_lo = tl.full((n,), -1065353216, tl.int32)
    p0, p1, p2, p3 = _e2m1_lut_words(w, t_hi, t_lo)
    tl.store(out_ptr + 4 * offs, p0)
    tl.store(out_ptr + 4 * offs + 1, p1)
    tl.store(out_ptr + 4 * offs + 2, p2)
    tl.store(out_ptr + 4 * offs + 3, p3)


def test_lut_decode_every_byte():
    # Every byte value in every byte lane of a word.
    b = torch.arange(256, dtype=torch.uint8)
    words = torch.stack([b, b.roll(1), b.roll(2), b.roll(3)], dim=-1)
    out = torch.empty(256, 4, dtype=torch.int32, device=device)
    _lut_kernel[(1,)](words.view(torch.int32).flatten().to(device), out, n=256)

    lo, hi = E2M1[(words & 0xF).long()], E2M1[(words >> 4).long()]
    # Pairs are elements (0, 2), (4, 6), (1, 3), (5, 7) of the 8 in a word.
    pairs = torch.stack(
        [
            torch.stack([lo[:, 0], lo[:, 1]], -1),
            torch.stack([lo[:, 2], lo[:, 3]], -1),
            torch.stack([hi[:, 0], hi[:, 1]], -1),
            torch.stack([hi[:, 2], hi[:, 3]], -1),
        ],
        dim=1,
    )
    got = out.cpu().view(torch.int16).view(256, 4, 2)
    assert torch.equal(got, pairs.to(torch.bfloat16).view(torch.int16))


def test_decode_every_nibble_and_scale():
    nib = torch.arange(16, dtype=torch.int32).repeat(256)
    scale = torch.arange(256, dtype=torch.int32).repeat_interleave(16)
    out = torch.empty(nib.numel(), dtype=torch.bfloat16, device=device)
    _decode_kernel[(1,)](nib.to(device), scale.to(device), out, n=nib.numel())

    # Exact for every scale the decoder does not clamp.
    ref = E2M1[nib.long()] * torch.exp2(scale.clamp(2, 252).float() - 127)
    assert torch.equal(
        out.cpu().view(torch.int16), ref.to(torch.bfloat16).view(torch.int16)
    )


def _make_moe_config(E, K, inter, top_k, num_local_experts=None):
    return FusedMoEConfig(
        num_experts=E,
        experts_per_token=top_k,
        hidden_dim=K,
        intermediate_size=inter,
        num_local_experts=num_local_experts or E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


def _make_experts(E, K, inter, top_k, w13_scale, w2_scale, num_local_experts=None):
    moe_config = _make_moe_config(E, K, inter, top_k, num_local_experts)
    quant_config = mxfp4_w4a16_moe_quant_config(w1_scale=w13_scale, w2_scale=w2_scale)
    return TritonMxfp4W4A16Experts(moe_config, quant_config)


def _make_weights(E, K, inter, gen):
    # Scales 2^-8..2^-4 keep activations in a realistic range.
    def rand(n, k):
        w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, generator=gen)
        s = torch.randint(119, 124, (E, n, k // 32), dtype=torch.uint8, generator=gen)
        return w.to(device), s.to(device)

    return rand(2 * inter, K) + rand(K, inter)


def _routing(M, E, top_k, gen):
    topk_ids = torch.rand(M, E, generator=gen).topk(top_k, dim=-1).indices
    topk_weights = torch.rand(M, top_k, generator=gen).softmax(dim=-1)
    return topk_weights.to(device), topk_ids.to(torch.int32).to(device)


def _run(experts, x, w13, w2, topk_weights, topk_ids, E, expert_map=None):
    M, K = x.shape
    top_k = topk_ids.size(1)
    ws1, ws2, out_shape = experts.workspace_shapes(
        M, w13.size(1), K, top_k, E, w13.size(0), None, MoEActivation.SILU
    )
    new = lambda shape: torch.empty(shape, dtype=x.dtype, device=device)  # noqa: E731
    out = new(out_shape)
    experts.apply(
        output=out,
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=new(ws1),
        workspace2=new(ws2),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return out


def _moe_ref(x, w13, s13, w2, s2, topk_weights, topk_ids, expert_map=None):
    """float32 MoE that rounds to bf16 wherever the kernel path does."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=device)
    for e in topk_ids.unique().tolist():
        local = e if expert_map is None else expert_map[e].item()
        if local < 0:
            continue
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        h = (x[tok].float() @ _dequant_ref(w13[local], s13[local]).T).bfloat16()
        gate, up = h.float().chunk(2, dim=-1)
        act = (F.silu(gate) * up).bfloat16().float()
        y = (act @ _dequant_ref(w2[local], s2[local]).T) * topk_weights[tok, slot, None]
        out.index_add_(0, tok, y.bfloat16().float())
    return out


def _assert_close(out, ref):
    out = out.float()
    rel_l2 = (torch.norm(out - ref) / torch.norm(ref)).item()
    max_err = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel_l2 < 5e-3 and max_err < 2e-2, f"rel_l2={rel_l2:.2e} max={max_err:.2e}"


def _gemm_config(gemv, bm, bn, bk, warps):
    return pytest.param(
        dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=1,
            GEMV=gemv,
            num_warps=warps,
            num_stages=2,
        ),
        id=f"{'gemv' if gemv else 'dot'}-m{bm}-n{bn}-k{bk}",
    )


GEMM_CONFIGS = [
    _gemm_config(True, 1, 16, 128, 1),
    _gemm_config(True, 1, 16, 512, 4),
    _gemm_config(True, 2, 16, 256, 2),
    _gemm_config(True, 4, 32, 128, 1),
    _gemm_config(False, 16, 64, 64, 4),
    _gemm_config(False, 32, 64, 128, 8),
    _gemm_config(False, 64, 64, 64, 4),
]


@pytest.mark.parametrize("config", GEMM_CONFIGS)
@pytest.mark.parametrize("N, K", [(128, 512), (96, 2080)])
def test_gemm_paths(config, N, K):
    E, M, top_k = 8, 37, 2
    gen = torch.Generator().manual_seed(N + K)
    w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=gen)
    s = torch.randint(119, 124, (E, N, K // 32), dtype=torch.uint8, generator=gen)
    w, s = w.to(device), s.to(device)
    x = torch.randn(M, K, generator=gen).bfloat16().to(device)
    _, topk_ids = _routing(M, E, top_k, gen)

    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E, None
    )
    out = torch.empty(M, top_k, N, dtype=torch.bfloat16, device=device)
    invoke_fused_moe_mxfp4_w4a16_kernel(
        x, w, out, s, None, sorted_ids, expert_ids, num_post_padded,
        False, top_k, config, tl.bfloat16,
    )  # fmt: skip

    per_expert = torch.stack([x.float() @ _dequant_ref(w[e], s[e]).T for e in range(E)])
    ref = per_expert[topk_ids.long(), torch.arange(M, device=device)[:, None]]
    _assert_close(out, ref)


# (E, K, inter, top_k)
SHAPES = [
    pytest.param(256, 4096, 512, 8, id="mimo-tp4"),
    pytest.param(64, 4096, 2048, 8, id="mimo-ep4"),
    # K and 2*inter are not multiples of the K/N tiles.
    pytest.param(16, 2080, 736, 4, id="ragged"),
]


@pytest.mark.parametrize("E, K, inter, top_k", SHAPES)
@pytest.mark.parametrize("M", [1, 7, 64, 333])
def test_moe_matches_reference(E, K, inter, top_k, M):
    gen = torch.Generator().manual_seed(M * 1000 + E)
    w13, s13, w2, s2 = _make_weights(E, K, inter, gen)
    x = torch.randn(M, K, generator=gen).bfloat16().to(device)
    topk_weights, topk_ids = _routing(M, E, top_k, gen)

    experts = _make_experts(E, K, inter, top_k, s13, s2)
    out = _run(experts, x, w13, w2, topk_weights, topk_ids, E)
    _assert_close(out, _moe_ref(x, w13, s13, w2, s2, topk_weights, topk_ids))


@pytest.mark.parametrize("M", [7, 333])
def test_fp16_activations(M):
    # fp16 has no GEMV path; the reference's bf16 rounding bounds the error.
    E, K, inter, top_k = 64, 1024, 256, 8
    gen = torch.Generator().manual_seed(M)
    w13, s13, w2, s2 = _make_weights(E, K, inter, gen)
    x = torch.randn(M, K, generator=gen).half().to(device)
    topk_weights, topk_ids = _routing(M, E, top_k, gen)

    experts = _make_experts(E, K, inter, top_k, s13, s2)
    out = _run(experts, x, w13, w2, topk_weights, topk_ids, E)
    _assert_close(out, _moe_ref(x, w13, s13, w2, s2, topk_weights, topk_ids))


@pytest.mark.parametrize("M", [3, 200])
def test_expert_parallel_skips_remote_experts(M):
    E, K, inter, top_k = 32, 1024, 256, 4
    local = torch.arange(E) % 2 == 1
    expert_map = torch.full((E,), -1, dtype=torch.int32)
    expert_map[local] = torch.arange(int(local.sum()), dtype=torch.int32)
    expert_map = expert_map.to(device)

    gen = torch.Generator().manual_seed(M)
    w13, s13, w2, s2 = _make_weights(int(local.sum()), K, inter, gen)
    x = torch.randn(M, K, generator=gen).bfloat16().to(device)
    topk_weights, topk_ids = _routing(M, E, top_k, gen)

    experts = _make_experts(E, K, inter, top_k, s13, s2, num_local_experts=E // 2)
    out = _run(experts, x, w13, w2, topk_weights, topk_ids, E, expert_map)
    ref = _moe_ref(x, w13, s13, w2, s2, topk_weights, topk_ids, expert_map)
    _assert_close(out, ref)


def test_auto_selects_triton_mxfp4_on_gfx1100():
    from vllm.platforms.rocm import on_gfx1100

    if not on_gfx1100():
        pytest.skip("gfx1100 only")
    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(
        _make_moe_config(8, 256, 64, 2)
    )
    assert backend == Mxfp4MoeBackend.TRITON_MXFP4_BF16
    assert experts_cls is TritonMxfp4W4A16Experts
