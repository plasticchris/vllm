# SPDX-License-Identifier: Apache-2.0
"""One-projection BF16 workspace experiment; never a model-sized weight cache.

Prefill only. Weight conversion cost is included in benchmark comparisons.
Decode retains packed GEMMs. Scaling/rounding match the packed weight-only path.
"""

import torch
from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from .moe import _e2m1_bits, _e2m1_bf16_bits, _e4m3_bits, _e4m3, INTEGER_FP4
from .tuning import expert_launch_config


@triton.jit
def _unpack(
    W,
    S,
    G,
    D,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL: tl.constexpr,
    NV: tl.constexpr,
    B: tl.constexpr,
    INTEGER_DECODE: tl.constexpr = False,
):
    z = tl.program_id(0) * B + tl.arange(0, B)
    kk = z % K
    nn = z // K
    ex = nn // N
    if NV:
        raw = tl.load(W + nn * (K // 2) + kk // 2, z < TOTAL, 0)
        c = (raw >> ((kk % 2) * 4)) & 15
        s = tl.load(
            S.to(tl.pointer_type(tl.uint8)) + nn * (K // 16) + kk // 16, z < TOTAL, 0
        )
        val = _e2m1_bf16_bits(c) if INTEGER_DECODE else _e2m1_bits(c)
        v = val * (_e4m3_bits(s) * tl.load(G + ex, ex < TOTAL // N // K, other=1))
    else:
        raw = tl.load(W.to(tl.pointer_type(tl.uint8)) + z, z < TOTAL, 0)
        s = tl.load(S + nn * (K // 32) + kk // 32, z < TOTAL, 0)
        v = _e4m3(raw) * tl.exp2(s.to(tl.float32) - 127.0)
    tl.store(D + z, v.to(D.dtype.element_ty), z < TOTAL)


@triton.jit
def _gemm(
    A,
    W,
    Sort,
    Expert,
    Padded,
    Y,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    DIV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    if tile * BM >= tl.load(Padded):
        return
    ex = tl.load(Expert + tile)
    if ex < 0:
        return
    rows = tl.load(Sort + tile * BM + tl.arange(0, BM))
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(K, BK)):
        k = i * BK + kk
        av = tl.load(
            A + (rows // DIV)[:, None] * K + k[None, :],
            (rows[:, None] < ROWS) & (k[None, :] < K),
            0,
        )
        bv = tl.load(
            W + (ex * N + cols[None, :]) * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            0,
        )
        acc = tl.dot(av, bv, acc)
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < ROWS) & (cols[None, :] < N),
    )


def staged_expert_gemm(
    x, weight, scale, global_scale, ids, input_div, nvfp4, layout=None
):
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("staged expert path requires contiguous BF16 input")
    e, n, packed_k = weight.shape
    k = packed_k * (2 if nvfp4 else 1)
    cfg, _ = expert_launch_config(ids.shape[0], k, nvfp4)
    bm, bn, bk = cfg
    if layout is None:
        layout = moe_align_block_size(ids, bm, e, ignore_invalid_experts=True)
    dense = torch.empty((e, n, k), device=x.device, dtype=x.dtype)
    _unpack[(triton.cdiv(dense.numel(), 1024),)](
        weight,
        scale,
        global_scale,
        dense,
        n,
        k,
        dense.numel(),
        nvfp4,
        1024,
        INTEGER_DECODE=INTEGER_FP4,
        num_warps=4,
    )
    out = torch.zeros((ids.numel(), n), device=x.device, dtype=x.dtype)
    if ids.numel():
        si, ei, pad = layout
        _gemm[(triton.cdiv(si.numel(), bm), triton.cdiv(n, bn))](
            x,
            dense,
            si,
            ei,
            pad,
            out,
            ids.numel(),
            n,
            k,
            input_div,
            bm,
            bn,
            bk,
            num_warps=4,
            num_stages=1,
        )
    return out, layout
