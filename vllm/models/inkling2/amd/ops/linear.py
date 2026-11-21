"""Portable ordered 1024-wide bf16 GEMM partials."""

import torch
from vllm.triton_utils import tl, triton


def linear(x, weight, out=None):
    if out is None:
        return torch.nn.functional.linear(x, weight)
    return torch.mm(x, weight.t(), out=out)


@triton.jit
def _k_group_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    pid_n,
    pid_m,
    pid_k,
):
    # Row tiles of at most 64 keep tl.dot on the same mma.sync instruction
    # sequence as the reference's 16x32x32 tiling. With SPLIT, each program computes one
    # K group's partial; otherwise it sums the groups in order.
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, 128)
    first = pid_k if SPLIT else 0
    NUM_GROUPS: tl.constexpr = 1 if SPLIT else (K + 1023) // 1024
    total = tl.zeros((BM, BN), tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        group = first + g
        partial = tl.zeros((BM, BN), tl.float32)
        for tile in range(tl.minimum(8, tl.cdiv(K - group * 1024, 128))):
            k = group * 1024 + tile * 128 + ks
            x = tl.load(
                X + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + cols[None, :] * K + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        total = partial if g == 0 else total + partial
    tl.store(
        Y + (first * M + rows[:, None]) * N + cols[None, :],
        total.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _groups(X, W, Y, M, N: tl.constexpr, K: tl.constexpr):
    _k_group_tile(
        X,
        W,
        Y,
        M,
        N,
        K,
        16,
        32,
        True,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


def k_group_partials(x, weight, out_dtype=torch.float32, launch_dependents=False):
    assert x.is_contiguous() and weight.is_contiguous()
    m, k = x.shape
    n = weight.shape[0]
    groups = triton.cdiv(k, 1024)
    out = torch.empty((groups, m, n), dtype=torch.float32, device=x.device)
    if m:
        _groups[(triton.cdiv(n, 32), triton.cdiv(m, 16), groups)](
            x, weight, out, m, n, k, num_warps=4, num_stages=1, enable_fp_fusion=False
        )
    return out


def linear_k_groups(x, weight, out_dtype=torch.float32):
    partials = k_group_partials(x, weight)
    out = partials[0]
    for i in range(1, partials.shape[0]):
        out = out + partials[i]
    return out.to(out_dtype)
