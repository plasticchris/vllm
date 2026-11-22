# SPDX-License-Identifier: Apache-2.0
"""Eight expert outputs: retain FP32 accumulation order, write BF16 once.

The interleaved four-accumulator order matches torch's contiguous-output
sum(dim=1) reduction here. Do not substitute a generic tree reduction: exponent
and cancellation stress fixtures distinguish them despite normal random parity.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _load_route(X, Local, row, col, H: tl.constexpr, J: tl.constexpr, SPARSE: tl.constexpr):
    mask = col < H
    if SPARSE:
        mask = mask & (tl.load(Local + row * 8 + J) >= 0)
    return tl.load(X + (row * 8 + J) * H + col, mask, 0).to(tl.float32)


@triton.jit
def _combine8(X, Y, Local, H: tl.constexpr, B: tl.constexpr, SPARSE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    a0 = _load_route(X, Local, row, col, H, 0, SPARSE)
    a1 = _load_route(X, Local, row, col, H, 1, SPARSE)
    a2 = _load_route(X, Local, row, col, H, 2, SPARSE)
    a3 = _load_route(X, Local, row, col, H, 3, SPARSE)
    a4 = _load_route(X, Local, row, col, H, 4, SPARSE)
    a5 = _load_route(X, Local, row, col, H, 5, SPARSE)
    a6 = _load_route(X, Local, row, col, H, 6, SPARSE)
    a7 = _load_route(X, Local, row, col, H, 7, SPARSE)
    value = (((a0 + a4) + (a1 + a5)) + (a2 + a6)) + (a3 + a7)
    # Torch's accumulators start at +0, so an all-negative-zero row sums to +0.
    # Preserve nonzero underflow signs at the final BF16 cast, and leave NaNs.
    value = tl.where(value == 0.0, 0.0, value)
    tl.store(Y + row * H + col, value, col < H)


def combine_experts(x: torch.Tensor, *, local: torch.Tensor | None = None):
    if x.ndim != 3 or x.shape[1:] != (8, 2304) or not x.is_contiguous():
        raise ValueError("expected contiguous [tokens,8,hidden] expert output")
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("qualified only for CUDA/ROCm BF16 expert outputs")
    t, _, h = x.shape
    if local is not None and (local.shape != (t, 8) or not local.is_contiguous() or local.device != x.device):
        raise ValueError("expected contiguous [tokens,8] local expert IDs")
    out = torch.empty((t, h), device=x.device, dtype=x.dtype)
    if t:
        _combine8[(t, triton.cdiv(h, 256))](
            x, out, local, h, 256, SPARSE=local is not None,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
