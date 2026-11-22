# SPDX-License-Identifier: Apache-2.0
"""Eight expert outputs: retain FP32 accumulation order, write BF16 once.

The interleaved four-accumulator order matches torch's contiguous-output
sum(dim=1) reduction here. Do not substitute a generic tree reduction: exponent
and cancellation stress fixtures distinguish them despite normal random parity.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _combine8(X, Y, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    a0 = tl.load(X + (row * 8 + 0) * H + col, col < H, 0).to(tl.float32)
    a1 = tl.load(X + (row * 8 + 1) * H + col, col < H, 0).to(tl.float32)
    a2 = tl.load(X + (row * 8 + 2) * H + col, col < H, 0).to(tl.float32)
    a3 = tl.load(X + (row * 8 + 3) * H + col, col < H, 0).to(tl.float32)
    a4 = tl.load(X + (row * 8 + 4) * H + col, col < H, 0).to(tl.float32)
    a5 = tl.load(X + (row * 8 + 5) * H + col, col < H, 0).to(tl.float32)
    a6 = tl.load(X + (row * 8 + 6) * H + col, col < H, 0).to(tl.float32)
    a7 = tl.load(X + (row * 8 + 7) * H + col, col < H, 0).to(tl.float32)
    value = (((a0 + a4) + (a1 + a5)) + (a2 + a6)) + (a3 + a7)
    # Torch's accumulators start at +0, so an all-negative-zero row sums to +0.
    # Preserve nonzero underflow signs at the final BF16 cast, and leave NaNs.
    value = tl.where(value == 0.0, 0.0, value)
    tl.store(Y + row * H + col, value, col < H)


def combine_experts(x: torch.Tensor):
    if x.ndim != 3 or x.shape[1:] != (8, 2304) or not x.is_contiguous():
        raise ValueError("expected contiguous [tokens,8,hidden] expert output")
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("qualified only for CUDA/ROCm BF16 expert outputs")
    t, _, h = x.shape
    out = torch.empty((t, h), device=x.device, dtype=x.dtype)
    if t:
        _combine8[(t, triton.cdiv(h, 256))](
            x, out, h, 256, num_warps=4, enable_fp_fusion=False
        )
    return out
