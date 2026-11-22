# SPDX-License-Identifier: Apache-2.0
"""Compile the selected MXFP8 conversion into an exact 128KiB BF16 codebook."""

import torch
from vllm.triton_utils import triton, tl
from .moe import _e4m3


@triton.jit
def _build_mx(Y, B: tl.constexpr):
    z = tl.program_id(0) * B + tl.arange(0, B)
    code = (z % 256).to(tl.uint8)
    scale = z // 256
    # Keep tl.exp2's existing scale-code0 flushing/nonfinite behavior.
    value = _e4m3(code) * tl.exp2(scale.to(tl.float32) - 127.0)
    tl.store(Y + z, value.to(tl.bfloat16))


def build_mx_codebook(device):
    out = torch.empty((256, 256), device=device, dtype=torch.bfloat16)
    _build_mx[(64,)](out, 1024, num_warps=4)
    return out
