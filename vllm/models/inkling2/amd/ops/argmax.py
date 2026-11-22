# SPDX-License-Identifier: Apache-2.0
"""Exact greedy selection across contiguous vocabulary shards.

Exchange FP32 (value, global-id) pairs, not the vocabulary. IDs below 2**24
are exactly representable. NaNs count as -inf; ties select the first global ID,
matching the large-vocabulary greedy_argmax path used by Inkling DSpark.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.greedy_argmax import _argmax_partial


@triton.jit
def _finish_local(
    PV, PI, OUT, SPLITS: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(PV + row * SPLITS + col, col < SPLITS, float("-inf"))
    part = tl.argmax(value, axis=0)
    score = tl.load(PV + row * SPLITS + part)
    index = tl.load(PI + row * SPLITS + part) + OFFSET
    tl.store(OUT + row * 2, score)
    tl.store(OUT + row * 2 + 1, index.to(tl.float32))


@triton.jit
def _finish_global(PAIRS, OUT, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    values = tl.load(PAIRS + row * WORLD * 2 + rank * 2, rank < WORLD, float("-inf"))
    owner = tl.argmax(values, axis=0)
    index = tl.load(PAIRS + row * WORLD * 2 + owner * 2 + 1)
    tl.store(OUT + row, index.to(tl.int64))


def local_argmax_pair(logits: torch.Tensor, offset: int, valid_vocab: int):
    rows, width = logits.shape
    if not logits.is_cuda or logits.stride(1) != 1:
        raise ValueError("local_argmax_pair requires a CUDA/ROCm row-major vocabulary")
    if (
        not 0 <= valid_vocab <= width
        or not 0 <= offset < 2**24
        or offset + width > 2**24
    ):
        raise ValueError("invalid vocabulary shard or inexact FP32 index range")
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("unsupported greedy logit dtype")
    out = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
    if rows == 0:
        return out
    if valid_vocab == 0:
        out[:, 0].fill_(-float("inf"))
        out[:, 1].fill_(offset)
        return out
    splits = triton.cdiv(valid_vocab, 4096)
    values = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
    _argmax_partial[(rows, splits)](
        logits,
        values,
        indices,
        valid_vocab,
        logits.stride(0),
        splits,
        4096,
        num_warps=4,
    )
    _finish_local[(rows,)](
        values,
        indices,
        out,
        splits,
        offset,
        triton.next_power_of_2(splits),
        num_warps=1,
    )
    return out


def gathered_argmax(pairs: torch.Tensor):
    rows, cols = pairs.shape
    if (
        pairs.dtype != torch.float32
        or not pairs.is_contiguous()
        or cols % 2
        or cols == 0
    ):
        raise ValueError("expected contiguous FP32 rank-ordered (value,id) pairs")
    world = cols // 2
    out = torch.empty(rows, dtype=torch.int64, device=pairs.device)
    if rows:
        _finish_global[(rows,)](
            pairs, out, world, triton.next_power_of_2(world), num_warps=1
        )
    return out
