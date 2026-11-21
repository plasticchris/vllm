# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable counterpart of Inkling2's split-KV log2-LSE combine kernel."""

import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _combine(
    Parts,
    Lses,
    Gate,
    Sink,
    Out,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    GATED: tl.constexpr,
    ACTUAL: tl.constexpr,
    GS: tl.constexpr,
    SS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    out_ptr = Out + (row * H + head) * D + d
    if row >= ACTUAL:
        tl.store(out_ptr, 0.0)
        return
    splits = tl.arange(0, BS)
    part = (splits * T + row) * H + head
    lse = tl.load(Lses + part, splits < S, float("-inf"))
    top = tl.max(lse, 0)
    top = tl.where(top == float("-inf"), 0.0, top)
    weights = tl.exp2(lse - top)
    denom = tl.sum(weights, 0)
    values = tl.load(
        Parts + part[:, None] * D + d[None, :], (weights > 0)[:, None], 0.0
    )
    value = tl.sum(values * (weights / tl.where(denom > 0, denom, 1.0))[:, None], 0)
    if GATED:
        gate = tl.load(Gate + row * GS + head * D + d).to(tl.float32)
        sink = tl.load(Sink + row * SS + head).to(tl.float32)
        total_lse = (tl.log2(denom) + top) * 0.6931471805599453
        value *= 1.0 / (1.0 + tl.exp(-gate))
        value *= 1.0 / (1.0 + tl.exp(sink - total_lse))
    tl.store(out_ptr, value)


def merge_partials(parts, lses, out, gate=None, sinks=None, num_actual_tokens=None):
    s, t, h, d = parts.shape
    assert parts.dtype == lses.dtype == torch.float32
    assert parts.is_contiguous() and lses.is_contiguous() and out.is_contiguous()
    assert tuple(lses.shape) == (s, t, h) and out.numel() == t * h * d
    assert (gate is None) == (sinks is None)
    gated = gate is not None
    if gated:
        assert gate.stride(1) == sinks.stride(1) == 1
        assert gate.numel() == t * h * d and sinks.numel() == t * h
    actual = t if num_actual_tokens is None else num_actual_tokens
    assert 0 <= actual <= t
    _combine[(t, h)](
        parts,
        lses,
        parts if gate is None else gate,
        parts if sinks is None else sinks,
        out,
        t,
        h,
        d,
        s,
        triton.next_power_of_2(s),
        gated,
        actual,
        0 if gate is None else gate.stride(0),
        0 if sinks is None else sinks.stride(0),
        num_warps=4,
    )
