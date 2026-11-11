# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for lifetime and values across QSA score row chunks."""

import weakref

import pytest
import torch

from vllm import envs
from vllm.models.qwen4_exp.amd.ops import qsa
from vllm.platforms.rocm import on_gfx1100


@pytest.mark.parametrize("value", [1, 16, 32, 128])
def test_score_workspace_registry_and_valid_values(value, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_QSA_LOGITS_WORKSPACE_MIB", str(value))
    assert value == envs.VLLM_ROCM_QSA_LOGITS_WORKSPACE_MIB
    assert qsa._qsa_logits_workspace_bytes(value) == value * 1024**2


@pytest.mark.parametrize("value", [0, -1, 129, 10**30])
def test_score_workspace_rejects_out_of_range(value):
    with pytest.raises(ValueError, match="in \\[1, 128\\]"):
        qsa._qsa_logits_workspace_bytes(value)


def test_score_workspace_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_QSA_LOGITS_WORKSPACE_MIB", raising=False)
    assert envs.VLLM_ROCM_QSA_LOGITS_WORKSPACE_MIB == 128


def test_previous_score_tile_is_released_before_next_allocation(monkeypatch):
    # Force three chunks of 2 rows and 8 score columns, including a final tail.
    monkeypatch.setattr(qsa, "_LOGITS_WORKSPACE_BYTES", 2 * 8 * 4)
    monkeypatch.setattr(qsa, "on_gfx1100", lambda: True)
    old = []
    chunks = []

    def score(query, *args, **kwargs):
        if old:
            assert all(ref() is None for ref in old), "previous score tile still live"
        logits = torch.arange(8, dtype=torch.float32).repeat(query.shape[0], 1)
        visible = torch.full((query.shape[0],), 8, dtype=torch.int32)
        old[:] = [weakref.ref(logits), weakref.ref(visible)]
        chunks.append(query.shape[0])
        return logits, visible

    def expand(blocks, positions, lengths, reqs, ratio, token_topk, out):
        out.fill_(-1)
        out[:, :2].copy_(blocks)

    monkeypatch.setattr(qsa, "qsa_mqa_paged", score)
    monkeypatch.setattr(qsa, "expand_qsa_block_indices_cuda", expand)
    result = qsa.qsa_select_paged_tokens(
        torch.zeros(5, 1, 1),
        torch.empty(1, 4, 1, 1),
        torch.empty(1, 2, dtype=torch.int32),
        torch.zeros(5, dtype=torch.int32),
        torch.arange(5, dtype=torch.int32),
        torch.tensor([32], dtype=torch.int32),
        token_topk=8,
        compress_ratio=4,
        allow_all_blocks=False,
    )
    assert chunks == [2, 2, 1]
    assert torch.equal(result[:, :2], torch.tensor([[7, 6]] * 5, dtype=torch.int32))
    assert torch.all(result[:, 2:] == -1)
    assert all(ref() is None for ref in old)


@pytest.mark.skipif(not on_gfx1100(), reason="requires gfx1100 GPU")
def test_real_scorer_chunk_boundary_peak_and_selection(monkeypatch):
    # The retained-tile control models Python's old assignment lifetime while
    # running exactly the same real scorer/top-k/expansion operations.
    torch.manual_seed(8)
    device = "cuda"
    context, rows, width, chunk_rows = 8192, 257, 2048, 128
    pages = width // 16
    query = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(pages, 16, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    positions = torch.arange(context - rows, context, device=device, dtype=torch.int32)
    lengths = torch.tensor([context], device=device, dtype=torch.int32)
    output = torch.empty((rows, 2051), device=device, dtype=torch.int32)
    monkeypatch.setattr(qsa, "_LOGITS_WORKSPACE_BYTES", chunk_rows * width * 4)
    real_score = qsa.qsa_mqa_paged
    hold = []
    allocation_points = []
    retain_tile = False

    def measured_score(*args, **kwargs):
        value = real_score(*args, **kwargs)
        allocation_points.append(torch.cuda.memory_allocated())
        if retain_tile:
            hold[:] = [value[0], value[1]]
        return value

    def run(retain):
        nonlocal retain_tile
        retain_tile = retain
        allocation_points.clear()
        monkeypatch.setattr(qsa, "qsa_mqa_paged", measured_score)
        hold.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        qsa.qsa_select_paged_tokens(
            query,
            keys,
            table,
            requests,
            positions,
            lengths,
            2048,
            4,
            out=output,
            max_seq_len=context,
            allow_all_blocks=False,
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        points = [value - base for value in allocation_points]
        result = output.clone()
        hold.clear()
        return peak, points, result

    # Warm up both paths before peak collection (compilation and allocations).
    run(False)
    run(True)
    optimized, optimized_points, actual = run(False)
    old_peak, retained_points, expected = run(True)
    assert torch.equal(actual, expected)
    tile_bytes = chunk_rows * width * 4
    print(
        f"peak optimized={optimized} retained={old_peak} tile={tile_bytes} bytes; "
        f"boundary optimized={optimized_points} retained={retained_points}"
    )
    assert len(optimized_points) == len(retained_points) == 3
    assert retained_points[1] - optimized_points[1] >= tile_bytes
    assert optimized < old_peak, (optimized, old_peak)
