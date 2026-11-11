# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU gates for the all-block QSA selection shortcut (no model required)."""

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops import qsa
from vllm.platforms.rocm import on_gfx1100

pytestmark = pytest.mark.skipif(not on_gfx1100(), reason="gfx1100 fast path")


@pytest.mark.parametrize("ctx", [1, 4, 2048, 2051, 2052])
@pytest.mark.parametrize("rows", [1, 3])
def test_all_blocks_matches_full_selection(ctx, rows):
    torch.manual_seed(ctx)
    pages = ((ctx + 3) // 4 + 15) // 16
    keys = torch.randn(pages, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    # Multi-request, permuted page IDs, including MTP-width query rows.
    table = torch.randperm(pages, device="cuda", dtype=torch.int32)[None].repeat(2, 1)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
    positions = torch.arange(
        max(0, ctx - rows), max(0, ctx - rows) + rows, device="cuda", dtype=torch.int32
    ).clamp(max=ctx - 1)
    lengths = torch.full((2,), ctx, dtype=torch.int32, device="cuda")
    kwargs = dict(max_seq_len=ctx)
    expected = qsa.qsa_select_paged_tokens(
        query,
        keys,
        table,
        requests,
        positions,
        lengths,
        2048,
        4,
        allow_all_blocks=False,
        **kwargs,
    )
    actual = qsa.qsa_select_paged_tokens(
        query, keys, table, requests, positions, lengths, 2048, 4, **kwargs
    )
    if ctx > 2051:
        # Above the bound, compare selected scores: the unchanged torch
        # fallback can choose different tied boundary blocks. Selecting the
        # first 512 blocks instead of the true top-k fails this comparison.
        logits, _ = qsa.qsa_mqa_paged(
            query, keys, table, requests, positions, lengths, 4
        )
        a_blocks = actual[:, :2048:4].long() // 4
        b_blocks = expected[:, :2048:4].long() // 4
        assert torch.equal(
            logits.gather(1, a_blocks).sort().values,
            logits.gather(1, b_blocks).sort().values,
        )
        return
    assert torch.equal(actual.sort(-1).values, expected.sort(-1).values)
    for i, pos in enumerate(positions.tolist()):
        assert actual[i, : pos + 1].tolist() == list(range(pos + 1))
        assert (actual[i, pos + 1 :] == -1).all()
    # The selected SET is identical; different accumulation order may round
    # differently. Gate numerical equivalence, not seedwise identity.
    main_pages = (ctx + 15) // 16
    kv = torch.randn(2, main_pages, 16, 1, 256, device="cuda", dtype=torch.bfloat16)
    bt = torch.randperm(main_pages, device="cuda", dtype=torch.int32)[None].repeat(2, 1)
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.bfloat16)
    a = qsa.qsa_sparse_paged_attention(q, kv[0], kv[1], actual, bt, requests)
    b = qsa.qsa_sparse_paged_attention(q, kv[0], kv[1], expected, bt, requests)
    torch.testing.assert_close(a, b, rtol=0.01, atol=0.001)


def test_all_blocks_padding_and_strided_output():
    # Rejected speculative rows and empty requests may be marked inert. They
    # must not read another request or expose a future token.
    positions = torch.tensor([0, 3, 5, 6, 2050], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 7, 2051], device="cuda", dtype=torch.int32)
    requests = torch.tensor([-1, 0, 1, 1, 2], device="cuda", dtype=torch.int32)
    storage = torch.full((5, 4102), -99, device="cuda", dtype=torch.int32)
    out = storage[:, ::2]
    actual = qsa.qsa_select_paged_tokens(
        torch.empty(5, 4, 128, device="cuda", dtype=torch.bfloat16),
        torch.empty(33, 16, 1, 128, device="cuda", dtype=torch.bfloat16),
        torch.zeros(3, 33, device="cuda", dtype=torch.int32),
        requests,
        positions,
        lengths,
        2048,
        4,
        out=out,
        max_seq_len=2051,
    )
    assert actual is out
    assert (actual[:2] == -1).all()
    for row, count in [(2, 6), (3, 7), (4, 2051)]:
        assert actual[row, :count].tolist() == list(range(count))
        assert (actual[row, count:] == -1).all()
    assert (storage[:, 1::2] == -99).all()


@pytest.mark.parametrize("rows", [32, 64, 128, 129, 257, 513, 2048])
@pytest.mark.parametrize("fp8", [False, True])
def test_windowed_prefill_preserves_attention_and_bounds_scratch(
    rows, fp8, monkeypatch
):
    torch.manual_seed(rows)
    kv = torch.randn(2, 129, 16, 1, 256, device="cuda", dtype=torch.bfloat16)
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(2051, device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
    # Include inert/padded rows and partially filled histories.
    indices[::3, 1000:] = -1
    indices[::7] = -1
    bt = torch.randperm(129, device="cuda", dtype=torch.int32)[None].repeat(2, 1)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(qsa, "_QSA_RDNA3_PREFILL_TUNED", False)
    expected = qsa.qsa_sparse_paged_attention(
        q, kv[0], kv[1], indices, bt, requests, k_scale=scale, v_scale=scale
    )
    storage = torch.full((rows, 12, 256), float("nan"), device="cuda", dtype=q.dtype)
    out = storage[:, ::2]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    monkeypatch.setattr(qsa, "_QSA_RDNA3_PREFILL_TUNED", True)
    actual = qsa.qsa_sparse_paged_attention(
        q, kv[0], kv[1], indices, bt, requests, out=out, k_scale=scale, v_scale=scale
    )
    torch.cuda.synchronize()
    assert actual is out
    assert torch.cuda.max_memory_allocated() - baseline < 16 * 1024**2
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
    assert (actual[::7] == 0).all()
    assert storage[:, 1::2].isnan().all()
