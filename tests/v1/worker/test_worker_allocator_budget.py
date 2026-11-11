# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the allocator helper used by real GPU-worker initialization."""

import pytest
import torch

from vllm.v1.worker.gpu_worker import _set_worker_allocator_budget


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf"])
def test_invalid_fraction_fails_without_initializing_cuda(value, monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MEMORY_FRACTION", value)
    before = torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="finite and in"):
        _set_worker_allocator_budget(torch.device("cpu"))
    assert torch.cuda.is_initialized() == before


def test_default_is_noop(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MEMORY_FRACTION", raising=False)
    before = torch.cuda.is_initialized()
    _set_worker_allocator_budget(torch.device("cpu"))
    assert torch.cuda.is_initialized() == before


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocator enforcement requires a GPU"
)
def test_worker_fraction_is_enforced_by_real_allocator(monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setenv("VLLM_WORKER_MEMORY_FRACTION", "0.02")
    try:
        _set_worker_allocator_budget(device)
        assert torch.cuda.get_per_process_memory_fraction(device) == pytest.approx(0.02)
        small = torch.empty(4 * 1024**2, dtype=torch.uint8, device=device)
        assert small.numel() == 4 * 1024**2
        # This fits an idle card physically, but exceeds its 2% budget.
        request = int(torch.cuda.get_device_properties(device).total_memory * 0.03)
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(request, dtype=torch.uint8, device=device)
        del small
    finally:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device)
