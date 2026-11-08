# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing contract; no GPU context or weight checkpoint required."""
import importlib
import pytest
import torch


@pytest.mark.parametrize('M,K,native', [
    (3, 5120, True), (6, 5120, True), (8, 5120, True),
    (9, 5120, True), (10, 5120, True), (11, 5120, True), (12, 5120, True),
    (12, 8192, True), (12, 8320, False), (13, 5120, False), (64, 5120, False),
])
def test_native_mtp_ladder_and_lds_guard(monkeypatch, M, K, native):
    module = importlib.import_module(
        'vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w8a16')
    import vllm._custom_ops as ops
    calls = []
    def skinny(w, x, s, cu_count, group_size, bias):
        calls.append('native')
        return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype)
    def triton(a, b_q, scales, group_size):
        calls.append('triton')
        return torch.empty((a.shape[0], b_q.shape[0]), dtype=a.dtype)
    monkeypatch.setattr(ops, 'wvSplitK_int8_g', skinny)
    monkeypatch.setattr(module, 'triton_w8a16_gemm', triton)
    # Registration exists on this ROCm build; no op is actually executed.
    assert getattr(torch.ops._rocm_C, 'wvSplitK_int8_g', None) is not None
    x = torch.empty((M, K), dtype=torch.bfloat16)
    w = torch.empty((16, K), dtype=torch.int8)
    s = torch.empty((16, K//128), dtype=torch.float32)
    out = module._rdna_hybrid_w8a16_apply_impl(x, w, s, None, 96, 128)
    assert out.shape == (M, 16)
    assert calls == ['native' if native else 'triton']
