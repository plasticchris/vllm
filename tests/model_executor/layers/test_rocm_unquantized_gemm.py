# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils


def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "shape,expected_cu_count",
    [
        ((4, 6144, 2560), 128),
        ((4, 2560, 6144), 128),
        ((4, 4608, 4608), 112),
    ],
)
def test_rocm_unquantized_gemm_uses_tuned_wvsplitk_cu_count(
    monkeypatch, shape, expected_cu_count
):
    n, m, k = shape
    x = torch.empty(n, k, dtype=torch.bfloat16)
    weight = torch.empty(m, k, dtype=torch.bfloat16)
    expected = torch.empty(n, m, dtype=torch.bfloat16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 48)

    wvsplitk_mock = MagicMock(return_value=expected)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)

    assert out.data_ptr() == expected.data_ptr()
    wvsplitk_mock.assert_called_once()
    call_args = wvsplitk_mock.call_args.args
    assert call_args[0] is weight
    assert call_args[1].data_ptr() == x.data_ptr()
    assert call_args[1].shape == x.shape
    assert call_args[2:] == (expected_cu_count, None)


def test_rocm_unquantized_gemm_uses_tuned_rocblas_shape(monkeypatch):
    x = torch.empty(3, 2560, dtype=torch.bfloat16)
    weight = torch.empty(12288, 2560, dtype=torch.bfloat16)
    expected = torch.empty(3, 12288, dtype=torch.bfloat16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 96)

    linear_mock = MagicMock(return_value=expected)
    monkeypatch.setattr(torch.nn.functional, "linear", linear_mock)
    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)

    assert out is expected
    linear_mock.assert_called_once_with(x, weight, None)
    wvsplitk_mock.assert_not_called()


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_gfx1x_n_gt_12_falls_back(monkeypatch):
    # The tuned wvSplitK dispatch handles n in [1, 12]; larger batches must
    # fall back to torch.nn.functional.linear.
    x = torch.randn(13, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
