#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm Hybrid W8A16 kernel (HIP skinny int8 + Triton prefill).

Run `pytest tests/kernels/quantization/test_rdna_hybrid_w8a16.py`.

The int8 W8A16 dense-linear (e.g. qwen3.8-27b-int8-w8a16) must resolve to
RDNAHybridW8A16LinearKernel, not the Conch/Exllama fallback. These tests pin:
  - the process_weights_after_loading layout the kernel reads (int8 [N,K] + fp32 scales)
  - the full apply_weights path (HIP skinny decode + Triton prefill) vs an
    exact-dequant float32 reference
  - can_implement / group-size policy
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

from vllm.platforms.rocm import on_gfx1x  # noqa: E402

device = "cuda"

hybrid_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w8a16"
)
RDNAHybridW8A16LinearKernel = hybrid_module.RDNAHybridW8A16LinearKernel
SUPPORTED_GROUP_SIZES = hybrid_module.SUPPORTED_GROUP_SIZES
MAX_SKINNY_BATCH_SIZE = hybrid_module.MAX_SKINNY_BATCH_SIZE


# ---------------------------------------------------------------------------
# Reference: exact int8 dequant then matmul in float32, cast back.
# w_int8: int8 [N, K] (one signed byte per value), scales: [N, K/G] fp32
# ---------------------------------------------------------------------------


def _w8a16_reference(x_mk, w_int8_nk, scales_nkg, group_size, bias):
    """Math ground truth: exact int8 dequant in float32, then the GEMM.

    The kernel paths differ in how the dequanted weight is cast before the
    GEMM (skinny: exact int8 MACs; Triton prefill: dequant cast to the
    activation dtype), so callers compare with a bf16-noise-calibrated
    tolerance rather than bit-exact equality.
    """
    N, K = w_int8_nk.shape
    s_full = scales_nkg.repeat_interleave(group_size, dim=1).to(torch.float32)
    w_fp = w_int8_nk.to(torch.float32) * s_full  # [N, K] f32 dequant
    out = x_mk.to(torch.float32) @ w_fp.t()
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


def assert_w8a16_close(out, ref, l2rel_max=2e-2, max_abs_max=1.0):
    """bf16-noise-calibrated comparison against the f32-exact reference.

    A real bug (wrong scale layout / wrong shuffle / OOB read) produces
    O(1) errors -> l2rel ~ 0.5+. The kernel's legitimate bf16 dequant noise
    is < 1% in relative L2 for these magnitudes.
    """
    diff = (out.float() - ref.float()).abs()
    l2rel = (diff.pow(2).sum().sqrt() / ref.float().pow(2).sum().sqrt()).item()
    max_abs = diff.max().item()
    assert l2rel < l2rel_max, f"l2rel {l2rel:.4f} >= {l2rel_max}"
    assert max_abs < max_abs_max, f"max_abs {max_abs:.4f} >= {max_abs_max}"


def _build_dummy_layer(w_int8_nk, scales_ckpt_nkg):
    """Layer matching the CompressedTensorsWNA16 int8 checkpoint:
       weight_packed: PackedvLLMParameter int32 [N, K/4] (4 int8 per int32, K-major)
       weight_scale:  GroupQuantScaleParameter [N, K/G]
    """
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    weight_loader = lambda *args, **kwargs: None

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=w_int8_nk.contiguous().view(torch.int32),  # [N, K/4] int32
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
            packed_factor=4,
            packed_dim=1,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales_ckpt_nkg,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
        ),
    )
    return layer


def _make_weights(N, K, G, dtype, seed=0):
    set_random_seed(seed)
    w_int8_nk = torch.randint(
        -127, 128, (N, K), device=device, dtype=torch.int32
    ).to(torch.int8)
    scales_ckpt_nkg = (
        0.02 * torch.rand((N, K // G), device=device, dtype=torch.float32)
    ).to(dtype)
    return w_int8_nk, scales_ckpt_nkg


def _make_kernel(N, K, G, dtype):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint8b128,
        act_type=dtype,
        group_size=G,
        zero_points=False,
    )
    return RDNAHybridW8A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
    ), config


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
def test_w8a16_process_weights_layout(group_size, dtype, dist_init):
    """process_weights_after_loading leaves int8 [N,K] weights + fp32 scales."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    N, K, G = 1024, 5120, group_size
    assert K % G == 0
    w_int8_nk, scales_ckpt_nkg = _make_weights(N, K, G, dtype)
    layer = _build_dummy_layer(w_int8_nk, scales_ckpt_nkg)
    kernel, config = _make_kernel(N, K, G, dtype)
    assert RDNAHybridW8A16LinearKernel.can_implement(config)[0]
    kernel.process_weights_after_loading(layer)

    # Weights stored once as int8 [N, K], K-major (natural order, no shuffle).
    assert layer.weight_packed.dtype == torch.int8
    assert tuple(layer.weight_packed.shape) == (N, K)
    torch.testing.assert_close(
        layer.weight_packed.cpu(), w_int8_nk.cpu(), rtol=0, atol=0
    )
    # Scales promoted to fp32 [N, K/G].
    assert layer.weight_scale.dtype == torch.float32
    assert tuple(layer.weight_scale.shape) == (N, K // G)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize(
    "M",
    [1, 2, MAX_SKINNY_BATCH_SIZE, MAX_SKINNY_BATCH_SIZE + 1, 64],
    ids=["M1", "M2", "decode", "prefill", "M64"],
)
def test_w8a16_apply_matches_reference(group_size, dtype, M, dist_init):
    """Full apply_weights (skinny decode + Triton prefill) matches the reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    N, K, G = 1024, 5120, group_size
    w_int8_nk, scales_ckpt_nkg = _make_weights(N, K, G, dtype)
    layer = _build_dummy_layer(w_int8_nk, scales_ckpt_nkg)
    kernel, _ = _make_kernel(N, K, G, dtype)
    kernel.process_weights_after_loading(layer)

    x = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    bias = (0.05 * torch.randn(N, device=device, dtype=torch.float32)).to(dtype)

    out = kernel.apply_weights(layer, x, bias)
    ref = _w8a16_reference(x, w_int8_nk, scales_ckpt_nkg.to(torch.float32), G, bias)
    assert out.shape == (M, N)
    assert_w8a16_close(out, ref)


@pytest.mark.skipif(not on_gfx1x(), reason="Hybrid path is gfx11/gfx12 only")
@pytest.mark.parametrize(
    "group_size,expected_ok",
    [(32, True), (64, True), (128, True), (256, False)],
)
def test_w8a16_can_implement_group_size(group_size, expected_ok):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    K, N = 5120, 1024
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint8b128,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=False,
    )
    ok, _ = RDNAHybridW8A16LinearKernel.can_implement(config)
    assert ok is expected_ok
