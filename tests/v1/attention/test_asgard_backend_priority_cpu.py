# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression for the gfx1100 long-context backend-default change.

Mock backend capability probes, not the ordering code under test. No HIP
context, model, distributed group, or device tensor is required.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import rocm
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum as Backend


@pytest.mark.parametrize(
    "arch,kv_dtype,opt_in,expected",
    [
        ("gfx1100", "bfloat16", False, Backend.ROCM_ATTN),
        ("gfx1100", "auto", False, Backend.ROCM_ATTN),
        ("gfx1100", "fp8", False, Backend.ROCM_ATTN),
        ("gfx1100", "fp8_e5m2", False, Backend.ROCM_ATTN),
        ("gfx1100", "bfloat16", True, Backend.TRITON_ATTN),
        ("gfx1201", "bfloat16", False, Backend.TRITON_ATTN),
        ("gfx1201", "fp8_e5m2", False, Backend.ROCM_ATTN),
        ("gfx1101", "bfloat16", False, Backend.ROCM_ATTN),
    ],
)
def test_priority_without_gpu(monkeypatch, arch, kv_dtype, opt_in, expected):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm, "on_gfx1201", lambda: arch == "gfx1201")
    monkeypatch.setattr(rocm, "on_gfx1100", lambda: arch == "gfx1100")
    monkeypatch.setattr(rocm, "on_rdna4", lambda: arch == "gfx1201")
    monkeypatch.setattr(rocm_aiter_ops, "is_mha_enabled", lambda: False)
    monkeypatch.setattr(rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False)
    monkeypatch.setattr("vllm._aiter_ops.is_aiter_found_and_supported", lambda: False)
    if opt_in:
        monkeypatch.setenv("ROCM_PREFER_TRITON_ATTN", "1")
    else:
        monkeypatch.delenv("ROCM_PREFER_TRITON_ATTN", raising=False)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(rocm, "_get_invalid_reasons", lambda *a, **kw: [])
    selector = SimpleNamespace(
        use_mla=False, use_sparse=False, use_kv_connector=False,
        kv_cache_dtype=kv_dtype,
    )
    with (
        patch.object(Backend, "get_class", return_value=object),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU init")),
    ):
        valid, _ = rocm.RocmPlatform.get_valid_backends(
            DeviceCapability(11, 0), selector
        )
    assert min(valid, key=lambda x: x[1])[0] == expected


def test_explicit_triton_still_overrides_default(monkeypatch):
    monkeypatch.setattr(rocm.RocmPlatform, "get_device_capability",
                        lambda: DeviceCapability(11, 0))
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(rocm, "_get_invalid_reasons", lambda *a, **kw: [])
    with patch.object(Backend, "get_class", return_value=object):
        path = rocm.RocmPlatform.get_attn_backend_cls(
            Backend.TRITON_ATTN, SimpleNamespace()
        )
    assert path == Backend.TRITON_ATTN.get_path()
