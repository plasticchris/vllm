# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical checks for AMD QSA FP8-E4M3 KV-cache reads."""

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops.qsa import qsa_sparse_paged_attention
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_rocm_qsa = pytest.mark.skipif(
    not current_platform.is_rocm() or not HAS_TRITON,
    reason="AMD QSA FP8 tests require ROCm and Triton",
)


@requires_rocm_qsa
@pytest.mark.parametrize("num_rows", [1, 64])
def test_qsa_fp8_kv_matches_bf16_kernel_on_dequantized_values(num_rows: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_blocks = 6
    page_size = 16
    num_query_heads = 8
    num_kv_heads = 1
    head_dim = 256
    selection_width = 65

    query = torch.randn(
        num_rows, num_query_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    key_fp8 = key.to(torch.float8_e4m3fn)
    value_fp8 = value.to(torch.float8_e4m3fn)
    dequantized_key = key_fp8.to(torch.bfloat16)
    dequantized_value = value_fp8.to(torch.bfloat16)
    logical_indices = torch.randint(
        0,
        num_blocks * page_size,
        (num_rows, selection_width),
        dtype=torch.int32,
        device=device,
    )
    logical_indices[:, selection_width // 2 :] = -1
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    token_to_req = torch.zeros(num_rows, dtype=torch.int32, device=device)
    one = torch.ones((), dtype=torch.float32, device=device)

    expected = qsa_sparse_paged_attention(
        query,
        dequantized_key,
        dequantized_value,
        logical_indices,
        block_table,
        token_to_req,
    )
    actual = qsa_sparse_paged_attention(
        query,
        key_fp8,
        value_fp8,
        logical_indices,
        block_table,
        token_to_req,
        k_scale=one,
        v_scale=one,
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@requires_rocm_qsa
def test_qsa_bf16_path_ignores_quantization_scales() -> None:
    device = torch.device("cuda")
    query = torch.randn(1, 8, 256, dtype=torch.bfloat16, device=device)
    key = torch.randn(2, 16, 1, 256, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    logical_indices = torch.arange(17, dtype=torch.int32, device=device).view(1, -1)
    logical_indices[:, -1] = -1
    block_table = torch.arange(2, dtype=torch.int32, device=device).view(1, -1)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=device)
    arbitrary = torch.tensor(7.0, dtype=torch.float32, device=device)

    expected = qsa_sparse_paged_attention(
        query, key, value, logical_indices, block_table, token_to_req
    )
    actual = qsa_sparse_paged_attention(
        query,
        key,
        value,
        logical_indices,
        block_table,
        token_to_req,
        k_scale=arbitrary,
        v_scale=arbitrary,
    )

    assert torch.equal(actual, expected)
