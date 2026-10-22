# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Leyline-style KV-cache splice and in-place RoPE delta rotation."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.v1.attention.ops.leyline_rope import (
    apply_kv_cache_rope_rotations_inplace,
    rotate_key_cache_block_inplace,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.core.kv_cache_utils import KVCacheBlockRoPERotation


def _make_rope(
    head_size: int = 256,
    rotary_dim: int = 64,
    max_pos: int = 4096,
    base: float = 10000000.0,
    is_neox_style: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> RotaryEmbedding:
    with set_current_vllm_config(VllmConfig()):
        return RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_pos,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("delta_pos", [-256, -73, -1, 16, 128])
def test_rotate_key_cache_token_major_exact_algebraic_identity(
    dtype: torch.dtype, delta_pos: int
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    head_size = 256
    rotary_dim = 64
    num_kv_heads = 4
    block_size = 16
    num_blocks = 4
    block_id = 2

    rope = _make_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        dtype=torch.float32,
    ).to(device)
    inv_freq = rope._compute_inv_freq(rope.base).to(device)

    orig_positions = torch.arange(512, 512 + block_size, device=device, dtype=torch.long)
    shifted_positions = orig_positions + delta_pos

    raw_q = torch.randn(
        block_size, num_kv_heads * head_size, device=device, dtype=torch.float32
    )
    raw_k = torch.randn(
        block_size, num_kv_heads * head_size, device=device, dtype=torch.float32
    )

    _, k_at_orig = rope.forward_native(orig_positions, raw_q.clone(), raw_k.clone())
    _, k_at_shifted = rope.forward_native(
        shifted_positions, raw_q.clone(), raw_k.clone()
    )

    assert k_at_orig is not None and k_at_shifted is not None
    k_orig_4d = k_at_orig.view(block_size, num_kv_heads, head_size).to(dtype)
    k_expected_4d = k_at_shifted.view(block_size, num_kv_heads, head_size).to(dtype)

    key_cache = torch.zeros(
        (num_blocks, block_size, num_kv_heads, head_size),
        device=device,
        dtype=dtype,
    )
    key_cache[block_id] = k_orig_4d.clone()

    rotate_key_cache_block_inplace(
        key_cache=key_cache,
        block_id=block_id,
        delta_pos=delta_pos,
        inv_freq=inv_freq,
        start_offset=0,
        end_offset=block_size,
        is_neox_style=True,
        kv_cache_dtype="auto",
    )

    atol = 1e-4 if dtype == torch.float32 else (2e-2 if dtype == torch.bfloat16 else 5e-3)
    torch.testing.assert_close(
        key_cache[block_id], k_expected_4d, atol=atol, rtol=atol
    )
    # Pass-through dimensions [rotary_dim:] must be byte-identical!
    torch.testing.assert_close(
        key_cache[block_id, :, :, rotary_dim:],
        k_orig_4d[:, :, rotary_dim:],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rotate_key_cache_rocm_head_major_layout_and_runner_hook(dtype: torch.dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    head_size = 256
    rotary_dim = 64
    num_kv_heads = 2
    block_size = 16
    num_blocks = 6
    block_id = 3
    delta_pos = -144

    rope = _make_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        dtype=torch.float32,
    ).to(device)
    inv_freq = rope._compute_inv_freq(rope.base).to(device)

    orig_positions = torch.arange(300, 300 + block_size, device=device, dtype=torch.long)
    shifted_positions = orig_positions + delta_pos

    raw_q = torch.randn(
        block_size, num_kv_heads * head_size, device=device, dtype=torch.float32
    )
    raw_k = torch.randn(
        block_size, num_kv_heads * head_size, device=device, dtype=torch.float32
    )
    _, k_at_orig = rope.forward_native(orig_positions, raw_q.clone(), raw_k.clone())
    _, k_at_shifted = rope.forward_native(
        shifted_positions, raw_q.clone(), raw_k.clone()
    )
    assert k_at_orig is not None and k_at_shifted is not None

    # Build LHBNC KV cache tensor `[num_blocks, 2, block_size, num_kv_heads * head_size]`
    kv_cache = torch.zeros(
        (num_blocks, 2, block_size, num_kv_heads * head_size),
        device=device,
        dtype=dtype,
    )
    key_cache, _ = PagedAttention.split_kv_cache(
        kv_cache.transpose(0, 1), num_kv_heads, head_size
    )
    x = 16 // kv_cache.element_size()

    # Pack k_at_orig into 5D key_cache[block_id]: [num_kv_heads, head_size // x, block_size, x]
    k_orig_packed = (
        k_at_orig.to(dtype)
        .view(block_size, num_kv_heads, head_size // x, x)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    k_expected_packed = (
        k_at_shifted.to(dtype)
        .view(block_size, num_kv_heads, head_size // x, x)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    key_cache[block_id].copy_(k_orig_packed)

    class _DummyImpl:
        kv_cache_dtype = "auto"

    class _DummyAttnLayer:
        def __init__(self, cache: torch.Tensor):
            self.kv_cache = cache
            self.num_kv_heads = num_kv_heads
            self.head_size = head_size
            self.impl = _DummyImpl()
            self._k_scale = torch.ones(1, device=device, dtype=torch.float32)

    forward_ctx = {"layers.3.self_attn.attn": _DummyAttnLayer(kv_cache)}
    rotations = [
        KVCacheBlockRoPERotation(
            block_id=block_id,
            delta_pos=delta_pos,
            start_offset=0,
            end_offset=block_size,
        )
    ]

    apply_kv_cache_rope_rotations_inplace(
        static_forward_context=forward_ctx,
        num_blocks=num_blocks,
        rotations=rotations,
        inv_freq=inv_freq,
        is_neox_style=True,
    )

    atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(
        key_cache[block_id], k_expected_packed, atol=atol, rtol=atol
    )
