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


# ---------------------------------------------------------------------------
# Offset-aware fused copy+rotate op (design §2b "one new fused op").
#
# The whole-block ``copy_kv_cache_blocks_inplace`` cannot express a non-zero
# intra-block shift (A1); §2b needs a donor range copied into a shifted target
# range with the RoPE delta applied in the same pass. These tests verify the
# op on CPU (GPT-J style forces the torch fallback, so no GPU process is
# spawned) by checking the RoPE algebraic identity
#     R(dst_pos) k == R(delta) R(src_pos) k
# plus verbatim V and intactness of non-copied slots / the donor block.
# ---------------------------------------------------------------------------


def _make_inv_freq(head_size: int, base: float = 10_000_000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))


def _rope_gptj(pos: int, x: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """GPT-J (interleaved) rotate of x [..., head_size] by position pos (fp32)."""
    rd = x.shape[-1]
    cs = torch.tensor([float(pos)]).float() * inv_freq
    cos = torch.cos(cs)
    sin = torch.sin(cs)
    out = x.to(torch.float32).clone()
    out[..., 0::2] = x.to(torch.float32)[..., 0::2] * cos - x.to(torch.float32)[..., 1::2] * sin
    out[..., 1::2] = x.to(torch.float32)[..., 1::2] * cos + x.to(torch.float32)[..., 0::2] * sin
    return out


def test_fuse_copy_rotate_gptj_cpu_verbatim_v_and_rotated_k():
    from vllm.v1.attention.ops.leyline_rope import fuse_copy_rotate_key_value_block_range

    head_size = 64
    num_kv_heads = 2
    block_size = 16
    inv_freq = _make_inv_freq(head_size)

    # Donor cached K = R(src_pos) raw; donor cached V = v (RoPE never touches it).
    # The donor token is copied from slot src_slot to slot dst_slot; the absolute
    # positional shift is delta = dst_slot - src_slot (negative: tokens spliced
    # out, the realistic §2b case). R(dst_abs) raw == R(src_abs + delta) raw.
    src_pos = 100
    src_slot = 4
    delta = -4  # dst_slot = src_slot + delta = 0 (non-zero intra-block shift)
    dst_slot = src_slot + delta  # 0

    raw = torch.randn(num_kv_heads, head_size)
    donor_k = _rope_gptj(src_pos, raw, inv_freq)
    donor_v = torch.randn(num_kv_heads, head_size)

    # Key: 4-dim token-major [num_blocks, block_size, num_kv_heads, head_size]
    # (op's 4-dim key branch). Value: the real split_kv_cache layout
    # [num_blocks, num_kv_heads, head_size, block_size] (slot is the last dim).
    key_cache = torch.zeros(2, block_size, num_kv_heads, head_size)
    value_cache = torch.zeros(2, num_kv_heads, head_size, block_size)
    key_cache[0, src_slot] = donor_k
    value_cache[0, :, :, src_slot] = donor_v
    # Sentinels that must remain untouched.
    value_cache[0, :, :, 11] = torch.randn(num_kv_heads, head_size)
    keep_v011 = value_cache[0, :, :, 11].clone()
    donor_keep = key_cache[0].clone()

    fuse_copy_rotate_key_value_block_range(
        key_cache=key_cache,
        value_cache=value_cache,
        src_block_id=0,
        dst_block_id=1,
        src_start=src_slot,
        src_end=src_slot + 1,
        dst_start=dst_slot,
        dst_end=dst_slot + 1,
        delta_pos=delta,
        inv_freq=inv_freq,
        is_neox_style=False,  # GPT-J -> torch fallback (CPU)
    )

    # K: the target slot holds R(dst_pos) raw = R(src_pos + delta) raw (fp32).
    expected_k = _rope_gptj(src_pos + delta, raw, inv_freq)
    torch.testing.assert_close(key_cache[1, dst_slot], expected_k, atol=1e-5, rtol=1e-5)
    # V: verbatim copy of the donor slot.
    torch.testing.assert_close(value_cache[1, :, :, dst_slot], donor_v, atol=0, rtol=0)
    # Integrity: donor block and non-copied target slots are untouched.
    torch.testing.assert_close(key_cache[0], donor_keep, atol=0, rtol=0)
    torch.testing.assert_close(value_cache[0, :, :, 11], keep_v011, atol=0, rtol=0)
    torch.testing.assert_close(
        key_cache[1, dst_slot + 1:],
        torch.zeros(block_size - 1 - dst_slot, num_kv_heads, head_size),
        atol=0,
        rtol=0,
    )


def test_fuse_copy_rotate_head_major_cpu_verbatim_delta0():
    """Head-major layout, delta_pos=0 (pure verbatim copy) — exercises the
    head-major copy-index path on CPU with a multi-token intra-block range."""
    from vllm.v1.attention.ops.leyline_rope import fuse_copy_rotate_key_value_block_range
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    head_size = 256
    num_kv_heads = 4
    block_size = 16
    x = 16 // 2  # fp16

    packed = torch.randn(2, 2, num_kv_heads, head_size, block_size, dtype=torch.float16)
    packed = packed.reshape(2, 2, num_kv_heads * head_size * block_size)
    key_cache, value_cache = PagedAttention.split_kv_cache(packed, num_kv_heads, head_size)
    assert key_cache.ndim == 5 and key_cache.shape[3] == block_size

    # Copy donor block 0 slots [4:8] into target block 1 slots [0:4], no rotation.
    donor_k = key_cache[0, :, :, 4:8, :].clone()
    donor_v = value_cache[0, :, :, 4:8].clone()
    # Sentinels.
    target_keep = key_cache[1, :, :, 4:, :].clone()
    donor_keep = key_cache[0].clone()

    fuse_copy_rotate_key_value_block_range(
        key_cache=key_cache,
        value_cache=value_cache,
        src_block_id=0,
        dst_block_id=1,
        src_start=4,
        src_end=8,
        dst_start=0,
        dst_end=4,
        delta_pos=0,
        inv_freq=_make_inv_freq(head_size),
        is_neox_style=True,
    )

    torch.testing.assert_close(key_cache[1, :, :, 0:4, :], donor_k, atol=0, rtol=0)
    torch.testing.assert_close(value_cache[1, :, :, 0:4], donor_v, atol=0, rtol=0)
    torch.testing.assert_close(key_cache[1, :, :, 4:, :], target_keep, atol=0, rtol=0)
    torch.testing.assert_close(key_cache[0], donor_keep, atol=0, rtol=0)
