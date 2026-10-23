# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leyline-style in-place KV-cache key RoPE delta rotation.

When a mid-history context edit omits or replaces a token span [s, e) with a
shorter span of length r < (e - s), suffix tokens originally cached at position
p >= e shift to p' = p + delta_pos, where delta_pos = r - (e - s). Because RoPE
is an orthogonal 2D rotation per frequency pair:

    R(p + delta_pos) * k = R(delta_pos) * (R(p) * k)

the cached post-RoPE keys in a cloned (CoW) KV block can be rotated in-place by
delta_pos without re-projecting or re-running the prefix transformer layers.
"""

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockRoPERotation,
    KVCacheBlockSplice,
)

FP8_MIN, FP8_MAX = get_fp8_min_max()


@triton.jit
def _rotate_key_cache_neox_head_major_kernel(
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    cos_ptr,  # [half_rd] float32
    sin_ptr,  # [half_rd] float32
    k_scale_ptr,  # scalar float32
    block_id: tl.int64,
    start_offset: tl.int32,
    end_offset: tl.int32,
    stride_blk: tl.int64,
    stride_head: tl.int64,
    stride_dim_x: tl.int64,
    stride_slot: tl.int64,
    num_kv_heads: tl.constexpr,
    half_rd: tl.constexpr,
    x: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    FP8_MIN_VAL: tl.constexpr,
    FP8_MAX_VAL: tl.constexpr,
):
    slot_idx = start_offset + tl.program_id(0)
    if slot_idx >= end_offset:
        return
    head_idx = tl.program_id(1)
    if head_idx >= num_kv_heads:
        return

    pair_offs = tl.arange(0, half_rd)
    cos_val = tl.load(cos_ptr + pair_offs).to(tl.float32)
    sin_val = tl.load(sin_ptr + pair_offs).to(tl.float32)

    dim1 = pair_offs
    dim2 = pair_offs + half_rd

    base_ptr = key_cache_ptr + block_id * stride_blk + head_idx * stride_head
    addr1 = (
        base_ptr
        + (dim1 // x) * stride_dim_x
        + slot_idx * stride_slot
        + (dim1 % x)
    )
    addr2 = (
        base_ptr
        + (dim2 // x) * stride_dim_x
        + slot_idx * stride_slot
        + (dim2 % x)
    )

    k1_raw = tl.load(addr1).to(tl.float32)
    k2_raw = tl.load(addr2).to(tl.float32)

    if FP8_KV_CACHE:
        scale = tl.load(k_scale_ptr).to(tl.float32)
        k1 = k1_raw * scale
        k2 = k2_raw * scale
    else:
        k1 = k1_raw
        k2 = k2_raw

    new_k1 = k1 * cos_val - k2 * sin_val
    new_k2 = k2 * cos_val + k1 * sin_val

    if FP8_KV_CACHE:
        inv_scale = 1.0 / tl.load(k_scale_ptr).to(tl.float32)
        out1 = tl.clamp(new_k1 * inv_scale, FP8_MIN_VAL, FP8_MAX_VAL)
        out2 = tl.clamp(new_k2 * inv_scale, FP8_MIN_VAL, FP8_MAX_VAL)
    else:
        out1 = new_k1
        out2 = new_k2

    tl.store(addr1, out1)
    tl.store(addr2, out2)


@triton.jit
def _rotate_key_cache_neox_token_major_kernel(
    key_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
    cos_ptr,  # [half_rd] float32
    sin_ptr,  # [half_rd] float32
    k_scale_ptr,  # scalar float32
    block_id: tl.int64,
    start_offset: tl.int32,
    end_offset: tl.int32,
    stride_blk: tl.int64,
    stride_slot: tl.int64,
    stride_head: tl.int64,
    num_kv_heads: tl.constexpr,
    half_rd: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    FP8_MIN_VAL: tl.constexpr,
    FP8_MAX_VAL: tl.constexpr,
):
    slot_idx = start_offset + tl.program_id(0)
    if slot_idx >= end_offset:
        return
    head_idx = tl.program_id(1)
    if head_idx >= num_kv_heads:
        return

    pair_offs = tl.arange(0, half_rd)
    cos_val = tl.load(cos_ptr + pair_offs).to(tl.float32)
    sin_val = tl.load(sin_ptr + pair_offs).to(tl.float32)

    base_ptr = (
        key_cache_ptr
        + block_id * stride_blk
        + slot_idx * stride_slot
        + head_idx * stride_head
    )
    addr1 = base_ptr + pair_offs
    addr2 = base_ptr + pair_offs + half_rd

    k1_raw = tl.load(addr1).to(tl.float32)
    k2_raw = tl.load(addr2).to(tl.float32)

    if FP8_KV_CACHE:
        scale = tl.load(k_scale_ptr).to(tl.float32)
        k1 = k1_raw * scale
        k2 = k2_raw * scale
    else:
        k1 = k1_raw
        k2 = k2_raw

    new_k1 = k1 * cos_val - k2 * sin_val
    new_k2 = k2 * cos_val + k1 * sin_val

    if FP8_KV_CACHE:
        inv_scale = 1.0 / tl.load(k_scale_ptr).to(tl.float32)
        out1 = tl.clamp(new_k1 * inv_scale, FP8_MIN_VAL, FP8_MAX_VAL)
        out2 = tl.clamp(new_k2 * inv_scale, FP8_MIN_VAL, FP8_MAX_VAL)
    else:
        out1 = new_k1
        out2 = new_k2

    tl.store(addr1, out1)
    tl.store(addr2, out2)


def compute_delta_cos_sin(
    inv_freq: torch.Tensor,
    delta_pos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos(delta_pos * inv_freq), sin(delta_pos * inv_freq)) in fp32."""
    inv_freq_f32 = inv_freq.to(device=device, dtype=torch.float32)
    angles = float(delta_pos) * inv_freq_f32
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


def rotate_key_cache_block_inplace(
    key_cache: torch.Tensor,
    block_id: int,
    delta_pos: int,
    inv_freq: torch.Tensor,
    start_offset: int = 0,
    end_offset: int | None = None,
    is_neox_style: bool = True,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
) -> None:
    """Rotate cached keys in `key_cache[block_id]` in-place by `delta_pos`.

    Supports both:
    - 5D head-major x-packed layout:
      `[num_blocks, num_kv_heads, head_size // x, block_size, x]` (ROCm PagedAttention)
    - 4D token-major Flash layout:
      `[num_blocks, block_size, num_kv_heads, head_size]`
    as well as FP8 quantized KV caches (`fp8`, `fp8_e4m3`, `fp8_e5m2`).
    """
    if delta_pos == 0:
        return

    is_fp8 = is_quantized_kv_cache(kv_cache_dtype)
    if is_fp8:
        fp8_dtype = current_platform.fp8_dtype()
        if key_cache.dtype != fp8_dtype:
            key_cache = key_cache.view(fp8_dtype)
    if k_scale is None:
        k_scale = torch.ones(1, dtype=torch.float32, device=key_cache.device)

    cos_delta, sin_delta = compute_delta_cos_sin(
        inv_freq, delta_pos, device=key_cache.device
    )
    half_rd = int(cos_delta.numel())
    rotary_dim = half_rd * 2

    use_head_major = key_cache.ndim == 5
    if use_head_major:
        num_kv_heads = int(key_cache.shape[1])
        block_size = int(key_cache.shape[3])
        x = int(key_cache.shape[4])
    elif key_cache.ndim == 4:
        block_size = int(key_cache.shape[1])
        num_kv_heads = int(key_cache.shape[2])
        x = 1
    else:
        raise ValueError(f"Unsupported key_cache shape: {tuple(key_cache.shape)}")

    s_off = max(0, int(start_offset))
    e_off = block_size if end_offset is None else min(block_size, int(end_offset))
    num_slots = e_off - s_off
    if num_slots <= 0:
        return

    # Use Triton fast path when NeoX split-half and half_rd is a power of 2
    if is_neox_style and (half_rd & (half_rd - 1)) == 0:
        grid = (num_slots, num_kv_heads)
        if use_head_major:
            _rotate_key_cache_neox_head_major_kernel[grid](
                key_cache_ptr=key_cache,
                cos_ptr=cos_delta,
                sin_ptr=sin_delta,
                k_scale_ptr=k_scale,
                block_id=block_id,
                start_offset=s_off,
                end_offset=e_off,
                stride_blk=key_cache.stride(0),
                stride_head=key_cache.stride(1),
                stride_dim_x=key_cache.stride(2),
                stride_slot=key_cache.stride(3),
                num_kv_heads=num_kv_heads,
                half_rd=half_rd,
                x=x,
                FP8_KV_CACHE=is_fp8,
                FP8_MIN_VAL=float(FP8_MIN),
                FP8_MAX_VAL=float(FP8_MAX),
                num_warps=1,
            )
            return
        else:
            _rotate_key_cache_neox_token_major_kernel[grid](
                key_cache_ptr=key_cache,
                cos_ptr=cos_delta,
                sin_ptr=sin_delta,
                k_scale_ptr=k_scale,
                block_id=block_id,
                start_offset=s_off,
                end_offset=e_off,
                stride_blk=key_cache.stride(0),
                stride_slot=key_cache.stride(1),
                stride_head=key_cache.stride(2),
                num_kv_heads=num_kv_heads,
                half_rd=half_rd,
                FP8_KV_CACHE=is_fp8,
                FP8_MIN_VAL=float(FP8_MIN),
                FP8_MAX_VAL=float(FP8_MAX),
                num_warps=1,
            )
            return

    # PyTorch fallback for non-power-of-2 rotary dims or GPT-J interleaved style
    scale_val = float(k_scale.item()) if is_fp8 else 1.0
    if use_head_major:
        # [num_kv_heads, head_size // x, num_slots, x] -> [num_slots, num_kv_heads, head_size]
        blk_view = key_cache[block_id, :, :, s_off:e_off, :]
        head_size = blk_view.shape[1] * x
        k_slots = (
            blk_view.permute(2, 0, 1, 3)
            .reshape(num_slots, num_kv_heads, head_size)
            .to(torch.float32)
        )
    else:
        blk_view = key_cache[block_id, s_off:e_off, :, :]
        head_size = blk_view.shape[-1]
        k_slots = blk_view.to(torch.float32)

    if is_fp8:
        k_slots = k_slots * scale_val

    k_rot = k_slots[..., :rotary_dim]
    if is_neox_style:
        k1 = k_rot[..., :half_rd]
        k2 = k_rot[..., half_rd:rotary_dim]
        new_k1 = k1 * cos_delta - k2 * sin_delta
        new_k2 = k2 * cos_delta + k1 * sin_delta
        k_slots[..., :half_rd] = new_k1
        k_slots[..., half_rd:rotary_dim] = new_k2
    else:
        k1 = k_rot[..., 0::2]
        k2 = k_rot[..., 1::2]
        new_k1 = k1 * cos_delta - k2 * sin_delta
        new_k2 = k2 * cos_delta + k1 * sin_delta
        k_rot[..., 0::2] = new_k1
        k_rot[..., 1::2] = new_k2

    if is_fp8:
        k_slots = torch.clamp(k_slots / scale_val, float(FP8_MIN), float(FP8_MAX))
        out_tensor = k_slots.to(key_cache.dtype)
    else:
        out_tensor = k_slots.to(key_cache.dtype)

    if use_head_major:
        reshaped = out_tensor.reshape(num_slots, num_kv_heads, head_size // x, x).permute(
            1, 2, 0, 3
        )
        key_cache[block_id, :, :, s_off:e_off, :] = reshaped
    else:
        key_cache[block_id, s_off:e_off, :, :] = out_tensor


def apply_kv_cache_rope_rotations_inplace(
    static_forward_context: dict[str, object],
    num_blocks: int,
    rotations: Sequence[KVCacheBlockRoPERotation],
    inv_freq: torch.Tensor,
    is_neox_style: bool = True,
) -> None:
    """Apply scheduled RoPE delta rotations across all full-attention layers."""
    if not rotations:
        return

    seen_ptrs: set[tuple[torch.device, int]] = set()
    for layer in static_forward_context.values():
        kv_cache = getattr(layer, "kv_cache", None)
        impl = getattr(layer, "impl", None)
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            continue
        if not hasattr(layer, "num_kv_heads") or not hasattr(layer, "head_size"):
            continue
        # Only full/sliding-window attention layers with 2 head slots (K and V)
        # store RoPE keys; Mamba state tensors are skipped.
        if kv_cache.ndim < 4 or kv_cache.shape[1] != 2:
            continue
        ptr_key = (kv_cache.device, kv_cache.data_ptr())
        if ptr_key in seen_ptrs:
            continue
        seen_ptrs.add(ptr_key)

        num_kv_heads = int(layer.num_kv_heads)
        head_size = int(layer.head_size)
        kv_cache_dtype = getattr(impl, "kv_cache_dtype", "auto")
        k_scale = getattr(layer, "_k_scale", None)

        # Split [B, 2, N, H*hs] -> key_cache [K_blocks, num_kv_heads, hs//x, kernel_block_size, x]
        key_cache, _ = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), num_kv_heads, head_size
        )
        total_kernel_blocks = int(key_cache.shape[0])
        kernel_block_size = int(key_cache.shape[3])
        kernel_blocks_per_sched_block = max(1, total_kernel_blocks // max(1, num_blocks))
        sched_block_tokens = kernel_blocks_per_sched_block * kernel_block_size

        for rot in rotations:
            if rot.delta_pos == 0:
                continue
            s_tok = max(0, int(rot.start_offset))
            e_tok = (
                sched_block_tokens
                if int(rot.end_offset) <= 0
                else min(sched_block_tokens, int(rot.end_offset))
            )
            if s_tok >= e_tok:
                continue

            base_kblk = int(rot.block_id) * kernel_blocks_per_sched_block
            first_sub = s_tok // kernel_block_size
            last_sub = (e_tok - 1) // kernel_block_size
            for sub in range(first_sub, last_sub + 1):
                sub_start = max(0, s_tok - sub * kernel_block_size)
                sub_end = min(kernel_block_size, e_tok - sub * kernel_block_size)
                rotate_key_cache_block_inplace(
                    key_cache=key_cache,
                    block_id=base_kblk + sub,
                    delta_pos=int(rot.delta_pos),
                    inv_freq=inv_freq,
                    start_offset=sub_start,
                    end_offset=sub_end,
                    is_neox_style=is_neox_style,
                    kv_cache_dtype=kv_cache_dtype,
                    k_scale=k_scale,
                )


def fuse_copy_rotate_key_value_block_range(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    src_block_id: int,
    dst_block_id: int,
    src_start: int,
    src_end: int,
    dst_start: int,
    dst_end: int,
    delta_pos: int,
    inv_freq: torch.Tensor,
    is_neox_style: bool = True,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
) -> None:
    """Offset-aware fused copy+rotate of a donor KV block range into a target
    block (the single new op from design §2b).

    The donor's already-RoPE'd cached keys for token slots ``[src_start,
    src_end)`` are copied into the target's slots ``[dst_start, dst_end)``:
    V is copied verbatim (RoPE never touches values), K is copied with the in
    place RoPE delta rotation of ``delta_pos`` applied in the same pass
    (R(dst_pos)k == R(delta)R(src_pos)k). This expresses a non-zero intra-block
    shift (``dst_start - src_start == delta_pos``) that the whole-block
    ``copy_kv_cache_blocks_inplace`` cannot (A1). The range is a token-aligned
    sub-block: ``src_end - src_start == dst_end - dst_start``. ``delta_pos == 0``
    is a pure verbatim copy. ``dst_block_id == src_block_id`` (in place) is a
    rotation-only fast path.
    """
    if dst_start >= dst_end or src_start >= src_end:
        return
    assert src_end - src_start == dst_end - dst_start, (
        f"splice range length mismatch: src ({src_start}, {src_end}) vs "
        f"dst ({dst_start}, {dst_end})"
    )

    use_head_major = key_cache.ndim == 5

    # --- V: verbatim copy. value_cache is [num_blocks, num_kv_heads, head_size,
    # block_size] in both layouts; the slot index is always the last dim. ---
    value_cache[dst_block_id, :, :, dst_start:dst_end] = value_cache[
        src_block_id, :, :, src_start:src_end
    ]

    # --- K: copy (unless in place), then rotate the copied range in place. ---
    if dst_block_id != src_block_id:
        if use_head_major:
            # key_cache [num_blocks, num_kv_heads, head_size // x, block_size, x]
            key_cache[dst_block_id, :, :, dst_start:dst_end, :] = key_cache[
                src_block_id, :, :, src_start:src_end, :
            ]
        else:
            # key_cache [num_blocks, block_size, num_kv_heads, head_size]
            key_cache[dst_block_id, dst_start:dst_end, :, :] = key_cache[
                src_block_id, src_start:src_end, :, :
            ]

    if delta_pos == 0:
        return

    rotate_key_cache_block_inplace(
        key_cache=key_cache,
        block_id=dst_block_id,
        delta_pos=delta_pos,
        inv_freq=inv_freq,
        start_offset=dst_start,
        end_offset=dst_end,
        is_neox_style=is_neox_style,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
    )


def apply_kv_cache_block_splices_inplace(
    static_forward_context: dict[str, object],
    num_blocks: int,
    splices: Sequence[KVCacheBlockSplice],
    inv_freq: torch.Tensor,
    is_neox_style: bool = True,
) -> None:
    """Apply scheduled fused copy+rotate splices across all full-attention
    layers (the §2b phase-2 worker op).

    Each ``KVCacheBlockSplice`` copies the donor's already-RoPE'd cached KV for
    token slots ``[src_start, src_end)`` of ``src_block_id`` into the target's
    slots ``[dst_start, dst_end)`` of ``dst_block_id``, rotating K by the
    absolute RoPE delta ``delta_pos`` in the same pass. Slot offsets are
    expressed in scheduler-block token units, so the mapping to the
    kernel's sub-blocks (``kernel_blocks_per_sched_block``) mirrors
    ``apply_kv_cache_rope_rotations_inplace``.
    """
    if not splices:
        return

    seen_ptrs: set[tuple[torch.device, int]] = set()
    for layer in static_forward_context.values():
        kv_cache = getattr(layer, "kv_cache", None)
        impl = getattr(layer, "impl", None)
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            continue
        if not hasattr(layer, "num_kv_heads") or not hasattr(layer, "head_size"):
            continue
        # Only full/sliding-window attention layers with 2 head slots (K and V)
        # store RoPE keys; Mamba state tensors are skipped.
        if kv_cache.ndim < 4 or kv_cache.shape[1] != 2:
            continue
        ptr_key = (kv_cache.device, kv_cache.data_ptr())
        if ptr_key in seen_ptrs:
            continue
        seen_ptrs.add(ptr_key)

        num_kv_heads = int(layer.num_kv_heads)
        head_size = int(layer.head_size)
        kv_cache_dtype = getattr(impl, "kv_cache_dtype", "auto")
        k_scale = getattr(layer, "_k_scale", None)

        # Split [B, 2, N, H*hs] -> key_cache/value_cache in kernel layout.
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), num_kv_heads, head_size
        )
        total_kernel_blocks = int(key_cache.shape[0])
        kernel_block_size = int(key_cache.shape[3])
        kernel_blocks_per_sched_block = max(1, total_kernel_blocks // max(1, num_blocks))
        sched_block_tokens = kernel_blocks_per_sched_block * kernel_block_size

        for sp in splices:
            s_tok = max(0, int(sp.src_start))
            e_tok = (
                sched_block_tokens
                if int(sp.src_end) <= 0
                else min(sched_block_tokens, int(sp.src_end))
            )
            d_tok = max(0, int(sp.dst_start))
            f_tok = (
                sched_block_tokens
                if int(sp.dst_end) <= 0
                else min(sched_block_tokens, int(sp.dst_end))
            )
            if s_tok >= e_tok or d_tok >= f_tok:
                continue
            assert e_tok - s_tok == f_tok - d_tok, (
                f"splice range length mismatch: src ({s_tok}, {e_tok}) vs "
                f"dst ({d_tok}, {f_tok})"
            )
            src_base = int(sp.src_block_id) * kernel_blocks_per_sched_block
            dst_base = int(sp.dst_block_id) * kernel_blocks_per_sched_block
            # The [src_start, src_end) and [dst_start, dst_end) ranges are in
            # scheduler-block token units and may (a) straddle several kernel
            # sub-blocks when a scheduler block maps to multiple kernel blocks
            # and (b) have a non-zero intra-block shift (dst_start - src_start).
            # Decompose the logical range [0, L) (L = range length) at every
            # point where the src local position OR the dst local position
            # crosses a kernel sub-block boundary, so each fragment lies wholly
            # inside one src sub-block and one dst sub-block.
            L = e_tok - s_tok
            cuts = {0, L}
            b = kernel_block_size
            while b < sched_block_tokens:
                for origin in (s_tok, d_tok):
                    t = b - origin  # logical offset where origin + t == b
                    if 0 < t < L:
                        cuts.add(t)
                b += kernel_block_size
            cuts = sorted(cuts)
            for t0, t1 in zip(cuts, cuts[1:]):
                src_sub = (s_tok + t0) // kernel_block_size
                dst_sub = (d_tok + t0) // kernel_block_size
                fuse_copy_rotate_key_value_block_range(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    src_block_id=src_base + src_sub,
                    dst_block_id=dst_base + dst_sub,
                    src_start=(s_tok + t0) - src_sub * kernel_block_size,
                    src_end=(s_tok + t1) - src_sub * kernel_block_size,
                    dst_start=(d_tok + t0) - dst_sub * kernel_block_size,
                    dst_end=(d_tok + t1) - dst_sub * kernel_block_size,
                    delta_pos=int(sp.delta_pos),
                    inv_freq=inv_freq,
                    is_neox_style=is_neox_style,
                    kv_cache_dtype=kv_cache_dtype,
                    k_scale=k_scale,
                )
