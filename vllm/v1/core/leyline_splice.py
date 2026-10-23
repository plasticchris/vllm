# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, two-phase planning for hybrid amortize splices.

Planning pins only immutable donor blocks. Destination validation and publication
are atomic, after allocation; failure leaves no worker operations behind.
"""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    KVCacheBlockSplice,
    MambaStateTransplant,
    resolve_block_hashes,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LeylineSplicePlan:
    middle_end: int
    target_end: int
    delta: int
    # (group, donor block, source offset, target position, token count)
    ranges: list[tuple[int, KVCacheBlock, int, int, int]]
    states: list[tuple[int, KVCacheBlock, int]]
    pins: list[KVCacheBlock]


def _fallback(request: "Request", reason: str) -> None:
    request._leyline_max_cacheable_tokens = None
    request.kv_splice_stats = {"fallback": reason, "skipped_tokens": 0}
    logger.info(
        "[leyline] req=%s fallback=%s nc=%s prompt=%s",
        request.request_id,
        reason,
        getattr(request, "num_computed_tokens", None),
        getattr(request, "num_prompt_tokens", None),
    )


def cancel_splice(manager: "KVCacheManager", request: "Request") -> None:
    plan = getattr(request, "leyline_splice_plan", None)
    if plan is not None:
        manager.block_pool.free_blocks(plan.pins)
        request.leyline_splice_plan = None
    if not getattr(request, "_leyline_splice_applied", False):
        request._leyline_max_cacheable_tokens = None


def prepare_splice(
    manager: "KVCacheManager", request: "Request"
) -> LeylineSplicePlan | None:
    """Validate a single edit and pin its donor before any target allocation."""
    plan = getattr(request, "leyline_splice_plan", None)
    if plan is not None:
        return plan
    if getattr(request, "leyline_splice_checked", False):
        return None
    request.leyline_splice_checked = True
    params = request.sampling_params
    extra = (params.extra_args or {}) if params is not None else {}
    if extra.get("kv_splice_mode", "off") != "amortize":
        return None
    if not manager.kv_cache_config.has_mamba_layers:
        return None
    directives = request.kv_splice_directives
    if not directives:
        return None
    if len(directives) != 1:
        _fallback(request, "multiple_spans")
        return None
    if not manager.enable_caching or request.skip_reading_prefix_cache:
        _fallback(request, "caching_disabled")
        return None
    if request.mm_features or request.prompt_embeds is not None:
        _fallback(request, "multimodal_or_embeds_unsupported")
        return None
    if request.pooling_params is not None or params.prompt_logprobs is not None:
        _fallback(request, "pooling_or_prompt_logprobs_unsupported")
        return None
    donor = request.kv_splice_donor_tokens
    hashes = request.donor_block_hashes
    if not isinstance(donor, list) or not hashes:
        _fallback(request, "no_donor")
        return None
    try:
        item = directives[0]
        if isinstance(item, dict):
            s = int(item.get("span_start", item.get("s", -1)))
            e = int(item.get("span_end", item.get("e", -1)))
            r = int(item.get("replacement_len", item.get("r", -1)))
            length = int(item.get("suffix_len", item.get("l", -1)))
        else:
            s, e, r, length = map(int, item)
    except (TypeError, ValueError):
        _fallback(request, "invalid_span")
        return None
    middle_end = s + r
    donor_end = e + length
    target = request.all_token_ids
    if not (
        0 <= s < e <= donor_end <= len(donor)
        and 0 <= r < e - s
        and length > 0
        and middle_end + length <= request.num_prompt_tokens
        and list(target[:s]) == donor[:s]
        and list(target[middle_end : middle_end + length]) == donor[e:donor_end]
    ):
        _fallback(request, "token_mismatch_or_invalid_span")
        return None

    groups = manager.kv_cache_config.kv_cache_groups
    managers = manager.coordinator.single_type_managers
    fa = []
    mamba = []
    circular = []
    for gid, (group, mgr) in enumerate(zip(groups, managers)):
        if isinstance(group.kv_cache_spec, CircularBufferSpec):
            circular.append((gid, mgr))
        elif isinstance(group.kv_cache_spec, FullAttentionSpec):
            fa.append((gid, mgr))
        elif isinstance(group.kv_cache_spec, MambaSpec):
            mamba.append((gid, mgr))
        else:
            _fallback(request, "unsupported_cache_topology")
            return None
    if not fa or not mamba:
        _fallback(request, "unsupported_cache_topology")
        return None
    block_size = mamba[0][1].block_size
    if any(mgr.block_size != block_size for _, mgr in fa + mamba):
        _fallback(request, "block_size_mismatch")
        return None
    compress_ratios = [
        int(group.kv_cache_spec.tokens_per_state)
        for gid, _ in fa
        for group in (groups[gid],)
        if isinstance(group.kv_cache_spec.tokens_per_state, int)
        and group.kv_cache_spec.tokens_per_state > 1
    ]
    if compress_ratios:
        cr_lcm = 1
        for cr in compress_ratios:
            if block_size % cr != 0:
                _fallback(request, "compressed_group_unaligned")
                return None
            cr_lcm = math.lcm(cr_lcm, cr)
        # If the suffix match starts inside a compression group, advance
        # both donor `e` and target `middle_end` into the matching suffix
        # until `middle_end` is group-aligned (the prefix before the new
        # `middle_end` is simply forwarded in Phase 1). Both `e` and
        # `middle_end` must share the same modulo `cr_lcm` so `delta` is a
        # multiple of `cr_lcm`.
        if (e - middle_end) % cr_lcm != 0:
            _fallback(request, "compressed_group_unaligned")
            return None
        align_adv = (-middle_end) % cr_lcm
        if align_adv:
            if length <= align_adv:
                _fallback(request, "compressed_group_unaligned")
                return None
            e += align_adv
            r += align_adv
            middle_end += align_adv
            length -= align_adv
    delta = middle_end - e
    resolved = resolve_block_hashes(
        hashes, manager.block_pool.hash_block_size, block_size
    )
    # Full boundaries only: partial-tail entries can still be a producer's
    # writable running state. Sparse retention (retention_interval=0) only
    # caches the donor's reachable replay boundary at (donor_end - 1) // bs * bs,
    # while eagle groups drop one block during prefill and only cache the last
    # block on decode; step down by block_size so we find the highest boundary
    # where BOTH Mamba and all FA groups are resident.
    states = []
    boundary = (donor_end - 1) // block_size * block_size
    while boundary > e:
        states = []
        bidx = boundary // block_size - 1
        if bidx < 0 or bidx >= len(resolved):
            boundary -= block_size
            continue
        for gid, mgr in mamba:
            found = manager.block_pool.get_cached_block(resolved[bidx], [gid])
            if not found or found[0].block_hash in mgr.cached_blocks_this_step:
                break
            states.append((gid, found[0], block_size))
        if len(states) == len(mamba) and all(
            manager.block_pool.get_cached_block(resolved[bidx], [gid])
            for gid, _ in fa
        ):
            break
        boundary -= block_size
    if boundary <= e or len(states) != len(mamba):
        mamba_cached = [
            (i + 1) * block_size
            for i in range(min(len(resolved), donor_end // block_size))
            if manager.block_pool.get_cached_block(resolved[i], [mamba[0][0]])
        ]
        fa_cached = [
            (i + 1) * block_size
            for i in range(min(len(resolved), donor_end // block_size))
            if manager.block_pool.get_cached_block(resolved[i], [fa[0][0]])
        ]
        logger.info(
            "[leyline] req=%s no_donor_boundary e=%d donor_end=%d bs=%d mamba_cached=%s fa_cached=%s",
            request.request_id,
            e,
            donor_end,
            block_size,
            mamba_cached,
            fa_cached,
        )
        _fallback(request, "no_donor_boundary")
        return None
    ranges = []
    for gid, _ in fa:
        pos = e
        while pos < boundary:
            found = manager.block_pool.get_cached_block(
                resolved[pos // block_size], [gid]
            )
            if not found or found[0].is_null:
                _fallback(request, "donor_kv_missing")
                return None
            size = min(block_size - pos % block_size, boundary - pos)
            ranges.append((gid, found[0], pos % block_size, pos + delta, size))
            pos += size
    # touch(), not bare ref_cnt += 1: cached donors at ref_cnt=0 must be
    # removed from the allocator's free queue before it can evict them.
    pins = list({block.block_id: block for _, block, *_ in ranges + states}.values())
    manager.block_pool.touch(pins)
    plan = LeylineSplicePlan(middle_end, boundary + delta, delta, ranges, states, pins)
    request._leyline_max_cacheable_tokens = middle_end
    request.leyline_splice_plan = plan
    logger.info(
        "[leyline] req=%s planned s=%d e=%d r=%d l=%d middle_end=%d target_end=%d delta=%d pins=%d",
        request.request_id,
        s,
        e,
        r,
        length,
        middle_end,
        boundary + delta,
        delta,
        len(pins),
    )
    return plan


def commit_splice(
    manager: "KVCacheManager", request: "Request", num_tail_tokens: int
) -> bool:
    """Publish only after ALL destination blocks and states have validated."""
    plan = getattr(request, "leyline_splice_plan", None)
    if plan is None or num_tail_tokens <= 0:
        return False
    managers = manager.coordinator.single_type_managers
    splices = []
    transplants = []
    destinations = []
    for gid, src, src_offset, target_pos, size in plan.ranges:
        mgr = managers[gid]
        blocks = mgr.req_to_blocks.get(request.request_id, ())
        done = 0
        while done < size:
            pos = target_pos + done
            idx, offset = divmod(pos, mgr.block_size)
            if idx >= len(blocks):
                _fallback(request, "destination_missing")
                return False
            dst = blocks[idx]
            if dst.is_null or dst.ref_cnt != 1 or dst.block_hash is not None:
                _fallback(request, "destination_not_private")
                return False
            count = min(size - done, mgr.block_size - offset)
            splices.append(KVCacheBlockSplice(
                src.block_id, dst.block_id, src_offset + done,
                src_offset + done + count, offset, offset + count, plan.delta,
            ))
            destinations.append(dst)
            done += count
    for gid, src, block_size in plan.states:
        blocks = managers[gid].req_to_blocks.get(request.request_id, ())
        # Use the actual next chunk's write-side column, not a guessed prompt
        # endpoint. This also handles a tail split across target boundaries.
        idx = (plan.target_end + num_tail_tokens - 1) // block_size
        if idx >= len(blocks):
            _fallback(request, "state_destination_missing")
            return False
        dst = blocks[idx]
        if dst.is_null or dst.ref_cnt != 1 or dst.block_hash is not None:
            _fallback(request, "state_destination_not_private")
            return False
        transplants.append(MambaStateTransplant(
            request.request_id, gid, src.block_id, dst.block_id, idx,
        ))
        destinations.append(dst)
    # Publication starts here; no fallible lookups follow. Keep every endpoint
    # alive until the scheduler's processed-step fence acknowledges the drains.
    manager.block_pool.touch(destinations)
    manager._pending_kv_cache_block_splices.extend(splices)
    manager._pending_mamba_state_transplants.extend(transplants)
    manager._pending_kv_cache_block_splice_retained.extend(plan.pins + destinations)
    request.kv_splice_stats = {
        "skipped_tokens": plan.target_end - plan.middle_end,
        "copied_ranges": len(splices),
        "state_transplants": len(transplants),
    }
    logger.info(
        "[leyline] req=%s committed skipped=%d splices=%d transplants=%d tail=%d",
        request.request_id,
        plan.target_end - plan.middle_end,
        len(splices),
        len(transplants),
        num_tail_tokens,
    )
    plan.pins = []
    request.leyline_splice_plan = None
    request._leyline_splice_applied = True
    return True
