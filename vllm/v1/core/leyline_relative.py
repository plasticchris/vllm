"""Conservative Leyline planning for explicitly relative-position cache groups.

Unlike rotary/Mamba models, every stored state is a token-indexed cell. Full
attention needs the entire reused suffix; sliding attention and short convs
need their boundary history only. Every required donor cell is checked and
pinned before allocation. Commit uses the existing atomic publication/fence.
"""

import math
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.core.kv_cache_utils import resolve_block_hashes


def has_relative_cache_groups(config):
    return any(
        getattr(g.kv_cache_spec, "leyline_position_encoding", None) == "relative"
        for g in getattr(config, "kv_cache_groups", ())
    )


def is_relative_topology(config):
    groups = getattr(config, "kv_cache_groups", ())
    encodings = [
        getattr(g.kv_cache_spec, "leyline_position_encoding", None) for g in groups
    ]
    return "relative" in encodings and all(
        isinstance(g.kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec))
        and encoding in ("relative", "rotary")
        for g, encoding in zip(groups, encodings)
    )


def prepare_relative_splice(manager, request, s, e, r, length):
    from vllm.v1.core.leyline_splice import LeylineSplicePlan, _fallback, logger

    groups = manager.kv_cache_config.kv_cache_groups
    managers = manager.coordinator.single_type_managers
    full = [
        i
        for i, g in enumerate(groups)
        if isinstance(g.kv_cache_spec, FullAttentionSpec)
    ]
    if not full:
        _fallback(request, "relative_no_full_attention")
        return None
    grid = math.lcm(*(mgr.block_size for mgr in managers))
    max_window = max(
        (getattr(g.kv_cache_spec, "sliding_window", 0) or 0) for g in groups
    )
    boundary = (e + length - 1) // grid * grid
    # Require the entire sliding history to lie in the matched suffix.
    if boundary - e < max_window + grid:
        _fallback(request, "relative_suffix_too_short")
        return None
    delta = s + r - e
    ranges = []
    for gid, (group, mgr) in enumerate(zip(groups, managers)):
        bs = mgr.block_size
        hashes = resolve_block_hashes(
            request.donor_block_hashes, manager.block_pool.hash_block_size, bs
        )
        window = getattr(group.kv_cache_spec, "sliding_window", None)
        pos = e if window is None else max(e, boundary - (window - 1))
        while pos < boundary:
            index, offset = divmod(pos, bs)
            if index >= len(hashes):
                _fallback(request, "relative_donor_hash_missing")
                return None
            found = manager.block_pool.get_cached_block(hashes[index], [gid])
            if (
                not found
                or found[0].is_null
                or found[0].block_hash in getattr(mgr, "cached_blocks_this_step", ())
            ):
                _fallback(request, "relative_donor_cache_missing")
                return None
            size = min(bs - offset, boundary - pos)
            ranges.append((gid, found[0], offset, pos + delta, size))
            pos += size
    pins = list({block.block_id: block for _, block, *_ in ranges}.values())
    manager.block_pool.touch(pins)
    plan = LeylineSplicePlan(s + r, boundary + delta, delta, ranges, [], pins)
    request._leyline_max_cacheable_tokens = s + r
    request.leyline_splice_plan = plan
    logger.info(
        "[leyline-relative] req=%s planned middle=%d target=%d groups=%d ranges=%d",
        request.request_id,
        plan.middle_end,
        plan.target_end,
        len(groups),
        len(ranges),
    )
    return plan
