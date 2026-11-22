"""Group-routed copies for explicitly marked Inkling2 target/draft caches.

All pass caches use [B,H,T,D]. Target attention and conv history copy
verbatim; the DSpark draft's rotary keys additionally receive a RoPE delta.
Never use this path for an unmarked cache topology.
"""

import torch
from vllm.v1.core.leyline_relative import is_relative_topology


def splice_jump_request_ids(output):
    """Explicit splice jumps only; never infer jumps from async counters."""
    return {
        op.req_id
        for op in (
            *(getattr(output, "mamba_state_transplants", None) or ()),
            *(getattr(output, "kv_cache_block_splices", None) or ()),
        )
        if getattr(op, "req_id", None) is not None
    }


def apply_relative_splices(context, config, splices):
    if not is_relative_topology(config):
        raise ValueError("Relative splice requires an explicitly marked topology")
    # Validate the complete publication before writing any destination.
    copies = []
    for sp in splices:
        gid = sp.kv_cache_group_id
        if gid is None or not 0 <= gid < len(config.kv_cache_groups):
            raise ValueError("Relative splice is missing its cache group")
        group = config.kv_cache_groups[gid]
        if sp.src_end - sp.src_start != sp.dst_end - sp.dst_start:
            raise ValueError("Relative splice length mismatch")
        for name in group.layer_names:
            cache = context[name].kv_cache
            if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
                raise ValueError("Relative splice expects [B,H,T,D] storage")
            inv_freq = None
            head_dim = None
            if (
                getattr(group.kv_cache_spec, "leyline_position_encoding", None)
                == "rotary"
            ):
                inv_freq = getattr(context[name], "leyline_rope_inv_freq", None)
                head_dim = getattr(context[name], "head_size", None)
                if (
                    inv_freq is None
                    or head_dim is None
                    or cache.shape[-1] != 2 * head_dim
                ):
                    raise ValueError(
                        "Draft rotary cache has no verified rotation/layout contract"
                    )
            factor = cache.shape[0] // config.num_blocks
            if factor < 1 or factor * config.num_blocks != cache.shape[0]:
                raise ValueError("Relative splice has an invalid block mapping")
            page = cache.shape[2]
            limit = page * factor
            if not (
                0 <= sp.src_start < sp.src_end <= limit
                and 0 <= sp.dst_start < sp.dst_end <= limit
            ):
                raise ValueError("Relative splice offsets outside scheduler block")
            done = 0
            while done < sp.src_end - sp.src_start:
                src = sp.src_start + done
                dst = sp.dst_start + done
                count = min(
                    sp.src_end - sp.src_start - done,
                    page - src % page,
                    page - dst % page,
                )
                sb = sp.src_block_id * factor + src // page
                db = sp.dst_block_id * factor + dst // page
                if (
                    not (0 <= sb < cache.shape[0] and 0 <= db < cache.shape[0])
                    or sb == db
                ):
                    raise ValueError(
                        "Relative splice endpoints must be separate valid blocks"
                    )
                copies.append(
                    (
                        cache,
                        sb,
                        db,
                        src % page,
                        dst % page,
                        count,
                        inv_freq,
                        head_dim,
                        sp.delta_pos,
                    )
                )
                done += count
    for cache, sb, db, src, dst, count, inv_freq, head_dim, delta in copies:
        cache[db, :, dst : dst + count, :].copy_(cache[sb, :, src : src + count, :])
        if inv_freq is not None and delta:
            from vllm.v1.attention.ops.leyline_rope import (
                rotate_key_cache_block_inplace,
            )

            rotate_key_cache_block_inplace(
                key_cache=cache.transpose(1, 2)[..., :head_dim],
                block_id=db,
                delta_pos=delta,
                inv_freq=inv_freq,
                start_offset=dst,
                end_offset=dst + count,
                is_neox_style=True,
                kv_cache_dtype="auto",
            )
