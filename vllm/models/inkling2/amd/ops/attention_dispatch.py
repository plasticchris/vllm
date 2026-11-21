"""Bounded split-KV dispatch shared by the target and DSpark draft."""

import torch
from .attention_sm90 import rel_attention, MODE_GATED, MODE_PARTIAL
from .attention_merge import merge_partials
from .tuning import OPTIMIZED, attention_launch_config


def run_attention(
    q,
    key_cache,
    value_cache,
    out,
    rel=None,
    gate=None,
    sinks=None,
    *,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    max_query_len,
    max_seq_len,
    softmax_scale,
    window,
    num_actual_tokens=None,
):
    t, h, d = q.shape
    splits, config = (
        attention_launch_config(
            t,
            cache_seqlens.shape[0],
            max_query_len,
            max_seq_len,
            h,
            key_cache.shape[2],
            d,
            window,
        )
        if OPTIMIZED
        else (1, None)
    )
    kwargs = dict(
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max_query_len,
        softmax_scale=softmax_scale,
        window=window,
        num_splits=splits,
        launch_config=config,
    )
    if splits == 1:
        rel_attention(
            q,
            key_cache,
            value_cache,
            out,
            None,
            rel,
            gate,
            sinks,
            mode=MODE_GATED,
            **kwargs,
        )
        return
    partials = torch.empty(splits, t, h, d, device=q.device, dtype=torch.float32)
    lses = torch.empty(splits, t, h, device=q.device, dtype=torch.float32)
    rel_attention(
        q, key_cache, value_cache, partials, lses, rel, mode=MODE_PARTIAL, **kwargs
    )
    merge_partials(partials, lses, out, gate, sinks, num_actual_tokens)
