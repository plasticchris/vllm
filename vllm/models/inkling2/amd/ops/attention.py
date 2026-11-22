"""DSpark block attention on the portable Triton kernel."""

from .attention_dispatch import run_attention


def plan_decode_attention(*args, **kwargs):
    return 1


def block_decode_attention(
    q,
    key_cache,
    value_cache,
    out,
    *,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    block,
    window,
    num_splits,
    max_seq_len=0,
):
    run_attention(
        q,
        key_cache,
        value_cache,
        out,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=block,
        softmax_scale=q.shape[-1] ** -0.5,
        window=window,
        max_seq_len=max_seq_len,
    )
