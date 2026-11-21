"""Measured gfx1100 launch policy; baseline remains available for rollback.

Keep this policy CPU-only: no .item()/device synchronizations in a decode.
The EP row-alignment block must be identical for both expert projections.
"""

import os

OPTIMIZED = os.environ.get("INKLING_OPTIMIZED", "0") == "1"


def expert_launch_config(num_tokens, k, nvfp4):
    if nvfp4:
        return (64 if num_tokens >= 512 else 16, 64, 128), True
    return (
        32 if num_tokens >= 512 else 16,
        64,
        128 if num_tokens < 512 and k > 1152 else 64,
    ), False


def attention_launch_config(
    num_tokens,
    num_reqs,
    max_query_len,
    max_seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    window,
):
    # The narrower K tile won the bounded long-prefill sweep; increasing M
    # did not. Keep short/mixed prefills on the baseline policy.
    if max_query_len >= 1024:
        return 1, (64, 16, 4)
    if window or max_query_len > 16 or max_seq_len < 4096:
        return 1, None
    rows = max_query_len * (num_heads // num_kv_heads)
    config = (16 if rows <= 16 else 32 if rows <= 32 else 64, 32, 4)
    tokens_per_tile = config[0] // (num_heads // num_kv_heads)
    ctas = max(
        1,
        ((max_query_len + tokens_per_tile - 1) // tokens_per_tile)
        * num_kv_heads
        * num_reqs,
    )
    goal = 128 if max_query_len == 1 and max_seq_len <= 262144 else 256
    desired = max(1, (goal + ctas - 1) // ctas)
    splits = min(128, 1 << (desired - 1).bit_length())
    per_split = max(1, num_tokens * num_heads * (head_dim + 1) * 4)
    cap = min(max_seq_len // 1024, (32 * 2**20) // per_split)
    while splits > max(1, cap):
        splits //= 2
    return splits, config if splits > 1 else None
