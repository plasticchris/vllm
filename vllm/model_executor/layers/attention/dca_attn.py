"""DCA attention forward over v1 paged KV (DECODE path) on ROCm.

Runs the three DCA passes (intra/succ/inter) against the paged KV cache using the
patched unified_attention (which emits per-(head,token) LSE) and combines them with
merge_attn_states. Decode only; chunked-prefill is the next stage.

q5 in: [T, 5, Hq, D] -- the DualChunkRotaryEmbedding output split into
(intra, succ, inter, succ_critical, inter_critical). The dense path uses the first
three; the *_critical variants are sparse-only.

Range starts are c*chunk_len and (c-1)*chunk_len; both are multiples of block_size
for the real config (chunk_len=253952, block_size=16 -> 253952%16==0), so the per-pass
block-table slices are exact (a pass reads its keys from a block boundary).

NOTE: per-request YaRN scale is folded into the query rows (equivalent to scaling the
softmax scale per request), so batched decode with differing seq_lens is handled. The
mixed-batch empty-pass case (a request whose succ/inter range is empty while others in
the batch are not) sets that request's LSE to -inf so merge ignores it; the target serve
runs max_num_seqs=1, so batch is 1 in practice.
"""
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

from vllm.model_executor.layers.attention.dca_metadata import build_dca_decode_metadata


def _mask_empty(lse, seq_lens):
    empty = seq_lens == 0
    if bool(empty.any()):
        lse[:, empty] = float("-inf")
    return lse


def _pass(q, k_cache, v_cache, cu, seqused_k, max_k, scale, block_table, causal):
    T, Hq, D = q.shape
    out = torch.empty(T, Hq, D, device=q.device, dtype=q.dtype)
    lse = torch.empty(Hq, T, device=q.device, dtype=torch.float32)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=max_k,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_lse=lse,
    )
    return out, lse


def _merge(cur_o, cur_l, o, l, seq_lens):
    l = _mask_empty(l, seq_lens)
    mo = torch.empty_like(cur_o)
    ml = torch.empty_like(cur_l)
    merge_attn_states(mo, cur_o, cur_l, o, l, output_lse=ml)
    return mo, ml


def dca_decode_forward(
    q5,                                  # [T,5,Hq,D] one decode token per request
    k_cache,                             # [num_blocks, block_size, Hkv, D]
    v_cache,
    cache_seq_lens,                      # [T] int total KV length per request
    block_table,                         # [T, max_blocks] int paged table
    base_scale,
    chunk_size,
    local_size,
    original_max_position_embeddings,
    block_size,
):
    T, _, Hq, D = q5.shape
    dev = q5.device
    meta = build_dca_decode_metadata(
        cache_seq_lens, block_table, chunk_size, local_size,
        original_max_position_embeddings, block_size,
    )
    sf = meta.scaling_factor.to(dev).view(T, 1, 1)
    q_intra = (q5[:, 0] * sf).contiguous()
    q_succ = (q5[:, 1] * sf).contiguous()
    q_inter = (q5[:, 2] * sf).contiguous()
    cu = torch.arange(T + 1, device=dev, dtype=torch.int32)

    cur_o, cur_l = _pass(
        q_intra, k_cache, v_cache, cu, meta.intra.seq_lens, meta.intra.max_seq_len,
        base_scale, meta.intra.block_table, causal=True,
    )
    # Decode is a single trailing query, so causal=True attends every key in each
    # pass's range -- identical to non-causal here, and the kernel is causal-only.
    # (Non-causal succ/inter only matters for multi-query prefill, a later stage.)
    if meta.succ.max_seq_len > 0:
        o_s, l_s = _pass(
            q_succ, k_cache, v_cache, cu, meta.succ.seq_lens, meta.succ.max_seq_len,
            base_scale, meta.succ.block_table, causal=True,
        )
        cur_o, cur_l = _merge(cur_o, cur_l, o_s, l_s, meta.succ.seq_lens)
    if meta.inter.max_seq_len > 0:
        o_n, l_n = _pass(
            q_inter, k_cache, v_cache, cu, meta.inter.seq_lens, meta.inter.max_seq_len,
            base_scale, meta.inter.block_table, causal=True,
        )
        cur_o, cur_l = _merge(cur_o, cur_l, o_n, l_n, meta.inter.seq_lens)
    return cur_o
