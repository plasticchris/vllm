"""DCA attention forward over v1 paged KV (PREFILL path) on ROCm.

Mirrors the reference _dual_chunk_flash_attn_prefill: walk the query region one
chunk at a time and, per chunk, run

    intra : keys [prev_chunk_end, end)         causal      (own chunk, q variant 0)
    succ  : keys [prev_chunk_end-CL, prev)     non-causal  (prev chunk,  q variant 1)
    inter : keys [0, prev_chunk_end-CL)        non-causal  (older chunks, q variant 2)

combining the three with merge_attn_states (LSE). The succ/inter passes are
multi-query and non-causal, which needs the gated CAUSAL=False kernel mode
(apply_noncausal_patch.py). chunk_len % block_size == 0 keeps every range
block-aligned, so the per-pass block-table slices are exact.

The per-request YaRN scalar (function of the total sequence length) folds into
the softmax scale, matching the reference.

q5 in: [Qtot, 5, Hq, D] -- DualChunkRotaryEmbedding output (intra, succ, inter,
succ_critical, inter_critical); the dense path uses the first three.
"""
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states


def _mq_pass(q, k_cache, v_cache, seqused_val, scale, bt_row, causal):
    """One multi-query attention pass for a single sequence over a key range.

    q: [Qc, Hq, D]; bt_row: 1-D block ids whose block 0 maps to the range start.
    """
    Qc, Hq, D = q.shape
    dev = q.device
    out = torch.empty(Qc, Hq, D, device=dev, dtype=q.dtype)
    lse = torch.empty(Hq, Qc, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, Qc], device=dev, dtype=torch.int32)
    seqused = torch.tensor([seqused_val], device=dev, dtype=torch.int32)
    unified_attention(
        q=q, k=k_cache, v=v_cache, out=out,
        cu_seqlens_q=cu, max_seqlen_q=Qc,
        seqused_k=seqused, max_seqlen_k=seqused_val,
        softmax_scale=scale, causal=causal, window_size=(-1, -1),
        block_table=bt_row.unsqueeze(0), softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None, output_lse=lse,
    )
    return out, lse


def _merge(o0, l0, o1, l1):
    mo = torch.empty_like(o0)
    ml = torch.empty_like(l0)
    merge_attn_states(mo, o0, l0, o1, l1, output_lse=ml)
    return mo, ml


def dca_prefill_forward(
    q5,                                  # [Qtot,5,Hq,D] prefill queries
    k_cache,                             # [num_blocks, block_size, Hkv, D]
    v_cache,
    k_length,                            # int total KV length (== seq end pos)
    block_table_row,                     # [n_blocks] int paged table for this seq
    base_scale,
    chunk_size,
    local_size,
    original_max_position_embeddings,
    block_size,
):
    Qtot, _, Hq, D = q5.shape
    dev = q5.device
    chunk_len = chunk_size - local_size
    if original_max_position_embeddings > 0:
        ratio = float(k_length) / float(original_max_position_embeddings)
        sf = max(0.1 * torch.log(torch.tensor(ratio)).item() + 1.0, 1.0)
    else:
        sf = 1.0
    scale = base_scale * sf

    q_intra = q5[:, 0].contiguous()
    q_succ = q5[:, 1].contiguous()
    q_inter = q5[:, 2].contiguous()
    out = torch.empty(Qtot, Hq, D, device=dev, dtype=q5.dtype)

    q_offset = k_length - Qtot           # abs key pos of the first query row
    begin = q_offset
    while begin < k_length:
        prev = (begin // chunk_len) * chunk_len      # own-chunk start (aligned)
        end = min(prev + chunk_len, k_length)
        qb = begin - q_offset
        qe = end - q_offset

        # intra: causal over the own chunk [prev, end)
        bt_intra = block_table_row[prev // block_size: (end - 1) // block_size + 1]
        cur_o, cur_l = _mq_pass(
            q_intra[qb:qe], k_cache, v_cache, end - prev, scale, bt_intra, causal=True
        )
        # succ: non-causal over the immediately-preceding chunk
        if prev - chunk_len >= 0:
            s = prev - chunk_len
            bt_succ = block_table_row[s // block_size: prev // block_size]
            o_s, l_s = _mq_pass(
                q_succ[qb:qe], k_cache, v_cache, chunk_len, scale, bt_succ, causal=False
            )
            cur_o, cur_l = _merge(cur_o, cur_l, o_s, l_s)
        # inter: non-causal over everything before the preceding chunk
        if prev - 2 * chunk_len >= 0:
            e2 = prev - chunk_len
            bt_inter = block_table_row[0: (e2 - 1) // block_size + 1]
            o_i, l_i = _mq_pass(
                q_inter[qb:qe], k_cache, v_cache, e2, scale, bt_inter, causal=False
            )
            cur_o, cur_l = _merge(cur_o, cur_l, o_i, l_i)

        out[qb:qe] = cur_o
        begin = end
    return out
