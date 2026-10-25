"""Sparse DCA prefill (gfx1100): vertical-slash sparse per region, LSE-merged.

Drop-in alternative to dca_prefill.dca_prefill_forward, gated by VLLM_DCA_SPARSE=1
at the engine. Mirrors Qwen's sparse recipe but with per-region (not global)
vertical-slash selection, which maps directly onto the two validated kernels:

  intra : causal block+column sparse   (vs_dca.causal_vs_attention_lse)     q variant 0
  succ  : non-causal shared-column     (vs_dca.noncausal_col_attention_lse) q variant 1, select with crit 3
  inter : non-causal shared-column     (vs_dca.noncausal_col_attention_lse) q variant 2, select with crit 4

Selection follows Qwen: intra picks vertical+slash from q_intra's last-q scores;
succ/inter pick vertical columns from the *critical* query's last-q scores (sinks
forced in). Each pass emits O + natural-log LSE; passes combine with the same
merge_attn_states the dense driver uses. Regions smaller than `sparse_min` fall
back to the validated dense paged pass (no accuracy or speed reason to sparsify).

paged KV gather: regions are block-aligned (chunk_len % block_size == 0), so a
region's contiguous K/V is k_cache[bt_slice].view(-1,Hkv,D)[:n]. The gather is
O(n) bandwidth; the win is the O(n^2)->O(n*budget) attention on big chunks.
"""
import os
import sys
import torch

from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

# vs_sparse / vs_dca live in the standalone sparse-port (kernels + C++ convert ext).
# Vendoring copies this module into the vllm attention package; keep the kernels in
# one place and resolve them by path so the build cache (csrc/) stays single-sourced.
_SPARSE_PORT = os.environ.get("DCA_SPARSE_PORT_DIR", "/home/hz6bst/mimir-train/sparse-port")
if _SPARSE_PORT not in sys.path:
    sys.path.insert(0, _SPARSE_PORT)
import vs_sparse as _vs
import vs_dca as _vd

_V_SIZE = int(os.environ.get("VLLM_DCA_SPARSE_V", "2048"))
_S_SIZE = int(os.environ.get("VLLM_DCA_SPARSE_S", "1024"))
_MIN = int(os.environ.get("VLLM_DCA_SPARSE_MIN", "4096"))
_LASTQ = int(os.environ.get("VLLM_DCA_SPARSE_LASTQ", "64"))
_SINK = int(os.environ.get("VLLM_DCA_SPARSE_SINK", "64"))
_SYNC = os.environ.get("VLLM_DCA_SPARSE_SYNC", "") not in ("", "0")


def _ck(tag):
    if _SYNC:
        torch.cuda.synchronize()
        print(f"[dca-sparse] {tag}", flush=True)


def _gather(cache, bt_row, start, n, block_size):
    """Contiguous [n, Hkv, D] from paged cache for keys [start, start+n)."""
    b0 = start // block_size
    b1 = (start + n - 1) // block_size + 1
    blk = bt_row[b0:b1].long()
    flat = cache[blk].reshape(-1, cache.shape[-2], cache.shape[-1])
    off = start - b0 * block_size
    return flat[off:off + n].contiguous()


def _merge_fmt(o, lse):
    # kernel out o:[1,Hq,N,D] lse:[1,Hq,N] -> merge fmt out:[N,Hq,D] lse:[Hq,N]
    return o[0].transpose(0, 1).contiguous(), lse[0].contiguous()


def _merge(o0, l0, o1, l1):
    mo = torch.empty_like(o0)
    ml = torch.empty_like(l0)
    merge_attn_states(mo, o0, l0, o1, l1, output_lse=ml)
    return mo, ml


_DENSE_BS = 256  # paged block size for the dense fallback (NOT one giant block)


def _dense_pass(q, k_reg, v_reg, scale, causal):
    """Dense fallback for small regions. q:[Qc,Hq,D] k_reg/v_reg:[n,Hkv,D].
    Returns out:[Qc,Hq,D] lse:[Hq,Qc] (natural log).

    The contiguous region is re-paged into real _DENSE_BS-sized blocks (padded,
    with seqused_k masking the pad). A single giant block (block_size = n) faults
    the ROCm unified_attention kernel for large n (GPU page fault at n>~8k), so
    keep blocks small like the live paged cache.
    """
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    Qc, Hq, D = q.shape
    n, Hkv, _ = k_reg.shape
    dev = q.device
    nb = (n + _DENSE_BS - 1) // _DENSE_BS
    pad = nb * _DENSE_BS - n
    if pad:
        z = k_reg.new_zeros(pad, Hkv, D)
        k_reg = torch.cat([k_reg, z], 0)
        v_reg = torch.cat([v_reg, v_reg.new_zeros(pad, Hkv, D)], 0)
    kc = k_reg.view(nb, _DENSE_BS, Hkv, D)
    vc = v_reg.view(nb, _DENSE_BS, Hkv, D)
    out = torch.empty(Qc, Hq, D, device=dev, dtype=q.dtype)
    lse = torch.empty(Hq, Qc, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, Qc], device=dev, dtype=torch.int32)
    seqused = torch.tensor([n], device=dev, dtype=torch.int32)   # masks the pad
    bt = torch.arange(nb, device=dev, dtype=torch.int32).unsqueeze(0)
    unified_attention(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=Qc,
        seqused_k=seqused, max_seqlen_k=n, softmax_scale=scale, causal=causal,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None, output_lse=lse,
    )
    return out, lse


def _intra_sparse(q_intra_r, k_reg, v_reg, scale):
    # q_intra_r:[Qc,Hq,D]; region is square causal (Qc == n_keys)
    Qc, Hq, D = q_intra_r.shape
    n = k_reg.shape[0]
    q = q_intra_r.transpose(0, 1).unsqueeze(0).contiguous()   # [1,Hq,Qc,D]
    k = k_reg.transpose(0, 1).unsqueeze(0).contiguous()       # [1,Hkv,n,D]
    v = v_reg.transpose(0, 1).unsqueeze(0).contiguous()
    v_idx, s_idx = _vs.calc_index_local(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        _V_SIZE, _S_SIZE, last_q_size=min(_LASTQ, Qc), sink_tokens=_SINK,
    )
    v_idx = v_idx.to(torch.int32).sort(-1).values
    s_idx = s_idx.to(torch.int32).sort(-1, descending=True).values
    seqlens = torch.tensor([n], dtype=torch.int32, device=q.device)
    bc, bo, cc, ci = _vs.convert_vertical_slash_indexes(seqlens, v_idx, s_idx, n)
    o, lse = _vd.causal_vs_attention_lse(q, k, v, seqlens, bc, bo, cc, ci, scale)
    return _merge_fmt(o, lse)


def _noncausal_sparse(q_r, q_crit_r, k_reg, v_reg, scale, vbudget=None):
    # q_r/q_crit_r:[Qc,Hq,D]; k_reg/v_reg:[n,Hkv,D]; n != Qc allowed (incl. Qc>n).
    # vbudget overrides the vertical budget; vbudget>=n selects every column (exact),
    # the dense-equivalent path for sub-MIN non-causal regions (unified_attention
    # can't serve those -- it NaNs when Qc>n due to its right-aligned-query model).
    Qc, Hq, D = q_r.shape
    n = k_reg.shape[0]
    vb = _V_SIZE if vbudget is None else vbudget
    q = q_r.transpose(0, 1).unsqueeze(0).contiguous()          # [1,Hq,Qc,D]
    qc = q_crit_r.unsqueeze(0).contiguous()                     # [1,Qc,Hq,D] for estimator
    k = k_reg.transpose(0, 1).unsqueeze(0).contiguous()        # [1,Hkv,n,D]
    v = v_reg.transpose(0, 1).unsqueeze(0).contiguous()
    col_index, col_count = _vd.calc_vertical_noncausal(
        qc, k_reg.unsqueeze(0), vb, n, sink_tokens=_SINK,
        last_q_size=min(_LASTQ, Qc),
    )
    klen = torch.tensor([n], dtype=torch.int32, device=q.device)
    o, lse = _vd.noncausal_col_attention_lse(q, k, v, klen, col_count, col_index, scale)
    return _merge_fmt(o, lse)


def dca_prefill_forward_sparse(
    q5, k_cache, v_cache, k_length, block_table_row, base_scale,
    chunk_size, local_size, original_max_position_embeddings, block_size,
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

    out = torch.empty(Qtot, Hq, D, device=dev, dtype=q5.dtype)

    # Slice q variants per chunk (not five full-length copies): at the trained
    # chunk_len the whole-prefill query is multi-GB/rank, so materialize only the
    # current chunk's variant and free it before the next region.
    q_offset = k_length - Qtot
    begin = q_offset
    while begin < k_length:
        prev = (begin // chunk_len) * chunk_len
        end = min(prev + chunk_len, k_length)
        qb = begin - q_offset
        qe = end - q_offset

        # Intra, offset-aware (chunked prefill): a query slice may start mid-chunk.
        # Query at abs p attends own-chunk keys [prev, p] = [prev, begin) (seen by
        # ALL slice queries -> non-causal) + [begin, p] (square causal with the
        # slice). Split so both reuse the validated square/non-causal kernels.
        _ck(f"iter begin={begin} prev={prev} end={end} n_pre={begin - prev} n_sq={end - begin}")
        qi = q5[qb:qe, 0].contiguous()
        n_pre = begin - prev                 # non-causal own-chunk prefix
        n_sq = end - begin                   # square-causal own-chunk block
        ksq = _gather(k_cache, block_table_row, begin, n_sq, block_size)
        vsq = _gather(v_cache, block_table_row, begin, n_sq, block_size)
        _ck(f"  gathered sq n_sq={n_sq}")
        if n_sq >= _MIN:
            cur_o, cur_l = _intra_sparse(qi, ksq, vsq, scale)
        else:
            cur_o, cur_l = _dense_pass(qi, ksq, vsq, scale, causal=True)
        _ck("  intra/square done")
        del ksq, vsq
        if n_pre > 0:
            kpre = _gather(k_cache, block_table_row, prev, n_pre, block_size)
            vpre = _gather(v_cache, block_table_row, prev, n_pre, block_size)
            _ck(f"  gathered pre n_pre={n_pre}")
            # non-causal own-chunk prefix has Qc(square)>=n_pre and often Qc>>n_pre;
            # unified_attention NaNs when Qc>n, so always use the column kernel
            # (full budget == exact when sub-MIN).
            o_p, l_p = _noncausal_sparse(
                qi, qi, kpre, vpre, scale,
                vbudget=(None if n_pre >= _MIN else n_pre),
            )
            _ck("  prefix done")
            cur_o, cur_l = _merge(cur_o, cur_l, o_p, l_p)
            _ck("  prefix merged")
            del kpre, vpre, o_p, l_p
        del qi

        if prev - chunk_len >= 0:
            s = prev - chunk_len
            ks = _gather(k_cache, block_table_row, s, chunk_len, block_size)
            vs = _gather(v_cache, block_table_row, s, chunk_len, block_size)
            _ck(f"  gathered succ s={s} len={chunk_len}")
            qs = q5[qb:qe, 1].contiguous()
            if chunk_len >= _MIN:
                qsc = q5[qb:qe, 3].contiguous()
                o_s, l_s = _noncausal_sparse(qs, qsc, ks, vs, scale)
                del qsc
            else:
                o_s, l_s = _noncausal_sparse(qs, qs, ks, vs, scale, vbudget=chunk_len)
            _ck("  succ done")
            cur_o, cur_l = _merge(cur_o, cur_l, o_s, l_s)
            del qs, ks, vs, o_s, l_s

        if prev - 2 * chunk_len >= 0:
            e2 = prev - chunk_len
            ki2 = _gather(k_cache, block_table_row, 0, e2, block_size)
            vi2 = _gather(v_cache, block_table_row, 0, e2, block_size)
            qii = q5[qb:qe, 2].contiguous()
            if e2 >= _MIN:
                qic = q5[qb:qe, 4].contiguous()
                o_i, l_i = _noncausal_sparse(qii, qic, ki2, vi2, scale)
                del qic
            else:
                o_i, l_i = _noncausal_sparse(qii, qii, ki2, vi2, scale, vbudget=e2)
            cur_o, cur_l = _merge(cur_o, cur_l, o_i, l_i)
            del qii, ki2, vi2, o_i, l_i

        out[qb:qe] = cur_o
        begin = end
    return out
