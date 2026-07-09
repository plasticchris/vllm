"""DCA engine hook for the v1 Attention layer (ROCm).

Called from Attention.forward when the layer was built with a
dual_chunk_attention_config (qwen2 then produces a 5-wide query via
DualChunkRotaryEmbedding). The query arrives H*5*D wide, variant-major:
the five flattened (T, H*D) sub-queries concatenated along the last dim in the
order [intra, succ, inter, succ_critical, inter_critical] (see dual_chunk_rope
forward_native), i.e. reshape(-1, 5, H, D).

Paths:
  * profiling (no metadata) or single chunk (max KV length <= chunk_len): take the
    intra sub-query (variant 0; standard rope for positions < chunk_len, YaRN factor
    clips to 1.0) and run the *exact* stock dispatch, so the output is token-identical
    to a non-DCA layer.
  * multi-chunk decode (max KV length > chunk_len, one query token per request):
    write the current KV, then run the 3-pass intra/succ/inter DCA over the paged
    cache (dca_attn.dca_decode_forward, validated offline at err ~3e-7).
  * multi-chunk prefill (max_query_len > 1): write the current KV, then per sequence
    run the chunked DCA prefill (dca_prefill.dca_prefill_forward) -- intra causal +
    succ/inter non-causal (the gated CAUSAL=False kernel mode), LSE-merged. Validated
    offline (Stage 3a kernel non-causal, Stage 3b full prefill).

Live kv_cache layout (ROCm backend): the (2, num_blocks, block_size,
num_kv_heads, head_size) tensor is written in the classic x-split layout
(K: [nb,Hkv,D//x,bs,x], V: [nb,Hkv,D,bs] via PagedAttention.split_kv_cache),
NOT flash. _to_flash_cache converts it to the [nb,bs,Hkv,D] flash layout the
unified_attention kernel reads. (unbind(0) would read transposed garbage.)

Only runs when self.dca is set; the non-DCA path in Attention.forward is untouched.
"""
import os
import sys

import torch

_DBG = os.environ.get("VLLM_DCA_DEBUG", "") not in ("", "0")


def _dbg(msg):
    if _DBG:
        print(f"[dca] {msg}", file=sys.stderr, flush=True)


def _to_flash_cache(kv_cache, num_kv_heads, head_size):
    """Convert the ROCm classic x-split paged cache to flash [nb,bs,Hkv,D].

    split_kv_cache returns K as [nb,Hkv,D//x,bs,x] and V as [nb,Hkv,D,bs];
    unified_attention expects [nb,bs,Hkv,D]. The permute+reshape materializes a
    contiguous flash copy (correct data); strided views can't express the x-split.
    """
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    ks, vs = PagedAttention.split_kv_cache(kv_cache, num_kv_heads, head_size)
    nb = ks.shape[0]
    bs = ks.shape[3]
    k_flash = ks.permute(0, 3, 1, 2, 4).reshape(nb, bs, num_kv_heads, head_size)
    v_flash = vs.permute(0, 3, 1, 2).reshape(nb, bs, num_kv_heads, head_size)
    return k_flash.contiguous(), v_flash.contiguous()


class DCAEngine:
    def __init__(self, dca_cfg):
        self.chunk_size = int(dca_cfg["chunk_size"])
        self.local_size = int(dca_cfg["local_size"])
        self.orig_max = int(dca_cfg.get("original_max_position_embeddings", 0))
        self.chunk_len = self.chunk_size - self.local_size
        # VLLM_DCA_SPARSE=1 routes the multi-chunk PREFILL through the vertical-slash
        # sparse path (per-region select + sparse kernels + LSE merge); decode and the
        # single-chunk/dense paths are unaffected.
        self.sparse_prefill = os.environ.get("VLLM_DCA_SPARSE", "") not in ("", "0")

    def forward(self, layer, query, key, value, output_shape):
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )

        H, D, Dv = layer.num_heads, layer.head_size, layer.head_size_v
        q5 = query.reshape(-1, 5, H, D)          # [T, 5, H, D], variant-major
        T = q5.shape[0]

        attn_metadata, _layer, kv_cache, _slot = get_attention_context(layer.layer_name)

        # profiling/dummy run or single chunk -> stock dispatch with the intra query.
        if attn_metadata is None or int(attn_metadata.max_seq_len) <= self.chunk_len:
            _dbg(f"single-chunk T={T} "
                 f"max_seq={None if attn_metadata is None else int(attn_metadata.max_seq_len)} "
                 f"chunk_len={self.chunk_len}")
            q_intra = q5[:, 0, :, :].reshape(T, H * D).contiguous()
            return self._dispatch(layer, q_intra, key, value, output_shape)

        # multi-chunk: append the current step's KV, then DCA over the paged cache.
        self._kv_write(layer, key, value)
        # The ROCm backend stores the paged cache in the classic x-split layout
        # (K: [nb,Hkv,D//x,bs,x], V: [nb,Hkv,D,bs]) -- NOT the flash layout the
        # unified_attention kernel reads. Convert to flash [nb,bs,Hkv,D] here.
        k_cache, v_cache = _to_flash_cache(kv_cache, layer.num_kv_heads, D)
        scale = layer.impl.scale
        block_size = k_cache.shape[1]

        if int(attn_metadata.max_query_len) == 1:
            _dbg(f"decode T={T} max_seq={int(attn_metadata.max_seq_len)} "
                 f"chunk_len={self.chunk_len}")
            from vllm.model_executor.layers.attention.dca_attn import (
                dca_decode_forward,
            )
            o = dca_decode_forward(
                q5, k_cache, v_cache, attn_metadata.seq_lens,
                attn_metadata.block_table, scale, self.chunk_size, self.local_size,
                self.orig_max, block_size,
            )  # [T, H, D]
        else:
            o = self._prefill_batch(
                q5, k_cache, v_cache, attn_metadata, scale, block_size, H, D,
            )

        if output_shape is None:
            output_shape = torch.Size((T, H * Dv))
        output = torch.empty(output_shape, dtype=query.dtype, device=query.device)
        output.view(-1, H, Dv).copy_(o)
        return output.view(-1, output_shape[-1])

    def _prefill_batch(self, q5, k_cache, v_cache, md, scale, block_size, H, D):
        """Per-sequence chunked DCA over a (possibly mixed) batch.

        Sequences with a single query token take the decode path; the rest take
        the chunked prefill path. The target serve runs max_num_seqs=1, so in
        practice this is one prefill sequence.
        """
        from vllm.model_executor.layers.attention.dca_attn import dca_decode_forward
        from vllm.model_executor.layers.attention.dca_prefill import (
            dca_prefill_forward,
        )

        T = q5.shape[0]
        o = torch.empty(T, H, D, device=q5.device, dtype=q5.dtype)
        qsl = md.query_start_loc.tolist()
        seq_lens = md.seq_lens
        bt = md.block_table
        n_seqs = seq_lens.shape[0]
        for i in range(n_seqs):
            s, e = int(qsl[i]), int(qsl[i + 1])
            if e <= s:
                continue
            klen = int(seq_lens[i].item())
            if e - s == 1:
                _dbg(f"prefill-batch seq{i} decode-row klen={klen}")
                o[s:e] = dca_decode_forward(
                    q5[s:e], k_cache, v_cache, seq_lens[i:i + 1], bt[i:i + 1],
                    scale, self.chunk_size, self.local_size, self.orig_max,
                    block_size,
                )
            else:
                n_chunks = (klen + self.chunk_len - 1) // self.chunk_len
                _dbg(f"prefill seq{i} qlen={e - s} klen={klen} "
                     f"chunk_len={self.chunk_len} n_chunks={n_chunks} "
                     f"sparse={self.sparse_prefill}")
                if self.sparse_prefill:
                    from vllm.model_executor.layers.attention.dca_prefill_sparse import (  # noqa: E501
                        dca_prefill_forward_sparse,
                    )
                    o[s:e] = dca_prefill_forward_sparse(
                        q5[s:e], k_cache, v_cache, klen, bt[i], scale,
                        self.chunk_size, self.local_size, self.orig_max, block_size,
                    )
                else:
                    o[s:e] = dca_prefill_forward(
                        q5[s:e], k_cache, v_cache, klen, bt[i], scale,
                        self.chunk_size, self.local_size, self.orig_max, block_size,
                    )
        return o

    def _kv_write(self, layer, key, value):
        """Append the current step's K/V into the paged cache (stock op)."""
        if (
            layer.attn_backend.forward_includes_kv_cache_update
            or layer.kv_sharing_target_layer_name is not None
            or key is None
            or value is None
        ):
            return
        from vllm.model_executor.layers.attention.attention import (
            unified_kv_cache_update,
            _encode_layer_name,
        )

        key = key.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.view(-1, layer.num_kv_heads, layer.head_size_v)
        if layer.use_direct_call:
            unified_kv_cache_update(key, value, layer.layer_name)
        else:
            torch.ops.vllm.unified_kv_cache_update(
                key, value, _encode_layer_name(layer.layer_name)
            )

    def _dispatch(self, layer, query, key, value, output_shape):
        """Verbatim tail of the stock Attention.forward, with the intra query."""
        from vllm.model_executor.layers.attention.attention import (
            unified_kv_cache_update,
            unified_attention_with_output,
            _encode_layer_name,
        )

        output_dtype = query.dtype
        if output_shape is None:
            output_shape = torch.Size(
                (query.shape[0], layer.num_heads * layer.head_size_v)
            )
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]
        query = query.view(-1, layer.num_heads, layer.head_size)
        output = output.view(-1, layer.num_heads, layer.head_size_v)
        if key is not None:
            key = key.view(-1, layer.num_kv_heads, layer.head_size)
        if value is not None:
            value = value.view(-1, layer.num_kv_heads, layer.head_size_v)

        need_kv = (
            not layer.attn_backend.forward_includes_kv_cache_update
            and layer.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        )
        kv_dep = None
        if layer.use_direct_call:
            if need_kv:
                kv_dep = unified_kv_cache_update(key, value, layer.layer_name)
            unified_attention_with_output(
                query, key, value, output, layer.layer_name, kv_cache_dummy_dep=kv_dep
            )
        else:
            enc = _encode_layer_name(layer.layer_name)
            if need_kv:
                kv_dep = torch.ops.vllm.unified_kv_cache_update(key, value, enc)
            torch.ops.vllm.unified_attention_with_output(
                query, key, value, output, enc, kv_cache_dummy_dep=kv_dep
            )
        return output.view(-1, hidden_size)
