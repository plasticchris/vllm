# Local patch: flash-decoding (split-KV) path for small-query / large-context
# attention on RDNA (gfx1100). The stock ROCm decode + prefix-prefill kernels do
# not split over the KV sequence, so decode/MTP-spec steps (query_len 1..1+num_spec)
# over long context launch few workgroups that serially scan the whole KV -> the
# long-context decode cliff. This routes those steps to the in-tree SGLang-style
# flash-decoding kernel (decode_attention_fwd), which is correct for head_dim 256,
# GQA, paged blocks and fp8 KV (validated in /mnt/scratch/vllm/kernel-work).
#
# Env:
#   ROCM_FLASH_DECODE=0        disable (fall back to stock path)
#   ROCM_FLASH_DECODE_CHECK=1  run BOTH paths, assert match, log max rel err
#   ROCM_FLASH_DECODE_MAXQ=8   max query_len to treat as decode/spec (else prefill)
import os
import torch
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

_LOGGED = {"ok": False, "off": False}


def _num_kv_splits(max_seq_len: int) -> int:
    # enough splits to fill ~96 CUs for single/low batch; capped to limit reduce cost
    s = (int(max_seq_len) + 1023) // 1024
    if s < 8:
        s = 8
    if s > 64:
        s = 64
    return s


def try_flash_decode(
    query, output, kv_cache, block_table, query_start_loc, seq_lens,
    max_query_len, max_seq_len, num_heads, num_kv_heads, head_size,
    sm_scale, k_scale, v_scale, kv_cache_dtype, fp8_dtype,
    sliding_window, alibi_slopes, sinks, causal,
) -> bool:
    if os.environ.get("ROCM_FLASH_DECODE", "1") == "0":
        return False
    if alibi_slopes is not None or sinks is not None or not causal:
        return False
    if sliding_window not in (0, None, -1):
        return False
    maxq = int(os.environ.get("ROCM_FLASH_DECODE_MAXQ", "8"))
    Q = int(max_query_len)
    num_seqs = seq_lens.shape[0]
    num_tokens = query.shape[0]
    _reason = None
    if Q < 1 or Q > maxq:
        _reason = f"Q={Q}>{maxq}"
    elif num_tokens != num_seqs * Q:
        _reason = f"nonuniform tok={num_tokens} seqs={num_seqs} Q={Q}"
    if _reason is not None:
        if not _LOGGED.get("rej"):
            import logging; logging.getLogger(__name__).warning("[rocm_flash_decode] skip: %s", _reason)
            _LOGGED["rej"]=True
        return False
    try:
        kc = kv_cache[0]
        vc = kv_cache[1]
        if "fp8" in kv_cache_dtype:
            kc = kc.view(fp8_dtype)
            vc = vc.view(fp8_dtype)
        page_size = kc.shape[1]            # [num_blocks, block_size, num_kv_heads, head_size]
        NKV = _num_kv_splits(max_seq_len)
        qv = query.view(num_seqs, Q, num_heads, head_size)
        ov = output.view(num_seqs, Q, num_heads * head_size)
        dev = query.device
        for j in range(Q):
            qj = qv[:, j, :, :].contiguous()
            blen = (seq_lens - (Q - 1) + j).to(torch.int32)
            oj = torch.empty(num_seqs, num_heads, head_size, device=dev, dtype=query.dtype)
            lse = torch.empty(num_seqs, num_heads, device=dev, dtype=torch.float32)
            al = torch.empty(num_seqs, num_heads, NKV, head_size + 1, device=dev, dtype=torch.float32)
            decode_attention_fwd(qj, kc, vc, oj, lse, block_table, blen, al, NKV,
                                 sm_scale, page_size=page_size, k_scale=k_scale, v_scale=v_scale)
            ov[:, j, :] = oj.reshape(num_seqs, num_heads * head_size)
        if not _LOGGED["ok"]:
            import logging
            logging.getLogger(__name__).warning(
                "[rocm_flash_decode] active: Q=%d NKV=%d page=%d maxseq=%d", Q, NKV, page_size, int(max_seq_len))
            _LOGGED["ok"] = True
        return True
    except Exception as e:  # any failure -> safe fallback to stock path
        if not _LOGGED["off"]:
            import logging
            logging.getLogger(__name__).warning("[rocm_flash_decode] disabled after error: %r", e)
            _LOGGED["off"] = True
        return False
