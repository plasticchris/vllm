# Local patch: split-KV (flash-decoding) attention that reads vLLM's *native*
# ROCm paged KV-cache layout directly (zero-copy), for small-query decode/MTP-spec
# steps over long context on RDNA (gfx1100, head_size 256).
#
# Why this file exists:
#   The stock ROCm decode path for head_size 256 has no KV-splitting kernel
#   (the C++ paged_attention_rocm kernel is gated to head_size 128, and MTP's
#   query_len=3 step is classified as "prefill" and sent to context_attention_fwd,
#   which launches one workgroup per (seq,kv_head) and serially scans the whole KV
#   -> the long-context decode cliff). The in-tree SGLang flash-decoding kernel
#   (triton_decode_attention) *does* split over KV, but its stage-1 assumes a flat
#   [total_tokens, num_kv_heads, head_size] buffer with contiguous head_dim. vLLM's
#   ROCm cache is instead:
#       key_cache:   [num_blocks, num_kv_heads, head_size // x, block_size, x]
#       value_cache: [num_blocks, num_kv_heads, head_size,      block_size]
#   so head_dim is x-split for K and block_size is interleaved for both. Copying the
#   whole cache into the flat layout every step is far too expensive, so instead this
#   stage-1 kernel is the SGLang grouped kernel with the K/V addressing rewritten to
#   the native strides (same offset math kernel_paged_attention_2d already uses). The
#   split reduction (stage-2) is reused unchanged.
#
# MTP (query_len Q>1) is handled by looping the decode kernel once per query position
# j, with the visible KV length set to (seq_len - (Q-1) + j) so intra-step causal
# masking is exact. This assumes the Q new tokens' K/V are already written into the
# cache before attention (vLLM writes them via reshape_and_cache first).
#
# Env:
#   ROCM_FLASH_DECODE=0        disable (fall back to stock path)
#   ROCM_FLASH_DECODE_CHECK=1  also run a torch fp32 oracle, assert match, log max rel err
#   ROCM_FLASH_DECODE_MAXQ=8   max query_len treated as decode/spec (else prefill)
import os
import logging

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

logger = logging.getLogger(__name__)
_LOGGED = {"ok": False, "off": False, "rej": False, "check": False}


def _num_kv_splits(max_seq_len: int) -> int:
    # enough splits to fill ~96 CUs at low batch; capped to bound the reduce cost
    s = (int(max_seq_len) + 1023) // 1024
    return max(8, min(64, s))


@triton.jit
def _native_grouped_stage1(
    Q,
    K_Cache,
    V_Cache,
    sm_scale,
    Block_Table,
    B_Seqlen,
    Att_Out,
    stride_bt_b,
    stride_qbs,
    stride_qh,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_k4,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    # native K x-split addressing: element (d) -> (d // X) * stride_k2 + (d % X) * stride_k4
    d_outer = offs_d // X
    d_inner = offs_d % X
    k_row = cur_kv_head * stride_k1 + d_outer[:, None] * stride_k2 + d_inner[:, None] * stride_k4
    v_col = cur_kv_head * stride_v1 + offs_dv[None, :] * stride_v2

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            logical_block = offs_n // BLOCK_SIZE
            pos = offs_n % BLOCK_SIZE
            phys_block = tl.load(
                Block_Table + cur_batch * stride_bt_b + logical_block,
                mask=mask_n, other=0,
            )
            # K tile: [BLOCK_DMODEL, BLOCK_N]
            k_col = phys_block[None, :] * stride_k0 + pos[None, :] * stride_k3
            offs_k = k_col + k_row
            k = tl.load(
                K_Cache + offs_k,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * (2 * tl.sigmoid(2 * qk / logit_cap) - 1)
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            # V tile: [BLOCK_N, BLOCK_DV]
            v_row = phys_block[:, None] * stride_v0 + pos[:, None] * stride_v3
            offs_v = v_row + v_col
            v = tl.load(
                V_Cache + offs_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            if v.dtype.is_fp8():
                v = (v.to(tl.float32) * vs).to(q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None],
                 mask=(mask_h[:, None]) & (mask_dv[None, :]))
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


def _native_decode_one(q, key_cache, value_cache, o, block_table, blen,
                       num_kv_splits, sm_scale, k_scale, v_scale, head_size, x):
    # q: [B, H, D]  o: [B, H, D]  (fp16)
    B, H, D = q.shape
    num_kv_heads = value_cache.shape[1]
    block_size = value_cache.shape[3]
    kv_group_num = H // num_kv_heads
    dev = q.device

    logits = torch.empty(B, H, num_kv_splits, head_size + 1, device=dev, dtype=torch.float32)
    lse = torch.empty(B, H, device=dev, dtype=torch.float32)

    BLOCK_DMODEL = triton.next_power_of_2(head_size)
    BLOCK_DV = triton.next_power_of_2(head_size)
    BLOCK_N = 16
    BLOCK_H = 16
    grid = (B, triton.cdiv(H, min(BLOCK_H, kv_group_num)), num_kv_splits)
    extra = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _native_grouped_stage1[grid](
        q, key_cache, value_cache, sm_scale, block_table, blen, logits,
        block_table.stride(0),
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.stride(3), key_cache.stride(4),
        value_cache.stride(0), value_cache.stride(1),
        value_cache.stride(2), value_cache.stride(3),
        logits.stride(0), logits.stride(1), logits.stride(2),
        k_scale, v_scale,
        kv_group_num=kv_group_num, q_head_num=H,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits, BLOCK_SIZE=block_size, X=x,
        logit_cap=0.0, Lk=head_size, Lv=head_size,
        num_warps=4, num_stages=1, **extra,
    )
    _fwd_kernel_stage2[(B, H)](
        logits, o, lse, blen,
        logits.stride(0), logits.stride(1), logits.stride(2),
        o.stride(0), o.stride(1), lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits, BLOCK_DV=BLOCK_DV, Lv=head_size,
        num_warps=4, num_stages=2,
    )


def _torch_oracle(qj, key_cache, value_cache, block_table, blen, sm_scale,
                  k_scale, v_scale, head_size, x):
    # Slow fp32 ground truth for one query position. Dequants + de-interleaves the
    # native cache into [seqlen, H, D] per sequence and does plain attention.
    B, H, D = qj.shape
    num_kv_heads = value_cache.shape[1]
    block_size = value_cache.shape[3]
    g = H // num_kv_heads
    ks = float(k_scale); vs = float(v_scale)
    out = torch.empty_like(qj)
    for b in range(B):
        L = int(blen[b].item())
        nb = (L + block_size - 1) // block_size
        blocks = block_table[b, :nb].to(torch.long)
        # gather referenced blocks first, then cast (avoid fp32-casting the whole cache)
        kc = key_cache[blocks].to(torch.float32)   # [nb,Hk,D//x,bs,x]
        vc = value_cache[blocks].to(torch.float32)  # [nb,Hk,D,bs]
        # K: [nb,Hk,D//x,bs,x] -> [nb,bs,Hk,D]
        kk = kc.permute(0, 3, 1, 2, 4).reshape(nb, block_size, num_kv_heads, head_size)
        vv = vc.permute(0, 3, 1, 2).reshape(nb, block_size, num_kv_heads, head_size)
        kk = kk.reshape(nb * block_size, num_kv_heads, head_size)[:L] * ks
        vv = vv.reshape(nb * block_size, num_kv_heads, head_size)[:L] * vs
        for h in range(H):
            kh = kk[:, h // g, :]  # [L,D]
            vh = vv[:, h // g, :]
            qh = qj[b, h].to(torch.float32)  # [D]
            s = (kh @ qh) * sm_scale  # [L]
            p = torch.softmax(s, dim=0)
            out[b, h] = (p[:, None] * vh).sum(0).to(qj.dtype)
    return out


def try_native_flash_decode(
    query, output, key_cache, value_cache, block_table,
    query_start_loc, seq_lens, max_query_len, max_seq_len,
    k_scale, v_scale, sm_scale, kv_cache_dtype, fp8_dtype,
    alibi_slopes, sliding_window, sinks, causal,
) -> bool:
    """Returns True iff it filled `output` and the caller should return early.
    In CHECK mode it validates against a torch oracle but still returns False
    so the stock path produces the served result."""
    if os.environ.get("ROCM_FLASH_DECODE", "1") == "0":
        return False
    if alibi_slopes is not None or sinks is not None or not causal:
        return False
    if sliding_window not in (0, None, -1, (-1, -1)):
        return False
    Q = int(max_query_len)
    maxq = int(os.environ.get("ROCM_FLASH_DECODE_MAXQ", "8"))
    num_seqs = seq_lens.shape[0]
    num_tokens = query.shape[0]
    if Q < 1 or Q > maxq or num_tokens != num_seqs * Q:
        if not _LOGGED["rej"]:
            logger.warning("[native_flash_decode] skip: Q=%s tok=%s seqs=%s",
                           Q, num_tokens, num_seqs)
            _LOGGED["rej"] = True
        return False

    check = os.environ.get("ROCM_FLASH_DECODE_CHECK") == "1"
    try:
        kc, vc = key_cache, value_cache
        if "fp8" in kv_cache_dtype:
            kc = kc.view(fp8_dtype)
            vc = vc.view(fp8_dtype)
        num_heads = query.shape[1]
        head_size = query.shape[2]
        x = kc.shape[4]
        NKV = _num_kv_splits(max_seq_len)
        qv = query.view(num_seqs, Q, num_heads, head_size)
        ov = output.view(num_seqs, Q, num_heads, head_size)
        dst = ov if not check else torch.empty_like(ov)
        max_rel = 0.0
        for j in range(Q):
            qj = qv[:, j, :, :].contiguous()
            blen = (seq_lens - (Q - 1) + j).to(torch.int32)
            oj = torch.empty(num_seqs, num_heads, head_size, device=query.device, dtype=query.dtype)
            _native_decode_one(qj, kc, vc, oj, block_table, blen, NKV, sm_scale,
                               k_scale, v_scale, head_size, x)
            dst[:, j, :, :] = oj
            if check:
                ref = _torch_oracle(qj, kc, vc, block_table, blen, sm_scale,
                                    k_scale, v_scale, head_size, x)
                denom = ref.abs().amax().clamp_min(1e-4)
                rel = (oj.to(torch.float32) - ref.to(torch.float32)).abs().amax() / denom
                max_rel = max(max_rel, float(rel))
        if check:
            ckey = f"worst_Q{Q}"
            prev = _LOGGED.get(ckey, 0.0)
            if max_rel > prev * 1.5 or max_rel > prev + 1e-3:
                logger.warning("[native_flash_decode] CHECK Q=%d NKV=%d maxseq=%d max_rel=%.4g",
                               Q, NKV, int(max_seq_len), max_rel)
                _LOGGED[ckey] = max_rel
            assert max_rel < 5e-2, f"native_flash_decode mismatch rel={max_rel}"
            return False  # let stock path fill the served output
        if not _LOGGED["ok"]:
            logger.warning("[native_flash_decode] active: Q=%d NKV=%d head=%d x=%d maxseq=%d",
                           Q, NKV, head_size, x, int(max_seq_len))
            _LOGGED["ok"] = True
        return True
    except Exception as e:
        if not _LOGGED["off"]:
            logger.warning("[native_flash_decode] disabled after error: %r", e)
            _LOGGED["off"] = True
        return False
