"""DCA per-pass paged metadata (decode path), ported from the v0 backend to the
v1 TritonAttentionMetadata field layout.

For a cached sequence length L, the decode query (position L-1) lives in chunk
c = (L-1)//chunk_len. The three passes attend disjoint key ranges that partition
[0, L):

    intra : [c*chunk_len, L)                 causal, query variant = intra
    succ  : [(c-1)*chunk_len, c*chunk_len)   full,   query variant = succ   (c>=1)
    inter : [0, (c-1)*chunk_len)             full,   query variant = inter  (c>=2)

Each pass gets its own seqused_k (the length above) and a block_table sliced to
start at range_start//block_size, mirroring the v0 builder. A YaRN-style scalar
rescales the softmax scale once L exceeds original_max_position_embeddings.

Pure torch; no vLLM import, so it runs anywhere (tests force CPU).
"""
from dataclasses import dataclass

import torch


@dataclass
class DCAPassMeta:
    seq_lens: torch.Tensor      # [batch] int32 — keys this pass attends, per request
    block_table: torch.Tensor   # [batch, n_blocks] — left-aligned, zero-padded
    max_seq_len: int


@dataclass
class DCADecodeMeta:
    intra: DCAPassMeta
    succ: DCAPassMeta
    inter: DCAPassMeta
    scaling_factor: torch.Tensor  # [batch] float — multiply the softmax scale


def _slice_block_table(block_table, range_start, max_seq_len_pass, last_valid_block, block_size):
    """Left-align block_table[i, start_blk : ed] where start_blk = range_start[i]//block_size,
    ed bounded by the request's own last valid block. Mirrors v0 exactly."""
    batch = block_table.shape[0]
    if max_seq_len_pass <= 0:
        return block_table[:, :0].clone()
    nblk = (max_seq_len_pass - 1) // block_size + 1
    out = torch.zeros(batch, nblk, dtype=block_table.dtype, device=block_table.device)
    start_blk = (range_start // block_size).to(torch.int64)
    for i in range(batch):
        st = int(start_blk[i].item())
        ed = min(st + nblk, int(last_valid_block[i].item()))
        if ed > st:
            out[i, : ed - st] = block_table[i, st:ed]
    return out


def build_dca_decode_metadata(
    cache_seq_lens,                      # [batch] int — total KV length per request
    block_table,                         # [batch, max_blocks] int — v1 paged table
    chunk_size,
    local_size,
    original_max_position_embeddings,
    block_size,
):
    device = block_table.device
    batch = cache_seq_lens.shape[0]
    chunk_len = chunk_size - local_size
    L = cache_seq_lens.to(torch.int64)
    c = ((L - 1) // chunk_len).clamp(min=0)                  # current chunk index
    last_valid_block = (L - 1) // block_size + 1            # exclusive block bound per request

    if original_max_position_embeddings > 0:
        scaling_factor = (
            0.1 * torch.log(L.float() / original_max_position_embeddings) + 1.0
        ).clamp(min=1.0)
    else:
        scaling_factor = torch.ones(batch, dtype=torch.float32, device=device)

    # intra — current chunk [c*chunk_len, L)
    seq_lens_intra = (L - c * chunk_len).to(torch.int32)
    max_intra = int(seq_lens_intra.max().item()) if batch else 0
    bt_intra = _slice_block_table(block_table, c * chunk_len, max_intra, last_valid_block, block_size)

    # succ — previous chunk [(c-1)*chunk_len, c*chunk_len); length chunk_len for c>=1, else 0
    seq_lens_succ = ((c - (c - 1).clamp(min=0)) * chunk_len).to(torch.int32)
    max_succ = int(seq_lens_succ.max().item()) if batch else 0
    succ_start = (c - 1).clamp(min=0) * chunk_len
    bt_succ = _slice_block_table(block_table, succ_start, max_succ, last_valid_block, block_size)

    # inter — everything before succ [0, (c-1)*chunk_len); starts at block 0
    seq_lens_inter = ((c - 1).clamp(min=0) * chunk_len).to(torch.int32)
    max_inter = int(seq_lens_inter.max().item()) if batch else 0
    if max_inter > 0:
        nblk = (max_inter - 1) // block_size + 1
        bt_inter = block_table[:, :nblk].clone()
    else:
        bt_inter = block_table[:, :0].clone()

    return DCADecodeMeta(
        intra=DCAPassMeta(seq_lens_intra, bt_intra, max_intra),
        succ=DCAPassMeta(seq_lens_succ, bt_succ, max_succ),
        inter=DCAPassMeta(seq_lens_inter, bt_inter, max_inter),
        scaling_factor=scaling_factor,
    )
