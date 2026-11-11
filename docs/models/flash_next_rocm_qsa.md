# Flash-Next QSA on gfx1100

The AMD Qwen4Exp owner already implements sparse QSA. These optimizations do
not change weights, the selection budget, KV precision, or sampling policy.

## Reuse the scorer's mask

The gfx1100 PyTorch top-k fallback reuses the paged scorer's causal `-inf`
mask, avoiding a redundant full-plane mask. Other callers of the fallback
retain its original masking behavior. Unique-score indices are unchanged;
tied indices were not stable across repeated ROCm PyTorch top-k calls even
before this change.

## All-block selection at short contexts

With a 2048-token budget and compression ratio 4, every complete block is
selected through 2051 visible tokens. If the caller supplies a known host
upper bound in that range, the selector directly emits all those blocks and
the causal tail. It does not allocate or score a logits plane. Above the
bound, or without a known bound, ordinary score/top-k selection is retained.

`VLLM_ROCM_QSA_ALL_BLOCKS=0` disables this optimization; the default is `1`.
The caller still updates compressed keys and recurrent tails before selection.
The existing indexer supplies the compact bound for prefill, not captured
uniform decode: this is not a claim that all decode steps take the shortcut.
Selection is emitted in token order, so floating-point accumulation order can
change. Numerical parity is tested; seedwise generated-token identity is not
promised.

## Bounded-memory prefill tuning (opt-in)

`VLLM_ROCM_QSA_PREFILL_TUNED=1` opts into gfx1100 tuning for the TP4 Flash-Next
geometry (six query heads, one KV head). Its default is `0`, pending full
endpoint/context qualification. The tuning uses a 16-token attention tile,
16 splits and two warps on 32--128 query rows. Wider steps are processed in
128-row windows. Each row still reads its whole selected history, and strided
output views and padded rows are supported. Other geometries and small decode
shapes retain the old policy.

The window limits FP32 split scratch to less than 16 MiB rather than allocating
approximately 192 MiB for a 2048-row split16 launch. Raising temporary memory
without accounting for the long-context cache can OOM the serving profile.

A different experimental BF16 tile32/split1 launch faulted during tuning. It
is not part of this policy and must not be copied from intermediate results.

## Validation and benchmark scope

Correctness gates live in `tests/models/qwen4_exp/test_qsa_all_blocks_amd.py`
and `test_qsa_amd.py`: short-context boundary 2051/2052, multiple requests,
padding, FP8/BF16 attention, row-window boundaries, strided output, and a
less-than-16-MiB scratch check. CPU/GPU top-k tests cover ragged rows, ties,
and budgets wider than the score plane.

The bounded component benchmark is
`~/rukhm/benchmarks/flash_next_qsa_microbench.py`. It requires an idle serving
stack, pins one discrete GPU and caps the PyTorch allocator. Raw ABBA receipts
and limitations are recorded in the corresponding rukhm qualification report.
Component speedups must not be presented as endpoint tokens/second gains.
