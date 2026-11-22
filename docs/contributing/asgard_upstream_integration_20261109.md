# Asgard upstream integration, 2026-11-09

## Scope and isolation

Merge `vllm-project/vllm` main at `3b33a845c6` into the Asgard serving source
at `2e01aaee01`. Candidate worktree: `/mnt/scratch/vllm/vllm-upstream-20261109`,
branch `merge/upstream-20261109`. The live `vllm-src` checkout remains at its
original revision. No live library replacement, environment update, or server
restart is part of this source merge.

## Port decisions

- Honor upstream's removal of Model Runner V1. Keep the Leyline copy/rotation
  and Mamba transplant logic in `worker/gpu/model_runner.py`; do not resurrect
  the deleted monolithic runner or its buffer classes.
- Keep upstream's request-count/query-length graph descriptor scheme. Migrate
  the CPU INT8 ladder regression to `decode_query_lens={3: 4}`. Preserve replay
  diagnostics, but correct their old eager-only claim (that diagnosis was
  retracted; live execution-mode logs showed FULL replay).
- Keep upstream's split HIP translation units. Port BF16 widths 6..12/16 to
  the new launcher, instantiate INT4 widths 6..8 in separate units, and retain
  all six local INT8 compilation units and the WMMA W4A16 source.
- Keep upstream's standalone ROCm QSA implementation and Octave support rather
  than reintroducing a FlashAttention inheritance workaround. Retain the local
  scaled FP8 cache path and FP8/FNUZ dtype handling.
- Port fused draft metadata helpers through `qsa_cache` -> `qsa_indexer`
  renaming. Preserve the new metadata structure and its upstream tests.
- Combine upstream FlashInfer checkpoint/export and state-gather/scatter with
  local short recurrent-prefill and Triton in-place output paths in GDN.
- Preserve upstream Mamba checkpoint publication and request-order routing.
  Apply the local subblock cap BEFORE encoder clipping: a second alignment
  pass after clipping made valid multimodal chunks stall at zero tokens.
- Preserve upstream lone-prefill/full-budget and adaptive fair-share behavior;
  use the separate explicit decode-active cap only when configured.
- Replace removed config `compute_hash()` calls with `hash_config()`. Replace
  the old request cache-skip boolean with `prefix_cache_hit_limit`: fail closed
  for ANY bounded request so splicing cannot bypass prompt scoring.
- Retain all upstream tests in heavily interleaved test conflicts, then add
  the local-only tests. Update obsolete output-index and token-list APIs.
  Synthetic all-boundary splice fixtures explicitly select the manager's
  dense allocation oracle to model populated donor checkpoints; this is NOT
  restoration of a supported public `mamba_cache_mode=all`. The separate
  unaligned/speculative-scratch test still exercises real align-mode behavior
  with an explicit 16-token step budget.

## Completed gates

Commands and logs are under `/mnt/scratch/vllm/upstream-integration-20261109/`.

- `verify_cpu.sh`: 383 tests in scheduler/cache-bound/config/graph suites,
  plus 4 Mamba transplant/guard tests, plus 3 QSA draft-update tests: **390 PASS**.
- Ruff undefined-name checks and `git diff --check`: PASS.
- Isolated CMake `_rocm_C` Release build, gfx1100, 4 jobs: PASS (52 steps).
- CPU-only `torch.ops.load_library` of that extension and lookup of `wvSplitK`,
  `wvSplitK_int4_g`, `wvSplitK_int8_g`: PASS.

These gates borrowed Torch 2.11 and Transformers 5.16.1 from an existing venv
through a separate staging environment. Upstream expects Torch 2.14. A successful
`_rocm_C` build does not prove that the entire new vLLM extension set or engine
works against the old Torch ABI. Model GPU correctness/performance, the complete
Torch 2.14 stack, and a supervised deployment/rollback test remain pending.
GPU tests were not run against the shared live server. Full QSA-reference runs
with devices hidden fail GPU-only cases as expected; only the explicitly named
CPU subset above is counted as passing. No throughput/quality improvement is
claimed.
