# Inkling ROCm TP4 small-message peer writes (Asgard)

This is an optional backend for Inkling's target attention and expert-output
all-reduces on four `gfx1100` devices with direct same-host PCIe P2P. It is not
an RCCL replacement, a lossy transport, or a flag for other model families.

## Build and opt in

From the same vLLM source checkout as the worker, with ROCm's `hipcc`:

```sh
bash tools/asgard/build_gfx1100_peer_push.sh
```

The script builds only `vllm/_gfx1100_peer_push.so` for `gfx1100`, with GPU
visibility masked during the build. It does not modify `_C`, `_rocm_C`, a ROCm
runtime or an installed wheel. The optional module has a narrow C ABI and an
embedded source hash; no PyTorch C++ ABI dependency. It must be rebuilt after
changes to its HIP source, or when a fresh checkout has no build artifact.

`VLLM_ROCM_INKLING_PEER_PUSH_MAX_ROWS=0` (default) selects the ordinary backend
and does not load the optional library. `8` selects contiguous rank-local
BF16 `[1,2304]` and `[8,2304]` target tensors. `32` also accepts `[32,2304]` for
controlled experiments; the Asgard serving profile is scoped to `8`. Other
shapes, embeddings, the drafter, and larger prefill collectives keep their
existing paths. This is a per-engine startup policy; restart to change it.

The worker validates four distinct PCI devices, hardware peer accessibility,
ABI/build consensus, and the supported Inkling TP4 generation configuration.
Do not combine it with eager-only experiments unless separately qualified, or
with multiple compute streams, DP/PP/context parallelism, DBO, sleep mode,
elastic EP, batch invariance, or torch.compile. Rank-local fallback after any
IPC setup failure is not permitted.

## Execution and errors

Each source uses logical rank order for FP32 accumulation and converts the
sum to BF16 once. Dedicated IPC arenas contain separate state per shape and
separate payload slots for adjacent epochs. The only launch streams are the
ordered worker compute stream and its coordinated graph capture stream.
Gathering IPC handles and opening/closing mappings are communicator setup and
teardown, not graph operations.

A missing publisher is bounded by device wall-clock and an iteration cap.
Faults become sticky on the local and peer arenas, including across shapes.
The sampling output path snapshots this status on its existing D2H copy stream,
waits for the existing copy event, and rejects poisoned output before emitting
any sampled tokens. A fault invalidates the engine; do not retry a partially
completed reduction or switch backends inside that engine. Restart with `0`
for the ordinary path. When normal cleanup cannot establish that peers have
closed their imports, the worker processes own reclamation at exit.

## Tests

- CPU shape, missing build and output-fault controls:
  `python -m pytest tests/distributed/test_gfx1100_peer_push.py`.
- `tools/asgard/test_gfx1100_peer_push.py` is a **four-GPU** check for dedicated
  idle hardware, under Asgard's existing memory guard (`torchrun --standalone
  --nproc-per-node=4`). It verifies exact reference arithmetic, changing-input
  graph replay across shapes and a bounded missing rank on otherwise healthy
  workers. It must not run beside a live service.
- A deployment check should verify real replay/backend selection, matched
  emitted-token throughput and tails, engine failure propagation, prefix and
  native-context correctness, and normal protocol/long-cache regression gates.
  Faster isolated collectives alone do not justify changing the default.
