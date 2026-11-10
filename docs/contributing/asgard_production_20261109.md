# Asgard Qwen3.8 27B production integration — 2026-11-09

This records the **tested production configuration**, including a dependency
patch that originally existed only in a local virtual environment. It does not
claim reproducibility on a different host or qualification of other models.
The code loaded at the final serving cutover was vLLM `d151776367`; this
runbook and the accompanying patch add no vLLM runtime changes on top.

## Source branches

All are personal forks; the branch heads listed below were used in the tested
stack. Kineto and MSLK must be pushed before PyTorch: its `.gitmodules` points
at those forks and its gitlinks pin the commits shown.

| Project | Personal fork / branch | Tested source commit |
| --- | --- | --- |
| vLLM | `plasticchris/vllm`, `merge/upstream-20261109` | `d151776367118322ac66a281282d808724ccecb3` plus this documentation commit |
| PyTorch | `plasticchris/pytorch`, `integrate/torch-2.14-20261109` | `4d16e8cf99a3f65adea0e2d76b84433ecb0d3751` |
| Kineto | `plasticchris/kineto`, `integrate/torch-2.14-20261109` | `1ac15e0280d97ab5e7ca4b7d51ff5d3c6481ea6d` |
| MSLK | `plasticchris/MSLK`, `integrate/torch-2.14-20261109` | `5ce51bcf2145bc509cf27cb8906ef53db97c0bce` |
| Triton | `plasticchris/triton`, `integrate/triton-3.9-20261109` | `b7ddcf86c095de79527ba3098503e73158bb94bd` |
| torchvision | `plasticchris/vision`, `integrate/vision-0.29-20261109` | `230957624c30adba69ecd119fe8e9b5b1b867439` |
| RCCL | `plasticchris/rccl`, `integrate/rccl-develop-20261109` | `826ec54b0ded66433bd6827ee8971f01ab198d22` |
| ROCr | `plasticchris/rocm-systems`, `integrate/rocr-develop-20261109` | `0900204aed48b04e20cc435d8673ed99ca57605d` |
| Serving launcher | `plasticchris/rukhm`, `main` | `5d0534104e2e62fb1882b794acc2283ce5e3fcd3` |

AITER `plasticchris/aiter`, branch `integrate/aiter-20261109`, commit
`965f3a6649b1059decb0eeedb8233ce1b97612d6`, was separately integrated and its
packaging-only wheel built. It is **not deployed or GPU-qualified** for this
model. Its composable-kernel submodule is `96afae1d54fc79c26857de7e1683b0dde1ff3c6f`.

## Binary artifacts and environment

The serving environment is Python 3.12 under
`/mnt/scratch/vllm/upstream-integration-20261109/venv` on asgard (ROCm 7.2.4,
four gfx1100 GPUs). Its `.pth` adds the vLLM source worktree **and**
`/mnt/scratch/vllm/venv-dsv41/lib/python3.12/site-packages`; therefore this
is not an isolated fully locked or self-contained environment. Preserve or
reconstruct those dependencies rather than assuming these sources alone are
sufficient for a fresh deployment.

| Artifact | Version / SHA256 |
| --- | --- |
| torch wheel | `2.14.0+rocm7.2.4.asgard`; `716d22f1a92c0aefa22c3aae8baca1b19a589363530636d0b8a094482e0b6455` |
| torchvision wheel | `0.29.0a0+2309576`; `caa55ec8f8de0342ea9453a21dfb78a7c475f1ebda3e4284f8f51c442b635cdb` |
| Triton wheel | `3.9.0+gitb7ddcf86`; `c6a944d70a5622a493a20995a2dfaa1040d810d4429ebec5857db38ee1ef4131` |
| RCCL library | `29611bd7aa434ec2ac87c979e97f7ce804291f7cda2b428d200dc1419ae61a4d` |
| ROCr library | `e45351d1574b99b22921be154c5a9e69f45bc0ad70af127d9e3a4d7de8880a5c` |
| vLLM `_rocm_C.abi3.so` | `a6fa2dd716d28874b85a190c7ebe67a3671766b4c2c7f8b04d0a65778f671a56` |
| transformers / fastokens | `5.16.1` / `0.3.6` |

These are hashes of local artifacts, **not binaries stored in this repository**.
The ROCr build includes the 39 default image-blit targets supported by the
installed compiler. `gfx1170/1171` image blits are explicitly disabled; no
fake replacement blobs are provided. Its build uses the opt-in
`ROCR_DISABLE_GFX117X_IMAGE_BLIT=ON` and `-Wl,-z,defs`. The local build recipe
and logs are in `/mnt/scratch/vllm/upstream-integration-20261109/`; the system
ROCr files were not replaced.

## FlashAttention Triton 3.9 compatibility patch

Production uses a local copy of `flash_attn` 2.8.3. Its
`flash_attn_triton_amd/bwd_prefill_onekernel.py` autotune keys contain names
absent from their decorated kernels. Triton 3.9 validates those keys at import
and rejects the unpatched file. The patch in this branch is:

`tools/asgard/patches/flash-attn-2.8.3-triton-3.9-autotune-keys.patch`

Apply it **to an isolated copy**, from the directory containing `flash_attn/`:

```bash
patch -p1 --ignore-whitespace --forward < /path/to/vllm/tools/asgard/patches/flash-attn-2.8.3-triton-3.9-autotune-keys.patch
```

Verify the file hashes before and after applying; do not blindly patch an
unknown build:

- Unpatched local 2.8.3 file: `ff843c3e5754e607adf7605abbe1e67f99d4280c9424507ddef9f098f407f86a`
- Production patched file: `df0a8f973e216f5a3c6f19603ee8fcc68e69695916b249d7d595ee49a99c96e4`
- Patch: `04b35fc0cb33de54be36e13c9849e26d092cb051c1931078e5d91d9b64c71137`

The local `flash_attn` copy came from `venv-dsv41`, not from a source fork or a
recorded reproducible wheel. Its Python package files were compared: this was
the only modified `.py` file. Its top-level import requires a CUDA extension
that was not built; **do not claim a complete FlashAttention installation or
backward-kernel GPU qualification** from the production serving test.

## Qualification and rollback

The repaired stack passed a matched 120K-context C1 and 24K-context C4
cold/warm model performance gate, plus deterministic short baseline matches and
streaming. The ROCr candidate separately passed four-GPU allocation, transfer,
stream/event, GEMM and RCCL-all-reduce tests, followed by the same-stack
long-context model gate. The actual deployment passed a post-cutover canary.
These checks do **not** substitute for a long soak or broader model/operation
coverage. Full logs and the currently deployed local artifacts remain in
`/mnt/scratch/vllm/upstream-integration-20261109/`.

On asgard, the launcher selects ROCr per process. `RUKHM_ROCR=system` in a
clean environment returns the merged stack to system ROCr; `RUKHM_STACK=legacy`
in a clean environment selects the prior vLLM/PyTorch stack. Both require a
coordinated server restart. Never overwrite a library currently mapped by a
running process, and do not replace the global ROCm installation to apply this
patch.
