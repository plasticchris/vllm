#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds only the optional native HIP TP4 module, without touching existing vLLM libraries.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
SRC="$ROOT/csrc/rocm/gfx1100_peer_push.hip"
OUT="${1:-$ROOT/vllm/_gfx1100_peer_push.so}"
[[ "$OUT" == *.so ]] || { echo 'expected an output .so path' >&2; exit 2; }
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
export HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=
TMP=$(mktemp -d "$(dirname -- "$OUT")/.peerpush-build.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
hash=$(sha256sum "$SRC"); hash=${hash%% *}
"$HIPCC" -O3 --offload-arch=gfx1100 -fno-fast-math -ffp-contract=off -fPIC -shared \
    -DVLLM_PP_SOURCE_SHA256=\""$hash"\" "$SRC" -o "$TMP/peer_push.so"
# A build that can't be loaded or is stale must never be silently selected.
python3 - "$TMP/peer_push.so" "$hash" <<'PY'
import ctypes as c
import sys

lib = c.CDLL(sys.argv[1])
lib.vllm_pp_source_sha256.restype = c.c_char_p
if lib.vllm_pp_abi_version() != 1 or lib.vllm_pp_source_sha256().decode() != sys.argv[2]:
    raise RuntimeError("gfx1100 peer-push build identity mismatch")
PY
mv -f "$TMP/peer_push.so" "$OUT"
printf 'gfx1100 peer push ABI 1, source sha256=%s\n' "$hash"
sha256sum "$OUT"
