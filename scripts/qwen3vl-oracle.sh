#!/usr/bin/env bash
# Generate the llama.cpp reference embeddings for the Qwen3-VL parity test (WI-2).
#
# WHY THIS USES THE SIBLING TREE
#
# The primary reference, old/repo/llama.cpp-master, DOES NOT BUILD. Its
# ggml/src/ggml-cpu/ggml-cpu.c:7 does `#include "iqp.h"`, and iqp.h was never
# vendored (missing since 2026-09-29), so the core CPU backend does not compile:
#
#   ggml/src/ggml-cpu/ggml-cpu.c:7:10: fatal error: iqp.h: No such file or directory
#
# The sibling tree old/repo/llama.cpp-xing4_0-port carries the genuine upstream
# iqp.h, and that makes it a valid oracle because the forward spec is
# BYTE-IDENTICAL between the two:
#
#   cmp old/repo/llama.cpp-master/tools/mtmd/models/qwen3vl.cpp \
#       old/repo/llama.cpp-xing4_0-port/tools/mtmd/models/qwen3vl.cpp   # identical
#
# So the spec grim is compared against does not depend on which tree supplies it.
# Both trees were checked: the spatial-merge block (qwen3vl.cpp:18-31) and the
# MTMD_DEBUG_EMBEDDINGS dump path are the same in each.
#
# old/repo/ is used READ-ONLY. llama.cpp's own AGENTS.md forbids AI-authored
# changes to that repository, so this script builds out-of-tree into a mktemp
# directory and never writes into old/repo/. Nothing here is committed there.
#
# Usage:
#   scripts/qwen3vl-oracle.sh <out-dir>      # explicit output directory
#   OUT=$(scripts/qwen3vl-oracle.sh)          # or print a fresh mktemp path
#
# Then run the test:
#   GRIM_QWEN3VL_ORACLE=$OUT/oracle.bin \
#   GRIM_QWEN3VL_IMAGE=$OUT/probe.png \
#     cargo test --release -p grim-models-vision --test qwen3vl_parity \
#       -- --ignored --nocapture --test-threads=1
#
# Release is required: the CPU forward is ~330 GFLOP of scalar f32 and takes
# minutes in debug.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/old/repo/llama.cpp-xing4_0-port"
MODEL="$REPO/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf"
MMPROJ="$REPO/models/qwen38-27b/mmproj-F16.gguf"

if [ $# -ge 1 ]; then OUT="$1"; else OUT="$(mktemp -d -t qwen3vl-oracle-XXXXXX)"; fi
mkdir -p "$OUT"
[ -f "$MMPROJ" ] || { echo "missing $MMPROJ" >&2; exit 2; }
[ -d "$SRC" ]   || { echo "missing oracle source tree $SRC" >&2; exit 2; }

# Fail loudly if the spec ever diverges between the two trees, because the whole
# justification for using the sibling tree rests on that identity.
if ! cmp -s "$REPO/old/repo/llama.cpp-master/tools/mtmd/models/qwen3vl.cpp" \
            "$SRC/tools/mtmd/models/qwen3vl.cpp"; then
  echo "FATAL: the two llama.cpp trees' qwen3vl.cpp have diverged." >&2
  echo "This script only uses the sibling tree because the spec is identical." >&2
  echo "Re-check which tree matches grim's implementation before trusting output." >&2
  exit 3
fi

# A deterministic, asymmetric image. A solid colour is invariant under any
# spatial permutation, so a permuted spatial merge would agree with the oracle
# and the parity test would PASS while the encoder were wrong.
python3 "$REPO/scripts/qwen3vl-test-image.py" "$OUT/probe.png" 768

BT="$OUT/build"
cmake -S "$SRC" -B "$BT" -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF >"$OUT/cmake.log" 2>&1
cmake --build "$BT" --target llama-mtmd-cli -j "$(nproc)" >>"$OUT/cmake.log" 2>&1

CLI="$BT/bin/llama-mtmd-cli"
[ -x "$CLI" ] || { echo "llama-mtmd-cli did not build; see $OUT/cmake.log" >&2; exit 2; }

DUMP="$OUT/oracle.bin"
# --no-mmproj-offload and -ngl 0 keep everything on CPU: the oracle must not
# depend on a GPU, so it cannot be perturbed by device selection or a peer agent.
MTMD_DEBUG_EMBEDDINGS="$DUMP" "$CLI" \
    -m "$MODEL" --mmproj "$MMPROJ" \
    --image "$OUT/probe.png" \
    -p "Describe this image in one sentence." \
    --no-mmproj-offload -ngl 0 -t "$(nproc)" >"$OUT/run.log" 2>&1 || {
      echo "oracle run failed; see $OUT/run.log" >&2; tail -20 "$OUT/run.log" >&2; exit 1
    }

[ -s "$DUMP" ] || { echo "no oracle dump produced" >&2; exit 1; }

# Report the header so the test asserts it rather than discovering it.
python3 - "$DUMP" <<'PY'
import os, struct, sys
p = sys.argv[1]
with open(p, "rb") as fh:
    n_tok, n_emb = struct.unpack("<ii", fh.read(8))
n_data = (os.path.getsize(p) - 8) // 4
print(f"oracle: n_tokens={n_tok} n_embd={n_emb} data_floats={n_data}")
assert n_tok > 0 and n_emb > 0, "oracle header is empty"
assert n_data == n_tok * n_emb, f"oracle data {n_data} != {n_tok}*{n_emb}; truncated"
PY

echo "ORACLE: $DUMP"
echo "IMAGE:  $OUT/probe.png"
[ -n "${1:-}" ] || echo "DIR:    $OUT"
