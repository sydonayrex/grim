#!/usr/bin/env bash
# WI-E2 / FIND-3: parity harness — run the same serving load against grim and
# vLLM, print both result sets for a manual docs/benchmarks/gfx1036.md table.
#
# Usage: scripts/parity-vs-vllm.sh [port] [concurrency] [duration_secs]
# Requires: a running `grim-cli serve` on $PORT (grim leg), and vllm serve on
# $VLLM_PORT for the vllm leg. Each leg is optional; missing servers are
# skipped with a note.

set -u
PORT="${1:-11434}"
CONC="${2:-4}"
DUR="${3:-60}"
VLLM_PORT="${VLLM_PORT:-8000}"

GRIM_BIN="$(dirname "$0")/../target/release/grim-cli"

echo "== grim (127.0.0.1:$PORT, conc=$CONC, ${DUR}s) =="
if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    "$GRIM_BIN" bench --mode serve --port "$PORT" --concurrency "$CONC" --duration "$DUR"
else
    echo "SKIP: no grim server on :$PORT"
fi

echo
echo "== vllm (127.0.0.1:$VLLM_PORT) =="
if command -v python3 >/dev/null 2>&1 && python3 -c "import vllm" 2>/dev/null; then
    # vllm's bench_serving: same shape load (random 128-token outputs).
    python3 -m vllm.entrypoints.cli.main bench serving \
        --backend vllm \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --num-prompts "$((CONC * 10))" \
        --request-rate "$CONC" \
        --dataset-name random \
        --random-input-len 256 --random-output-len 128 2>&1 | tail -20
else
    echo "SKIP: vllm not installed in this environment"
fi

echo
echo "Paste both blocks into docs/benchmarks/gfx1036.md with date + model + backend."
