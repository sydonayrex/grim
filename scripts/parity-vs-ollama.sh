#!/usr/bin/env bash
# Parity harness — run concurrent serving load against grim and Ollama,
# comparing throughput (tokens/sec), request latency, and ITL percentiles.
#
# Usage:
#   scripts/parity-vs-ollama.sh [model] [concurrency] [duration_secs] [grim_port] [ollama_port]
#
# Examples:
#   scripts/parity-vs-ollama.sh qwen3.8:latest 4 30 11435 11434
#   scripts/parity-vs-ollama.sh gemma4:latest 2 20
#
# Defaults:
#   MODEL:       qwen3.8:latest
#   CONCURRENCY: 4
#   DURATION:    30s
#   GRIM_PORT:   11435
#   OLLAMA_PORT: 11434

set -euo pipefail

MODEL="${1:-qwen3.8:latest}"
CONC="${2:-4}"
DUR="${3:-30}"
GRIM_PORT="${4:-11435}"
OLLAMA_PORT="${5:-11434}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRIM_BIN="$REPO_ROOT/target/release/grim-cli"

if [ ! -f "$GRIM_BIN" ]; then
    GRIM_BIN="$REPO_ROOT/target/debug/grim-cli"
fi

echo "=========================================================="
echo " Serving Benchmark: Grim vs Ollama Parity"
echo " Model:       $MODEL"
echo " Concurrency: $CONC"
echo " Duration:    ${DUR}s"
echo " Grim Port:   $GRIM_PORT"
echo " Ollama Port: $OLLAMA_PORT"
echo "=========================================================="

echo
echo "=== [1/2] Grim (127.0.0.1:$GRIM_PORT) ==="
if curl -sf -m 3 "http://127.0.0.1:$GRIM_PORT/health" >/dev/null 2>&1; then
    if [ -x "$GRIM_BIN" ]; then
        "$GRIM_BIN" bench --mode serve --port "$GRIM_PORT" --concurrency "$CONC" --duration "$DUR"
    else
        echo "grim-cli binary not built. Build with 'cargo build --release -p grim-cli'."
    fi
else
    echo "SKIP: No Grim server detected on port $GRIM_PORT."
    echo "      Start Grim with: cargo run --release -p grim-cli -- serve --port $GRIM_PORT --model <path_or_model>"
fi

echo
echo "=== [2/2] Ollama (127.0.0.1:$OLLAMA_PORT) ==="
if curl -sf -m 3 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    if [ -x "$GRIM_BIN" ]; then
        # Ollama exposes OpenAI-compatible /v1/chat/completions natively on its HTTP port.
        # grim bench --mode serve connects to /v1/chat/completions, providing an apples-to-apples load generator.
        "$GRIM_BIN" bench --mode serve --port "$OLLAMA_PORT" --concurrency "$CONC" --duration "$DUR"
    else
        echo "grim-cli binary not built. Build with 'cargo build --release -p grim-cli'."
    fi
else
    echo "SKIP: No Ollama server detected on port $OLLAMA_PORT."
    echo "      Start Ollama with: 'ollama serve'"
fi

echo
echo "=========================================================="
echo " Parity run complete."
echo "=========================================================="
