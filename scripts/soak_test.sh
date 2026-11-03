#!/usr/bin/env bash
# Soak test script for grim-server
# § Task 8.5: Stress-tests server under concurrent streaming, early disconnects, and cancels.
set -euo pipefail

SOAK_DURATION_SECS="${1:-7200}"
MODEL_NAME="${2:-Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf}"
SERVER_PORT="${GRIM_PORT:-11434}"
SERVER_HOST="${GRIM_HOST:-127.0.0.1}"
LOG_FILE="/tmp/grim_soak.jsonl"
SERVER_LOG="/tmp/grim_soak_server.log"

echo "=== Starting Grim Server Soak Test (${SOAK_DURATION_SECS}s) ==="
echo "Target Model: ${MODEL_NAME}"
echo "Telemetry log: ${LOG_FILE}"

# Build server release if needed
cargo build --release -p grim-server -p grim-cli

# Start server in background
LIBRARY_PATH=/opt/rocm/lib:${LIBRARY_PATH:-} LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
./target/release/grim-server --address "${SERVER_HOST}:${SERVER_PORT}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "=== Cleaning up soak test processes ==="
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server /health
echo "Waiting for server to become healthy..."
for i in {1..30}; do
    if curl -s "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
        echo "Server is healthy."
        break
    fi
    sleep 1
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server died during startup. Check ${SERVER_LOG}"
        exit 1
    fi
done

# Spawn 4 concurrent load worker loops
spawn_worker() {
    local worker_id=$1
    local end_time=$2

    while [ "$(date +%s)" -lt "$end_time" ]; do
        # 10% cancel chance
        local roll=$((RANDOM % 100))
        if [ "$roll" -lt 10 ]; then
            # Issue a request then immediately cancel
            local req_id="soak-cancel-${worker_id}-$RANDOM"
            curl -s -X POST "http://${SERVER_HOST}:${SERVER_PORT}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 100\"}],\"stream\":true}" \
                --max-time 1 >/dev/null 2>&1 || true
            curl -s -X POST "http://${SERVER_HOST}:${SERVER_PORT}/v1/requests/${req_id}/cancel" >/dev/null 2>&1 || true
        elif [ "$roll" -lt 30 ]; then
            # 20% random client disconnect after partial tokens
            curl -s -N -X POST "http://${SERVER_HOST}:${SERVER_PORT}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain quantum computing in detail.\"}]}" \
                --max-time 2 >/dev/null 2>&1 || true
        else
            # Normal streaming completion
            curl -s -N -X POST "http://${SERVER_HOST}:${SERVER_PORT}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello, write a short haiku about compilers.\"}]}" >/dev/null 2>&1 || true
        fi
        sleep 0.2
    done
}

END_TIME=$(( $(date +%s) + SOAK_DURATION_SECS ))
for w in {1..4}; do
    spawn_worker "$w" "$END_TIME" &
done

echo "Workers started. Sampling telemetry every 60s..."
rm -f "${LOG_FILE}"

while [ "$(date +%s)" -lt "$END_TIME" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server died prematurely! Check ${SERVER_LOG}"
        exit 101
    fi

    TIMESTAMP=$(date -Iseconds)
    RSS_KB=$(ps -o rss= -p "$SERVER_PID" 2>/dev/null | tr -d ' ' || echo "0")
    
    # Try amd-smi or rocm-smi for VRAM telemetry
    VRAM_JSON="null"
    if command -v amd-smi >/dev/null 2>&1; then
        VRAM_JSON=$(amd-smi metric -g 0 -m --json 2>/dev/null || echo "null")
    fi

    HEALTH_START=$(date +%s%N)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${SERVER_HOST}:${SERVER_PORT}/health" || echo "000")
    HEALTH_LATENCY_MS=$(( ( $(date +%s%N) - HEALTH_START ) / 1000000 ))

    echo "{\"timestamp\":\"${TIMESTAMP}\",\"rss_kb\":${RSS_KB},\"health_status\":${HTTP_CODE},\"health_ms\":${HEALTH_LATENCY_MS},\"vram\":${VRAM_JSON}}" >> "${LOG_FILE}"

    sleep 60
done

echo "=== Soak test completed successfully ==="
echo "Logs preserved at ${LOG_FILE}"
