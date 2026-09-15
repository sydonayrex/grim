# Observability Reference

This document describes the structured logging targets, tracing spans, Prometheus metrics, and training telemetry pipelines implemented in Grim.

---

## 1. Tracing & Structured Logging

Grim uses the Rust `tracing` and `tracing-subscriber` ecosystem. Logs are formatted in standard plain text or structured JSON.

### Primary Tracing Targets

| Target | Crate | Description |
|---|---|---|
| `grim_engine` | `grim-engine` | Engine request lifecycle, token generation ticks, and batch admissions. |
| `grim_scheduler` | `grim-scheduler` | Continuous batching queue transitions (waiting -> running -> swapped). |
| `grim_server` | `grim-server` | HTTP request routing, connection handling, and SSE stream chunks. |
| `grim_backend_rocm` | `grim-backend-rocm` | ROCm GPU kernel launches, JIT cache hits/misses, and stream sync. |
| `grim_backend_cuda` | `grim-backend-cuda` | CUDA stream dispatch, PTX compilation, and memory copies. |
| `grim_backend_vulkan` | `grim-backend-vulkan` | Vulkan compute pipeline creation and device probe events. |
| `grim_autograd` | `grim-autograd` | Backward tape node execution and optimizer steps. |

### Log Levels

- `ERROR`: Unrecoverable errors (model load failures, GPU OOM, driver crashes).
- `WARN`: Recoverable degradation (JIT compilation fallbacks, memory oversubscription warnings).
- `INFO`: Lifecycle milestones (model loaded, server bound, training epoch completed).
- `DEBUG`: Per-batch token generation counts, time-to-first-token (TTFT), inter-token latency (ITL).
- `TRACE`: Individual operator dispatches, byte offsets, memory address mappings.

---

## 2. Prometheus Metrics

When running `grim serve`, Prometheus metrics are exposed via the `GET /metrics` endpoint.

> [!NOTE]
> Binding `/metrics` to public IP interfaces requires setting `GRIM_ALLOW_PUBLIC_METRICS=true` or `--allow-public` for security.

### Exposed Metric Keys

#### Throughput & Token Counters
- `grim_prefill_tokens_per_second` (Gauge): Exponential moving average of prompt tokens prefilled per second.
- `grim_decode_tokens_per_second` (Gauge): Exponential moving average of decode tokens generated per second during decode passes.
- `grim_response_tokens_per_second` (Gauge): End-to-end response generation throughput (response tokens produced per second).
- `grim_tokens_generated_total` (Counter): Cumulative decode and response tokens produced across all completed requests.
- `grim_tokens_prefilled_total` (Counter): Cumulative prompt tokens prefilled across all scheduled passes.

#### Latency Metrics
- `grim_time_to_first_token_seconds` (Gauge): Latency from request receipt to first token emission in seconds.
- `grim_time_to_first_token_ms` (Gauge): Latency to produce first token in milliseconds.
- `grim_inter_token_latency_seconds` (Gauge): Inter-token decode latency in seconds.
- `grim_inter_token_latency_ms` (Gauge): Inter-token decode latency in milliseconds.

#### Memory & KV Cache Telemetry
- `grim_kv_cache_used_bytes` (Gauge): Current KV cache memory allocated in bytes.
- `grim_kv_cache_total_bytes` (Gauge): Total KV cache capacity in bytes.
- `grim_kv_cache_blocks_used` (Gauge): Count of KV cache blocks currently allocated.
- `grim_kv_cache_blocks_total` (Gauge): Total count of KV cache blocks available in block pool.
- `grim_block_pool_usage` (Gauge): Ratio of used KV cache blocks to total capacity (`0.0` to `1.0`).
- `grim_vram_used_bytes` (Gauge): Current VRAM memory allocated in bytes.
- `grim_vram_total_bytes` (Gauge): Total available VRAM memory in bytes.
- `grim_gpu_util_pct` (Gauge): Current GPU compute utilization percentage.

#### Scheduler & Queues
- `grim_scheduler_active_requests` (Gauge): Count of currently active requests executing in the scheduler.
- `grim_scheduler_waiting_requests` (Gauge): Count of pending requests waiting in the scheduler queue.
- `grim_scheduler_admitted_requests` (Counter): Cumulative count of admitted requests.
- `grim_preemption_count` (Counter): Cumulative request preemptions.
- `grim_active_sessions` (Gauge): Count of active LoRA adapters and inference sessions.

#### Speculative Decoding
- `grim_speculative_accept_rate_ema` (Gauge): Exponential moving average of speculative draft token acceptance rate.
- `grim_speculative_drafted_tokens_total` (Counter): Total draft tokens proposed.
- `grim_speculative_accepted_tokens_total` (Counter): Total draft tokens accepted.

---

## 3. Training Telemetry (`grim-garage`)

The `grim-garage` web dashboard communicates with `grim-autograd` via telemetry events:

```rust
pub struct TrainingProgressEvent {
    pub step: usize,
    pub total_steps: usize,
    pub epoch: usize,
    pub current_loss: f32,
    pub smoothed_loss: f32,
    pub learning_rate: f32,
    pub tokens_per_second: f32,
    pub vram_used_bytes: u64,
}
```

---

## 4. CLI Monitoring & Inspection

Grim provides CLI utilities for inspecting live engine and server state:

- **`grim status`**: Shows loaded models, compute processor, active backend, and numeric VRAM / KV-cache memory usage.
- **`grim scheduler`**: Displays live scheduler queue statistics (`running`, `waiting`, `admitted`, `paused`) and KV block pool utilization.
- **`GET /metrics`**: Standard Prometheus scrape target for dashboard integration (e.g. Grafana).

