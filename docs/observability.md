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

- `grim_tokens_generated_total` (Counter): Total number of tokens generated across all requests.
- `grim_request_duration_seconds` (Histogram): End-to-end request latency distribution.
- `grim_time_to_first_token_seconds` (Histogram): Latency from request receipt to first token emission.
- `grim_inter_token_latency_seconds` (Histogram): Time elapsed between consecutive decoded tokens.
- `grim_running_requests` (Gauge): Current count of requests actively evaluating in the engine.
- `grim_vram_allocated_bytes` (Gauge): Active GPU memory allocated in bytes.

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

