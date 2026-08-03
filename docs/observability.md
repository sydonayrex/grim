# Observability

This document describes the logging, metrics, and telemetry instrumentation actually present in the Grim source code.

## Logging

### Log Targets

Logging is implemented via `eprintln!` (stderr) across most crates. The `RUST_LOG` environment variable controls log levels for crates that use the `log` crate facade. The `tracing` crate is used in `grim-garage`.

### Log Sources

- **`grim-server/src/lib.rs`**: Emit progress, model-loading, and error status via `eprintln!` with `[grim-server]` and `[Server]` prefixes.
- **`grim-engine/src/lib.rs`**: Step-level and scheduler diagnostics via `eprintln!`.
- **`grim-nn/src/varbuilder.rs`**: Model loading and weight materialization progress.
- **`grim-nn/src/modules.rs`**: Module initialization messages.
- **`grim-backend-rocm/src/device/cubecl.rs`**: Backend selection diagnostics, including RDNA2 warmup-retry messages.
- **`grim-garage/src/rocm.rs`**: Uses `tracing::warn` for device probing warnings.

### Enabling Verbose Logging

```bash
RUST_LOG=debug grim serve
```

For GPU debugging:

```bash
# ROCm
ROCM_LOG_LEVEL=debug grim serve
```

## Metrics Endpoints

### `GET /metrics`

Telemetry metrics endpoint (§8). Returns JSON:

```json
{
    "engine_state": "healthy",
    "active_sessions": <usize>,
    "block_pool_usage": 0.05,
    "preemption_count": 0,
    "hardware": {
        "rocm_gpu_count": <usize>,
        "xack_enabled": <bool>
    }
}
```

Source: `grim-server/src/lib.rs` lines 1396–1419.

### `GET /status`

Status endpoint with detailed resource telemetry. Returns JSON with:

- `status`: health string (`"healthy"`)
- `processor`: `"CPU"` or GPU name
- `default_model`: current default model name
- `system_ram_used_gb`, `system_ram_total_gb`
- `vram_used_gb`, `vram_total_gb`
- `gpu_util_pct`: VRAM utilization percentage
- `loaded_models`: array of model objects with name, params, VRAM, TPS, context limit
- `kv_cache`: used_bytes, total_bytes, blocks_used, blocks_total
- `context_limit`: typically 8192

Source: `grim-server/src/lib.rs` lines 1602–1729.

## Telemetry Counters

### `LIVE_CLEANUP_GUARDS`

A `pub static LIVE_CLEANUP_GUARDS: AtomicUsize` in `grim-server/src/lib.rs` (line 134) tracks the number of live `RequestCleanupGuard` instances. It is incremented on guard creation and decremented on `Drop`. Used by tests to assert exactly-once cleanup after request completion or cancellation.

### Engine Counters

- `engine.adapter_count()` — number of active LoRA adapter sessions.
- `engine.kv_cache_telemetry()` — returns `(used_bytes, total_bytes, blocks_used, blocks_total)`.
- `engine.tokens_per_sec()` — estimated generation throughput.

Source: `grim-engine/src/lib.rs`.

## Debug Mode

Enable verbose output:

```bash
RUST_LOG=debug grim serve
```
