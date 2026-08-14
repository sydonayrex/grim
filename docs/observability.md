# Observability

## Tracing and Logging

The system relies on the `tracing` ecosystem for application-level observability.

### Targets
*   `grim_engine`: Core model execution, context allocation, and scheduling.
*   `grim_backend_*`: Hardware-specific dispatch logic.
*   `grim_cli`: Command-line setup and initialization paths.

### Levels
*   `ERROR`: Unrecoverable faults (e.g., failed model load, GPU OOM).
*   `WARN`: Performance degradation or fallback mechanisms triggered.
*   `INFO`: High-level workflow stages (e.g., model load completion, server binding).
*   `DEBUG`: Kernel execution times, tensor shapes, token generation intervals.
*   `TRACE`: Individual operator dispatches, byte-level I/O.

## Metrics

Prometheus-compatible metrics are emitted to track throughput and system utilization.

*   `grim_tokens_per_second` (Gauge): Token generation throughput.
*   `grim_inference_latency_ms` (Histogram): Milliseconds spent on a single forward pass.
*   `grim_memory_allocated_bytes` (Gauge): Current tracked memory usage across devices.

Source definitions map to specific metric layers registered at server initialization.
