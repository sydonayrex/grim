# Configuration Reference

This document describes all environment variables and configuration options used by Grim.

## Environment Variables

### Path Configuration

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_MODELS_DIR` | string | `/var/lib/grim/models` or `~/.grim/models` | `grim-core`, `grim-garage` | Directory for cached models |
| `GRIM_CONFIG_DIR` | string | `/etc/grim` or `~/.grim` | `grim-core` | Directory for config files |
| `GRIM_LOG_DIR` | string | `/var/log/grim` | `grim-core` | Directory for log files |
| `GRIM_PLUGINS_DIR` | string | `/var/lib/grim/plugins` or `~/.grim/plugins` | `grim-plugin` | Directory for plugins |
| `GRIM_HSACO_CACHE_DIR` | string | `$HOME/.cache/hsaco` | `grim-backend-rocm` | ROCm JIT kernel cache directory |

### Backend Selection

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_BACKEND` | string | `auto` | `grim-core`, `grim-cli` | Backend selection: `rocm`, `cuda`, `vulkan`, `metal`, `cpu`, `auto` |
| `GRIM_FORCE_DEVICE` | string | unset | `grim-engine`, `grim-cli` | Force a specific device (overrides backend selection) |
| `GRIM_GPU_TARGET` | string | `gfx900` | `grim-backend-rocm` | Target GPU GCN architecture (e.g., `gfx1100`, `gfx1201`) |
| `GRIM_RUN_GPU_TESTS` | flag | unset | `grim-backend-rocm` | Enable GPU tests when set |
| `GRIM_ROCM_ORDINAL_OVERRIDE` | string | unset | `grim-backend-rocm` | Override GPU ordinal |
| `GRIM_ROCM_DEVICE_NAME` | string | unset | `grim-backend-rocm` | Device name filter |
| `GRIM_ROCM_GCN_NAME` | string | unset | `grim-backend-rocm` | GCN name filter |
| `GRIM_CAPTURE_GRAPH` | flag | unset | `grim-backend-rocm` | Enable hip graph capture |
| `GRIM_ALLOC_POOL_CAP_BYTES` | number | unset | `grim-backend-rocm` | Override allocation pool capacity |
| `ROCM_PATH` | string | unset | `grim-backend-rocm` | ROCm installation path |
| `HIP_PATH` | string | unset | `grim-backend-rocm` | HIP installation path |

### ROCm Memory Management

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_ROCM_MANAGED_ALLOCATIONS` | `always`/`auto` | unset | `grim-backend-rocm` | Use HIP managed memory for global allocations; `auto` spills when free VRAM or the configured budget is insufficient |
| `GRIM_ROCM_MANAGED_WEIGHTS` | `always`/`auto` | unset | `grim-nn` | Use HIP managed memory for F32 model-weight materialization only |
| `GRIM_ROCM_VRAM_BUDGET_BYTES` | number | 90% of reported VRAM in `auto` mode | `grim-backend-rocm`, `grim-nn` | Soft per-device VRAM ceiling in bytes before new allocations use host-backed managed memory |
| `GRIM_ROCM_ORDINAL_OVERRIDE` | string | unset | `grim-backend-rocm` | Override GPU ordinal selection |

### Runtime Control

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_HOST` | string | `127.0.0.1` | `grim-core` | Server bind host address |
| `GRIM_PORT` | number | `11434` | `grim-core` | Server bind port |
| `GRIM_CONTEXT` | number | unset | `grim-core` | Override model context window (KV cache length) — takes precedence over GGUF `max_position_embeddings` |
| `GRIM_GPUS` | string | unset (all) | `grim-core`, `grim-engine` | Comma-separated ordinal list of GPUs to use (e.g., `0,1`) |
| `GRIM_TP_SIZE` | number | `0` (single-device) | `grim-core`, `grim-engine`, `grim-nn` | Tensor-parallel world size. >1 requires matching GPUs and RCCL/NCCL |
| `GRIM_TP_RANK` | number | `0` | `grim-core`, `grim-nn`, `grim-models` | Tensor-parallel rank for sharded execution |
| `GRIM_PARALLEL` | `yes`/`no` | unset | `grim-core` | Advisory multi-GPU parallelism hint; no-op on single-GPU/CPU |
| `GRIM_MEM_BUDGET_MIB` | number | unset | `grim-core` | Per-device GPU memory budget cap in MiB |
| `GRIM_KERNEL_TIMEOUT` | number (seconds) | `300` | `grim-core` | Soft GPU kernel timeout before host aborts a launch |
| `GRIM_WEIGHT_STREAMING` | flag | unset | `grim-engine` | Enable weight streaming mode |
| `GRIM_AVAILABLE_VRAM` | number (bytes) | unset | `grim-engine` | Override available VRAM for allocation decisions |

### Path Environment Variables

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `HOME` / `USERPROFILE` | string | N/A | `grim-core` | User home directory |
| `CARGO_MANIFEST_DIR` | string | set by Cargo | Build | Directory containing Cargo.toml |
| `OUT_DIR` | string | set by Cargo | Build | Output directory for build artifacts |

### Test Configuration

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_DATASETS_DIR` | string | unset | `grim-garage` | Directory for test/training datasets |
| `GRIM_MAX_CONCURRENT_JOBS` | number | unset | `grim-garage` | Limit concurrent training jobs |

## Configuration File

Grim looks for `grim.toml` in the following locations:
1. Current working directory (`./grim.toml`)
2. Config directory (`/etc/grim/grim.toml`)
3. User config (`~/.grim/grim.toml`)

### Configuration File Format

```toml
default_model = "llama3:latest"

[server]
address = "127.0.0.1:11434"
max_batched_tokens = 4096

[scheduler]
target_ttft_ms = 2000
target_itl_ms = 100
```

### Supported Keys

| Key | Type | Description |
|---|---|---|
| `default_model` | string | Default model name for the server |
| `server.address` | string | HTTP server bind address |
| `server.max_batched_tokens` | integer | Maximum tokens per batch |
| `scheduler.target_ttft_ms` | integer | Target time-to-first-token in ms |
| `scheduler.target_itl_ms` | integer | Target inter-token latency in ms |

## Precedence Rules

Configuration is applied in the following order (later sources override earlier ones):

1. Default values in code
2. `grim.toml` (if found)
3. Environment variables
4. CLI flags

## Grim Models Directory Structure

```
$GRIM_MODELS_DIR/
├── models/           # Model cache directory
│   ├── llama3/       # Model name
│   │   ├── *.gguf     # GGUF checkpoint
│   │   ├── *.grim     # ROCm-optimized format
│   │   └── *.safetensors
├── plugins/          # Model architecture plugins
│   └── *.grimplugin
└── datasets/         # Training datasets
```
