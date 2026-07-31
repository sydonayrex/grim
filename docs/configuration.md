# Configuration Reference

This document describes all environment variables and configuration options used by Grim.

## Environment Variables

### Path Configuration

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_MODELS_DIR` | string | `/var/lib/grim/models` or `~/.grim/models` | `grim-core`, `grim-server` | Directory for cached models |
| `GRIM_CONFIG_DIR` | string | `/etc/grim` or `~/.grim` | `grim-core`, `grim-server` | Directory for config files |
| `GRIM_LOG_DIR` | string | `/var/log/grim` | `grim-core` | Directory for log files |
| `GRIM_PLUGINS_DIR` | string | `/var/lib/grim/plugins` or `~/.grim/plugins` | `grim-plugin` | Directory for plugins |
| `GRIM_HSACO_CACHE_DIR` | string | `$HOME/.cache/hsaco` | `grim-backend-rocm` | ROCm JIT kernel cache directory |

### GPU/Device Selection

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_GPU_TARGET` | string | `gfx900` | `grim-backend-rocm` | Target GPU GCN architecture (e.g., `gfx1100`, `gfx1201`) |
| `GRIM_RUN_GPU_TESTS` | flag | unset | `grim-backend-rocm` | Enable GPU tests when set |
| `GRIM_ROCM_ORDINAL_OVERRIDE` | string | unset | `grim-backend-rocm` | Override GPU ordinal |
| `GRIM_ROCM_DEVICE_NAME` | string | unset | `grim-backend-rocm` | Device name filter |
| `GRIM_ROCM_GCN_NAME` | string | unset | `grim-backend-rocm` | GCN name filter |
| `GRIM_CAPTURE_GRAPH` | flag | unset | `grim-backend-rocm` | Enable hip graph capture |
| `GRIM_ALLOC_POOL_CAP_BYTES` | number | unset | `grim-backend-rocm` | Override allocation pool capacity |
| `GRIM_FORCE_DEVICE` | string | unset | Various | Force specific device |
| `GRIM_CUDA_ORDINAL_OVERRIDE` | string | unset | `grim-backend-cuda` | Override CUDA device ordinal |

### Runtime Control

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_WEIGHT_STREAMING` | flag | unset | `grim-engine` | Enable weight streaming mode |
| `GRIM_AVAILABLE_VRAM` | number | unset | `grim-engine` | Override available VRAM for allocation decisions |

### Path Environment Variables

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `HOME` / `USERPROFILE` | string | N/A | `grim-core` | User home directory |
| `ROCM_PATH` | string | unset | `grim-backend-rocm` | ROCm installation path |
| `HIP_PATH` | string | unset | `grim-backend-rocm` | HIP installation path |

### Test Configuration

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_DATASETS_DIR` | string | unset | Various | Directory for test datasets |
| `GRIM_MAX_CONCURRENT_JOBS` | number | unset | `grim-garage` | Limit concurrent training jobs |

### Build-Time Variables (set by Cargo)

| Variable | Description |
|---|---|
| `CARGO_MANIFEST_DIR` | Directory containing Cargo.toml |
| `OUT_DIR` | Output directory for build artifacts |

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