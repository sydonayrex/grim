# Configuration Reference

This document provides a consolidated reference for all environment variables, configuration file options, and precedence rules used across the Grim workspace.

## Precedence Order

When configuration values are supplied across multiple sources, settings resolve in the following order:

1. **Explicit CLI Flags** (e.g. `--port 8080`, `--device rocm`, `--seed 42`)
2. **Environment Variables** (e.g. `GRIM_PORT`, `GRIM_BACKEND`) — override the file
3. **Configuration File** (`grim.toml`) — see [RuntimeEnv & grim.toml](#runtimeenv--grimtoml-wi-x16)
4. **Hardcoded Internal Defaults**

> WI-X16 note: `grim-core::env_config::RuntimeEnv` reads `grim.toml` **first**
> and lets `GRIM_*` environment variables override per key. Unknown keys in
> `grim.toml` produce a one-time warning; they are never silently dropped.
> `grim doctor` prints the effective value and its source (`file` / `env` /
> `default`) for every core key.

### RuntimeEnv & grim.toml (WI-X16)

Core runtime knobs are centralized in `grim-core::env_config::RuntimeEnv`.
Lookup order per key: `GRIM_CONFIG` path → `./grim.toml` → `~/.grim/grim.toml`;
a template ships as `grim.toml.example`. Keys:

| Key | Type | Default | Env override |
|---|---|---|---|
| `host` | String | `127.0.0.1` | `GRIM_HOST` |
| `port` | u16 | `11434` | `GRIM_PORT` |
| `context` | usize | auto (model metadata) | `GRIM_CONTEXT` |
| `backend` | String | `auto` | `GRIM_BACKEND` |
| `gpus` | list | all visible | `GRIM_GPUS` |
| `tp_size` | usize | `0` (off) | `GRIM_TP_SIZE` |
| `parallel` | bool | model-dependent | `GRIM_PARALLEL` |
| `mem_budget_mib` | usize | unlimited | `GRIM_MEM_BUDGET_MIB` |
| `kernel_timeout` | u64 | `300` s | `GRIM_KERNEL_TIMEOUT` |

Behavior-specific escapes that remain environment-only (debug/test gates such
as `GRIM_RUN_GPU_TESTS`, `GRIM_ATTENTION_AUTOTUNE`, backend feature probes)
are intentionally not part of the file contract.

---

## Environment Variables

### Path & Directory Configuration

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_MODELS_DIR` | String | `~/.grim/models` or `/var/lib/grim/models` | `grim-core`, `grim-cli`, `grim-garage` | Local storage path for downloaded and converted models. |
| `GRIM_CONFIG_DIR` | String | `~/.grim` or `/etc/grim` | `grim-core`, `grim-cli` | Directory containing `grim.toml` configuration files. |
| `GRIM_LOG_DIR` | String | `/var/log/grim` | `grim-core` | Destination directory for system logs. |
| `GRIM_PLUGINS_DIR` | String | `~/.grim/plugins` or `/var/lib/grim/plugins` | `grim-plugin` | Directory for native and WASM plugin binaries. |
| `GRIM_HSACO_CACHE_DIR` | String | `~/.cache/hsaco` | `grim-backend-rocm` | Disk cache directory for JIT-compiled AMD HSACO kernel binaries. |

### Backend & Hardware Selection

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_BACKEND` | String | `auto` | `grim-core`, `grim-cli` | Target backend: `rocm`, `cuda`, `vulkan`, `metal`, `cpu`, or `auto`. |
| `GRIM_FORCE_DEVICE` | String | Unset | `grim-engine`, `grim-cli` | Forces a specific device ordinal (e.g. `rocm:0`). |
| `GRIM_GPU_TARGET` | String | `gfx1100` | `grim-backend-rocm` | Target AMD GCN architecture (e.g. `gfx1030`, `gfx1100`, `gfx1200`, `gfx90a`, `gfx942`). |
| `ROCM_PATH` | String | `/opt/rocm` | `grim-backend-rocm` | ROCm installation directory for resolving HIP/rocBLAS libraries. |
| `HIP_PATH` | String | `/opt/rocm/hip` | `grim-backend-rocm` | HIP toolkit installation path. |
| `HIP_VISIBLE_DEVICES` | String | Unset | `grim-backend-rocm` | Comma-separated list of visible AMD GPU ordinals. |
| `CUDA_PATH` | String | `/usr/local/cuda` | `grim-backend-cuda` | CUDA installation root for resolving cuBLAS and runtime libraries. |
| `NVCC` | String | `nvcc` | `grim-backend-cuda` | Path to the NVIDIA CUDA compiler binary for JIT PTX builds. |
| `CUDA_VISIBLE_DEVICES` | String | Unset | `grim-backend-cuda` | Comma-separated list of visible NVIDIA GPU ordinals. |
| `VK_ICD_FILENAMES` | String | Unset | `grim-backend-vulkan` | Path to specific Vulkan Installable Client Driver JSON file (e.g. `radeon_icd.json`). |
| `VK_LOADER_LAYERS_DISABLE` | String | `~all~` | `grim-backend-vulkan` | Disables implicit Vulkan layers that can stall headless execution. |

### Memory & Execution Controls

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_ROCM_MANAGED_ALLOCATIONS` | String (`always`/`auto`) | Unset | `grim-backend-rocm` | Enables HIP managed memory allocations when physical VRAM is constrained. |
| `GRIM_ROCM_VRAM_BUDGET_BYTES` | Number (Bytes) | 90% of total VRAM | `grim-backend-rocm` | Soft VRAM budget cap before falling back to system memory. |
| `GRIM_MEM_BUDGET_MIB` | Number (MiB) | Unset | `grim-core`, `grim-engine` | Maximum GPU memory budget allocation limit in MiB. |
| `GRIM_KERNEL_TIMEOUT` | Number (Seconds) | `300` | `grim-core` | GPU kernel dispatch execution timeout before aborting. |
| `GRIM_CONTEXT` | Number (Tokens) | Model Default | `grim-core`, `grim-engine` | Overrides maximum context length and KV cache buffer allocation. |

### Serving & Network Controls

| Variable | Type | Default | Used By | Description |
|---|---|---|---|---|
| `GRIM_HOST` | String | `127.0.0.1` | `grim-server`, `grim-cli` | Network IP address to bind HTTP API server. |
| `GRIM_PORT` | Number | `11434` | `grim-server`, `grim-cli` | TCP port number for HTTP server. |
| `GRIM_ALLOW_PUBLIC_METRICS` | Flag (`1`/`true`) | Unset | `grim-server` | Explicit opt-in required to expose Prometheus `/metrics` on public IP interfaces. |
| `GRIM_TP_SIZE` | Number | `0` | `grim-core`, `grim-engine` | Tensor-parallel world size across identical GPUs. |
| `GRIM_TP_RANK` | Number | `0` | `grim-core`, `grim-engine` | Tensor-parallel rank for distributed rank execution. |

---

## Configuration File (`grim.toml`)

Grim automatically searches for a `grim.toml` file in:
1. Current working directory (`./grim.toml`)
2. User directory (`~/.grim/grim.toml`)
3. Global configuration (`/etc/grim/grim.toml`)

### Example `grim.toml`

```toml
default_model = "models/llama-3.2-1b.gguf"

[server]
host = "127.0.0.1"
port = 11434
max_batched_tokens = 4096
allow_public_metrics = false

[scheduler]
target_ttft_ms = 2000
target_itl_ms = 100
max_running_requests = 128

[memory]
gpu_memory_fraction = 0.90
block_size = 16
prefix_caching = true

[training]
seed = 42
lr = 0.0001
weight_decay = 0.01
```
