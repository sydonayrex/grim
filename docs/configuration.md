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
| `GRIM_CUDA_ORDINAL_OVERRIDE` | string | unset | `grim-backend-cuda` | Override CUDA GPU ordinal |
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
| `GRIM_ALLOW_PUBLIC_METRICS` | `1`/`true`/`yes` | unset | `grim-server` | Required opt-in before binding the server/metrics listener to a non-loopback address |
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

## Multi-Process Tensor Parallelism (TP)

Grim uses a **multi-process, one-OS-process-per-rank** model (Design A) for tensor
parallelism. Each rank runs its own `grim` process; the processes rendezvous via
RCCL (ROCm) or NCCL (CUDA) at forward time. There is no in-process fan-out — the
engine does not spawn or orchestrate subprocesses; the operator launches N
processes manually (or via a script).

### Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `GRIM_TP_SIZE` | number | `0` | TP world size. `0`/`1` = single-device. Set to `N` for `N` GPUs. |
| `GRIM_TP_RANK` | number | `0` | This process's shard index (0 to `N-1`). |
| `GRIM_GPUS` | string | unset | Comma-separated GPU ordinals, one per rank (e.g. `0,1`). Rank `i` loads on `GRIM_GPUS[i]`; if absent, rank `i` uses ordinal `i`. |
| `GRIM_PORT` | number | `11434` | Server bind port — each rank process must use a distinct port to avoid bind collision. |

### Quick Start (2-GPU)

```sh
# Rank 0 — GPU 0
GRIM_TP_SIZE=2 GRIM_TP_RANK=0 GRIM_GPUS=0,1 GRIM_PORT=8000 \
  grim serve --backend rocm model.gguf

# Rank 1 — GPU 1 (separate terminal / process)
GRIM_TP_SIZE=2 GRIM_TP_RANK=1 GRIM_GPUS=0,1 GRIM_PORT=8001 \
  grim serve --backend rocm model.gguf
```

Both ranks must load the **same model files**. Each process loads only its own
weight shard (`ColumnParallelLinear`/`RowParallelLinear` pre-sharded at load);
RCCL's `ncclAllReduce` in `RowParallelLinear::forward` synchronizes partial
outputs across ranks.

### Supported Architectures

| Architecture | `load_tp` | Notes |
|---|---|---|
| Llama | Full | Column-parallel Q/K/V/Gate/Up, row-parallel O/Down. KV-head sharding via `plan_kv_head_sharding`. |
| GPT-2 | Stub (error) | Fused QKV needs head-axis reshape; `load_tp` returns `Unsupported`. |
| Gemma / DeepSeek / LFM2 | Stub (error) | `forward` uses plain `Linear` with no all-reduce hook — load-side sharding would be silently wrong. |
| T5 | Full | Encoder + decoder attention sharded. |
| Mamba / RWKV | Stub (error) | SSM/RWKV recurrent path has no all-reduce semantics. |
| BERT / ViT / Whisper | Stub (error) | Encoder-only / encoder-decoder serving path is `CausalLm`; TP is out of scope. |

### Validation

- `GRIM_TP_RANK >= GRIM_TP_SIZE` → hard error at `Engine::new` (issue #6 fix).
- `num_heads` must be divisible by `GRIM_TP_SIZE`.
- KV-head sharding: either `num_kv_heads % world_size == 0` (shard) or
  `world_size % num_kv_heads == 0` (replicate). Otherwise, error.
- Fewer GPUs visible than `GRIM_TP_SIZE` → error from `RocmDevice::probe`.
- If only one of N ranks starts, `ncclAllReduce` at first forward will block —
  this is expected (same as `torchrun` with a missing rank).

### Verification

The CPU-only parity test `test_llama_block_tp_parity_concat_shards_equals_full`
runs in CI without GPUs. It loads a tiny Llama block with `world_size=1` and with
`world_size=2` (ranks 0+1) using a fake provider, then asserts that
concatenating the two shards' weight matrices reproduces the full weight
exactly — catching overlap, gap, or off-by-one sharding bugs.

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
