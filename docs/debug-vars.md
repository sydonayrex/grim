# Debug & Tuning Environment Variables

Full reference for every `GRIM_*` environment variable actually read
somewhere under `crates/` (124 variables, inventoried 2026-09-24).
`docs/configuration.md` covers the 27 core runtime knobs plus
`grim.toml` precedence; this document is the complete debug/tuning
surface: name, polarity (which values turn it on/off), default, and
owning crate(s).

Provenance: `env_vars.json` (read-site inventory grouped by crate),
`env_sites.txt` (exact `std::env::var` call sites), `env_contexts.txt`
(default/polarity source contexts). Polarity strings below quote the
match arms at those call sites.

How to read the tables:

- **Polarity** describes the exact check in source, e.g. `== "1"`
  (only the literal `1` enables), `!= "0"` (anything except `0`
  enables, so the flag is on by default), `set` (mere presence
  enables, even empty), `1/true/on` (case-insensitive true spellings).
- **Default** is the behavior when the variable is unset or
  unparseable (unparseable almost always falls back to the default).
- **Owner(s)** lists every crate with a read site; the variable is
  grouped under the first (primary) owner.

The no-dead-docs test `crates/grim-cli/tests/debug_vars_docs.rs`
fails if any variable documented here stops being read in `crates/`.

## grim-backend-rocm

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_ALLOC_NO_POOL` | `set` disables the caching pool (every free is a synchronized real release) | pool enabled | grim-backend-rocm |
| `GRIM_ALLOC_POOL_CAP_BYTES` | bytes integer | `total_VRAM / 6` | grim-backend-rocm |
| `GRIM_ALLOC_TRACE` | `set` enables `[ctx-trace]` / `[jit-trace]` / `[launch-done]` / `[alloc-trace]` lines | off | grim-backend-rocm |
| `GRIM_ATTENTION_AUTOTUNE` | `1` / `true` enables flash-decode tile autotuning | off | grim-backend-rocm |
| `GRIM_AUTOTUNE_CACHE_DIR` | path; tuner JSON persisted per arch | unset (memory only) | grim-backend-rocm |
| `GRIM_CAPTURE_GRAPH` | backend device flag: `1` / `true` enables; engine scheduler: on unless `0` / `false` / `off` | backend off; scheduler on | grim-backend-rocm, grim-engine |
| `GRIM_DISABLE_CALIBRATION` | `set` skips the capability micro-benchmark | calibration on | grim-backend-rocm |
| `GRIM_DOT_GEMV` | on unless `0` / `false` / `off` | on | grim-backend-rocm, grim-models |
| `GRIM_DOT_GEMV_LEGACY` | `1` / `true` / `on` enables the legacy path | off | grim-backend-rocm |
| `GRIM_F16_KV` | any value except `0` / `false` enables (LFM2 arenas only) | off | grim-backend-rocm |
| `GRIM_F32_GEMV` | on unless `0` / `false` / `off` | on | grim-backend-rocm |
| `GRIM_FLASH_DECODE_MIN_KV` | integer KV length threshold | `256` (RDNA3/4) / `512` (other) | grim-backend-rocm |
| `GRIM_FP16_ACT` | `== "1"` enables on-device FP32-to-FP16 pre-quantize path | off | grim-backend-rocm |
| `GRIM_FP16_VERIFY` | `== "1"` enables quantize/dequantize round-trip verify | off | grim-backend-rocm |
| `GRIM_GLA_FUSED` | any value except `0` / `false` enables the fused state-update/output path | on | grim-backend-rocm |
| `GRIM_GLA_STATE_FP8` | any value except `0` / `false` enables FP8 GLA state | off | grim-backend-rocm |
| `GRIM_GPUS` | comma-separated ordinals, one per TP rank | all visible | grim-backend-rocm, grim-cli, grim-core, grim-engine |
| `GRIM_GPU_TARGET` | arch string (`gfx1030`, `gfx1100`, `gfx1200`, `gfx90a`, `gfx942`, ...) | `gfx900` | grim-backend-rocm |
| `GRIM_GPU_TEST` | `== "1"` enables GPU test bodies (aliases `GRIM_RUN_GPU_TESTS`, `GRIM_RUN_GPU_TEST`) | off | grim-backend-rocm, grim-engine, grim-nn |
| `GRIM_GRAPH_MAX_RETRIES` | integer | `3` | grim-backend-rocm |
| `GRIM_GRAPH_RETRY_INTERVAL` | integer eager steps between recapture attempts | `32` | grim-backend-rocm |
| `GRIM_HIP_TRACE` | `set` enables `[hipMemcpy-trace]` / `[hipDeviceSync-trace]` backtraces | off | grim-backend-rocm |
| `GRIM_HSACO_CACHE_DIR` | path to JIT HSACO disk cache | temp `grim_hsaco_cache` | grim-backend-rocm, grim-cli |
| `GRIM_JIT_CACHE_DIR` | path to compiled HSA code-object cache | `~/.cache/grim/rocm_code_objects` | grim-backend-rocm |
| `GRIM_KV_QUANT_FORMAT` | `q4khalf` / `q4k` / `q8_0` / `legacy` (also `1` / `2` / `3`) | derived from `quant_bits` | grim-backend-rocm |
| `GRIM_MXFP4_FUSED_GEMM` | `1` / `true` forces on; `0` / `false` / `off` / `no` forces off | arch-confirmed (RDNA4 gfx12x, UDNA gfx13x, CDNA4 gfx95x) | grim-backend-rocm |
| `GRIM_Q4K_REPRO` | `set` (`=1`) enables Q4K repro diag tests | off | grim-backend-rocm |
| `GRIM_Q4K_TILED` | `1` / `true` / `on` enables the tiled Q4K path | off | grim-backend-rocm |
| `GRIM_RCCL_TOPOLOGY` | `symmetric` / `asymmetric` (test assertion selector) | unset (no assertion) | grim-backend-rocm |
| `GRIM_ROCM_MANAGED_ALLOCATIONS` | `1` / `true` / `always` forces managed memory; `auto` uses VRAM checks | unset (off) | grim-backend-rocm |
| `GRIM_ROCM_ORDINAL_OVERRIDE` | ordinal integer (probe + enumerate) | unset (HIP enumerate) | grim-backend-rocm |
| `GRIM_ROCM_VRAM_BUDGET_BYTES` | bytes integer | 90% of total VRAM | grim-backend-rocm, grim-nn |
| `GRIM_ROWS4_MIN_N` | integer N gate for the 4-col-per-thread Q8_0 path; `0` disables | `0` (disabled) | grim-backend-rocm |
| `GRIM_RUN_GPU_TEST` | `== "1"` enables GPU test bodies | off | grim-backend-rocm, grim-nn |
| `GRIM_RUN_GPU_TESTS` | `== "1"` (or merely set, in older tests) enables GPU test bodies | off | grim-backend-rocm, grim-backend-tests, grim-backend-vulkan |
| `GRIM_TP_RANK` | integer rank | `0` | grim-backend-rocm, grim-cli, grim-nn |
| `GRIM_TP_SIZE` | integer world size; `> 1` enables tensor-parallel | `0` (off) | grim-backend-rocm, grim-cli, grim-core, grim-engine, grim-nn |
| `GRIM_TRACE_FUSED_QKV` | `set` enables fused-QKV trace lines | off | grim-backend-rocm |
| `GRIM_WAVEFRONT_SIZE` | `32` / `64` | arch-detected | grim-backend-rocm |
| `GRIM_WMM_MAX_M` | integer | `16384` | grim-backend-rocm |

## grim-engine

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_AVAILABLE_VRAM` | bytes integer override for speculative registration | unset (device probe) | grim-engine |
| `GRIM_GRAVE` | `1` / `true` / `TRUE` selects GDL attention mode | off (softmax) | grim-engine |
| `GRIM_GRAVE_SIDECAR` | path to GRAVE gate-triple sidecar (fail-closed: set-but-unreadable is an error) | unset | grim-engine |
| `GRIM_HOST_ALLOWANCE_GB` | integer GB | `8` | grim-engine |
| `GRIM_KV_KEEP_TAIL_BLOCKS` | integer blocks kept hot on watermark demotion | `2` | grim-engine |
| `GRIM_KV_QUANT` | mode string (`int8`, ...) | unset (off) | grim-engine |
| `GRIM_KV_SPILL` | `1` / `true` / `on` attaches the shared spill manager | off | grim-engine |
| `GRIM_KV_SPILL_DIR` | scratch path for spilled KV blocks | temp `grim_kv_spill_<pid>` | grim-engine |
| `GRIM_KV_WATERMARK_BLOCKS` | integer blocks; demotion only beyond the watermark | unset (no watermark demotion) | grim-engine |
| `GRIM_LFM2_ATTENTION_MODE` | `gdl` / `grave` selects GDL mode | softmax | grim-engine |
| `GRIM_LFM2_ATTENTION_MODE_LAYERS` | per-layer mode list | unset (uniform mode) | grim-engine |
| `GRIM_LFM2_MXFP4_QKV` | on unless `0` / `false` / `off` | on | grim-engine |
| `GRIM_MEMORY_RESERVE_GB` | integer GB | `2` | grim-engine |
| `GRIM_RADIX` | on unless `0` / `false` / `off` | on | grim-engine, grim-server |
| `GRIM_SCYTHE_INFERENCE` | `1` / `true` / `on` enables multi-GPU capability profiling | off | grim-engine |
| `GRIM_SCYTHE_RING_RESIDENT` | `== "1"` enables the resident-ring experiment (test) | off | grim-engine |
| `GRIM_SCYTHE_SPREAD` | on unless `== "0"` | on | grim-engine |
| `GRIM_SESSION_GRAPH_SLOTS` | integer `>= 1` | `2` | grim-engine |
| `GRIM_SESSION_PIN_SECS` | integer seconds of session block pinning | `300` | grim-engine |
| `GRIM_STEP_TRACE` | `set` enables per-step timing | off | grim-engine |
| `GRIM_TEST_FREE_DEVICE_BYTES` | bytes integer mock for free-device-memory probes | unset (real probe) | grim-engine |
| `GRIM_WEIGHT_STREAMING` | `set` marks weight streaming active | off | grim-engine |

## grim-cli

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_BACKEND` | backend string (`rocm`, `cuda`, `vulkan`, `metal`, `cpu`, `auto`); `GRIM_FORCE_DEVICE` is a legacy alias | `auto` | grim-cli, grim-core, grim-engine, grim-server |
| `GRIM_CPU_SAMPLER` | CLI: `set` forces the CPU sampler; server: `1` / `true` forces it | off (GPU sampler eligible) | grim-cli, grim-server |
| `GRIM_CTX_TOKENS` | integer prompt-pad target for the long-context harness | unset | grim-cli |
| `GRIM_DEBUG_LOGITS_PATH` | path; final-step logits dumped as raw f32 bytes | unset | grim-cli |
| `GRIM_DEFAULT_MODEL` | model name string | `default` | grim-cli |
| `GRIM_DOCTOR_METRICS_URL` | URL for the `/metrics` GPU-verdict probe | `http://127.0.0.1:11434/metrics` | grim-cli |
| `GRIM_EVAL_MAX_WINDOWS` | integer deterministic prefix cap (recorded in metrics JSON) | unset (uncapped) | grim-cli |
| `GRIM_FORCE_DEVICE` | legacy alias of `GRIM_BACKEND` (`rocm:0`, ...) | unset | grim-cli, grim-engine |
| `GRIM_GRAVE_GATES` | per-layer `d,e,w` triples separated by `;` | unset (block defaults) | grim-cli, grim-models |
| `GRIM_GRAVE_TC_INIT` | `1` / `true` seeds distill gates from depth-scaled TC gates | off | grim-cli |
| `GRIM_KV_CACHE_LEN` | integer KV positions | `4096` | grim-cli, grim-models |
| `GRIM_MOUSE` | `== "1"` enables TUI mouse capture (disables terminal text selection) | off | grim-cli |
| `GRIM_SCORE_CHUNK_LEN` | integer `64..=32768` | `1024` | grim-cli |
| `GRIM_SCORE_FILE` | path to text scored for aggregate perplexity | unset | grim-cli |
| `GRIM_TUI_LOG` | path for the stderr-redirected TUI log | `/tmp/grim-tui.log` | grim-cli |

## grim-core

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_CONFIG` | explicit `grim.toml` path | unset (`./grim.toml` search) | grim-core |
| `GRIM_CONFIG_DIR` | config directory override | `~/.grim` or `/etc/grim` | grim-core |
| `GRIM_CONTEXT` | integer context tokens | model default | grim-core |
| `GRIM_FALLBACK_LOG` | `1` / `json` / `true` emits one machine-readable JSON line per fallback to stderr | off | grim-core |
| `GRIM_HOST` | bind IP | `127.0.0.1` | grim-cli, grim-core |
| `GRIM_INSECURE_TLS` | `1` / `true`; debug builds only (release always verifies) | off | grim-core |
| `GRIM_KERNEL_TIMEOUT` | integer seconds | `300` | grim-core |
| `GRIM_LOG_DIR` | log directory | `/var/log/grim` | grim-core |
| `GRIM_MEM_BUDGET_MIB` | integer MiB GPU budget | unlimited | grim-core |
| `GRIM_MODELS_DIR` | model storage path | `~/.grim/models` (see `grim_models_dir`) | grim-core, grim-garage |
| `GRIM_PARALLEL` | `yes` / `true` / `1` enables; `no` / `false` / `0` disables | model-dependent | grim-core |
| `GRIM_PLUGINS_DIR` | plugin directory | `~/.grim/plugins` or `/var/lib/grim/plugins` | grim-core, grim-engine |
| `GRIM_PORT` | integer port | `11434` | grim-cli, grim-core |

## grim-server

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_ALLOW_PUBLIC_METRICS` | `1` / `true` / `yes` exposes `/metrics` on public interfaces | off (loopback only) | grim-server |
| `GRIM_API_KEY` | single API key string | unset | grim-server |
| `GRIM_API_KEYS` | comma-separated API key list | unset | grim-server |
| `GRIM_COMPRESS_PROMPT` | `== "1"` enables extractive prompt compression | off | grim-server |
| `GRIM_CONFIG_PATH` | config-file path for model defaults and TLS | `grim.toml` search | grim-server |
| `GRIM_DEBUG_PROMPT` | `== "1"` logs prompt tokens and text to stderr | off | grim-server |
| `GRIM_SAMPLE_SEED` | integer base seed | `0x9E3779B97F4A7C15` | grim-server |
| `GRIM_SAMPLE_TEMPERATURE` | float | `0.7` | grim-server |
| `GRIM_SAMPLE_TOP_K` | integer | `0` | grim-server |
| `GRIM_SAMPLE_TOP_P` | float | `1.0` | grim-server |
| `GRIM_SPEC` | `off` / `0` / `false` / `disable` / `disabled` disables speculative decoding | on | grim-server |
| `GRIM_TOKEN_PACING_MS` | integer milliseconds of fixed per-token sleep | `0` | grim-server |

## grim-models

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_DBRX_REAL_ROUTING` | `== "1"` enables real softmax top-k routing (ROCm only) | off (fixed ensemble) | grim-models |
| `GRIM_DECODE_GRAPH` | on unless `0` / `false` / `off` (case variants) | on | grim-models |
| `GRIM_DELTANET_GDL` | `1` / `true` / `on` opts the DeltaNet base into GDL | off | grim-models |
| `GRIM_FUSED_FFN` | on unless `== "0"` (ROCm Q8_0 gate/up fusion) | on | grim-models |
| `GRIM_FUSED_QKV` | on unless `== "0"` | on | grim-models |
| `GRIM_LFM2_F32_HEAD` | `== "1"` forces the f32 head table | off (packed Q8_0 path) | grim-models |
| `GRIM_MOE_CHARON` | on unless `== "0"` (per-expert loop reference when off) | on | grim-models |
| `GRIM_MXFP4_PARITY_DEBUG` | `set` enables the plain-F32 harness-isolation leg | off | grim-models |
| `GRIM_RMSNORM_ROPE` | on unless `== "0"` | on | grim-models |
| `GRIM_ROPE_DEV_BASE` | on unless `== "0"` (device-side RoPE base table) | on | grim-models |

## grim-nn

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_EP_RANK` | integer expert-parallel rank | `0` | grim-nn |
| `GRIM_EP_SIZE` | integer world size; `> 1` enables expert-parallel | `1` (off) | grim-nn |
| `GRIM_ROCM_MANAGED_WEIGHTS` | `1` / `true` / `always` forces managed weights; `auto` uses VRAM checks | unset (device memory) | grim-nn |

## grim-garage

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_DATASETS_DIR` | dataset search path | `~/.grim/datasets` | grim-garage |
| `GRIM_FUSED_CE` | on unless `== "0"` / `false` | on (ROCm) / off (CPU) | grim-garage |
| `GRIM_MAX_CONCURRENT_JOBS` | integer | `4` | grim-garage |
| `GRIM_ROCM_DEVICE_NAME` | device display-name override for garage probes | `AMD ROCm Accelerator #<ordinal>` | grim-garage |
| `GRIM_ROCM_GCN_NAME` | GCN arch override for garage probes | `gfx1030` | grim-garage |

## grim-backend-cuda

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_CUDA_ORDINAL_OVERRIDE` | ordinal integer (probe) | unset (CUDA enumerate) | grim-backend-cuda |

## grim-backend-cpu

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_DBG_ATTN` | `set` enables the paged-attention score dump | off | grim-backend-cpu |

## grim-kvtransport

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_SHM_DIR` | shared-memory KV inbox root | `/dev/shm/grim-kv` (else OS temp) | grim-kvtransport |

## grim-quant

| Variable | Polarity | Default | Owner(s) |
|---|---|---|---|
| `GRIM_Q4K_MODEL` | model path for the Q4K reference test | `models/MiniCPM5-1B-Q4_K_M.gguf` | grim-quant |
