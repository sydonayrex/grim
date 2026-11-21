# Q0 Baseline: Flash-Next Host Sync & Stage Latency (`GRIM_FLASH_PROFILE=1`)

## COMMAND
```bash
# GPU 2-layer exact-Flash-Next-geometry Q8_0 baseline on Card 1 (gfx1200)
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  GRIM_FLASH_PROFILE=1 GRIM_KERNEL_TIMES=1 \
  cargo test --release -p grim-models-transformer --lib bench_flash_next_baseline_gpu -- --ignored --nocapture --test-threads=1

# CPU passthrough & probe accumulation unit tests
cargo test -p grim-models-transformer --lib qwen4exp_baseline_probe -- --nocapture
```

Note on real-GGUF checkpoint: `real-GGUF checkpoint: NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test: [get-trace] ShapeMismatch at name=token_embd.weight expected=[2560, 248320] got=[248320, 2560])`.

## EXECUTED
- `bench_flash_next_baseline_gpu`: `1 passed; 0 failed; 0 ignored; 0 measured; 324 filtered out; finished in 51.93s` (executed count = `1`).
- `qwen4exp_baseline_probe` unit tests: `2 passed; 0 failed; 1 ignored; 0 measured; 322 filtered out; finished in 0.00s` (`enabled_profile_accumulates_time_and_transfers`, `unset_profile_is_passthrough_returning_value_unchanged`).
- Pre-commit regression checks after Q0.2 probe wiring:
  - `cargo test -p grim-models-transformer --lib qwen4exp -- --nocapture` -> `53 passed; 0 failed`
  - `cargo test -p grim-models-transformer --test qwen4exp_synthetic_forward_test --test qwen4exp_gdn_wiring_test --test qwen4exp_qsa_wiring_test -- --nocapture` -> `5 + 5 + 5 passed; 0 failed`
  - `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 cargo test -p grim-models-transformer --lib qwen38_moe_device_dispatch_matches_host_reference -- --nocapture` -> `1 passed; 0 failed (0.13s)`

## BEFORE
Baseline establishment phase (no prior Q0 profile harness existed).

## AFTER

### 2-Layer Measured Profile (1 GDN layer + 1 QSA layer, `h=2560, num_experts=512, top_k=10, inter=640, vocab=248320`, Q8_0 on `Device::Rocm(0)` mapped to Card 1 `gfx1200`)

#### `decode_1k` (`ctx=1024`, `seq_len=1`)
- 3-run wall times: `[600.444, 614.257, 611.139]` ms -> median **`611.139 ms`**

| Stage | Calls | Total (ms) | D2H Count | D2H Bytes | H2D Count | H2D Bytes |
|---|---:|---:|---:|---:|---:|---:|
| `Router` | 4 | 0.211 | 2 | 4,096 | 0 | 0 |
| `ExpertStage` | 2 | 570.021 | 0 | 0 | 20 | 104,448,000 |
| `Gdn` | 1 | 34.321 | 6 | 230,272 | 1 | 24,576 |
| `Qsa` | 1 | 3.261 | 8 | 34,304 | 1 | 24,576 |
| `LmHead` | 1 | 2.346 | 0 | 0 | 0 | 0 |

#### `decode_19k` (`ctx=19456`, `seq_len=1`)
- 3-run wall times: `[1174.477, 1193.569, 1182.885]` ms -> median **`1182.885 ms`**

| Stage | Calls | Total (ms) | D2H Count | D2H Bytes | H2D Count | H2D Bytes |
|---|---:|---:|---:|---:|---:|---:|
| `Router` | 4 | 0.224 | 2 | 4,096 | 0 | 0 |
| `ExpertStage` | 2 | 568.871 | 0 | 0 | 20 | 104,448,000 |
| `Gdn` | 1 | 34.037 | 6 | 230,272 | 1 | 24,576 |
| `Qsa` | 1 | 576.402 | 8 | 34,304 | 1 | 24,576 |
| `LmHead` | 1 | 2.352 | 0 | 0 | 0 | 0 |

#### `prefill_64` (`seq_len=64`, full 2-layer forward)
- 3-run wall times: `[13583.116, 13571.180, 13600.385]` ms -> median **`13583.116 ms`**

| Stage | Calls | Total (ms) | D2H Count | D2H Bytes | H2D Count | H2D Bytes |
|---|---:|---:|---:|---:|---:|---:|
| `Router` | 130 | 1.842 | 2 | 262,144 | 0 | 0 |
| `ExpertStage` | 128 | 13308.945 | 1282 | 14,417,920 | 1405 | 6,661,181,440 |
| `Gdn` | 1 | 133.134 | 6 | 4,383,232 | 1 | 1,572,864 |
| `Qsa` | 1 | 16.871 | 8 | 2,130,944 | 1 | 1,572,864 |
| `LmHead` | 1 | 137.198 | 0 | 0 | 0 | 0 |

#### `lm_head_2048` (`[2048, 2560]` through `model.output.forward`, 248320 vocab)
- 3-run wall times: `[892.859, 877.549, 867.728]` ms -> median **`877.549 ms`**
- Run-2 `LmHead` span: `867.708 ms` (`0 D2H`, `0 H2D`)

#### Top GPU Kernel Times (`GRIM_KERNEL_TIMES=1`)
- `__dev0_total__`: `calls=16173`, `total_ms=2.751`
- `grim_wmma_fused_dequant_q8_0`: `calls=2`, `total_ms=1.745`
- `grim_fp32_mm_tile`: `calls=11938`, `total_ms=0.622`
- `grim_dot4_q80_f32act_gemv_v2`: `calls=105`, `total_ms=0.310`
- `grim_silu_mul`: `calls=3979`, `total_ms=0.040`

### 48-Layer Linear Projection (36 GDN + 12 QSA + 48 MoE + 1 LmHead)
- `decode_1k`: `Router` `5.06 ms` + `ExpertStage` `13680.50 ms` + `Gdn` `1235.56 ms` + `Qsa` `39.13 ms` + `LmHead` `2.35 ms` = **`~14.96 s/tok`** (`~1.28 s/tok` excluding `ExpertStage` cold staging, where `Gdn` is 96% of non-expert compute).
- `decode_19k`: `Router` `5.38 ms` + `ExpertStage` `13652.90 ms` + `Gdn` `1225.33 ms` + `Qsa` `6916.82 ms` + `LmHead` `2.35 ms` = **`~21.80 s/tok`** (`Qsa` + `Gdn` = `~8.14 s/tok`).

### Key Baseline Findings
1. **`GRIM_ALLOC_POOL=1` & `staged_to` OOM root cause:** `Linear::staged_to` -> `move_to_device` (`crates/grim-nn/src/modules.rs:419`) dequantizes Q8_0 `[640, 2560]` experts to F32 (`6,553,600 B` > `SMALL_MAX = 4 MiB`) plus `w_t`. Without `GRIM_ALLOC_POOL=1`, unpooled `hipMalloc`/`hipFree` leaks ~2.4 GiB/step in the ROCm driver until `hipMalloc(slab) failed: 2` on `decode_19k`. With `GRIM_ALLOC_POOL=1`, Card 1 VRAM stays flat at `2363 MB` (`13941 MB` free).
2. **Host-sync bottleneck confirms Q1–Q6 order:**
   - Prefill (`prefill_64`): token-major MoE loop stages experts `1280+` times (`13.31 s`), plus `[S, 248320]` full `lm_head` projection (`877.5 ms` at `S=2048` on device).
   - Decode: `ExpertStage` clear-all at `cache.len() >= 8` + QSA host indexer/gather at 19k (`576.4 ms`/layer, `8 D2H`) + GDN host `short_conv1d`/recurrence (`34.3 ms`/layer, `6 D2H` + `1 H2D`).
3. **Integrator note (`qwen4exp_flash_next.rs`, originally `:1247`, now `:1348-1457`):** comments explicitly keep `short_conv1d`, recurrence, and output gate on host to avoid mixed-device storage errors, explaining the 6 D2H + 1 H2D transfers per GDN layer.

## NUMERIC GATE
- `GRIM_FLASH_PROFILE` unset compiles to an inline pass-through (`ProbeStage` spans execute closures directly with zero synchronization or extra allocations); all 53 `qwen4exp` unit tests and 15 integration tests pass unchanged.

## TOKEN PARITY
- N/A for Q0 (instrumentation-only probe, zero arithmetic or control-flow change).

## VERDICT
Pass — Q0 baseline metrics captured on Card 1 (`gfx1200`) with `executed=1`, host sync hotspots quantified, ready for Q1a.
