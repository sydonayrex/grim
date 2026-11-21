# Q6 Benchmark: Device Argmax for Qwen4 Greedy (`GRIM_QWEN4_DEVICE_ARGMAX=1`)

## COMMAND
```bash
# CPU env-gate test
cargo test -p grim-models-transformer --test qwen4exp_device_argmax_test -- --nocapture

# ROCm Card 1 (gfx1200) full 248320-vocab parity (spread, ties, uniform, last-idx) & 100-step 3-run A/B
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test qwen4exp_device_argmax_test \
  device_argmax_full_248320_vocab_parity_and_ab -- --ignored --nocapture --test-threads=1
```

## EXECUTED
- CPU (`qwen4exp_device_argmax_test`): `1 passed; 0 failed; 1 ignored; finished in 0.00s`.
- GPU Card 1 (`device_argmax_full_248320_vocab_parity_and_ab`): `1 passed; 0 failed; 0 ignored; finished in 0.18s`.

## BEFORE (`GRIM_QWEN4_DEVICE_ARGMAX=0`, 993,280 B D2H + CPU argmax)
- 100 steps × 248,320 vocab: 3-run `[18.357, 18.270, 18.278]` ms -> median **`18.278 ms`** (`0.183 ms/token`)

## AFTER (`GRIM_QWEN4_DEVICE_ARGMAX=1`, `gpu_argmax_into_dev` + 4-byte D2H)
- 100 steps × 248,320 vocab: 3-run `[15.476, 15.482, 15.720]` ms -> median **`15.482 ms`** (`0.155 ms/token`, **1.18x speedup**, `-0.028 ms/token`, D2H reduced from `993,280 B` to `4 B`).

## NUMERIC GATE
- Full 248,320 vocab exact match against CPU `Iterator::max_by`:
  - `spread`: `123456 == 123456`
  - `ties_highest_wins`: `200000 == 200000`
  - `uniform`: `248319 == 248319`
  - `last_index_max`: `248319 == 248319`

## PLAN CORRECTION (Tie-Break Semantics)
- `grim_argmax_stage1` (`crates/grim-backend-rocm/src/kernels/compute_kernels.rs:405-445`) breaks ties using `>=` / `s_idx[tid+s] > s_idx[tid]` (**highest index wins**), matching Rust's `Iterator::max_by`. This retracts the PLAN's "lowest index" note.
- `qwen35.rs:7577-7582` already subtracts `nextn_reserve_bytes` in `plan_layer_devices` (no production edit needed).

## TOKEN PARITY
- 4/4 full-vocab cases exact (`0` mismatches).

## VERDICT
Pass — modest eager speedup (`1.18x`, `-28 us/token`, `993 KB` -> `4 B` D2H) with exact 248,320-vocab parity; flag remains default OFF (`GRIM_QWEN4_DEVICE_ARGMAX=1`).
