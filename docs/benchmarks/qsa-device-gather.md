# Q4 Benchmark: QSA Device Gather on Decode (`GRIM_QSA_DEVICE_GATHER=1`)

## COMMAND
```bash
# CPU QSA wiring & env-gate tests
cargo test -p grim-models-transformer --test qwen4exp_qsa_wiring_test -- --nocapture

# ROCm Card 1 (gfx1200) kernel gather & score parity
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test -p grim-backend-rocm --test qsa_gather_parity -- --ignored --nocapture --test-threads=1

# ROCm Card 1 (gfx1200) QSA device gather integration parity & 12-layer 1k/19k decode A/B
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test qsa_device_gather_integration -- --ignored --nocapture --test-threads=1

# ROCm Card 1 (gfx1200) cli_decode_parity probe
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  GRIM_PROBE_MODEL=/drive/bigfast/grim/models/qwen38-27b/qwen38-27b-probe8.gguf \
  cargo test -p grim-models-transformer --test cli_decode_parity -- --ignored --nocapture --test-threads=1
```

Scope note: Q4 removes K/V readback and host masked attention for `seq_len == 1` decode only; indexer scoring stays host and prefill stays dense. **This is a decode win — no prefill TTFT claim.**

Note on real-GGUF checkpoint: `real-GGUF checkpoint: NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test)`.

## EXECUTED
- CPU (`qwen4exp_qsa_wiring_test`): `6 passed; 0 failed; 0 ignored; finished in 0.00s`.
- GPU Card 1 (`qsa_gather_parity`): `3 passed; 0 failed; 0 ignored; finished in 0.12s` (`0/640` bitwise K mismatches, `3.052e-5` score diff).
- GPU Card 1 (`qsa_device_gather_integration`): `1 passed; 0 failed; 0 ignored; finished in 9.68s`.
- GPU Card 1 (`cli_decode_parity` on `qwen38-27b-probe8.gguf`): `1 passed; 0 failed; 0 ignored; finished in 4.89s`.

## BEFORE (`GRIM_QSA_DEVICE_GATHER=0`, K/V D2H + host `masked_gqa_attention` + H2D)
- `ctx=1k` (`kv_len=1024, n_ids=256`, 12 Flash-Next full-attention layers, `24x256` Q, `2x256` KV): 3-run `[107.616, 107.043, 106.960]` ms -> median **`107.043 ms`**
- `ctx=19k` (`kv_len=19456, n_ids=2048`, 12 Flash-Next full-attention layers): 3-run `[2953.015, 2978.675, 3017.118]` ms -> median **`2978.675 ms`**

## AFTER (`GRIM_QSA_DEVICE_GATHER=1`, `launch_kv_gather_f32` + `qkv_attention(..., cache_offset = n_ids - 1)`)
- `ctx=1k` (12 layers): 3-run `[4.328, 4.233, 4.224]` ms -> median **`4.233 ms`** (**25.3x speedup**, `-102.810 ms/token`)
- `ctx=19k` (12 layers): 3-run `[8.331, 8.254, 7.583]` ms -> median **`8.254 ms`** (**360.9x speedup**, `-2970.421 ms/token`)

## NUMERIC GATE
- `ctx=1k`: `max_abs_diff = 2.086163e-7` (`<= 1e-4` gate).
- `ctx=19k`: `max_abs_diff = 3.576279e-7` (`<= 1e-4` gate).
- `qsa_gather_parity`: `0/640` bitwise mismatches; `qsa_index_score` worst `3.052e-5` (`< 1e-4`).

## TOKEN PARITY
- Matches host `masked_gqa_attention` oracle to `<= 3.58e-7` at 1k and 19k context; falls back loudly via `[qsa-device-gather] fallback: <reason>` to the host masked oracle (never dense) on any error.

## VERDICT
Pass — 25.3x (`1k`) and 360.9x (`19k`) decode speedup across 12 QSA layers with `<= 3.58e-7` max diff; flag remains default OFF (`GRIM_QSA_DEVICE_GATHER=1`).
