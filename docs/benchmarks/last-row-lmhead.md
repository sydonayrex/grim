# Q1a Benchmark: Last-Row LM Head on Prefill (`GRIM_FLASH_LAST_ROW_LMHEAD=1`)

## COMMAND
```bash
# CPU unit & parity tests
cargo test -p grim-models-transformer --test qwen4exp_last_row_lmhead -- --nocapture

# ROCm Card 1 (gfx1200) parity & 3-run A/B benchmark (Q8_0 [248320, 2560] lm_head)
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test qwen4exp_last_row_lmhead \
  bench_last_row_lmhead_gpu_card1 -- --ignored --nocapture --test-threads=1
```

Design note: reused existing `SessionT::full_output_logits()` and `SessionT::skip_output_projection()` (`crates/grim-core/src/session.rs`), so no `CausalLm` trait change in `crates/grim-core/src/model.rs` was required. `session.set_last_hidden_state(normed.clone())` preserves the full `[seq_len, hidden]` normed hidden state.

Note on real-GGUF checkpoint: `real-GGUF checkpoint: NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test)`.

## EXECUTED
- CPU (`qwen4exp_last_row_lmhead`): `4 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s`.
- GPU Card 1 (`bench_last_row_lmhead_gpu_card1`): `1 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out; finished in 4.11s` (executed count = `1`).

## BEFORE (`GRIM_FLASH_LAST_ROW_LMHEAD=0`, full `[seq_len, 248320]` projection)
- `seq_len=512`: 3-run `[237.429, 221.742, 221.052]` ms -> median **`221.742 ms`**
- `seq_len=2048`: 3-run `[872.256, 874.845, 876.152]` ms -> median **`874.845 ms`**

## AFTER (`GRIM_FLASH_LAST_ROW_LMHEAD=1`, device-sliced `[1, 248320]` projection)
- `seq_len=512`: 3-run `[2.251, 2.248, 2.245]` ms -> median **`2.248 ms`** (**98.6x speedup**, `-219.494 ms`)
- `seq_len=2048`: 3-run `[2.292, 2.253, 2.245]` ms -> median **`2.253 ms`** (**388.3x speedup**, `-872.592 ms`)

## NUMERIC GATE
- CPU (`seq_len=8`, F32): bitwise identical (`to_bits() ==`) on last-row logits AND full `[8, hidden]` `last_hidden_state`; `full_output_logits(true)` returns bitwise identical `[8, vocab]`.
- GPU Card 1 (`seq_len=8`, Q8_0 `[248320, 2560]`): `max_abs_diff = 0.000000e0` (bitwise exact).

## TOKEN PARITY
- GPU Card 1 (`seq_len=8`, 248320 vocab): `top1_on = 213232 == top1_off = 213232` (exact match).

## VERDICT
Pass — 98.6x (`seq=512`) and 388.3x (`seq=2048`) LM-head prefill speedup with bitwise parity (`0.0` max diff) and exact top-1 match; flag remains default OFF (`GRIM_FLASH_LAST_ROW_LMHEAD=1`).
