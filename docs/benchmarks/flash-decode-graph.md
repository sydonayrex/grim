# Q8 Benchmark: Flash-Next ROCm Decode Graph Capture (`GRIM_FLASH_DECODE_GRAPH=1`)

## COMMAND
```bash
# CPU env-gate & wiring tests
cargo test -p grim-models-transformer --test qwen4exp_decode_graph_test --test qwen4exp_gdn_wiring_test -- --nocapture

# ROCm Card 1 (gfx1200) 10-step captured graph parity & 36-layer GDN A/B
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test qwen4exp_decode_graph_test \
  flash_decode_graph_gdn_10step_parity_and_36layer_ab -- --ignored --nocapture --test-threads=1
```

Architecture note: uses string graph keys (`flash_gdn_{tag:x}_d{ord}_l{layer}`) and `RocmDevice::{begin,end,replay}_graph_capture_force` with zero-alloc `short_conv1d_causal_step_into` + `launch_kda_delta_rule_raw_into` + `mul_into` + `sigmoid_mul_into` into fixed-address device buffers (**no changes to `DecodeGraphKey`**). PLE lookup, QSA host indexer, and host-staged MoE stay outside captured regions.

## EXECUTED
- CPU (`qwen4exp_decode_graph_test`): `1 passed; 0 failed; 1 ignored; finished in 0.00s`.
- GPU Card 1 (`flash_decode_graph_gdn_10step_parity_and_36layer_ab`): `1 passed; 0 failed; 0 ignored; finished in 0.17s` (`is_layer_captured(0) == true`, zero fallbacks).

## BEFORE (`GRIM_FLASH_DECODE_GRAPH=0`, eager `gdn_decode_step_on_device`)
- 36-layer Flash-Next GDN decode (`48x128`): 3-run `[6.874, 6.552, 6.573]` ms -> median **`6.573 ms`**

## AFTER (`GRIM_FLASH_DECODE_GRAPH=1`, captured HIP graph replay)
- 36-layer Flash-Next GDN decode (`48x128`): 3-run `[2.977, 2.739, 2.738]` ms -> median **`2.739 ms`** (**2.40x speedup** over eager GPU, **451x** cumulative over Q0 host baseline `1236.1 ms`).

## NUMERIC GATE
- 10-step captured graph vs eager GPU GDN: `max_out_diff = 0.000000e0`, `max_state_diff = 0.000000e0` (bitwise exact).

## TOKEN PARITY
- Bitwise identical (`0.0` diff on output and recurrent state across 10 steps).

## VERDICT
Pass — 2.40x speedup over eager GPU (`6.573 ms` -> `2.739 ms` for 36 GDN layers) with bitwise parity (`0.0` diff) and zero `DecodeGraphKey` edits; flag remains default OFF (`GRIM_FLASH_DECODE_GRAPH=1`).
