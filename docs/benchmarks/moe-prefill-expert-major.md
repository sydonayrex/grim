# Q1b Benchmark: Expert-Major MoE Prefill (`GRIM_MOE_EXPERT_MAJOR=1`)

## COMMAND
```bash
# CPU bitwise unit tests (4-tok x 6-exp: 1-tok expert, all-tok expert, top-k tie)
cargo test -p grim-models-transformer --test moe_expert_major_prefill -- --nocapture

# ROCm Card 1 (gfx1200) parity & 3-run A/B benchmark (1-layer Flash-Next geometry, 512 experts, top_k=10, Q8_0)
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test moe_expert_major_prefill \
  bench_moe_expert_major_prefill_gpu_card1 -- --ignored --nocapture --test-threads=1
```

Note on real-GGUF checkpoint: `real-GGUF checkpoint: NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test)`.

## EXECUTED
- CPU (`moe_expert_major_prefill`): `2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s`.
- GPU Card 1 (`bench_moe_expert_major_prefill_gpu_card1`): `1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 24.90s` (executed count = `1`).

## BEFORE (`GRIM_MOE_EXPERT_MAJOR=0`, token-major MoE prefill loop)
- `seq_len=64` (1-layer, 512 experts, `top_k=10`, Q8_0): 3-run `[7812.349, 6818.246, 6783.171]` ms -> median **`6818.246 ms`**

## AFTER (`GRIM_MOE_EXPERT_MAJOR=1`, expert-major grouping + slot-ordered reduction + batched shared expert)
- `seq_len=64` (1-layer, 512 experts, `top_k=10`, Q8_0): 3-run `[397.874, 399.384, 403.844]` ms -> median **`399.384 ms`** (**17.07x speedup**, `-6418.862 ms`)

## NUMERIC GATE
- CPU (4 tokens × 6 experts, `top_k=2`, single-token expert, all-token expert, exact tie): bitwise identical (`to_bits() ==`, `max_abs_diff = 0.0`), active expert call count reduced from `8` to `5`.
- GPU Card 1 (`seq_len=8`, 512 experts, `top_k=10`, Q8_0): `max_abs_diff = 0.000000e0` (bitwise exact).

## TOKEN PARITY
- Identical (`max_abs_diff = 0.000000e0` across all output logits).

## VERDICT
Pass — 17.07x MoE prefill speedup at `seq_len=64` with bitwise parity (`0.0` max diff); flag remains default OFF (`GRIM_MOE_EXPERT_MAJOR=1`).
