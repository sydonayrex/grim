# Q2 Benchmark: Device MoE Router (`GRIM_MOE_DEVICE_ROUTER=1`)

## COMMAND
```bash
# CPU env-gate unit test
cargo test -p grim-models-transformer --test moe_device_router_test -- --nocapture

# ROCm Card 1 (gfx1200) 512-expert top_k=10 tie & uniform parity test (grim-backend-rocm)
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test -p grim-backend-rocm --test moe_d2d_routing_gpu \
  route_topk_flash_next_512_top10_ties_and_uniform_matches_host -- --ignored --nocapture --test-threads=1

# ROCm Card 1 (gfx1200) Flash-Next router helper parity & 48-layer 3-run A/B benchmark
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test moe_device_router_test \
  bench_moe_device_router_gpu_card1 -- --ignored --nocapture --test-threads=1
```

Note on real-GGUF checkpoint: `real-GGUF checkpoint: NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test)`.

## EXECUTED
- CPU (`moe_device_router_test`): `1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s`.
- GPU Card 1 (`route_topk_flash_next_512_top10_ties_and_uniform_matches_host`): `1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.06s`.
- GPU Card 1 (`bench_moe_device_router_gpu_card1`): `1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.10s`.

## BEFORE (`GRIM_MOE_DEVICE_ROUTER=0`, 512-logit D2H + host `route_row_host`)
- 48-layer decode router (`num_experts=512, top_k=10`, 2048 B D2H/layer): 3-run `[1.884, 1.874, 1.873]` ms -> median **`1.874 ms`**

## AFTER (`GRIM_MOE_DEVICE_ROUTER=1`, `moe_route_topk_on_device` with `route_mode=0, norm_weights=false`)
- 48-layer decode router (`num_experts=512, top_k=10`, 80 B D2H/layer): 3-run `[16.155, 15.753, 15.728]` ms -> median **`15.753 ms`** (**8.41x slower in eager mode**, `+13.879 ms` due to per-call scratch allocation + kernel launch + `dev.synchronize()` + two small D2H copies per layer).

## NUMERIC GATE
- 512-expert `top_k=10` adjacent equal-logit ties (`[17]==[18]`, `[200]==[201]`): expert IDs **100% identical** (`lower index wins`), weight `max_w_diff = 2.235174e-8` (`< 1e-4`).
- 512-expert all-equal uniform row: expert IDs **100% identical** (`0..10`), weight `max_w_diff = 0.000000e0` (`< 1e-4`).

## TOKEN PARITY
- Identical expert IDs (`0` mismatches) and `< 1e-7` weight delta vs host oracle.

## VERDICT
Negative eager speed / Pass numeric & graph-capture prerequisite — exact routing parity (`max_w_diff <= 2.24e-8`, 0 ID mismatches) and reduces D2H payload from `2048 B` to `80 B` per layer, but eager mode is slower (`15.753 ms` vs `1.874 ms` over 48 layers). Keep `GRIM_MOE_DEVICE_ROUTER` **OFF** by default until Q8 graph capture eliminates per-layer host sync/allocation overhead.
