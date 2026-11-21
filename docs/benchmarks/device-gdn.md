# Q3 Benchmark: Device GDN Decode Step (`GRIM_GDN_DEVICE=1`)

## Q3.1 Launcher Location & Design Audit (Recorded Before Code)
- **27B decode graph caller (`crates/grim-models/transformer/src/decode_graph.rs:5132-5184`):** does **not** call `gla_launchers.rs:49,144` (`grim_gla_state_update_output`, specialized to `dk==64, dv==64`). Instead, 27B decode calls `short_conv1d_causal_step_into` + `kda_gated_delta_rule_batched_into` (`crates/grim-backend-rocm/src/device/device_recurrent.rs`), backed by existing HIP kernel `grim_kda_gated_delta_rule_batched` (`crates/grim-backend-rocm/src/kernels/compute_kernels.rs:2606`).
- **Exact parity with `qwen4exp_gdn.rs`:**
  - `conv_out` layout `[q | k | v]`, `Interleaved` head pairing `h % n_k_heads` (default in `qwen4exp_flash_next.rs:1153`).
  - `kda_silu_dev` applied inside `kda_delta_rule_batched_dev` (pass raw post-conv output without pre-SiLU).
  - `gate = softplus(alpha + dt_bias) * ssm_a`, `beta = sigmoid(beta)`, `decay = exp(gate)`.
  - `l2_norm` denominator `sqrtf(ss + eps)` -> pass `eps = cfg.rms_norm_eps * head_dim as f32`.
  - State layout `S[h, j, i]` (`[n_v_heads, head_dim, head_dim]`), output `acc[idx] = a * rsqrtf(head_dim)`.
- **Post-recurrence difference:** Flash-Next applies `acc * ssm_norm_weight[d] * sigmoid(gate)` (no RMS norm, `sigmoid` gate rather than `silu`). Implemented on device via existing `grim_mul` (`BackendDevice::mul`) + `sigmoid_on_device` with a Rust-only launcher `launch_kda_delta_rule_raw_into` in `gla_launchers.rs` (**zero HIP kernel source edits**; Gate D not triggered).

## COMMAND
```bash
# CPU env-gate & regression tests
cargo test -p grim-models-transformer --test qwen4exp_gdn_device --test qwen4exp_gdn_wiring_test --test qwen4exp_synthetic_forward_test -- --nocapture

# ROCm Card 1 (gfx1200) 1-step + 10-step drift oracle & 36-layer Flash-Next GDN A/B
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 \
  cargo test --release -p grim-models-transformer --test qwen4exp_gdn_device \
  gdn_device_step_matches_host_oracle_1step_and_10step_drift -- --ignored --nocapture --test-threads=1
```

Note on real-GGUF checkpoint: `real-GGUF checkpoint: NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test)`.

## EXECUTED
- CPU (`qwen4exp_gdn_device`, `qwen4exp_gdn_wiring_test`, `qwen4exp_synthetic_forward_test`): `1 + 5 + 5 = 11 passed; 0 failed`.
- GPU Card 1 (`gdn_device_step_matches_host_oracle_1step_and_10step_drift`): `1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 4.22s` (executed count = `1`).

## BEFORE (`GRIM_GDN_DEVICE=0`, host `short_conv1d` + host `gated_delta_net_forward` + host `sigmoid` + H2D)
- 36-layer Flash-Next GDN decode (`n_v_heads=48, n_k_heads=16, head_dim=128, conv_k=4`): 3-run `[1236.118, 1237.028, 1235.961]` ms -> median **`1236.118 ms`**

## AFTER (`GRIM_GDN_DEVICE=1`, device `short_conv1d_causal_step_into` + `launch_kda_delta_rule_raw_into` + device `mul`/`sigmoid`)
- 36-layer Flash-Next GDN decode (`n_v_heads=48, n_k_heads=16, head_dim=128, conv_k=4`): 3-run `[7.656, 6.805, 6.585]` ms -> median **`6.805 ms`** (**181.6x speedup**, `-1229.313 ms/token`)

## NUMERIC GATE
- 10-step recurrence drift vs `qwen4exp_gdn.rs` host oracle on signed inputs (`48x128` geometry):
  - `max_out_diff = 1.629815e-9` (`< 1e-5` gate)
  - `max_state_diff = 4.656613e-9` (`< 1e-5` gate)
  - `conv_state_diff < 1e-5` at every step

## TOKEN PARITY
- Matches host GDN oracle to `1.63e-9` output and `4.66e-9` recurrent state across 10 consecutive steps.

## VERDICT
Pass — 181.6x speedup across Flash-Next's 36 GDN layers (`1236.1 ms` -> `6.8 ms`) with `< 5e-9` 10-step drift and zero HIP kernel source changes; flag remains default OFF (`GRIM_GDN_DEVICE=1`).
