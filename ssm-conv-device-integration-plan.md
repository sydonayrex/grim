# SSM / shortconv device integration plan — falcon_h1.rs + lfm2.rs

Follow-up to the round-trip fixes landed 2026-09-03 (block.rs prefill, qwen38
SwiGLU/attention, muse_glimmer KV arena). This plan covers the remaining
forward-path host compute in `falcon_h1.rs` and `lfm2.rs`.

**Headline: most of this is integration, not new kernel work.** The backend
trait already carries `short_conv1d_causal_step` and `selective_scan`; the
mamba crate is the only consumer today. Exactly one new kernel variant is
needed (falcon_h1's per-state-B/C scan recurrence). lfm2's MoE and attention
are already device-first; its shortconv decode composes from existing kernels.

---

## Verified current state (2026-09-03)

### Kernel surface that already exists (`grim-tensor/src/backend.rs`)

| Kernel | Impls | Contract |
|---|---|---|
| `short_conv1d_causal_step` (L1516) | CPU, CUDA, Vulkan, ROCm | Single-token depthwise causal conv. `out[h] = Σ_k state[h,k]·w[h,k] + x[h]·w[h,last] + bias`. `w` is `[hidden, k]` row-major per channel — matches both models' weight layout. Does **not** update `state`; caller owns it. |
| `selective_scan` (L1557) | ROCm, CUDA, Vulkan, Metal | Decode-step only. `h[n,s] = a[n,s]·h_prev + dt[n]·x[n]·b[n]`, `y[n] = Σ_s c[n]·h[n,s] + D[n]·x[n]`. **B and C are per-channel scalars** (`[d_inner]`, broadcast across s). `a` is `[d_inner·d_state]`, precomputed `exp(a_log+1)`. State updated in place (`h_in_out`). |
| `silu_mul`, `add`, `mul`, `rope`, `qkv_attention`, `alloc_storage`, `copy_slice_into/range` | all backends | — |
| `cache_append_kv` (`block.rs:277`, `pub(crate)`) | composes on the above | Geometric-grow device KV arena append. |

### falcon_h1.rs (22 sites, all production, zero device kernels used)

- **Attention (L453-500)**: Q/K RoPE'd on device, then `to_vec_f32` → host
  `k_cache.extend_from_slice` → host `fused_or_scalar_attention`. Cache is
  `FalconH1LayerCache` (L76) — host `Vec` mirrors only, no device fields.
- **SSM conv (L506-517)**: `ssm_in` → D2H, triple-nested host loop
  `O(seq·conv_dim·d_conv)`, conv weight pulled per call (L507-508).
- **SSM scan (L525-560)**: per-token recurrence on host: `dt = softplus(dt_pre+dt_b)`
  per **head**; `d_a = exp(dt[h]·a_vec[h])` per head; `s_new = s·d_a + b_t[s]·x·dt`;
  `y = Σ_s c_t[s]·s_new + D[h]·x·dt`. **B_t and C_t are per-token `[d_state]`
  vectors** sliced from the xBC buffer — this is Mamba-2/SSD-style indexing and
  does **not** match the per-channel-B/C `selective_scan` contract.
- **SwiGLU (L562-570)**: `silu(z)·y` on host → `wrap()` H2D → `ssm_out` matmul
  → D2H again (L573-575).

### lfm2.rs (29 production sites; most gaps already closed upstream)

- **Attention**: already arena-first (`fused_or_scalar_attention_arena` L694);
  host path at L715 is the kernel fallback. No work.
- **MoE**: `forward_moe_ffn_device` (L941) is primary on GPU; the L1084-1087
  expert-stack pulls are the host fallback only. Router probs (L1043-1070)
  stay on host by design — `[steps, n_expert]` softmax is tiny and steers
  dispatch. No work beyond keeping the fallback honest.
- **Shortconv (L399-478) — the real remaining gap**: `proj` → D2H (L401),
  host split into `b/c/x_val`, host depthwise conv loop with `Vec::copy_within`
  state, gate `y[d] = c[d]·conv(bx)[d]`. All composable from existing kernels:
  `mul` (b·x), `short_conv1d_causal_step` (conv), `mul` (c gate epilogue).
  No new kernel needed for decode (steps == 1).
- Env-guarded debug pulls (`GRIM_DEBUG_SHORTCONV`, L405-435) and weight
  loading (L1162-1196) are legitimate; leave.

---

## Implementation status (2026-09-04)

| Item | Status | Notes |
|---|---|---|
| WI-A attention arena | **landed** | `FalconH1LayerCache` gained `k_device`/`v_device`; `gqa_attn_with_cache` is tensor-based (rope + `cache_append_kv` + `qkv_attention`); host mirrors only advance on the `Unimplemented` fallback. Parity: `test_gqa_arena_matches_host_reference` (8-step decode, atol 1e-5) + `test_gqa_device_path_leaves_mirrors_empty`. Also fixed the arena-fetch bug class: capacity-sized `to_cpu_vec_f32` results must be truncated to valid rows (same latent bug fixed in muse_glimmer's kernel-Err branch). |
| WI-B conv decode | **landed** | `ssm_conv_step_device`: D2D xBC slice + `short_conv1d_causal_step` at seq==1; prefill keeps the host loop. Parity: `test_ssm_decode_matches_prefill` (prefill×3 vs 3×decode, atol 1e-4 — batch-shape matmul ulp noise). |
| WI-C device state | folded into WI-D | While the scan is host-bound, device state adds round-trips instead of removing them. State goes resident with the scan kernel (`selective_scan*` already takes `state` in place). |
| WI-D scan | **D2 landed** | New trait method `BackendDevice::selective_scan_headed` (per-head dt/A/D, per-token B/C — default `Unimplemented`) + `ssm_scan_step_device` try-dispatch in the decode path. Every backend currently falls through to the host loop. HIP/CUDA/Vulkan/Metal kernel impls remain the follow-up (need GPU toolchain verification). |
| WI-E prefill conv kernel | deferred | per plan. |
| WI-F lfm2 shortconv | **landed** | `shortconv_step_device`: D2D b/c/x slices + `mul` + conv kernel + `mul` gate; decode no longer pulls `proj` to host; device `block_out` feeds the same residual + FFN tail. Parity: `shortconv_decode_matches_prefill`. |
| WI-G cleanup | **landed** | Dead `q_reshaped` removed; debug probes stripped; lint baselines updated with rationale comments (`falcon_h1` 21→25: test asserts + fallback branch; `lfm2` 25→27: device-path bx fetch + decode test — runtime decode round-trips went down). |

Suite: 146 lib + roundtrip_lint + 6 e2e green; full workspace builds.
Pre-existing clippy debt in `grim-tensor` (div_ceil, doc-comment lint) is untouched by this work.

---
---

## Work items (in execution order)

### WI-A — falcon_h1 attention → device arena  *(pure integration, no kernel changes, ~0.5-1 day)*

Highest win/effort ratio: kills the O(context) H2D re-upload per decode step.

1. `FalconH1LayerCache`: add `k_device`/`v_device:
   Option<Box<dyn BackendStorage>>` (mirror `LlamaLayerCache`, block.rs:169).
2. Attention fn: replace host extend + `fused_or_scalar_attention` with
   `cache_append_kv` + `dev.qkv_attention(q_rot.storage(), k_st, v_st, ...)`.
   Reference implementations: `block.rs:843-935` and the muse_glimmer
   `forward_with_kv` migration landed 2026-09-03 (device_attempt →
   `Ok(None)` → host helper structure).
3. Drop `q_roped.to_vec_f32()`/`k_roped.to_vec_f32()` from the primary path;
   host attention remains the `Unimplemented`-guarded fallback.
4. No-cache prefill branch: borrow `k_rot.storage()` directly (block.rs:880
   pattern landed today).

**Gate**: parity test — 8-step decode, device path vs current host path,
atol 1e-5. `cargo test -p grim-models-transformer`.

### WI-B — falcon_h1 conv1d → `short_conv1d_causal_step` (decode path)  *(~0.5 day)*

1. Decode path (`seq_len == 1`): one kernel call replaces the triple loop's
   single iteration. Weight layout already matches (`conv_w[i1·d_conv+i0]`
   is row-major per output channel).
2. Gate dispatch on `seq_len == 1`; prefill (`seq_len > 1`) keeps the host
   loop for now — per-token kernel launches would cost more than the host
   loop at prefill lengths. Revisit in WI-E.
3. Keep `conv_state` as host mirror initially; upload `(d_conv-1)·conv_dim`
   floats per step (negligible). Move to device in WI-C.

### WI-C — falcon_h1 SSM/conv state on device  *(~1 day, after WI-B)*

1. `FalconH1LayerCache.conv_state`/`ssm_state` → device arenas
   (`alloc_storage` at init, kernels read/write in place — `selective_scan`
   already takes `h_in_out`).
2. Update host mirror only for the host fallback path.

### WI-D — falcon_h1 scan: extend `selective_scan` for per-state B/C  *(the one real kernel item, ~2-4 days)*

Contract mismatch is the blocker for the scan only. Two options:

- **D1 (preferred)**: add `selective_scan_headed` (name TBD) to
  `BackendDevice` with explicit layout params:
  `x [d_inner]`, `dt [n_heads]` (post-softplus — keep softplus on host or a
  prior elementwise kernel), `a [n_heads]`, `d [n_heads]`,
  `b/c [d_state]` per token, `heads→channels` map via `head_dim_ssm` param.
  Recurrence: `h[n,s] = exp(dt[h(n)]·a[h(n)])·h_prev + b[s]·x[n]·dt[h(n)]`,
  `y[n] = Σ_s c[s]·h[n,s] + d[h(n)]·x[n]·dt[h(n)]`.
  Default `Err(Unimplemented)`; HIP reference in
  `grim-backend-rocm/src/kernels/selective_scan.rs` (fork the existing
  `grim_selective_scan` source; index changes only), then port to CUDA
  (`cuda_device.rs`), Vulkan (`lib.rs`), Metal (`lib.rs`) following the
  existing per-channel impls.
- **D2 (fallback if kernel bandwidth is unavailable)**: integrate behind
  try-dispatch — `if let Ok(out) = dev.selective_scan(...) { }` falls through
  to the host loop (mamba crate pattern, mamba/src/lib.rs `step_block`).
  Ships the conv/attention/state wins immediately; scan lands when the
  kernel does.

Either way: host loop stays as the CPU implementation (sequential recurrence
— CPU kernel would buy nothing) and as the GPU-failure fallback.

**Gate**: property test vs the host loop as reference — random tensors,
seq 1..64, `n_heads` ∈ {1, 4, 8}, atol 1e-5; plus the full-model falcon_h1
e2e parity.

### WI-E (optional, defer) — full-sequence shortconv/conv prefill kernel

Only if prefill profiling shows the host conv loop matters after decode is
device-side. A seq-parallel depthwise causal conv is straightforward
(im2col-free, per-channel) but touches all four backends.

### WI-F — lfm2 shortconv decode → existing kernels  *(no new kernels, ~1 day)*

Decode path (`steps == 1`) composed entirely from existing kernels:

1. `bx = mul(b_slice, x_val_slice)` — `dev.mul` (needs column-slice of
   `proj` into three `[1, h_dim]` tensors: verify `Tensor` narrow/slice
   support on device storage; if absent, one small `copy_slice_range`
   composition or a `slice_cols` helper in grim-nn).
2. `conv = short_conv1d_causal_step(bx, w, None, conv_state_storage, ...)`.
3. `y = mul(conv, c_slice)` — gate epilogue.
4. State slide: `copy_slice_range` D2D (replaces `copy_within`).
5. Gate dispatch on `steps == 1`; multi-token prefill keeps the host loop.
6. Remove `proj_v` D2H (L401) from the decode path.

**Gate**: parity vs host loop, atol 1e-5; `roundtrip_lint` baseline for
lfm2.rs ratchets **down** from 16.

### WI-G — cleanup + lint ratchet  *(~0.5 day, last)*

1. Re-run the site audit on both files; delete any now-dead host code paths
   that were only reachable via the replaced code (keep explicit fallbacks).
2. Lower `roundtrip_lint.rs` baselines for `falcon_h1.rs` and `lfm2.rs` to
   the new counts so regressions fail CI.
3. Update the performance-audit claims doc if one is checked into the repo.

---

## Sequencing and risk

| Order | Item | New kernel code? | Fallback preserved? |
|---|---|---|---|
| 1 | WI-A attention arena | no | yes |
| 2 | WI-B conv decode | no | yes |
| 3 | WI-C device state | no | yes |
| 4 | WI-D scan kernel | yes (D1) / no (D2) | yes |
| 5 | WI-F lfm2 shortconv | no | yes |
| 6 | WI-G cleanup | no | — |

Everything lands behind try-dispatch with the host path as fallback
(mamba-crate pattern), so no backend regression can break model loading or
CPU inference. All items are single-GPU — none depend on the multi-GPU
verification gap noted in `gpu-followup-workitems.md`.

## Verification

- Per-item parity tests vs the host reference (atol 1e-5).
- `cargo test -p grim-models-transformer` (149 tests + `roundtrip_lint`).
- `roundtrip_lint` baselines ratcheted down in WI-G.
- GPU smoke on ROCm for WI-B/C/D paths; CUDA/Vulkan/Metal follow the trait
  default (Unimplemented → host fallback) until their ports land.
