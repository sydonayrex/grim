# DATS-DEMMM: Grim Inference Pipeline Review

**Date**: 2026-09-16  
**Scope**: All model files in `crates/grim-models/transformer/src/` + ROCm backend in `crim-backend-rocm/src/`  
**Perspective**: MoE and Dense model inference on ROCm  
**Methodology**: Static audit of `unwrap()` usage, dead code detection, unwired code identification

---

## Executive Summary

The grim inference pipeline is a large Rust codebase with 180+ model implementations and a ROCm backend. While the core infrastructure is sophisticated (graph capture, fused kernels, device-residual KV caches), significant quality issues exist:

- **379 `unwrap()` calls** in model files, **33 in ROCm backend hot path** — these are potential panics in production inference
- **10+ dead config fields** in lfm2.rs alone — declared but never read
- **Complete graph capture infrastructure** that is only wired for LFM2 (and broken by ShortConv/MoE)
- **Unwired device-path functions** exist in block.rs, lfm2.rs, and shared_moe.rs that could benefit multiple models

---

## 1. Unnecessary `unwrap()` Calls

### 1.1 Model Files — Top Offenders

| File | `unwrap()` Count | Risk Level |
|------|------------------|------------|
| `block.rs` | 94 | Critical (core layer, ~90 models inherit) |
| `lfm2.rs` | 68 | Critical (benchmark model) |
| `falcon_h1.rs` | 25 | High |
| `kimi_k3.rs` | 23 | High |
| `gemma.rs` | 11 | High |
| `deepseek.rs` | 11 | High |
| `delta_net_base.rs` | 11 | High |
| `gpt2.rs` | 10 | High |
| `qwen38_flash_next.rs` | 10 | High |
| `falcon.rs` | 7 | Medium |
| `gemma2.rs` | 8 | Medium |
| `chameleon.rs` | 6 | Medium |
| `qwen35.rs` | 5 | Medium |
| `muse_glimmer.rs` | 12 | High |
| `dbrx.rs` | 2 | Low |

### 1.2 block.rs `unwrap()` — Critical Examples

```rust
// Line 629: Panics if wqkv_q80_fused is None (any non-Q8_0 model)
let fused = self.wqkv_q80_fused.as_ref().unwrap();

// Line 768: Panics if w_gate_up_q80_fused is None
let fused = self.w_gate_up_q80_fused.as_ref().unwrap();

// Line 1335: Panics if cache is None during decode
cache.as_mut().unwrap(),

// Line 1408-1409: Panics if owned_k/owned_v are None
owned_k.as_ref().unwrap().as_ref(),
owned_v.as_ref().unwrap().as_ref(),

// Line 1565-1567: Panics if cache fields are None
let past_dev = cache.past_dev.as_ref().unwrap();
let k_arena = cache.k_device.as_ref().unwrap();
let v_arena = cache.v_device.as_ref().unwrap();
```

**Problem**: These are in the **production decode path**. A `None` value (e.g., checkpoint without Q8_0 weights, cache not initialized) causes a panic instead of graceful fallback.

### 1.3 lfm2.rs `unwrap()` — Critical Examples

```rust
// Line 311-315: Panics if any attention weight is None
wq.as_ref().unwrap(),
wk.as_ref().unwrap(),
wv.as_ref().unwrap(),
attn_q_norm.as_ref().unwrap(),
attn_k_norm.as_ref().unwrap(),

// Line 795-797: Panics in forward pass
let q = self.wq.as_ref().unwrap().forward(&norm_x)?;
let k = self.wk.as_ref().unwrap().forward(&norm_x)?;
let v = self.wv.as_ref().unwrap().forward(&norm_x)?;

// Line 873: Panics if cache is None
if cache.as_mut().unwrap().past_dev_needs_init() {

// Line 925, 939: Panics on cache access
let pos_base_dev_mut = match cache.as_mut().unwrap() {
let pos_base = pos_base_dev_mut.as_ref().unwrap();
```

### 1.4 ROCm Backend `unwrap()` — Hot Path

| File | Count | Notes |
|------|-------|-------|
| `kernels/qkv_attention.rs` | 46 | **Worst offender** — attention kernel hot path |
| `kernels/dot_gemv.rs` | 37 | GEMM hot path |
| `device/device_compute.rs` | 7 | Launch path |
| `memory/storage.rs` | 6 | Device pointer access |
| `kernels/charon.rs` | 8 | MoE dispatch |
| `kernels/device_sampler.rs` | 3 | Sampling |

**storage.rs device pointer access** (lines 155, 189, 226, 454, 509):
```rust
let dev_ptr_void = self.device_ptr.unwrap() as *mut c_void;
```
This panics if a tensor reaches a device op without a device pointer. The doc comment on line 52 says "surfaces an error instead of panicking" — but `.unwrap()` does the opposite.

---

## 2. Dead Code

### 2.1 Dead Config Fields (declared, never read)

**lfm2.rs — 4 dead config fields:**

| Field | Line | Status |
|-------|------|--------|
| `n_swa: usize` | 37 | **Dead** — declared, never read after load |
| `swa_type: u32` | 38 | **Dead** — declared, never read after load |
| `expert_weights_scale: f32` | 35 | **Dead** — declared, never read |
| `expert_gating_func: u32` | 36 | **Dead** — declared, never read |

These fields are part of `Lfm2Config` but the forward path never checks them. They appear to be remnants of a planned sliding-window attention and expert gating feature that was never wired.

**qwen35.rs — 1 dead config field:**

| Field | Line | Status |
|-------|------|--------|
| `devices: Vec<Device>` | 38 | **Effectively dead** — only used in `load_tp` to check if non-empty, never used in forward |

### 2.2 Dead Functions (defined, no callers)

**block.rs:**

| Function | Line | Status |
|----------|------|--------|
| `with_alibi(mut self) -> Self` | 591 | **Dead** — public method, zero callers outside block.rs |
| `alibi_slopes_for(num_heads: usize)` | 22 | **Effectively dead** — only called from `with_alibi` (which is dead) |

**shared_moe.rs:**

| Function | Line | Status |
|----------|------|--------|
| `normalize_weights(topk: &[(usize, f32)])` | 775 | **Dead** — public function, no callers outside shared_moe.rs |
| `charon_grouped_dispatch(...)` | 472 | **Effectively dead** — only called from `ensure_charon_scratch` which is only called from `lfm2_graph.rs` |

### 2.3 Dead Fields in Cache Structs

**10 models declare `k_device`/`v_device` fields but never use them:**

| File | Field | Status |
|------|-------|--------|
| `delta_net_base.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `diffusion_gemma.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `gemma3n.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `inkling_small.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `interns2_mobius.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `kimi_k3.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `native_mtp.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |
| `qwen2.rs` | `k_device`, `v_device` | Dead — thin wrapper, inherits block.rs |
| `qwen38_flash_next.rs` | `k_device`, `v_device` | Dead — forward uses CPU path |

---

## 3. Unwired Code (Complete but Only Called by Tests)

### 3.1 Graph Capture Infrastructure — Complete but Only Wired for LFM2

The graph capture system is **fully implemented and tested** but only used by one model:

| Component | File | Status |
|-----------|------|--------|
| `GraphCaptureManager` | `graph_capture.rs` | Complete, tested, only called from `lfm2_graph.rs` |
| `DecodeGraphBuffers` | `decode_graph_buffers.rs` | Complete, tested, only called from `lfm2_graph.rs` |
| `DecodeGraph::replay()` | `decode_graph_buffers.rs:780` | Complete, tested, only called from LFM2 |
| `forward_capture()` | `lfm2_graph.rs:262` | Complete, tested |
| `forward_replay()` | `lfm2_graph.rs:318` | Complete, tested |
| `forward_capture_batch()` | `lfm2_graph.rs:279` | Complete, tested (P3 batch) |
| `forward_replay_batch()` | `lfm2_graph.rs:347` | Complete, tested (P3 batch) |
| `GraphRetryPolicy` | `graph_capture.rs:94` | Complete, tested, env-overridable |

**Impact**: 90+ thin wrapper models (Qwen, Phi, Olmo, StableLM, etc.) could benefit from graph capture but don't have it wired.

### 3.2 Device-Path Functions — Complete but Underwired

**block.rs:**

| Function | Line | Status |
|----------|------|--------|
| `device_graph_decode_attention()` | 1502 | **Wired only for block.rs layers** — complete with parity tests, but lfm2.rs has its own implementation |
| `apply_rmsnorm_rope_multi_head_opt()` | 1211 | **Wired only in block.rs** — could be used by lfm2.rs and other models |

**lfm2.rs:**

| Function | Line | Status |
|----------|------|--------|
| `shortconv_step_device()` | 1803 | **Wired but bisected** — line 627-633: "bisection isolated `shortconv_step_device`, NOT the RoPE" — opt-in via `GRIM_DECODE_GRAPH=1` |
| `decode_attention_device()` | 1654 | **Wired only for LFM2** — complete with fallback, but other models don't use it |
| `fused_qkv()` | 1369 | **Wired only for LFM2** — block.rs has its own version |

### 3.3 Shared MoE Functions — Complete but Underwired

| Function | Line | Status |
|----------|------|--------|
| `fused_moe_dispatch_from_logits()` | 263 | **Wired for DeepSeek2/32/4, KimiK3, LFM2** — complete, but Gemma2, GLM5_2, Qwen35Moe use host routing |
| `charon_grouped_dispatch()` | 472 | **Wired only via ensure_charon_scratch** — complete kernel dispatch, but only LFM2 uses it |
| `ensure_charon_scratch()` | 121 | **Wired only for LFM2 graph** — complete, tested |
| `route_topk()` | 751 | **Wired for DeepSeek2/32/4, KimiK3** — complete, but other MoE models do host routing |
| `normalize_weights()` | 775 | **Dead** — no callers |

### 3.4 Shared Attention Functions — Complete but Underwired

| Function | Line | Status |
|----------|------|--------|
| `fused_qkv_project()` | 103 | **Wired for Chameleon, CommandR, Dots3Note, HyV4** — complete, but most models use `fused_qkv_dot4_decode` |
| `fused_qkv_project_raw()` | 119 | **Wired for Chameleon, GptJ** — complete |
| `build_fused_qkv_q80()` | 127 | **Wired for Chameleon, CommandR, Dots3Note, GptJ, HyV4** — complete |
| `build_fused_qkv_q80_opt()` | 137 | **Unwired** — optimized version exists but no callers |
| `fused_attention_tensors_softcapped()` | 628 | **Wired only for Gemma2** — complete, could benefit other models |
| `gather_paged_history()` | 526 | **Wired only for MiniCPM** — complete, could benefit other paged-attention models |
| `fused_or_scalar_attention_scaled()` | 564 | **Unwired** — scaled attention variant, no callers |

---

## 4. MoE-Specific Pipeline Review

### 4.1 MoE Gate Routing — Host vs Device

**Host Routing (D2H every step):**
- `deepseek2.rs:517` — `logits.to_vec_f32()` for softmax+top-k
- `deepseek32.rs:653` — same pattern
- `deepseek4.rs:646` — same pattern
- `kimi_k3.rs:433` — same pattern
- `gemma2.rs:383` — `logits.to_vec_f32()`
- `glm5_2.rs:151` — `logits_v = logits.to_vec_f32()`
- `diffusion_gemma.rs:382` — `gate.to_vec_f32()`

**Device Routing (Charon):**
- `lfm2.rs:1974` — tries `fused_moe_dispatch_from_logits` first, falls back to host
- `lfm2_graph.rs:1009` — `moe_route_topk_on_device` for graph capture

### 4.2 MoE Expert Dispatch — Host vs Device

**Host Loop (per-expert D2H):**
- `dbrx.rs:124` — `expert.forward(x)?.to_vec_f32()`
- `glm4_moe_lite.rs:138,147` — shared + expert D2H
- `granite_moe_hybrid.rs:141,150` — shared + expert D2H
- `qwen35moe.rs:502-505` — CPU matmul for gate/up/down
- `qwen38_flash_next.rs:284,291` — per-expert + shared D2H

**Device Dispatch (Charon grouped):**
- `shared_moe.rs:472` — `charon_grouped_dispatch` — complete but only called from LFM2

### 4.3 MoE Unwired Infrastructure

The Charon grouped-kernel dispatch is **complete and tested** but only wired for LFM2:

- `device_routing.rs:807` — `launch_charon_grouped_dispatch` — complete
- `device_routing.rs:996` — `launch_charon_grouped_dispatch_fp8` — complete
- `device_routing.rs:1111` — `launch_charon_grouped_dispatch_mxfp4` — complete
- `device_routing.rs:604` — `moe_fused_grouped_dispatch_w8a8_int8` — complete
- `device_routing.rs:684` — `moe_fused_grouped_dispatch_w8a8_fp8` — complete
- `device_routing.rs:735` — `moe_fused_grouped_dispatch_awq` — complete

**None of these are called from any model file except LFM2.**

---

## 5. Dense Model Pipeline Review

### 5.1 Attention — Device vs CPU

**Device-side attention (fused):**
- `block.rs:1280` — `prefilled_self_attention` — device-first with fallback
- `block.rs:1502` — `device_graph_decode_attention` — device-only, graph-capture-ready
- `lfm2.rs:1654` — `decode_attention_device` — device-only with fallback
- `shared_attention.rs:601` — `fused_attention_tensors` — device-first

**CPU-side attention (scalar):**
- `bloom.rs:220` — `fused_or_scalar_attention` with D2H
- `chameleon.rs:241` — D2H for Q/K/V
- `commandr.rs:214` — D2H for Q/K/V
- `deepseek.rs:246` — D2H for Q/K/V
- `exaone4_5.rs:200` — D2H for Q/K/V
- `falcon.rs:279` — D2H for Q/K/V
- `gemma2.rs:222` — D2H for Q/K/V
- `gemma.rs:207` — D2H for Q/K/V
- `gptj.rs:182` — D2H for Q/K/V
- `hy_v4.rs:204` — D2H for Q/K/V
- `longcat_flash.rs:166` — D2H for Q/K/V

### 5.2 FFN — Fused vs Unfused

**Fused gate+up (1 launch):**
- `block.rs:767-769` — `fused_gate_up_dot4_decode` — only for Q8_0 weights
- `lfm2.rs:1568` — `fused_gate_up_dot4_decode` — only for Q8_0 weights

**Unfused (2 launches):**
- All other models use separate `w_gate.forward()` + `w_up.forward()`

**Activation:**
- `block.rs:783-797` — `silu_mul_quant_q81_decode` — fused silu+quant for Q8_0/Q4K
- All other models use `silu_mul_on_device` or host GeGLU

---

## 6. Recommendations

### Critical (Panic Risk)

1. **Replace `unwrap()` with `?` or graceful fallback in block.rs hot path** — 94 calls in the core layer used by ~90 models. A single panic kills the entire inference session.

2. **Replace `unwrap()` in storage.rs device pointer access** — 6 calls that panic instead of returning errors. The doc comment says "surfaces an error" but `.unwrap()` does the opposite.

3. **Replace `unwrap()` in kernels/qkv_attention.rs** — 46 calls in the attention kernel hot path.

### High (Dead Code Removal)

4. **Remove dead config fields from lfm2.rs** — `n_swa`, `swa_type`, `expert_weights_scale`, `expert_gating_func` are never read.

5. **Remove dead `with_alibi` function from block.rs** — public method with zero callers.

6. **Remove dead `normalize_weights` from shared_moe.rs** — public function with zero callers.

7. **Remove dead `k_device`/`v_device` fields from 10 models** — declared but never used.

### Medium (Wiring Existing Infrastructure)

8. **Wire graph capture for block.rs/Llama** — infrastructure exists, tested, but only LFM2 uses it. Would benefit ~90 models.

9. **Wire Charon MoE dispatch for all MoE models** — DeepSeek2/32/4, KimiK3, Gemma2, GLM5_2 all do host routing. The device path exists in `device_routing.rs`.

10. **Wire `shortconv_step_device` by default** — currently opt-in due to bisected regression. Needs parity cover.

11. **Use `fused_attention_tensors_softcapped` for all softcapped models** — currently only Gemma2 uses it.

### Low (Code Quality)

12. **Replace `unwrap()` in test-only code with `expect()`** — makes test failures more descriptive.

13. **Add `#[allow(clippy::unwrap_used)]` to test modules** — centralize the allow attribute.

14. **Document why `build_fused_qkv_q80_opt` exists** — optimized version with no callers.

---

## 7. Verification Notes

- All `unwrap()` counts from static grep of `.unwrap()` in `.rs` files.
- Dead code verified by checking field/function usage across all model files.
- Unwired code verified by checking callers outside the defining module and test modules.
- MoE routing paths verified by tracing `to_vec_f32()` on gate logits.
- Graph capture wiring verified by grep for `decode_graph_active`, `forward_capture`, `forward_replay` in model files.
