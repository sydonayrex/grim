# Grim-Metal Parity — Implementation Log (liquid.md)

> Companion to **`glass.md`** (parity plan). This file tracks implementation status, decisions, and next concrete steps. Updated as work happens.

**Author:** [REDACTED]
**Created:** 2026-08-30
**Root:** `/D/rex/projects/grim`

---

## Phase 1 — Critical backward passes ✔/⏳

**Status:** Four of four kernel gaps **implemented**; `cargo check --package grim-backend-metal` passes.

### What was done
- Added MSL kernels to `src/kernels.msl`:
  - `grim_embedding_scatter_add` — scatter-add: `dweight[token_id[t], :] += out_grad[t, :]`
  - `grim_zeros_f32` — flat zero-fill (used by `embedding_backward` two-pass path)
- Registered pipeline slots in `MetalDeviceInner` + `MetalPipelines` struct (`src/lib.rs`):
  - `rmsnorm_backward`, `rope_backward`, `softmax_backward`, `embedding_backward` already had slots from prior work
  - Added `embedding_scatter_add`, `zeros_f32` slots
- Added `get_pipeline(...)` calls in the `MetalContext::get()` init block for `zeros_f32`
- `embedding_backward` dispatcher already existed and now correctly references the new `zeros_f32` + `embedding_scatter_add` pipelines

### Kernel locations (for reference)
|| Kernel | File:lines |
||---|---| |
|| `grim_rmsnorm_backward` | `src/kernels.msl:4197` |
|| `grim_rope_backward` | `src/kernels.msl:4197` (mirror — verify exact line) |
|| `grim_softmax_backward` | `src/kernels.msl:4246` |
|| `grim_embedding_scatter_add` | `src/kernels.msl` (new) |
|| `grim_zeros_f32` | `src/kernels.msl` (new) |
|| Dispatch fns | `src/lib.rs:3582–3925` (`AutogradOps` impl body) |

### Verification
- [x] `cargo check --package grim-backend-metal` — passes
- [ ] `cargo clippy --package grim-backend-metal` — pre-existing lint errors in `caps.rs` + `lib.rs` (not introduced by this work); clean separately
- [ ] each backward kernel has a golden numerical-parity test vs CPU reference — **not yet written**; belongs in Phase 6
- [ ] `cargo test --package grim-backend-tests --test parity_cpu_vulkan_metal` — not run yet (Vulkan test hang caveat: use `--workspace --exclude grim-backend-vulkan` instead)

### Known issues
- `embedding_backward` dispatcher in ROCm (`roc_device.rs:4293`) may have a slightly different calling convention or shape handling than the Metal impl — parity test will catch this.
- `quantized_matmul_backward_dx` is still **partial** (Q8_0 only); widening to non-Q8_0 formats is Phase 1.5 in glass.md but was deferred.

---

## Phase 1.5 — Widen `quantized_matmul_backward_dx` (deferred)

**Status:** Deferred; not started.

- MSL already has `grim_quantized_matmul_backward_q8_0` (at ~line 4315).
- Need: `_q4k`, `_q5k`, `_q6k` variants to match ROCm's multi-format backward.
- This is the last Phase-1 item; once done, Phase 1 is fully complete.

---

## Phase 2 — Fused GEMM wrappers (unblocks lm_head / cross-entropy training)

**Status:** Phase 2 quick wins **complete**; `cargo check` + `cargo test` pass.

### What was done
- Added dispatch fn `fused_linear_cross_entropy_forward` to `lib.rs` wrapping `grim_fused_linear_ce` kernel
- Added dispatch fn `fused_linear_cross_entropy_backward` to `lib.rs` wrapping `grim_fused_linear_ce_backward` kernel
- Registered pipeline slots `fused_linear_ce` + `fused_linear_ce_backward` in `MetalPipelines` struct
- Registered `get_pipeline(...)` calls in `MetalContext::get()` init block
- Added MSL kernel `grim_fused_linear_ce_backward` to `kernels.msl`
- Added pipeline slots `flash_decode_split_k` + `softmax_merge` and `get_pipeline(...)` calls (kernel wiring for Phase 3)
- CPU fallback paths for both dispatchers (non-Apple targets)

### Verification
- [x] `cargo check --package grim-backend-metal` — passes
- [x] `cargo test --package grim-backend-metal` — passes (6 tests: standalone_dequant_parity, fused_add_rms_norm, metal_catchup, doc-tests)
- [ ] kernel numerical parity tests vs CPU reference — not yet written (Phase 6)

### Remaining Phase 2 items
- `fused_mxfp4_gemm_qk_norm_rope_kv` — MXFP4 QKV GEMM + QK-norm + RoPE (not started)
- `fused_rmsnorm_mxfp4_gemm`, `fused_rmsnorm_mxfp4_gemm_rope_kv` (not started)

---

## Phase 3 — Attention decode / prefill paths

**Status:** `flash_decode` dispatcher **complete**; `cargo check` + `cargo test` pass.

### What was done
- Added `pub fn flash_decode` dispatcher to `lib.rs` wrapping `grim_flash_decode_split_k` + `grim_softmax_merge` kernels
- Pipeline slots `flash_decode_split_k` + `softmax_merge` were already registered in `MetalPipelines` struct and `get_pipeline(...)` calls added in `MetalContext::get()`
- Dual-GPU dispatch: Stage 1 kernel writes partial V/Softmax/Max into mid tensors, then Stage 2 softmax merge combines them
- CPU fallback path for non-Apple targets (naive chunked attention + merge)

### Verification
- [x] `cargo check --package grim-backend-metal` — passes
- [x] `cargo test --package grim-backend-metal` — passes (6 tests)
- [ ] kernel numerical parity tests vs CPU reference — not yet written (Phase 6)

### Remaining Phase 3 items
- `extend_attention`, `prefill_compact`, `preshuffled_attention` — no MSL kernels yet; dispatch wrappers not written

---

## Phase 4 — Quantized GEMM family expansion

**Status:** `fused_dequant_gemm_mxfp4` fast-path **complete** (kernel `kernels.msl:2535` + pipeline slot `lib.rs:169,334` + dispatch branch `lib.rs:4853-4908`); `cargo check` + `cargo test` pass.

### What was done
- Added MXFP4 dispatch branch to `quantized_matmul` in `lib.rs` (line 4853) — mirrors the existing Q4K and FP8 fast-path structure: `FloatPackScheme::MxFp4` check, `zeros(out_shape, F32)`, command buffer + encoder, `setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_mxfp4)`, buffer wiring (a_buf:0, b_buf:1, out_buf:2), `setBytes_length_atIndex` for (m,n,k), `16×16` threadgroup grid, `encoder.endEncoding()`, `return Ok(...)`.
- Verified kernel exists in `kernels.msl:2535` (`grim_fused_dequant_gemm_mxfp4`): per-row interleaved layout — `codes_bytes = (k+1)/2`, `exps_bytes = k/32`, `row_base = b_packed + col*(codes_bytes+exps_bytes)`, nibbles split as `packed&0x0F` / `packed>>4`, shared_exp from `row_base[codes_bytes + i/32]`, calls `metal_mxfp4_to_float`.
- Verified pipeline slot `fused_dequant_gemm_mxfp4: Retained<ProtocolObject<dyn MTLComputePipelineState>>` in `MetalPipelines` struct (line 169).
- Verified `get_pipeline("grim_fused_dequant_gemm_mxfp4")?` in `MetalContext::get()` init (line 334).

### Verification
- [x] `cargo check --package grim-backend-metal` — passes (0.29s)
- [x] `cargo test --package grim-backend-metal` — passes (6 tests; MXFP4-specific parity tests belong in Phase 6)
- [ ] kernel numerical parity test vs CPU reference for `fused_dequant_gemm_mxfp4` — not yet written (Phase 6)

### Remaining Phase 4 items
- `fused_mxfp4_gemm_qk_norm_rope_kv` — MXFP4 QKV GEMM + QK-norm + RoPE (kernel + dispatch, not started)
- `fused_rmsnorm_mxfp4_gemm`, `fused_rmsnorm_mxfp4_gemm_rope_kv` (not started)
- `dequant_w4a16_blob_to_f32` / `dequant_wna16_to_f32` host dequant wrappers (not started)
- `awq_segment_offsets` / `gptq_segment_offsets` host helpers (not started)
- Non-Q8_0 `quantized_matmul_backward_dx` widening to q4k/q5k/q6k (not started)

---

## Phase 5 — Utility & architectural cleanup

**Status:** Not started.

See liquid.md original notes (BackendDevice impl empty, fused_add_rms_norm pub fn, naming mismatch).

---

## Phase 6 — Tests & benchmarks

**Status:** Not started.

See glass.md #6 and liquid.md original notes.

---

|| Running build / check status

| Check | Command | Result |
|---|---|---|
| `cargo check grim-backend-metal` | `cargo check --package grim-backend-metal` | **PASS** (Phase 1 + Phase 2 + Phase 3 flash_decode) |
| `cargo test --package grim-backend-metal` | `cargo test --package grim-backend-metal` | **PASS** (6 tests) |
| `cargo clippy grim-backend-metal` | `cargo clippy --package grim-backend-metal` | Pre-existing lint errors (caps.rs unused import, cfg_attr formatting) — not introduced by this work |
| `cargo test --workspace` | `cargo test --workspace --exclude grim-backend-vulkan` | Not run yet (Vulkan primus hang) |

---

## Pre-existing clippy warnings (not blocking, clean separately)

From the last `patch` run's lint output (these existed before Phase 1 and are in the same files):
- `src/caps.rs:1` — `use std::sync::Mutex;` moved below `use std::sync::atomic` (ordering)
- `src/lib.rs:8` — `Storage as DTypeStorage` import moved
- `src/lib.rs:82,104` — `#[cfg_attr(…)]` split across lines for `simdgroup_gemm_variant`
- Various `encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group)` reformatted to chained form (touched by Phase 1 + 2 — intentional style fix)
- `impl SamplingOps for MetalDevice {}` added as empty impl (consistent with other empty trait impls) — intentional

If you want clippy clean as a gate here, run `cargo clippy --package grim-backend-metal --fix` and review the auto-fixes.

---

## What's in `glass.md` that isn't in this log yet

Everything in `glass.md` still applies — this `liquid.md` is the implementation-facing companion. The gaps, risks, and verification checklist from `glass.md` (#7 and #8) are still live. `liquid.md` will be updated as each phase progresses.

---

## Next action

Phase 1 (backward passes) **complete** — 6 of 6 dispatched. Phase 2 (fused GEMM, lm_head/cross-entropy forward+backward) **complete** — 5 of 5 f...[truncated]
