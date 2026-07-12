# FinalRide — Post-Pass-4 Cleanup Plan

## Overview

**Status of the codebase at the start of `finalride.md`**: Pass 4 (`pass4.md`) of the oxidizer is **complete** and green. grim's Garage UI (`crates/grim-garage`) is **complete** and E2E-verified live (101 tests green pre/post CVKG bump). grim-implementation-plan phases 1–3 are **landed**. The P2 fixes from `fix.md` are merged (install.sh, spec.rs UX).

This plan captures the **remaining technical debt and small spec gaps** discovered during the post-bump inspection of the codebase on 2026-07-05. Each entry has been checked against the actual source — no items are speculative.

**Naming**: "FinalRide" because the next round of fixes — once they pass — should close the oxidizer → garage pipeline cleanly enough that the user can train on their AMD GPU end-to-end without hitting stubs, no-ops, or unsound FFI.

---

## Constraints (Reaffirmed)

- Strict RED→Green TDD on every code change.
- Native Rust CVKG components only — no React/Vite/Tailwind.
- grim's Garage: backend uses `cvkg_webkit_server::router::create_router` at CVKG **0.3.4**; UI shaders/native panels all CVKG 0.3.x.
- All CVKG 0.3.x pins at the same version (currently 0.3.4).
- No comments unless explicitly required by the code being changed.
- Workspace must stay `cargo check --workspace`-clean after every fix.

---

## Pre-flight Status (Verified 2026-07-05)

| Check | Result |
|---|---|
| `cargo check --workspace` | ✅ green (style warnings only, no errors) |
| Pass 4 all 7 targets | ✅ present in source |
| grim-garage UI panels | ✅ conforming to `cvkg-components-overview.md` |
| `WavefrontTiledLayout::tile/untile` | ⚠️ struct/impl exist in `grim-backend-rocm`; **not yet wired into `lora.rs:align_tensor_for_rocm_gemm`** |
| FP4/NF4/FP8 dequant functions | ✅ `dequant_fp4/nf4/fp8` implemented in `grim-quant`; **only `Storage::KQuant(*)` maps, no `Storage::Block` variant for these formats** |
| `grim-server` → `grim-engine` | ⚠️ SSE calls `engine.tick()`; **non-stream `chat_completions` path not yet plumbed to engine** |
| FFI `unsafe extern "C"` | ✅ `grim-backend-cuda:23`, `grim-backend-vulkan:121` fixed |
| ROCm `unsafe { set_var }` | ✅ lines 1586/1589/1591 already wrapped |

---

## Fix 1 — Wire `WavefrontTiledLayout` into LoRA alignment

**File**: `crates/grim-models/transformer/src/lora.rs`  
**Problem**: `align_tensor_for_rocm_gemm` is an identity no-op. ROCm GEMM kernels need wavefront-aware tensor padding.

**Spec** (from pass4.md): for non-trivial shapes (e.g. 70×60, 35×40), pad rows/cols to the next multiple of `wavefront_size` (typically 64) using `WavefrontTiledLayout::tile` from `grim-backend-rocm`. Shapes that are already wavefront-aligned pass through unchanged.

**Approach**:  
1. `lora.rs` is in `grim-models` (cannot depend on `grim-backend-rocm` directly — would be a circular dependency).  
2. Add `align_tensor_for_rocm_gemm` to `grim-tensor` or `grim-backend-rocm`'s public API as an architecturally neutral helper that takes `(rows, cols, wavefront_size) -> (padded_rows, padded_cols)`.  
3. Rewrite `lora.rs:align_tensor_for_rocm_gemm` to call that helper and repad the tensor bytes accordingly.  
4. Add RED tests asserting `(70, 60) → (128, 64)`, `(35, 40) → (64, 64)`, `(64, 64) → (64, 64)`.  
5. GREEN: round-trip `tile → untile` recovers original values.

**Verify**: `cargo test -p grim-models-transformer` passes; `cargo check --workspace` clean.

---

## Fix 2 — `Storage::Block` variant for FP4/NF4/FP8

**File**: `crates/grim-tensor/src/dtype.rs` (or wherever `Storage` enum lives) + `crates/grim-format/src/tprov.rs`  
**Problem**: `Storage::KQuant(KQuantScheme::Q4K)` is used for all K-quant types. But `dequant_fp4/nf4/fp8` are separate formats that need their own dispatch path.

**Spec** (from pass4.md): Add `Storage::Block(BlockDType)` where `BlockDType = Fp4 | Nf4 | Fp8`. `dtype_from_gguf` should map `GGUF_TYPE_Q4_K` → `Storage::Block(Fp4)`, etc., instead of `Storage::KQuant`. The `WeightSource::get_for_training` dispatch then calls `dequant_fp4/nf4/fp8` directly for those variants.

**Approach**:  
1. Add `pub enum BlockDtype { Fp4, Nf4, Fp8 }` and `Storage::Block(BlockDtype)` in `grim-tensor`.  
2. Update `dtype_from_gguf` mappings in `tprov.rs`.  
3. Add RED tests: `dtype_from_gguf(Q4_K) -> Block(Fp4)`, `dtype_from_gguf(Q8_0) -> KQuant(Q80)` (backward compat).  
4. GREEN: existing K-quant path unchanged; new block path calls `dequant_fp4/nf4/fp8` from `grim-quant`.

**Verify**: `cargo test -p grim-format` and `cargo test -p grim-quant` pass.

---

## Fix 3 — Plumb non-stream `chat_completions` to `engine.tick()`

**File**: `crates/grim-server/src/lib.rs`  
**Problem**: SSE path calls `engine.tick()` and reads `last_outcome()`. The non-stream path currently returns a static stub response.

**Spec**: Non-stream `chat_completions` should run `engine.tick()` in a loop until `done`, collecting `outcome.content()` tokens into a single `completion` string.

**Approach**:  
1. RED test: `POST /v1/chat/completions` with `"stream": false` and a short prompt returns non-empty completion.  
2. GREEN: reuse the existing tick/outcome machinery from the SSE path.  
3. Verify `cargo test -p grim-server` passes.

**Verify**: `cargo test -p grim-server`; E2E `curl -X POST ... -d '{"messages":[{"role":"user","content":"hi"}],"stream":false}'` returns a non-empty completion.

---

## Fix 4 — ponytail-audit of net-new code

**Files**:
- `crates/grim-garage/src/renderer_host/mod.rs`
- `crates/grim-garage/src/ui_state/poller.rs`
- `crates/grim-backend-rocm/src/fusion.rs`
- `crates/grim-backend-rocm/src/gptq_kernel.rs`

**Approach**: One-shot scan. Output a ranked list of findings — over-engineering, dead code, unnecessary dependencies, speculative abstractions. No fixes applied in this pass.

---

## Fix 5 — `grim-backend-rocm` module tests (HIP runtime required)

**File**: `crates/grim-backend-rocm/src/lib.rs` lines 1596+  
**Problem**: `probe_with_ordinal_override_returns_one_device` and `probe_without_hip_runtime_returns_empty_or_one` are gated on real HIP runtime and will be skipped on a non-ROCm host.

**Spec**: No code change needed — just ensure these tests are marked `#[test]` and properly gated (they already are, per the `hipSetDevice` failure mode). Confirm tests are discoverable: `cargo test -p grim-backend-rocm -- --list` shows them.

**Verify**: `cargo test -p grim-backend-rocm -- --list | rg probe` shows both tests listed (even if skipped on non-HIP host).

---

## Dependency Order

```
Fix 1 (WavefrontTiledLayout wire)
  └── Fix 2 (Storage::Block variant)      ← independent
       └── Fix 3 (chat_completions wire)   ← depends on engine tick being wired in pass4
Fix 4 (ponytail-audit)                     ← independent, runs after fixes 1–3
Fix 5 (module test listing)                ← independent
```

Fix 1 should land first because LoRA training on ROCm is the primary user story.

---

## Success Criteria

After all fixes land:
1. `cargo test --workspace` passes (HIP-gated tests may skip on non-ROCm hosts).
2. Non-stream `chat_completions` returns real tokens from `engine.tick()`.
3. `grim-garage` backend E2E: `POST /api/train/start` → job registered → `GET /api/train/jobs` returns it.
4. `align_tensor_for_rocm_gemm` does real wavefront padding — not identity.
5. ponytail-audit findings ranked and ready for a future sprint.