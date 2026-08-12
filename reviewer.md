# exploded.md Validation Review

> Reviewed against `/D/rex/projects/grim` HEAD `05b8218`.
> `exploded.md` was not present in the working tree at review time;
> the plan content was reconstructed from the prior conversation summary.

---

## 1. Summary Verdict

The plan's **"what already exists" section is largely correct about capabilities** — the infrastructure it describes (C2plrController, PlacementCache, ScytheRing, ScythePlacement, GpuCapability, WaveTune, Lagrangian dual ascent) **does exist in the codebase**. However, the plan's **"what was split from" premise is fictional**: it claims these were extracted from standalone files (`schedule.rs`, `peer_mapping.rs`, `ring_buffer.rs`, `arch.rs`) that **never existed** in this project. The work was always done inline in `scythe2.rs` and `backend.rs`.

The **"what is still missing" section mixes genuine gaps with items that already exist under different names**. The plan calls for `charon_wmma.rs`, `fp8_wmma.rs`, `mxfp8_wmma_dequant.rs`, and `mxfp_weights.rs` as new kernel files, but functionally similar kernels already exist as `wmma_gemm.rs`, `fp8_gemm_rdna4.rs`, `mxfp_standalone.rs`, and `fused_dequant_gemm.rs`. The genuine gaps are `charon_backward.rs` (backward pass) and the MXFP8 quantization kernel (`mxfp_weights.rs`).

**Bottom line:** the plan's intent (WMMA-accelerated fused MoE dispatch, proper quantization integration, backward pass) is valid, but its gap analysis is based on a fictional file layout. Before implementing new files, the existing WMMA/quant infrastructure must be audited for overlap.

---

## 2. Section-by-Section Validation

### 2.1 "What scythe2.rs Already Ships" — Mostly Correct, Wrong File Attribution

| Plan Claim | Actual Location | Status |
|---|---|---|
| `C2plrController` — per-layer placement/partition/route, WaveTune bilinear eval, Lagrangian budget, online MLP gradient | `crates/grim-engine/src/scythe2.rs:188` | **EXISTS** — 861-line file contains all of this |
| `PlacementCache` — fast-path array + HashMap, epoch invalidation | `crates/grim-engine/src/scythe2.rs:65` | **EXISTS** |
| `ScytheRing` + `ScytheTaskDescriptor` — lock-free VRAM ring, 64-byte descriptors | `crates/grim-engine/src/scythe2.rs:563,594` | **EXISTS** |
| `ScytheLink` (`PeerDirect`/`Pcie`/`Host`) — P2P link classification | `crates/grim-tensor/src/backend.rs:27` | **EXISTS** — also `RouteLink` in `p2p_route.rs:45` |
| `ScythePlacement` — per-layer GPU assignment + route selection | `crates/grim-tensor/src/backend.rs:50` | **EXISTS** |
| `GpuCapability` — tflops/hbm/vram/throttle | `crates/grim-tensor/src/backend.rs:9` | **EXISTS** |
| `CapabilityProfiler` | `crates/grim-backend-rocm/src/device/capability_profiler.rs:37` | **EXISTS** |

**Inconsistency:** The plan claims these were "split across `schedule.rs` + `peer_mapping.rs` + launch planner" and that `ring_buffer.rs` "was proposing to build this from scratch." **None of these files exist in the project.** The infrastructure was always in `scythe2.rs` (861 lines, engine crate) and `backend.rs` (1570 lines, tensor crate). The plan's narrative of "replacing duplicated work" is based on a file layout that was never real.

**Inconsistency:** The plan says `GpuCapability` had "partial overlap with `arch.rs`." There is no `arch.rs` in `grim-backend-rocm` or any `grim-architecture` crate. `GpuCapability` lives in `grim-tensor/src/backend.rs` and is consumed by `CapabilityProfiler` in `capability_profiler.rs`. The "overlap" the plan imagines does not exist.

### 2.2 "What Is Still Genuinely Missing" — Mixed Findings

| Plan Requested File | Status | Existing Overlap | Notes |
|---|---|---|---|
| `charon_wmma.rs` — fused WMMA forward | **DOES NOT EXIST** | `wmma_gemm.rs` (generic WMMA GEMM), `fp8_gemm_rdna4.rs` (RDNA4 FP8 GEMM) | These are generic GEMM kernels, not MoE-specific fused dispatch. **Genuine gap**, but scope needs clarification. |
| `charon_backward.rs` — WMMA backward | **DOES NOT EXIST** | None | **Genuine gap** — no backward kernels exist anywhere. |
| `fp8_wmma.rs` — native FP8 WMMA, gfx1200+ | **DOES NOT EXIST** | `fp8_gemm_rdna4.rs` (gfx1200+ FP8 GEMM), `fp8_standalone.rs` | These provide FP8 GEMM but NOT fused MoE dispatch. The plan's WMMA-specific fused variant is genuinely missing. |
| `mxfp8_wmma_dequant.rs` — MXFP8 → BF16 fallback | **DOES NOT EXIST** | `mxfp_standalone.rs` (dequant kernels), `fused_dequant_gemm.rs` (fused dequant+GEMM) | Dequantization exists but NOT fused into MoE dispatch. **Partial overlap** — the dequant logic can be reused but the MoE fusion is missing. |
| `mxfp_weights.rs` — MXFP8 quantize kernel | **DOES NOT EXIST** | `quant_standalone.rs` | **Genuine gap** — quantization kernel for converting F32 weights to MXFP8 format. |

**Key inconsistency:** The plan's "why it must live in `grim-backend-rocm/src/kernels`" column says "GPU kernel. Engine cannot emit HIP WMMA instructions." This is correct for new fused MoE kernels, but the plan doesn't acknowledge that `wmma_gemm.rs` already provides WMMA infrastructure (rocWMMA include, fragment setup, 16×16 tiles) that `charon_wmma.rs` would need to build on top of. A new file would duplicate the rocWMMA include and fragment boilerplate unless it imports from `wmma_gemm.rs`.

### 2.3 Charon Kernel Source Audit (`charon.rs`, 85 KB)

The existing `charon.rs` already ships **7 kernel variants** in its `KERNEL_SOURCE`:

| Kernel | Quant | Structure | Status |
|---|---|---|---|
| `grim_moe_fused_dispatch` | FP32 | Sortless (one block per token-expert pair) | Implemented |
| `grim_moe_fused_grouped` | FP32 | Grouped (token-sorted by expert) | Implemented |
| `grim_moe_fused_grouped_fp8` | FP8 E4M3 | Grouped + inline dequant | Implemented |
| `grim_moe_fused_grouped_mxfp4` | MXFP4 E2M1+E8M0 | Grouped + inline dequant | Implemented |
| `grim_moe_fused_grouped_mxfp8` | MXFP8 E4M3+E8M0 | Grouped + inline dequant | Implemented |
| `grim_moe_fused_grouped_q80` | Q8_0 (GGUF) | Grouped + inline dequant | Implemented |
| `grim_moe_fused_grouped_iqk` | IQ/K-quant (12 formats) | Grouped + inline dequant | Implemented |

**Inconsistency with plan:** The plan says `charon_wmma.rs` should provide "fused WMMA forward" and `fp8_wmma.rs` should provide "native FP8 WMMA" gated on gfx1200+. But the existing `charon.rs` kernels already implement the same fused dispatch math (gate+up SiLU combine, down projection, atomicAdd accumulation) for FP32, FP8, MXFP4, MXFP8, Q8_0, and IQ/K-quant. The WMMA versions the plan calls for would be **WMMA-accelerated variants of the same kernels**, not entirely new functionality. The plan doesn't make this distinction clear.

### 2.4 Roundtrip Helper (`charon_fused_dispatch_roundtrip`)

The plan's Step 2 asks for a `pub fn charon_fused_dispatch_roundtrip` on `RocmDevice`. The current `roc_device.rs` has a `pub(crate)` launcher but not a `pub` roundtrip. **This is genuinely missing** and needed for integration tests from outside the crate.

### 2.5 Golden Test File

The plan's Step 3 asks for `tests/golden_charon_moe_gpu.rs` with 3 tests. A file with this name already exists at `crates/grim-backend-rocm/tests/golden_charon_moe_gpu.rs` (36 lines modified in the last commit). **Partially exists** — needs audit to confirm it matches the plan's 3-test specification.

---

## 3. Items That Need Modification to Meet Plan Intent

### 3.1 High Priority

| Item | Current State | Required Change |
|---|---|---|
| `charon_wmma.rs` | Does not exist | Create WMMA-accelerated fused MoE dispatch variants. Must reuse rocWMMA setup from `wmma_gemm.rs` to avoid duplicating fragment boilerplate. |
| `charon_backward.rs` | Does not exist | Create backward pass kernels (gradient computation for MoE). No existing code to build on. |
| `fp8_wmma.rs` | Does not exist (overlap with `fp8_gemm_rdna4.rs`) | Either extend `fp8_gemm_rdna4.rs` with fused MoE dispatch, or create new file that imports WMMA primitives from `wmma_gemm.rs`. |
| `mxfp8_wmma_dequant.rs` | Does not exist (overlap with `mxfp_standalone.rs`) | Reuse dequant device functions from `mxfp_standalone.rs` (`fp8e4m3_to_f32`, `mxfp4_e2m1_to_f32`) in a new fused MoE WMMA kernel. |
| `mxfp_weights.rs` | Does not exist | Create MXFP8 quantization kernel (host-side weight conversion). |
| `charon_fused_dispatch_roundtrip` | `pub(crate)` only | Promote to `pub` on `RocmDevice` for test access. |

### 3.2 Medium Priority

| Item | Current State | Required Change |
|---|---|---|
| `schedule.rs`, `peer_mapping.rs`, `ring_buffer.rs`, `arch.rs` | Never existed | **No action needed** — these were never standalone files. The plan's premise that they "would need to be modified" is based on a fictional past. |
| `PeerMapping` type | Does not exist | The plan references `PeerMapping` as a distinct type from `ScytheLink`. Currently `ScytheLink` + `RouteLink` serve this role. If the plan wants a separate `PeerMapping` abstraction, it needs to be created; otherwise the plan should be updated to use existing types. |
| `CapabilityProfiler` | Exists in `capability_profiler.rs` | The plan mentions "Partial overlap with `arch.rs`" for `GpuCapability` — this reference should be updated to point to `capability_profiler.rs`. |
| `P2P link classification` | Implemented as `RouteLink`/`to_route_link()` in `p2p_route.rs` | The plan says this "would need to be modified." It already exists and works. The plan should reference `RouteLink` instead of the fictional `PeerMapping`. |

### 3.3 Low Priority / Cleanup

| Item | Current State | Required Change |
|---|---|---|
| `scythe2.rs` (engine) vs `scythe2.rs` (nn) | Two files with same name | These serve different purposes (controller vs. layer dispatch). The plan's confusion about file layout may stem from this naming collision. Consider renaming `crates/grim-nn/src/scythe2.rs` to `scythe2_dispatch.rs` or similar to avoid ambiguity. |
| Charon kernel source string | All 7 kernels in one 790-line string literal | If new WMMA variants are added, consider splitting `KERNEL_SOURCE` into per-kernel constants or separate files to avoid a 2000-line string literal. |

---

## 4. Inconsistencies in Plan Premises

### 4.1 Fictional File Layout

The plan's entire "What scythe2.rs Already Ships" section is built on the premise that infrastructure was "split from" files that never existed:

> "Was split across `schedule.rs` + `peer_mapping.rs` + launch planner"
> "`ring_buffer.rs` (WI-H) was proposing to build this from scratch"
> "`peer_mapping.rs` was duplicating this"
> "Launch planner was reinventing this"
> "Partial overlap with `arch.rs`"

**None of these files exist in the project's current or historical visible code.** The infrastructure was always in `scythe2.rs` (engine) and `backend.rs` (tensor). The plan should be corrected to say "this infrastructure lives in `scythe2.rs`" rather than "this was extracted from files that never existed."

### 4.2 WMMA Gap Analysis Is Incomplete

The plan lists `charon_wmma.rs` and `fp8_wmma.rs` as missing, but doesn't audit the existing `wmma_gemm.rs` (generic WMMA GEMM) or `fp8_gemm_rdna4.rs` (RDNA4 FP8 GEMM). These files provide the WMMA primitive layer that new fused MoE kernels would build on top of. The plan should either:

1. Specify that `charon_wmma.rs` should import/reuse `wmma_gemm.rs` primitives, or
2. Clarify that it's asking for MoE-specific WMMA fused dispatch (different from generic GEMM)

### 4.3 Quantization Path Overlap Not Acknowledged

The plan says `mxfp8_wmma_dequant.rs` must provide "MXFP8 → BF16 fallback" and `mxfp_weights.rs` must provide "MXFP8 quantize kernel." But:

- `mxfp_standalone.rs` already provides `grim_dequant_mxfp4` and `grim_dequant_mxfp8` device functions
- `quant_standalone.rs` exists (purpose unverified but likely provides host-side quantization)
- The dequant device functions in `mxfp_standalone.rs` are reusable in a fused MoE context

The plan doesn't acknowledge this overlap, which could lead to code duplication if new files are created without referencing existing code.

### 4.4 "Partial overlap with arch.rs" Is Wrong

The plan says `GpuCapability` had "partial overlap with `arch.rs`." The actual situation:

- `GpuCapability` is defined in `crates/grim-tensor/src/backend.rs:9`
- `CapabilityProfiler` is in `crates/grim-backend-rocm/src/device/capability_profiler.rs:37`
- There is no `arch.rs` in `grim-backend-rocm` or any `grim-architecture` crate
- The closest analog is `accel_features.rs` (MFMA capability detection) which is unrelated to `GpuCapability`

The plan should reference `capability_profiler.rs` instead of the fictional `arch.rs`.

---

## 5. Recommendations

1. **Correct the plan's file layout premise.** Remove references to `schedule.rs`, `peer_mapping.rs`, `ring_buffer.rs`, and `arch.rs` as source files. Update to say "infrastructure lives in `scythe2.rs` and `backend.rs`."

2. **Audit existing WMMA/quant files before creating new ones.** Before creating `charon_wmma.rs`, `fp8_wmma.rs`, or `mxfp8_wmma_dequant.rs`, review `wmma_gemm.rs`, `fp8_gemm_rdna4.rs`, `mxfp_standalone.rs`, and `fused_dequant_gemm.rs` for reusable code. New files should compose from existing primitives, not duplicate them.

3. **Clarify `charon_wmma.rs` scope.** The plan should specify whether this is (a) a WMMA-accelerated variant of the existing fused MoE dispatch (replacing the current inline GEMM loops with WMMA fragments), or (b) a separate GEMM utility. These are different scopes.

4. **Create `charon_fused_dispatch_roundtrip` as planned.** This `pub` helper is genuinely needed for integration tests from outside the crate.

5. **Create `charon_backward.rs` and `mxfp_weights.rs` as genuine new work.** These have no existing overlap.

6. **Update `PeerMapping` references.** The plan should use `ScytheLink`/`RouteLink` (which exist) rather than introducing a new `PeerMapping` type, unless there's a specific reason for a separate abstraction.

7. **Consider renaming `scythe2.rs` in `grim-nn`.** The dual `scythe2.rs` files (engine vs nn) cause confusion about which file contains what. A rename would clarify the architecture.

---

## 6. Conclusion

The plan's **intent is sound** — WMMA-accelerated fused MoE dispatch, proper backward kernels, and MXFP8 quantization are real gaps. But its **gap analysis is based on a fictional file history** and **doesn't audit existing WMMA/quant infrastructure**. Before implementing, the plan should be revised to:

1. Remove references to files that never existed (`schedule.rs`, `peer_mapping.rs`, `ring_buffer.rs`, `arch.rs`)
2. Document overlap with existing files (`wmma_gemm.rs`, `fp8_gemm_rdna4.rs`, `mxfp_standalone.rs`, `fused_dequant_gemm.rs`)
3. Clarify the exact scope of each new file (MoE-specific fused dispatch vs. generic GEMM utility)
4. Update type references to use existing types (`ScytheLink`, `RouteLink`, `CapabilityProfiler`) rather than fictional ones (`PeerMapping`)

The infrastructure the plan describes **already exists** in `scythe2.rs` (861 lines, engine crate) and `backend.rs` (1570 lines, tensor crate), plus `capability_profiler.rs` (ROCM device crate). The remaining work is genuine kernel development, not "replacing duplicated work."
