# Implementation Plan: dot2 / dot4 / sudot4 MoE Kernels in Charon (RDNA Backend)

Target: make `grim_sdot4`-family GEMV kernels (dot2 f32/f16, dot4 q8_0/int8, q4_K, and `sudot4` on RDNA3/4) the kernels of choice for Charon MoE grouped dispatch on the RDNA backend, in q8_0/int8 and q4_K quantizations.

## 0. Ground truth: what already exists

The survey of `crates/grim-backend-rocm` shows the MoE side of this plan is **mostly implemented**. Do not re-write these:

| Existing | Location | Status |
|---|---|---|
| `grim_sdot4()` ISA selection — `__builtin_amdgcn_sdot4` (RDNA2), `sudot4` (RDNA3/4, sdot4 removed on RDNA4) | `crates/grim-backend-rocm/src/kernels/dot_gemv.rs:10-17` | Keep as-is |
| `grim_dot4_q80_q81_gemv`, `grim_dot2_f32_f16`, `grim_dot4_q2k..q6k_q81_gemv` incl. **q4_K** (`dot_gemv.rs:680`) | `src/kernels/dot_gemv.rs` | Keep; extend |
| W4A4 `sudot8` GEMV (gfx1200/1201 only) | `dot_gemv.rs:1049-1210` | Keep |
| Rust launchers `launch_dot4_q80_q81_gemv` … `launch_sudot8_*` | `src/device/device_compute.rs:2286-2695` | Keep; add MoE wrappers |
| `quantized_matmul` dispatch with per-`QuantMode` branches | `src/device/device_quant.rs:32` (q4k @117, q8_0 @426) | **Modify** — MoE path |
| Charon fused grouped MoE: `grim_moe_fused_grouped_{fp8,mxfp4,mxfp8,q80,iqk,w8a8_int8,w8a8_fp8,awq}` | `src/kernels/charon.rs:401-1274` | **Modify** — q80 and int8 variants exist; add q4_K variant |
| Routing: `RoutingAssignment` (`charon.rs:1383`), `SortedRouting` (1460), `moe_align_block_size` (1475), `CharonLaunchPlan` (1552), `CharonVariant` (1751), `CharonSelector` (1974) | `src/kernels/charon.rs` | Keep |
| MoE grouped routing launchers | `src/device/device_routing.rs:105,259,312,513` | **Modify** — variant wiring |
| WMMA MoE competitor variant | `src/kernels/charon_wmma.rs:18,105` | Keep (benchmark baseline) |
| Quant format enums `QuantFormat` | `crates/grim-tensor/src/dtype.rs:160`; backend `QuantMode` `src/quantization.rs:187` | Keep; ensure Q4K variant maps through |
| Build/FFI: links `amdhip64`, `rocblas`, `hiprtc` | `crates/grim-backend-rocm/build.rs:36-38`; JIT include paths `src/device/util.rs:265`, `build_rocm_detect.rs:100-143` | Keep |
| GPU parity test harness + shared KATs (`TEST_QUANT_FORMATS` Q8_0/Q4K/Q5K/Q6K over K∈{256..4096}) | `crates/grim-backend-rocm/tests/`, `crates/grim-backend-tests/src/lib.rs` | **Extend** |

Known risks already flagged in-tree: rocwmma kernels can be excluded via TEMP-DIAG flag in `src/kernels/source_asm.rs:8` (a "GGUF fault hunt" leftover — verify it is off); `sdot4` is removed on RDNA4 so all q8_0/int8 paths must select `sudot4` there (`dot_gemv.rs:16-17`).

## 1. Changes

### 1.1 Add a q4_K Charon grouped MoE kernel (the main gap)
- **File:** `crates/grim-backend-rocm/src/kernels/charon.rs`
- **Edit:** Add `grim_moe_fused_grouped_q4k` next to `grim_moe_fused_grouped_q80` (line 640). Reuse the `grim_sdot4()` dequant-dot inner pattern from `dot_gemv.rs:680` (`grim_dot4_q4k_q81_gemv`) inside the grouped expert loop: per-expert base pointer from `SortedRouting`, superblock (QK_K=256, 8 sub-blocks of 16) unpack, `sudot4/sdot4` on the i8 sub-blocks with per-sub-block `d`/`dmin` scaling accumulated in i32 then scaled in f32. Keep the routing/align structure identical to the q80 variant so `CharonLaunchPlan` is unchanged.
- **Why:** q4_K is in `TEST_QUANT_FORMATS` and dispatched in `quantized_matmul`, but Charon has no q4_K grouped variant — dense dot4 q4k exists, MoE does not.

### 1.2 Register the kernel
- **File:** `crates/grim-backend-rocm/src/kernels/source_asm.rs`
- **Edit:** Nothing if `KERNEL_SOURCE` already aggregates `charon::KERNEL_SOURCE` — verify and only add the new kernel to the `charon.rs` source string. Also confirm the TEMP-DIAG exclusion at line 8 is disabled for release builds.
- **Verification:** `grep -c grim_moe_fused_grouped_q4k src/kernels/source_asm.rs` (or its `include!`) > 0.

### 1.3 Launcher + dispatch
- **Files:** `crates/grim-backend-rocm/src/device/device_compute.rs` (new `launch_moe_fused_grouped_q4k`, modeled on existing grouped launchers), `src/device/device_quant.rs` (`quantized_matmul` at line 32: add `QuantMode::Q4K` MoE branch), `src/device/device_routing.rs` (`grim_moe_fused_grouped` at 259 and `CharonSelector` variant table in `charon.rs:1844`: add the q4k `CharonVariant` with `grouped_dispatch_entry` at 1762).
- **Why:** selection is by `match` on `QuantMode`/`GcnArch`, not a trait registry — every new variant needs all three touch points.

### 1.4 Make dot4/sudot4 the kernels of choice (selection policy)
- **File:** `src/kernels/charon.rs` — `CharonSelector` (1974) + `WaveCostModel` (1790)
- **Edit:** For decode (GEMV-shaped, tokens-per-expert small) on RDNA2/3/4, prefer dot4-family variants over WMMA; keep WMMA preferred for prefill/large `m`. Gate: `sudot4` requires RDNA3+ (`gcn_arch` probe in `src/device/hardware_spec.rs`); RDNA2 falls back to `sdot4`; RDNA4 must use `sudot4` (sdot4 opcode removed). Encode this in the existing `grim_sdot4()` gfx guard and assert it in the selector.
- **File:** `src/device/gemm_tuning.rs` / `src/autotune.rs`
- **Edit:** Add dot4 variants to the autotune candidate list so "kernel of choice" is measured, not hardcoded, and record the winner in the tuning table.

### 1.5 dot2 path: keep, but scope it
- `grim_dot2_f32_f16` (`dot_gemv.rs:209`, inline asm) is for f32/f16 activations, not quantized MoE. **Keep unchanged**; only wire it as fallback when `sudot4/sdot4` probe fails (`grim_sdot4_probe`, line 1215).

### 1.6 Training path: backward grads must follow (llm-training intent)
Grim is an inference *and* training engine. A new forward kernel without a matching grad path forks precision between fwd/bwd.
- **File:** `crates/grim-backend-rocm/src/kernels/charon_backward.rs`
- **Edit:** expert-weight grads for q4_K experts must consume the *same* dot4 dequant inner loop (shared `#include`-style common source in `dot_gemv.rs`), not a separate f32 dequant path. If dot4 grouped fwd is selector-gated for training runs, either add dot4 backward or pin training runs to WMMA variant — decide once, encode in `CharonSelector` via a `training: bool` flag. i32 dot accumulators (sdot4/sudot4 output) are exact — good for grad accumulation; do not add fp32 emulation.
- **Reuse check (ponytail):** no new routing structs, no new launch-plan types. q4k variant reuses `RoutingAssignment`/`SortedRouting`/`CharonLaunchPlan` verbatim.

### 1.7 FFI: constraints, not changes (rust-ffi rules apply to review)
- Link-time `amdhip64`/`hiprtc`/`rocblas` in `build.rs:36-38` stays. New kernel is JIT source — zero new FFI surface.
- Review gates for touched code: every HIP call wrapped in status check (`hip_check` pattern); host-side rocBLAS params use ABI `rocblas_stride` = i64 as-is; **device kernel index math stays 32-bit** (RDNA waves32, 32-bit regs — 64-bit index math wastes reg pairs and s_mul_u64), one 64-bit base-pointer add at the end; launcher asserts total expert tensor bytes < 2^31 before using i32 offsets, else falls back to existing non-dot4 path. No panic across FFI in `launch_*` wrappers (return `Result`); SAFETY comments on all `unsafe` hipRTC launches.
- If RDNA4 needs newer hipRTC codegen for `sudot4`, bump only `--offload-arch` propagation in `build_rocm_detect.rs:100-143`. Do not introduce dlopen/SONAME probing — link-time already fails fast and grim pins per-arch toolchains.

### 1.8 Removals / cleanups
- None of the existing dot kernels should be removed. Remove only the TEMP-DIAG rocwmma exclusion switch in `source_asm.rs:8` once the fault hunt is closed.

## 2. Tests

### Unit tests (no GPU, compile-time + logic)
- **File:** `crates/grim-backend-rocm/src/kernels/dot_gemv.rs` (extend `#[cfg(test)]` block at 1239) and `charon.rs`
- Assertions: q4k MoE source contains `sudot4`/`sdot4` guard; `CharonVariant` table contains q4k entry with correct gfx gating; RDNA4 ⇒ `sudot4`, RDNA2 ⇒ `sdot4` string selection.
- **File:** `crates/grim-tensor/src/dtype.rs` — Q4K round-trip block layout constants (QK_K, scales) match kernel assumptions.

### Parity / regression tests (GPU-gated, existing pattern)
- **File:** `crates/grim-backend-rocm/tests/charon_dot4_q4k_grouped_parity.rs` (new) — mirror `charon_wmma_parity.rs`: random routed tokens, compare `grim_moe_fused_grouped_q4k` against CPU reference (`grim-backend-cpu/src/dequant_gemm.rs` as golden), tolerance per `grim-backend-tests` KATs (K ∈ {256..4096}, formats Q8_0/Q4K).
- **File:** `crates/grim-backend-rocm/tests/dot_gemv_parity.rs` (extend) — add gfx-matrix cases: run q8_0/int8 GEMV on probed arch and assert `sudot4` path taken on RDNA3/4 (via a flag kernel or `grim_sdot4_probe`).
- **Regression gate:** `tests/moe_quant_wiring_gate.rs` — add q4k to the wiring gate so MoE q4_K silently falling back to dequant-f32 is a test failure.

### Integration tests
- **File:** `crates/grim-backend-rocm/tests/golden_charon_moe_gpu.rs` (extend) — end-to-end MoE layer with Q8_0 and Q4_K experts; compare token outputs vs WMMA variant (agreement within tolerance) and vs stored golden.
- **File:** `crates/grim-backend-rocm/tests/charon_dot4_selector_gate.rs` (new) — `CharonSelector` picks a dot4 variant for decode shapes and WMMA for prefill shapes on each probed arch; HIP-graph capture of the q4k grouped path (`graph_capture.rs`) succeeds.

### Performance verification
- Use `scripts/parity-vs-ollama.sh` + `src/device/capability_profiler.rs` microbench: report tok/s and kernel µs for dot4 vs WMMA vs rocBLAS at decode batch 1/2/4, experts 8/64. "Kernel of choice" claims must be backed by these numbers checked into `gemm_tuning.rs`.

### CI / commands
- `cargo test -p grim-backend-rocm --test charon_dot4_q4k_grouped_parity -- --ignored --nocapture` (GPU-gated tests follow existing `--ignored` convention — verify convention in the test files).
- `cargo test -p grim-backend-rocm` on a GPU machine as the merge gate; `cargo mutants` already configured via root `mutants.toml` — add the new dispatch branches to its scope.

## 3. Sequencing

1. Selector policy + probes (1.4, 1.5) — lowest risk, immediately makes existing dot4 q8_0/q4k GEMV the decode path.
2. q4_K Charon grouped kernel (1.1–1.3) with parity tests.
3. Backward/training decision (1.6) — blocks any training-mode enablement.
4. Integration/golden tests + autotune candidates (1.4, §2).
5. Cleanup TEMP-DIAG (1.8) after fault hunt closure.
