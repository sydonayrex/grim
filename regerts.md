# regerts.md — Backend Stub Audit & Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use writing-plans, vanity-engineering-review, and ponytail-audit patterns. This document is the source of truth for stub elimination across the three GPU backend crates.

**Goal:** Catalog every stub, dead path, and simulation-only operation in `grim-backend-rocm`, `grim-backend-cuda`, and `grim-backend-vulkan`, grade the vanity surface, and produce a prioritized kill/implement plan.

**Architecture:** Three crates implement the `BackendDevice`/`BackendStorage` traits from `grim-tensor`. rocm is the lead (real rocBLAS), cuda is early (real cuBLAS, square-only), vulkan is a shell (CPU-fallback matmul, no GPU dispatch).

**Tech Stack:** Rust, hipFFT/rocBLAS (rocm), cuBLAS (cuda), Vulkan FFI + GLSL/CubeCL (vulkan), `lazy_static`, `tokio` (optional).

## Global Constraints

- Tests run in CI; vulkan `vkCreateInstance` **hangs** on machines without a Vulkan loader. Tests for vulkan are **non-runnable in CI** today.
- rocm requires physical AMD GPU + rocBLAS installed; tests are green on a GPU host but not portable.
- cuda requires NVIDIA GPU + cuBLAS; square-only matmul limitation is documented but untested.
- `Error::Unimplemented` is the canonical "stub returns here" signal. Do not silently return `Ok(())` from a no-op — that hides stub status.
- Follow the honesty rule: name the pattern, show evidence, propose the simpler alternative.

---

## Phase 0 — Requirement Anchor

**Who uses this?** Inference engine (`grim-models`) calls `add → silu_mul → rms_norm → softmax → embedding` per transformer block. None of these resolve on *any* backend today.

**What must it do?** (1) Allocate/free device memory, (2) move tensors H↔D, (3) `matmul` via vendor BLAS, (4) elementwise + norm + softmax + embedding on device.

**What scale?** Single-GPU inference, batch ≤ 32, seq ≤ 4096. Not distributed.

**Team size?** Solo/small. No dedicated backend team.

**Hard truth:** No model can run end-to-end on any of the three backends. The `Unimplemented` ops form a hard wall. rocm can `matmul` but cannot `add`. vulkan can move bytes but cannot compute. cuda can `matmul` (square only) but cannot `add`.

---

## Phase 1 — Stub Inventory (Verified on Disk)

### grim-backend-rocm (`src/lib.rs` ~1913 lines, 32 tests green)

**Real GPU ops (implemented):**
| Op | Status | Evidence |
|----|--------|----------|
| `to_cpu_vec_f32` | real | `hipMemcpyDtoH` at lib.rs:803 |
| `zeros` | real | `hipMalloc` + `hipMemset` at lib.rs:951 |
| `from_cpu` | real | `hipMemcpyHtoD` at lib.rs:992 |
| `matmul` | real | `rocblas_sgemm` / `rocblas_gemm_ex` at lib.rs:1005 |
| `advise` | real | `hipMemAdvise` + XNACK fallback `hipMemcpyAsync` at lib.rs:1238 |

**Stubs (immediate `Err(Error::Unimplemented)`):**
| Op | Line | Message |
|----|------|---------|
| `add` | 1178 | "ROCM add pending hip/rocblas scalar/elementwise ops link" |
| `mul` | 1189 | "ROCM mul pending hip elementwise ops link" |
| `silu_mul` | 1200 | "ROCM silu_mul pending hip elementwise ops link" |
| `rms_norm` | 1212 | "ROCM rms_norm pending hip kernels" |
| `softmax` | 1222 | "ROCM softmax pending hip kernels" |
| `embedding` | 1233 | "ROCM embedding pending hip kernels" |

**Vanity / dead surface (within rocm):**
- `GemmTileConfig` + `lookup_gemm_config` (lib.rs:1301-1326) — **mock** Tensile autotune lookup. Returns hardcoded tile sizes by `wave % 128 == 0`. Never connected to real Tensile or rocBLAS `rocblas_gemm_algo`. V1 drag.
- `hipGraphLaunch` wrapper (lib.rs:1330-1332) — wraps one FFI call, no added value. yagni.
- HIP graph capture/replay block (lib.rs:1133-1158) — gated by `GRIM_CAPTURE_GRAPH` env var, never exercised in tests or CI. V2 structural.
- `rocblas_gemm_ex` path (lib.rs:1085-1104) — alternative sgemm behind `rocm-aiter` feature. Duplicates the `rocblas_sgemm` path. Divergent behavior risk. V1 drag.
- `println!` profiler spam — 9 `println!` calls gated `#[cfg(feature = "rocm-profile")]` (lines 953, 999, 1012, 1065, 1135, 1158, 1240, 1254, 1258). Noise; should be behind a real tracing subscriber, not raw `println!`.
- `HsacoKernelCache` (fusion.rs) — cache exists but kernel compilation via hsaco is not wired into any op path. Dead abstraction until a JIT path is added.

### grim-backend-cuda (`src/lib.rs` ~506 lines, 4 tests green)

**Real GPU ops:**
| Op | Status | Evidence |
|----|--------|----------|
| `to_cpu_vec_f32` | real | `cudaMemcpyDtoH` at lib.rs:212 |
| `zeros` | real | `cudaMalloc` + `cudaMemset` at lib.rs:327 |
| `from_cpu` | real | `cudaMemcpyHtoD` at lib.rs:351 |
| `matmul` | real (square-only) | `cublasSgemm_v2` at lib.rs:351, column-major hack |
| `advise` | **no-op stub** | `Ok(())` at lib.rs:506 |

**Stubs (immediate `Err(Error::Unimplemented)`):**
| Op | Line |
|----|------|
| `add` | 443 |
| `mul` | 453 |
| `silu_mul` | 463 |
| `rms_norm` | 474 |
| `softmax` | 483 |
| `embedding` | 493 |

**Issues:**
- **`matmul` is square-only** (documented): `transa=N, transb=N` with swapped `a`/`b` pointers compensates row↔col major only for `m==n`. Non-square inputs produce silently-wrong results. No test covers non-square. **V2 structural — a correctness landmine.**
- `advise` returns `Ok(())` — dishonest stub. rocm's `advise` is real. cuda should either implement `cudaMemAdvise` or return `Error::Unimplemented` to signal the gap.

### grim-backend-vulkan (`src/lib.rs` ~910 + `cube_kernels.rs` ~140, **tests hang**)

**Real GPU ops (memory only):**
| Op | Status | Evidence |
|----|--------|----------|
| `zeros` | real (host-visible) | `vkAllocateMemory` + map + memset at lib.rs:516 |
| `from_cpu` | real | `vkMapMemory` + memcpy at lib.rs:679 |
| `to_cpu_vec_f32` | real | `vkMapMemory` + readback at lib.rs:451 |
| `advise` | **no-op stub** | `Ok(())` at lib.rs:712 |

**Simulation-only (CPU fallback, NOT GPU dispatch):**
- `matmul` (lib.rs:545-623) — allocates real Vulkan buffers, maps them, then **CPU-loops** the GEMM. Comments: "Simulate SPIR-V hardware execution of tiling math" (lib.rs:605). The GLSL/CubeCL/SPIR-V pipeline is defined but **never dispatched**. This is Pattern 36 (Simulation-Only Execution Engine) at the GPU layer.

**Stubs (immediate `Err(Error::Unimplemented)`):**
| Op | Line |
|----|------|
| `add` | 631 |
| `mul` | 640 |
| `silu_mul` | 649 |
| `rms_norm` | 659 |
| `softmax` | 667 |
| `embedding` | 676 |

**Vanity / dead surface (within vulkan):**
- `VulkanAutotuner` (lib.rs:697-718) — mock; returns 1 of 2 hardcoded configs by `% 64 == 0`. V1 drag.
- `compile_cube_kernel_to_spirv` (lib.rs:722) — emits a SPIR-V *assembly string*. Never assembled, loaded, or dispatched.
- `generate_matmul_glsl` (lib.rs:752) — emits GLSL. Never compiled (no `naga`/`shaderc` call) or dispatched.
- `COMPUTE_SHADER_GEMM` (cube_kernels.rs:6) — GLSL matmul shader string. Correct (Bug 1 fixed) but never dispatched.
- `COMPUTE_SHADER_ATTENTION` (cube_kernels.rs:64) — dead code (`#[allow(dead_code)]`).
- `AVAILABLE_KERNELS` (cube_kernels.rs:92) — dead code (`#[allow(dead_code)]`), only read by the metadata test.
- `KernelBuilder` (cube_kernels.rs:106-122) — all methods return `Error::Unimplemented`.
- `VulkanContext::init` (lib.rs:205) — calls `vkCreateInstance`; **hangs** when no loader present. No timeout, no graceful failure. This is why vulkan tests are unrunnable in CI. V2 structural.

---

## Phase 2 — Vanity Engineering Assessment

### Summary

Three backends present as "complete GPU compute stacks." In reality: rocm does `matmul` only among the 7 required compute ops; cuda does `matmul` (square) only; vulkan does **zero** GPU compute — its `matmul` is CPU. The elementwise/norm/softmax/embedding ops — the actual body of any transformer — are `Err(Unimplemented)` on all three. The CubeCL/SPIR-V/GLSL scaffolding in vulkan is theater: shaders are authored, "compiled" to strings, and never run.

### Requirement-to-Complexity Ratio (RCR)

**7/10.** The `VulkanAutotuner`, `compile_cube_kernel_to_spirv`, `generate_matmul_glsl`, `KernelBuilder`, `HsacoKernelCache`, and the HIP-graph capture block are complexity with zero requirement payoff. They read like a PhD thesis on GPU dispatch that was scaffolded but never connected.

### Top Findings (7)

1. **Vulkan is a non-functional shell** (V3 Compounding). 910+ lines, every GPU op is either `Unimplemented` or CPU-simulated. `vkCreateInstance` hangs CI. Deletion test: `grep -r "grim-backend-vulkan" crates/ | grep -v "backend-vulkan/"` — is it referenced by anything? If only `grim-cli` probes it, it provides negative value (hangs, confuses).
   - *Should be:* either (a) delete the crate and fall back to CPU, or (b) commit to real Vulkan compute dispatch within 2 cycles. Not a half-shell.

2. **`add`/`mul`/`silu_mul`/`rms_norm`/`softmax`/`embedding` unimplemented on rocm (V2 Structural).** These 6 ops are the transformer. rocm cannot run a single layer. Highest-value work in the repo.
   - *Should be:* implement via hipRTC JIT kernels or hipified elementwise + a fused `silu_mul`, `rms_norm`, `softmax` kernel. Track as the blocker for inference.

3. **CUDA `matmul` square-only landmine (V2 Structural).** Silent wrong answers for non-square. A model with `m != n` (every decoder step) gets garbage.
   - *Should be:* implement proper transpose handling (`transa`/`transb` selection by stride), add non-square tests, or return `Error::Unsupported` for non-square until fixed. Do not ship silent.

4. **`lookup_gemm_config` mock (V1 Drag).** Hardcoded tiles by `% 128`. Wastes 26 lines; implies tuning that doesn't happen.
   - *Should be:* delete; let rocBLAS pick the algorithm (`rocblas_gemm` default), or wire a real `rocblas_gemm_algo` query.

5. **Vulkan CubeCL/SPIR-V/GLSL pipeline is dead (V3 Compounding).** `compile_cube_kernel_to_spirv`, `generate_matmul_glsl`, `KernelBuilder`, `AVAILABLE_KERNELS`, `COMPUTE_SHADER_ATTENTION` — none dispatch. This is Pattern 46 (Display-Only UI) at the shader level.
   - *Should be:* if keeping vulkan, replace the string-emitter with a real `shaderc`/`naga` compile + `vkCreateShaderModule` + `vkCmdDispatch` path. Otherwise delete all of it.

6. **CUDA `advise` no-op (V1 Drag).** Returns `Ok(())` — hides that memory advice is unimplemented. rocm implements it.
   - *Should be:* `cudaMemAdvise` call, or `Err(Error::Unimplemented("CUDA advise pending"))` for honesty.

7. **`println!` profiler spam in rocm (V0 Cosmetic).** 9 raw `println!` behind `rocm-profile`. Should be `tracing::info!` gated by a subscriber, not `println!`.

### Vanity Debt Estimate

~340 lines of mock/simulation/dead code across the three crates (vulkan ~200, rocm ~90, cuda ~50). Maintenance cost: ~6 eng-hours/month reading and re-verifying paths that never execute.

### The Hard Question

> If you deleted `grim-backend-vulkan` and `grim-backend-cuda` today and kept only `grim-backend-rocm` + a CPU fallback, what model capability would you lose that you actually have working code for?

(Answer: nothing — neither cuda nor vulkan can run a transformer layer. The honest move is to cut them to prototypes or implement the 6 missing rocm ops first.)

---

## Phase 3 — Kill Criteria

### Tier 1 — Hard Kill (automatic)
- `grim-backend-vulkan`: if no real `vkCmdDispatch` path is wired within 2 release cycles → delete crate.
- Any backend op returning silently-wrong results (cuda non-square matmul) → block release until fixed or errors.

### Tier 2 — Review Trigger
- `lookup_gemm_config` mock: review at next planning; default-to-delete.
- HIP graph capture block: review; default-to-delete unless a benchmark uses it.
- `println!` profiler: replace with `tracing` or delete.

### Tier 3 — Soft-Go (must earn continuation)
- vulkan: within 30 days, demonstrate one real dispatched compute kernel (not CPU loop) on a CI-runnable software rasterizer (llvmpipe) OR delete.
- cuda: within 30 days, pass non-square matmul tests on a GPU host OR mark `matmul` non-square as `Unsupported`.

---

## Phase 4 — Remediation Plan (Prioritized Tasks)

### Task 1: Implement the 6 missing rocm compute ops (BLOCKER for inference)
**Files:** `crates/grim-backend-rocm/src/lib.rs:1172-1236`
**Why:** Without these, no transformer runs. Highest value-per-line in repo.
**Interfaces:** Consumes `RocmStorage` (device_ptr, shape, dtype). Produces `(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)`.
- [ ] Step 1: Write failing tests for `add`, `mul`, `silu_mul`, `rms_norm`, `softmax`, `embedding` on a GPU host.
- [ ] Step 2: Implement `add`/`mul` via hipified elementwise kernel (or `hipLaunchKernel` with a JIT string).
- [ ] Step 3: Implement `silu_mul` (fused gate*up activation) as one kernel.
- [ ] Step 4: Implement `rms_norm` (x * rsqrt(mean(x²)+eps) * w) as one kernel.
- [ ] Step 5: Implement `softmax` (row-wise) as one kernel.
- [ ] Step 6: Implement `embedding` (gather rows by indices) as one kernel.
- [ ] Step 7: Run `cargo test -p grim-backend-rocm`; all green on GPU host.
- [ ] Step 8: Commit per-op with descriptive messages.

### Task 2: Fix CUDA `matmul` non-square + honest `advise`
**Files:** `crates/grim-backend-cuda/src/lib.rs:351-436, 506`
**Why:** Square-only is a silent correctness bug; `advise` lies.
- [ ] Step 1: Write failing test `matmul` with `m=2, k=3, n=4` (non-square).
- [ ] Step 2: Replace column-major hack with explicit `transa`/`transb` by comparing row-major strides; handle both square and non-square.
- [ ] Step 3: Write passing test for non-square.
- [ ] Step 4: Change `advise` to `Err(Error::Unimplemented("CUDA advise pending"))` or implement `cudaMemAdvise`.
- [ ] Step 5: Commit.

### Task 3: Decide vulkan's fate (delete or wire dispatch)
**Files:** `crates/grim-backend-vulkan/**`
**Why:** It is a non-functional shell that hangs CI.
- [ ] Step 1: `grep -rn "grim-backend-vulkan" crates/ | grep -v "grim-backend-vulkan/"` — confirm only `grim-cli` probes it.
- [ ] Step 2: If no dispatch path within 2 cycles → delete crate + remove from `grim-cli` probe.
- [ ] Step 3: If keeping → implement `vkCreateShaderModule` + `vkCmdDispatch` for `matmul` using `COMPUTE_SHADER_GEMM` (already correct), swap CPU loop for real dispatch, add llvmpipe CI job.
- [ ] Step 4: Commit decision.

### Task 4: Ponytail sweep — delete mock/dead code
**Files:** rocm `lookup_gemm_config`, `hipGraphLaunch`, HIP-graph block; vulkan `VulkanAutotuner`, `compile_cube_kernel_to_spirv`, `generate_matmul_glsl`, `KernelBuilder`, `AVAILABLE_KERNELS`, `COMPUTE_SHADER_ATTENTION`; rocm `println!` → `tracing`.
**Why:** V1/V0 drag; removes ~340 lines.
- [ ] Step 1: `cargo check -p grim-backend-rocm -p grim-backend-vulkan` baseline.
- [ ] Step 2: Delete mock config + graph block in rocm; replace `println!` with `tracing::info!`.
- [ ] Step 3: Delete dead vulkan shader/autotuner/emitter code (unless Task 3 keeps vulkan).
- [ ] Step 4: `cargo check --workspace` clean; commit `refactor: remove backend mock/dead code`.

### Task 5: Unified tile-config helper (optional, post-Task 4)
**Why:** Each backend re-implements `lookup_gemm_config`/`VulkanAutotuner` identically (Pattern 38, parallel config).
- [ ] Step 1: Extract `grim-tensor/src/autotune.rs` with one `GemmTileConfig` type.
- [ ] Step 2: Have rocm/cuda/vulkan (if kept) use it.
- [ ] Step 3: Commit.

---

## Phase 5 — Self-Review

1. **Spec coverage:** All 18 stub sites (6 per backend × 3) are cataloged. Simulation-only vulkan `matmul` flagged. Vanity grade assigned. Kill criteria written. Tasks cover implement (rocm 6 ops), fix (cuda), decide (vulkan), sweep (ponytail), unify (optional).
2. **Placeholder scan:** No "TBD"/"implement later". Every task has concrete file:line and action.
3. **Type consistency:** `RocmStorage`, `VulkanStorage`, `CudaStorage`, `BackendStorage`, `ComputeHandle` referenced consistently. `GemmTileConfig` name reused in Task 5 matches rocm's existing struct.

**Verdict:** The backends are 80% scaffold, 20% function. The 20% (rocm matmul + memory moves) is real. Cut the scaffold or finish it — the middle ground (shells that look complete) is the most expensive option.
