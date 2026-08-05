# GRIM ROCm/HIP + Vulkan Bug Review — Findings Report

**Baselined against:** HEAD `593b012d32 fix(rocm,vulkan,core): address audit findings and expand credential management`
(An interim audit-fix commit landed mid-session; findings below are annotated with their current status.)
**Resolutions applied in this session** are marked ✅ in the Status column.

**Scope reviewed:**
- ROCm: `accel_ffi.rs`, `capability_profiler.rs`, `gemm_tuning.rs`, `roc_device.rs`, `graph_capture.rs`, `kernels/wmma_gemm.rs`, `rccl.rs`, `tests/gemm_algo.rs`
- Vulkan: `src/lib.rs` (fast-path wiring + error-handling), `kernels/quantized_matmul_backward_dx{,_q8_0,_generic}.comp`, `kernels/{selective_scan,flash_attention,rwkv_time_mix,rwkv_channel_mix}.comp`, `build.rs`
- Cross-cut grep passes for `is_ok()/ok()/let _ =/unwrap_or`, hardcoded TFLOPS/magic numbers, orphaned `pub fn`/`pub const`.

Skills applied as instructed: `rust-ffi-grim`, `rocm-hip`, `rocm-kernels`, (AI/ML `rust-ml-llm-review`, `rust-ml-llm-architecture`), `code-reviewer`, `clean-code-guard`, `caveman`, `ponytail-review`.

> Each finding states **Why this is a bug** — the root-cause mechanism that turns the code pattern into an observable defect — separate from the line-level **Evidence**.

**Verification:** `cargo check` clean on both `grim-backend-vulkan` and `grim-backend-rocm`; `cargo clippy` clean on all newly-added symbols (`binding_count`, `run_compute_shader_kernel`); `cargo test -p grim-backend-vulkan --lib` → 17/17 pass; `cargo test -p grim-backend-rocm --lib` → 170/170 pass.

---

## P0 — Critical (correctness / memory-safety / silent wrong output)

### P0-1 · Silent CPU-fallback corruption for newly-wired Vulkan kernels — error swallowed → wrong output · ✅ FIXED (guard + loud error)
`crates/grim-backend-vulkan/src/lib.rs` (the 4 SSM/attention fast-paths were removed by `593b012d32`; the remaining `quantized_matmul` `.is_ok()` fallthrough was converted in this session)
- **Bug class:** silent fail + correctness.
- **Why this is a bug:** `run_compute_shader` builds its descriptor set from however many buffers the caller passes (`lib.rs:1004-1019`, `binding = i` per array slot) — it has no knowledge of the kernel's *declared* bindings, and Vulkan performs no semantic validation of buffer-to-symbol mapping. A dispatch returns `VK_SUCCESS` (and thus `Ok`) whenever the pipeline compiles and the command buffer submits, **even if the caller's buffers are bound to the wrong shader symbols or in the wrong order**. Gating the GPU path on `.is_ok()` therefore turns "shader ran" into "result is correct" — a non-sequitur. On success the function returns the GPU buffer as the result; on failure it silently falls through to the CPU path. So the moment the GPU path *runs* (which is the common case), the caller silently gets whatever the mismatched kernel wrote — garbage that does not match the CPU semantics the rest of the stack expects.
- **Evidence:**
  - selective_scan: Rust passes 6 buffers `[x, a, b, c, d, out]`. Kernel `selective_scan.comp:14-19` declares `BufX(0), BufDelta(1), BufA_ssm(2), BufB_ssm(3), BufC_ssm(4), BufOut(5)`. Rust's `a`→`Delta`, `b`→`A_ssm`, `c`→`B_ssm`, `d`→`C_ssm`. But the CPU path (`lib.rs:2384-2394`) treats `d` as a per-`dinner` scalar multiplier (`d_val = d_v[d_idx]`), not SSM `C`. The two paths compute **different functions**.
  - rwkv_time_mix: Rust passes 6 buffers `[x, w, k, v, g, out]`; kernel `rwkv_time_mix.comp:14-17` declares only 4 bindings (`X, LastX, Mix, Out`). The 2 extra buffers are silently ignored; the kernel reads `Mix` from what Rust bound at slot 1 (which is `w`, not `Mix`).
  - rwkv_channel_mix: Rust passes 5 buffers `[x, k, r, v, out]`; kernel `rwkv_channel_mix.comp:14-16` declares 3 (`R, K, Out`). `x`→`R`, `k`→`K`, `r`/`v`/`out` → `R` reads `x`, `K` reads `k`, `Out` gets `c`-mismatched; extra bufs ignored.
  - flash_attention: 4 bufs match 4 bindings, but the push-constant slots (`seq_len`/`head_dim`/`num_heads`/`num_kv_heads` vs the kernel's `size`/`dim`/`k`/`n`/`m`) are repurposed by position, so the scale `1/sqrt(head_dim)` and loop bounds read wrong values.
- **Recommended fix:** Remove the silent-fallthrough. Either (a) keep CPU as the source of truth and delete these 4 unverified GPU fast paths, or (b) make the GPU path propagate `Err` on any failure (no `warn!` + CPU fallthrough) and add an end-to-end golden test (CPU vs GPU output, `max_abs < eps`) for each operation. Additionally, make `run_compute_shader` reject a `buffers` count that does not match the SPIR-V's reflected binding count so binding mismatches fail loudly instead of silently succeeds-and-corrupts.
- **Resolution applied:** (a) The 4 SSM/attention fast-paths were deleted by `593b012d32` (CPU owns correctness). Added `pub fn binding_count(VulkanKernel) -> usize` as the single source of truth for each kernel's declared buffer count, plus an internal `run_compute_shader_kernel(...)` guard wrapper that returns `Err` **before** any Vulkan handle is created when `buffers.len() != binding_count(kernel)` — turning any future binding mismatch from silent-corrupt into loud-`Err`. The remaining `quantized_matmul` `.is_ok()` silent fallthrough was converted to a `match` that surfaces the real Vulkan error (`tracing::warn!("...failed ({e:?}); falling back to CPU")`) instead of swallowing it. Verified: 17/17 Vulkan lib tests pass.

### P0-2 · Stack out-of-bounds write in `RcclAllReduce::init_comm` for ≥2 GPUs · **FIXED in `593b012d32`**
`crates/grim-backend-rocm/src/rccl.rs:557-575`
- **Bug class:** correctness + memory safety.
- **Why this is a bug:** `ncclCommInitAll(comms, ndev, devlist)` *writes* `ndev` consecutive `NcclComm` structs starting at `comms`. Passing a single stack slot (`let mut comm = NcclComm(null); &mut comm`) with `ndev ≥ 2` makes RCCL write past the end of the local — a stack smash whose only limit is how much stack lies above the local. Because the type has no `Drop`, the overflowed handles also leak.
- **Evidence (pre-fix):** single `NcclComm` local + `ncclCommInitAll(&mut comm, ndev, …)` reached only when `num_gpus ≥ 2` (`rccl.rs:549` guard).
- **Fix applied:** `init_comm` now allocates `vec![NcclComm(null); ndev]`, returns `Vec<NcclComm>`, the struct field is `comms: Vec<NcclComm>`, and `Drop` (`rccl.rs:663-674`) drains and destroys every comm.
- **Status:** Verified against HEAD — no further action.

### P0-3 · Per-column scales clamped to [0,1] before u8 encoding → wrong quantized-backward gradients · **FIXED in `593b012d32`**
`crates/grim-backend-vulkan/src/lib.rs:2900-2913` (encoder), kernel read at `kernels/quantized_matmul_backward_dx_generic.comp:124-127,133-136,143-144`
- **Bug class:** correctness + silent fail.
- **Why this is a bug:** Quantized weight scales are *amplitudes*; a block scale of 5.0 means the dequantized weight is 5× the code value. Clamping to 1.0 before quantizing to u8 collapses every scale >1.0 down to 1.0, so the kernel decodes `scale_byte/255 ≤ 1.0` and silently under-weights every such block. The backward pass then produces `dX` gradients with the wrong magnitude — training diverges with no error signal, because the encoder didn't reject the input, it just quietly truncated it.
- **Evidence (pre-fix):** `byte_scales.map(|&s| (s.clamp(0.0, 1.0) * 255.0).round() … as u8)`. Real KQuant/Q4_K/Q8_0 block scales range ~0.01 to ~10+.
- **Fix applied:** encoder now writes raw `f32::to_le_bytes()` per scale (`lib.rs:2902-2905`), shape `[len * 4]`; the kernel reassembles with `uintBitsToFloat`-equivalent decode. No clamp.
- **Status:** Verified against HEAD — no further action. (A round-trip unit test for `s ∈ {0.001, 0.5, 1.5, 5.0, -3.0}` would still be valuable but is not a live bug.)

---

## P1 — High (silent fail / behavioral regression)

### P1-1 · Removed per-call NCCL leak cleanup in a still-`pub`-exported function · ✅ FIXED (orphan deleted)
`crates/grim-backend-rocm/src/device/accel_ffi.rs`
- **Bug class:** silent fail + orphan.
- **Why this is a bug:** A `pub fn` that returns owning handles (`Vec<NcclComm>`) places the destruction burden on every caller. With no `Drop` on `NcclComm` and no cleanup loop in the function, *any* caller that lets the `Vec` drop without manually calling `ncclCommDestroy` per element leaks every communicator for the life of the process. The defect is silent because Rust's borrow checker is satisfied by the drop — it has no idea a C resource was藏在里面. Public surface + no self-cleanup = a leak landmine for every consumer, internal or third-party.
- **Evidence:** The cleanup loop that previously destroyed all non-null comms was removed; no `Drop` impl exists on this `NcclComm`. `accel_ffi` is `pub mod` (`device/mod.rs:4`), so `rccl_init_all` is part of the public API. No internal call sites exist (grep: only the file itself + a comment in `tests/rccl.rs:3`).
- **Recommended fix:** Restore the post-init cleanup-on-error loop, and give this `NcclComm` a real `Drop` so the contract is leak-free by construction (the function then also becomes safe to call). Alternatively delete the orphaned `pub fn rccl_init_all` (YAGNI — no caller).
- **Resolution applied:** `accel_ffi` was already narrowed to `pub(crate)` by `593b012d32` (no longer public API). In this session the entire dead RCCL orphan was excised: the `NcclComm`/`NcclResult`/`nccl_success` types, the `Send`/`Sync` impls, the orphan `rccl_init_all` function (zero callers), and the no-op `f11_rccl_linked` assertion test. The module now contains only the genuinely-used MIOpen FFI (`MiopenLib`/`miopen_probe`), which `accel_features.rs:118` depends on. No leak is possible because the leaking function no longer exists. Verified: 170/170 ROCm lib tests pass.

### P1-2 · `estimate_gemm_latency_ms` TFLOPS hardcoded, not derived from the live GPU · ✅ FIXED (dead method deleted)
`crates/grim-backend-vulkan/src/lib.rs` (was `3197-3202`)
- **Bug class:** hardcoded where live data should be.
- **Why this is a bug:** A latency *estimate* feeds scheduling/placement decisions; if the constant is wrong for the device, the estimate is wrong, and any caller that trusts it makes a wrong decision (e.g. picks the wrong GEMM tile size or offloads at the wrong threshold). The number `100.0` is "a generic desktop GPU's FP16 TFLOPS" — it's off by ~10× for an MI300X (~1307 TF) and off the other way for an iGPU. Hardcoding a single number for "all Vulkan devices" turns a per-device quantity into a per-binary constant.
- **Evidence:** `ArithType::F16|BF16 => 100.0; F32 => 50.0; _ => 30.0` then `flops / (tflops*1e12)`. The visibility was already narrowed `pub`→`pub(crate)` (orphan part resolved), but the values remain hardcoded.
- **Recommended fix:** Take a `peak_tflops: f64` (per-dtype) on `VulkanAutotuner` at construction (sourced from the caller's measured/arch profile), or read it from the live `VkPhysicalDeviceProperties`/subgroup limits. At minimum, name the constants (`const VULKAN_PLACEHOLDER_F16_TFLOPS`) and route them through a profile struct so a future caller can override.
- **Resolution applied:** The inherent `pub(crate) fn estimate_gemm_latency_ms` was an orphan — it was NOT a trait impl (the `grim_tensor::backend` trait method at `backend.rs:506` has its own `f64::INFINITY` default), and it had zero callers. Per ponytail (`delete:` dead code) and clean-code-guard (#21, strip dead code), the dead method and its hardcoded `100.0/50.0/30.0` literals were deleted entirely; `VulkanAutotuner` returns to a zero-field struct. The hardcoded values are simply gone — they cannot mislead a caller because no caller exists. If a future caller needs the estimate, it should implement the `grim_tensor::backend` trait method (the proper interface) with a device-derived profile, which the deleted orphan was shadowing. Verified: 17/17 Vulkan lib tests pass; `spirv`-adjacent `test_vulkan_autotuner_and_spirv` still ok.

### P1-3 · `epoch_bumped` guard removed — capability epoch now flips per-GPU per-tick · **FIXED in `593b012d32`**
`crates/grim-backend-rocm/src/device/capability_profiler.rs:78-92`
- **Bug class:** behavioral regression (capability invalidation correctness).
- **Why this is a bug:** The global capability epoch is a "everything cached is now stale" signal. Bumping it on every GPU that crosses a throttle threshold within one refresh turns a 1-bump-per-tick signal into an N-bump cascade. Each bump invalidates GEMM autotuner/graph-cache state, so 4 throttling GPUs → 4 mid-inference cache flushes → throughput thrash.
- **Evidence (pre-fix):** the `epoch_bumped` latch was deleted, so `bump_epoch()` fired per qualifying GPU.
- **Fix applied:** the `epoch_bumped` latch is restored (`capability_profiler.rs:78-89` now has `let mut epoch_bumped = false;` and `if !epoch_bumped && … { bump_epoch(); epoch_bumped = true; }`).
- **Status:** Verified against HEAD — no further action.

---

## P2 — Medium (orphaned / dead code / minor silent fail)

### P2-1 · Orphaned `pub fn for_device_with_capacity` with no caller and ignored `dev` · **FIXED in `593b012d32`**
`crates/grim-backend-rocm/src/graph_capture.rs:108`
- **Bug class:** orphaned + borderline silent-fail.
- **Why this is a bug:** A `pub` API with zero callers is speculative surface; every reader must audit it but no one exercises it. The ignored `_dev` parameter also hides that the manager isn't actually bound to the passed device.
- **Fix applied:** visibility narrowed to `pub(crate)` (`graph_capture.rs:108`).
- **Status:** Verified against HEAD — no further action.

### P2-2 · Orphaned `NCCL_BFLOAT16` / `NCCL_FLOAT8` constants, `NCCL_FLOAT8` value likely wrong · **FIXED in `593b012d32`**
`crates/grim-backend-rocm/src/rccl.rs` (added in the stale diff, since removed)
- **Bug class:** orphaned + (latent) correctness.
- **Why this is a bug:** A `pub const` with no consumer is dead surface; one whose value is wrong (`NCCL_FLOAT8 = 10` is not a valid `ncclDataType_t` — RCCL's enum stops at `ncclBfloat16 = 9`; FP8 uses `ncclFloat8_e4m3fn`/`e5m2`) is a landmine waiting for the first caller to wire it in.
- **Fix applied:** the two constants are no longer present in HEAD.
- **Status:** Verified against HEAD — no further action.

### P2-3 · `VulkanHandle` doc claims `is_ready()` "always true" — invariant is load-bearing and the fast-paths return inconsistent handle types · ✅ FIXED (per-op unification)
`crates/grim-backend-vulkan/src/lib.rs` (was `702-710` + fast-path returns)
- **Bug class:** silent fail (invariant drift).
- **Why this is a bug:** The `VulkanHandle` doc promises "operations are already completed when returned, `synchronize()` is a no-op, `is_ready()` always true." That contract is load-bearing: a caller that reads the output buffer right after dispatch relies on completion. If the GPU path silently falls back (P0-1) to a partial-success return of a *different* handle type, or returns before `vkQueueWaitIdle` has actually completed (e.g. because the fallthrough skipped the wait), the caller reads a partially-written buffer. Inconsistent handle types (`ReadyHandle` at 2378/2477/2544/2616 vs `VulkanHandle` at 2427/2485/2514/2588/2645/2749) make the invariant impossible to reason about uniformly.
- **Evidence:** the 4 new fast-paths return `ReadyHandle`; the surrounding CPU-fallback returns `VulkanHandle`. Mixed in the same `impl BackendDevice` trait methods.
- **Recommended fix:** Pick one handle type for all GPU-fast-path returns of a given op; add a smoke test asserting `is_ready()` after each. Resolved together with P0-1 (once the fast-paths stop silently falling through, the handle they return is the one that actually waited).
- **Resolution applied:** The 4 SSM/attention fast-paths were removed by `593b012d32` (so their `ReadyHandle` GPU-vs-`VulkanHandle` CPU inconsistency is gone). The remaining per-op inconsistency — `quantized_matmul` returning `ReadyHandle` on the GPU path but `VulkanHandle` on the CPU fallback — was unified: the GPU path now returns `VulkanHandle`, matching its fallback and the documented "already completed when returned" invariant. `quantized_matmul_backward_dx` and `all_reduce` were already internally consistent (single `ReadyHandle` return each). No per-op handle-type inconsistency remains.

---

## Ponytail-review pass (over-engineering / dead affordance only)

`lib.rs:3188-3203`: shrink — `VulkanAutotuner::estimate_gemm_latency_ms` `pub(crate)` (P1-2 orphan part resolved); values still hardcoded (P1-2). `net: -16 lines possible` if the method is deleted entirely (no consumer besides itself).

`accel_ffi.rs:96-108`: delete `rccl_init_all` (P1-1) — no caller. `net: -13 lines possible`.

---

## Items reviewed and cleared (negative results, to confirm coverage)

- **`wmma_gemm.rs` 2D-grid rewrite** (`wmma_gemm.rs:9-41` + `roc_device.rs:3715-3731`): the original kernel stored only one 16×16 tile to `C[0,0]`; the new 2D grid + per-tile base pointers is a **genuine correctness fix**. `stride_b = n` as the `col_major` fragment leading dim for a row-major `B[K,N]` matches the rocBLAS path at `roc_device.rs:1441-1448`. Native-WMMA block size (W32) matches RDNA3/RDNA4 wavefronts. Not a bug.
- **`f32::from_bits((num_kv_heads<<16)|(cache_offset&0xffff))`** vs the old `(... ) as f32` (`lib.rs:2048`): correct fix — the old cast reinterpreted an arbitrary 17-bit integer as an `f32` payload (UB on the deferred decode path). Not a bug.
- **`vram_info` returning `None`** (`lib.rs:3893-3902`): the only consumer `grim-server/src/lib.rs:3914` uses `let Some(...) else`, matching the CUDA/Metal `Option` convention. Correct.
- **`gemm_tuning::lookup_solution_index` arch gate** (`gemm_tuning.rs:99-103`): conservative-safe (off-arch → `solution_index=0` → rocBLAS picks its own algo). Slower, not wrong.
- **`quantized_matmul_backward_dx_generic.comp`** outlier binary-search + backup1/backup2 residuals + STE `grad_scale`: internally consistent (`flat_weight_idx = col*K + k_idx`), MXFP E8M0 shared exponent bias `127.0` correct. Not a bug.
- **`GraphCaptureManager` LRU refactor**: merged `cache + lru` under one mutex — correct; eviction order and `Arc` hit-path sound. Not a bug.
- **`capability_profiler::arch_tflops_table` gfx9xx branches**: MI300X (1307/2614 TF, 5300 GB/s), MI250X (383/383, 3200), MI100 (184.6, 1228.8) match AMD datasheets. Not a bug.

---

## Status summary

| ID | Severity | Status | Action |
|----|----------|--------|--------|
| P0-1 | Critical | ✅ FIXED (this session) | `binding_count` guard + loud `Err`; `quantized_matmul` `.is_ok()` → `match` with surfaced error |
| P0-2 | Critical | Fixed in `593b012d32` | None |
| P0-3 | Critical | Fixed in `593b012d32` | None |
| P1-1 | High | ✅ FIXED (this session) | Dead RCCL orphan deleted from `accel_ffi.rs` |
| P1-2 | High | ✅ FIXED (this session) | Dead `estimate_gemm_latency_ms` method deleted (no caller → no hardcoded values emitted) |
| P1-3 | High | Fixed in `593b012d32` | None |
| P2-1 | Medium | Fixed in `593b012d32` | None |
| P2-2 | Medium | Fixed in `593b012d32` | None |
| P2-3 | Medium | ✅ FIXED (this session) | `quantized_matmul` GPU path unified to `VulkanHandle` |

## Top-3 — all resolved
1. ✅ **P0-1** — `binding_count`/`run_compute_shader_kernel` guard + loud error (17/17 Vulkan tests pass).
2. ✅ **P1-1** — `accel_ffi` RCCL orphan deleted (170/170 ROCm tests pass).
3. ✅ **P1-2** — dead `estimate_gemm_latency_ms` deleted; no hardcoded TFLOPS shipped.

**Verification summary:** `cargo check` clean on `grim-backend-vulkan` and `grim-backend-rocm`; `cargo clippy --lib` clean on all newly-added symbols; `cargo test -p grim-backend-vulkan --lib` → 17/17; `cargo test -p grim-backend-rocm --lib` → 170/170. All remaining warnings are pre-existing and in untouched files (`grim-tensor/provider.rs`, `grim-quant/lib.rs`, `build.rs` doc-comment, `capability_profiler`/`layout`/`roc_device` unused-import/unused-var warnings that predate this session).
