# GRIM ROCm/HIP + Vulkan Bug Review — Findings Report

**Scope reviewed (uncommitted working tree + last 5 commits touching Vulkan quantized-backward):**
- ROCm: `accel_ffi.rs`, `capability_profiler.rs`, `gemm_tuning.rs`, `roc_device.rs`, `graph_capture.rs`, `kernels/wmma_gemm.rs`, `rccl.rs`, `tests/gemm_algo.rs`
- Vulkan: `src/lib.rs` (uncommitted fast-path wiring + error-handling), `kernels/quantized_matmul_backward_dx{,_q8_0,_generic}.comp`, `kernels/{selective_scan,flash_attention,rwkv_time_mix,rwkv_channel_mix}.comp`, `build.rs`
- Cross-cut grep passes for `is_ok()/ok()/let _ =/unwrap_or`, hardcoded TFLOPS/magic numbers, and orphaned `pub fn`/`pub const` symbols.

Skills applied as instructed: `rust-ffi-grim`, `rocm-hip`, `rocm-kernels`, (AI/ML `rust-ml-llm-review`, `rust-ml-llm-architecture`), `code-reviewer`, `clean-code-guard`, `caveman`, `ponytail-review`.

---

## P0 — Critical (correctness / memory-safety / silent wrong output)

### P0-1 · Silent CPU-fallback corruption for newly-wired Vulkan kernels (validates clean!) — error swallowed → wrong output
`crates/grim-backend-vulkan/src/lib.rs:2363-2382` (selective_scan), `2461-2480` (flash_attention), `2526-2545` (rwkv_time_mix), `2598-2617` (rwkv_channel_mix)
- **Bug class:** silent fail + correctness.
- **Evidence:** Every new fast path does `if run_compute_shader(...).is_ok() { return Ok(out_storage); }` and, on `Err`, falls through to `tracing::warn!(...falling back to CPU)`. The kernel dispatch "succeeds" (VK_SUCCESS) regardless of whether the SPIR-V's buffer bindings and push-constant layout match what the Rust caller supplied — Vulkan exposes no semantic validation. When bindings mismatch, the shader executes, writes a buffer, and the function returns `Ok` with **silently wrong data**.
- **Concrete mismatch (selective_scan):** Rust passes `[x, a, b, c, d, out]` bound at indices 0..5 (confirmed at `lib.rs:1004-1019` — `binding = i` per array slot). The kernel `kernels/selective_scan.comp:9-13` declares `BufX(0), BufDelta(1), BufA_ssm(2), BufB_ssm(3), BufC_ssm(4), BufOut(5)`. So Rust's `a` lands in `Delta`, `b` in `A_ssm`, `c` in `B_ssm`, `d` in `C_ssm` — but the CPU path (`lib.rs:2384-2394`) treats `d` as a per-`dinner` multiplier, not SSM `C`. The two paths compute **different functions**, and the GPU path always wins on a clean `VK_SUCCESS`.
- **Recommended fix:** Do not gate on `is_ok()`. Either (a) keep CPU as the source of truth and delete these unverified GPU fast paths, or (b) make the GPU path return `Err` on any binding/layout mismatch and propagate (no `warn!` + CPU fallthrough), and add an end-to-end golden test (CPU vs GPU output, assert `max_abs < eps`) for each of the four operations before enabling. Also assert in `run_compute_shader` that `buffers.len()` equals the declared descriptor set's binding count.

### P0-2 · Stack out-of-bounds write in `RcclAllReduce::init_comm` for ≥2 GPUs
`crates/grim-backend-rocm/src/rccl.rs:560-577` (callers: `roc_device.rs:316`, `tests/rccl.rs:77,99`, `grim-cli/src/train.rs:543`, `grim-garage/src/jobs.rs:1716`)
- **Bug class:** correctness + memory safety.
- **Evidence:** `let mut comm = NcclComm(std::ptr::null_mut());` (single stack slot, 16 bytes), then `unsafe { ncclCommInitAll(&mut comm, ndev, devlist.as_ptr()) }`. `ncclCommInitAll` *writes* `ndev` communicators starting at the given pointer. The guard at `rccl.rs:549` (`if num_gpus <= 1`) means this runs **only when `ndev >= 2`**, so RCCL writes 2+ consecutive `NcclComm` structs into one 16-byte slot → stack smash beyond the local. The `Drop` at `rccl.rs:664-676` destroys only `self.comm` (the first slot) → the other `ndev-1` handles leak.
- **Recommended fix:** Allocate `let mut comms = vec![NcclComm(ptr::null_mut()); ndev as usize];` and pass `comms.as_mut_ptr()`. Store all of them (e.g. `comm: Vec<NcclComm>`) and destroy each in `Drop`. Compare against the sibling `accel_ffi::rccl_init_all` (`accel_ffi.rs:96-108`) which already does the right thing — prefer reusing it or sharing the constructor.

### P0-3 · Per-column scales clamped to [0,1] before u8 encoding → wrong quantized-backward gradients
`crates/grim-backend-vulkan/src/lib.rs:2909-2916` (encoder), kernel read at `kernels/quantized_matmul_backward_dx_generic.comp:124-127,133-136,143-144`
- **Bug class:** correctness + silent fail.
- **Evidence:** `byte_scales.map(|&s| (s.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8)` — every scale `> 1.0` is silently truncated to `1.0`. Real KQuant / Q4_K / Q8_0 block scales routinely exceed 1.0 (typical fp16/f32 group scales span ~0.01 to several, super-block scales up to ~10+). The kernel then computes `w_val = unpack_weight(...)  * (scale_byte / 255.0)` — so any layer with `|scale| > 1.0` has its weight magnitude truncated to ≤1.0, producing **wrong `dX` gradients** that mis-train without any error. The new negative-scale guard (`lib.rs:2902-2908`) is partially defensive (the clamp already discarded negatives) and does not catch this.
- **Recommended fix:** Encode each scale by its raw `f32` IEEE-754 bits into a `[u8; 4]` block (or use an `f16`/u16 with a shared exponent), and have the kernel `uintBitsToFloat`/reassemble it. Replace the u8/255 encoding entirely, then add a round-trip unit test that encodes `s ∈ {0.001, 0.5, 1.5, 5.0, -3.0}` and asserts decoded == original within tolerance.

---

## P1 — High (silent fail / behavioral regression)

### P1-1 · Removed per-call NCCL leak cleanup in a still-`pub`-exported function
`crates/grim-backend-rocm/src/device/accel_ffi.rs:96-108` (diff removed lines 108-115)
- **Bug class:** silent fail + orphan.
- **Evidence:** The deleted block robustly destroyed any non-null comm if a caller dropped the returned `Vec` without explicit teardown. `NcclComm` here has **no `Drop`** (unlike rccl.rs's `RcclAllReduce`). `accel_ffi` is `pub mod` (`device/mod.rs:4`), so `rccl_init_all` is part of the public API. No internal call sites exist (grep returns only the file itself + a comment in `tests/rccl.rs:3`), so this is today an **orphaned public function** that no longer self-cleans — any third-party caller leaks every communicator.
- **Recommended fix:** Either delete the orphaned `pub fn rccl_init_all` and the module-`pub mod accel_ffi` export (YAGNI), or restore the cleanup loop and additionally give `NcclComm` a real `Drop` so the contract is leak-free by construction.

### P1-2 · `estimate_gemm_latency_ms` TFLOPS hardcoded, not derived from the live GPU
`crates/grim-backend-vulkan/src/lib.rs:3282-3289`
- **Bug class:** hardcoded where live data should be.
- **Evidence:** `ArithType::F16|BF16 => 100.0; F32 => 50.0; _ => 30.0` and `flops / (tflops*1e12)`. These are generic desktop-GPU guesses; the ROCm crate's `capability_profiler::arch_tflops_table` (`capability_profiler.rs:189-210`) already measures per-gfx FP16/FP32 peak TFLOPS. If the heuristic ever feeds the autotuner's placement decision, every non-integrated-GPU device is mis-scored.
- **Recommended fix:** Drive `tflops` from the physical device's Vulkan `maxComputeSharedMemorySize`/subgroup / `vendorID`/`deviceID` or at minimum accept it as a constructor parameter from the caller's measured profile. Mark the constants `const VULKAN_GUESS_*` and add a TODO, or delete the method if it has no real consumer (its only caller is itself).

### P1-3 · `epoch_bumped` guard removed — capability epoch now flips per-GPU per-tick
`crates/grim-backend-rocm/src/device/capability_profiler.rs:78-89` (diff removed `epoch_bumped` flag)
- **Bug class:** silent fail / behavioral regression (correctness of capability invalidation).
- **Evidence:** Previously, one epoch bump per refresh across all GPUs. Now `bump_epoch()` fires for **every** GPU whose throttle delta exceeds 10% in a single refresh, so a refresh that sees 4 GPUs cross the threshold flips the global epoch 4×. Since `bump_epoch` invalidates cached capabilities / retune hints (per the §3.6 comment), this can thrash the GEMM autotuner / graph-cache mid-inference.
- **Recommended fix:** Restore the `epoch_bumped` latch (one bump per refresh), or document why per-GPU cascading bumps are intended and debounce `bump_epoch` (e.g. idempotent within N ms).

---

## P2 — Medium (orphaned / dead code / minor silent fail)

### P2-1 · Orphaned `pub fn for_device_with_capacity` with no caller and ignored `dev`
`crates/grim-backend-rocm/src/graph_capture.rs:101-110`
- **Bug class:** orphaned + borderline silent-fail.
- **Evidence:** `pub fn for_device_with_capacity(_dev: &RocmDevice, max_entries: usize) -> Self` — the `_dev` parameter is taken (comment says no HIP call fires here), but it is the only public constructor besides `for_device`. `grep` shows zero callers outside the `for_device` wrapper at `:104`. Per clean-code-guard YAGNI (no present-day caller) and the DIP rule, this speculative capacity-tuning knob should be inlined or removed until a second caller exists.
- **Recommended fix:** Make `for_device_with_capacity` `pub(crate)` (or delete, bake `64` into `for_device`), or actually use `dev` to bind the capture stream to that device's ordinal.

### P2-2 · Orphaned `NCCL_BFLOAT16` / `NCCL_FLOAT8` constants, `NCCL_FLOAT8` value likely wrong
`crates/grim-backend-rocm/src/rccl.rs:25-26` (added in diff `+2`)
- **Bug class:** orphaned + (latent) correctness.
- **Evidence:** `grep` across all `crates/` finds only the definitions; no `match`/call site. `NCCL_FLOAT8 = 10` is not a valid RCCL `ncclDataType_t` (the official enum stops at `ncclBfloat16 = 9`; FP8 is handled via `ncclFloat8_e4m3fn`/`e5m2` extended APIs, not value 10). Unused + wrong → dead landmine.
- **Recommended fix:** Delete both constants (no consumer), or wire them into `arith_to_nccl_dtype` with a real FP8 path gated on `GcnArch::RDNA4|CDNA3` per the `rust-ml-llm-review` ROCm checklist (`fp8 paths gated on target_gfx >= gfx1200`).

### P2-3 · `VulkanHandle` doc claims `is_ready()` "always true" — invariant is load-bearing and untested
`crates/grim-backend-vulkan/src/lib.rs:702-706` + the 4 fast-paths returning `VulkanHandle` / `ReadyHandle` inconsistently
- **Bug class:** silent fail (invariant drift).
- **Evidence:** The new doc says `run_compute_shader` calls `vkQueueWaitIdle` synchronously, so `synchronize()` no-ops / `is_ready()` true. But the four fast paths return **different** handle types: `selective_scan`→`ReadyHandle` (`lib.rs:2378`), `flash_attention`→`VulkanHandle` (`lib.rs:2476`), `rwkv_time_mix`→`ReadyHandle` (`lib.rs:2538`), `rwkv_channel_mix`→`ReadyHandle` (`lib.rs:2609`). That inconsistency plus the silent-fallthrough (P0-1) means a caller relying on the documented invariant can read partially-written output.
- **Recommended fix:** Pick one handle type for all GPU-fast-path returns; add a smoke test that asserts `is_ready()` after each.

---

## Ponytail-review pass (over-engineering / dead affordance only)

`graph_capture.rs:79-118`: shrink — `GraphCacheState { cache, lru }` merged-lock refactor is good but `pub fn for_device_with_capacity` (P2-1) is the only new surface; inline it into `for_device`. `net: -22 lines possible` (drop the 2nd constructor + its test).

`lib.rs:3374-3389`: shrink — `VulkanAutotuner::estimate_gemm_latency_ms` exposed `pub` for no caller (P1-2). Make it `pub(crate)` or delete.

`rccl.rs:25-26`: delete (P2-2). `net: -2 lines possible`.

`accel_ffi.rs:96-108`: delete (P1-1). `net: -13 lines possible` (or restore cleanup + add `Drop`).

---

## Items reviewed and cleared (negative results, to confirm coverage)

- **`wmma_gemm.rs` 2D-grid rewrite** (`wmma_gemm.rs:9-41` + `roc_device.rs:3715-3731`): the original kernel only stored one 16×16 tile to `C[0,0]`; the new 2D (`blockIdx.y`=M-tile, `blockIdx.x`=N-tile) + `a_tile_ptr/b_tile_ptr/c_tile_ptr` base offsets is a **genuine correctness fix**. `stride_b = n` as the `col_major` fragment leading dim for a row-major `B[K,N]` is the standard rocBLAS/rocWMMA row↔col transpose convention — verified against the rocBLAS path at `roc_device.rs:1441-1448` which uses the same `(n, K*N)` layout. Native-WMMA block size (W32, blockSize 32) matches RDNA3/RDNA4 wavefronts. Not a bug.
- **`f32::from_bits((num_kv_heads<<16)|(cache_offset&0xffff))`** vs the old `(... ) as f32` (`lib.rs:2048`): correct fix — the old cast reinterpreted an arbitrary 17-bit integer as an `f32` exponent/mantissa payload (UB on the deferred decode path); `from_bits` is the right SPIR-V push-constant payload carry. Not a bug.
- **`vram_info` returning `None` instead of `(total,total)`** (`lib.rs:3893-3902`): the only consumer `grim-server/src/lib.rs:3914` already uses `let Some(...) else`, matching the CUDA/Metal `Option` convention. Correct.
- **`gemm_tuning::lookup_solution_index` arch gate** (`gemm_tuning.rs:99-103` `if !arch.contains("1036") return 0`): conservative-safe (off-arch gets the default `solution_index=0` → rocBLAS picks its own algo). Slower, not wrong. The passing `arch` plumbing at `roc_device.rs:870,1392,3674,3746,3888` is consistent.
- **`quantized_matmul_backward_dx_generic.comp` outlier binary-search + backup1/backup2 residuals + STE `grad_scale`**: internal math is internally consistent (`flat_weight_idx = col*K + k_idx` matches the forward packing), MXFP E8M0 shared exponent bias `127.0` correct (verified `kernels/shared_device_fns.rs:47`). Negative-scale guard at `lib.rs:2902-2908` is redundant with the clamp but harmless.
- **`GraphCaptureManager` LRU refactor** (`graph_capture.rs:80-237`): merged `cache + lru` under one mutex (avoids the prior two-lock deadlock-prone dance) — correct; eviction order and the cloned `Arc` hit-path are sound.
- **`capability_profiler::arch_tflops_table` gfx9xx branches** (`capability_profiler.rs:189-203`): MI300X (1307/2614 TF, 5300 GB/s), MI250X (383/383, 3200), MI100 (184.6, 1228.8) values check out against published AMD datasheets.

---

## Top-3 to fix first
1. **P0-2** `RcclAllReduce::init_comm` — multi-GPU training stack-smash + handle leak; ship-blocking for any 2+-GPU run.
2. **P0-1** Four Vulkan fast-paths silently return wrong data on binding mismatch; gate on a golden test, not `is_ok()`.
3. **P0-3** u8-clamped scales corrupt quantized-backward gradients for any `|scale|>1`; rewrite the scale encoding.
