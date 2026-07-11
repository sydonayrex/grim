# Task: Fix `rocblas_gemm_ex` ABI mismatch and close remaining ROCm performance gaps

## Context

This is `grim-backend-rocm`, a ROCm/HIP backend crate for LLM inference (`src/lib.rs`, `src/gptq_kernel.rs`, `src/fusion.rs`, `src/rocm_pad.rs`). A prior pass already fixed several issues (fp32-only GEMM dispatch, hot-path logging, HIP graph capture correctness for a single GEMM, stream-pool round-robin, JIT `.hsaco` cache reuse, wavefront padding, GCN target coverage, and disabling a broken fused-attention kernel). This ticket covers what's still outstanding, ordered by priority. **Work through items in order — later items assume earlier ones are done, and item 0 is a correctness bug that must land before anything else is trusted.**

Do not batch unrelated changes into one commit. Each numbered item below should be its own change with its own tests, so a regression can be bisected to one cause.

---

## Item 0 (BLOCKING — do this first): `rocblas_gemm_ex` FFI declaration does not match the real ABI

**This is a correctness bug, not a performance bug.** The `extern "C"` declaration for `rocblas_gemm_ex` in `src/lib.rs` currently has 17 parameters with the `A_type`/`B_type`/`C_type` datatype enums bundled at the end. The real rocBLAS signature is:

```c
rocblas_status rocblas_gemm_ex(
    rocblas_handle    handle,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int       m,
    rocblas_int       n,
    rocblas_int       k,
    const void*       alpha,
    const void*       a,
    rocblas_datatype  a_type,
    rocblas_int       lda,
    const void*       b,
    rocblas_datatype  b_type,
    rocblas_int       ldb,
    const void*       beta,
    const void*       c,
    rocblas_datatype  c_type,
    rocblas_int       ldc,
    void*             d,
    rocblas_datatype  d_type,
    rocblas_int       ldd,
    rocblas_datatype  compute_type,
    rocblas_gemm_algo algo,
    int32_t           solution_index,
    uint32_t          flags);
```

The current binding is missing `d`, `d_type`, `ldd`, `compute_type`, `algo`, `solution_index`, and `flags` entirely, and interleaves the type enums in the wrong positions relative to their pointer/leading-dimension arguments. Calling the real `librocblas.so` symbol through the current binding pushes arguments into the wrong slots per the platform calling convention — this is undefined behavior: it can silently produce wrong GEMM results, corrupt memory, or crash, and it invalidates any correctness claims about the F16/BF16 dispatch path added in a previous pass.

### Required changes

1. Correct the `extern "C"` declaration in `src/lib.rs` to match the real signature exactly, including parameter order and types (`rocblas_gemm_algo` is an `i32`-backed enum; add it plus `rocblas_gemm_flags` as needed — define minimal enum/constants if not already present, e.g. `ROCBLAS_GEMM_ALGO_STANDARD = 0`).
2. Update every call site (`RocmDevice::matmul`'s normal path and its HIP-graph-capture duplicate) to pass:
   - `d = c` (same pointer — in-place output, matching current single-buffer-out behavior) and `d_type` = the output dtype (same value currently passed as `c_type`/what will become `out_type`).
   - `compute_type` = the existing `arith_to_rocblas_dtype`-derived compute dtype logic.
   - `algo = 0` (`rocblas_gemm_algo_standard`), `solution_index = 0`, `flags = 0` as safe defaults for now (Item 7 revisits `solution_index`).
3. Grep the whole crate for every call to `rocblas_gemm_ex` (there are at least two: the normal dispatch path and the HIP-graph-capture path) and fix both — don't fix only one and leave the other on the old broken signature.
4. Add a build-time or test-time static assertion / doc comment pinning this signature to the rocBLAS version this crate targets, and a comment explaining why every parameter is there, so a future edit doesn't silently drop one again.

### Acceptance criteria for Item 0

- A test that runs a small F16 or BF16 GEMM through `RocmDevice::matmul` (on the `gemm_ex` path) and compares the result against a CPU-computed reference within tolerance. This is the regression test that would have caught the original bug — it must exist and must actually exercise `use_gemm_ex = true`, not silently fall through to `rocblas_sgemm`.
- Confirm (by code inspection, not just by it compiling) that both call sites — normal dispatch and graph-capture — were updated identically.

---

## Item 1: No caching/pooling GPU allocator — `hipMalloc`/`hipFree` on every op

`RocmStorage::alloc_gpu` calls raw `hipMalloc` and `Drop for RocmStorage` calls raw `hipFree`, and every op (`matmul`, `add`, `mul`, `silu_mul`, `rms_norm`, `softmax`, `embedding`, `rmsnorm_matmul`, `qkv_attention`, `zeros`) goes through `alloc_gpu` for its output. `hipMalloc`/`hipFree` are effectively device-synchronizing driver calls. In a token-by-token decode loop this means dozens of malloc/free round-trips per token — almost certainly the largest fixed per-token cost in the crate.

### Required changes

1. Implement a caching allocator on `RocmDevice`: a size-bucketed free-list (e.g. round each allocation up to the next power-of-two or a fixed set of size classes, keep a `Mutex<HashMap<usize, Vec<*mut c_void>>>` of freed-but-not-`hipFree`'d buffers per device).
2. `RocmStorage::alloc_gpu` should first check the pool for a free buffer of adequate size before calling `hipMalloc`.
3. `Drop for RocmStorage` should return the buffer to the pool instead of calling `hipFree`, unless the pool is above a configurable size cap (then actually free it, to bound memory growth).
4. Add an explicit `RocmDevice::empty_cache()` method (mirroring `torch.cuda.empty_cache()`) that walks the pool and calls `hipFree` on everything, for callers that need to release memory back to the OS/driver between distinct workloads.
5. Make the size-class scheme configurable or at least documented, since GPTQ/quantized weight buffers and activation buffers will have very different size distributions.

### Acceptance criteria

- A test or benchmark that runs N forward/decode steps in a loop and asserts the number of raw `hipMalloc` calls (instrument with a counter behind a test-only feature flag, or log-and-count under `rocm-profile`) is roughly constant after a warmup period, not O(N) per allocation site.
- No memory growth over a long-running loop with fixed input shapes (steady-state pool reuse, not unbounded caching).

---

## Item 2: `hipModuleLoad`/`hipModuleUnload` on every kernel launch, plus an unconditional sync

In `launch_compute_kernel` (`src/lib.rs`), the compiled `.hsaco` bytes are cached now, but the *loaded module* is not: `hipModuleLoad` and `hipModuleUnload` run around every single dispatch, re-registering the code object with the driver each call. The function also unconditionally calls `hipStreamSynchronize` before returning, which — combined with the stream pool added in a prior pass — means round-robining across streams buys nothing, since each call blocks until its own stream finishes before the next op is even issued.

### Required changes

1. Add a module cache on `RocmDevice` (or alongside `HsacoKernelCache`): `Mutex<HashMap<String /* cache_key */, (*mut c_void /* module */, *mut c_void /* function */)>>`. Load once per unique kernel per process lifetime; look up on subsequent calls instead of reloading.
2. Unload modules only in `Drop for RocmDevice` (alongside the existing stream-pool and rocblas-handle cleanup), not per-call.
3. Remove the unconditional `hipStreamSynchronize` from `launch_compute_kernel`. Callers that need the result synchronously (e.g. before a `to_cpu_vec_f32` read-back) should synchronize explicitly at the point they need the data, not inside every intermediate kernel dispatch. Audit every current caller of `launch_compute_kernel` to confirm none of them depend on the implicit sync for correctness (e.g. two dependent kernels launched back-to-back on different pooled streams need either the same stream or an explicit event/dependency — see the note on stream affinity below).
4. Because dispatches now round-robin across pooled streams *and* no longer sync per-call, add explicit dependency handling where two kernel launches have a data dependency but land on different streams (e.g. `hipEventRecord`/`hipStreamWaitEvent`, or simplest: pin a whole fused op chain to a single stream rather than round-robining sub-kernels of the same logical op across different streams).

### Acceptance criteria

- Loaded-module count (instrumented similarly to Item 1's malloc counter) stays constant after warmup across repeated calls to the same kernel entry point.
- Correctness tests for every op that uses `launch_compute_kernel` (add, mul, silu_mul, rms_norm, softmax, embedding, rmsnorm_matmul) still pass with the sync removed — if any fail intermittently, that's a real missing dependency, not a flaky test; fix it with an explicit sync/event rather than reintroducing the blanket sync.

---

## Item 3: `zeros()` round-trips through host memory instead of `hipMemset`

`BackendDevice::zeros` (`src/lib.rs`) allocates a full host `Vec<f32>` of zeros and does an H2D `hipMemcpy` to zero device memory.

### Required changes

1. Add `hipMemset` to the `extern "C"` block (`hipError_t hipMemset(void* dst, int value, size_t sizeBytes)`).
2. Replace the host-buffer-allocate-and-copy logic in `zeros()` with a direct `hipMemset(dev_ptr_void, 0, storage.bytes)` call. Note `hipMemset` zeroes bytes, so this is only valid for dtypes whose zero representation is all-zero bytes (true for `f32`/`f16`/`bf16`/integer types — confirm this holds for every `DType` this function can be called with before relying on it).

### Acceptance criteria

- `zeros()` no longer allocates a host-side `Vec`.
- Existing correctness tests for `zeros()` (or add one if missing) confirm the resulting buffer is actually all-zero after the change, for every dtype `zeros()` supports.

---

## Item 4: Synchronous, pageable-memory `hipMemcpy` for host transfers

`RocmStorage::copy_from_host` and `to_cpu_vec_f32` use blocking `hipMemcpy` from/to plain (pageable) `Vec` buffers. Pageable transfers get a fraction of achievable PCIe/xGMI bandwidth versus pinned (`hipHostMalloc`) memory, and blocking copies stall the calling thread instead of overlapping with other work. An async path already exists elsewhere in the file (`hipMemcpyAsync`, used in a fallback branch) but isn't used for these two functions.

### Required changes

1. For the token-generation hot path specifically (feeding a sampled token back in, reading logits/next-token back out) — identify these call sites and switch their host-side staging buffers to `hipHostMalloc`-allocated (pinned) memory instead of a plain `Vec`. This likely means adding a small pinned-buffer wrapper type (allocate via `hipHostMalloc`, free via `hipHostFree` in `Drop`) reused across steps rather than allocated fresh each time.
2. Switch these transfers to `hipMemcpyAsync` on an appropriate stream, with an explicit `hipStreamSynchronize` only at the point the host actually needs the data (e.g. right before returning `Vec<f32>` from `to_cpu_vec_f32`, or before sampling from logits).
3. Leave `copy_from_host`/`to_cpu_vec_f32` as synchronous, pageable-memory versions for cold-path uses (one-off weight loading, debugging, tests) if a fully async version would complicate their call sites — but the decode-loop-critical paths (wherever per-token host round trips happen) must use the pinned+async version.

### Acceptance criteria

- Identify (grep/trace) every call site in the per-token decode path that touches host memory, and confirm each either uses the new pinned+async path or is justified as off the hot path in a comment.
- A benchmark comparing per-token host-round-trip latency before/after, on whatever ROCm hardware is available for testing.

---

## Item 5: HIP graph capture only wraps a single GEMM

The existing `GRIM_CAPTURE_GRAPH` path captures one GEMM call. Real payoff from HIP graphs comes from capturing an entire decode step (QKV projection → attention → output projection → MLP) into one graph and replaying it, eliminating per-kernel launch overhead across the whole step.

**Do this only after Items 1 and 2 are done** — HIP graphs generally require stable device pointer addresses across replays, which conflicts with an allocator that calls `hipMalloc` mid-graph (Item 1's caching allocator, once buffers are stable/reused per step, resolves this) and with per-launch module load/unload churn (Item 2).

### Required changes

1. Design a `capture_decode_step(&self, ...) -> HipGraphExec` (or similar) API that runs one full decode step under `hipStreamBeginCapture`/`hipStreamEndCapture`, covering all the ops involved (not just one GEMM), and stores the resulting `hipGraphExec` for repeated `hipGraphLaunch` calls across subsequent tokens where shapes are unchanged.
2. Handle the case where shapes change between steps (e.g. prefill vs. decode, or KV-cache length growth) by falling back to recapture rather than replaying a stale graph.
3. Make this opt-in behind a feature/env var initially (as the single-GEMM version was), with a clear migration path to being the default decode path once validated.

### Acceptance criteria

- Correctness: output of a graph-captured decode step matches the non-captured (eager) path exactly (or within f32 tolerance) for the same inputs.
- Performance: measurable reduction in per-token wall-clock time versus the eager path, on real hardware, not just launch-count reduction.

---

## Item 6: No batched/strided-batched GEMM

All matmuls go through single-GEMM `rocblas_sgemm`/`rocblas_gemm_ex`. Any batch size above 1 (even small decode-time batches across multiple concurrent sequences) currently means one GEMM dispatch per sequence per layer.

### Required changes

1. Add `rocblas_gemm_strided_batched_ex` (and/or `rocblas_sgemm_strided_batched`) to the `extern "C"` block with a correct signature (cross-check against rocBLAS docs the same way Item 0 required — do not repeat that mistake).
2. Add a batched matmul path on `RocmDevice` that collapses a batch of same-shape GEMMs into one strided-batched call instead of a loop of single GEMMs.
3. Wire this into wherever batched inference (multiple concurrent sequences) would call matmul, if that call site exists elsewhere in the workspace; if it doesn't yet exist, expose the API and leave integration as a follow-up, but don't leave the FFI binding unverified against the real signature.

### Acceptance criteria

- Correctness test: batched GEMM output matches running the equivalent single GEMMs in a loop, for several batch sizes.
- The FFI signature is verified against official rocBLAS documentation/headers before merging, with the same rigor as Item 0 (this function has the same historical risk of parameter-order mistakes).

---

## Item 7: `solution_index`/`algo` tuning (depends on Item 0)

Once Item 0 gives `rocblas_gemm_ex` real access to `algo` and `solution_index`, these are currently hardcoded to `0`/standard. Tie `solution_index` into the existing `lookup_gemm_config` shape-dispatch table: offline, use `rocblas_gemm_ex_get_solutions` to enumerate valid solutions for representative (m, n, k, dtype) shapes seen in real inference (prefill and decode), benchmark them, and record the fastest solution index per shape bucket. At runtime, `lookup_gemm_config` (or a new sibling table) returns a `solution_index` alongside its existing tile-config heuristics, and that value is passed into `rocblas_gemm_ex` instead of `0`.

### Acceptance criteria

- A documented offline tuning process (script or tool) that produces the shape→solution_index table, checked into the repo or regenerable, not hand-guessed.
- Fallback to `solution_index = 0` for any shape not present in the table, so untuned shapes still work correctly, just without the tuned speedup.
- Benchmark showing improvement over the default heuristic for at least the shapes actually exercised by real prefill/decode traffic.

---

## Overall sequencing summary

0 (blocking correctness) → 1 and 2 (biggest per-token latency wins, can be done in parallel with each other) → 3 and 4 (smaller, independent, do anytime) → 5 (depends on 1 and 2) → 6 and 7 (independent of 5, but 7 depends on 0).

Every item above must ship with the correctness test described in its acceptance criteria — none of these are "trust it because it compiles" changes, especially given Item 0 shows this codebase has already shipped one FFI signature that was wrong and passed casual review.
