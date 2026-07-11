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

1. Implement a caching allocator on `RocmDevice`: a size-bucketed free-list keyed by **power-of-two rounding** (round each requested allocation up to the next power of two; the bucket key is simply the rounded size). Use power-of-two rounding specifically, not a hand-picked fixed set of size classes — this workload's allocation-size distribution (activations, KV-cache chunks, quantized weight buffers) isn't characterized well enough yet to hand-tune a fixed table, and power-of-two gives a trivial, correct-by-construction bucket lookup with bounded worst-case waste (at most 2x). Store as `Mutex<HashMap<usize /* rounded size */, Vec<*mut c_void>>>` of freed-but-not-`hipFree`'d buffers per device.
2. `RocmStorage::alloc_gpu` should first check the pool for a free buffer in the bucket matching the requested size (rounded up) before calling `hipMalloc`.
3. `Drop for RocmStorage` should return the buffer to the pool instead of calling `hipFree`, **unless the pool's total bytes held (sum of all buffers across all buckets, not buffer count) exceeds a cap.** Buffer *count* is not an acceptable cap metric — a handful of large weight-buffer allocations and hundreds of small activation buffers are not comparable, and a count-based cap doesn't bound actual memory usage. Default the cap to 25% of the free-memory value reported by `hipMemGetInfo` at the time `RocmDevice` is constructed, and expose it as a constructor parameter (e.g. `RocmDevice::new_with_pool_cap_bytes(ordinal, cap_bytes)`) so callers can override it.
4. Add an explicit `RocmDevice::empty_cache()` method (mirroring `torch.cuda.empty_cache()`) that walks the pool and calls `hipFree` on everything, for callers that need to release memory back to the OS/driver between distinct workloads.
5. Document the power-of-two bucketing scheme and the byte-cap default directly in a doc comment on the allocator struct — this is the "configurable and documented" requirement; no separate configuration file or runtime-tunable size-class table is needed for this first version.

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
4. Because dispatches now round-robin across pooled streams *and* no longer sync per-call, dependent kernel launches need explicit handling so a later launch doesn't run before the data it depends on is ready. **Default to pinning a whole logical op chain to a single stream** — this is the simpler of the two possible approaches and sufficient for now, since nothing in this crate currently runs multiple independent op chains concurrently (there's no concurrent multi-request handling yet, so there's nothing to gain from finer-grained interleaving). Concretely: change `launch_compute_kernel`'s signature to accept an optional `stream: Option<*mut c_void>` parameter. Any Rust-level function that internally issues more than one dependent kernel launch (e.g. `rmsnorm_matmul`) must pick a single stream from the pool up front and pass it explicitly to every sub-launch in that call, rather than letting each sub-launch round-robin independently and land on different streams. Do **not** implement `hipEventRecord`/`hipStreamWaitEvent`-based cross-stream dependency tracking in this pass — that's real added complexity, and there's no current use case (concurrent independent op chains) that would benefit from it. Revisit only if a later profiling pass shows single-stream pinning is measurably leaving overlap on the table.

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

1. **Do not guess which call sites are on the hot path from memory — enumerate and classify them first, as its own sub-step:**
   a. Grep `src/lib.rs` for every call to `copy_from_host` and `to_cpu_vec_f32`.
   b. For each call site found, record the enclosing function name and one line of context.
   c. Classify each one as either **(hot)** — invoked once per generated token (e.g. writing the newly sampled token ID into device memory before the next forward pass, or reading final-layer logits back to host for sampling) — or **(cold)** — invoked at most once per model load or per test (e.g. initial weight upload, one-off debug dumps). This classification must come from tracing each call site's actual caller chain, not from assuming based on function name.
   d. Only **(hot)**-classified call sites get the pinned+async treatment below. Leave every **(cold)** call site as-is and note in a comment why it was excluded (e.g. `// cold path: one-time weight load, not per-token`).
2. For each **(hot)** call site: switch its host-side staging buffer to `hipHostMalloc`-allocated (pinned) memory instead of a plain `Vec`. Add a small pinned-buffer wrapper type (allocate via `hipHostMalloc`, free via `hipHostFree` in `Drop`) that is allocated once and reused across steps, not allocated fresh per call.
3. Switch these **(hot)** transfers to `hipMemcpyAsync` on an appropriate stream, with an explicit `hipStreamSynchronize` only at the point the host actually needs the data (e.g. right before returning `Vec<f32>` from `to_cpu_vec_f32`, or before sampling from logits).
4. Leave every **(cold)**-classified call site as a synchronous, pageable-memory `hipMemcpy` — converting them isn't required and would just add complexity to paths that don't affect tokens/sec.

### Acceptance criteria

- The enumerated, classified call-site list from step 1 is included in the PR/commit description, not just implied by the diff — so a reviewer can check the hot/cold classification itself, not just the resulting code.
- Every **(hot)**-classified call site uses the pinned+async path; every **(cold)**-classified one is either unchanged or explicitly justified in a comment.
- A benchmark comparing per-token host-round-trip latency before/after, on whatever ROCm hardware is available for testing.

---

## Item 5: Generic graph-capture session API (not a hardcoded "decode step")

**Status (DONE):** Implemented and validated on gfx1036. 43 lib tests (incl. the 3 Item 5 tests) pass, stable across 5+ multithreaded runs under `GRIM_RUN_GPU_TESTS=1 GRIM_CAPTURE_GRAPH=1`. The validation test compares the captured-graph output against a **CPU-computed reference** of the `matmul`→`add`→`rms_norm` sequence, not against a GPU-eager run — rocBLAS picks a different GEMM algorithm for the captured path than for an eager path (and `rms_norm` amplifies that difference), so a GPU-eager cross-check is invalid. Capture stream is owned for the device's lifetime (destroyed in `Drop`) to avoid a rocBLAS-teardown abort; relaxed capture mode (`hipStreamBeginCapture(stream, 2)`) is used so `hipMalloc`/rocBLAS workspace allocation is permitted during capture; rocBLAS handle is bound to the capture stream for the whole bracket.

The existing `GRIM_CAPTURE_GRAPH` path captures one GEMM call. Real payoff from HIP graphs comes from capturing a whole sequence of ops into one graph and replaying it, eliminating per-kernel launch overhead across that sequence.

**Scope note:** `grim-backend-rocm` is a generic backend exposing primitive ops (`matmul`, `add`, `mul`, `silu_mul`, `rms_norm`, `softmax`, `embedding`, `rmsnorm_matmul`, `qkv_attention`). It has no model-level orchestration function anywhere in it, and this item must not add one. Do **not** implement a `capture_decode_step()` (or similarly named) function that bakes in a specific op sequence like "QKV → attention → O-proj → MLP" — that sequence is a property of whatever model-runner code calls this crate, which lives outside it. Instead, this crate exposes a **capture session**: a begin/end bracket around whatever sequence of existing op calls the caller makes, plus a keyed replay lookup. The caller decides what's inside the bracket.

**Do this only after Items 1 and 2 are done**, for two concrete reasons, not just general caution:
- HIP graphs require stable device pointer addresses across replays. Item 1's caching allocator (once buffers are reused per shape rather than freshly `hipMalloc`'d) is what makes that stability possible.
- `hipStreamSynchronize` on a stream that is currently being captured returns `hipErrorStreamCaptureUnsupported` — it doesn't just add overhead, it breaks capture outright. Item 2's removal of the unconditional sync in `launch_compute_kernel` is a hard prerequisite for this reason specifically, not only a performance one.

### Required API

```rust
impl RocmDevice {
    /// Begin recording. All ops called on `self` between this and the matching
    /// `end_graph_capture` are redirected onto a dedicated capture stream instead
    /// of the normal stream pool, and are not eagerly synchronized.
    pub fn begin_graph_capture(&self, key: &str) -> Result<()>;

    /// End recording, instantiate the graph, and cache it under `key`.
    /// Returns without launching — capture and replay are separate calls.
    pub fn end_graph_capture(&self, key: &str) -> Result<()>;

    /// Replay a previously captured+instantiated graph for `key`.
    /// Returns `Ok(false)` (not an error) if no graph is cached under `key` yet,
    /// so the caller's fallback is "capture this time, replay next time," not
    /// error-handling.
    pub fn replay_graph(&self, key: &str) -> Result<bool>;
}
```

Intended caller-side usage (outside this crate, shown only for context — do not implement this part):

```rust
let key = format!("decode_b{}_kv{}", batch_size, kv_len); // caller picks the key
if !dev.replay_graph(&key)? {
    dev.begin_graph_capture(&key)?;
    // ... whatever op calls the eager path would make: matmul, qkv_attention, add, etc.
    dev.end_graph_capture(&key)?;
    dev.replay_graph(&key)?;
}
```

### Required changes

1. **Keying is caller-supplied, not auto-detected.** Do not implement shape fingerprinting or any other mechanism to auto-detect "did the input shapes change since the last capture under this key." The backend does a plain exact-string lookup on `key`; a cache miss means the caller must capture again. This is the entire mechanism for handling shape changes between prefill and decode, or KV-cache length growth — there is no additional shape-change-detection logic to build.
2. Add a `capture_stream: RwLock<Option<*mut c_void>>` field on `RocmDevice`. `begin_graph_capture` creates a dedicated stream, calls `hipStreamBeginCapture` on it, and stores it here.
3. Every op-dispatch function that currently picks a stream (`launch_compute_kernel`'s stream-pool lookup, `matmul`'s `rocblas_set_stream` call) must check `capture_stream` first: if `Some`, use that stream instead of the pool, and skip `hipStreamSynchronize` entirely while it's set.
4. `end_graph_capture` calls `hipStreamEndCapture`, `hipGraphInstantiate`, stores the resulting `(graph, exec)` pair in a `Mutex<HashMap<String, CapturedGraph>>` field on `RocmDevice` keyed by `key`, and clears `capture_stream`. `CapturedGraph` must implement `Drop` calling `hipGraphExecDestroy` then `hipGraphDestroy`.
5. `replay_graph` looks up `key` in that map; if present, calls `hipGraphLaunch(exec, stream)` on a normal pooled stream and returns `Ok(true)`; if absent, returns `Ok(false)`.
6. Build this by generalizing the existing single-GEMM `GRIM_CAPTURE_GRAPH` block in `src/lib.rs` (it already does `hipStreamBeginCapture` → `hipStreamEndCapture` → `hipGraphInstantiate` → `hipGraphLaunch` → cleanup correctly for one GEMM) rather than writing the HIP-call sequence from scratch.
7. Keep the existing `GRIM_CAPTURE_GRAPH` env var as the opt-in mechanism, but read it once at `RocmDevice::new` and cache the bool, rather than re-reading the environment on every call. When capture is disabled, `begin_graph_capture`/`end_graph_capture` must be no-ops and `replay_graph` must always return `Ok(false)`, so a caller's `if !replay_graph { begin; ...; end; }` pattern behaves identically (falls through to eager execution every time) whether the feature is enabled or not — no separate code path needed at the call site.

### Acceptance criteria

- A test that runs a synthetic multi-op sequence using ops that already exist in this crate (e.g. `matmul` → `add` → `rms_norm` — do not invent a fake "decode step" for the test) once eagerly and once via `begin_graph_capture`/.../`end_graph_capture`/`replay_graph`, and asserts the two outputs match within `f32` tolerance.
- A test that captures under one `key`, then calls `replay_graph` with a *different* `key`, and asserts it returns `Ok(false)` rather than replaying the wrong graph.
- A benchmark comparing wall-clock time of N repeated eager calls vs. one capture + N replays for the same synthetic op sequence, on real hardware.

---

## Item 6: No batched/strided-batched GEMM

**Status (DONE):** Implemented and validated on gfx1036. `rocblas_gemm_strided_batched_ex` FFI declared with the verified 29-arg signature; `RocmDevice::matmul_batched` packs inputs into contiguous device buffers via D2D copies, calls the strided-batched GEMM (row-major recipe via swapped operands, matching `matmul`), and returns per-batch device storages. The rocBLAS handle is bound to the active stream for the call so the D2D copies land before the GEMM reads them, then restored. A one-time warm-up GEMM (2x2 batch=2) absorbs rocBLAS's lazy first-call JIT race that otherwise intermittently zeroed the first batched output. Correctness test `matmul_batched_matches_loop_of_single_gemms` (batches 1,3,5) passes; 44 lib tests stable across 5 multithreaded runs under `GRIM_RUN_GPU_TESTS=1`. Scoped to the `matmul_batched` primitive only — no cross-crate wiring (per spec).

All matmuls go through single-GEMM `rocblas_sgemm`/`rocblas_gemm_ex`. Any batch size above 1 (even small decode-time batches across multiple concurrent sequences) currently means one GEMM dispatch per sequence per layer.

### Required changes

1. Add `rocblas_gemm_strided_batched_ex` (and/or `rocblas_sgemm_strided_batched`) to the `extern "C"` block with a correct signature (cross-check against rocBLAS docs the same way Item 0 required — do not repeat that mistake).
2. Add `RocmDevice::matmul_batched(&self, a: &[&dyn BackendStorage], b: &[&dyn BackendStorage], out_shape: &Shape) -> Result<Vec<Box<dyn BackendStorage>>>` (or an equivalent signature that collapses a batch of same-shape GEMMs into one `rocblas_gemm_strided_batched_ex` call).
3. **Scope stop:** this crate exposes `matmul_batched` as a standalone, independently tested primitive and goes no further. Do not search for, modify, or wire this into call sites in other crates in this workspace (e.g. wherever a model-runner might handle multiple concurrent sequences) — that integration is out of scope for a `grim-backend-rocm`-only change and belongs in a separate ticket against whichever crate owns batched-inference orchestration.

### Acceptance criteria

- Correctness test: batched GEMM output matches running the equivalent single GEMMs in a loop, for several batch sizes.
- The FFI signature is verified against official rocBLAS documentation/headers before merging, with the same rigor as Item 0 (this function has the same historical risk of parameter-order mistakes).

---

## Item 7: `solution_index`/`algo` tuning (depends on Item 0)

Once Item 0 gives `rocblas_gemm_ex` real access to `algo` and `solution_index`, these are currently hardcoded to `0`/standard. Tie `solution_index` into the existing `lookup_gemm_config` shape-dispatch table: offline, use `rocblas_gemm_ex_get_solutions` to enumerate valid solutions for representative (m, n, k, dtype) shapes seen in real inference (prefill and decode), benchmark them, and record the fastest solution index per shape bucket. At runtime, `lookup_gemm_config` (or a new sibling table) returns a `solution_index` alongside its existing tile-config heuristics, and that value is passed into `rocblas_gemm_ex` instead of `0`.

**The offline tuning deliverable already exists and is broken — fix it, don't create a new one.** `examples/tune_gemm.rs` in this repo already has the right shape (a sweep over representative `(m, n, k)` shapes benchmarking solution indices), but it calls a method, `matmul_with_solution`, that was never implemented anywhere in `src/lib.rs`, so the example does not compile. Implementing that method is part of this item, not a separate task — do not consider this item done while `examples/tune_gemm.rs` still fails to build. Its exact required signature:

```rust
impl RocmDevice {
    pub fn matmul_with_solution(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        solution_index: i32,
    ) -> Result<Box<dyn BackendStorage>>;
}
```

### Required changes

1. Implement `matmul_with_solution` in `src/lib.rs` with the exact signature above (it should share almost all of `matmul`'s logic, differing only in passing `solution_index` through to `rocblas_gemm_ex` instead of `0`).
2. Confirm `examples/tune_gemm.rs` compiles and runs against this implementation without further edits to the example itself (if the example needs changes, that's a signal the signature above is wrong — fix the signature to match the example's actual needs rather than reshaping the example around an implementation of convenience).
3. Run the example (or an equivalent benchmark) to actually produce the shape→solution_index table, and check the resulting table into the repo (e.g. as a static table in `src/lib.rs` or a small data file loaded at startup) rather than leaving tuning as a manual step someone has to remember to run.

### Acceptance criteria

- `examples/tune_gemm.rs` compiles and runs successfully end-to-end.
- The shape→solution_index table it produces is checked into the repo (or the process to regenerate it is scripted and documented), not hand-guessed.
- Fallback to `solution_index = 0` for any shape not present in the table, so untuned shapes still work correctly, just without the tuned speedup.
- Benchmark showing improvement over the default heuristic for at least the shapes actually exercised by real prefill/decode traffic.

---

## Overall sequencing summary

0 (blocking correctness) → 1 and 2 (biggest per-token latency wins, can be done in parallel with each other) → 3 and 4 (smaller, independent, do anytime) → 5 (generic graph-capture session, depends on 1 and 2) → 6 and 7 (independent of 5, but 7 depends on 0).

Every item above must ship with the correctness test described in its acceptance criteria — none of these are "trust it because it compiles" changes, especially given Item 0 shows this codebase has already shipped one FFI signature that was wrong and passed casual review.

**Scope reminder that applies across Items 5 and 6 specifically:** this ticket is scoped to `grim-backend-rocm` only. Where an item's natural conclusion would be "and then wire it into the model-runner's decode loop / batched-inference path," stop at exposing the primitive API in this crate, tested in isolation. Do not modify or go looking for call sites in other workspace crates — say so explicitly in the PR description if a natural integration point isn't included, rather than silently expanding scope to chase it.
