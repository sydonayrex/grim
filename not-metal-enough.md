# not-metal-enough.md — Review of `grim-backend-metal`

**Subject:** `crates/grim-backend-metal` — the intended Grim Apple-Silicon (Metal/MSL) backend for serving `.grim` files on macOS.
**Compared against:** `grim-backend-vulkan` and `grim-backend-cuda`.
**Date:** 2026-07-19
**Verdict:** Not production-viable. It is a **CPU fallback shim wearing a Metal badge**. Every compute op is routed to `grim-backend-cpu`; the `objc2`/Metal code paths are allocation + `memcpy` only. The `mods/` directory is science-fiction about the Apple Neural Engine that never executes. On Apple Silicon it would be *slower* than CPU for the same work because it round-trips through CPU anyway.

---

## 1. Capability parity matrix

| Capability (trait method / extra) | CUDA | Vulkan | **Metal** |
|---|---|---|---|
| Real device compute (GPU) | ✅ cuBLAS + hand-written kernels | ✅ SPIR-V compute shaders (+ sim fallback) | ❌ **CPU fallback only** |
| `zeros` | ✅ GPU alloc | ✅ GPU alloc | ⚠️ Apple: alloc+bzero; else Vec |
| `from_cpu` | ✅ `cudaMemcpy` H2D | ✅ `vkMapMemory` | ⚠️ Apple: alloc+copy; else Vec |
| `to_cpu_vec_f32` | ✅ `cudaMemcpy` D2H | ✅ `vkMapMemory` | ⚠️ Apple: `buffer.contents()` copy; else Vec |
| `matmul` | ✅ cuBLAS `sgemm` (row/col-major corrected) | ✅ SPIR-V + autotuner | ❌ **`cpu_dev.matmul`** |
| `add` | ✅ kernel `grim_add` | ✅ SPIR-V | ❌ **`cpu_dev.add`** |
| `mul` | ✅ kernel `grim_mul` | ✅ SPIR-V | ❌ **`cpu_dev.mul`** |
| `silu_mul` | ✅ kernel `grim_silu_mul` | ✅ SPIR-V | ❌ **`cpu_dev.silu_mul`** |
| `rms_norm` | ✅ kernel `grim_rms_norm` | ✅ SPIR-V | ❌ **`cpu_dev.rms_norm`** |
| `softmax` | ✅ kernel `grim_softmax` | ✅ SPIR-V | ❌ **`cpu_dev.softmax`** |
| `embedding` | ✅ kernel `grim_embedding` | ✅ SPIR-V | ❌ **`cpu_dev.embedding`** |
| `matmul_with_solution` | ✅ trait default → `matmul` | ✅ trait default → `matmul` | ✅ trait default (inherited) |
| `advise` (MemAdvice) | ✅ no-op | ✅ no-op | ✅ no-op |
| `qkv_attention` (extended) | ✅ real kernel | ⚠️ (referenced as ROCm parity target; not in this crate) | ❌ **absent** |
| Autotuner / tile config | n/a (cuBLAS) | ✅ `VulkanAutotuner` | ❌ absent |
| Kernel JIT cache | ✅ `JIT_CACHE` (seahash) | ✅ precompiled SPIR-V blobs | ❌ absent |
| FFI error safety (no panics) | ✅ docs + guards | ✅ ret codes | ⚠️ `expect()` on device creation |
| `probe()` real device enumeration | ✅ `cudaGetDeviceCount` | ✅ real Vk enumeration | ❌ **returns `vec![new(0)]` hardcoded** |

**Conclusion of the matrix:** CUDA/Vulkan implement the *entire* trait surface with real device execution (with graceful host-simulation fallback). Metal implements the *trait signatures* but **zero** device kernels. It is a behavioral clone of `grim-backend-cpu` with a Metal buffer as the storage carrier.

---

## 2. What the "Metal" path actually does

`src/lib.rs` is a thin UMA wrapper. On `target_vendor = "apple"`:

- `zeros` / `from_cpu` → `device.newBufferWithLength_options(... StorageModeShared)` then a `memcpy`/`write_bytes`. (Buffer creation only.)
- Every *compute* op (`matmul`, `add`, `mul`, `silu_mul`, `rms_norm`) → `run_fallback_binary`, which:
  1. `to_cpu_vec_f32` out of the Metal buffer (which is already CPU-visible Shared memory, so this is a copy),
  2. builds a `CpuDevice`, runs the op **on the CPU**,
  3. copies the result back into a new Metal buffer.
- `softmax` / `embedding` → same shape, inlined rather than via the helper.

So on Apple Silicon the "Metal backend" is strictly **worse than just using the CPU backend**: it adds two extra buffer copies per op and an `objc2` dependency for no compute benefit. There is **no MSL shader, no `MTLLibrary`/`MTLComputePipeline`, no `MTLCommandEncoder` dispatch, no `dispatch_async`/encoder calls anywhere in the crate**. The word "Metal" in the compute sense appears nowhere.

This is the textbook `rust-gpu-discipline` violation: the work is claimed done (`is_ready` returns `true` on non-Apple; `synchronize` is a no-op) while secretly executing on CPU.

---

## 3. `mods/` — dead, misleading, and non-executing

`src/mods/{ane,bridge,npu}.rs` are referenced by `lib.rs` (`pub mod mods;`) but **never used** by the backend. They are:

- **`ane.rs`** — `AneGraphBuilder` emits fake `.mil` text strings to a `File`. No compilation, no ANE runtime link, no `CoreML`/`MLModel`. It is a string formatter with a `#[test]` that asserts a file was written.
- **`bridge.rs`** — `AneBridgeClient` is a `println!` wrapper. On Apple it returns `Ok` with a `null_mut()` raw pointer and `dispatch()` prints a message. The doc-comment claims "private FFI bridge to `AppleNeuralEngine.framework`" — there is **no FFI, no `extern`, no linkage**. On non-Apple it correctly errors with `Unimplemented`; on Apple it lies with `Ok`.
- **`npu.rs`** — `NpuExecutor` / `NpuDeviceDescriptor` are a `println!` + struct holder. No hardware, no backend integration.

These violate the ponytail-reviewer rule explicitly: **delete speculative abstractions that do nothing.** They also violate `honesty`/docs-guard — the doc comments describe capabilities ("compilation of mega-kernels for ANE hardware execution", "FFI bindings to the private ANE driver") that do not exist. That is not a stub-marked `TODO`; it is documentation that asserts shipped functionality.

There is no `// TODO`/`todo!()`/`unimplemented!()` discipline here — it's worse: it *looks* complete and *tests green*.

---

## 4. Robustness / correctness gaps (ponytail + rust-review lens)

1. **`MetalDevice::new` panics on Apple.** `MTLCreateSystemDefaultDevice().expect(...)` — an `expect` on an FFI boundary is UB-adjacent and breaks the `rust-ffi-grim` §1.2 "never panic across FFI" rule that the CUDA crate follows (`map_err` + `Error::Backend`). A headless Mac (no GPU, e.g. CI, VM, remote) turns this into a process abort instead of a graceful `Err`.
2. **No `Drop` for `MetalStorage`.** Vulkan and CUDA free their device memory in `Drop` (Vulkan: `vkDestroyBuffer`/`vkFreeMemory`; CUDA: `cudaFree`). Metal storage just drops the `Retained<MTLBuffer>` — *probably* fine via `objc2` ARC, but there is no explicit `Drop`, and the non-Apple variant leaks the `Mutex<Vec<f32>>` handle across the `Box<dyn BackendStorage>` boundary with no lifetime owner, unlike the Apple path.
3. **`dtype_byte_size` is Apple-only.** On non-Apple, `zeros`/`from_cpu` hardcode `vec![0.0f32; elem_count]` — so F16/BF16/U8/U32 inputs are silently misinterpreted as F32 by element count. CUDA/Vulkan size by `dtype_byte_size` (F16=2, U8=1, …). The Metal non-Apple path will corrupt non-F32 shapes. (Also `dtype_byte_size` is marked `#[cfg(target_vendor = "apple")]` and unused-`dead_code`-free only because the Apple call sites are also Apple-gated.)
4. **No F32 dtype guard.** CUDA rejects non-F32 with `Error::DTypeMismatch` and documents why. Metal accepts anything and either (a) on Apple, allocates by `elem_count * 4` bytes regardless of `dtype`, or (b) on non-Apple, stores `Vec<f32>`. Both are silently wrong for quantized `.grim` payloads.
5. **`probe()` is hardcoded.** CUDA enumerates real devices; Vulkan enumerates + checks; Metal returns `vec![MetalDevice::new(0)]` unconditionally — it reports one device even on Linux/Windows where the crate still compiles and "passes" tests via the `not(apple)` branch. That means `grim` could "select" the Metal backend on a non-Apple host and run everything on CPU while advertizing Metal.
6. **`run_fallback_binary` always allocates via `CpuDevice` copy** — O(N) extra copies on every dispatch. No reuse, no streaming, no chunking. For `.grim` model serving (large matmuls) this is a correctness-safe but performance-catastrophic design.

---

## 5. Test quality (strong-tests / tdd lens)

Tests pass (6/6) — but they prove almost nothing about Metal:

- All run on the **non-Apple** path (this host is Linux). The `not(target_vendor="apple")` branch just uses `Vec<f32>`. So `test_metal_matmul` is really `test_cpu_matmul_through_a_vec`. The Apple code paths (`newBufferWithLength_options`, `buffer.contents()`, `waitUntilCompleted`, `status()`) have **zero** test coverage and **zero** compile-exercised logic on CI.
- Tests assert happy-path numerics only. No:
  - shape-mismatch / non-2D matmul rejection (CUDA/Vulkan test and reject),
  - dtype-mismatch rejection,
  - `to_cpu_vec_f32` null-buffer error path (the `Error::Backend("Metal buffer contents is null")` branch is untested),
  - `synchronize`/`is_ready` semantics on a real command buffer,
  - device-absent / `new()` failure handling.
- `mods/ane.rs` test only asserts a file exists after writing — it does not validate the MIL is parseable or that any ANE path exists.
- **No integration test** that exercises `BackendDevice` as a trait object the way serving code would (the other backends are selected dynamically; this one would be selected and silently CPU-bound).

Following `strong-tests`: the suite has **high coverage %, near-zero mutation resistance** — mutating `run_fallback_binary` to skip the CPU call would still pass every test because none assert device execution.

---

## 6. Software_Factory / project-planning read

As the designated macOS serving backend, `grim-backend-metal` is **incomplete by definition**:

- It does not serve `.grim` on Apple Silicon with GPU acceleration. It serves them via CPU, indistinguishable from `grim-backend-cpu` except for extra copies.
- The ANE path (the *actual* differentiator for Apple Silicon inference — NPU is dramatically more efficient than GPU for transformer decode) is pure fiction in `mods/`.
- There is no `build.rs`, no shader compilation, no MSL source, no MTL capture, no performance parity target vs CUDA/Vulkan.

This is a **scaffold mislabeled as an implementation**. The honest status is: *Metal buffer allocation + CPU fallback exist; Metal compute does not.*

---

## 7. Recommendation — what "metal enough" looks like

Priority order (TDD: write the failing tests first):

1. **Rename or label honestly.** Either mark the crate `#![doc = "CPU-fallback compatibility shim"]` or gate `pub mod mods;` behind a feature and label it experimental. Stop asserting ANE/FFI functionality that does not exist.
2. **Implement real Metal compute** (the gap that makes it "not metal enough"):
   - Add MSL kernel sources (`kernels.msl`) mirroring CUDA's `kernels.rs` (`grim_add`, `grim_mul`, `grim_silu_mul`, `grim_rms_norm`, `grim_softmax`, `grim_embedding`), compiled via `objc2-metal` `MTLLibrary`/`newLibraryWithSource:` + `newComputePipelineStateWithFunction:`.
   - Dispatch through `MTLCommandQueue` + `MTLCommandBuffer` + `MTLComputeCommandEncoder` in each `BackendDevice` op. Keep the UMA Shared-storage zero-copy path (already correct) so `to_cpu_vec_f32` stays a direct read.
   - Add `matmul` via Metal Performance Shaders (`MPSMatrixMultiplication`) or a tiled MSL kernel with the autotuner pattern Vulkan already has.
3. **Add `qkv_attention`** for transformer serving parity (CUDA has it; Metal needs it for `.grim` decode).
4. **Fix robustness:** `new()` returns `Result` (no `expect`), add `Drop`/explicit resource handling, `dtype`-aware sizing on both cfg branches, `probe()` returns empty on non-Apple (don't advertise a Metal device where none exists).
5. **Delete or feature-gate `mods/`** until the ANE bridge has real `extern` FFI and a working dispatch; replace the misleading docs with `TODO`s.
6. **Tests:** add Apple-path tests gated behind `#[cfg(target_vendor="apple")]` (run in CI on a Mac runner), plus negative tests (shape/dtype mismatch, null contents) and a behavioral test asserting the op actually ran on-device (e.g., timestamp or a `MTLCommandBuffer` status assertion) so a future regression to CPU-fallback would fail.

---

### One-line summary
`grim-backend-metal` satisfies the *type signature* of a backend but not its *contract*: it allocates Metal buffers and then computes everything on the CPU, while `mods/` narrates an ANE/NPU story that never runs. Compared to the real GPU execution in `grim-backend-cuda` and `grim-backend-vulkan`, it is not metal enough — it is a CPU backend with a Metal-shaped shell.
