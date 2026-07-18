# Plan: Port grim's ROCm HIP Kernels from C++ to Rust (cubecl)

**What we are building:** A plan and execution strategy to re-implement the 11
custom ROCm/HIP compute kernels currently authored as C++ source strings (JIT-compiled
via hipRTC) in `grim-backend-rocm` as native Rust GPU code using **cubecl**, so that no
C++ kernel source string remains in the crate. The ROCm/rocBLAS FFI layer for GEMM and
device memory is preserved — only the *custom kernels* move to cubecl.

> This document was produced with the `writing-product-specs` skill (spec structure),
> the `architecture-blueprint` skill (module layout + ADRs), and the ROCm conversion
> skills (`rocm-hip-kernels`, `rust-gpu-discipline`, `rust-ffi`/`rust-ffi-bindings`,
> `port-c-module`, `analyze-`/`minimize-rust-ffi-crate-surface`) for the conversion
> work-plan. The conversion skills live in `/home/nelson/.agents/skills/`.

---

## 0. Environment probe (rust-gpu-discipline §0 — measured, not assumed)

Run before planning any code:

```
rocminfo         -> gfx1036 (AMD Radeon 610M, RDNA2). Name amdgcn-amd-amdhsa--gfx1036
hipinfo          -> not installed
/opt/rocm        -> hip_runtime_api.h + librocblas.so PRESENT
libvulkan*       -> ABSENT  (no Vulkan loader)
vulkaninfo       -> present but no loader -> wgpu/vulkan backend unusable here
```

Consequences that shape this plan:
- **Target arch is `gfx1036` (RDNA2).** No native fp8 MFMA. All 11 kernels are currently
  `f32`-only, so this is not a blocker for the port, but fp8-attention work (Phase-2 of the
  existing kernel spec) is *not* a win on this box and must stay gated on `gcnArchName >= gfx1200`.
- **cubecl's WGPU/Vulkan backend cannot run on this machine** (no Vulkan loader). The
  **cubecl `hip` backend** (which links the HIP runtime directly) is the only viable cubecl
  path. The plan treats cubecl-hip as the execution backend and gates any fallback on it.

### Per-arch toolchain layout (discovered — drives toolchain selection)
The repo ships three side-by-side ROCm toolchains as hidden folders at the repo root, all
now **Linux** builds, each targeting a different RDNA generation:
- `.rocm-2/` — **RDNA2 (gfx10xx)**. Linux `libamdhip64.so`, `amdclang`. **HIP 7.15.** This is
  the arch-matched toolchain for this box's gfx1036 (RDNA2).
- `.rocm-3/` — **RDNA3 (gfx11xx)**. Linux, HIP 7.15. Wrong arch for this box.
- `.rocm-4/` — **RDNA4 (gfx12xx)**. Linux, HIP 7.15 (ships `libhipblaslt.so`, `librccl.so`,
  `llvm/`, nightly tarball). Wrong arch for this box.

`/opt/rocm` is **also** a Linux install but is **HIP 7.2** (older) — `build.rs` currently
hardcodes `rustc-link-search=native=/opt/rocm/lib`. cubecl-hip v0.10.0 compiles **only**
against HIP 7.2 (`/opt/rocm`), not the 7.15 `.rocm-N` folders (its `cubecl_hip_sys` bindings
don't resolve 7.15 symbols — verified in Phase 0). So **cubecl-hip links `/opt/rocm`**; the
`.rocm-N` set stays the toolchain for the existing hipRTC/rocBLAS C++ path. See ADR-5.
- ROCm is installed, so kernel builds + GPU runs are possible on this machine (subject to
  cubecl-hip backend maturity on gfx1036 — see Risk R1).

---

## 1. Background

### Context
`grim-backend-rocm` is the primary GPU backend for grim. It today drives the GPU two ways:
1. **rocBLAS / hipBLASLt** (via hand-written FFI) for GEMMs — correct, near-peak, keep as-is.
2. **Custom HIP kernels**, authored as **C++ source strings** embedded in `.rs` files, JIT-compiled
   at runtime through **hipRTC** (`hiprtcCreateProgram` → `hiprtcCompileProgram` →
   `hipModuleLoadData` → `hipModuleLaunchKernel`).

The custom kernels are:
- `src/kernels/compute_kernels.rs` → `OTHER_KERNEL_SOURCE` (7 kernels)
- `src/kernels/qkv_attention.rs` → `KERNEL_SOURCE` (3 kernels)
- `src/gptq_kernel.rs` → `GPTQ_CORRECTION_KERNEL` (1 kernel)

Dispatch flows through `launch_compute_kernel` (`lib.rs`) / `jit_compile_hsaco`
(`gptq_kernel.rs`) → hipRTC → module load → launch. Host launchers
(`launch_paged_attention`, `launch_tree_attention`) pack `&mut i32` args.

The user has chosen (via clarification) to **rewrite the kernel logic in a Rust GPU DSL
(cubecl) targeting ROCm — a true C++→Rust port with no C++ strings remaining**.

### Audience
- **grim backend engineers** maintaining `grim-backend-rocm`.
- **grim ML/kernel engineers** adding future attention/fusion/quant kernels.
- **CI/release owners** who must keep GPU tests green on gfx1036.

### Problem statements
- The custom kernels are C++ strings, not Rust: no type safety, no reuse, no IDE support,
  no compile-time checking of launch args against the Rust launcher.
- hipRTC JIT adds launch-time latency and a runtime compiler dependency; a kernel bug only
  surfaces at first launch, not at `cargo build`.
- The dispatch path duplicates a near-identical hipRTC wrapper (`device/helpers.rs` vs
  `gptq_kernel.rs` vs `device/roc_device.rs`) — maintenance drag.
- Some existing kernels only use wave 0 of a 256-thread block (see `qkv_attention.rs`:
  `if (wave_id > 0) return;`) — a correctness-of-intent / occupancy smell worth fixing while
  re-authoring, not carrying forward.

---

## 2. Hypothesis

Re-expressing the 11 kernels in cubecl (a Rust GPU DSL that compiles to HIP on RDNA) removes
the C++ strings, gives compile-time launch-arg checking, unifies dispatch, and lets future
kernels (paged/tree/quant attention) be added in Rust. GEMM stays on rocBLAS via FFI
(rocm-hip-kernels **Rule 0 — don't reinvent GEMM**), so porting risk is limited to the
elementwise/attention/correction kernels, which are exactly the ones the vendor libs don't
provide. Keeping rocBLAS FFI means we only need to bridge cubecl's device tensors to the
existing `RocmStorage`/`BackendStorage` and bind cubecl's client to the same HIP stream.

---

## 3. Success criteria (rust-gpu-discipline §4 must pass for each kernel)

- **No C++ kernel string remains** in `grim-backend-rocm` after the final phase.
- **Each ported kernel executes on the GPU** through cubecl-hip (not a CPU fallback).
  Mechanical evidence: the cubecl `launch`/`launch_kernel` call is in the path, and a test
  constructs a device tensor and asserts `result.device().is_hip()` (or cubecl-equivalent)
  plus value-equivalence against the existing CPU reference.
- **No silent CPU fallback** (rust-gpu-discipline §3). If cubecl-hip is unavailable, the op
  returns `Err(...)`, mirroring PyTorch `NotImplementedError`. Opt-in fallback (env var +
  `tracing::warn!`) is the *only* allowed fallback and is default-off.
- **Wave64 honored** (rocm-hip-kernels): every cubecl workgroup dimension is a multiple of
  64; the plan records the CubeDim per kernel.
- **Numerically equivalent** to the prior C++ kernel within tolerance (f32) on gfx1036,
  verified by the existing test files (`tests/qkv_attention.rs`, `tests/paged_attention.rs`,
  `tests/tree_attention.rs`, `tests/fusion_smoke.rs`, `tests/quantization.rs`, etc.) — each
  gets a cubecl-path assertion that is *not* `#[ignore]`d and not gated behind an env var.
- **Build & GPU tests green**: `cargo build -p grim-backend-rocm` and
  `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm` (or the crate's GPU-test switch)
  pass on gfx1036.

---

## 4. Requirements

### Functional
- Re-implement all 11 kernels in cubecl: `grim_add`, `grim_mul`, `grim_silu_mul`,
  `grim_rms_norm`, `grim_softmax`, `grim_embedding`, `grim_rmsnorm_matmul`,
  `grim_qkv_attention`, `grim_qkv_attention_paged`, `grim_tree_attention`,
  `gptq_wavefront_correction_kernel`.
- Preserve each kernel's exact *contract* (layout, dtypes, masking, GQA mapping) as documented
  in `grim_qkv_attention_kernel_spec.md`.
- Keep GEMM on rocBLAS FFI (unchanged).
- Wire cubecl so it shares the existing `RocmDevice` HIP context/stream (no second device).

### Technical
- Add `cubecl` (+ `cubecl-hip` backend) as a workspace dependency; pin a version after a
  spike (Phase 0) confirms gfx1036 compilation.
- Bridge layer (`kernels/cubecl_bridge`): convert `RocmStorage` device pointers ↔ cubecl
  tensors/handles without host round-trip (rust-gpu-discipline forbidden pattern #7/#13:
  no per-call DtoH readback in the hot path).
- Replace `launch_compute_kernel` / `jit_compile_hsaco` with a cubecl-based dispatch that
  still satisfies the `RocmDevice` trait methods the rest of the crate calls.
- Per-arch gating: fp8 paths (future) gated on `gcnArchName >= gfx1200`; this box stays f32.

### Verification
- Each kernel: a runnable on-GPU test + a CPU-reference equivalence test (existing tests
  extended, not replaced).
- `rocminfo`/arch probe at startup selects the cubecl-hip target; kernels cached by
  (source-hash, arch) like today's `jit_cache.rs`.

---

## 5. Non-requirements (explicitly out of scope)

- **Rewriting GEMM in cubecl.** rocBLAS stays (Rule 0).
- **Switching to WGPU/Vulkan cubecl backend.** Not viable without a Vulkan loader on this box;
  revisit only if a Vulkan loader appears.
- **Porting the rocBLAS/hipRTC FFI itself to Rust.** The FFI to rocBLAS remains; only hipRTC
  (the kernel JIT) is removed once all kernels are on cubecl.
- **fp8/BF16/MFMA kernels.** Phase-2 of the existing spec; deferred, and invalid on gfx1036
  anyway. Gate on arch.
- **Changing tensor/dtype schemas or the `BackendStorage` trait.** Bridge only.

---

## 6. Architecture (architecture-blueprint skill)

### Module layout (additions inside `grim-backend-rocm/src`)
```
src/
  kernels/
    mod.rs
    compute_kernels.rs      # (post-port) re-exports cubecl impls; C++ string DELETED
    qkv_attention.rs        # (post-port) cubecl impls; C++ string DELETED
    gptq_kernel.rs          # (post-port) cubecl impl; C++ string DELETED
    cubecl_runtime.rs       # NEW: cubecl Client/device init, stream binding, cache
    cubecl_bridge.rs        # NEW: RocmStorage <-> cubecl tensor/handle zero-copy bridge
    cubecl_kernels/         # NEW: the actual .cl (cubecl Rust) kernel modules
        add.rs  mul.rs  silu_mul.rs  rms_norm.rs  softmax.rs
        embedding.rs  rmsnorm_matmul.rs
        qkv_attention.rs  qkv_attention_paged.rs  tree_attention.rs
        gptq_correction.rs
```
- **Domain-first, locality over layering**: all custom-kernel code lives under `kernels/`,
  co-located. The bridge and runtime are platform glue, not business logic, so they sit beside
  the kernels (not in a separate `ffi/` tree) — they only exist to satisfy the existing
  `RocmDevice`/`BackendStorage` boundary.
- **No new abstraction unless it earns its place.** `cubecl_runtime` and `cubecl_bridge` are
  the minimum shim to keep `RocmDevice`'s public trait stable for the rest of the crate.

### ADRs (record as decisions)
- **ADR-1: cubecl-hip backend, not WGPU/Vulkan.** *Because* no Vulkan loader on target box.
  Revisit if Vulkan loader added.
- **ADR-2: GEMM stays on rocBLAS FFI; cubecl only for custom kernels.** *Because* Rule 0
  (don't reinvent GEMM) and to limit port risk.
- **ADR-3: cubecl shares the existing HIP stream via a handle bridge, no second device.**
  *Because* multi-device/context splits break rocBLAS↔kernel interleaving (rust-ffi ROCm
  section: bind handle to the active stream).
- **ADR-4: incremental port behind a dispatch toggle; remove C++ strings only after the
  cubecl path is verified per-kernel.** *Because* port-c-module discipline + rust-gpu-discipline
  §3 (don't flip a gate to true until verified).
- **ADR-5: cubecl-hip v0.10.0 binds `/opt/rocm` (HIP 7.2), NOT the `.rocm-N` folders.**
  *Why:* the `.rocm-2/3/4` Linux toolchains are **HIP 7.15**, but `cubecl-hip`'s
  `cubecl_hip_sys` bindings were generated against the 7.2-era API — they fail to compile
  against 7.15 (`hipSetDevice`, `hipMemGetInfo`, `hipGetDevicePropertiesR0600` are
  unresolved). The Phase-0 spike proved `cubecl-hip 0.10.0` **compiles and launches on
  gfx1036 only against `/opt/rocm` (HIP 7.2)**. Therefore `build.rs` keeps linking
  `/opt/rocm/lib`, and `ROCM_PATH` defaults there. (If a future cubecl version supports HIP
  7.15, this flips to `.rocm-2`.) The `.rocm-N` folders remain the per-arch toolchain for
  the **existing hipRTC/rocBLAS** C++ path, not for cubecl.
- **ADR-6: cubecl `#[cube(launch)]` kernels take array buffers via `ArrayArg`; bare `f32`
  scalar launch args don't work** (produced all-zero output in Phase 1 — kernel body never
  wrote). Use cubecl's proper scalar-binding form, or pass scalars as single-element
  `Array`s / hardcode constants inside the kernel. Plan all custom kernels (attention
  `inv_sqrt_d`, etc.) accordingly.

---

## 7. Kernel inventory & C++ → cubecl mapping

| # | Kernel | Today (file) | CubeDim (Wave64) | cubecl module | Notes |
|---|--------|--------------|------------------|---------------|-------|
| 1 | `grim_add` | compute_kernels.rs | (n+63)/64 × 1 | `add.rs` | trivial elementwise |
| 2 | `grim_mul` | compute_kernels.rs | (n+63)/64 × 1 | `mul.rs` | trivial elementwise |
| 3 | `grim_silu_mul` | compute_kernels.rs | (n+63)/64 × 1 | `silu_mul.rs` | gate `silu` in kernel |
| 4 | `grim_rms_norm` | compute_kernels.rs | (rows,1,1) blk 256 | `rms_norm.rs` | row reduce → use cubecl cube/unit |
| 5 | `grim_softmax` | compute_kernels.rs | (rows,1,1) blk 256 | `softmax.rs` | row reduce + online max/sum |
| 6 | `grim_embedding` | compute_kernels.rs | (total+63)/64 | `embedding.rs` | gather |
| 7 | `grim_rmsnorm_matmul` | compute_kernels.rs | (m,n) tiles | `rmsnorm_matmul.rs` | norm fused before matmul epilogue |
| 8 | `grim_qkv_attention` | qkv_attention.rs | (seq,heads) blk 256 | `qkv_attention.rs` | online softmax, causal, GQA, f32 |
| 9 | `grim_qkv_attention_paged` | qkv_attention.rs | (batch,heads) blk 256 | `qkv_attention_paged.rs` | block-table gather |
| 10 | `grim_tree_attention` | qkv_attention.rs | (1+γ,heads,batch) blk 256 | `tree_attention.rs` | ancestor-mask attention |
| 11 | `gptq_wavefront_correction_kernel` | gptq_kernel.rs | (cols,rows) | `gptq_correction.rs` | group-map elementwise |

All are **f32** on this arch. No MFMA needed for the port (MFMA is a later perf optimization,
not a correctness requirement — rocm-hip-kernels).

---

## 8. Conversion work-plan (phased, skill-driven)

### Phase 0 — Spike: cubecl-hip on gfx1036 (rust-gpu-discipline §0/§1)
- Add `cubecl` + `cubecl-hip` to a throwaway example; compile and run a trivial kernel on
  gfx1036. Confirm the hip backend links against `/opt/rocm` and launches.
- **Gate:** if cubecl-hip cannot compile/launch on gfx1036, STOP and revisit (Risk R1) —
  do NOT silently fall back to keeping C++ and reporting "done".

### Phase 1 — Runtime + bridge scaffolding
- `kernels/cubecl_runtime.rs`: init cubecl `Client` on the HIP device; bind to the existing
  `RocmDevice` stream; kernel cache keyed by (hash, arch) replacing `jit_cache.rs` role for
  custom kernels.
- `kernels/cubecl_bridge.rs`: zero-copy `RocmStorage` (device_ptr + len + dtype) ↔ cubecl
  `Tensor`. No host readback (forbidden patterns #7/#13).
- Skill: `rust-ffi` (ROCm section — opaque handles, status checks), `rust-ffi-bindings`
  (safe boundary, `#[repr(C)]` where crossing to existing FFI), `rust-gpu-parallelism`
  (stream-ordered, graph capture compat with existing `graph_capture.rs`).

### Phase 2 — Port the 7 elementwise/reduction kernels (lowest risk first)
- Port kernels 1–7. Each: cubecl module + bridge call + test.
- Skill: `port-c-module` (analyze → plan → implement → equivalent tests → wire-up).
- For each: add a runnable GPU test asserting device residency + CPU-reference equivalence.

### Phase 3 — Port the 3 attention kernels (highest value)
- Port kernels 8–10. Carry forward the *contract* from `grim_qkv_attention_kernel_spec.md`
  (causal mask, GQA `kv_head = h/(num_heads/num_kv_heads)`, online softmax, f32-only).
- **Fix the known bugs while re-authoring** (do NOT port them):
  - The spec's confirmed `num_kv_heads` hardcode bug.
  - **`head_dim > 64` returns NaN** in the current C++ kernels (audit `grim-sonnet.md`
    FINDING 2): all three attention kernels `return nan` for `head_dim > 64` because they
    map `d = lane_id` against a single 64-lane wavefront. Modern LLMs (Llama-3 head_dim=128,
    Mistral=96) silently NaN today. The cubecl port MUST tile `head_dim` across the
    wavefront (cubecl `cube`/shared-memory tiling), so `head_dim` up to 256 works — this is
    a *hard requirement* of the port, not a nice-to-have, because it unblocks every
    production model.
  - The `wave_id > 0` early-return (uses only 1 of 4 wavefronts) — replace with proper
    cubecl cube reduce.
- Skill: `rocm-hip-kernels` (Wave64, LDS/tiling analog in cubecl shared memory, occupancy),
  `rust-gpu-discipline` §2 (no fake-GPU, no deferred kernel, no CPU detour).

### Phase 4 — Port the GPTQ correction kernel
- Kernel 11. Keep the CPU fallback in `grim-quant` as the documented slow path; the cubecl
  path is the GPU fast path, gated by `enabled` (ADR-4), returns `Err` if cubecl-hip absent.

### Phase 5 — Remove C++ strings + minimize FFI surface
- After Phase 4 verified, delete `OTHER_KERNEL_SOURCE`, `KERNEL_SOURCE`,
  `GPTQ_CORRECTION_KERNEL` and the `launch_compute_kernel`/`jit_compile_hsaco` hipRTC paths.
- Skill: `analyze-rust-ffi-crate-surface` then `minimize-rust-ffi-crate-surface` — enumerate
  now-unused hipRTC symbols (`hiprtcCreateProgram`, etc.) and remove them from `device/handles.rs`,
  `device/helpers.rs`, `device/roc_device.rs`, `lib.rs` re-exports. Keep rocBLAS FFI.
- Collapse the duplicated hipRTC wrappers.

### Phase 6 — Verification sweep (rust-gpu-discipline §4)
- 4a Backend evidence: grep each op for a cubecl `launch`/GPU call — zero matches = not ported.
- 4b CPU detour: grep for `.cpu()`/DtoH in hot path — justify any survivor (only end-of-pipeline
  user readback allowed).
- 4c Stub residue: no `todo!`/`unimplemented!` in any ported path.
- 4d Test reachability: every new path has a runnable on-GPU test, no `#[ignore]`.
- 4e Adversarial review: answer "where is the GPU compute? show the launch. show the test that
  fails on CPU-only."

---

## 9. Skill → work mapping

| Phase | Skills applied |
|-------|----------------|
| 0 | `rust-gpu-discipline` (§0 probe, §1 pre-flight) |
| 1 | `rust-ffi`, `rust-ffi-bindings`, `rust-gpu-parallelism` |
| 2–4 | `port-c-module`, `rocm-hip-kernels`, `rust-gpu-discipline` (§2 forbidden, §3 no-silent-fallback) |
| 3 | `rocm-hip-kernels` (Wave64/LDS/occupancy), `rust-ml-llm-debugging` (CPU ref vs GPU) |
| 5 | `analyze-rust-ffi-crate-surface`, `minimize-rust-ffi-crate-surface` |
| 6 | `rust-gpu-discipline` (§4 mechanical checks), `rocm-profiling-perf` (occupancy/tok-s) |

Repo-level: `architecture-blueprint` (layout + ADRs), `writing-product-specs` (this doc).

---

## 10. Tradeoffs & concerns

- **R1 (biggest): cubecl-hip backend maturity on gfx1036 is unverified.** If the Phase-0 spike
  fails to launch a kernel, the whole "true Rust port" goal is blocked on this hardware and we
  must either (a) get a Vulkan loader + use cubecl-wgpu, or (b) accept that the only
  Rust-authored path available is a different DSL. This is the first thing to resolve, not the last.
- **CubeDim/Wave64 semantics differ from raw HIP.** cubecl abstracts the wavefront; we must
  still ensure workgroup sizes are multiples of 64 and that reductions use cubecl's
  cube/unit primitives rather than hand-rolled `__shfl_xor`.
- **Bridge cost:** zero-copy bridge between `RocmStorage` and cubecl tensors must avoid
  re-allocation; if cubecl requires its own storage type, we pay a wrapping cost but no data copy.
- **Two dispatch styles during transition** (C++ hipRTC + cubecl) increase surface until Phase 5;
  the dispatch toggle (ADR-4) keeps them isolated per-kernel.
- **Future fp8 attention** remains a rocm-hip-kernels/rocm-quantization-inference concern,
  gated on gfx1200 — explicitly *not* part of this port.

---

## 11. Open items resolved by existing policy (rust-gpu-discipline §3)
- "Should cubecl fall back to CPU if hip unavailable?" → **No.** Return `Err`. Opt-in env-gated
  fallback only, default-off, `tracing::warn!` per call. This is not an open question.
- "Keep GEMM in cubecl or rocBLAS?" → rocBLAS (Rule 0). Not an open question.

---

## 12. Execution progress log (evidence)

### Phase 0 — PASSED (gate met)
- Spike crate `/tmp/grim-cubecl-spike` added `cubecl` v0.10.0 (`features=["hip"]`) +
  `cubecl-hip` v0.10.0. Both compile and **launch a real kernel on gfx1036**.
- Verified output correct: `PASS: cubecl-hip launched add kernel on gfx1036, 256 elems correct`.
- **Toolchain finding:** cubecl-hip 0.10.0 compiles **only** against HIP 7.2 (`/opt/rocm`);
  the HIP 7.15 `.rocm-N` toolchains break its `cubecl_hip_sys` bindings. See ADR-5.
- Saved reusable skill `gpu/cubecl-hip-spike` (crate names, feature flag, launch API,
  version trap, bare-f32 scalar gotcha).

### Phase 1 — PASSED (spike)
- `cubecl` runtime client + bridge primitive proven: allocate cubecl buffers, launch,
  read back, assert. The `RocmStorage.device_ptr` (u64 hipMalloc) is not adoptable by
  cubecl's opaque `Handle` at 0.10, so the bridge allocates cubecl buffers + D2D HIP copy
  (no host readback in hot path — forbidden #7). See ADR-6 re: bare-f32 scalar args
  silently producing all-zero output.

### Phase 2 — IN PROGRESS (RED→GREEN proven for elementwise)
- Added `src/lib.rs` kernels + `tests/phase2_elems.rs` TDD integration tests (run on gfx1036,
  assert device output == CPU reference).
- `grim_add` + `grim_mul`: **PASS** (`test result: ok. 2 passed`).
  ```
  running 2 tests
  test grim_add_matches_cpu_reference ... ok
  test grim_mul_matches_cpu_reference ... ok
  ```
- Remaining P2 kernels: `silu_mul`, `rms_norm`, `softmax`, `embedding`, `rmsnorm_matmul`.
  Reductions (`rms_norm`, `softmax`) will use `plane_sum`/`plane_max` + `SharedMemory`
  (cubecl's LDS analog) — the same primitives Phase 3 attention needs. Spike lives in
  `/tmp/grim-cubecl-spike`; modules get lifted into `grim-backend-rocm` (Phase "lift").

### Next decision
- Autonomously continue P2 reductions -> P3 attention (with head_dim>64 NaN fix) -> P4 GPTQ ->
  P5 FFI cleanup -> P6 sweep. OR lift the proven foundation into `grim-backend-rocm` first.
  Both paths keep TDD (function + integration, on-GPU assertions).
