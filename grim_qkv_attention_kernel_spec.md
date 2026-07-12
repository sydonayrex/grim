# Task: Implement `grim_qkv_attention` (fused QKV attention HIP kernel)

## Context

`grim-backend-rocm` has a fused QKV-attention path. The Rust entry point `RocmDevice::qkv_attention()` in `src/lib.rs` launches `grim_qkv_attention` **unconditionally, with no gate** — there is no `enabled` field on `QkvAttentionFusionConfig` and no `Err` guard anywhere in this function. (An earlier draft of this task doc claimed the kernel was already "disabled on purpose" behind `enabled: false` — that was checked against the real source and is false. Correcting it here since it changes what Step 3 needs to do: add a gate, not preserve one.) The existing `grim_qkv_attention` HIP source (in `COMPUTE_KERNEL_SOURCE` in `src/lib.rs`) is a non-functional stub that runs live today:

- It never indexes `k_tensor` by sequence position — the same K vector is reused for every loop iteration.
- It never applies softmax.
- It never reads `v_tensor` at all.
- It overwrites the same `out[head * head_dim]` slot on every loop iteration instead of writing a distinct result per position.

**Also confirmed as a real, separate bug**, not just a missing feature: `RocmDevice::qkv_attention()` currently hardcodes `num_kv_heads: num_heads / 4` in the `QkvAttentionFusionConfig` it builds, instead of taking `num_kv_heads` as an actual input. `grim-cli/src/run.rs` reads `num_heads` and `num_kv_heads` from GGUF as independent metadata fields (`attention.head_count`, `attention.head_count_kv`) with no fixed ratio between them. Any model with a GQA ratio other than 4:1 — or no GQA at all — would get silently wrong `kv_head` mapping from this hardcode. Fix this as part of Step 3: `qkv_attention()` must take `num_kv_heads` as a real parameter (or derive it from the actual K/V tensor shape it receives), never assume a ratio.

There is currently no production call site that reaches this function at all: `grim-cli`'s model loader forces CPU device for the real forward pass regardless of ROCm availability, and `grim-memory`'s `PagedKvCache::current_k()`/`current_v()` — the paged KV cache that would eventually need to feed this — are stubbed as `Err("not implemented in v1 paged cache")`. So this is genuinely dead code today, not "disabled" dead code — nothing currently calls it, correctly or incorrectly, at runtime. That's better than "live and being called wrong," but it also means Step 0's investigation will not find a real caller to confirm the layout against; treat the contract in Step 0 as authorized-by-you, not verified-in-the-wild.

Your job is to replace the kernel stub with a correct, reasonably performant fused attention kernel, fix the `num_kv_heads` hardcode, wire it up behind a real `enabled` gate (new, not existing), and only flip that gate to `true` once verified.

## Files involved

- `src/lib.rs` — `COMPUTE_KERNEL_SOURCE` (find `grim_qkv_attention`), and `RocmDevice::qkv_attention()`.
- `src/fusion.rs` — `QkvAttentionFusionConfig` and `hip_launch_params()`.
- `tests/fusion_smoke.rs` — existing geometry tests; update these and add correctness tests.

## Skills map

Each section below is informed by the following skills (created or updated in `~/.agents/skills/` during the ROCm skills rebuild). Use these as the scope-of-authority when scoping or reviewing each section.

| Section | Skills to apply (`~/.agents/skills/`) |
|---|---|
| Context, Step -1 (env probe), Step 0 (memory contract) | `rust-gpu-discipline` (env probe, PyTorch parity, no silent fallback); `rust-ml-llm-architecture` (backend isolation); `rust-gpu-parallelism` (RDNA-Wave64 + device dispatch) |
| Step 1 (algorithm) | Math reference — no skill-driven edits beyond what's in `rust-ml-llm-architecture` and `rust-ai-ml-inference-guide` Action 1/5 |
| Step 2 (parallelization) | `rocm-hip-kernels` (Wave64 mandate, LDS, occupancy, online softmax); `rust-gpu-parallelism` (CPU↔GPU dispatch) |
| Step 3 (wire-up: `enabled` gate, real `num_kv_heads`) | `rust-gpu-discipline` (§3 hide-when-disabled is PyTorch-parity); `rust-ml-llm-architecture`; `rust-ml-llm-review` |
| Step 4 (testing) | `rust-gpu-discipline` (§4 test reachability, JIT warm-up); `rust-ml-llm-debugging` (CPU reference vs GPU); `rocm-profiling-perf` (warm-up, isolation) |
| Acceptance criteria | `rust-gpu-discipline`; `rust-ml-llm-review` |
| Status of open items | `rust-gpu-discipline` (§3 defer-for-coordination policy: PyTorch-clear answers aren't open questions) |
| Phase 2 (performance) | `rocm-profiling-perf` (autotune loop, bottleneck metrics); `rocm-hip-kernels` (MFMA, tiling, fused kerns); `cuda-on-rocm` (only when reusing CUDA source/binaries) |
| 3.1 Device scratch pool | `rust-ai-ml-inference-guide` Action 3 (memory pool, KV-cache scratch); `rust-gpu-parallelism` (stream-ordered `hipMallocAsync`); `rocm-profiling-perf` (allocation overhead) |
| 3.2 HIP graph capture | `rust-gpu-parallelism` (graph capture section); `rust-ai-ml-inference-guide` Action 9 (graph capture for decode); `rocm-profiling-perf` (JIT warm-up guard) |
| 3.3 Quantized GEMM via MFMA | `rocm-quantization-inference` (fp8/int8, hipBLASLt scale descriptors, per-arch gating); `rocm-hip-kernels` (MFMA intrinsics, per-arch dispatch) |
| 3.4 Paged attention | `rocm-hip-kernels` (kernel authoring patterns); `rust-ai-ml-inference-guide` Action 3 (paged KV-cache as a layout) |
| 3.5 Speculative decoding | `rust-ai-ml-inference-guide` Actions 1, 5 (model formats, continuous batching, Jeet Kune Do); `rust-ml-llm-architecture` (orchestration) |
| 3.6 Autotuner per GPU arch | `rocm-profiling-perf` (autotune loop methodology and `rocblas_gemm_ex_get_solutions` runtime enumeration) |
| 3.7 Profiling CI gate | `rocm-profiling-perf` (regression gates, metric discipline); `rust-gpu-discipline` (no fake-GPU claims) |
| 3.8 Multi-GPU with RCCL | `rocm-multi-gpu-rccl`; `rust-gpu-parallelism` (one stream per device); `rust-gpu-scheduling` (per-slot VRAM) |
| Anti-Pattern Enforcement Checklist | `rust-gpu-discipline` (no silent fallback, file-size discipline); `rust-ffi` (SAFETY when modularizing FFI) |

⚠ **Corrections from the skills creation pass** that affect this spec are called out inline next to the affected sections. The most material ones:
- **3.3** — `QuantMode::Fp8` must gate on `target_gfx >= gfx1200`; emulated fp8 on gfx1036 is *slower than f16*.
- **3.6 / cross-task** — grim's existing `lookup_solution_index` table in `lib.rs` is currently a silent no-op (passes `algo::standard` together with a non-zero `solution_index`; rocBLAS ignores `solution_index` unless `algo == rocblas_gemm_algo_solution_index`).
- **3.8** — acceptance numbers assume MI300/xGMI but grim targets **consumer RDNA** (no Infinity Fabric between GPUs; PCIe only) — adjust expectations.
- **Phase 2 system-level** — modernize `rocprof`/`rocprofv2` references to `rocprof-compute`/`rocprofiler-sdk`/`rocprofv3` and update `--offload-arch` example to match grim's RDNA2 (`gfx1036`) or RDNA4 (`gfx1200`).

## Step -1 — Environment Probe (required per rust-gpu-discipline)

Before planning or writing code, run and record:
```
rocminfo | grep -A5 "Name.*gfx"   # use the .rocm-2/.rocm-3/.rocm-4 toolchain matching your GPU's RDNA gen (gfx1036 / gfx110x / gfx1200)
hipinfo                            # query hipDeviceProp.gcnArchName, memoryPoolsSupported, wavefront_size, totalGlobalMem
hipcc --version
```
Confirm target arch (e.g. `gfx1036` for RDNA2, `gfx1100` for RDNA3, `gfx1200` for RDNA4) and ROCm version match workspace pin. This is the only acceptable basis for "I can't verify on this machine" — assumptions are forbidden.

## Step 0 — memory layout and parameter contract (CONFIRMED — implement as specified, no re-deriving)

This was previously left as "go find the caller, or guess the general case if you can't." That is not acceptable for a kernel where a wrong guess silently corrupts model outputs. The contract below is now authorized and final — do not treat it as a guess to re-derive:

1. Search the wider `grim` workspace for `.qkv_attention(` and read every call site, as a sanity check against the contract below.
2. If a call site contradicts the contract below, stop and flag the discrepancy before proceeding — do not silently pick one over the other.
3. If no call site is found (e.g. no caller exists yet in this repo), implement exactly the contract below. It is authorized; this is not a placeholder pending further sign-off.

**Contract (final):**

- `q`: `[seq_len, num_heads, head_dim]` — matches `out_shape`. General case (prefill or decode), not decode-only.
- `k`, `v`: `[kv_seq_len, num_kv_heads, head_dim]`.
- New required kernel/API parameters (do not omit or rename):
  - `kv_seq_len: u32` — number of valid K/V positions, may exceed `seq_len`.
  - `cache_offset: u32` — number of previously-cached K/V positions that precede this chunk's queries. Query position `i` (0-indexed within this call) corresponds to absolute position `cache_offset + i`.
- **Causal masking happens inside this kernel, unconditionally.** The caller is not responsible for pre-slicing K/V to the causal range. This removes the ambiguity in the old spec's point 3 — if a future caller wants unmasked/bidirectional attention, that requires a separate kernel or an explicit `causal: bool` flag added later, not silent reinterpretation of this one.

If your investigation in step 1/2 finds the real contract differs from the default above (e.g. K/V pre-sliced by caller, or a `[num_kv_heads, kv_seq_len, head_dim]` layout), **that confirmed reality wins** — update the doc comment accordingly and note the deviation in your PR description. The default above only applies when no caller can be found and is authorized by a reviewer.

**Evidence checked: `grim-kvquant` crate.** This crate does *not* call `RocmDevice::qkv_attention()` (confirmed by grep — no hits for `qkv_attention` or `RocmDevice` anywhere in it), so it is not "the caller" and does not fully resolve Step 0. It's still relevant as corroborating/conflicting evidence, both noted rather than silently absorbed:

- *Supports* the shape contract above: its `IdentityCompressor::fused_attention` CPU reference uses `[num_tokens, num_heads_or_kv_heads, head_dim]` for Q/K/V, matching what's locked in here.
- *Complicates* the "masking always happens inside this kernel" decision: that same reference function attends every query over the **entire** KV block with no causal mask at all — consistent with a "caller pre-slices K/V to the causal range, kernel doesn't mask" pattern elsewhere in this codebase. This doesn't override the decision above (different call path, and the decision is intentional — see below), but it means "kernel always masks" is a deliberate choice being made here, not a fact already established elsewhere in the codebase. Worth a second look if the real ROCm caller turns up and disagrees.
- *Do not copy* its GQA indexing: it indexes K/V by `h` directly instead of the `kv_head = h / (num_heads / num_kv_heads)` mapping this spec requires. That's either a latent bug in `grim-kvquant` or evidence GQA grouping isn't exercised there yet — either way it's not authoritative for this kernel.

## Step 1 — algorithm

Standard scaled-dot-product multi-head attention with GQA head-sharing and causal masking:

```
kv_head = h / (num_heads / num_kv_heads)     // GQA mapping — unchanged, already correct
for i in 0..seq_len:                          // query position within this call
    abs_i = cache_offset + i
    for j in 0..kv_seq_len where j <= abs_i:  // causal limit, absolute positions
        score[j] = dot(q[i, h, :], k[j, kv_head, :]) / sqrt(head_dim)
    softmax score[] in place, numerically stable (subtract max before exp)
    out[i, h, :] = sum_j softmax_score[j] * v[j, kv_head, :]
```

Requirements:
- Numerically stable softmax: subtract running max before `expf`, running-sum accumulate, normalize at the end.
- Causal masking uses `j <= cache_offset + i`, per the fixed contract in Step 0 — no other interpretation.
- Keep `kv_head = head / (num_heads / num_kv_heads)`.
- Float accumulation for dot products and softmax regardless of input dtype.
- **Input/output dtype: `f32` for this version.** (If lower-precision inputs are needed later, that's a follow-up task with its own vector-load and conversion design — do not silently add f16/bf16 support as a side effect of this task.)

## Step 2 — parallelization strategy

Fixed decision (not left open): **one thread block per `(query_position, head)` pair**, flattened into a 2-D grid (`grid = (seq_len, num_heads)`). This is chosen over "one block per head with an inner query loop" because it parallelizes the part that dominates cost at real `seq_len` (the softmax/reduction over `kv_seq_len` per query), at the cost of more blocks for large `seq_len` — acceptable since block count scaling is cheap on ROCm relative to serial inner loops.

- Within a block: parallelize the `head_dim` dot product and the `kv_seq_len` reduction using shared memory (LDS) for the score buffer and the max/sum reduction.
- **Shared memory budget**: `ATTENTION_SHARED_MAX_BYTES` stays capped at 32768 bytes (8192 floats).
- **Mandatory**: implement online (flash-attention-style) softmax — running max + running weighted sum, no full score vector materialized — so behavior is correct and uniform regardless of `kv_seq_len`. Do not implement a "capped `kv_seq_len`" fallback path; this removes the either/or the old spec left open. One code path, always correct, no silent truncation and no separate bound-checked-assert path to maintain.
- **RDNA2/RDNA3 Wave64 mandate**: kernel MUST use `@workgroup_size(64, 1, 1)`. RDNA2/3 executes exactly 64-thread wavefronts; 32 wastes half the execution width, 128 runs as two sequential wavefronts. 64 maps one workgroup to one wavefront = maximum SIMD utilization.
- **Subgroup reduction for dot product & softmax**: use `__shfl_down_sync` (HIP provides CUDA-compatible shfl builtins which on AMD map to AMDGCN `ds_swizzle`/`readlane`/`ds_permute` over the 64-lane wavefront). This avoids LDS bank conflicts and cuts shared-memory traffic. The final max/sum reduction across the wavefront uses the same shuffle tree. ⚠ Per `rocm-hip-kernels`: prefer `__shfl_xor` for butterfly reductions and AMDGCN's `__builtin_amdgcn_ds_swizzle`/`_readlane` directly when you want explicit control; verify the wave is actually 64 lanes on RDNA before relying on shfl behavior.
- Update `QkvAttentionFusionConfig::hip_launch_params()` to emit `grid = (seq_len, num_heads)`, block dim sized for the `head_dim`/reduction work, and shared-mem sizing for the online-softmax scratch (running max/sum + per-thread partial dot products, not per-`kv_seq_len` storage).

## Step 3 — wire it up in `src/lib.rs`

- Replace the `grim_qkv_attention` source in `COMPUTE_KERNEL_SOURCE`.
- Update `RocmDevice::qkv_attention()` to pass `kv_seq_len` and `cache_offset` as explicit kernel args (per Step 0's fixed contract).
- **Fix the hardcoded GQA ratio**: remove `num_kv_heads: num_heads / 4` from the `QkvAttentionFusionConfig` construction. `num_kv_heads` must come from a real parameter (add one to `qkv_attention()`'s signature) or be derived from the actual shape of the `k`/`v` storage passed in — never assumed as a fixed ratio of `num_heads`.
- Add a doc comment directly above the kernel stating: the confirmed (or reviewer-authorized default) Q/K/V shapes, the meaning of `cache_offset`, and that causal masking is always applied inside the kernel.
- **Add an `enabled` gate to `QkvAttentionFusionConfig` and check it in `qkv_attention()`, returning `Err` when false.** This does not exist today — `qkv_attention()` currently launches the broken kernel unconditionally, with no gate at all. Default it to `false` until Step 4 passes, then flip to `true`. Keep the field and the check afterward (don't delete it once enabled) so a future regression can be gated off again without an emergency patch.

## Step 4 — testing (required before enabling)

1. CPU-side reference implementation of causal GQA attention (plain Rust, no HIP), for small shapes, using the same `cache_offset` semantics as Step 0/1.
2. Correctness tests comparing `RocmDevice::qkv_attention()` against the reference, tolerance `1e-3` relative error (`f32`). Required cases:
   - `num_heads == num_kv_heads` and `num_heads > num_kv_heads` (GQA), **including at least one ratio other than 4:1** (e.g. 2:1 or 8:1) — specifically to catch a regression back to the hardcoded-ratio bug being fixed in Step 3.
   - `seq_len == 1, cache_offset > 0` (decode-style, non-empty cache) and `seq_len > 1, cache_offset == 0` (prefill-style).
   - `seq_len > 1, cache_offset > 0` (chunked prefill continuing a cache) — this case was missing before and is exactly where an off-by-one in the causal bound would hide.
   - `kv_seq_len` large enough to exceed 8192 floats, to exercise the online-softmax path (this is now always-on, not optional).
   - Causal check: later query position's output changes when an earlier position's K/V changes; earlier position's output does not depend on later K/V.
3. Update `tests/fusion_smoke.rs` geometry tests to assert against the new fixed `(seq_len, num_heads)` grid and the online-softmax shared-mem sizing — not just patched to compile.
4. Only after all pass: set `enabled` default to `true`. Keep the field and the check in `qkv_attention()`.

## Acceptance criteria

- No correctness regression vs. CPU reference across the full Step 4 test matrix, including the new chunked-prefill-with-cache-offset case and the non-4:1 GQA ratio case.
- Kernel never reads/writes out of bounds for any `kv_seq_len` (guaranteed structurally by always using online softmax — no separate cap/assert path to fall out of sync).
- `enabled` gate exists and is checked (new — did not exist before this task) and remains as a mechanism after being flipped to `true`.
- `num_kv_heads` is a real parameter or shape-derived value, never a hardcoded ratio.
- Doc comment on the kernel states the actual (confirmed-by-caller or reviewer-authorized) layout and `cache_offset` semantics — this was the root cause of the original bug and is no longer optional documentation.

## Status of open items

- **Memory layout (Q/K/V shapes, `cache_offset` semantics)**: authorized per Step 0 above. No production caller exists to verify against (confirmed — see Context), so this is a locked design decision, not a verified fact; revisit if a real caller is written later and disagrees.
- **`enabled` gate**: does not exist yet; this task adds it (see Step 3).
- **`num_kv_heads` hardcode**: confirmed real bug in current code; this task fixes it (see Step 3).
- **Paged KV cache integration** (block-table-based storage in `grim-memory`, vs. this kernel's flat-buffer contract): still an open design question, not resolved by anything found in `grim-backend-rocm` or `grim-cli`. This kernel is being built as a flat-buffer primitive on the assumption that a separate, not-yet-written gather step will materialize a contiguous K/V buffer from the paged cache before calling it (Option A from prior discussion). Redesigning this kernel to be paging-aware directly (Option B) is out of scope for this task unless explicitly requested.

---

# Phase 2 (follow-up task, not part of this one) — performance

Everything above is Phase 1: correctness first, "reasonably performant" only. **Do not start Phase 2 work until Phase 1's Step 4 has passed and `enabled` is `true`.** Folding these into the same PR risks reintroducing correctness bugs under cover of a perf rewrite, and makes review much harder. Treat this section as a backlog, not instructions to execute now.

There's no objective "most performant ROCm implementation ever" benchmark — the real target is competitive with hand-tuned flash-attention-style kernels on the deployed hardware (CDNA MI200/MI300 class, per `gpu_target`), for the shapes this workload actually produces. Define a concrete before/after benchmark (tokens/sec at representative `seq_len`/`kv_seq_len`/batch, on real hardware) before claiming any of this "worked."

Roughly in order of expected impact:

**Architecture-level (would replace Phase 1's Step 2 design, not extend it):**
- Use MFMA matrix-core instructions for the QK^T and score×V matmuls instead of scalar FMA loops — the single biggest lever on CDNA hardware; Phase 1's per-thread dot product leaves matrix cores idle.
- **Phase 1 uses scalar FMA only.** MFMA tensor-core instructions are Phase 2 replacement; do not add MFMA to Phase 1 kernel.
- Tiled flash-attention-style blocking over `(query_tile, kv_tile)` with online softmax kept in registers/LDS, replacing Phase 1's one-block-per-single-query-position scheme — amortizes K/V loads across queries/heads sharing KV data.
- Paged/block-table-aware kernel (Option B from the Step 0 discussion above) instead of requiring a pre-gather into a flat buffer — avoids a full extra memory copy per call once the paged KV cache in `grim-memory` is actually wired up.

**RDNA 4 (gfx1200) forward-compatibility notes:**
- RDNA 4 retains Wave64 (`@workgroup_size(64, 1, 1)` remains correct).
- Expanded MFMA support: BF16/FP8/FP6 matrix ops in addition to FP16/FP32. Phase 2's MFMA path should target BF16/FP8 for RDNA 4 inference workloads.
- Improved dual-issue and LDS banking; subgroup shuffle patterns from Phase 1 remain valid.
- When Phase 2 adds BF16/FP8, use `v_mfma_f32_16x16x16bf16` (or equivalent) with proper K-dimension tiling for RDNA 4's matrix cores.
- No Phase 1 changes needed for RDNA 4 — scalar FMA + Wave64 + subgroup shuffles work identically on gfx1200.

**Fits inside the existing kernel shape:**
- Vectorized global loads (`float4`) for `head_dim`-contiguous reads instead of scalar loads.
- Warp/wavefront shuffle reductions for the dot-product and softmax reduction instead of routing everything through LDS.
- Split-KV ("flash-decoding" style) parallelism for the decode case (`seq_len == 1`, large `kv_seq_len`): split the KV reduction for a single query across multiple blocks with a second-pass combine, instead of serializing the whole reduction in one block.
- Persistent kernels / HIP graph capture — `capture_enabled` / `GRIM_CAPTURE_GRAPH` already exists in `grim-backend-rocm`; wire `qkv_attention` into that path to avoid per-call launch overhead.
- Autotune block/tile sizes per `wavefront_size` and `gpu_target` — kernel caching is already keyed on `gpu_target` for JIT purposes; extend that to launch geometry too.

**Lower effort:**
- Fuse RoPE application into this kernel if it currently runs as a separate pass immediately before/after attention.
- bf16/fp16 compute with fp32 accumulation, once Phase 1's fp32-only correctness baseline is solid and can be used as the regression reference.
- Investigate AMD's own AITER library before hand-rolling further — `grim-memory` already has a `block_major_layout` flag tied to a `rocm-aiter` feature that nothing currently implements. Wrapping/calling a hand-tuned AITER attention kernel may beat further hand-rolling here for less engineering risk.

**System-level (outside this kernel, but likely bigger impact than any single-kernel optimization above — worth doing first or in parallel):**
- **Avoid per-call scratch allocation.** `upload_device_buffer()` in `grim-backend-rocm/src/lib.rs` currently does `hipMalloc` + `hipMemcpy` + implicit sync on every call for scratch buffers. Pool and reuse device scratch memory across calls instead of alloc/free per launch — this is a real, measurable overhead source today, independent of how fast `qkv_attention` itself becomes.
- **HIP graph capture across the whole per-token decode step**, not just individual kernels. `capture_enabled`/`GRIM_CAPTURE_GRAPH` already exists in this codebase — extend graph capture to the full fused-op sequence (rmsnorm+matmul → qkv_attention → ...) so CPU-side launch overhead is amortized once per graph replay instead of once per kernel. Matters most at small batch sizes / short decode steps where kernels are small and launch latency dominates over kernel runtime.
- **Continuous batching**: batch multiple in-flight sequences into the same kernel launch wherever `grim-scheduler`'s admission control allows it. GPU utilization at batch size 1 is poor regardless of kernel quality — this generally has more impact on end-to-end throughput than kernel-internal tuning.
- **Stream pooling**: make sure enough pooled streams (`active_stream()`) exist to actually overlap independent launches (e.g. across concurrent requests) rather than serializing on one stream.
- **Compiler/build flags**: target the exact GPU arch (e.g. `--offload-arch=gfx1036` for grim's RDNA2, `--offload-arch=gfx1200` for RDNA4, `--offload-arch=gfx942` for the CDNA Instinct builds) rather than a generic target so arch-specific codegen (MFMA availability, LDS size) is used; verify `hipcc` optimization level. Match the arch to the running GPU's `gcnArchName` (queried at startup) — having one shipped `--offload-arch` mismatch silently falls back to suboptimal codegen. Per `rocm-hip-kernels`/`rocm-profiling-perf`: also compile and cache multiple arch variants, selected by `.grim` model's `target_gfx` metadata.
- **Environment tuning**: `HSA_ENABLE_SDMA`, `GPU_MAX_HWQUEUES`, and XNACK/HMM settings (`xnack_enabled` is already tracked in `RocmDeviceProps`) all affect copy-engine usage and page-fault behavior — usually want XNACK off for latency-sensitive inference. Pin NUMA affinity between GPU and the feeding CPU thread on multi-socket/multi-GPU hosts.
- **Pinned host memory** for any remaining host↔device transfers in the pipeline.
- **Profile before optimizing further**: use `rocprof-compute` for kernel occupancy/roofline and stall reasons, `rocprofiler-sdk` (HIP/HSA/RCCL counters) for trace-driven metric capture, and `rocprofv3`/`rocpd` (newer ROCm timelines) when available — confirm where the actual bottleneck is before investing in any specific item above. A well-tiled kernel can still lose to register-spill-induced low occupancy, which only shows up under profiling.
- **Multi-GPU** (if this deployment uses tensor/pipeline parallelism): use RCCL for collectives, confirm Infinity Fabric/xGMI links are used for GPU-to-GPU traffic instead of falling back to PCIe, and overlap communication with compute rather than treating all-reduce as a blocking barrier.

Define the before/after benchmark (Phase 2's acceptance criteria above) against the *system*, not just the kernel in isolation — some of these system-level changes will show up as bigger throughput gains than the kernel rewrite itself, so sequence or parallelize accordingly rather than assuming kernel work alone gets "most performant."

**Acceptance criteria for Phase 2 (draft — refine when this is actually scoped):**
- Numerical parity with the Phase 1 CPU reference maintained at the same tolerance, across the same test matrix plus whatever new shapes the tiled/paged design introduces.
- Documented before/after benchmark on real hardware, not just theoretical occupancy.
- Phase 1's `enabled` gate semantics preserved — a Phase 2 regression must be gated off the same way a Phase 1 one would be.

---

# Phase 3: System-Level Performance Roadmap (Post-Phase 2)

These items target end-to-end LLM inference throughput/latency on ROCm. Each section includes: **why** it matters, **exact crate/file/lines** to target, **starter code**, and **modularization rules** (no file > 1500 lines).

## 3.1 Device Scratch Memory Pool — Immediate Win (P0)

### Why
`upload_device_buffer()` in `grim-backend-rocm/src/lib.rs` does `hipMalloc` + `hipMemcpy` + implicit sync on **every call** for scratch buffers. At decode batch=1, this adds 20–40 µs/token of pure driver overhead. Pooling eliminates alloc/free churn and lets the driver reuse VRAM mappings.

### Target
- **Crate**: `grim-backend-rocm`
- **File**: `crates/grim-backend-rocm/src/lib.rs` (currently 4630 lines — **must modularize**)
- **Lines to edit**: `upload_device_buffer` (~line 1790), `alloc_device_buffer` (~line 1840), plus new pool module

### Modularization (required before adding pool)
Split `lib.rs` into:
```
crates/grim-backend-rocm/src/
├── lib.rs                    # ~300 lines: re-exports, public API
├── device.rs                 # ~800 lines: RocmDevice, allocator, stream pool
├── kernels/                  # NEW directory
│   ├── mod.rs                # ~200 lines: kernel registry, JIT cache
│   ├── qkv_attention.rs      # ~400 lines: QKV attention kernel + launch
│   ├── rmsnorm_matmul.rs     # ~300 lines: fused RMSNorm+MatMul
│   └── compute_kernels.rs    # ~500 lines: COMPUTE_KERNEL_SOURCE string
├── memory/
│   ├── mod.rs                # ~100 lines
│   ├── pool.rs               # ~300 lines: DeviceScratchPool (NEW)
│   └── allocator.rs          # ~400 lines: RocmCachingAllocator
├── fusion.rs                 # keep (62 lines)
├── graph_capture.rs          # ~400 lines: capture/replay logic
└── tests/                    # keep
```

**Rule**: No new code in `lib.rs`. All new files < 500 lines.

### Starter Code — `memory/pool.rs`
```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::{hipMalloc, hipFree, hipSuccess, DevicePtr};

/// Layout key for the scratch pool: (size, alignment)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolLayout {
    pub size: usize,
    pub align: usize,
}

impl PoolLayout {
    pub fn new(size: usize, align: usize) -> Self {
        // Round size up to next power of 2 for bucketization
        let bucket = size.next_power_of_two().max(256);
        Self { size: bucket, align }
    }
}

/// RAII guard returning buffer to pool on drop
pub struct PooledBuffer {
    ptr: *mut std::ffi::c_void,
    layout: PoolLayout,
    pool: Arc<DeviceScratchPool>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            self.pool.return_buffer(self.ptr, self.layout);
        }
    }
}

impl PooledBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void { self.ptr }
    pub fn as_device_ptr(&self) -> DevicePtr { DevicePtr::new(self.ptr) }
}

/// Thread-safe scratch buffer pool with power-of-2 bucketization
pub struct DeviceScratchPool {
    buckets: Mutex<HashMap<PoolLayout, Vec<*mut std::ffi::c_void>>>,
    peak_bytes: std::sync::atomic::AtomicUsize,
    current_bytes: std::sync::atomic::AtomicUsize,
}

impl DeviceScratchPool {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            buckets: Mutex::new(HashMap::new()),
            peak_bytes: std::sync::atomic::AtomicUsize::new(0),
            current_bytes: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Get a buffer of at least `size` bytes, `align`-aligned
    pub fn get(self: &Arc<Self>, size: usize, align: usize) -> Result<PooledBuffer, crate::Error> {
        let layout = PoolLayout::new(size, align);
        let ptr = {
            let mut buckets = self.buckets.lock().unwrap();
            buckets.get_mut(&layout).and_then(|v| v.pop())
        };
        let ptr = match ptr {
            Some(p) => p,
            None => {
                // Allocate new (rounded to bucket size)
                let mut p = std::ptr::null_mut();
                let res = unsafe { hipMalloc(&mut p, layout.size) };
                if res != hipSuccess {
                    return Err(crate::Error::Backend(format!("hipMalloc failed: {}", res)));
                }
                self.current_bytes.fetch_add(layout.size, std::sync::atomic::Ordering::Relaxed);
                let peak = self.current_bytes.load(std::sync::atomic::Ordering::Relaxed);
                self.peak_bytes.fetch_max(peak, std::sync::atomic::Ordering::Relaxed);
                p
            }
        };
        Ok(PooledBuffer { ptr, layout, pool: self.clone() })
    }

    fn return_buffer(&self, ptr: *mut std::ffi::c_void, layout: PoolLayout) {
        let mut buckets = self.buckets.lock().unwrap();
        buckets.entry(layout).or_default().push(ptr);
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}
```

### Integration Point
In `device.rs` (new), add to `RocmDevice`:
```rust
pub struct RocmDevice {
    // ... existing fields
    scratch_pool: Arc<DeviceScratchPool>,
}

impl RocmDevice {
    pub fn new(ordinal: usize) -> Result<Self, Error> {
        // ... existing init
        let scratch_pool = DeviceScratchPool::new();
        Ok(Self { /* ... */ scratch_pool, /* ... */ })
    }

    /// Replace `alloc_device_buffer` for scratch/temp buffers
    pub fn get_scratch(&self, size: usize, align: usize) -> Result<PooledBuffer, Error> {
        self.scratch_pool.get(size, align)
    }
}
```

### Update `upload_device_buffer` (in `memory/mod.rs` after split)
```rust
// OLD: hipMalloc + hipMemcpy + implicit sync per call
// NEW: borrow from pool, memcpy, return to pool on drop
pub fn upload_to_scratch(dev: &RocmDevice, data: &[u8], align: usize) -> Result<PooledBuffer, Error> {
    let buf = dev.get_scratch(data.len(), align)?;
    let res = unsafe { hipMemcpy(buf.as_ptr(), data.as_ptr() as *const _, data.len(), HipMemcpyKind::HostToDevice) };
    if res != hipSuccess { return Err(Error::Backend(...)); }
    Ok(buf)
}
```

### Acceptance
- `cargo test -p grim-backend-rocm` passes
- Microbenchmark: `upload_to_scratch` 1000× < 5 ms total (vs ~200 ms with per-call `hipMalloc`)
- Pool peak memory < 10 MB for typical decode workloads

---

## 3.2 HIP Graph Capture for Full Decode Step (P0)

### Why
Per-kernel launch overhead on ROCm is 10–20 µs. A decode step launches 8–12 kernels (rmsnorm, q_proj, k_proj, v_proj, qkv_attention, o_proj, residual, rmsnorm, ffn_gate, ffn_up, ffn_down, residual). Capturing the **entire sequence** into a `hipGraph` and replaying with `hipGraphLaunch` reduces launch overhead to ~1 µs/replay — **15–30 µs/token saved at batch=1**.

### Target
- **Crate**: `grim-backend-rocm`
- **File**: New `crates/grim-backend-rocm/src/graph_capture.rs` (new, ~400 lines)
- **Integration**: `RocmDevice::capture_decode_graph()` + `replay_decode_graph()`

### Starter Code — `graph_capture.rs`
```rust
use crate::{hipGraphCreate, hipGraphInstantiate, hipGraphLaunch, hipGraphExecDestroy, hipGraphDestroy, hipGraph_t, hipGraphExec_t, hipStream_t, hipSuccess, Error, DevicePtr};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A captured decode-step graph for a specific (batch, seq_len, kv_len) bucket
pub struct DecodeGraph {
    pub graph: hipGraph_t,
    pub exec: hipGraphExec_t,
    pub stream: hipStream_t,
    pub input_addrs: Vec<*mut std::ffi::c_void>,  // pointers to update per replay
    pub output_addrs: Vec<*mut std::ffi::c_void>,
}

/// Key for graph cache: (batch, seq_len, kv_seq_len, head_dim, num_heads, num_kv_heads)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeGraphKey {
    pub batch: u32,
    pub seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
}

pub struct GraphCaptureManager {
    graphs: Mutex<HashMap<DecodeGraphKey, Arc<DecodeGraph>>>,
    capture_stream: hipStream_t,
}

impl GraphCaptureManager {
    pub fn new() -> Result<Self, Error> {
        let mut stream = std::ptr::null_mut();
        let res = unsafe { crate::hipStreamCreate(&mut stream) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!("hipStreamCreate failed: {}", res)));
        }
        Ok(Self {
            graphs: Mutex::new(HashMap::new()),
            capture_stream: stream,
        })
    }

    /// Get or capture a graph for the given shape bucket
    pub fn get_or_capture(
        &self,
        key: DecodeGraphKey,
        capture_fn: impl FnOnce(hipStream_t) -> Result<(), Error>,
    ) -> Result<Arc<DecodeGraph>, Error> {
        let mut graphs = self.graphs.lock().unwrap();
        if let Some(g) = graphs.get(&key) {
            return Ok(g.clone());
        }
        drop(graphs);

        // Capture on dedicated stream
        let res = unsafe { crate::hipStreamBeginCapture(self.capture_stream, crate::hipStreamCaptureMode::Global) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!("hipStreamBeginCapture failed: {}", res)));
        }

        capture_fn(self.capture_stream)?;

        let mut graph = std::ptr::null_mut();
        let res = unsafe { crate::hipStreamEndCapture(self.capture_stream, &mut graph) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!("hipStreamEndCapture failed: {}", res)));
        }

        let mut exec = std::ptr::null_mut();
        let res = unsafe { crate::hipGraphInstantiate(&mut exec, graph, 0) };
        if res != crate::hipSuccess {
            unsafe { crate::hipGraphDestroy(graph) };
            return Err(Error::Backend(format!("hipGraphInstantiate failed: {}", res)));
        }

        let decode_graph = Arc::new(DecodeGraph {
            graph,
            exec,
            stream: self.capture_stream,
            input_addrs: Vec::new(),  // filled by caller after instantiation
            output_addrs: Vec::new(),
        });

        let mut graphs = self.graphs.lock().unwrap();
        graphs.insert(key, decode_graph.clone());
        Ok(decode_graph)
    }

    pub fn replay(&self, graph: &DecodeGraph) -> Result<(), Error> {
        let res = unsafe { crate::hipGraphLaunch(graph.exec, graph.stream) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!("hipGraphLaunch failed: {}", res)));
        }
        Ok(())
    }
}
```

### Integration in `device.rs` (new)
```rust
impl RocmDevice {
    pub fn capture_decode_step(
        &self,
        batch: u32,
        seq_len: u32,
        kv_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Result<(), Error> {
        let key = DecodeGraphKey { batch, seq_len, kv_seq_len, head_dim, num_heads, num_kv_heads };
        
        self.graph_mgr.get_or_capture(key, |stream| {
            // Set stream for all ops in this capture
            self.bind_stream_to_rocblas(stream)?;
            self.bind_stream_to_kernels(stream)?;
            
            // Execute the full decode sequence (rmsnorm → q_proj → k_proj → v_proj → qkv_attention → o_proj → residual → rmsnorm → ffn → residual)
            self.execute_full_decode_sequence(stream)?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn replay_decode_step(&self, key: DecodeGraphKey) -> Result<(), Error> {
        let graphs = self.graph_mgr.graphs.lock().unwrap();
        let graph = graphs.get(&key).ok_or(Error::Backend("Graph not captured".into()))?;
        self.graph_mgr.replay(graph)
    }
}
```

### Acceptance
- `hipGraphLaunch` replay latency < 2 µs vs 15–30 µs multi-kernel launch
- Graph cache hit rate > 95% for steady-state decode
- `cargo test -p grim-backend-rocm` passes

---

## 3.3 Quantized GEMM via MFMA (Phase 2 Priority)

### Why
BF16 MFMA on MI200/MI300 (CDNA2/3) and RDNA4 gives 4–8× throughput vs FP32 scalar FMA. Q4_K_M block-wise dequant in LDS before MFMA yields 4× memory bandwidth reduction.

⚠ **Correction (from skills creation, `rocm-quantization-inference` + `rust-gpu-discipline` pattern #12):**
- **FP8 MFMA is only natively supported on RDNA4 (`gfx1200`/`gfx1201`).** The `__hip_fp8_e4m3_fnuz` / `__hip_fp8_e5m2` *types* exist in RDNA2/RDNA3 ROCm headers, but there is no native fp8 MFMA on those arches — fp8→f32 is emulated and *slower than f16*. **`QuantMode::Fp8` must gate on `target_gfx >= gfx1200`** (use `hipGetDeviceProperties.gcnArchName`); on gfx1036/gfx110x, dispatch to `QuantMode::Bf16` rather than silently running emulated fp8.
- The kernel's MFMA shapes/instruction names differ between arches: CDNA MI300x (gfx942) has the richest MFMA set (incl. f8f6f4 packed-fp8 for newer CDNA4 parts); RDNA2/RDNA3 have a smaller subset; RDNA4 adds BF16/FP6/FP8. **Arch-dispatch** the MFMA call sites per `gcnArchName` — do not assume one MFMA shape works everywhere.

### Target
- **Crate**: `grim-backend-rocm`
- **File**: `crates/grim-backend-rocm/src/kernels/qkv_attention_quantized.rs` (new, ~500 lines)
- **Integration**: `QkvAttentionFusionConfig` gains `quant_mode: QuantMode` field

### Starter Code — `kernels/qkv_attention_quantized.rs`
```rust
use crate::{RocmDevice, Error, Shape, DType};

/// Quantization mode for attention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Fp32,      // baseline
    Bf16,      // BF16 compute, FP32 accumulate
    Fp8,       // FP8 E4M3/E5M2
    Q4K,       // llama.cpp Q4_K_M block-wise
}

/// Kernel source with MFMA + block dequant
pub const QKV_ATTENTION_QUANTIZED_KERNEL: &str = r#"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// Block-wise dequant: each 32-element block has (min, max, scale)
// Dequant in LDS, then MFMA for QK^T and score×V

extern "C" __global__ void grim_qkv_attention_q4k(
    const uint8_t* q, const uint8_t* k, const uint8_t* v, float* out,
    const float* q_scales, const float* k_scales, const float* v_scales,
    int num_heads, int num_kv_heads, int head_dim, int seq_len,
    int kv_seq_len, int cache_offset
) {
    // Wave64: @workgroup_size(64, 1, 1)
    // Each workgroup handles one (query_pos, head)
    // LDS layout: 
    //   - dequant Q tile [Bq, D] (BF16)
    //   - dequant K tile [Bk, D] (BF16)  
    //   - scores [Bq, Bk] (FP32 online softmax)
    //   - dequant V tile [Bk, D] (BF16)
    // MFMA: v_mfma_f32_16x16x16bf16 for QK^T and score×V
}

// MFMA intrinsic wrapper for BF16
__device__ void mfma_bf16_16x16x16(float* d, const __hip_bf16* a, const __hip_bf16* b, float* c) {
    asm volatile(
        "v_mfma_f32_16x16x16bf16 %0, %1, %2, %3"
        : "=v"(d[0]), "=v"(d[1]), "=v"(d[2]), "=v"(d[3])
        : "v"(a[0]), "v"(a[1]), "v"(b[0]), "v"(b[1]), "v"(c[0]), "v"(c[1]), "v"(c[2]), "v"(c[3])
        :
    );
}
"#;

impl RocmDevice {
    pub fn qkv_attention_quantized(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        out_shape: &Shape,
        quant_mode: QuantMode,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Dispatch based on quant_mode
        match quant_mode {
            QuantMode::Fp32 => self.qkv_attention(q, k, v, out_shape),
            QuantMode::Bf16 | QuantMode::Fp8 | QuantMode::Q4K => {
                // Launch quantized kernel
                todo!("wire quantized kernel launch")
            }
        }
    }
}
```

### MFMA Wrapper — `kernels/mfma.rs` (new, ~200 lines)
```rust
//! MFMA intrinsic wrappers for BF16/FP8/FP16/FP32
//!
//! Maps to AMDGCN `v_mfma_f32_16x16x16bf16`, `v_mfma_f32_16x16x16f16`, etc.

#[cfg(target_arch = "amdgcn")]
mod mfma_intrinsics {
    use std::arch::asm;

    /// BF16 MFMA: D = A × B + C  (16×16×16, f32 accumulation)
    #[inline]
    pub unsafe fn mfma_bf16_16x16x16(
        a: [u32; 4],   // 8 BF16 per register × 4 = 32 elements (16×16 / 8)
        b: [u32; 4],
        c: [f32; 8],   // 16×16 f32 = 256 elements → 8 f32 registers (32 each)
    ) -> [f32; 8] {
        let mut d = [0.0f32; 8];
        asm!(
            "v_mfma_f32_16x16x16bf16 {{
                {d0}, {d1}, {d2}, {d3}, {d4}, {d5}, {d6}, {d7},
                {a0}, {a1}, {a2}, {a3},
                {b0}, {b1}, {b2}, {b3},
                {c0}, {c1}, {c2}, {c3}, {c4}, {c5}, {c6}, {c7}
            }}",
            a0 = in(vreg) a[0], a1 = in(vreg) a[1], a2 = in(vreg) a[2], a3 = in(vreg) a[3],
            b0 = in(vreg) b[0], b1 = in(vreg) b[1], b2 = in(vreg) b[2], b3 = in(vreg) b[3],
            c0 = in(vreg) c[0], c1 = in(vreg) c[1], c2 = in(vreg) c[2], c3 = in(vreg) c[3],
            c4 = in(vreg) c[4], c5 = in(vreg) c[5], c6 = in(vreg) c[6], c7 = in(vreg) c[7],
            d0 = out(vreg) d[0], d1 = out(vreg) d[1], d2 = out(vreg) d[2], d3 = out(vreg) d[3],
            d4 = out(vreg) d[4], d5 = out(vreg) d[5], d6 = out(vreg) d[6], d7 = out(vreg) d[7],
            options(nostack, nomem, preserves_flags)
        );
        d
    }
}
```

### Acceptance
- BF16 MFMA kernel matches FP32 reference within 1e-2 relative error
- Q4_K_M kernel within 1e-1 (expected quantization loss)
- 4× memory bandwidth reduction vs FP32 baseline on MI300

---

## 3.4 Paged Attention Kernel (Phase 2 Architecture)

### Why
Current design requires gathering K/V from paged cache into flat buffer before attention — **full extra memory copy per call**. Paged attention kernel reads directly from block table — eliminates gather copy, enables continuous batching.

### Target
- **Crate**: `grim-backend-rocm`
- **File**: `crates/grim-backend-rocm/src/kernels/qkv_attention_paged.rs` (new, ~500 lines)
- **Integration**: `grim-memory::PagedKvCache` provides `block_table: &[u32], page_size: u32`

### Starter Code — `kernels/qkv_attention_paged.rs`
```rust
//! Paged Attention kernel — reads K/V directly from block table (no gather copy)

use crate::kernels::mfma::*;  // reuse MFMA tiles

/// Block table entry: (physical_block_id, page_size)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockTableEntry {
    pub block_id: u32,
    pub page_size: u32,
}

/// Kernel launch config for paged attention
pub fn paged_attention_launch_config(
    batch: usize,
    num_heads: usize,
    head_dim: usize,
) -> (u32, u32, u32, usize) {
    // 1 block per (batch, head) — same as flash attention
    let grid_x = batch as u32;
    let grid_y = num_heads as u32;
    let block_dim = 256;
    // Shared mem: Q tile + K/V tile + block table cache
    let shared_mem = (64 * 128 * 2) * 2 + (256 * 4);  
    (grid_x, grid_y, block_dim, shared_mem)
}

pub const PAGED_ATTENTION_KERNEL: &str = r#"
extern "C" __global__ void grim_qkv_attention_paged(
    const float* q,           // [seq_len, num_heads, head_dim]
    const BlockTableEntry* __restrict__ block_tables, // [batch, max_blocks]
    const float* __restrict__ k_pages,     // [num_pages, page_size, num_kv_heads, head_dim]
    const float* __restrict__ v_pages,     // [num_pages, page_size, num_kv_heads, head_dim]
    float* out,               // [seq_len, num_heads, head_dim]
    const float* scale,
    uint32_t seq_len, uint32_t kv_seq_len,
    uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size,
    uint32_t cache_offset
) {
    // Grid: (seq_len, num_heads)
    // Block: 256 threads (4 waves)
    
    // Each thread block:
    // 1. Load Q tile for this (query_pos, head) into registers/LDS
    // 2. For each KV block in block_table:
    //    a. Load K/V page from global memory (coalesced)
    //    b. Dequant if needed (Q4_K, FP8)
    //    c. Compute QK^T for this page
    //    d. Online softmax update (running max + running sum)
    //    e. Accumulate score × V into output register
    // 3. Write final output
    
    // Key optimization: 
    // - Page loads are coalesced (128-byte transactions)
    // - Online softmax: no full score matrix in LDS
    // - Page size 16–64 tokens tuned per head_dim
}
"#;

pub fn launch_paged_attention(
    dev: &crate::RocmDevice,
    q: &crate::RocmStorage,          // [seq_len, num_heads, head_dim]
    block_tables: &crate::RocmStorage, // [max_blocks] of BlockTableEntry
    k_pages: &crate::RocmStorage,     // [num_pages, page_size, num_kv_heads, head_dim]
    v_pages: &crate::RocmStorage,
    out: &mut crate::RocmStorage,
    seq_len: u32,
    kv_seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    cache_offset: u32,
) -> Result<(), crate::Error> {
    let (grid_x, grid_y, block_dim, shared_mem) = paged_attention_launch_config(
        1, num_heads as usize, head_dim as usize
    );
    let module = dev.get_or_compile_module("paged_attention", PAGED_ATTENTION_KERNEL)?;
    let function = dev.get_function(&module, "paged_attention")?;
    
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scale_dev = dev.upload_scalar(scale)?;
    
    let args = [
        q.device_ptr().unwrap() as *const _,
        block_tables.device_ptr().unwrap() as *const _,
        k_pages.device_ptr().unwrap() as *const _,
        v_pages.device_ptr().unwrap() as *const _,
        out.device_ptr().unwrap() as *mut _,
        scale_dev.as_ptr() as *const _,
        &seq_len, &kv_seq_len, &num_heads, &num_kv_heads,
        &head_dim, &page_size, &cache_offset,
    ];
    
    dev.launch_kernel(function, 1, 1, 1, block_dim, 1, 1, shared_mem, &args)
}
```

### Integration with `grim-memory`
In `grim-memory/src/lib.rs` (currently 534 lines — add paged cache methods):
```rust
impl PagedKvCache {
    /// Get block table for current sequence (for paged attention kernel)
    pub fn block_table(&self) -> &[BlockTableEntry] { ... }
    
    /// Get K/V pages as flat buffers (for kernel)
    pub fn k_pages(&self) -> &RocmStorage { ... }
    pub fn v_pages(&self) -> &RocmStorage { ... }
}
```

### Acceptance
- Zero host↔device copies for KV cache during decode
- Continuous batching: multiple sequences with different `kv_seq_len` in same kernel launch
- Throughput ≥ 5× flat-buffer + gather baseline at batch=8, ctx=4096

---

## 3.5 Speculative Decoding Primitive (P2)

### Why
Small draft model (1/8 size) generates `k` tokens ahead; large model verifies tree of draft tokens in **one forward pass** via tree attention. 2–3× latency reduction at same quality.

### Target
- **Crate**: `grim-speculative` (new) + `grim-backend-rocm` tree attention kernel
- **Files**: 
  - `crates/grim-speculative/src/lib.rs` (new, ~400 lines)
  - `crates/grim-backend-rocm/src/kernels/tree_attention.rs` (new, ~400 lines)

### Starter Code — `grim-speculative/src/lib.rs`
```rust
use grim_backend_rocm::{RocmDevice, RocmStorage, Shape, DType};

/// Draft model + target model verification pipeline
pub struct SpeculativeDecoder {
    draft_model: DraftModel,      // small (e.g., 1/8 size)
    target_model: TargetModel,    // large GGUF
    gamma: usize,                 // draft steps per verification
    acceptance_threshold: f32,
}

impl SpeculativeDecoder {
    pub fn new(
        draft_path: &str,
        target_path: &str,
        gamma: usize,
        threshold: f32,
    ) -> Result<Self, Error> { ... }

    /// Single decode step: draft → verify → accept/reject
    pub fn step(&mut self, input_ids: &[u32]) -> Result<Vec<u32>, Error> {
        // 1. Draft: generate gamma tokens
        let draft_tokens = self.draft_model.generate(input_ids, self.gamma)?;
        
        // 2. Build tree attention prefix (draft + input)
        let tree_prefix = build_tree_attention_prefix(input_ids, &draft_tokens);
        
        // 3. Target: single forward pass with tree attention
        let (accepted, target_logits) = self.target_model.verify_tree(
            &tree_prefix, 
            self.gamma,
            self.acceptance_threshold,
        )?;
        
        // 4. Return accepted tokens (prefix of draft + maybe 1 target token)
        Ok(accepted)
    }
}

/// Tree attention: single kernel verifies multiple draft branches
pub fn tree_attention_kernel(...) { ... }
```

### Kernel — `grim-backend-rocm/src/kernels/tree_attention.rs`
```rust
pub const TREE_ATTENTION_KERNEL: &str = r#"
extern "C" __global__ void tree_attention(
    const float* Q,           // [batch, 1+gamma, num_heads, head_dim]
    const float* K,           // [kv_seq_len, num_kv_heads, head_dim]
    const float* V,
    float* Out,               // [batch, 1+gamma, num_heads, head_dim]
    const uint32_t* tree_parents, // [1+gamma] parent index in tree
    const float* draft_logits,    // [1+gamma, vocab] from draft model
    float* target_logits,         // [1+gamma, vocab] output
    const float* acceptance_threshold,
    uint32_t gamma, uint32_t seq_len, ...
) {
    // Each block handles one head
    // Compute attention for root (real token) + all draft tokens
    // Tree-structured causal mask: token i attends to its ancestors in tree
    // Output: target logits for each draft position
    // Host then does acceptance sampling (or kernel does it)
}
"#;
```

### Acceptance
- End-to-end latency 2–3× lower than greedy decoding at same quality
- Draft model accuracy ≥ 90% token acceptance rate
- Tree attention kernel latency < 2× single-token attention

---

## 3.6 Autotuner per GPU Arch (P2)

### Why
Optimal `block_dim`, `tile_kv`, `grid_stride` vary by GPU (MI200 vs MI300 vs gfx1036 vs gfx1200). Hardcoding leaves 10–20% on the table.

⚠ **Cross-task correction (grim backend, from skills-creation audit):** this autotuner targets **kernel launch geometry** (block_dim / tile_kv / grid_stride). Separately, grim's existing `lookup_solution_index` table in `crates/grim-backend-rocm/src/lib.rs` (Item 7 / `matmul_with_solution`) attempts to autotune **rocBLAS GEMM** but currently calls `rocblas_gemm_ex` with `algo = rocblas_gemm_algo::standard` together with a non-zero `solution_index` — per `rocBLAS` semantics, `solution_index` is **ignored** unless `algo == rocblas_gemm_algo_solution_index`. It is a silent no-op today. Per `rocm-profiling-perf` (autotune loop) and `rust-gpu-discipline` pattern #11 (silent no-op): replace the hardcoded table with **runtime `rocblas_gemm_ex_get_solutions` enumeration + benchmark**, and pass `algo = rocblas_gemm_algo_solution_index` when honoring a tuned index. Cache by `(arch, M, N, K, dtype)`. That fix is *not* inside this spec's scope but blocks any claim that "GEMM is tuned" — tracked as a separate backend task.

### Target
- **Crate**: `grim-backend-rocm`
- **File**: `crates/grim-backend-rocm/src/autotune.rs` (new, ~300 lines)
- **Cache**: `~/.cache/grim/autotune/{gfx1036,gfx1100,gfx1200,gfx942}.json`

### Starter Code — `autotune.rs`
```rust
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotuneConfig {
    pub kernel_name: String,
    pub block_dim: u32,
    pub tile_kv: u32,
    pub grid_stride: u32,
    pub cycles_per_invocation: u64,
    pub timestamp: u64,
}

pub struct Autotuner {
    cache_dir: PathBuf,
    device: Arc<crate::RocmDevice>,
}

impl Autotuner {
    pub fn new(device: Arc<crate::RocmDevice>) -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("grim/autotune");
        fs::create_dir_all(&cache_dir).ok();
        Self { cache_dir, device }
    }

    pub fn get_or_tune(
        &self,
        kernel_name: &str,
        problem_shape: (usize, usize, usize), // (seq_len, num_heads, head_dim)
    ) -> AutotuneConfig {
        let key = format!("{}_{}_{}_{}_{}", kernel_name, problem_shape.0, problem_shape.1, problem_shape.2, self.device.gpu_target());
        let cache_file = self.cache_dir.join(format!("{}.json", key));
        
        if let Ok(data) = fs::read_to_string(&cache_file) {
            if let Ok(cfg) = serde_json::from_str::<AutotuneConfig>(&data) {
                return cfg;
            }
        }

        // Run microbenchmark
        let best = self.benchmark_kernel(kernel_name, problem_shape);
        fs::write(&cache_file, serde_json::to_string_pretty(&best).unwrap()).ok();
        best
    }

    fn benchmark_kernel(&self, kernel_name: &str, shape: (usize, usize, usize)) -> AutotuneConfig {
        let mut best = AutotuneConfig { cycles_per_invocation: u64::MAX, ..Default::default() };
        
        for block_dim in [64, 128, 256, 512] {
            for tile_kv in [32, 64, 128, 256] {
                for grid_stride in [1, 2, 4] {
                    if let Ok(cycles) = self.run_once(kernel_name, shape, block_dim, tile_kv, grid_stride) {
                        if cycles < best.cycles_per_invocation {
                            best = AutotuneConfig {
                                kernel_name: kernel_name.into(),
                                block_dim,
                                tile_kv,
                                grid_stride,
                                cycles_per_invocation: cycles,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                            };
                        }
                    }
                }
            }
        }
        best
    }

    fn run_once(&self, kernel: &str, shape: (usize, usize, usize), block: u32, tile: u32, stride: u32) -> Result<u64, Error> {
        // Launch kernel with rocprof counters: SQ_WAVES, VALU_ACTIVE, LDS_ACTIVE
        // Return elapsed cycles from PM4 counter
        todo!("wire rocprof or use hipEventElapsedTime as proxy")
    }
}
```

### Integration
```rust
// In device.rs
impl RocmDevice {
    pub fn autotuned_launch_config(&self, kernel: &str, shape: (usize, usize, usize)) -> LaunchConfig {
        AUTOTUNER.get_or_tune(kernel, shape).into()
    }
}
```

### Acceptance
- Autotune runs < 5 s at first `RocmDevice::new()` call
- Config cached across runs; 10–20% kernel speedup vs hardcoded defaults
- CI gate: `cargo test --features autotune` verifies configs load

---

## 3.7 Profiling CI Gate (P2)

### Why
Prevent performance regressions. Kernel cycles/invocation must not increase > 5% without justification.

### Target
- **Crate**: `grim-backend-rocm`
- **Files**: 
  - `.github/workflows/rocm-perf.yml` (new)
  - `crates/grim-backend-rocm/tests/perf_gate.rs` (new)

### CI Workflow — `.github/workflows/rocm-perf.yml`
```yaml
name: ROCm Perf Gate
on: [push, pull_request]
jobs:
  perf:
    runs-on: [self-hosted, rocm, gfx1036]  # or MI200/MI300 runner
    steps:
      - uses: actions/checkout@v4
      - name: Build release
        run: cargo build --release -p grim-backend-rocm
      - name: Run microbenchmarks
        run: cargo test -p grim-backend-rocm --release --test perf_gate -- --nocapture
      - name: Upload rocprof traces
        uses: actions/upload-artifact@v4
        with:
          name: rocprof-traces
          path: target/perf/*.csv
      - name: Compare with baseline
        run: |
          python scripts/perf_compare.py \
            --baseline artifacts/baseline_${{ github.base_ref }}.json \
            --current target/perf/results.json \
            --threshold 0.05
```

### Perf Test — `tests/perf_gate.rs`
```rust
use grim_backend_rocm::{RocmDevice, Shape, DType, ArithType};
use std::time::Instant;

#[test]
fn perf_qkv_attention_baseline() {
    let dev = RocmDevice::new(0).unwrap();
    let (seq_len, num_heads, head_dim) = (2048, 32, 128);
    
    let q = dev.random_tensor(&[seq_len, num_heads, head_dim], DType::F32).unwrap();
    let k = dev.random_tensor(&[seq_len, num_heads, head_dim], DType::F32).unwrap();
    let v = dev.random_tensor(&[seq_len, num_heads, head_dim], DType::F32).unwrap();
    
    // Warmup
    for _ in 0..10 {
        let _ = dev.qkv_attention(&q, &k, &v, &Shape::from(&[seq_len, num_heads, head_dim])).unwrap();
    }
    
    // Timed
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = dev.qkv_attention(&q, &k, &v, &Shape::from(&[seq_len, num_heads, head_dim])).unwrap();
    }
    dev.synchronize().unwrap();
    let elapsed = start.elapsed().as_micros() as f64 / iters as f64;
    
    println!("PERF: qkv_attention(seq={}, heads={}, dim={}) = {:.2} µs", seq_len, num_heads, head_dim, elapsed);
    
    // Assert against baseline (stored in target/perf/baseline.json)
    let baseline = std::fs::read_to_string("target/perf/baseline.json")
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("qkv_attention_2048_32_128").and_then(|x| x.as_f64()));
    
    if let Some(b) = baseline {
        assert!(elapsed <= b * 1.05, "Performance regression: {:.2} µs > {:.2} µs (5% threshold)", elapsed, b * 1.05);
    }
}
```

### Acceptance
- CI fails on > 5% regression
- Baseline updated via `cargo test --release -- --update-baseline`
- Artifacts uploaded for roofline analysis

---

## 3.8 Multi-GPU with RCCL (P3 — Scale-Out)

### Why
Single GPU VRAM limits model size. Tensor/pipeline parallelism across GPUs scales throughput.

### Target
- **Crate**: `grim-distributed` (new) + `grim-backend-rocm` RCCL integration
- **Files**: 
  - `crates/grim-distributed/src/lib.rs` (new, ~500 lines)
  - `crates/grim-backend-rocm/src/rccl.rs` (new, ~300 lines)

### Starter Code — `grim-backend-rocm/src/rccl.rs`
```rust
use rccl::*;  // rccl-sys or rccl crate

pub struct RcclComm {
    comm: rcclComm_t,
    stream: hipStream_t,
}

impl RcclComm {
    pub fn new(rank: usize, world_size: usize, devices: &[RocmDevice]) -> Result<Self, Error> {
        let mut comm = std::ptr::null_mut();
        let dev_ids: Vec<i32> = devices.iter().map(|d| d.ordinal() as i32).collect();
        let res = unsafe { rcclCommInitRank(&mut comm, world_size as u32, dev_ids.as_ptr(), rank as u32) };
        if res != rcclSuccess { return Err(Error::Backend("RCCL init failed".into())); }
        Ok(Self { comm, stream: devices[rank].active_stream() })
    }

    pub fn all_reduce(&self, buffer: &mut RocmStorage, count: usize, dtype: rcclDataType_t, op: rcclRedOp_t) -> Result<(), Error> {
        let res = unsafe {
            rcclAllReduce(
                buffer.device_ptr().unwrap() as *const _,
                buffer.device_ptr().unwrap() as *mut _,
                count,
                dtype,
                op,
                self.comm,
                self.stream,
            )
        };
        if res != rcclSuccess { return Err(Error::Backend("RCCL all_reduce failed".into())); }
        Ok(())
    }
}
```

### Integration in `grim-distributed`
```rust
pub struct TensorParallelAttention {
    rccl: Arc<RcclComm>,
    local_heads: usize,
    head_dim: usize,
}

impl TensorParallelAttention {
    pub fn forward(&self, q: &RocmStorage, k: &RocmStorage, v: &RocmStorage) -> Result<RocmStorage, Error> {
        // 1. Local QK^T for assigned heads
        // 2. All-reduce partial scores across GPUs
        // 3. Local softmax + local score×V
        // 4. All-reduce partial outputs
        self.rccl.all_reduce(partial_scores, ...)?;
        // ...
    }
}
```

### Acceptance
- TP=2 on 2×MI300: ≥ 1.9× throughput vs single GPU
- Pipeline parallel: bubble < 10% at 8 micro-batches

⚠ **Correction (from skills creation, `rocm-multi-gpu-rccl`):** the acceptance numbers above assume **MI300 / CDNA Infinity Fabric** (xGMI, ~600 GB/s peer bandwidth), which mirrors the Instinct data-center target. grim's actual deployment targets **consumer RDNA** (gfx1036 / gfx1200), which has **no Infinity Fabric between GPUs** — peer access uses PCIe 4.0 x16 (≈32 GB/s on paper, ~1–10 GB/s in practice after overhead), orders of magnitude lower than xGMI. Adjust TP=2 throughput expectations accordingly: confirm `hipDeviceCanAccessPeer` (then `hipDeviceEnablePeerAccess`), but do **not** assume xGMI-class bandwidth. Use smaller `rcclAllReduce` chunks (so comm latency doesn't dominate) and overlap compute with comm aggressively (`rocm-profiling-perf` RCCL tracing). For Instinct installs the numbers above hold; for grim's default consumer target, scale expectations down to roughly PCIe-bounded and verify against the real PCIe topology before claiming a percentage gain.

---

## Anti-Pattern Enforcement Checklist

| Rule | Enforcement |
|------|-------------|
| **No file > 1500 lines** | `cargo clippy -- -W clippy::too_many_lines` fails if any file > 1500 |
| **New code never pushes file over 1500** | Pre-commit hook: `git diff --name-only | xargs wc -l | awk '$1 > 1500'` |
| **Modularize before adding** | Any edit to `lib.rs` (4630 lines) requires split first |
| **Kernel files ≤ 500 lines** | Each kernel in own file under `kernels/` |
| **Test files ≤ 500 lines** | One test module per feature |

### Immediate Modularization Task (before any Phase 3 code)
```bash
# Split grim-backend-rocm/src/lib.rs (4630 lines) into:
mkdir -p crates/grim-backend-rocm/src/{kernels,memory,graph_capture}
# 1. Move COMPUTE_KERNEL_SOURCE + kernel helpers → kernels/compute_kernels.rs
# 2. Move allocator + upload → memory/allocator.rs + memory/pool.rs
# 3. Move graph capture → graph_capture.rs
# 4. Move device struct + methods → device.rs
# 5. lib.rs becomes re-exports only (~300 lines)
```

All Phase 3 code lands in the new modular structure — **zero lines added to `lib.rs`**.

---

*End of spec additions.*
