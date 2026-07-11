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
- Tiled flash-attention-style blocking over `(query_tile, kv_tile)` with online softmax kept in registers/LDS, replacing Phase 1's one-block-per-single-query-position scheme — amortizes K/V loads across queries/heads sharing KV data.
- Paged/block-table-aware kernel (Option B from the Step 0 discussion above) instead of requiring a pre-gather into a flat buffer — avoids a full extra memory copy per call once the paged KV cache in `grim-memory` is actually wired up.

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
- **Compiler/build flags**: target the exact GPU arch (e.g. `--offload-arch=gfx942`) rather than a generic target so arch-specific codegen (MFMA availability, LDS size) is used; verify `hipcc` optimization level.
- **Environment tuning**: `HSA_ENABLE_SDMA`, `GPU_MAX_HW_QUEUES`, and XNACK/HMM settings (`xnack_enabled` is already tracked in `RocmDeviceProps`) all affect copy-engine usage and page-fault behavior — usually want XNACK off for latency-sensitive inference. Pin NUMA affinity between GPU and the feeding CPU thread on multi-socket/multi-GPU hosts.
- **Pinned host memory** for any remaining host↔device transfers in the pipeline.
- **Profile before optimizing further**: use `rocprof`/`rocprofv2` for kernel-level timing and `omniperf` for roofline analysis (compute-bound vs. memory-bound) to confirm where the actual bottleneck is before investing in any specific item above — a well-tiled kernel can still lose to register-spill-induced low occupancy, which only shows up under profiling.
- **Multi-GPU** (if this deployment uses tensor/pipeline parallelism): use RCCL for collectives, confirm Infinity Fabric/xGMI links are used for GPU-to-GPU traffic instead of falling back to PCIe, and overlap communication with compute rather than treating all-reduce as a blocking barrier.

Define the before/after benchmark (Phase 2's acceptance criteria above) against the *system*, not just the kernel in isolation — some of these system-level changes will show up as bigger throughput gains than the kernel rewrite itself, so sequence or parallelize accordingly rather than assuming kernel work alone gets "most performant."

**Acceptance criteria for Phase 2 (draft — refine when this is actually scoped):**
- Numerical parity with the Phase 1 CPU reference maintained at the same tolerance, across the same test matrix plus whatever new shapes the tiled/paged design introduces.
- Documented before/after benchmark on real hardware, not just theoretical occupancy.
- Phase 1's `enabled` gate semantics preserved — a Phase 2 regression must be gated off the same way a Phase 1 one would be.
