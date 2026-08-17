# grim-backend-rocm — remaining implementation plan

A single document for the work left after the review pass. Not a backlog — a **prioritized, scope-bound, deliverable-shaping** list for the ROCm crate only. Anything here is post-review: the 2×16GB Qwen3.8-27B load/decode path is addressed by the changes already landed; this page is what’s left and in what order.

## What we already did

For the record, so the list below is bounded and not re-litigated:

- **MXFP4 decode path** — zero-copy framed-buffer split (no per-GEMM alloc/copy/sync on the stored blob), scalar kernels replaced with `float4`/`uint4` loads + one `exp2f` per E8M0 micro-block + `__launch_bounds__`, RoPE partner columns via `__shfl_xor_sync` instead of full partner-column dot recompute, split-K decode lane for m≤8/k≥2048. Verified numerically identical to a CPU oracle (7 golden-parity tests pass).
- **Per-op syncs** — 44+ `hipStreamSynchronize`/`hipDeviceSynchronize` cleared across QKV/emb/rope/bias/yarn/batched/GEMMs/replay; replaced with `hipFreeAsync` where the buffer still needs releasing.
- **Launcher fast cache** — cached `(entry, grid)` → `hipFunction` so repeat launches skip the source-regenerate/seahash/CString work; `str_interner` so `Box::leak` per call becomes one leak per unique (entry, arch). `env::var("GRIM_GPU_TARGET")` per GEMM eliminated.
- **RCCL multi-GPU correctness** — BF16 → `NCCL_BFLOAT16` (5 dtype arms), peer access pinned to src before `hipDeviceEnablePeerAccess` with original-device restore, RCCL device all-reduce for F16/BF16 TP activations (no more D2H→CPU-sum→H2D round trip per RowParallel layer per token), split-K `solution_index` now passed through (was hardcoded 0).
- **Load / VRAM / staging** — pool cap floor 128MB→512MB, ceiling 4GB; `zeros` uses async `hipMemsetAsync` on the active stream; `advise` no longer fires a whole-tensor self-copy on the null stream when XNACK is off; `copy_route` host-bounce staging cached per stream.
- **Verify** — `cargo check -p grim-backend-rocm` clean; 246/246 lib tests pass; 7/7 MXFP4 golden-parity tests pass.

Nothing in this document should be treated as blocking that path.

---

## How to read this plan

- Each item has: **scope**, **minimal safe change**, **dependency / risk**, **test hook**, **done looks like**.
- Ordered by **payoff × risk**, not by urgency. Severity here is about throughput/numerical-correctness per step, not about whether the crate currently runs.
- Items are **independent enough to stop at any one** without regressing what is already landed. Some have a two-stage shape — land stage 1 if time is short and still have a cleaner result.
- Training-side items are **numerics-grade**, not bit-exact. Do not use golden-exact assertions where floating-point non-associativity applies; use tolerance-first verification and a reference when one exists.

---

### P1 — M+Adamfp16: in-place f16 master-weight truncation + redundant O(N) dw loop + racy scale write

**Severity / impact** — High for any sustained training run on this code path. Truncating the master weight to fp16 in-place each step compounds precision loss over training; the redundant loop applies the same `dw` to every column N times; the racy `if (new_scale > scale_val) { scale[...] = ... }` is a data race on a per-weight scale.

**Scope**
- `kernels/fused_dequant_gemm.rs` — around lines 252–269 (the per-column `dw` loop), lines 278–280 (the racy scale write), line 269 (the `(_Float16)new_w` truncation).

**Minimal safe change**
1. Stop truncating the master weight in-place to fp16. Keep the master weight f32 in-place; derive any need for a fp16 view from that master rather than writing fp16 back into the storage the master lives in.
2. Dedupe the `dw` application so each unique weight element is touched once per update, not N times. The loop-invariant `dw` should not be applied per-column in an inner loop.
3. Replace the racy per-weight scale read-modify-write with a single-owner scale maintenance path — either a separate kernel, or keep scale maintenance out of the update kernel entirely.

**Dependency / risk**
- Lowest blast radius of the training items, but it changes the semantics of the weight storage. Before landing, verify nothing downstream is reading the weight buffer as fp16 in-place (load/export/push paths that expect the fp16 layout).
- Keep stage 1 (master-weight semantics + scale-maintenance isolation) separate from stage 2 (the dedup loop). Stage 1 is correctness; stage 2 is perf.

**Test hook**
- Per-step weight drift vs a reference f32 master after N steps on a small constant-gradient test where you can assert the master stays f32 and the quantized view is stable. If no reference run exists, gate on the constant-gradient case first.

**Done looks like**
- Master weights are f32 in-place; in-place fp16 truncation gone.
- Per-unique-weight update loop is not O(N·same-dw); scale maintenance is not a racy per-element read-modify-write in the update kernel.
- No downstream crash or layout assertion on the load/export paths after the change.

---

### P2 — Charon MoE backward: scalar triple-nested + global `atomicAdd` per weight element

**Severity / impact** — High for MoE training throughput; currently recomputed forward intermediates in the backward kernel plus per-weight global atomics. This is the one item on the list that needs on-device testing to verify numerically — it renders correctly today but costs a lot per step.

**Scope**
- `kernels/charon_backward.rs` — around lines 143–176 (the forward recomputation loops `hg`, `hu`, plus the `atomicAdd` sites at ~156/168–176).
- `roc_device.rs` charon MoE launcher block (~4900–5600) — the per-call `hipMalloc`/`hipStreamSynchronize`/`hipFree` per routing array, which compounds the cost but is a separate loop.

**Minimal safe change**
1. Save forward intermediates (act gate / up activations, or routed per-expert per-token output) into a persistent scratch buffer per layer during the forward pass, rather than recomputing the full `inter×hidden×hidden` serial dot in backward.
2. Replace per-weight `atomicAdd` with per-expert tiled accumulation: compute `ddw`/`dgw`/`duw`/`dx` as GEMMs into per-expert scratch, then a single thread-owner write (or an atomic only at tile granularity).

**Dependency / risk**
- Affects training only. Golden verification is numeric, not bit-exact — use tolerance-first verification and a reference run if one exists.
- Do not touch the forward MoE dispatch unless you are also fixing the per-call routing-buffer alloc/free; that is a separate hot loop and should not be bundled into this item.
- Two-stage: stage 1 is saved intermediates (cost is forward storage, not a correctness regression); stage 2 is the GEMM-based weight gradient (biggest perf win, highest verification burden). Land stage 1 if time is short.

**Test hook**
- Backward gradient-norm check on a single MoE layer with known inputs, before and after. If you can’t run on-device here, defer to a machine that can — no local smoke is worth more than a real backward sanity on the target arch.

**Done looks like**
- Backward does not recompute the full forward dot-product per weight element from scratch.
- Weight gradients are not hammered by one global atomic per weight element; at worst an atomic per tile.
- Forward scratch is accounted for in the scratch/allocator path, not leaked.

---

### P3 — CPU `cross_entropy_gpu` over full logits → wire the on-device fused CE kernel

**Severity / impact** — High for training steps on this path, but not the 2-GPU load/decode path. The cost is a full D2H + H2D of the logits plus a CPU softmax over the whole batch×vocab matrix per step; an on-device fused CE kernel already exists in-file and is not wired.

**Scope**
- `roc_device.rs` `cross_entropy_gpu` (around lines 10645–10712, the CPU softmax path).
- The call site that currently routes to `cross_entropy_gpu`.

**Minimal safe change**
- Keep the CPU path as a fallback (CPU-only boxes / reference runs), but route the ROCm training path to the device fused CE kernel for the shapes it supports.
- Gate on shape support: if the fused kernel doesn’t cover a shape/class, fall back to the CPU path rather than failing the step.

**Dependency / risk**
- Training path only. Verification is numerics-centered (token-level CE vs reference), not a golden kernel parity test.
- Start with a single-batch, small-vocab sanity first before claiming end-to-end parity.

**Test hook**
- Token-level CE against a reference on a contrived small case (small batch, small vocab) where you can assert the fused path matches within tolerance.

**Done looks like**
- ROCm training steps use the device fused CE kernel for supported shapes; CPU path remains for unsupported shapes and CPU-only boxes.
- No step crash on shapes the fused kernel doesn’t cover.

---

### P4 — IQ / GQuant backward kernels: still scalar `div/mod` per element

**Severity / impact** — Moderate. Forward path already improved (standalone dequant upgraded to 64-thread blocks, fused dispatch is the default forward path). Backward is the gap: per-MAC `int sb_idx = k / 256; int in_sb = k % 256;`, one dequant per MAC, one thread per output.

**Scope**
- `kernels/iq_gemm.rs` and the KQuant backward variants — the per-MAC `div/mod` pattern and per-thread-per-output layout.
- `kernels/iq_dequant.rs` — already restructured forward standalone dequant; backward is the remaining gap.

**Minimal safe change**
- Same shape of transformation as the forward path: process one 32- or 256-element superblock per thread, load the scale once, avoid per-element `div/mod`.
- Do not restructure all 16 variants at once — pick one representative (e.g. `iq4xs`, or the variant your workload actually uses) and prove the pattern first, then expand.

**Dependency / risk**
- Backward numerics — tolerance-grade, not bit-exact. Start with a single backward kernel + gold comparison on a contrived small case.

**Test hook**
- A single backward kernel against a gold comparison on a small contrived case, with tolerance-first verification.

**Done looks like**
- The backward path for the representative variant does not do per-element `div/mod` and per-MAC full dequant; at least one variant is proven before expansion.

---

## Ordering and blocking notes

1. **M+Adamfp16** is the first training item to land — smallest blast radius, correctness-first, and doesn’t need the full charon rework to be valuable.
2. **Charon MoE backward** is second — two stages, so you can stop after stage 1 if short on time and still have a cleaner (if not yet fastest) backward.
3. **CPU cross-entropy → device fused CE** is wiring-only for an existing kernel — good training-perf win, but not the 2-GPU decode scenario.
4. **IQ/GQuant backward superblock** is last — one representative variant first, then expand; gating on the actual quant format your workload uses keeps this from becoming boilerplate.

None of these block the 2×16GB Qwen3.8-27B load/run path. If you want any single one turned into its own implementation plan with exact file/line targets and a minimal test first, ask before starting.

---

## Skills used as the frame for this document

- **writing-plans / project-planning** — item shape (scope / minimal safe change / dependency / test hook / done), prioritized by payoff×risk, stop-at-any-one.
- **clean-code-guard** — stage 1 / stage 2 separation where a correctness change and a perf change share the same hot path; don’t bundle them.
- **caveman** — short sentences, no filler, concrete file/line scope.
- **ponytail / humanizer** — tone is direct but not terse to the point of leaving ambiguity; plain language where the technical term isn’t the point.
- **writing-guidelines** — what this is and is not, for the record up front, so the list is bounded.
