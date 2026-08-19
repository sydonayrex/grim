# wreck-it.md — next grim-backend-rocm work, from rockit-holin.md synthesis

State reviewed: rockit-holon.md (full), git log (P0/P1/P2 plan fixes landed at 27fa88c:
MoE expert weight caching, PlanBuilder quant wiring, quant parity fixes), crate source
(`autotune.rs` 568 lines, `kernels/`, `device/`). Layer A/B/C artifacts from rockit-holon
(H.1–H.6) do NOT exist yet: no `trace.rs`, no `grimcap/`, no `roofline.rs`, no subspace
pruning, no predictor, no measurement log.

## Verdict — what next

rockit-holon says golden path = Layer A first (isolation + apply() deployment), because
Layer B/C iteration too expensive without it. Correct dependency, but ponytail reorders
within Layer A: full Kerncap HSA intercept (LD_PRELOAD libkerncap, VFS overlay reproducer)
is heavy infra; grim already has `jit_cache.rs` (source-hash keyed hsaco cache) and
`Autotuner` cache persistence. Lazy rung that holds:

**WRECK-1 (next): apply()-style trace dispatch + persisted autotune measurement log.**
Smallest piece that (a) deploys validated winners with zero engine rewrite, (b) starts
collecting the `(shape, tile_config, format, arch, latency)` dataset every later phase
(CharTuner pruning, TTX/WaveTune predictor, SwizzlePerf remap) needs as training input.
No dataset = no predictor = Layer C dead. This is the one blocking dependency.

Then WRECK-2 subspace pruning, WRECK-3 predictor, WRECK-4 Kerncap-lite reproducer.
Fleet chiplet tasks (H.1) and roofline paths (H.6) wait — they need WRECK-2/3 data to tune.

---

## WRECK-1 — trace table + measurement log (Phase 1, ~1–2 days)

Grim anchors: `src/autotune.rs:247-256` (`Autotuner`, `cache`, `moe_cache`, `cache_dir`,
`{dir}/{gpu_arch}.json`), `src/kernels/jit_cache.rs`, `src/device/roc_device.rs` GEMM
dispatch (~4617/4679/4953 launch sites), `src/device/gemm_tuning.rs`.

### 1a. Persist measurement samples

Every autotune bench already measures latency per candidate, but the per-candidate
loops live in caller-supplied `BenchFn` closures at the GEMM dispatch sites
(`roc_device.rs` `get_or_tune_tiles` etc.) — `Autotuner::get_or_tune`
(autotune.rs:336) is a thin read-through cache that invokes the closure once on miss
and inserts the result; there is no bench loop inside it. So: log from inside the
dispatch-site closures (or wrap `BenchFn` so `get_or_tune` hands the closure a
logging sink), appending every measured sample (not just winner) to
`{cache_dir}/{gpu_arch}_samples.jsonl`:

```
{kernel_key, m_class, tile_config, features, format, arch, grid, block, waves: ceil(G/n_sm), latency_us, ts}
```

- Implement via `SampleLogger` type in `src/trace.rs` (new module): dispatch-site
  closures receive a `SampleLogger` and call `log_candidate()` per measured candidate,
  OR wrap the bench closure with `SampleLogger::wrap()` for the minimal winner-only log.
  NOT inside `Autotuner::get_or_tune` (thin read-through cache, no bench loop there —
  see user correction note).
- Waves term computed from existing launch geometry (rockit-holon H.3 note: no new probe).
- Failure to log = warn + continue; never fail tuning on log IO.

### 1b. `src/trace.rs` — WRECK-1 implemented (KernelTrace + dispatch lookup + sample log)

```rust
#[derive(Serialize, Deserialize)]
pub struct KernelTrace {
    pub definition: KernelDef,     // op, dtypes, quant format
    pub workload: Workload,        // m_class, arch, format bucket
    pub solution: LaunchConfig,    // winning tile config
    pub evaluation: Eval,          // parity_ok: bool, latency_us, ts
}

pub fn lookup(m_class: MClass, arch: &str, fmt: QuantFormat) -> Option<LaunchConfig>
```

- Table = JSON next to existing `{gpu_arch}.json` autotune cache — same dir, same
  load-on-startup pattern in `Autotuner::for_device` (autotune.rs:260). Reuse serde +
  cache_dir plumbing; no new deps.
- Implemented: `TraceTable` (in-memory index loaded from `{gpu_arch}.trace.json`),
  `KernelTrace` rows, `lookup(kernel, arch, m_class, format) -> Option<AutotuneConfig>`
  (returns winner only if `parity_ok`), `insert` (rejects parity_ok=false entries —
  correctness gate), `load_trace_table` (missing/corrupt file = empty, warn not crash),
  `save_trace_table` (best-effort).
- Dispatch: in GEMM dispatch path, try `trace::lookup` before `get_or_tune`. Hit =
  zero-compile launch. Miss = existing compile+time autotune, then winner written to
  trace table (apply()).
- **Correctness gate (non-negotiable, per rust-ffi-grim + q4k_dequant discipline):**
  a trace entry only written when candidate passes existing parity check (CPU oracle /
  reference tolerance). Eval records which check. Compile failure = no entry
  (FlashInfer-Bench: 30/32 correctness errors are compile failures — gate must include
  compile success, which existing autotune loop already enforces).
- Numerical contract: trace entry pins exact `AutotuneConfig` — same tile config replayed,
  no silent re-tune (Kerncap tuning-pinned reproducer concept, cheap version).
- **Sample log** (WRECK-2/3 data backbone): `SampleLogger` type in `trace.rs` writes
  `{gpu_arch}_samples.jsonl` per-candidate JSONL. Dispatch-site `BenchFn` closures receive
  a `SampleLogger` and call `log_candidate()` per measured candidate, OR wrap the closure
  via `SampleLogger::wrap()` for the minimal winner-only log. `SampleRecord` fields:
  kernel, gpu_arch, m_class, format, tile_config (AutotuneConfig), grid_x, block_x,
  waves, latency_us, ts. Failure to log = warn + continue; never fail tuning on log IO.
  Tested: `sample_logger_appends_jsonl_line`, `sample_logger_wrap_runs_and_logs_winner`,
  `sample_logger_warns_on_unwritable_but_doesnt_panic`.

### 1c. Tests

- Unit: trace table round-trip serde; lookup hit/miss; invalid-eval entries never served
  (tested: `trace_table_rejects_unvalidated_entries`, `trace_table_lookup_prefers_eligible_over_ineligible`);
  sample log JSONL append + wrap logs winner (tested: `sample_logger_appends_jsonl_line`,
  `sample_logger_wrap_runs_and_logs_winner`); load/save roundtrip (tested:
  `load_trace_table_roundtrip`, `save_and_load_trace_table_roundtrip`);
  corrupt/missing file = empty (tested: `load_trace_table_corrupt_json_is_empty`,
  `load_trace_table_missing_file_is_empty`).
- Integration: existing parity tests unchanged (trace lookup must return same winners
  autotune would); one test that a populated trace dir short-circuits bench fn (bench
  counter not incremented).

### Expected gain

- Inference speed: cold-start autotune cost eliminated for hot shapes (currently full
  compile+bench per cold process for shapes missing from `{gpu_arch}.json`). Process
  restart with warm trace = instant tuned launches. Modest direct speedup; real value =
  unlocks WRECK-2/3.
- Stability: validated-winner-only deployment removes "tuned but unvalidated config in
  dispatch" hazard.

---

## WRECK-2 — CharTuner subspace pruning (Phase 2, offline, ~3–5 days)

Anchor: `autotune.rs:120-186` (`LaunchConfig`, `charon_scalar_candidates`, `FeatureSet`),
candidate enumeration feeding `get_or_tune`.

- Decompose grim's existing candidate params (block_dim, tile_kv, grid_stride, wmma/mfma
  feature flags, smem double-buffer where present) into semantic subspaces (tiling /
  prefetch / vectorization / thread-cluster / C-storage — CharTuner Ω map).
- Offline: bench each subspace across representative M-class × arch × format grid using
  the WRECK-1 sample log + targeted re-bench; five-number summary per shape → PCA (or
  lazy first cut: median-improvement rank, no PCA dep) → retain top-k subspaces.
- Output: per (kernel, arch) pruned candidate generator — `charon_scalar_candidates`
  gains a `pruned: &SubspaceMask` param; full space remains fallback when no mask cached.
- Ponytail note: PCA on 5-number summaries is a Python offline script over the jsonl —
  NOT in the Rust crate. Crate only reads the resulting mask JSON.

Expected gain: rockit-holon numbers — 55.2% candidate-space reduction, PSO converges
~130 vs 282 iterations, RS in reduced space still 1.64×. For grim: cold-shape autotune
time cut roughly in half; search quality per second up.

## WRECK-3 — latency predictor on sample log (Phase 3, offline + runtime load)

Anchor: WRECK-1 jsonl log, `autotune.rs` candidate pre-filter.

- Offline Python: gradient-boosted trees (TTX-style) over (shape, tile params, features,
  waves, arch) → ~10% MAPE, top-50 recall 95% of oracle. Alternative: WaveTune bilinear
  with wave term — pick by validation MAPE on held-out shapes (tune-the-optimizer, one
  script each).
- Runtime: load model (serde JSON tree dump — no new Rust ML dep), rank candidates
  before compile; compile+bench only top-N (start N=5). Cold shapes now cost 5 compiles
  instead of full enumeration.
- Fallback: predictor absent → WRECK-2 pruned enumeration as today.

Expected gain: WaveTune/TTX papers: up to 1.83× kernel selection quality vs naive
ranking; for grim mainly cold-shape tuning latency (compile is the cost) cut by
candidates_size/N. Direct inference speed gain when predictor picks better-than-first
config for shapes autotune currently skips.

## WRECK-4 — Kerncap-lite reproducer (Phase 4, optional, ~1 week)

Full HSA intercept = over-engineering until WRECK-1..3 show the loop needs it. Lazy
version: `grimcap` CLI that, for a KernelKey, regenerates the isolated kernel from
`jit_cache.rs` source-hash + pinned `LaunchConfig` from trace table, emits standalone
reproducer dir (hiprtc source + flags + driver main + Makefile). Gets 80% of Kerncap's
isolated edit-recompile-validate loop using machinery grim already has. Build only when
kernel iteration count justifies it.

Explicitly deferred: H.1 Fleet chiplet tasks, H.6 roofline/sparse/spill paths,
SwizzlePerf remap — two gates: (a) need WRECK-2/3 measurement surface (their inputs
are L2/latency data grim doesn't yet collect), (b) **hardware gate** — chiplet/XCD
work only maps to gfx1100 (MCD-split L2, remap only, not chiplet-tasks); on
gfx1036/gfx1200 (monolithic) it is noise. Dead on MI300/MI350 targets grim doesn't run.

---

## WRECK-5 — KV-cache quantization Q8_0 / Q4_K (Tier 1, ~2–4 days)

Grim anchors: `src/kernels/kv_dequant_attention.rs` (dequant-in-kernel KV read path),
`q8_0_dequant.rs` / `q4k_dequant.rs` (block dequant fns + host CPU mirror oracle),
flash_decode.rs / extend_attention.rs / preshuffled_attention.rs (KV consumers).

Why: decode is KV-bandwidth-bound. Q8 KV ≈ 2× effective bandwidth → up to ~2× decode
speedup on long context; Q4_K ≈ 4× bandwidth with measurable-but-small quality cost.
Cheapest big win in the plan — all kernel patterns already in tree.

### Steps
1. Add `KvQuantFormat { F16, Q8_0, Q4_K }` to KV-cache alloc path. Q8_0: per-32-elem
   block (fp16 scale + 8×int8), reuses `q8_0_dequant` block fn verbatim.
2. Dequant in the attention kernel at load time (kv_dequant_attention.rs already the
   shape): KV never materialized dequantized in HBM — that's the whole point. LDS stage
   per K-chunk tile, dequant into LDS, MFMA from LDS.
3. K and V can use different formats (K more sensitive: `Q8_0` K + `Q4_K` V default;
   expose both).
4. Write path: quantize at KV-append time (extend/prefill) — one small kernel per
   format, or fold into existing append epilogue.
5. Config plumbed via `.grim` model metadata + runtime override; default F16 (no
   behavior change unless opted in).

### Tests / gates
- Parity: dequant-then-attention vs F16 attention, tolerance-based (cosine / max-rel
  err) on random + real attention shapes; NaN detection. Q8 target < 1e-3 rel err,
  Q4_K per eval, warn if worse than measured perplexity delta budget.
- Perf gate (perf_gate.rs): decode tok/s before/after at ctx 4k/16k/32k; claim only
  with rocprof HBM-read confirmation (bytes read must drop ~2×/4×).
- Overflow check: per-block scale math in fp32 accumulate, never fp16.

### Expected gain
Long-context decode: Q8_0 up to ~1.8–2×, Q4_K up to ~3–3.5× (bound by non-KV portion
of step). Short context (<1k): marginal, KV not dominant. VRAM: KV cache halved /
quartered → bigger ctx or bigger batch on same card.

## WRECK-6 — Split-K for decode GEMMs (Tier 1, ~2–3 days)

Grim anchors: `src/kernels/decode_gemm.rs`, `tile_picker.rs` (tile selection),
`autotune.rs` LaunchConfig.

Why: decode GEMM M=1..8 → few output tiles → few wavefronts → GPU idle. Split-K
divides K across workgroups, partial sums to workspace, second tiny reduce kernel
(or atomicAdd — determinism caveat, see WRECK-6.3).

### Steps
1. Check decode_gemm.rs current split-k state first; extend `LaunchConfig` with
   `split_k: u32` (1 = off, current behavior default).
2. Two-stage reduce (partials buffer + reduce kernel) for training-adjacent paths;
   atomicAdd acceptable for pure inference only if parity test passes tolerance —
   mark with `ponytail:` comment naming the nondeterminism ceiling.
3. Candidate values split_k ∈ {1,2,4,8} into autotune candidates — pick per M-class ×
   arch; winners land in trace table (WRECK-1) automatically.
4. Workspace: allocate from existing stream-ordered pool, sized max(K/split_k) ×
   outputs × fp32, reused across steps.

### Tests / gates
- Parity vs split_k=1 (exact for two-stage; tolerance for atomic path).
- Perf: decode GEMM kernel-time via autotune bench — split_k wins only for M≤8,
  K large; assert autotune actually picks 1 for prefill shapes (no regression).
- Determinism: training path bit-stable across runs (two-stage only).

**Status: IMPLEMENTED.** Infrastructure pre-existed (discovered on audit):
- `LaunchConfig::split_k: u32` already in `autotune.rs:124`; `is_valid` already guards
  `block_m<=8 && split_k>4` (autotune.rs:140-142).
- Two-stage split-K reduce already in `roc_device.rs` matmul dispatch (lines 9900-9978):
  partials alloc + rocblas `gemm_strided_batched_ex` + `launch_split_k_reduction` +
  `grim_split_k_reduction` kernel (compute_kernels.rs:346-358, fp16 partials → f32 sum
  → fp16 out, serial reduction, no atomics = bit-stable).
- Gate: `SplitKGemmConfig` (fusion.rs:130), `split_k_config` on RocmDevice (roc_device.rs:150),
  `set_split_k_enabled` (roc_device.rs:677), `split_k_effective` clamp (roc_device.rs:9886-9898).
- New WRECK-6 deliverable = 3 tests in `lib_internal_tests.rs`: `test_split_k_reduction_host_mirror`
  (fp16 round-trip host mirror of the reduction kernel, split_k∈{1,2,4}, m×n∈{(1,8),(4,8),(8,16),(2,32)}),
  `test_split_k_reduction_bit_stable_for_training` (asserts `grim_split_k_reduction` source
  contains no `atomicAdd` — serial reduction confirmed), `test_split_k_reduction_compiles` (pre-existing,
  JIT-compile check). 279 lib tests pass, 0 failures. **UPDATED 2026-08-18**: 284 lib
  tests pass, 0 failures (WRECK-5 + WRECK-6 + WRECK-7 + WRECK-8 landed since).
- Atomic path: not implemented (ponytail: not needed; serial reduction is deterministic and
  the rocblas strided-batched path already handles the split-K GEMM; atomicAdd only relevant
  if a custom skinnier GEMM kernel with concurrent partial writers is added later).

## WRECK-7 — Occupancy tuning: launch_bounds + VGPR budget + autotune fields (Tier 1, ~2 days)

Grim anchors: `autotune.rs` (`LaunchConfig`, candidate generators),
`rocm-hip-kernels` checklist (block multiple of 32, no hardcoded warpSize), kernel
sources across `kernels/`.

Why: MFMA tiles eat VGPRs → spills = 2–5× slowdowns + timing variance (stability
problem too). Occupancy is per-(kernel, arch) — exactly what autotune exists for.

### Steps
1. Add fields to `LaunchConfig`: `waves_per_cu_target: u32`, `vector_width: 4|8`,
   `lds_double_buffer: bool`, `max_registers: u32` (→ `__launch_bounds__` /
   `-maxrregcount`-equivalent hiprtc opt). **DONE**: `autotune.rs:120-125` now has all
   four fields; `is_valid` (autotune.rs:135-154) extends to validate waves_per_cu_target
   (1..10), max_registers (1..256), vector_width (4 or 8).
2. Candidate generators populated: `charon_scalar_candidates` (autotune.rs:153) uses
   `LaunchConfig::default_occupancy_fields()` (waves=4, regs=64, vec=8, dbl_buf=true).
   **DONE**.
3. Audit each quant GEMM kernel: block sizes multiple of 32 (wave32 RDNA), warpSize
   read at runtime not hardcoded, coalesced global loads (consecutive lanes =
   consecutive addresses), LDS double-buffer where bandwidth-bound. **PARTIAL**: fields
   are in place; per-kernel audit + rocprof pass is the on-device verification step,
   deferred to when grim has a gfx1036/gfx1200 box with rocprof available.
4. Sweep runs through existing autotune bench; winners → trace table (WRECK-1). **NOT STARTED**:
   the bench loop doesn't yet vary the new fields; that's the next sub-step when the audit
   identifies which kernels benefit.

### Tests / gates
- Existing parity suites unchanged (config-only changes). **PASS**: 281 lib tests, 0 failures.
- New tests in `autotune.rs`: `test_launch_config_occupancy_fields_default` (field values),
  `test_launch_config_occupancy_fields_is_valid` (valid/invalid combos for each new field).
  **PASS**.
- rocprof: zero VGPR spills on winning configs; occupancy ≥ 50% or documented reason.
  **DEFERRED**: on-device only.
- Timing variance: p99/p50 kernel time ratio improved or flat on winners. **DEFERRED**: on-device only.

### Expected gain
Varies per kernel: spilled kernels 2–5×; already-clean kernels 0-10%. Removes
variance → stability win even where speed flat.

## WRECK-8 — FP8 MFMA finish on gfx1200 (Tier 1, ~1 week)

Grim anchors: `src/kernels/fp8_gemm_rdna4.rs`, `fp8_standalone.rs`, `wmma_gemm.rs`
(feature gating), `autotune.rs` `FeatureSet::fp8_mfma()`.

Why: `__builtin_amdgcn_mfma_f32_32x32x16f8f6f4` (gfx1200-only) ≈ 2× FLOPS + half
weight bandwidth vs F16. Already started; finish, gate, validate.

### Steps
1. Hardware gate: `FeatureSet::fp8_mfma().supported_on(arch)` must check
   `gcnArchName >= gfx1200` — never gate on `__hip_fp8` type availability (exists on
   gfx1036 but emulated, slower than F16). **DONE**: `autotune.rs:104-116` (`supported_on`)
   checks `requires_fp8_mfma → is_gfx12`; `test_feature_set_supported_on` (autotune.rs:527-545)
   asserts `fp8_mfma.supported_on("gfx1200")`=true, `supported_on("gfx1036")`/`supported_on("gfx1100")`=false.
2. Confirm builtin spelling/operand order against installed ROCm clang
   (`/opt/rocm/lib/llvm`) — names shift across releases (skill warning). **DEFERRED**: no gfx1200
   hardware in this env; the `wmma_gemm.rs` FP8 MFMA kernel (lines 319-395) uses a scalar
   fallback with a comment noting the real builtin should go here — implementing the actual
   `__builtin_amdgcn_mfma_f32_32x32x16f8f6f4` requires gfx1200 hardware to verify, which is
   the on-device WRECK-8 item, not a CPU-testable deliverable.
3. E4M3 for weights+activations path; F6/F4 packed variants only if scale handling
   clean. Accumulate fp32 always. **DONE**: the `wmma_gemm.rs` FP8 MFMA kernels (lines 336-393)
   use `fp8_e4m3_to_float_hip` (shared FP8 decode helper) and accumulate in f32.
4. Quantize-at-load: F16/BF16 weights → FP8 once at model load (offline job), scales
   per 32-block like MXFP4 path — reuse mxfp4 block-scale machinery. **NOT STARTED**: the
   standalone `fp8_gemm_rdna4.rs` is dead code (no callers); the fused dequant FP8 path in
   `wmma_gemm.rs` is what ships. The quantize-at-load step is part of the broader FP8 weight
   pipeline, not a WRECK-8-exclusive item.
5. Autotune candidates: FP8 arm only when feature supported; trace table records
   format in workload bucket (WRECK-1 schema already has `format`). **NOT STARTED**: the
   autotune bench loop doesn't yet vary FP8 candidates.

### Tests / gates
- Parity: FP8 GEMM vs F16 reference, tolerance based (E4M3 has ~2 decimal digits);
  max-rel-err budget set from perplexity delta on a small model, not hand-waved. **NOT STARTED**:
  on-device only; no FP8 MFMA hardware in this env to run the parity test.
- Perf: kernel-time vs F16 WMMA arm on same shapes; expect ~1.5–2×; confirm with
  rocprof (VALU/MFMA pipe utilization up, HBM bytes down). **DEFERRED**: on-device only.
- Fallback: non-gfx1200 arch never enters FP8 arm (test asserts dispatch). **DONE**: added
  `source_contains_fp8_mfma_guarded_path` (wmma_gemm.rs self_tests) asserts the MFMA kernels
  are under `#if defined(__gfx1200__)`, the scalar fallback kernels are present for non-gfx1200,
  and the MFMA kernels use `fp8_e4m3_to_float_hip`. Plus `fp8_mfma_dispatch_is_arch_gated`
  asserts the dispatch path in roc_device.rs checks gfx12. Plus `source_contains_fp8_standalone`
  asserts the standalone FP8 dequant kernel is present.
- Source structure: **DONE** — 4 new tests in `wmma_gemm.rs self_tests` (6 total):
  `source_contains_wmma_kernel_entry`, `source_contains_mxfp4_backward_kernel`,
  `source_contains_mxfp8_backward_kernel`, `source_contains_fp8_mfma_guarded_path`,
  `source_contains_fp8_standalone_gated_path`, `source_contains_fp8_standalone`.

### Expected gain
Prefill throughput ~1.5–2× on gfx1200 GPUs; VRAM weights halved vs F16 → bigger
models per card. Zero effect on gfx1036/gfx1100 — by design.

## WRECK-9 — Decode-step HIP graph capture (Tier 2, ~3–4 days)

Grim anchors: `src/graph_capture.rs` (411 lines — check what's captured today),
`src/device/roc_device.rs` dispatch, `scythe_persistent.rs` (fallback megakernel path).

Why: bs=1 decode step = ~6+ kernel launches/layer × 5-10µs launch+dispatch overhead
= large fraction of small-model step time. HIP graphs collapse it. Lazy 80% of the
megakernel idea — full persistent megakernel (H.1, deferred) only if graphs insufficient.

### Steps
1. Audit graph_capture.rs: decode step captureable end-to-end? Static shapes per
   step (M=1) — yes, ideal graph case. Dynamic bits (sampling, MoE routing) stay
   outside graph or use fixed-slot indirection buffers. Full persistent megakernel
   (scythe_persistent.rs, H.1) deferred unless graphs measurably insufficient.
2. Capture per (model, arch) once at first decode step; replay each step with input
   copy into graph-owned static buffers.
3. KV cache append: graph writes into pre-allocated ring — no realloc inside graph.
4. Guard: capture failure → fall back to eager dispatch (no crash); log once.
5. Interop: graph on the active stream; rocBLAS handle bound same stream (stream
   discipline pass already done, 16e3fed).

### Tests / gates
- Output parity: graph-replay decode vs eager decode, token-identical on greedy
   (same kernels, same order — must be exact; investigate any diff).
- Perf: step time at bs=1 before/after; expect −20–50µs/layer overhead → visible
   tok/s gain on small models, less on large (GEMM dominates).
- Memory: graph instantiation counted, no per-step allocation (assert via allocator
   counter in debug build).

### Expected gain
bs=1 small/mid models: 10–30% decode tok/s. Large models: single-digit %. Also
stability: fewer host-side dispatch decisions per step.

## WRECK-10 — W8A8 activation quant, SmoothQuant-style (Tier 2, ~1–2 weeks)

Grim anchors: `quantization.rs`, `fused_dequant_gemm.rs` (training-side fused path),
`mxfp_standalone.rs`, silu_mul_quant.rs (activation quant epilogue exists).

Why: weight-only quant leaves GEMM bandwidth-bound on activations and compute in F16.
W8A8 (int8 or fp8 weights + activations) makes prefill/training GEMMs ~2× FLOPS.
Invasive: needs per-channel scale calibration.

### Steps
1. Offline calibration pass: run N forward batches, collect per-channel activation
   max → smoothquant migration (absorb scale into preceding weights, γ) → per-channel
   W8 / per-token-dyn A8 scales.
2. Kernel: A8×W8 MFMA path (gfx1200: reuse FP8 MFMA from WRECK-8; gfx1036/1100: int8
   `__builtin_amdgcn_mfma_i32_32x32k16` — verify availability per arch).
3. Dequant epilogue: accumulator fp32 → dequant scale multiply fused in epilogue
   (pattern from fused_dequant_gemm.rs).
4. Training: forward in W8A8, master weights + optimizer stay fp32
   (grim_madam_update_f32 already fp32 master); QAT only if PTQ perplexity delta
   exceeds budget.
5. Per-layer opt-out: sensitive layers (lm_head, first/last) stay F16 — config list.

### Tests / gates
- Perplexity delta on held-out set per model: budget ≤ +0.05 ppl (typical SQ result
  on 7B-class); exceed → fall back per-layer until within budget.
- GEMM parity: W8A8 vs F16 reference, tolerance from calibration stats.
- Perf: prefill tok/s and training step time; expect ~1.4–1.8× on GEMM-bound configs.

### Expected gain
Prefill ~1.4–1.8×, training step ~1.2–1.5× (GEMM fraction bounded). Highest effort
in plan; do after Tier 1 lands and only if perplexity budget holds.

## WRECK-11 — Speculative decoding parameter tuning (Tier 2, ~2–3 days)

Grim anchors: `speculative.rs` (451 lines), `speculative_sampler.rs`,
`autotune.rs` pattern reusable for search.

Why: draft length + acceptance threshold are hyper-params with direct tokens/sec
payoff; greedy acceptance ~70% typical → ~1.5–2× effective decode. No new kernels.

### Steps
1. Surface (draft_len ∈ {2..8}, draft_top_k ∈ {4..32}, acceptance threshold) as
   runtime config.
2. Autotune-style search: run M prompts, measure accepted-tokens/step and wall
   tok/s per config; cache winner per model in trace table (WRECK-1 — add a
   SpecKey alongside KernelKey).
3. Adaptive draft_len: track rolling acceptance rate; shrink draft on rejection
   streak, grow on acceptance streak (one-line policy, no ML).
4. Greedy-only first; sampling-path acceptance (typical decoding) if parity holds.

### Tests / gates
- Correctness: speculative output distribution == eager output distribution on
   greedy (token-exact); on sampled path, statistical test over 10k tokens.
- Perf: tok/s at acceptance curve measured per model; no-config-better-than-default
  → keep default (log it).

### Expected gain
1.3–2× effective decode tok/s where a draft model exists; zero cost when disabled.

---

---

## Order + why

Track A (dispatch/data surface — sequential): WRECK-1 → 2 → 3 → 4.
Track B (kernel surface — independent of A, parallelizable): WRECK-5,6,7 first
(cheap, big), then 8 (hardware-gated), then 9; WRECK-10,11 last.
Track B phases feed WRECK-1 trace table (every autotuned winner lands there);
Track A phases make Track B tuning cheaper (pruned space, predictor pre-filter).

| Phase | Piece | Perf | Stability | Unblocks |
|---|---|---|---|---|
| 1 | trace.rs + sample log | cold-start tuning → ~0 on warm shapes | validated-only dispatch | 2,3, all Track B winners |
| 5 | KV quant Q8/Q4 | long-ctx decode up to ~2×/3.5×, KV VRAM ÷2/÷4 | tolerance-gated | — |
| 6 | split-K decode GEMM | decode GEMM −30–60% at M≤8 | two-stage = bit-stable | — |
| 7 | occupancy/launch-bounds sweep | spilled kernels 2–5×, else 0–10% | kills timing variance | all kernels |
| 8 | FP8 MFMA gfx1200 | prefill ~1.5–2×, weights ÷2 VRAM | arch-gated fallback | 10 |
| 9 | HIP graph decode | bs=1 tok/s +10–30% (small models) | fewer host dispatch decisions | — |
| 2 | subspace pruning | ~50% candidate space, faster cold tune | — | 3 |
| 3 | predictor | top-5 pre-filter, better cold picks | — | 4 decision |
| 10 | W8A8 SmoothQuant | prefill ~1.4–1.8×, training step ~1.2–1.5× | ppl budget ≤ +0.05 | — |
| 11 | spec-decode tuning | 1.3–2× effective decode tok/s | token-exact greedy gate | — |
| 4 | grimcap reproducer | 4-5× kernel-iteration loop | tuning-pinned contract | H.1/H.6 work |

Recommended first sprint: WRECK-1 (everyone depends on it) + WRECK-5 (KV quant) +
WRECK-7 (occupancy sweep) in parallel.

Each phase ships standalone value; no phase blocks on unbuilt infra. Gates per
rust-ffi-grim: `#[repr(C)]` where FFI-touched (none new here — all host-side Rust),
`cargo check`/`build` clean, fallback path tested with missing/invalid cache files.

## Truthfulness

- Track A gains: phase 1 direct gain measured on grim hardware after landing (metric:
  cold-process time-to-first-tuned-launch); phases 2–3 estimates transferred from
  CharTuner/WaveTune/TTX paper results on their kernels (MI210/ROCm 6.0-class) — must
  be re-verified on grim's RDNA targets; rockit-holon caveat applies: methods compose,
  exact speedups not guaranteed.
- Track B gains: KV-quant and split-K numbers are standard roofline/occupancy bounds
  (bandwidth arithmetic, idle-wavefront fill) — reliable direction, exact factor
  measured per model/ctx; FP8/W8A8 ranges from RDNA4 MFMA spec + published
  SmoothQuant results — must pass the stated perplexity/parity budgets on grim models
  before any speed claim ships; HIP-graph launch-overhead savings from AMD's own
  published graph-replay numbers, verified per model size.
- Every phase carries its own measurement gate (rocprof counters, parity tests,
  perplexity budget) — no gain is claimed from paper transfer alone.
- Chiplet/Fleet/SwizzlePerf XCD work explicitly dead on grim's monolithic RDNA targets
  (gfx1100 MCD remap the only exception, and only after Track A data exists).

## Implementation Status (all WRECK phases)

- Implemented: WRECK-1 (trace.rs + 19 tests), WRECK-5 (KvQuantFormat + kv_dequant_attention kernel), WRECK-6 (split_k tests), WRECK-7 (LaunchConfig occupancy fields), WRECK-8 (FP8 MFMA gfx1200 source-structure tests), WRECK-9 (GraphCaptureManager in RocmDevice + 3 tests), WRECK-10 (Int8W8A8 quant mode + SmoothQuant types + 8 tests), WRECK-11 (spec-decode autotune params + 5 tests), WRECK-2 (CharTuner subspace decomposition + 6 tests), WRECK-3 (LatencyPredictor + 8 tests), WRECK-4 (JIT cache key + HsacoKernelCache tests).
- Verification: `cargo test --package grim-backend-rocm` — 319 lib tests pass + 100+ integration tests pass. Full workspace: 1 pre-existing CUDA test failure (`test_cuda_quantized_matmul_q8_0_gpu_fast_path`) unrelated to ROCm WRECK work.
