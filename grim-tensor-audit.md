# grim-tensor + grim-tensor-graph Audit Report

Scope: crates/grim-tensor (backend, dtype, softmax_merge, provider, shape,
tensor, wavefront, error + 4 integration test files) and
crates/grim-tensor-graph (lib, ir) as of this audit.

Classes: L = logic fault (silent wrong result), B = latent bug (hits
only under specific conditions), G = gap (missing capability / untested
pathway), P = perf/correctness footgun, M = maturity/positioning.

Test status at audit time: 43/43 pass (29 unit + 4 golden/stress files
in grim-tensor; 8 unit in grim-tensor-graph).

---

## L1. shard_raw_tensor silently drops rows/columns on non-divisible shapes
   grim-tensor/src/provider.rs:146-151, 161-169

   `shard_rows = rows / world_size` truncates: a [5, 8] weight sharded
   across 2 ranks yields two [2, 8] shards — row 4 vanishes from every
   rank with no error. The crate even ships the exact check needed
   (`shard_boundary_valid`, line 84: divisibility AND block alignment)
   and `shard_raw_tensor` never calls it. Tensor-parallel loading of a
   checkpoint whose dims don't divide by world_size silently corrupts
   every weight matrix — the worst failure mode in a loader.

## L2. shard_raw_tensor mis-slices tensors of rank > 2
   grim-tensor/src/provider.rs:134-142

   Rejects rank < 2 but accepts rank > 2, then treats `shape[0]`/
   `shape[1]` as rows/cols while the byte buffer is the product of ALL
   dims. For a 3-D tensor the computed `row_stride = cols * elem_size`
   is wrong and both shard paths return confidently wrong bytes. Should
   be `ndim != 2 → Err`.

## B1. Arc blanket impl fails to forward 11 overridable trait methods
   grim-tensor/src/backend.rs:1530-2100

   `impl<T: BackendDevice + ?Sized> BackendDevice for Arc<T>` is ~580
   lines of manual forwarding — and it omits: `sample_on_device`,
   `qkv_attention_alibi`, `rerope`, `fused_add_rms_norm`,
   `fused_mxfp4_gemm_qk_norm_rope_kv`, `broadcast_bias`,
   `scale_bias_epilogue`, `short_conv1d_causal_step`,
   `kda_gated_delta_rule_step`, `mla_q_kv_norm_split`,
   `mla_absorbed_decode`. ROCm overrides nearly all of them. Verified
   latent: every live call site dispatches through `Arc<dyn
   BackendDevice>` (grim-nn `pick_device_for_*`) or `Box::new(concrete)`
   (engine tests, grim-cli train, grim-autograd), never through
   `Box::new(arc_of_concrete)` — but the impl's own doc advertises
   exactly that pattern ("hand out cheap Arc clones through the
   `Box<dyn BackendDevice>` API"). The moment anyone uses it as
   documented, backend overrides silently fall to the trait defaults
   (mostly `Err(Unimplemented)` — or worse, `sage_attention`'s silent
   f32 fallback). Fix: forward the 11, plus a debug assertion or
   compile-time test that the forwarding list is exhaustive (e.g. a
   test that calls every default-overridable method through
   `Arc<RocmDevice>`-shaped handle and asserts the override ran).

## B2. merge_partials head-dim mismatch is debug_assert-only
   grim-tensor/src/softmax_merge.rs:80-86

   In release builds a (head_dim_a != head_dim_b) merge silently
   `zip`-truncates to the shorter accumulator and returns a wrong-
   shaped partial. This module is the shared math for three attention
   paths; it should return `Result` or at least `assert!` unconditionally.

## B3. padded_dims assumes power-of-two wavefront size
   grim-tensor/src/wavefront.rs:36-41

   `(rows + wf - 1) & !(wf - 1)` silently mis-pads for any non-power-of-
   two `wavefront_size` (e.g. 48). No debug_assert on the precondition.

## B4. div_scalar default is mul_scalar(1/scalar)
   grim-tensor/src/backend.rs:249-256

   Two numeric faults: `div_scalar(x, 0.0)` computes `x * inf` instead
   of erroring, and `x / s` differs from `x * (1/s)` by up to 1 ulp for
   non-power-of-two s — so backends that don't override this produce
   different numerics than backends that do, for the same call. A
   silent per-backend numeric divergence in an otherwise
   backend-agnostic contract.

## B5. lora_accumulate default: no shape validation, per-call host round-trips
   grim-tensor/src/backend.rs:1324-1424

   `rank` vs `rank_b` and `x`'s `in_features` vs `a`'s `in_features_a`
   are never checked — mismatched LoRA A/B ranks flow straight into
   matmul (garbage or an opaque backend error). Also: Aᵀ/Bᵀ transposes
   round-trip through host Vecs on EVERY call, and the scale is applied
   by uploading a full `out_shape.elem_count()` broadcast buffer per
   call — a per-token allocation on the decode path wherever this
   default is used.

## P1. sample_on_device numeric edges
   grim-tensor/src/backend.rs:260-331

   - Greedy path uses `.unwrap()` (defended by the preceding
     is_empty check — safe but style-inconsistent with the audit bar).
   - All-(-inf) logits: `max_logit = -inf` → `l - max = NaN` → NaN
     probabilities → returns index 0 silently instead of an error.
   - top-p is applied AFTER top-k truncation on the renormalized
     subset (an intentional order, but undocumented in the contract).
   - Single splitmix64 draw from `seed`: deterministic (good) but every
     step with the same seed picks the same quantile — callers needing
     per-step variation must seed differently each call (contract not
     documented here).

## P2. sage_attention default silently degrades to plain f32 attention
   grim-tensor/src/backend.rs:353-376

   The name promises INT8/FP8 block-scaled attention; the default is a
   plain `qkv_attention` passthrough with no signal to the caller that
   quantization was skipped. Fine as a fallback — but it should be
   visible (return an error, or a documented quality flag), because a
   caller benchmarking "SageAttention" on an unmodified backend measures
   the wrong thing.

## P3. BackendDevice is a ~60-method god-trait with ~40 Unimplemented defaults
   grim-tensor/src/backend.rs:164-1517
   [RESOLVED — implemented as capability sub-traits, see "P3 executed" below]

   Required-method set is small (zeros, matmul, add, mul, silu_mul,
   rms_norm, softmax, embedding, from_cpu, advise) but the default-
   method surface spans attention variants, optimizers, quantization,
   collectives, recurrent/SSM kernels, MLA, graph capture, and latency
   prediction. Consequences already visible: the Arc forwarding rotted
   (B1); every new backend must audit 60 methods to know what it owes;
   `mul_scalar`-family defaults silently change `div_scalar` numerics
   (B4). Natural decomposition: AttentionOps, OptimizerOps, QuantOps,
   CollectiveOps, RecurrentOps sub-traits with blanket fallback impls —
   or at minimum a generated exhaustive-forwarding test.

## P4. QuantizedMatmulBackwardResiduals carries raw device pointers
   grim-tensor/src/backend.rs:2102-2181

   `unsafe Send + Sync` on raw GPU pointers, with an unusually honest
   SAFETY comment that documents the protecting invariant ("all live
   call paths pass through AppState.engine: Mutex<Engine>… do NOT
   remove that lock"). Correct today, but the invariant lives in a
   comment in a different crate; nothing enforces it. A typed guard or
   a debug-only poison check would make the contract load-bearing.

## M1. Two competing error taxonomies — and grim-tensor's doc is false
   grim-tensor/src/error.rs:2-3 vs grim-core/src/error.rs

   grim-tensor's error module claims "Every crate in the workspace
   ultimately returns `grim_tensor::Result<T>`". Reality: 200 files use
   `grim_core::error::Error` (with its own KvCache/Backend/etc.
   variants); ~4 files use grim_tensor::Error. The workspace has two
   Result types and every boundary converts (or stringifies) between
   them. Either grim-tensor's error should be THE error (it is
   dependency-free enough) or the doc should stop claiming it.

## G-gaps (features the crate's function implies, that exist in no other crate)

   G1. No shape algebra: no reshape/transpose/permute/squeeze/broadcast
       on `Shape` or `Tensor` — reshape is pure metadata and still
       missing. Every consumer (grim-nn modules, engine, autograd)
       hand-rolls vec copies for layout changes.
   G2. No device-side reductions or `sub`: the trait has add/mul but no
       sub, sum, max, or argmax — `sample_on_device`'s argmax runs on
       the host after `to_cpu_vec_f32`, and every "top-k on device"
       idea dead-ends here.
   G3. No packed-size calculator: `Storage` byte layouts are documented
       prose (excellent docs!) but there is no
       `expected_bytes(shape, &DType) -> usize` to validate
       `RawTensor.bytes` length at load — layout violations surface as
       garbage deep inside kernels instead of a load-time error.
   G4. QuantFormat ↔ Storage scheme duplication: `QuantFormat`'s 16
       variants nearly mirror `KQuantScheme` + `FloatPackScheme` +
       `BlockDtype` with no `From` conversions — drift risk every time
       a format is added.
   G5. `TensorProvider::tensor_names` default returns empty (making
       prefetch silently a no-op) — reasonable, but there is no test
       that a provider forgetting the override still loads correctly.

## C1. Strengths (notable good bits)
   - softmax_merge.rs is the best-tested numerics module in the audit
     series: identity/commutativity/associativity properties,
     hand-derived golden triples explicitly designed to kill
     scale-rescaling mutants, and an end-to-end split-KV ≡ direct
     softmax property test.
   - The golden RoPE / re-RoPE / LoRA tests use hand-computed expected
     values (mutation-resistant), and the stress test covers 128K
     position jumps, zero-delta identity, and reverse retargeting with
     finiteness assertions.
   - dtype.rs byte-layout contracts (MXFP4/W4A16/AWQ/ResidualPacked/
     GroupInt) are documented to kernel-consumer precision — rare and
     valuable.
   - The ArithType/Storage split is a genuinely good design that keeps
     new low-bit formats from forking dispatch.
   - RopeConfig.interleaved carries a bisected real-world bug lesson in
     its doc; `is_plain()` lets legacy backends refuse non-plain configs
     instead of silently corrupting.
   - Honest safety documentation (Tier 1/2/3 taxonomy; the
     QuantizedMatmulBackwardResiduals SAFETY comment names the actual
     protecting lock).

---

## grim-tensor-graph

   Verdict: a decorative stub that has survived as an integration
   point. 327 lines, two parallel IRs, no dataflow.

   B6. Fusion detection pairs tensors by accident of naming order
       (lib.rs:57-109). `find_first` takes the FIRST name containing
       ANY needle, so `detect_rmsnorm_matmul` pairs the first-seen norm
       with the first-seen q projection independently — for a
       multi-layer checkpoint both are always layer 0's, and a naming
       scheme where needle order differs from layer order pairs tensors
       from different layers. The consumer (grim-cli oxidizer.rs:484)
       only uses op-level `recommended_fusion_ops()`, which limits the
       blast radius to "detection misses when naming conventions
       differ" — but the `FusionGroup.tensors` field, the crate's only
       tensor-level output, is untrustworthy.

   M2. `ComputationGraph` (ir.rs) is a sequence matcher, not a graph:
       no edges (`GraphNode.input_tensors` is never populated — `push`
       hardcodes `Vec::new()` and no API sets it), fusion detection is
       by push ORDER, and `fusion_candidates` are consumed by nothing
       in the workspace. `TensorGraphIr` (lib.rs) is a substring
       matcher. Two IRs, one crate, zero shared code, zero
       cross-references. The backends' actually-fused pairs (silu_mul,
       add+rmsnorm, scale_bias_epilogue) are not detectable at all —
       only the two patterns named in GrimFusionOp.

   M3. Hard-coded target strings ("fused_rmsnorm_matmul_rocm") duplicate
       naming that should come from grim-format; `OpType` (5 variants)
       cannot express the graphs the engine actually runs; no serde /
       persistence; no test with a real multi-layer tensor-name list.

   Missing features relative to its stated function ("checkpoint-derived
   tensor graph IR and fusion-pattern detection"): dataflow edges, shape
   inference (`GraphNode.shape`/`dtype` fields exist but no API sets
   them), topological ordering, a path from `FusionSequence` to actual
   dispatch, and coverage of the fusion pairs the backends really
   implement. Either build those out or fold `TensorGraphIr` into
   grim-format beside `GrimFusionOp` and delete `ComputationGraph` —
   the current state is the worst of both: too big to be free, too
   small to be real.

---

## Priority

CRITICAL: L1 — silent sharding data loss (the check one function over
          exists and is unused).
HIGH:     B1 (Arc forwarding rot — latent but documented-pattern
          triggered), L2, B2.
MEDIUM:   B3, B4, B5, B6, M1, M2, G1-G3.
LOW:      P1-P4, M3, G4, G5.
INFO:     C1 strengths; mutation-testing (mutants.toml) covers only
          grim-quant — grim-tensor hosts the workspace's shared
          attention numerics and would be the natural next candidate.

---

## Fixes applied (post-audit)

- L1: `shard_raw_tensor` now errors when rows or cols are not divisible
  by `world_size` (was: truncate and silently drop the tail from every
  rank). Tests: `sharded_non_divisible_dims_error`.
- L2: `shard_raw_tensor` requires `ndim == 2` (was: accepted rank > 2
  and sliced with wrong byte offsets). Test: `sharded_rank3_tensor_errors`.
- B1: the `Arc<T>` blanket impl now forwards all 11 previously-missing
  methods (`sample_on_device`, `qkv_attention_alibi`, `rerope`,
  `fused_add_rms_norm`, `fused_mxfp4_gemm_qk_norm_rope_kv`,
  `broadcast_bias`, `scale_bias_epilogue`, `short_conv1d_causal_step`,
  `kda_gated_delta_rule_step`, `mla_q_kv_norm_split`,
  `mla_absorbed_decode`). New `tests/arc_forwarding.rs` dispatches each
  through `Box<dyn BackendDevice>` built from `Arc<ProbeDevice>` and
  requires the override's marker error — a missing forward now fails CI
  instead of rotting silently.
- B2: `merge_partials` head-dim check is `assert_eq!` (was
  `debug_assert_eq!`; release builds would zip-truncate silently).
- B3: `padded_dims` carries a `debug_assert!(is_power_of_two)`
  precondition (const fn, so plain literal message).
- B4: `div_scalar` default errors on `scalar == 0.0` and documents the
  1-ulp decomposition caveat (backends needing exact division override).
- B5: `lora_accumulate` validates `rank == rank_b` and
  `in_features == in_features_a`, returning the actual shapes on
  mismatch instead of feeding matmul garbage.
- P1: `sample_on_device` greedy path no longer unwraps (guarded
  `max_by`), and a non-finite logit maximum is an error instead of NaN
  probabilities that silently return index 0.

Verification: grim-tensor 34 (31 unit + arc_forwarding probe + 2 golden
files) + stress + grim-tensor-graph 8 — all passing; grim-nn,
grim-format, grim-tensor-graph compile clean against the changed crate
(no signatures changed; only new error paths).

Not fixed (deliberate, ponytail scope): P2 (`sage_attention` silent f32
fallback — needs a cross-backend contract decision), P3 (trait
decomposition — large refactor), P4 (raw-pointer invariant enforcement),
G1-G5 gaps, and B5's per-call host round-trips (perf work, needs
backend-side transpose caching). grim-tensor-graph: unchanged — see
recommendation below.

---

## Fixes applied (second pass — the deferred items)

Following the post-fix recheck (which confirmed all deferred items were
still present), the fixable ones were addressed:

- P2: `sage_attention` default now prints a loud warning ("no native
  quantized-attention kernel … falling back to plain f32") and its doc
  states that benchmarks without an override measure f32 attention.
- P4: `outlier_indices_ptr`/`outlier_values_ptr` are now private behind
  an `unsafe fn set_outlier_pointers` (contract documented at the
  attachment site) plus read accessors; the ROCm reader sites updated.
  The raw pointers can no longer be poked freely from outside the crate.
- B5 (partial): `lora_accumulate` scale application now prefers
  `mul_scalar` (CPU/ROCm have kernels — no broadcast buffer upload) and
  falls back to the broadcast `mul` otherwise, with the ceiling and
  upgrade path commented. The Aᵀ/Bᵀ host transposes remain; caching
  them needs backend-owned state and stays deferred.
- G2: trait gained `sub` (default `Err(Unimplemented)` — no silent host
  round-trip), plus `reduce_sum` / `reduce_max` / `argmax` with
  documented host fallbacks (empty tensors error, ties resolve last-index
  per `max_by`). All four forwarded in the `Arc<T>` blanket impl and
  covered by the probe forwarding test.
- expected_bytes exactness: `W4A16` arm made exact (codes + per-group
  scales); the doc now states precisely which variants are exact and
  which return an upper bound. Golden tests added for
  Q80/Q4K/Fp8/MxFp4/W4A16.

New tests: `tests/reduction_defaults.rs` (host-fallback correctness,
tie semantics, empty-tensor errors), `test_expected_bytes_golden` in
dtype.rs, and 4 new probe assertions. Suite: 32 unit + 8 integration
files green; grim-nn, grim-engine, grim-backend-rocm, grim-server all
compile clean.

P3 (BackendDevice trait decomposition): IMPLEMENTED — see "P3 executed:
BackendDevice decomposition into capability sub-traits" below.
B5's backend-side transpose caching still deferred (needs backend-owned
state).

---

## P3 executed: BackendDevice decomposition into capability sub-traits

**Strategy.** `BackendDevice` was a 68-method god-trait (10 required, 58
defaulted) with a hand-maintained `Arc<T>` forwarding impl — the setup
that already produced B1 once. The constraint that shaped the design:
every call site in the workspace dispatches through `dyn BackendDevice`
(`Box<dyn>` / `Arc<dyn>`), so a decomposition must not split the dyn
object. Chosen design: **umbrella supertrait**. The 68 methods are
grouped into 12 dyn-compatible capability sub-traits, and
`BackendDevice` becomes an empty umbrella requiring all twelve plus
`Send + Sync`. Because dyn method lookup walks supertraits, every
existing `dev.method(...)` call through `dyn BackendDevice` compiles
unchanged — zero churn at ~200 call sites. Each capability stays
separately importable, so a backend or consumer can now depend on e.g.
`AttentionOps` alone, and the per-group semantics of the defaults are
documented per trait instead of buried in one 1,500-line impl.

Sub-traits (grim-tensor/src/backend.rs, re-exported from the crate
root): `CoreTensorOps` (12 methods; the 10 required + 2 convenience
defaults), `ElementwiseOps` (10, incl. scalar ops + reductions),
`SamplingOps` (1), `AttentionOps` (12, incl. rope/rerope/MLA),
`FusionOps` (5; supertraits CoreTensorOps + QuantOps because
`silu_mul_quantize` decomposes into them), `AutogradOps` (6; supertraits
Core + Elementwise because `lora_accumulate` decomposes), `OptimizerOps`
(3), `QuantOps` (4), `RecurrentOps` (5), `CollectiveOps` (3),
`MemoryOps` (3), `GraphCaptureOps` (4). Cross-trait default
decompositions are expressed as supertrait bounds — the compiler now
enforces what the old single trait only implied.

**Mechanical transformation.** All trait bodies were preserved
byte-for-byte; the split was done with a brace-depth parser (segment =
docs + one method, classified by an explicit 68-entry method→trait map,
validated to be a total bijection). The same script split each backend's
single `impl BackendDevice for X` into 12 per-sub-trait impl blocks:
CpuDevice (38 overrides), RocmDevice (52), CudaDevice (29),
VulkanDevice (36), MetalDevice (35), plus the audit's ProbeDevice (25).
No method body was edited — a backend override before the split is the
identical override after it.

**Arc forwarding.** The hand-written `impl BackendDevice for Arc<T>`
was replaced by twelve `impl<T: Sub + ?Sized> Sub for Arc<T>` blanket
impls that forward every method of the group explicitly, plus an empty
umbrella impl. Doing this immediately surfaced **live B1-class rot**:
`fused_adamw_step`, `fused_lion_step`, and `fused_madam_step` were NOT
forwarded in the old impl — post-B1-fix methods whose overrides had
already fallen through the documented `Arc` pattern. The generated impls
forward all 68, and the probe test (`tests/arc_forwarding.rs`) now
overrides all three optimizer steps and asserts their marker errors
dispatch through `Box<dyn BackendDevice>` built from `Arc<ProbeDevice>`.

**Call-site fallout (the only non-mechanical part).** Three patterns
needed touching, all semantics-preserving: (1) UFCS
`BackendDevice::method(...)` paths no longer resolve (the umbrella
declares no methods) — rewritten to the owning sub-trait path
(`CoreTensorOps::from_cpu(...)`, ~60 sites across grim-nn, grim-models,
grim-autograd, backends, tests); (2) method-syntax calls on *concrete*
device types need the sub-trait in scope (dyn callers are unaffected) —
imports added where required; (3) `use ... BackendDevice` imports that
became unused were dropped. Inherent-method callers are untouched.

**Verification (per-backend).** grim-tensor: 32 unit + 6 integration
files + doctests green, including the extended arc_forwarding probe.
grim-backend-cpu: 44 unit + 4 test files green. grim-backend-cuda: 32
unit + 3 files green (includes real CUDA runs). grim-backend-metal: 13
unit + 6 files green. grim-backend-vulkan: 18 unit + files green.
grim-backend-rocm: 365 lib unit tests green + full test-target compile
clean. grim-nn (68), grim-autograd (151), grim-format (125+), and the
whole workspace `cargo check` (lib + tests) clean.

**Residual notes.** Doc-comment references to `BackendDevice::method`
were left as-is (they resolve through the umbrella). If a future method
is added to a sub-trait, the compiler forces a decision its group makes
explicit (default or required), and the probe test fails if the `Arc`
forward is forgotten — the two failure modes that made P3 dangerous are
now structurally closed.

---

Original verification record (pre-fix):

All deferred items re-verified present: P2 (sage_attention default still
delegates to plain qkv_attention), P3 (trait now 64 methods, 43
Unimplemented defaults), P4 (raw pointers + "Do NOT remove that lock"
comment unchanged), B5 (lora_accumulate host round-trips unchanged), and
the device-side gaps (no `sub`/`sum`/`max`/`argmax`; sampling argmax
still host-side via `to_cpu_vec_f32`). Suite green (34).

G-gap update: G1 (reshape/transpose) and G3/G4 (expected_bytes,
QuantFormat↔Storage conversions) were closed by later concurrent edits —
`Shape::reshape`/`transpose` (both properly validated),
`DType::expected_bytes`, `From<QuantFormat> for Storage`,
`TryFrom<&Storage> for QuantFormat`. Caveat on `expected_bytes`: the
fallback arm returns `elem_count * arith.byte_size()` for
W4A16/GroupInt/AWQ/CompressedTensors — a large over-count for packed
formats — so it is an upper-bound check, not an exact size, for those
variants. Remaining open gaps: device-side sub/reductions (G2), the
`_`-arm exactness, and everything in the deferred P/B list above.

## Recommendation: grim-tensor-graph

**Recommendation: fold TensorGraphIr into grim-format next to
GrimFusionOp, delete ComputationGraph, keep the crate (renamed or
absorbed) only if a second consumer appears. Do not build it out now.**

Reasoning:

1. One real consumer, one function. The entire crate is exercised by
   grim-cli/src/oxidizer.rs:484 calling
   `build_transformer_ir(names).recommended_fusion_ops()`. That
   consumer needs: (tensor names) -> (set of fusion ops). It does not
   need a graph, nodes, edges, or sequences — and has no path from
   FusionSequence to backend dispatch.

2. The crate's ownership is wrong. Its only vocabulary type,
   `GrimFusionOp`, lives in grim-format/gguf.rs. Detection is a pure
   function of checkpoint tensor names — that is format-domain
   knowledge (naming conventions are the GGUF/safetensors layouts'
   concern). Co-locating the detector with GrimFusionOp removes the
   grim-tensor dependency entirely (grim-tensor-graph's only crate-level
   use of grim-tensor is Shape/ArithType in the unused ComputationGraph)
   and kills a whole crate from the build graph of anything linking
   grim-format.

3. ComputationGraph is worse than dead — it is a false affordance.
   GraphNode.input_tensors is never populated by any API, "fusion
   detection" matches push order, and its output is consumed by
   nothing. Dead code that looks like a framework invites someone to
   build on top of a matcher that has no dataflow semantics. Delete it;
   git keeps it.

4. The YAGNI test: when would a real graph IR be justified? When fusion
   decisions need shape/dtype inference or dataflow analysis the
   backends can't do at dispatch time. The backends currently fuse by
   hard-coded call sites (silu_mul, add+rmsnorm, scale_bias_epilogue)
   and the `.grim` fusion_mask is op-level. Nothing on the roadmap
   needs edge-level IR. If that changes — 50-line detector in
   grim-format is trivial to migrate into whatever real IR emerges,
   and its substring heuristics carry no capital.

5. Middle path rejected: hardening in place (fix find_first layer
   pairing, document limits) still leaves two IRs, a misnamed crate
   ("tensor-graph" with no graph), and a dependency edge that shouldn't
   exist. Fixing the wrong shape costs more than moving 100 lines.

Migration is mechanical: move `build_transformer_ir` +
`detect_rmsnorm_matmul` + `detect_qkv_attention` + `find_first` +
`TensorGraphIr`/`FusionGroup` into grim-format/gguf.rs (or a
grim-format/fusion.rs), repoint grim-cli's import, delete the crate,
drop it from workspace members. Effort: under an hour including test
moves; the 8 existing unit tests move verbatim.

## Migration executed

grim-tensor-graph is deleted. `FusionGroup`, `TensorGraphIr`, and
`build_transformer_ir` (+ detectors + tests) now live in
`grim-format/src/fusion.rs`, re-exported from the grim-format root.
grim-cli's oxidizer imports `grim_format::fusion::build_transformer_ir`
and its Cargo.toml drops the grim-tensor-graph dependency; the crate is
removed from workspace members and workspace.dependencies; the stale
grim-autograd doc reference is updated. The original detection test
moved verbatim; three added (missing wv ⇒ no QkvAttention, empty names,
dedup of recommended ops across layers). Verified: grim-format 125 unit
+ golden suites green; grim-cli `cargo check` clean; no references to
`grim_tensor_graph` remain (grep).

