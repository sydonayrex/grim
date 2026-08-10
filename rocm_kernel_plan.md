# Charon — P-DAFD Kernel Experiment, Implementation Plan (ROCm/grim)

**Derived from:** `old/moeres/synthesis.md` §4 (P-DAFD: Predictive Distribution-Aware Fused Dispatch — the corpus-derived novel evolution beyond expert-parallel dispatch).
**Codename: Charon** — the ferryman of the underworld who rows souls (tokens) to their destination (expert) across the Styx (HBM). P-DAFD remains the long-form spec name; *Charon* is the public/codebase-facing moniker (flag `moe_charon`, prefix `charon::`).
**Date:** 2026-08-09
**Status:** Plan — no code landed yet
**Target backend:** `grim-backend-rocm` (primary GPU target, hipRTC JIT, Wave64/MFMA, RDNA gfx1036/gfx1200)

## 0. Why this experiment

The 39-paper corpus shows three independent, individually-proven legs that no published system
combines:

1. **Fuse** — sortless single-pass fused dispatch GEMM (TritonMoE 2605.23911, FlashMoE 2506.04667,
   UniEP 2604.19241): ~35% GMEM-traffic cut, 89–131% of MegaBlocks throughput on batch ≤512.
2. **Adapt** — polymorphic kernel population + runtime selection from the live routing histogram
   with *no CPU–GPU sync*. RaMP 2604.26039 and DA-MoE 2607.23099 report large wins (RaMP: 1.22×
   kernel / 1.30× e2e, 0.93% regret; DA-MoE: 1.16–1.56×) — **all measured on NVIDIA
   (Ada/Hopper, CuTe/CUTLASS/vLLM) against their own polished kernels.** These are upper-bound
   evidence *for the mechanism*, not expected RDNA outcomes. This plan reimplements the cost model
   and measures its win against grim's own WI-A baseline — never a cited figure (see §5 risks).
3. **Predict** — router-distilled lookahead predictor driving resident-set / mixed-precision plan
   (PROBE 2602.00509: 1.32× prefill / 1.26× decode; MxMoE 2505.05799: 3.4×, mixed-precision
   Group-GEMM; DynaExq 2511.15015).

**P-DAFD = all three, one kernel.** Dispatch decision = fused-kernel variant + per-expert tile
precision + resident/prefetch set, all driven by a layer-ahead routing prediction, GPU-resident.
Codename: **Charon** — one fused crossing carries every routed token to its expert.

**Why ROCm here:** grim's ROCm backend already has hipRTC JIT (`device/helpers.rs::jit_compile_hsaco`),
an on-disk hsaco cache (`kernels/jit_cache.rs::HsacoKernelCache`), a slot autotuner
(`autotune.rs::KernelKey`/`AutotuneConfig`), and a tile table (`device/gemm_tuning.rs::GemmTileConfig`).
There is **no grouped/expert GEMM path at all today** (`grep` of `kernels/` for grp/expert → empty),
and `grim-nn/src/moe.rs` is explicitly a *correct-but-unoptimized CPU reference* (WI-M5 fused/grouped
GPU kernel is a registered non-blocking perf item). So this experiment fills a real, pre-scoped gap.

**Non-goals / scoping (KISS):**
- NOT a training kernel (UniEP territory) — prefill+decode inference only.
- NOT an all-to-all / EP communication redesign (NCCL EP 2603.13606, UCCL 2512.19849 are out).
- Single-GPU, single-process; multi-GPU dispatch is out of scope for v1.
- Vendor BLAS still owns *dense* GEMM (Rule 0 of `rocm-hip-kernels`); the custom kernel is the
  *fused dispatch* path, not a rocBLAS clone.
- fp8 is gfx1200-only — mixed-precision stage must gate on `gcnArchName`.

---

## 1. Architecture of the experiment (what P-DAFD looks like in grim)

```
MoeFfn (grim-nn)                     P-DAFD GPU path (grim-backend-rocm)
  | route() -> topk/weights            |
  |   (today: host-side)               v
  |                              LookaheadPredictor (router distiller)
  |                                   | predicts next layer's routing dist
  |                                   v
  |                              PlanBuilder (budget-feasible, per-expert)
  |                                   | kernel variant idx, per-expert prec,
  |                                   | resident/prefetch list
  |                                   v
  |                              FusedDispatchKernel (hipRTC, sortless)
  |                                   | gate+up fused, in-KLU SiLU, down,
  |                                   | grouped w/ per-block expert assignment
  |                                   v
  |                              Combine (in-kernel, weights applied) → out
  v
  CPU parity baseline (moe.rs forward)
```

Three sub-experiments, each independently verifiable, ordered by increasing novelty/risk:

- **WI-A (Prove Fuse).** Sortless fused dispatch GEMM kernel, fixed precision. Gate = parity vs CPU
  reference + GMEM-traffic reduction vs per-expert rocBLAS calls. No prediction/selection yet.
- **WI-B (Prove Adapt).** A small polymorphic population (2–3 kernel variants) + GPU-resident
  selector keyed on a live histogram — the DA-MoE/RaMP cost model (form borrowed; model re-fit on
  RDNA). Gate = ≤5% regret vs *grim's own* offline-best variant on synthetic routing distributions
  (Dirichlet, mirroring 2607.23099 §III) — not RaMP's 0.93%.
- **WI-C (Prove Predict).** Router-distilled lookahead predictor → resident-set + mixed-precision plan
  (MxMoE/DynaExq flavor). Gate = prediction Hit@k vs *actual* next-layer routing (G-C2) AND the
  feature must beat its own off-switch on pre-registered thresholds (G-C3) — output parity (G-C4) is
  necessary but never sufficient for the prediction claim.

Wire all three behind **one feature flag** (`moe_charon`), CPU reference unchanged as default.

---

## 2. Work items (convention: Why / Where / What-exists / What-to-build / Limits / Gates)

### WI-A — Sortless fused dispatch kernel, fixed precision (gfx1036-first)

**Why:** Establish the fused baseline + parity oracle before adding any selection/prediction. This is
the TritonMoE/FlashMoE result translated to hipRTC.

**Where:** `crates/grim-backend-rocm/src/kernels/charon.rs` (new; register in `kernels/mod.rs` as the
`charon` module), launcher lifted from the `fused_dequant_gemm.rs` pattern (`KERNEL_SOURCE: &str`
const + hipRTC load, blockDim multiple of 64 / Wave64).

**What-exists:** `jit_compile_hsaco(source, entry, arch)` + `HsacoKernelCache` keyed
`(name, seahash(source))`; `autotune.rs` slot registry; `gemm_tuning.rs` tile tables; quantized
dequant GEMM kernels (`q*k_gemm.rs`, `fused_dequant_gemm.rs`) to reuse block-dequant device fns
(`shared_device_fns.rs`). CPU reference forward = `grim_nn::moe::MoeFfn::forward`.

**What-to-build:**
1. Device kernel `charon_fused_dispatch` (long form: `grim_moe_fused_dispatch`) — fused `gate` + `up` GEMM with in-register `SiLU` combine
   (TritonMoE result), grouped per expert via **sortless block-to-expert assignment** (block-idx-driven,
   no host sort, no per-expert kernel launch). Single `extern "C"` entry taking: activation ptr,
   per-expert `gate/up/down` weight ptrs, router indices (device uint), weights (device f32),
   expert offsets, dims, strides. Wave64 tile sizes from `lookup_gemm_config`.
2. Host `CharonLauncher` (a `pub(crate) fn`, per grim-codebase-fixes testability rule) that:
   assembles the one kernel launch + returns device output.
3. Counter harness counting GMEM bytes touched (device side) to prove the traffic reduction claim.

**Limits:** Fixed precision (f16/bf16 first, f32 fallback). Shared expert handled as a separate
dense path for v1 (not fused). No prediction/selection — pure fused GEMM.

**Gates (ordered):**
- G-A1 compile: `cargo build -p grim-backend-rocm` clean.
- G-A2 host logic unit tests (no GPU): `cargo test -p grim-backend-rocm --lib kernels::charon::tests`
  — launcher builds correct parameter blob, dims, offsets, indices marshalling.
- G-A3 **oracle-integrity precondition (no GPU):** before any GPU parity diff, the reference must
  prove itself self-consistent — `cargo test -p grim-nn --lib moe` passes, with
  `routed_scaling_factor_scales_routed_not_shared` (`moe.rs:575`) explicitly named in the run. This
  guards against a fused kernel "confirming" a *regressed* reference (the oracle and kernel
  agreeing again after sharing the same bug). If the named test fails, stop: the oracle is broken,
  the GPU diff is meaningless, and the reference must be fixed first.
- G-A4 **parity (GPU, unverified in this env — flagged):** `CharonLauncher` output ==
  `MoeFfn::forward` output on synthetic router distributions, max-abs-err ≤ 1e-2 f16 / 3e-3 bf16,
  run *only after* G-A3 is green. GPU parity alone is not evidence of correctness.
- G-A5 perf (GPU): fused kernel bytes-moved ≤ 70% of per-expert rocBLAS baseline at batch ≤512 —
  TritonMoE's ~35% reduction, *measured*, before claiming success. If G-A4/A5 cannot run here,
  they are recorded as device-verify TODO, not skipped silently (honest-verification rule).

### WI-B — Polymorphic population + GPU-resident variant selector

**Why:** The corpus's biggest *on-top-of-fusing* win is runtime kernel selection (RaMP/DA-MoE).
Without it, P-DAFD is just "a faster fused gemm". **Honesty caveat:** RaMP's 0.93% regret /
1.22×–1.30× and DA-MoE's 1.16–1.56× were measured on NVIDIA (Ada/Hopper, CuTe/CUTLASS/vLLM) on
their own kernels. We borrow the *form* of RaMP's 4-param wave cost model, not its constants or its
numbers; the model must be re-fit on RDNA and its success measured against grim's own baselines
(locally-tuned offline argmin for regret, WI-A for perf). No cited figure is a target here.

**Where:** `crates/grim-backend-rocm/src/kernels/charon.rs` (extend) +
`crates/grim-backend-rocm/src/device/gemm_tuning.rs` (add variant table) or a new
`autotune::CharonKey`.

**What-exists:** `KernelKey {kernel, gpu_arch, m, n, k}` + `AutotuneConfig {block_dim, tile_kv}` —
exactly the slot shape for `(variant, arch, M, per-expert-shape)`.

**What-to-build:**
1. 2–3 kernel variants: (a) small-batch/decode tile, (b) large-group prefill tile, (c) high-skew
   (few experts, many tokens) tile — each a separate spawnable variant of the WI-A kernel (RaMP-style
   "polymorphic megakernel" population, ~130 configs → collapse to the ~3 that matter for RDNA).
2. A **4-param wave cost model** (form borrowed from RaMP 2604.26039): predict cycles from
   `(nlok, nrb, flops_stalled, waittime)` — device-side, no host readback per dispatch. **The four
   coefficients are ours to fit on RDNA; nothing is transliterated from the paper, and no RaMP
   number is a target** (RaMP validated only NVIDIA Ada/Hopper).
3. GPU-resident selector: matching the **live routing histogram** to offline-tuned distribution
   buckets (DA-MoE 2607.23099) and emitting `variant_idx` into the next launch, without a CPU
   round-trip. Use `grim_softmax`-style existing primitive for histogram normalization.

**Limits:** 3 variants max; selector generalizes over synthetic Dirichlet distributions only in v1
(no per-model offline tuning DB yet). No mixed precision here.

**Gates:**
- G-B1 compile + host unit tests for the cost model (log-parity: model monotonic in each param).
- G-B2 (GPU) synthetic-Distribution regret: selector picks ≤5% worse than *grim's own offline
  argmin over the local variant table* for that distribution — the ≤5% bar and the referenced argmin
  are our own, not RaMP's 0.93% (their on-NVIDIA number); regret is always measured against a
  locally-tuned table, then locked.
- G-B3 (GPU) no CPU–GPU sync in the select→launch path (assert zero `hipMemcpy` D2H per dispatch).

### WI-C — Router-distilled lookahead predictor + mixed-precision resident set

**Why:** The predict leg (PROBE/MxMoE/DynaExq). This is the genuinely novel composition — no published
system fuses dispatch *and* predicts *and* varies per-expert precision.

**Where:** `crates/grim-nn/src/moe.rs` (predictor host component, keep unit-testable without GPU) +
`crates/grim-backend-rocm/src/kernels/charon.rs` (mixed-precision variant, fp16/bf16 gated + int8
via existing `q*k_gemm` dequant) + `crates/grim-memory` (resident-set budget).

**What-exists:** `MoeRouter` (SoftmaxTopK / SigmoidTopKWithBias) — route logits already computed on
host; positional target for distillation. `grim-quant` quantization. `grim-memory/budget.rs`.

**What-to-build:**
1. **Gate-Initialized Lookahead Predictor** (PROBE 2602.00509): a tiny distilled copy of the router
   (`MoeRouter::gate` linear, low-rank) that forecasts *next layer's* activated-expert distribution
   from current gate logits. Runs host-side; output = predicted histogram + per-expert hotness vector.
2. **PlanBuilder** (DynaExq-budget-feasible top-n): under HBM envelope, keep hot experts high-prec
   resident (fp16), cold experts low-prec/int8 fallback; async promote/demote via stable expert
   handles — v1: only the *prefetch list + precision mask* feeding WI-B selector and a mixed-precision
   kernel variant.
3. **SRP/SCH confidence gate** (2505.16056): compute the model's local-routing-consistency once at
   load; if below threshold → **disable prediction**, fall back to WI-B reactive matching only.
   (This is the honesty valve: don't *claim* prediction works on models it measurably can't.)

**Limits:** Predictor is host-side (GPU-resident distillation is follow-up). No dynamic replication
(PROBE's balance-solver is out for v1). int8 mixed path reuses existing dequant rather than new
codepath.

**Gates:**
- G-C1 (no-GPU) predictor unit tests: router-distillation PPL/Hit@k on synthetic gates; PlanBuilder
  respects the HBM budget function; SRP/SCH gate toggles prediction off when consistency is low.
- **G-C2 (no-GPU) prediction-accuracy, measured against ground truth, not parity:** evaluate the
  predictor directly on what it *does* — forecast the next layer's activated-expert distribution —
  and score it against the *actual* next-layer routing on held-out router traces (the only
  ground truth that exists, since the mechanism is novel and has no reference impl). Required bar:
  predicted top-k set must match realized top-k at Hit@k ≥ 0.80 (PROBE reports ≈90%; we claim less,
  on hardware we don't pre-validate). This is the gate that makes "a predictor wrong in an
  interesting way" fail: if Hit@k is low, no amount of loose output parity rescues it, because the
  kernel's correctness is *not* evidence the prediction works.
- **G-C3 (no-GPU) prediction-utility (falsifiable):** the *feature* must beat its own off-switch —
  PlanBuilder-with-prediction must demonstrably reduce resident-set misses / improve budget-kept
  quality vs the reactive WI-B fallback on the same traces. If prediction adds nothing over plain
  reactive matching (no measurable Δ in hit-rate or quality-under-budget), WI-C is **not** a pass:
  the honest outcome is "prediction adds no signal on this arch" recorded in the evidence doc,
  not "≈acceptable". Fixed thresholds, chosen before measuring: Hit@k Δ ≥ +0.05 and/or
  budget-kept PPL Δ ≥ 0.05 vs reactive baseline. This kills the G-C3-unfalsifiability: it is a
  pre-registered go/no-go against the off-switch, not "≥1.2× or a documented smaller-but-real gain".
- G-C4 (GPU) parity: mixed-precision P-DAFD output within DynaExq-style tolerance vs full-fp16
  reference on resident==full set. (Necessary, but — see G-C2/G-C3 — output parity alone never
  counts as evidence the *prediction* behaves.)

### WI-D — End-to-end flag + evidence packaging

**Why:** No dead code, honest verification, KISS. Follows `moe_probs.md` WI "5/6" convention: feature
flag defaulting to CPU reference; `Err(Unimplemented)` not silent dense fallback.

**Where:** `crates/grim-models/transformer` (Qwen3MoE/Laguna arms load path), `crates/grim-engine`
config, `crates/grim-cli`.

**What-to-build:**
1. `--moe-backend charon` / `moe_charon` cfg that routes `MoeFfn` forward → `CharonLauncher`.
2. Wire per-arch: at minimum Qwen3MoE (softmax) + DeepSeek-Laguna arm (sigmoid+bias, shared expert)
   — Laguna is the structurally-demanding acceptance arch per `moe_probs.md`.
3. Evidence doc `old/moeres/experiment_results.md` — copy of this plan's gates with per-gate
   PASS/FAIL/UNVERIFIED + device notes, so a no-GPU agent session never claims green falsely.
   **WI-C gates C2/C3 are pre-registered thresholds, not ~"almost any outcome passes":** a
   prediction leg that passes output parity but fails Hit@k/utility is recorded as FAIL with the
   reason the mechanism doesn't pay — agreeing with a CPU reference was never the point.

**Gates:**
- G-D1 `cargo build -p grim-models-transformer` (with flag) clean; CPU path byte-identical without flag.
- G-D2 hostile-check: `Err(Unimplemented)` surfaces on partial wiring, never silent behavior change.
- G-D3 final: ≥30 papers / all 2024+ corpus intact + full evidence table in `experiment_results.md`.

---

## 3. Suggested execution order & effort

| Step | Work | Effort | GPU needed? |
|---|---|---|---|
| 1 | WI-A: fused sortless kernel, host logic + tests | 1–2 sessions | only for A4/A5 (A3 oracle check runs on CPU) |
| 2 | WI-A device verify (on a gfx1036/gfx1200 box) | 1 session | yes |
| 3 | WI-B: 3 variants + wave model + selector | 1–2 sessions | B2/B3 need device |
| 4 | WI-C: predictor + PlanBuilder + SRP/SCH gate (+ C2/C3 pre-gates) | 2 sessions | C4 needs device |
| 5 | WI-D: flag wiring + Qwen3/Laguna + evidence doc | 1 session | G-D2 no; D3 no |

All host-logic tests run in the current no-GPU environment (`cargo test -p grim-backend-rocm --lib`,
**never** `cargo test --workspace` — `grim-backend-vulkan` hangs on headless). Device-gated gates are
recorded as explicit UNVERIFIED-in-sandbox TODOs, never dropped.

## 4. Verification discipline (grim-codebase-fixes rules)

- Extract launcher/planner logic into `pub(crate) fn` + `#[cfg(test)]` so host logic is provable
  without a device (A2, B1, C1, D2). **Oracle-integrity first:** the CPU reference (`MoeFfn::forward`)
  is the parity oracle for G-A4, and it must pass its own suite (incl.
  `routed_scaling_factor_scales_routed_not_shared`, `moe.rs:575`) before any GPU diff — else a
  regressed reference and a regressed kernel can silently agree (A3 guards this).
- Per-crate verification only: `grim-backend-rocm`, `grim-nn`, `grim-models-transformer`,
  `grim-memory`. Default workspace cargo commands are forbidden here.
- fp8/MFMA paths gated on `gcnArchName >= gfx1200` (RDNA4 only), never on type availability.
- "Done" = every gate PASS or an explicit UNVERIFIED with a device-verify TODO + reason. Never claim
  green where evidence is missing.
- **Cited numbers are never acceptance criteria.** Every perf/regret/accuracy threshold in §2 is
  measured against a locally-computed baseline (grim's own offline argmin, the off-switch, held-out
  ground-truth routing), with the bar fixed *before* measuring — corpus figures are motivation and
  calibration priors only.

## 5. Risks

- **No GPU in sandbox** → A4/A5/B2/B3/C4 can't be executed here. Mitigation: parity harness is
  written and gated-ready; evidence doc marks device TODOs. (A3 oracle check, and all of WI-C's
  mechanism gates C1/C2/C3, run on CPU — the falsifiable core of WI-C does *not* require hardware.)

- **Unverified citation-forwarded as claim** (the Q5_K "219/256 wrong" / routed_scaling_factor
  doc-vs-code class): RaMP's 0.93%/1.22×/1.30×, PROBE's ≈90% and 1.2×, TritonMoE's ~35% are all
  measured on their authors' hardware and kernels. **No corpus number is a target or a gate
  threshold here** — every perf/regret/accuracy bar is defined against grim's own baselines
  (local argmin, off-switch, held-out traces) and either measured locally or explicitly
  UNVERIFIED. The plan's §0/§2 language has been corrected so cited numbers read as *upper-bound
  motivation*, never as expected outcome.

- **WI-C unfalsifiability** (a predictor wrong in an interesting way passing loose parity): G-C2
  (Hit@k vs ground-truth next-layer routing) and G-C3 (must beat its own off-switch on
  pre-registered thresholds, else recorded as FAIL "prediction adds no signal") close this. Output
  parity (G-C4) is explicitly stripped of any power to certify the prediction mechanism.
- **Custom GEMM perf risk** (Rule 0): fused dispatch is legitimate custom-kernel work, but per-expert
  dense GEMM must *still* be rocBLAS wherever the fuse doesn't pay. The fused path is only selected
  under the flag; rocBLAS path remains.
- **Selection hysteresis** (DA-MoE caution): histogram→variant matching must include a de-sync guard
  (don't thrash variants between adjacent layers). Put min-hold-count into the selector table.
- **Prediction overclaim** (2505.16056): without the SRP/SCH gate, predictor gains are model-dependent.
  The gate is mandatory, not optional.

## 6. Corpus traceability

Each work item maps to its evidence:
- WI-A ← TritonMoE 2605.23911, FlashMoE 2506.04667, UniEP 2604.19241, SonicMoE 2512.14080
- WI-B ← RaMP 2604.26039, DA-MoE/Decoding the Skew 2607.23099, memory-bound 2512.09277, Samoyeds 2503.10725
- WI-C ← PROBE 2602.00509, MxMoE 2505.05799, DynaExq 2511.15015, FineMoE 2502.05370, SRP/SCH 2505.16056, ALF-LB 2512.03915
- WD+context ← DeepSeek-V3 2412.19437, MegaScale 2504.02263, FinDEP 2512.21487, Tied experts 2606.16825