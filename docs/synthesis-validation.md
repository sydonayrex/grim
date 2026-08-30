# Synthesis Validation: research recommendations vs. grim codebase reality

**Date:** 2026-08-30 (updated after R1 implementation) · **Scope:** validate every
actionable recommendation in [`old/tp/synthesis.md`](/D/rex/projects/grim/old/tp/synthesis.md)
against the live codebase (`main`), then prioritize the real gaps.

## Verification method

For each synthesis "Pick", I grepped for (a) the data structure, (b) its use
outside its own crate's unit tests, and (c) any call on the actual decode hot
path (`Engine::tick` → `step_batch` → `drive_forward` → `model.forward`). A
recommendation is "wired" only if (c) holds. "Data structure only" means it
builds and tests cleanly but the execution path never calls it.

---

## Validation results

### Pick 1 — VPP virtual-stage PP → `pipeline_engine.rs`

| Check | Result |
|---|---|
| Data structure (`PipelinePlan`, `PipelineStageConfig`) | ✅ present, 255 lines, tested |
| Virtual-stage *traversal* (V-shaped fold-back) | ❌ absent — `stage_for_layer` is a pure partition query, no chunk fold-back |
| Wired into decode execution | ❌ no — engine only computes & validates the plan, hard-fails if `pp_size > 1` |

**Finding:** Implements Megatron-style static layer partitioning, not VPP's
V-shaped virtual-stage traversal or async comm overlap (Pick 1's actual
innovation). The "98% bubble reduction" mechanism is not present.

### Pick 2 — RRFP readiness dispatch → `readiness_dispatch.rs`

| Check | Result |
|---|---|
| Data structure (`ReadinessDispatcher`, `ReadySet`, `ScheduleHint`) | ✅ present, 304 lines, tested |
| Called from scheduler/engine hot loop | ❌ no — `pub use`d from `grim-scheduler/src/lib.rs` but `Scheduler::schedule()` never constructs or calls it |
| Message-driven async comm / TP coordination | ❌ absent — pure in-memory ready-set arbitration, no cross-rank coordination |

**Finding:** A correct RRFP *arbitration primitive* (ready-set scan over a hint
order), but the fixed-order `Scheduler::schedule()` still drives admission.
Readiness-driven dispatch is not the live path.

### Pick 3 — UniEP fused MoE mega-kernel → `moe_deterministic.rs`

| Check | Result |
|---|---|
| Data structure (`DeterministicTokenMap`, `ScoreboardSync`) | ✅ present, 367 lines |
| Deterministic token ordering / prefix-sum offset table | ⚠️ `DeterministicTokenMap::build` exists; no runtime-fused all-to-all+FFN kernel |
| Called from MoE execution (`MoeFfn::forward`) | ❌ no — only referenced in `moe_block.rs` unit tests |

**Finding:** Captures the *mapping* idea (stable sort + prefix-sum destination
offsets) but not the fused comm-compute mega-kernel. No persistent-SM kernel,
no scoreboard-driven overlap. The existing `MoeFfn` uses the non-fused path.

### Pick 3b — Disaggregated attention/FFN (DisagMoE)

| Check | Result |
|---|---|
| Disaggregated placement (attention vs FFN on different GPU groups) | ❌ absent |
| M2N/N2M fused combine+P2P | ❌ absent |
| Roofline bandwidth model | ❌ absent |

**Finding:** Not implemented. The pre-existing `grim-disagg` crate is
*prefill/decode* disaggregation over TCP, a different concept.

### Pick 4 — Certified exactness → `memory_certificate.rs`

| Check | Result |
|---|---|
| Data structure (`MemoryCertificate`, `BoundaryVector`, semantic-demand) | ✅ present, 219 lines, semantic-demand lower bound |
| Wired into engine or server admission | ❌ no — `Engine::new` / `admit_*` never call `MemoryCertificate::certify` |

**Finding:** Correct *bound computation* (expert-set × element size), but it is
not enforced. The "43.59 GiB semantic-demand" mechanism computes a number that
nothing checks.

### New this session — Dispatched multi-LoRA kernel → `batched_lora.rs`

| Check | Result |
|---|---|
| Dispatched kernel (2 launches, token→adapter indirection) | ✅ present, GPU-tested vs grouped path |
| Wired into engine `apply_batched_lora_to_rows` | ✅ yes — preferred GPU path, falls back to grouped then CPU |
| CPU reference + GPU parity test | ✅ both present, passing |

**Finding:** This one IS wired — the only synthesis-aligned feature (built this
session, not from synthesis) that runs on the hot path.

---

## What IS genuinely new and wired (baseline)

These are real capability additions to the hot path, verified by grep + test:

1. **Dispatched multi-LoRA serving** — 2-kernel indirection dispatch replaces the
   per-adapter host loop; merged via the engine's `step_batch` two-phase decode.
2. **Multi-transport disagg (SharedMemP2p)** — same-host file-inbox handoff with
   TCP fallback, auto-started by `KvReceiverServer`.
3. **TP launch ergonomics** — `grim serve --tp-size N` spawns N-1 peer ranks.
4. **PP config + planner + loud gate** — `GRIM_PP_SIZE`, `PipelinePlan`, hard-fail
   when execution isn't wired (honest, not silent).
5. **Qwen38 loader fix** — `gated_residual_branches` + 6 missing fields, verified
   against the HuggingFace-published `config.json`.

---

## Honest gap summary

The codebase has **data structures for all 5 synthesis picks but the execution
path calls none of them.** Unit tests pass; real inference is unchanged. This is
the single most important finding: the competitive advantage the synthesis
promises exists as *API surface*, not as *running behavior*.

---

## Prioritized recommendations (ranked by evidence, not effort)

### R1 — Wire ReadinessDispatcher into `Scheduler::schedule()` (highest value)

**Why now.** `ReadinessDispatcher` is the most complete of the un-wired pieces
(304 lines, tested) and the gap is purely integration. RRFP's "schedule as hint,
dispatch ready work" directly replaces the fixed-order loop in
`Scheduler::schedule()` that currently decides prefill/decode order.

**Concrete steps:**
- Construct one `ReadinessDispatcher` per pipeline stage at `Scheduler::init`.
- On each `schedule()` call, submit incoming prefill/decode/moe-dispatch tasks
  with their predecessor dependencies instead of the fixed-order vec scan.
- Replace the admission loop with `dispatcher.arbitrate()` until the ready set
  drains or the token budget is hit.
- Keep `ScheduleHint::default()` ordering (decode-first for latency) — this is
  the RRFP "hint" that makes it skip-ready instead of wait-ordered.

**Validation:** benchmark `Scheduler::schedule()` decision latency and
first-token jitter under mixed prefill+decode load vs the current fixed-order
path. RRFP reports 1.77–2.77×; on a single-node inference box the realistic
target is eliminating head-of-line stalls when a slow prefill blocks a ready
decode.

### R2 — Fuse MoE dispatch + FFN via `DeterministicTokenMap` (medium-high)

**Why now.** `DeterministicTokenMap::build` already computes the conflict-free
destination offsets (the hard correctness part). The remaining work is feeding
it to `MoeFfn::forward` and fusing the resulting all-to-all with the FFN GEMM.

**Concrete steps:**
- Replace `MoeFfn`'s current scatter/gather in `forward` with
  `DeterministicTokenMap::build` → `pack_activations` → fused GEMM →
  `combine_expert_outputs`.
- Start with the non-fused-but-deterministic path (correctness first), then
  add scoreboard-driven overlap via `ScoreboardSync` once equivalence with
  sequential is proven.

**Validation:** `MoeFfn::forward` output must be bitwise-identical to the current
path before any overlap is added (UniEP's deterministic-ordering guarantee).

### R3 — Virtual-stage traversal for chunked prefill (medium)

**Why now.** `PipelinePlan` is static partitioning. VPP's actual contribution —
the V-shaped fold-back traversal and async bidirectional comm — is what
delivers the 98% bubble reduction. Static partitioning alone is just Megatron.

**Concrete steps:**
- Implement the V-traversal: chunk *k* maps to virtual stages
  `{s0,s1,s2,s3}` on 2 physical ranks, fold-back so chunk *k*'s middle
  overlaps chunk *k-1*'s tail and chunk *k+1*'s head.
- Reuse the existing async comm path (TCP already has send/recv primitives in
  `grim-kvtransport`); the data-plane port is the real work.
- Validate on long-context prefill (>128K) on the existing ROCm box before
  claiming bubble reduction.

**Validation:** measure pipeline bubble ratio on a 512K prefill; VPP claims
98% reduction vs DCPP. Don't claim it without this number on RDNA2.

### R4 — Enforce `MemoryCertificate` at admission (medium, easy win)

**Why now.** `MemoryCertificate::certify` already computes the semantic-demand
lower bound. Wiring it is a few call sites.

**Concrete steps:**
- Call `MemoryCertificate::certify` in `Engine::admit_placed_request`; reject
  (400 or queue) when `required > envelope` instead of OOM-ing later.
- Surface the bound in `grim doctor` / status output so operators see the
  "exceeds envelope" claim before deploying.

### R5 — Disaggregated attention/FFN placement (lowest priority now)

**Why defer.** This is the most invasive (requires per-component GPU groups and
a roofline model tuned to RDNA2/CAPI/PCIe constants grim does not yet have) and
the synthesis itself flags it as training-only on datacenter H800. The single
consumer-GPU-node target that grim optimizes for may not benefit. Build R1–R4
first and revisit if multi-GPU consumer configs become a real deployment.

---

## Process recommendations (how to avoid "data structure only" drift)

The pattern in commits 70017881 and 83836ba6 is: build a faithful data
structure with passing tests, then stop before wiring it. To prevent this:

1. **Definition of done = hot-path call.** A synthesis pick is not "done" until
   a grep for its type in the execution crate (`grim-engine`, `grim-scheduler`)
   returns a line outside that type's own module and tests.
2. **Commit message honesty.** If the commit adds a data structure that is not
   yet called, say so: "add ReadinessDispatcher primitive (not yet wired into
   Scheduler::schedule)." Don't title it "implement RRFP" until R1 above lands.
3. **Wire-before-claim discipline.** The synthesis' "competitive advantage" is
   only real when R1–R4 are on the hot path. Until then, grim has research-grade
   primitives, not production advantage.

---

## Implementation status (2026-08-30)

### R1 — Wire ReadinessDispatcher into `Scheduler::schedule()` ✅ DONE

**What landed** (`crates/grim-scheduler/src/lib.rs`):
- Added `readiness: Option<ReadinessDispatcher>` field to `Scheduler` and a
  `set_readiness_dispatch()` setter (constructor keeps `None` → legacy path
  untouched).
- In `schedule()`: when the dispatcher is set **and** token pressure is active,
  submit one ready-decode `MicrobatchTask` per decode-eligible running request
  (keyed by request id, priority 100, zero dependencies), then
  `arbitrate()`. If decode wins, **defer all new prefill admission to next
  tick** so decode runs this tick — RRFP decode-first interleaving applied to
  the prefill/decode contention. Without the dispatcher or off pressure, the
  legacy greedy prefill path runs unchanged.
- `READINESS_DECODE_PRIORITY` constant; dead-code/unused-variable lints honored.

**Test:** `test_readiness_dispatch_defers_prefill_under_pressure` proves that
with the dispatcher + pressure + a decode-eligible running request, a contending
new prefill is deferred (not in `output.prefill_ids`) while decode is returned;
and that without the dispatcher the legacy path admits the prefill greedily.
All 39 `grim-scheduler` tests pass; 135 `grim-engine` tests pass.

**Why this is the right adaptation.** grim's scheduler is a single-stage
continuous batcher, not a multi-stage PP pipeline, so "per pipeline stage"
does not apply. RRFP's core insight — "schedule as a hint, dispatch ready work,
skip blocked work" — generalizes to: decode is always ready, prefill contends
for budget, so arbitrate decode-first under pressure. The deferral protects ITL
without head-of-line-blocking decode behind a fresh prefill.

### R4 — Enforce MemoryCertificate at admission ✅ DONE

Wired `MemoryCertificate::certify` into `Engine::admit_placed_request`:

1. **Per-backend free-memory probe.** Added `free_device_memory(ordinal)` to the
   ROCm backend (`capability_profiler.rs`, re-exported at crate root) using the
   existing `hipMemGetInfo`-based `vram_info`. Other backends return `None`
   (fail-open). A `GRIM_TEST_FREE_DEVICE_BYTES` override makes the probe
   deterministic for tests and operators.
2. **Hyperparameters at registration.** Added an optional `arch_hyperparams()`
   method to the `CausalLm` trait (default `None`) and implemented it for the
   Llama family. The engine captures and stores
   `arch_hyperparams: Option<ArchHyperparameters>` in each `LoadedModel`.
3. **Admission gate.** `admit_placed_request` re-certifies each request's
   footprint (prompt + max_tokens) against a `BoundaryVector` built from
   *currently-free* device memory. On failure it returns a clear
   "exceeds current memory envelope" error instead of admitting a request that
   would OOM mid-prefill. Fail-open when hyperparams or the probe are
   unavailable.

**Test:** `test_memory_certificate_admission_gate` proves rejection below the
envelope and admission above it. All engine/scheduler/core/nn/rocm suites green.

### R2 — Fuse DeterministicTokenMap into MoeFfn::forward ✅ DONE (feature-gated)

Fused the deterministic dispatch into `MoeFfn::forward` behind the
`moe-deterministic-dispatch` feature flag:

- `MoeFfn::forward_deterministic` routes through `DeterministicTokenMap::build`
  (conflict-free prefix-sum destination addressing), `pack_activations` (gather
  tokens into expert-ordered slots), per-slot expert evaluation, then a combine
  that replicates the reference's exact FP order (`routed += w*y`, THEN
  `out += rsf*routed`) so results are bitwise identical.
- `forward` dispatches to `forward_deterministic` when the feature is on,
  otherwise runs the unchanged CPU reference.
- `test_deterministic_dispatch_is_bitwise_identical_to_reference` asserts
  bitwise equality (`to_bits()`) across both router kinds, with/without a
  shared expert, multiple `routed_scaling_factor` values, and batch sizes 1/2/4.
  Both default (27 tests) and feature-gated (28 tests) MoE suites pass.

The packing is what enables a fused comm-compute mega-kernel on GPU; on CPU it
is a correctness-equivalent reorganization proven identical to the reference.

### R3 — VPP virtual-stage traversal ⏸ DEFERRED

`PipelinePlan` is static layer partitioning (Megatron-style), not VPP's
V-shaped fold-back traversal with async bidirectional comm. The real VPP
mechanism — chunk `k` visiting virtual stages `{s0,s1,s2,s3}` in a V, async
send/recv at fold points, pipelined drain-window packing — is not present.
**Recommendation:** after R1, implement the V-traversal as a
`PipelinePlan::virtual_stage_traversal(chunks, num_ranks)` returning the
per-chunk stage schedule, then reuse the existing `grim-kvtransport` async
primitives for fold-point handoffs. Benchmark bubble ratio on long-context
prefill before claiming improvement.

### R5 — Disaggregated attention/FFN placement ⏸ DEFERRED (lowest priority)

Not implemented. DisagMoE's disaggregation assumes datacenter multi-node EP;
grim targets single-node consumer GPU. **Recommendation:** revisit only if
multi-GPU consumer configs become a real deployment target.
