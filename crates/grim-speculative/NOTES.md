# Work Item 4 — TIDE Gap Analysis (Gate 4.6.1)

Per `grim_rocm_consumer_perf_planv2.md` §4.3: *"Your first concrete deliverable
for this work item is a short written comparison listing, side by side: what
TIDE proposes, what already exists, and the specific gap."*

This document is the Gate 4.6.1 deliverable. It was written after reading, in
full: `distill.rs`, `confidence_scheduler.rs`, `entropy_confidence_head.rs`,
and `grim-backend-rocm/src/speculative.rs`, plus the live decode loop in
`speculative_wrapper.rs::decode_dspark` and the `CausalLm::forward` trait in
`grim-core/src/model.rs`.

## TIDE's three core ideas

1. **Reuse the target model's already-computed hidden states** as a free
   training/adaptation signal for the draft model — no separate labeled dataset.
2. **Activate draft adaptation only when beneficial** — not on a fixed schedule.
   Triggered by a runtime signal (acceptance-rate drift, divergence).
3. **Zero overhead** — no extra target-model forward pass. The adaptation
   signal comes from the forward that already runs for verification.

## Side-by-side comparison

| TIDE proposes | What exists in grim-speculative | Gap |
|---|---|---|
| (1) Reuse target hidden states | **Nothing.** `CausalLm::forward` returns logits only; the penultimate hidden state is computed and discarded inside the model. `distill.rs` takes only `&str` paths, no tensors. `decode_dspark` calls `target.forward(...)` and binds only `target_logits` (line 273). | **Hidden-state capture surface.** The hidden state is not exposed anywhere in the type system. Capturing it requires extending `CausalLm::forward`'s return type or a side-channel — a cross-crate change in `grim-core`/`grim-models`, upstream of `grim-speculative`. This is the load-bearing prerequisite. |
| (2) Adapt only when beneficial | **Partial / mis-targeted.** `ConfidenceScheduler::choose_verify_len` is a load-gated runtime decision — the *habit* exists. But it gates *verify length*, not *adaptation*. No `should_adapt`-style method exists. The signal that should drive it (measured accept rate) is computed in `decode_dspark` and discarded (`_accepted_per_sec` at line 109). | **Adaptation-trigger decision.** No function returns "draft has drifted; refresh now." The raw accept-rate signal exists in the decode loop but is not fed back to the scheduler. |
| (3) Zero overhead | **Nothing to violate yet.** The decode loop calls `target.forward` exactly once per step. No adaptation code exists, so no accidental second forward pass. | **Same as gap (1).** Zero-overhead is satisfied by construction once the hidden state is captured from the *existing* forward call, not a new one. Risk: a naive fix that re-invokes `target.forward` would violate Gate 4.6.2. |

## What this work item will build (per plan §4.4)

Given the gap analysis above, the scoped work for this pass is:

1. **§4.4.2 — Adaptation-trigger decision** in `confidence_scheduler.rs`:
   a `should_adapt_draft` method that fires when measured acceptance rate
   drifts below a configurable threshold. Deterministic (seeded RNG only,
   no wall-clock gating — Gate 4.6.3). This is the "adapt only when beneficial"
   runtime control, using the accept-rate signal the decode loop already
   computes.

2. **§4.4.3 — Draft-update interface** in `distill.rs`: the *interface*
   definition for a signal-triggered draft refresh — what triggers it, what
   data it consumes — without a full gradient-based training loop (per the
   plan's right-limit: "define the interface ... leave the actual weight-update
   mechanism as an explicitly flagged follow-up").

## What is explicitly deferred (out of scope for this work item)

- **§4.4.1 — Hidden-state capture** (gap 1). This requires changing
  `CausalLm::forward` in `grim-core` and updating every model implementation in
  `grim-models/`. That is a cross-crate API change larger than this work item's
  scope. The interface designed in §4.4.3 anticipates receiving hidden states
  as a parameter so it is ready to consume them once the upstream surface lands.
- **Full gradient-based training loop** — the plan's §4.4.3 and right-limit
  explicitly defer this. The draft-update interface defines *what* a refresh
  would consume, not *how* the weights change.
- **TIDE's heterogeneous-cluster mapping** — not applicable to single-GPU
  consumer hardware (plan §4.5 right-limit).
