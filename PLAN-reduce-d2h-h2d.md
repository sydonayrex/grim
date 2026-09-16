# Implementation Plan: Eliminate Self-Inflicted D2H/H2D and Fix Graph-Capture Defaults

Status: DRAFT — verified against uploaded `crates.zip` source on 2026-09-15.
Every finding below cites an exact file:line from the archive. Where the
smackdown.md dress-down got something wrong, that's called out explicitly
so this plan doesn't inherit its errors.

---

## 0. Ground truth correction (read this first)

The smackdown claimed "`GRIM_DECODE_GRAPH` defaults to off — the graph path
is opt-in." **That is false for the actual launch-reducing mechanism.**
Verified:

- `grim-backend-rocm/src/decode_graph_buffers.rs:638-647` —
  `decode_graph_enabled()` returns `true` unless `GRIM_DECODE_GRAPH` or
  `GRIM_CAPTURE_GRAPH` is explicitly set to `0/false/off`. Comment: *"both
  flags disable... = eager."* This is the actual gate `grim-cli/src/run.rs:46`
  checks before calling `try_graph_decode_step`, which is invoked
  unconditionally on every non-prefill CLI decode step (`run.rs:682-698`).
  **So: the other AI you consulted is right. Graph replay is already
  default-on in the CLI**, contradicting the smackdown's blanket claim.

- BUT `lfm2.rs` has two *different, inconsistent* gates for pieces the graph
  replay path depends on:
  - `lfm2.rs:619` and `lfm2.rs:822` use `== Ok("1")` → **opt-in, default off**
    (ShortConv device step, RoPE/QK-norm seeding path).
  - `lfm2.rs:974` and `block.rs:1328` use `!= Ok("0")` → **opt-out, default
    on** (attention device path).
  - `block.rs:1317`'s own comment says *"opt-in via GRIM_DECODE_GRAPH=1"*
    directly above code written as opt-out. Comment and code disagree.

- The reason "608 launches" still shows up in practice despite graph replay
  being default-on: **graph capture only succeeds for the narrow path** —
  ROCm + LFM2 + dense layers + `q_len==1` + no ALiBi + no sliding window
  (`block.rs:1324-1329`) + successful first capture. Any single miss sets
  `graph_failed = true` and the CLI falls back to full eager for the rest
  of the run (`run.rs:46`, `run.rs:56-72`). So the 600-launch number is real
  whenever capture fails or the model/config falls outside that narrow gate
  — not because the flag defaults off.

This plan therefore has two independent workstreams:
**(A)** stop D2H/H2D from happening by default when it doesn't need to,
and **(B)** make capture actually succeed (or fail loudly) instead of
silently degrading to eager for the whole run.

---

## 1. Inventory of self-inflicted defaults (the actual bugs to fix)

| # | Location | Current default | Problem |
|---|----------|------------------|---------|
| 1 | `lfm2.rs:619` `shortconv` decode_graph flag | `== Ok("1")`, OFF | Inconsistent with #4; ShortConv models never get the graph fast path unless the user knows to set this |
| 2 | `lfm2.rs:822` RoPE/QK-norm seeding decode_graph flag | `== Ok("1")`, OFF | Same — this flag gates whether `past_dev`/`pos_base_dev` get seeded correctly for graph mode |
| 3 | `lfm2.rs:889` `GRIM_ROPE_DEV_BASE` | `!= Ok("0")`, ON | Fine on its own, but only takes effect when `decode_graph` (line 890, which itself reads the *local* `decode_graph` from line 974's `!=Ok("0")` check) is true — so it's silently inert whenever #1/#2 haven't also been flipped on |
| 4 | `lfm2.rs:974` attention decode_graph flag | `!= Ok("0")`, ON | Correct polarity, but disagrees with #1/#2, so on a stock run part of the pipeline is graph-ready and part isn't — capture aborts on the mismatch and everything falls back to eager (this is very likely *why* users see 600 launches even though the master switch is on) |
| 5 | `block.rs:1328` decode_graph flag | `!= Ok("0")`, ON | Comment above it (line 1317) says opt-in; contradicts own code |
| 6 | `grim-cli/src/main.rs:206` `repeat_penalty` CLI default | `1.1` | Disables GPU/device sampling every step after the first (`run.rs:680`, `745`); forces CPU softmax fallback on full-vocab D2H every decode token |
| 7 | `grim-core/src/sampler.rs:94` comment vs `:99` value | comment says "mild 1.10", field default is `1.0` | Stale/misleading comment only (not functional) — separate from #6, which is the CLI arg default that actually reaches users |
| 8 | `grim-server/src/lib.rs:1663-1666` `GRIM_TOKEN_PACING_MS` | `unwrap_or(10)` | Artificial 10ms sleep per streamed token, unrelated to PCIe/launch cost — pure self-inflicted latency |
| 9 | `lfm2.rs:41` `mxfp4_qkv_attention` | `false` (doc comment confirms "off by default") | Prefill takes the 3-GEMV+2-norm+2-RoPE+full-Q/K/V-D2H path instead of the fused single-kernel path even when weights are MXFP4-compatible |
| 10 | `shared_attention.rs:323-378` `fused_or_scalar_attention_arena` fallback | unconditional D2H of full `k_arena`/`v_arena` on `qkv_attention` kernel failure | Not gated by an env var at all — a kernel-dispatch failure silently degrades to O(context) D2H **every subsequent call**, with no re-attempt of the device path and no visibility to the caller |

---

## 2. Workstream A — Collapse D2H/H2D to true fallback-only

**Goal:** on ROCm, with a dense LFM2 model, decode-step attention should
do **zero** PCIe transfers unless a real fallback condition is hit
(kernel dispatch failure, non-ROCm device, unsupported layer type,
prefill). Today it does 6 transfers/layer by default because of items
#1–#5 disagreeing with each other.

### A1. Unify the four `decode_graph` gate expressions into one
- **Where:** `lfm2.rs:619`, `lfm2.rs:822`, `lfm2.rs:974`, `block.rs:1328`.
- **What:** extract a single free function,
  `grim_models_transformer::decode_graph_active(device: &Device) -> bool`,
  that mirrors `grim_backend_rocm::decode_graph_enabled()`'s polarity
  (opt-out, `!= Ok("0")` semantics — matches items #4/#5 and matches the
  actual master switch in `decode_graph_buffers.rs:638`) AND checks
  `matches!(device, Device::Rocm(_))`.
- Replace all four call sites with this one function. This makes items
  #1 and #2 default ON, consistent with #4/#5, which is what actually lets
  capture succeed instead of aborting on the ShortConv/RoPE-seed mismatch.
- **Gate:** correctness — add a unit test asserting all four call sites
  agree for a given env-var state (can be done by grepping the compiled
  binary's env-var table in a test, or more simply by refactoring so
  there is only one physical call site the compiler enforces via the
  shared function — prefer the latter).

### A2. Make the arena-fallback D2H in `fused_or_scalar_attention_arena` a real fallback, not a silent every-call degradation
- **Where:** `shared_attention.rs:342-378`.
- **What:** today, on `dev.qkv_attention(...)` failure the function
  downloads the *entire* `k_arena`/`v_arena` and re-enters
  `fused_or_scalar_attention` (3 more H2D), **every single call**, with
  no caching of "the device kernel is broken for this shape" and no
  telemetry. Add:
  1. A `static` (per-device, keyed by launch config) sticky flag —
     once `qkv_attention` fails for a given `(num_heads, num_kv_heads,
     head_dim)` triple, log a `tracing::warn!` once and record the
     failure so repeated identical failures don't re-attempt the device
     kernel every token (the device kernel dispatch itself has a cost
     even when it fails fast).
  2. A counter/metric (`grim_metrics` or equivalent, or a simple
     `AtomicU64` exposed via `/api/stats`) so the smackdown's
     "PCIe traffic per token" estimates are backed by real runtime data
     rather than static analysis. This directly extends the pattern
     already used in `compute_utilization`/`probe_vram_and_gpus`
     described in prior verification work — falls back to `None`/`null`
     instead of fabricating a value.
- **Gate:** correctness (sticky-failure logic must not silently mask a
  legitimately-fixed kernel across separate process runs — key it by
  process lifetime only, never persist to disk) → compile →
  architecture-cleanliness → perf (non-blocking).

### A3. Route `GRIM_ROPE_DEV_BASE` through the same unified gate
- **Where:** `lfm2.rs:889`.
- **What:** once A1 lands, this flag's `&& decode_graph` condition
  (line 890) is checking the *correct*, now-consistent value, so no
  code change needed here beyond confirming it reads the unified
  function's result rather than a locally recomputed `decode_graph`
  bool. Add a regression test that with all defaults (no env vars set)
  on a simulated/mocked `Device::Rocm`, the RoPE path taken is
  `rope_dev_base` (device-base), not the host-position `Vec<u32>` build
  at `lfm2.rs:942-963`.

### A4. Turn on `mxfp4_qkv_attention` conditionally, not unconditionally
- **Where:** `lfm2.rs:41`, `lfm2.rs:266`.
- **What:** do **not** flip the struct default to `true` — the doc
  comment is explicit that F32 is deliberately kept as "golden"
  reference behavior for correctness validation, and flipping a
  numerics-affecting default is out of scope for a latency fix. Instead:
  add an autodetection path at model-load time (`grim-engine`'s model
  loader) that sets `mxfp4_qkv_attention = true` automatically **only
  when** the loaded weights are already in an MXFP4-compatible format
  (`WeightFormat::Rook` per the bird-tier taxonomy) — i.e., don't force
  a quality-affecting recompute path, just stop leaving free performance
  on the table when the data is already in the right layout. If weights
  are F32/Q8_0/etc., leave it off. This is a "self-inflicted lag" only
  in the specific case where the weights already support the fast path
  and grim ignores that fact.
- **Gate:** correctness (oracle-parity test: MXFP4 fused prefill output
  must match F32 reference path within tolerance — the codebase already
  has this pattern for other fused paths, reuse it) → compile →
  perf validation on `syd-beasty` with `TODO(gpu-verify)`.

### A5. Prefill: apply the same device-arena discipline as decode
- **Where:** the non-fused prefill path (`lfm2.rs`, per-layer loop
  described in the smackdown, RoPE output D2H feeding into
  `fused_or_scalar_attention`/`_arena`).
- **What:** the smackdown's own numbers (~150–300MB D2H+H2D for a
  100-token/32-layer prefill) are the single biggest line item in the
  whole report. A4 partially addresses this for MXFP4-formatted weights.
  For the general case, the fix is architectural, not a flag flip:
  prefill should write RoPE(Q/K/V) output tensors directly into the
  same device-resident arena machinery `decode_attention_device` already
  uses (`lfm2.rs:1536` — `k_dev`/`v_dev` growth, `past_dev`/`pos_base_dev`),
  rather than routing through host `Vec<f32>` at all. This is a genuinely
  new code path (prefill currently has no device-arena equivalent), not a
  default-value fix — track as its own work item, e.g.
  `WI-X2-PREFILL-ARENA`, since it changes control flow, not just gating.
- **Gate:** correctness (oracle parity vs current eager-prefill numerics,
  byte-identical KV cache contents after prefill) → compile →
  architecture-cleanliness (must compose with existing
  `Lfm2LayerCache::Attention` variant, not add a parallel cache type) →
  perf (non-blocking, `TODO(gpu-verify)` on `syd-beasty`).

---

## 3. Workstream B — Fix hardcoded/self-inflicted lag values unrelated to D2H

### B1. CLI `repeat_penalty` default (`main.rs:206`)
- **Problem:** `1.1` is a correctness-motivated default (matches Ollama
  behavior for quality) but it has an undocumented, unrelated *side
  effect*: it silently disables GPU/device sampling for the entire run
  (`run.rs:680`, `745`), forcing full-vocab D2H (256KB for a 65536-vocab
  model) + CPU softmax every single decode step.
- **Fix — do NOT change the default value.** Quality defaults shouldn't
  move to chase latency. Instead, decouple the two concerns: implement
  (or verify — check `grim-core/src/sampler.rs` for an existing
  device-side repeat-penalty kernel before assuming one doesn't exist)
  a GPU repeat-penalty kernel so `repeat_penalty > 1.0` no longer forces
  the CPU path. If no such kernel exists, this becomes a proper
  `WI-` work item (device-side repeat-penalty application over the
  logits tensor, using the existing token-history device buffer this
  plan's other work items are already making device-resident), not a
  one-line default change.
- **Gate:** correctness (device repeat-penalty must match CPU reference
  bit-for-bit on integer-representable penalties, or within float
  tolerance) → compile → perf.

### B2. Server `GRIM_TOKEN_PACING_MS` default (`grim-server/src/lib.rs:1663-1666`)
- **Problem:** literally a hardcoded 10ms `tokio::time::sleep` per
  streamed token with no relation to actual backpressure — pure
  self-inflicted lag, and the comment ("avoid overwhelming clients or
  the engine") doesn't correspond to any measured signal.
- **Fix:** replace the fixed sleep with real backpressure — e.g. only
  pace if the outbound stream's write buffer is actually full (most
  HTTP/SSE frameworks expose a `poll_ready`/flush-based backpressure
  signal), or at minimum drop the default to `0` and let ops who
  actually need client-side pacing opt in explicitly via the existing
  env var. Given this plan's bias toward "fallback only, not default,"
  the correct target default is **0ms** unless a specific downstream
  client problem is documented and reproduced.
- **Gate:** correctness (streaming still produces well-formed SSE/ndjson
  under load) → compile → perf (should show near-immediate throughput
  improvement in server benchmarks, non-blocking for correctness sign-off
  but worth an explicit before/after number since this one doesn't need
  GPU hardware to verify — it's a pure host-side timer).

### B3. Sticky graph-failure with no visibility (`run.rs:46`, `graph_failed`)
- **Problem:** once ANY single decode step fails graph capture or replay,
  `graph_failed = true` is set for the rest of the process lifetime
  (`run.rs:46`, `74-77`) with no log line, no metric, and no retry. A
  transient failure (e.g. one bad shape on step 3 of a 10,000-token
  generation) silently downgrades the entire remaining run to the
  600-launch eager path with zero operator visibility.
- **Fix:** add a `tracing::warn!` at every site that sets `graph_failed
  = true` (there are at least two: capture failure at `run.rs:56-58`,
  and replay failure at `run.rs:75-77`), including the underlying error.
  Also expose whether graph mode is active for the current run via
  the CLI's existing stats/summary output (the CLI already prints a
  decode-tokens/sec summary; add "graph: active/fell-back-at-step-N").
  This turns the smackdown's "the fast path is opt-in behind an env var"
  narrative — which this plan has shown is only half-true — into
  something operators can actually observe and act on rather than a fact
  buried in source.
- **Gate:** correctness (logging must not itself break graph capture —
  `tracing` calls inside a capture bracket could plausibly trigger a
  host sync depending on the subscriber; verify no D2H/host-sync is
  introduced inside the capture window specifically, log only on the
  eager fallback side) → compile → architecture-cleanliness.

---

## 4. Sequencing

1. **A1** first — it's the root cause of why the "default-on" master
   switch still produces 600 launches in practice. Everything else is
   secondary until the four gates agree.
2. **A3** falls out of A1 essentially for free (verify with a test, no
   new logic).
3. **B3** next — cheap, high-value observability fix; do this before
   further perf work so subsequent A/B items can be verified by watching
   real logs instead of re-deriving PCIe math by hand each time.
4. **A2** — bounds the worst-case cost of arena-kernel failures and adds
   the telemetry needed to confirm A1 actually eliminated the steady-state
   transfers on real hardware.
5. **B2** — independent, zero-risk, do anytime; recommend early since it's
   a pure host-side change needing no GPU to verify.
6. **B1** — depends on auditing whether a GPU repeat-penalty kernel
   already exists; do the audit before scoping this as new work.
7. **A4** — after A1-A3 are stable, since it changes prefill dispatch and
   should be validated against a known-good baseline.
8. **A5** — largest, most architecturally invasive item (new device-arena
   code path for prefill). Do last, after the smaller decode-path fixes
   are landed and validated on `syd-beasty`, since it's the one item here
   that isn't a gating fix but genuinely new code.

## 5. What NOT to do
- Do not flip `mxfp4_qkv_attention`'s struct default to `true` — that's a
  numerics/quality decision, not a latency-gating bug, and doing so
  without the oracle-parity work in A4 would violate "no performance
  claims without hardware validation" and the "F32 golden path" rationale
  documented in the source itself.
- Do not change `repeat_penalty`'s CLI default away from `1.1` — same
  reasoning; fix the underlying coupling (B1) instead of removing the
  quality behavior it protects.
- Do not persist A2's kernel-failure sticky flag to disk or across
  process restarts — a fixed kernel in a new binary must get a fresh
  attempt.
