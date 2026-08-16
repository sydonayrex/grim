# Fix Plan: Mellum2 MXFP4 Load Path — OOM, Silent Slowness, K-Quant Gaps

## Context

Investigating gibberish output from Mellum2-12B-A2.5B-Thinking-MXFP4_MOE on
ROCm surfaced two prior bugs (already understood/fixed: YaRN params not
threaded through, `Storage::FloatPack`'s attention_factor application —
confirmed correct, not a bug). Chasing the remaining gibberish led to
discovery that **the ROCm on-device MXFP4 fast path is not actually being
taken for this model** — loading instead falls through to a host-side
scalar dequant path that is both extremely slow (~10-11 min for this model,
observed) and memory-unsafe (confirmed OOM / host RAM flood, process
killed). This document is the fix plan for what was found, ranked by
what's blocking basic usability right now.

None of these fixes have been implemented or tested yet — this is a plan,
not a changelog.

---

## Priority 1 — Fix the triple-buffering OOM in `load_quantized`

**Location:** `grim-nn/src/moe.rs`, `ExpertBank::load_quantized`, lines
~288–392.

**The bug:** all three projections (gate, up, down) are fetched and held
simultaneously before any per-expert processing begins:

```rust
let raw_banks = [
    ws.get_raw_packed(projections[0].0)?,
    ws.get_raw_packed(projections[1].0)?,
    ws.get_raw_packed(projections[2].0)?,
];
```

`bank_datas` (built from `raw_banks` at line ~294) holds `&'a [u8]` slices
borrowing from `raw_banks`, so `raw_banks` must stay alive for the entire
`for e in 0..num_experts` loop (all 64 experts, lines ~348–389) before any
of the three full-bank buffers can drop. Confirmed via code read: this is
not a hypothesis, it's the actual borrow lifetime in the current code.

**Measured impact:** for Mellum2 (64 experts, `moe_intermediate_size=896`,
`hidden_size=2304`), each full projection bank is ~1.4 GB packed at MXFP4
density. Holding all three simultaneously means ~4.2 GB packed-byte peak
*per layer*, for 28 layers — and this is on top of, not instead of, the F32
dequant peak described in Priority 2. This was observed to cause an actual
OOM requiring the process to be killed on real hardware.

**Fix:** process one projection at a time instead of pre-fetching all
three. Restructure so `raw_banks[i]` (and its `bank_datas` entry) is
fetched, fully consumed across all 64 experts for that projection, and
dropped before fetching projection `i+1`:

```rust
// Sketch — not exact code, illustrates the shape of the fix:
let mut gate = Vec::with_capacity(num_experts);
let mut up = Vec::with_capacity(num_experts);
let mut down = Vec::with_capacity(num_experts);
let targets = [&mut gate, &mut up, &mut down];

for (i, target) in targets.into_iter().enumerate() {
    let raw = ws.get_raw_packed(projections[i].0)?;
    // ... shape check, build single BankData for this projection only ...
    for e in 0..num_experts {
        // ... slice, frame, materialize_raw, push to target ...
    }
    // `raw` (and its BankData borrow) drops here, before the next
    // projection is fetched.
}
```

This drops peak packed-byte residency from ~4.2 GB/layer to ~1.4 GB/layer
for this model — roughly a 3x reduction, and the reduction scales with
`num_experts` for larger MoE models where this would otherwise be worse.

**Verification:** re-run the load with the same `ps -o pcpu,pmem` monitoring
used to diagnose this; peak RSS during MoE-tensor loading should drop
roughly 3x. A unit test asserting `raw_banks`-equivalent data for only one
projection is live at a time (e.g. via a counting/tracking allocator in a
test harness, if one exists in the workspace) would catch a regression;
otherwise a manual RSS check on a known-large model is the practical gate.

---

## Priority 2 — Investigate why the ROCm on-device MXFP4 fast path isn't taken

**Location:** `grim-nn/src/varbuilder.rs`, `materialize()`, lines ~437–490,
and `Storage::FloatPack`'s doc comment in `grim-tensor/src/dtype.rs:91-93`.

**The bug (root cause not yet confirmed — this is scoped as
investigation + fix, not just fix):** the ROCm fast path exists and
*should* apply to `FloatPack(MxFp4)`:

```rust
#[cfg(feature = "rocm-mem")]
if let Device::Rocm(ordinal) = device {
    if !matches!(dtype.storage, Storage::GroupInt(_)) {
        // ... upload raw packed bytes, keep on-device ...
    }
}
```

The condition only excludes `GroupInt`, so `FloatPack(MxFp4)` should
qualify. But the observed behavior (10+ minute load, steady single-core
CPU burn advancing one `[alias]` log line at a time, eventual OOM) is only
consistent with the **fallthrough path** — `dequant_to_f32` → host-side
`dequant_mxfp4`, one full tensor at a time — actually being taken instead.

Two candidate explanations, not yet distinguished:
1. `Storage::FloatPack`'s doc comment (`grim-tensor/src/dtype.rs:91-93`)
   states outright: *"Dequantized to F32 on load"* — worded as an
   unconditional invariant of the type, not backend-specific. If something
   elsewhere (not yet located) enforces this as a real constraint for
   `FloatPack` specifically — e.g. a check gating the fast path that wasn't
   found in the `materialize()` read so far — that would fully explain the
   observed behavior, and the doc comment is accurate, just misleadingly
   incomplete (should say "on CPU" or "unless backend supports packed
   residency").
2. The fast path's `cfg`/runtime condition is correct as written and *is*
   being taken, but something downstream of `materialize()` (e.g. in
   `ExpertBank::load_quantized`'s call to `ws.materialize_raw(rt, shape)?`
   at line ~384, which is a *different* call than the `materialize()`
   function audited here) routes through a separate, non-fast-path
   function. `materialize_raw` was referenced but not itself read line by
   line in this investigation — worth checking whether it duplicates or
   diverges from `materialize()`'s fast-path logic.

**What to build:**
1. Add one `eprintln!` (temporary, or behind a debug flag) immediately
   inside the ROCm fast-path branch and immediately before the
   `dequant_to_f32` fallthrough, so a single test run against Mellum2
   definitively shows which branch fires for MXFP4 tensors.
2. Based on that result:
   - If the fast path isn't firing due to a real gating check: locate and
     either fix the check (if it's wrong) or, if `FloatPack` genuinely
     cannot support on-device residency for architectural reasons not yet
     understood, update the misleading doc comment and treat Priority 1 +
     Priority 3 as the only available mitigations for this format.
   - If `materialize_raw` (called from `load_quantized`, distinct from
     `materialize()`) is the actual entry point and has its own,
     non-equivalent dispatch logic: audit it with the same rigor applied
     to `materialize()` here, and bring it in line.

**Why this matters more than Priority 1 alone:** even after fixing the
triple-buffering issue, the host path still needs to build a full F32
`Vec<f32>` per tensor — ~1.5 GB for this model's largest MoE tensors,
transient but real, and the *total* work across all tensors is still the
~44 GB-equivalent of scalar dequant compute (see Priority 3 for the
time cost). If the on-device packed-residency path can be made to actually
fire, this becomes a `hipMemcpy`-bound operation instead of a
CPU-scalar-bound one — plausibly cutting load time from ~10 minutes to
under a minute, per the ROCm MXFP4 GEMM kernel path already confirmed to
exist and work correctly (`grim_mxfp4_gemm_tiled`, audited in an earlier
pass of this investigation).

**Left/right limit:** don't "fix" this by just making the host fallback
faster (Priority 3) and calling it done — that treats the symptom. If the
fast path can legitimately be made to fire, it obsoletes most of the need
for Priority 3's optimization work for MXFP4 specifically (though Priority
3's progress-logging fix is still worth doing regardless, for every format
that does hit the host fallback).

---

## Priority 3 — Progress logging for host-side dequant fallback

**Location:** `grim-backend-rocm/src/memory/storage.rs`, `dequant_cpu`,
lines ~514–563 (and wherever `dequant_to_f32` in `grim-nn/src/varbuilder.rs`
is called per-tensor during model load).

**The bug:** zero progress output during host-side dequant. The only
symptom of a legitimate ~10-11 minute load is silence between `[alias]`
log lines — indistinguishable, from the user's perspective, from a true
hang. This directly caused significant debugging time in this session
before `ps`-based CPU monitoring confirmed it was progressing, not stuck.

**Fix:** add a log line (`eprintln!`, consistent with the existing
`[grim]`/`[alias]` convention) either:
- Once per tensor, after dequant completes, with elapsed time for that
  tensor (cheap, minimal-diff option), or
- A coarser once-per-layer summary if per-tensor logging is judged too
  noisy for models with many small tensors.

Suggested format, matching existing log style:
```
[grim] Dequantized blk.11.ffn_gate_exps.weight (MXFP4, 132M elements) in 1.8s
```

**Scope note:** this fix is valuable regardless of the Priority 2 outcome
— even if the on-device fast path gets fixed for MXFP4 specifically, other
quantization formats and other backends (CUDA's equivalent fallback,
Vulkan, Metal) can still hit the same silent-dequant pattern for large
models. This is a small, low-risk, high-value fix independent of
everything else in this plan.

---

## Priority 4 — Wire up missing K-quant host dequant functions

**Location:** `grim-backend-rocm/src/memory/storage.rs`, `dequant_cpu`,
lines ~514–563.

**The bug:** `dequant_cpu`'s match statement only handles 2 of 12
`KQuantScheme` variants (`Q80`, `Q4K`) plus 2 `FloatPack` cases (`Fp8`,
`MxFp4`). Everything else — `Q2K`, `Q3K`, `Q5K`, `Q6K`, and all five
`IQ*` variants — falls through to:
```rust
_ => Err(Error::Backend(format!(
    "to_cpu_vec_f32: host dequant not yet implemented for {:?}",
    dtype.storage
))),
```
Confirmed triggered in this session by LFM2.5-VL-3B-**Q4_K_M**, which
(despite the name) uses Q6_K for select tensors — a common llama.cpp
mixed-precision convention, not specific to this one model.

**What already exists:** every one of the ten missing dequant functions
(`dequant_q2k`, `dequant_q3k`, `dequant_q5k`, `dequant_q6k`,
`dequant_iq4nl`, `dequant_iq4xs`, `dequant_iq3xxs`, `dequant_iq3s`,
`dequant_iq2xxs`, `dequant_iq2xs`, `dequant_iq2s`) is already implemented
in `grim-quant/src/lib.rs` — confirmed present via direct search, not
assumed. This is a wiring gap, not missing implementation work, for 10 of
the 11 missing arms.

**One caveat:** `dequant_iq2s`'s signature is
`fn dequant_iq2s(_data: &[u8], _num_weights: usize) -> Result<Vec<f32>>`
— underscore-prefixed, unused parameters, unlike the other ten. Strong
signal this is a stub (likely returns a placeholder / zeros / an error)
rather than a real decode. Needs a direct read before wiring it in as if
equivalent to the other ten.

**Fix:** add the missing match arms:
```rust
DTypeStorage::KQuant(KQuantScheme::Q2K) => grim_quant::dequant_q2k(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::Q3K) => grim_quant::dequant_q3k(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::Q5K) => grim_quant::dequant_q5k(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::Q6K) => grim_quant::dequant_q6k(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::IQ4NL) => grim_quant::dequant_iq4nl(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::IQ4XS) => grim_quant::dequant_iq4xs(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => grim_quant::dequant_iq3xxs(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::IQ3S) => grim_quant::dequant_iq3s(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => grim_quant::dequant_iq2xxs(raw, elem_count),
DTypeStorage::KQuant(KQuantScheme::IQ2XS) => grim_quant::dequant_iq2xs(raw, elem_count),
// IQ2S: verify dequant_iq2s is a real implementation, not a stub, before wiring in.
DTypeStorage::KQuant(KQuantScheme::IQ2S) => grim_quant::dequant_iq2s(raw, elem_count),
```

**Verification:** re-run the LFM2.5-VL-3B-Q4_K_M load that originally
surfaced this; should proceed past the point it previously errored.
Ideally also add a small unit test per newly-wired variant (round-trip a
known-good block through `grim_quant`'s own encode/decode if such a
harness exists, or at minimum a shape/no-panic smoke test) rather than
relying solely on one real-model load as the check.

---

## Explicitly not in this plan (deferred)

- **Whether Mellum2's gibberish is fully resolved.** The YaRN fix is
  confirmed correct and live (`yarn: Some(YaRNParams {...})` observed in
  the load log this session). Whether output is coherent after loading
  has not yet been observed end-to-end, because the load OOM'd before
  generation could run. This plan exists to get a clean load to actually
  test that — it is a prerequisite investigation, not a confirmed
  additional bug.
- **The `WI-SPINQUANT-AttentionGate` hang** (separate document, separate
  command path — `grim convert`/`grim oxidizer convert`, not `grim run`).
  Confirmed via code search this session that `grim run`'s load path does
  not call `pack_tensors`/`spinquant_rotate` at all — not applicable here,
  tracked separately.
- **The CLI-side transfer-overhead findings** (per-token CPU↔GPU
  round-trips, embedding round-trip, untuned GEMM tiles outside gfx1036) —
  real, previously verified, but orthogonal to the load-path issues in
  this document. Worth its own pass once a model can actually finish
  loading and generating.

---

## Suggested order

**Priority 1 (OOM fix) and Priority 4 (K-quant wiring) are independent and
can be done in parallel** — different files, no shared risk.

**Priority 2 (fast-path investigation) should start with just the
diagnostic `eprintln!` before committing to a fix direction** — it's the
one item here where the actual code change depends on what the diagnostic
reveals.

**Priority 3 (progress logging) is small enough to bundle with whichever
of the above lands first** — no reason to sequence it separately.

Realistic near-term goal: land Priority 1 + Priority 3 immediately (both
low-risk, well-understood), which turns "silent OOM-prone 10-minute load"
into "observable, RSS-bounded ~10-minute load" — enough to safely let a
Mellum2 load finish and finally get a real answer on whether the YaRN fix
resolved the original gibberish. Priority 2 (getting load time down to
under a minute) and Priority 4 (broader model compatibility) can follow
without blocking that experiment.
