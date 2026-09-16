# WI-DEV-REPETITION-PENALTY — device-side repeat penalty (PLAN-reduce-d2h-h2d B1)

Status: IMPLEMENTED 2026-09-16 (kernel + plumbing + parity green on gfx1201).
Do NOT change `repeat_penalty` CLI default (`1.1`) to fix latency — fixed via the coupling instead.

## B1 audit result (2026-09-16, verified in-tree)

No GPU repeat-penalty kernel exists. Evidence:

- `grim-backend-rocm/src/kernels/device_sampler.rs` — `grim_sample_logits_stochastic`
  takes `(logits, out_token, vocab_size, temperature, top_k, top_p, seed,
  position)` only. No history / penalty params.
- `device_serve.rs:541 launch_sample_stochastic` — same, no penalty args.
- `grim-tensor/src/backend.rs:408 SamplingOps::sample_on_device` trait —
  `(logits, temperature, top_p, top_k, seed)`. No penalty/history.
- `grim-cli/src/run.rs:680,745` — `repeat_penalty > 1.0 && !history.is_empty()`
  forces CPU sampler + full-vocab D2H every decode step (256KB @ 64K vocab).
- `grim-server/src/lib.rs:562` — server device path does NOT gate on
  repeat_penalty at all: penalty is silently IGNORED on device (quality bug,
  opposite direction from CLI's perf bug). Both must be fixed together.

CPU reference semantics (`grim-core/src/sampler.rs:479 apply_repeat_penalty`):
dedup history via HashSet; for each unique `tok < vocab`: `logit = logit < 0
? logit * p : logit / p`. Gate for parity tests below.

## Spec

New HIP kernel `grim_repeat_penalty_apply` (single block, 256 threads):

```c
extern "C" __global__ void grim_repeat_penalty_apply(
    float* __restrict__ logits,        // [vocab], device-resident, modified in place
    const unsigned int* __restrict__ hist_ids, // [hist_len] UNIQUE token ids (host-deduped)
    int hist_len,
    int vocab_size,
    float penalty);                    // > 1.0 guaranteed by caller
// per thread: for (i = tid; i < hist_len; i += block) {
//   u = hist_ids[i]; if (u < vocab_size) {
//     float l = logits[u]; logits[u] = (l < 0.0f) ? l * penalty : l / penalty; } }
```

- Host dedups history (mirror of CPU HashSet; integer-only, no PCIe) then one
  H2D of `hist_len` u32 (≤ history length, typically ≪ vocab) BEFORE the
  sampler launch, same stream. No extra launch: run as pre-pass in the same
  stream right before `grim_sample_logits_stochastic`.
- Unique ids ⇒ each logit touched by exactly one thread ⇒ no races.
- Out-of-range ids skipped (matches CPU `i < out.len()` guard).
- `penalty <= 1.0 || hist_len == 0` ⇒ skip launch entirely (matches CPU early-out).
- NaN logits: CPU leaves NaN comparisons false → `NaN / p = NaN`. Kernel
  `l < 0.0f` is false for NaN → `NaN / p` ⇒ identical. Document + test this.

Plumbing:

1. `SamplingOps::sample_on_device` gains `repeat_penalty: f32, history: &[u32]`
   (default-trait-method overload to avoid breaking vulkan/metal/cpu impls —
   or a separate `sample_on_device_with_penalty` method; prefer separate method,
   smaller blast radius).
2. `RocmDevice::sample_on_device` implements it: dedup → H2D hist →
   `grim_repeat_penalty_apply` → existing sampler path unchanged.
3. CLI `run.rs:680,745` gates drop the penalty clause once wired (keep
   `GRIM_CPU_SAMPLER` escape hatch).
4. Server `lib.rs:562` device path passes registry penalty + history instead
   of ignoring it (fixes silent quality bug; keep CPU fallback contract).

## Acceptance gates

1. Correctness: parity test — random logits (incl. NaN/±inf/negative),
   random histories with duplicates + out-of-range ids, penalties
   {1.0, 1.1, 1.5, 2.0}: device-penalized logits == CPU `apply_repeat_penalty`
   bit-for-bit (integer-representable penalties) or ≤1e-6 else. Gate on
   `GRIM_RUN_GPU_TESTS` (needs gfx arch).
2. Perf: repeat_penalty=1.1 decode tok/s within 5% of penalty=1.0 path on
   `syd-beasty` (proves full-vocab D2H eliminated); `TODO(gpu-verify)`.
3. Server: repeat_penalty actually changes server output distribution
   (regression test for the silent-ignore bug).

## Cost note

Per-step H2D is O(unique history) u32s, re-uploaded each token ⇒ O(n²) total
bytes over a run. For histories ≪ vocab this still beats 256KB/step D2H by
~100×. If histories routinely exceed ~16K tokens, promote to a persistent
device ring buffer with append-only uploads (follow-up, not this WI).

## Implementation notes (2026-09-16)

- Pre-pass kernel + launcher: `kernels/device_sampler.rs`
  (`DEVICE_REPEAT_PENALTY_SOURCE`, `apply_repeat_penalty_on_device` with
  persistent grow-on-demand `penalty_hist_buf` on `RocmDevice`).
- Same-TU registration in `kernels/source_asm.rs`; exports in `lib.rs`.
- `SamplingOps::sample_on_device_with_penalty` (grim-tensor): default
  `Unimplemented` so backends opt in; `RocmDevice` implements via
  `sample_logits_on_device[_at]_with_penalty` (host dedup → H2D → pre-pass →
  sampler; greedy mirrors CPU penalty-before-argmax). Pre-pass miss warns
  once per process, then `Err` → caller CPU-fallback (never silently
  unpenalized). Penalty mutates the [vocab] tail in place — safe: every
  decode step fully overwrites the buffer before sampling.
- CLI: penalty clauses removed from `allow_gpu_sample`/`gpu_sample_ok`;
  device-sample `Err` degrades to CPU (was `?`-propagate).
- Server: `SamplerParams.repeat_penalty` threaded from both chat
  (`register_request_sampler_params`) and completions (`CompletionRequest`)
  bodies; device path tries penalty-aware first, falls through to legacy;
  completions CPU path also fixed to honor `repeat_penalty` (was default 1.0).
- Verification: `tests/repeat_penalty_parity.rs` — bit-for-bit pre-pass parity
  (penalties × histories incl. NaN/±inf/dupes/OOR) + greedy token parity, all
  green on gfx1201; CLI e2e `repeat-penalty 1.1` device path output identical
  to `GRIM_CPU_SAMPLER=1` on LFM2.5-350M-Q8_0.
