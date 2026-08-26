# grim-autograd audit — findings and dispositions (2026-08-25)

Scope: `crates/grim-autograd` (~16.4k lines). Full read of the core autodiff
machinery — `tape.rs`, `backward.rs`, `param.rs`, `ops.rs` (matmul/DoRA
backward math) — plus targeted read of the AdamW fused-step path
(`adamw.rs::step_param`) that `backward_step` drives; pattern sweeps over the
remaining optimizer/trainer modules.

## Live — fixed

### A1. AdamW fused streaming path: bias correction frozen at t=1 forever
`adamw.rs::AdamW::step_param` derives bias corrections from `step_count`,
but only the batch entry `step()` increments it. `backward_step`
(the LOMO-style fused backward+optimizer path used by `Tape::drain_and_step`)
calls `step_param` directly per parameter, so on that path **every update
used t=1 corrections forever** — `m̂ = m/(1-β₁¹)` = ~10× mis-scale with
default β₁ that never adapts as moments converge. Fix: per-parameter step
counters (`AdamW.param_steps`) incremented inside `step_param`; the batch
path is unchanged in behavior because it steps every param exactly once per
call. Gate: `step_param_advances_per_param_bias_correction` — sequence
g=[1,0], hand-computed t=2 position (-0.166946) asserted AND the frozen-t=1
value (-0.189959) explicitly rejected. Note: constant-gradient sequences
cannot expose this bug (bias-corrected Adam gives -lr every step for
constant g); the gate uses a changing gradient for that reason.
Same-class suspect NOT changed: `MAdam::step_param` shares the shape of the
bug (its ctor was accidentally touched and reverted during this fix);
verify separately before relying on fused MAdam.

### A2. `backward_step` silently multi-steps tied/shared parameters
A parameter contributing through MULTIPLE tape entries (weight tying, one
LoRA A reused across injections) received an optimizer step PER ENTRY: each
partial gradient stepped then zeroed. Adam moments update once per fragment
instead of once per summed gradient — silent mis-training. The fused path
now refuses loudly (`stepped_params` set + descriptive error directing the
caller to plain `backward`, which sums correctly). Gate:
`backward_step_refuses_multi_entry_param`.

## Suspect — needs device verification (not changed)

### A3. Quantized-B GPU grad_b operand order in `matmul_backward`
When B is quantized and a fused GPU backward dispatch succeeds for grad_a,
grad_b is computed as `dev.matmul(a, out_grad, b.shape())`. Under the
row-major convention documented in roc_device's matmul_op this evaluates
`A[m,k] @ G[?]` whose inner dims only line up when m == m — i.e. it looks
dimensionally wrong for the standard dB = Aᵀ@G unless the backend matmul
performs an implicit transpose. Could not execute this path here (requires
quantized weights on ROCm/CUDA/Metal/Vulkan). VERIFY before trusting
quantized LoRA training gradients; if wrong, symptoms are silently-wrong
weight gradients only through the GPU fast path.

## Latent — recorded

- **A4** `backward()`/`backward_step()` duplicate ~160 lines of match arms;
  a divergence between them is how A1/A2 grew. Worth refactoring into one
  traversal with a stepping callback.
- **A5** `matmul_backward` transposed branch contains a vestigial empty
  `match (transpose_a, transpose_b)` loop (~lines 583-593) left from an
  earlier derivation — dead code, delete opportunistically.
- **A6** `TrainableParam::accumulate_grad` adds into `self.grad.shape()`
  without checking the incoming gradient's shape; backend add behavior on
  mismatch varies (error or silent truncation). Cheap explicit shape check
  recommended.
- **A7** `tape.rs` records RoPE/Softmax backward as exact inverses /
  Jacobians via saved outputs — fine as-is, but replay (`replay.rs`)
  reconstructs freed tensors only for recorded kinds; any NEW TapeKind must
  be added to both replay and backward or checkpointed runs diverge. The
  `checkpointed_gradients_match_uncheckpointed` gate covers existing kinds.

## Positive results

- Tape record/replay/checkpoint retention policy is coherent and gated by a
  checkpointed-vs-full parity test (`checkpointed_gradients_match_uncheckpointed`).
- DoRA forward/backward math verified against the paper formulas including
  the P1-17 grad_x fix noted inline.
- Matmul backward transposed cases carry derived formulas with cache-friendly
  loop reorderings; non-transposed dA/dB match textbook definitions.

## Verification

grim-autograd lib tests incl. new gates — see commit. NOTE (2026-08-25):
final test execution for A1/A2 landed while a parallel quantization-layer
refactor (Storage variant additions in grim-tensor/dtype.rs +
grim-compressed-tensors) was mid-flight in the same tree; if the commit's
CI is red inside grim-format/convert.rs, that is the other workstream's
transient state, not these fixes.
