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

## Suspect — VERIFIED LIVE ON HARDWARE, FIXED

### A3. Quantized-B GPU fast path: both gradients wrong under the documented contract
Device-verified on gfx1201 (`quantized_matmul_backward_gate`, Q4K, non-square
2×4×3): when `quantized_matmul_backward_dx` succeeded, **both** grads diverged
from the CPU reference — grad_a because the dx kernels serve the WEIGHT-STYLE
convention (B stored [n, k], C = A @ Bᵀ) while the fast path also admitted the
documented non-transposed contract (B [k, n]); grad_b because it was computed
as `dev.matmul(A, G)` (= A@G, not Aᵀ@G). Fix: the fused path now serves ONLY
the weight-style convention that production callers (`streaming_forward`
record_matmul, transpose_b = true) use — gate flipped to
`!transpose_a && transpose_b`; grad_b computed as dB_stored[p][q] =
Σ_i G[i][p]·A[i][q] on host (B is the small trainable matrix; no device
transpose primitive exists); the documented non-transposed case falls back to
the verified CPU loops. Gate asserts GPU-vs-CPU parity ≤5e-2 through the real
Q4K dx kernel. NOTE: the Q8_0 JIT source was mid-refactor by the parallel
compressed-tensors workstream during verification; the gate uses host-packed
Q4K instead. Same-class suspect NOT changed: `MAdam::step_param` shares the
frozen-t shape of A1 (fixed in the follow-up commit alongside 8Bit/PagedAdamW).

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


---

## Addendum — quant workstream wiring gates (2026-08-26)

Follow-on wiring for the compressed-tensors formats (workstream consolidated
under this session to stop parallel edits):

- **W4A16 (Marlin) dense: WIRED + GATED.** New `W4A16` dispatch arm in
  `quantized_matmul` routes the resident blob
  (`[codes N*K/8 u32][scales f32]`) through
  `launch_marlin_gemm_w4a16_blob` → `grim_marlin_gemm_w4a16_f32`.
  Gate `w4a16_dense_dispatch_matches_dequant_reference`: GPU output ==
  exact dequantized-weight reference.
- **WNA16 / EmbeddingWNA16Int: dequant services wired + gated.** Public
  `RocmDevice::dequant_wna16_to_f32` / `dequant_embedding_wna16_int_to_f32`
  expose the on-device decoders for expert/table materialization. Gates run
  GPU decode vs host MSB-first reference (4-bit/2-block and 3-bit/embedding).
- **Loud contracts replace silent garbage.** The old `_ =>` dispatch fallbacks
  fed PACKED BYTES into F32 matmuls. Kernel-less variants
  (`CompressedTensorsW8A8Int8/Fp8`, WNA16 fused-GEMM, all weight-only
  backward) now return descriptive errors. Gate:
  `kernel_less_variants_fail_loudly_not_silently`.
- **Repaired the parallel session's kernel sources** (compressed_gemm.rs):
  device helpers were defined after first use with Rust `f32::from_bits`
  syntax in HIP C — every kernel in the aggregate JIT unit failed to compile;
  the WNA16 bit-decoder had a negative-shift UB extracting wrong bits; both
  launchers passed 3-4 args against 5-6-param kernels. Rewritten and
  arity-corrected; blob headers in tests fixed to the documented u32 fields.

**Still open (kernel authoring, tracked):** fused GEMM kernels for
CompressedTensorsW8A8 Int8/Fp8 and WNA16; MoE grouped-kernel consumption of
packed expert blobs (today MoE sizes buffers for these formats via moe.rs and
materializes through the new dequant services — packed-resident grouped MoE
is future kernel work).
