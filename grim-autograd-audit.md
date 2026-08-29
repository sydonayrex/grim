# grim-autograd Audit Report

Scope: /D/rex/projects/grim/crates/grim-autograd (tape, ops, backward,
param, registry, injection, loss, replay, adamw, lomo, sophia, galore,
scythe, scythe1, soul_eater, oasis, omnigrad, mm_grpo, contrast_omni,
preference_loss, relora, turbo_finetune, collate, lr_schedule, scale,
omnilo_prune, tops_post, came, etc.)

Classes: L = logic fault (silent wrong result), B = latent bug (hits
only under specific conditions), G = gap (missing capability), P =
perf/correctness footgun.

---

## L1. add_backward ignores broadcasting — WRONG GRADIENT
   ops.rs:727-729
   `add_backward` returns `(out_grad.clone(), out_grad.clone())` for both
   LHS and RHS. When the forward Add was a broadcast (e.g. RHS shape [n]
   added to LHS shape [m, n]), the correct gradient for RHS is
   `sum(out_grad, dim=0)`. The tape records NO shape metadata for Add
   (TapeMetadata::Add is unit), so backward cannot detect or correct
   this. Any caller using broadcasting gets a silently wrong RHS gradient.

   Impact: LoRA path uses same-shape tensors today — not triggered. But
   `record_add` is general-purpose and any future caller broadcasting
   through it is corrupted.

## L2. dora_backward returns ZERO grad for base weight unconditionally
   ops.rs:346-355
   `dora_backward` hardcodes `grad_w_base = zeros` because "base weight is
   frozen." But when `AutogradScope == FullParameter`, the base weight IS
   trainable. The function has no scope parameter and returns zeros anyway,
   silently killing all base-weight learning in full-parameter DoRA mode.

## L3. AdaLomo bias correction frozen at t=1 in streaming path
   lomo.rs:233-234
   `bias_correction2 = 1.0 - beta2.powf(step)` where
   `step = self.step_count.max(1)`. `step_count` is only incremented by
   `step()` (batch mode). The fused streaming path `backward_step` ->
   `step_param` -> `update_param` never increments it. Result: step_count
   stays 0 forever, bias_correction2 is permanently stuck at `1 - beta2^1`,
   and the variance estimate is never bias-corrected for t>1. Effective
   learning rate is too large by a factor of `1/bias_correction2(t)` for
   all t > 1.

## B1. Single-GPU all_reduce_grads DOUBLES the gradient
   param.rs:374-381
   `all_reduce_grads` calls `all_reduce_grads_weighted(... 1.0/num_gpus)`.
   For single-GPU (num_gpus=1, weight=1.0), the RCCL fast path is skipped
   (num_gpus not > 1), falling through to the host accumulation loop:
   `grad_vec *= 1.0; param.accumulate_grad(grad_vec)`.
   Since `grad_vec` IS `param.grad`, this computes `grad += grad`.
   Doubled gradients → doubled effective LR → training instability.
   Footgun for any caller that unconditionally calls all_reduce_grads.

## B2. matmul_backward CPU fallback has a dead-code match arm
   ops.rs:608-618
   The loop body `(true, _) | (_, false) => { ... empty ... }` is a
   no-op match — the actual gradient math is at lines 620-701. The dead
   arm burns cycles and confuses readers but produces correct results
   because the real computation happens below. Harmless but sloppy.

## B3. rope_backward heuristic layout detection
   ops.rs:1567-1589 vs 1591-1606
   Two RoPE layouts are supported (half-split vs adjacent-interleaved
   pairs) with a heuristic: `if dim == 0 || g_vec.len() % (dim*2) != 0`
   picks adjacent-interleaved, else half-split. There is no metadata to
   specify intent — the heuristic can silently pick the wrong layout for
   unusual tensor shapes (e.g. grad length 64 with cos length 20).

## B4. RSLoRA broken under gradient checkpointing
   replay.rs:309
   `TapeMetadata::LoRAApply` hardcodes `scale = alpha / rank`, but RSLoRA
   needs `alpha / sqrt(rank)`. Documented limitation, but means any
   checkpointed segment with RSLoRA produces wrong activations on replay.
   The `LoRAInjectionConfig.use_rs_lora` flag is not serialized into tape
   metadata, so replay has no way to recover the correct scale.

## G1. No gradient accumulation across micro-batches
   `zero_all_grads` is called at the start of every step. There is no API
   to forward+backward multiple micro-batches before stepping. Standard
   technique for fitting large effective batch sizes in VRAM — missing.

## G2. No dropout backward
   The op set has no `Dropout` / `StochasticDepth` / `MaskedFill` op.
   Training without dropout works but regularization is limited to
   weight decay only.

## P1. `accumulate_tensor_grad` cross-device host round-trip
   backward.rs:467-496
   When existing grad and incoming grad are on different devices,
   `g.to_vec_f32()` + `dev.from_cpu()` does a full host round-trip per
   accumulation. In model-parallel or multi-device setups this kills
   throughput.

## P2. `fused_step_with_oasis` computes OASIS projection then discards it
   scythe.rs:342
   `let _ = oasis_proj;` — the projection is computed (mutating subspace
   state via `update_basis`) but the result is unused. Wasted work.

## P3. Tape stores ALL forward activations without checkpointing
   tape.rs:113
   `tensors: HashMap<TensorId, Tensor>` holds every intermediate. Without
   `set_checkpoint_segs`, peak activation memory = full forward pass.
   Fine for LoRA (small activation path), but a problem if the crate is
   ever used for full-parameter training (dense base activations).

## P4. Lion passes hardcoded beta2=0.99 to fused_lion_step
   adamw.rs:1090
   Lion is a sign-based optimizer (no beta2). The fused kernel call
   includes `beta2 = 0.99f32`. If the kernel ignores it, harmless. If it
   uses it (e.g. as a second moment), the semantics are wrong. The CPU
   fallback correctly ignores it.

## P5. all_reduce RCCL sync comment is misleading
   param.rs:352-354
   Comment claims "synchronize is called implicitly by the subsequent
   mul_scalar handle" but there IS NO subsequent mul_scalar in the RCCL
   branch. The sync must be internal to `sum_gradients_device`; the
   comment lies.

## C1. Strengths (notable good bits)
   - LoRA backward finite-difference tested (ops.rs dora_backward_matches_finite_difference).
   - dora_backward has a "P1-17 fix" comment showing a real past bug
     was caught and corrected.
   - backward_step correctly detects multi-entry params and refuses to
     step them (prevents partial-gradient corruption).
   - FIM diagonal approximation in SCYTHE1 is sound (sum of squared
     projected grads per rank component).
   - Tape replay mirrors production forward semantics byte-for-byte.
   - Extensive golden-file tests for MatMul transpose combinations.
   - `assert_rank_compatible` before collective all-reduce catches
     divergence early.

---

## Priority

CRITICAL: L1, L2, L3 — silent wrong gradients or bias-corruption.
HIGH: B1 — gradient doubling on single-GPU all-reduce.
MEDIUM: B3, B4, P1, G1 — fragility and missing standard features.
LOW: B2, P2, P4, P5 — dead code, wasted work, misleading comments.
INFO: G2, P3, C1.
