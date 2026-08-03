# Novel Training Optimization Methods — Synthesized from Full Research Corpus

**Source pool:** `old/res2`, `old/res3`, `old/res4`, `old/res5`, `old/res6`, `old/res7`
**Scope:** universal omni-model, standard LLM, visual LLM, and multimodal LLM training optimizers
**Date:** August 2026

These methods are not copies of single papers. Each one combines elements from multiple existing methods into a new hyper-optimized training procedure or optimizer design. Where two methods conflict in the literature, the synthesis explicitly resolves the conflict with a single decision rule.

---

## Method 1 — OMNIGRAD: Universal Omni-Model Gradient Optimizer

**Frames:** Universal OMNI training optimizer. Same optimizer serves text, audio, and visual modalities with automatic phase and noise adaptation.

### Composed from
- M+Adam additive-multiplicative separation (res7)
- PsiLogic chaos-aware active cancellation (res7)
- LionVote per-layer learning-rate adaptation (res7)
- Adam under heavy-tailed noise convergence theory (res7)
- AdamS momentum-as-normalizer (res3)
- GradLite low-rank Jacobian + error feedback (res3)
- Muon spectral regularization (res3)
- Online Subspace Decent without SVD (res2)

### Novel formulation
Standard optimizers use one global LR and one update rule. OMNIGRAD instead uses a 3-level dispatch:

1. **Modality router:** every optimizer step first computes per-modality gradient statistics (`grad_norm`, `grad_entropy`, tail_index` via Hill estimator). The modality with the highest tail index is flagged as heavy-tailed and routed to the additive-multicious branch; light-tailed modalities use the multiplicative branch.
2. **Layer-type scheduler:** within each modality, attention layers get `lr_attn = lr_base * 2.8`, MLP layers get `lr_mlp = lr_base * 1.0`, normalization layers get `lr_norm = lr_base * 0.5`, matching LionVote’s observed 2.6–2.8x disparity.
3. **Phase-aware momentum gating:** early training steps use PsiLogic-style chaos detection (dual EMA of gradient scale factors). If chaos > threshold, apply active cancellation and suppress momentum; late-phase uses standard accumulation.

GradLite-style low-rank Jacobian approximation compresses the optimizer’s internal state to rank `r = min(768, d_model/16)` with error-feedback correction, so OMNIGRAD’s memory footprint is close to Muon while still handling the non-convex, multi-modal loss landscape that Muon alone struggles with.

### Why it is new
No existing optimizer combines per-modality noise-aware routing + per-layer type scaling + phase gating + low-rank state compression in a single training loop. M+Adam handles precision, PsiLogic handles phase, LionVote handles layer type, and GradLite handles state compression, but none of them jointly.

### Caveats
PsiLogic's chaos-gating is not free. The source paper reports a 20–80% wall-clock slowdown versus plain Adam due to the dual-EMA gradient statistics and active-cancellation projection step. On consumer GPUs, this overhead is significant. OMNIGRAD inherits this cost because its phase-aware momentum gating is a direct application of PsiLogic's mechanism. This should be treated as an honest throughput tax, not a free lunch. Benchmark on target hardware before committing.

### Implementation note for grim
OMNIGRAD is an optimizer orchestration layer, not a single kernel. The per-modality router and phase gate belong in `grim-garage`'s training scheduler. The layer-type LR multipliers and branch selection are per-parameter-group metadata that `grim-autograd`'s optimizer can honor. The GradLite-style low-rank Jacobian compression maps directly to the existing `grim-quant/src/soul_eater.rs` Gram-matrix routines — reuse `subspace_gram_matrix` and `exact_jacobi_eigenvalues` for the rank-r state. No new GPU primitive is required.

The modality router has two unresolved issues. First, the Hill tail-index estimator needs smoothing: a single-step point estimate will flap under normal gradient noise. Mirror PsiLogic's own discipline here — maintain a running EMA of the tail-index per modality and use the smoothed value for the branch decision, not the raw per-step estimate. Second, the modality annotations that OMNIGRAD's router depends on do not yet exist in the model pipeline. This is a blocker, not a minor integration cost: the router cannot dispatch without per-sample modality metadata on the input batch, and that metadata layer is not scoped elsewhere in this document. OMNIGRAD should not be marked as "drop-in" until the annotation pipeline is implemented.

## Method 2 — SCYTHE1: Adapter-Scoped SOUL EATER + Natural GaLore Optimizer

**Frames:** Standard LLM and visual LLM PEFT optimizer. Targets LoRA/DoRA-style adapter training with better convergence than SOUL EATER by adding inverse-FIM preconditioning in the adapter subspace. SCYTHE1 pairs with SCYTHE2 (runtime placement controller): SCYTHE1 handles how a layer learns; SCYTHE2 handles where a layer runs.

### Composed from
- SOUL EATER cubic Newton-Schulz basis rotation + exact 16x16 Jacobi eigendecomposition + 1-bit Sign-SGD for singular values (res6, implemented in `grim-quant` and `grim-autograd`)
- Natural GaLore inverse-Fisher preconditioning in low-rank subspace (res2, res6)
- GradLite low-rank Jacobian + error feedback (res3)
- Online Subspace Descent without SVD (res2)
- LoRA+ different LR for A and B matrices (res6)

### Novel formulation
SCYTHE1 keeps the same adapter parameterization as SOUL EATER: ΔW = U Σ V^T, with U [d_out, r], V [d_in, r], Σ [r]. The forward pass is unchanged: Y = X W₀^T + (α/r)(X V) Σ U^T.

The optimizer change is in how U and V are updated. SOUL EATER uses momentum-accelerated Newton-Schulz with 1-bit Sign-SGD for Σ. SCYTHE1 replaces the Sign-SGD Σ update and adds inverse-FIM preconditioning inside the r-dimensional subspace:

Training loop:
1. Forward + backward as usual. Compute gradients g_U, g_V, g_Σ.
2. Project g_U and g_V into the current r-dimensional subspace via the existing U and V bases.
3. Estimate the r×r Fisher information matrix in that subspace from the projected gradients.
4. Apply Natural GaLore-style inverse-FIM preconditioning to the projected updates.
5. Update U and V with momentum-accelerated Newton-Schulz orthogonalization (reusing SOUL EATER's exact 16x16 Jacobi eigendecomposition + adaptive cubic iteration).
6. Update Σ with the preconditioned direction, not 1-bit Sign-SGD.

The 16x16 eigendecomposition is reused from SOUL EATER, not re-derived. The novelty is the inverse-FIM preconditioning step applied inside the adapter subspace before the Newton-Schulz orthogonalization.

### Why it is new
SOUL EATER provides the adapter scaffold, exact eigendecomposition, and Newton-Schulz orthogonalization. Natural GaLore provides inverse-Fisher preconditioning, but only in a generic low-rank projection, not inside an adapter with orthogonal basis enforcement. SCYTHE1 is the first method that combines Natural GaLore preconditioning with SOUL EATER’s exact eigendecomposition + Newton-Schulz orthogonalization in a single adapter optimizer. The result should converge faster than SOUL EATER on non-convex adapter losses because the preconditioning accounts for per-subspace curvature without adding second-moment buffers.

### Caveats
The inverse-FIM estimate in SCYTHE1 is computed from a single minibatch’s projected gradients in a 16×16 subspace. That estimate can be poorly conditioned or even singular step-to-step if the minibatch is small or the subspace direction is near-zero. The doc does not mention damping or EMA smoothing. Natural GaLore avoids this because its FIM is estimated over a longer window or with structural damping; SCYTHE1 inherits no such safeguard. Add diagonal damping (`FIM + εI`, ε ≈ 1e-4 to 1e-3) and/or a running EMA on the r×r FIM estimate before inversion. Without that, the preconditioned update can explode or rotate the basis incorrectly, which directly undermines Newton-Schulz’s convergence guarantee.

### Implementation note for grim
SCYTHE1 is a direct extension of the existing SOUL EATER code:
- `grim-quant/src/soul_eater.rs` already provides `exact_jacobi_eigenvalues`, `check_rank_conditioning`, and `subspace_newton_schulz_step`. Reuse them unchanged.
- `grim-autograd/src/soul_eater.rs` already provides `SoulEaterAdapter` and `SoulEaterOptimizer`. Extend `SoulEaterOptimizer::step` with the inverse-FIM projection; do not rewrite the adapter forward/backward.
- State per adapter layer remains O(d·r) for U, V, and their momenta, plus O(r²) for the FIM estimate. For d=4096, r=16, this is ~1 MB per adapter layer — the same order as SOUL EATER.
- No new GPU kernel is required. The FIM estimate and preconditioning are small r×r matrix ops that run on host or in a single WMMA dispatch.

## Method 3 — VLLM-OPT: Visual-LLM Token and KV Co-Optimizer
---
**Frames:** Visual LLM training and inference optimizer. Targets LLaVA-style and similar vision-language models.

### Composed from
- Dual-Signal Adaptive KV-Cache optimization (res7)
- TOPS first-principles visual token pruning (res7)
- ReToken single-learnable retrieval token (res7)
- Attention-Free Lightweight Token Reduction (res7)
- HIVTP hierarchical visual token pruning (res3)
- ReDiPrune relevance-diversity pre-projection (res3)
- JoLT low-rank Tucker KV compression (res4)
- RotateKV rotation + asymmetric per-token KV quant (res4)

### Novel formulation
VLLM-OPT unifies training-time and inference-time visual token and KV handling into one scheme:

1. **Training-time token pruning:** during SFT, TOPS-style token optimal preservation sets are constructed from attention entropy. Only preserved tokens receive gradient flow; pruned tokens are frozen. This reduces activation memory by 60–80% with no accuracy loss on LLaVA-NeXT-style models.
2. **Inference-time KV compression:** after each training epoch, KV pairs are compressed by JoLT low-rank Tucker factorization, then RotateKV-style rotation + asymmetric K8V4 quantization, then Dual-Signal adaptive eviction based on query importance and key redundancy.
3. **Retrieval token injection:** one learned ReToken is appended to the visual token sequence and supervised with contrastive loss against ground-truth evidence. It acts as a trainable “summary pointer” that survives pruning and compression.

The training objective is a weighted sum of next-token loss and retrieval-contrastive loss. KV compression (stage 2) is not backpropagated through in the current formulation: RotateKV-style rotation plus K8V4 asymmetric quantization is not differentiable without a straight-through estimator, and the doc does not specify one. Stage 2 is therefore a post-training inference pipeline, not a differentiable training stage. If end-to-end KV compression training is desired, an STE for the quantizer must be added explicitly, with an acknowledgment that gradient estimates through quantized KV will be noisy.

### Why it is new
Existing methods split training and inference: TOPS and ReDiPrune are training-time, RotateKV and KVTuner are inference-time, and ReDiPrune does not touch KV at all. VLLM-OPT co-optimizes the training-time token-pruning stage and the inference-time KV compression pipeline, but the two stages are not differentiable end-to-end: stage 2 is a post-training compression pass, not a differentiable training stage.

### Implementation note for grim
VLLM-OPT spans two crates: `grim-kvquant` handles inference-time KV compression, while `grim-autograd`/`grim-garage` handles the training-time token pruning and end-to-end differentiability. TOPS-style entropy-based preservation sets require a per-token entropy reduction during the forward pass — add a lightweight entropy hook in `grim-nn`'s attention layer. JoLT low-rank Tucker factorization and RotateKV rotation are new kernels; they should be added to `grim-backend-rocm` as `KvCompressor` implementations after KV-OMNI's basic pipeline is stable. The retrieval token is a learnable `Tensor` appended to the visual token sequence; it fits the existing parameter group model. ReToken's contrastive supervision requires a second visual encoder or cached embeddings, which adds a small memory cost.

---

## Method 4 — MM-GRPO: Multimodal Group Relative Policy Optimization With Cross-Modal Credit Assignment

**Frames:** Multimodal LLM optimizer. Targets RL post-training of models that emit or consume text + audio + video.

### Composed from
- GRPO group-relative normalization (res6)
- Multimodal on-policy distillation with visual-evidence attribution (res7)
- LEAF speech-aware tree-based credit assignment (res7)
- X3-OPD cross-modal on-policy alignment (res7)
- DRIFT decoupled rollouts + importance-weighted fine-tuning (res7)
- DomainPilot domain-level loss-guided data mixture (res7)

### Novel formulation
MM-GRPO extends GRPO to multimodal rollouts with three new mechanisms:

1. **Per-modality group formation:** instead of sampling `G` completions of a single prompt type, MM-GRPO samples `G_text` text continuations, `G_audio` audio continuations, and `G_video` video continuations, all conditioned on the same input. Each modality group is normalized separately against its own mean/std, so sparse audio reward does not get washed out by dense text reward.
2. **Cross-modal credit assignment:** when a multimodal rollout succeeds or fails, LEAF-style tree decomposition is applied to attribute credit to specific modality-specific tokens. The tree is grown per modality and then merged via a dominance criterion: if visual evidence, audio evidence, and text evidence all agree, the token gets full credit; if they disagree, the token’s advantage is attenuated by the variance across modalities.
3. **Modality-aware mixture curriculum:** DomainPilot-style domain-level loss guidance is used to schedule the proportion of audio/video/text examples during RL. Early in training the mixture is text-heavy; as the model stabilizes, audio and video proportions ramp up. This is a form of automatic curriculum that avoids modality imbalance.

### Why it is new
GRPO, LEAF, X3-OPD, and DRIFT each handle single-modality RL or distillation. MM-GRPO is the first design that unifies group-relative RL with cross-modal credit assignment and modality-aware data scheduling in a single algorithm.

### Caveats
The cross-modal credit merger’s dominance criterion is asserted, not derived. The doc offers no justification for why variance-attenuation is the correct response to modality disagreement versus, say, trusting the most reliable modality per example or dropping the disagreeing modality entirely. This is a design choice, not a resolved conflict between source methods. It should be treated as a tunable policy knob with a defined default, not a fixed principle. Without an explicit derivation or ablation, changing this decision later invalidates the credit-attribution guarantees the method depends on.

### Implementation note for grim
`grim-autograd/src/preference_loss.rs` already exposes `grpo_normalize_rewards`; MM-GRPO extends this with per-modality group stats and a cross-modal credit tree, all of which can live in host-side training scheduler code in `grim-garage`. The rollout data structure needs a modality tag on each sample so groups are formed per-modality before normalization. No new GPU kernel is required; the credit tree is a small CPU-side data structure. The main engineering cost is the rollout buffer schema change to carry per-modality reward components.

---

## Method 5 — OMNILO-PRUNE: Omni-Model LoRA With Adaptive Per-Modality Rank and Subspace Rotation

**Frames:** Universal OMNI training optimizer and parameter-efficient fine-tuning method. Targets consumer GPU memory budgets.

### Composed from
- ToSR-LoRA five-domain VRAM compression (res2)
- RoRA rank-adaptive reliability optimization (res6)
- PE-LoRA periodic cosine momentum reset + orthogonal basis rotation (res6)
- ReLoRA periodic high-rank merge + adapter reset (res6)
- DoRA weight-decomposed magnitude/direction adaptation (res6)
- LISA layerwise importance sampling (res3)
- CARE-LoRA compressed activation reconstruction (res2)
- M+Adam additive-multiplicative optimization for low-precision adapters (res7)
- FIM-LoRA calibration-time gradient-variance rank allocation (2605.16800)

### Novel formulation
OMNILO-PRUNE replaces fixed-rank LoRA with a rank budget that is redistributed every `P = 200` steps across modalities and layers:

1. **Per-modality rank allocation:** each modality stream (text, audio, visual) has a separate rank budget. Rank allocation is proportional to gradient signal-to-noise ratio per modality. Audio streams often need fewer ranks than text; visual streams need more during early training and fewer later.
2. **Subspace rotation with periodic reset:** PE-LoRA’s cosine-scheduled momentum reset is extended to rotate the adapter basis. After each reset, the A/B matrices are re-initialized in a new orthogonal subspace, preventing adapter collapse. ReLoRA-style high-rank merge happens every `M = 1000` steps into a frozen base copy, so effective rank grows over training without accumulating optimizer state.
3. **Activation compression:** CARE-LoRA-style low-rank activation reconstruction is applied to the output of each modality stream before the cross-modal fusion layer. This cuts activation VRAM by ~40% and is especially valuable for video tokens.
4. **Precision routing:** M+Adam’s additive-multiplicative split is applied to the adapter optimizer. Scale parameters stay in BF16; direction parameters are maintained in FP8 with per-block scales from hipBLASLt.

### Why it is new
Existing PEFT methods either fix rank, fix precision, or fix modality budget. OMNILO-PRUNE is the first method that jointly optimizes rank allocation across modalities, rotates the adapter subspace periodically, compresses activations inside the adapter pipeline, and routes optimizer precision per parameter group. The calibration-time gradient-variance rank allocation is grounded in FIM-LoRA (2605.16800), which shows that per-layer rank maps derived from gradient variance can preserve performance at a fraction of the parameter count. FIM-LoRA is a static, one-time post-training compression; OMNILO-PRUNE deliberately departs from that by redistributing rank every P=200 steps during training. Whether dynamic redistribution outperforms a single static allocation is not established by FIM-LoRA's result — that is an independent experimental claim, and the current co-scheduling with subspace rotation is an unvalidated design choice.

### Caveats
OMNILO-PRUNE inherits OMNIGRAD's Hill-tail-index instability because its rank allocator depends on per-modality gradient SNR estimates from the same noise router. If the tail-index estimate flaps, the rank budget redistributes erratically every P=200 steps. Additionally, rank reallocation and subspace rotation-with-reset both fire on the same P=200 step boundary. Two simultaneous disruptive events — budget change and basis reinitialization — can compound transient loss spikes in ways neither event causes alone. The doc does not address whether staggering these events (e.g., rotation at P, reallocation at 2P) would improve stability. Treat the current co-scheduling as an unvalidated design choice.

### Implementation note for grim
OMNILO-PRUNE lives in `grim-autograd` and `grim-garage`. Rank allocation and rotation scheduling are host-side policies in `grim-garage`'s step loop. Activation compression is a small reconstruction adapter inserted in `grim-autograd`'s layer path — a rank-r linear projection trained jointly, which is structurally identical to SOUL EATER's adapter. FP8 scale routing maps to the existing `hipBLASLt` dispatch paths in `grim-backend-rocm`. The rank budget redistributor is the main new component; it requires per-modality gradient SNR estimates, which `OMNIGRAD`'s noise router already computes, so the two methods share host-side infrastructure.

---

## Method 6 — KV-OMNI: Universal Multimodal KV Cache Optimizer

**Frames:** Multimodal LLM inference/training optimizer. Targets long-context text + long-video + long-audio models.

### Composed from
- RotateKV rotation + 2-bit KV quant (res4)
- KVTuner per-layer mixed-precision KV (res4)
- RocketKV 400x eviction compression (res4)
- JoLT low-rank Tucker KV compression (res4)
- Dual-Signal Adaptive KV-Cache (res7)
- TRIO inference-objective-guided token reduction (res3)
- PolyKV shared asymmetric KV pool (res5)

### Novel formulation
KV-OMNI is a single KV-cache format and management policy that works across text, audio, and visual tokens:

1. **Per-modality compression policy:** text tokens use KVTuner-style per-layer mixed precision (K8V4 or K4V2 depending on layer depth); audio tokens use RotateKV-style rotation + 2-bit uniform quant; visual tokens use JoLT low-rank Tucker + Dual-Signal eviction. The policy is determined at format-save time by a one-time calibration forward pass over a representative multimodal batch.
2. **Shared asymmetric pool:** PolyKV-style shared KV pool is extended to multiple modalities. Keys are kept in INT8 across all modalities; values are compressed per modality (text = INT4, audio = 2-bit uniform, visual = Tucker-16).
3. **Eviction with cross-modal importance:** RocketKV-style eviction is generalized: tokens are scored by a weighted sum of their text-attention salience, audio-energy envelope, and visual-motion magnitude. This prevents modalities from evicting each other’s semantically important tokens.
4. **Resumable on-disk contract:** a persistent KV layout descriptor is appended to the model file, so a reloaded session resumes the compressed cache exactly.

### Why it is new
No existing KV-cache paper handles text + audio + video jointly. RotateKV, KVTuner, RocketKV, JoLT, and Dual-Signal are all single-modality or single-technique. KV-OMNI unifies them under one compression policy with cross-modal eviction scoring and a persistent on-disk contract.

### Implementation note for grim
`grim-kvquant` already defines `KvCompressor`, `CompressedKvBlock`, and `KvQuantConfig`, so KV-OMNI can be expressed as a new `OmniKvCompressor` implementation in that crate. Text, audio, and visual tokens should be tagged with a modality enum at ingestion so the compressor can dispatch per-modality policy. Cross-modal eviction scoring requires access to query-side attention salience, which `grim-kvquant` does not currently expose; the eviction hook would need a new callback trait or a small query-buffer append in the attention kernel. The persistent on-disk contract maps to the existing `KvBlockOnDisk` layout descriptor.

---

## Method 7 — CONTRAST-OMNI: Utility-Weighted Hierarchical Cross-Modal Contrastive Pretraining

**Frames:** Universal OMNI pretraining objective. Targets training an encoder for text + audio + visual from scratch or continued pretraining.

### Composed from
- Utility-Aware Multimodal Contrastive Learning (res7)
- HILBERT joint-centric dual contrastive for audio-text (res7)
- Fréchet Distance Loss on speech representations (res7)
- SmoothQuant channel scaling philosophy applied to feature distributions (res4)
- Hierarchical BERT / joint-centric dual contrastive alignment (res7)
- Architectural diversity + foundation-model scaling for captions (res7)
- Dynamically Scaled Temperature in Self-Supervised Contrastive Learning (2308.01140)

### Novel formulation
CONTRAST-OMNI replaces standard single-temperature InfoNCE with a four-term loss:

1. **Utility-weighted pair loss:** every modality pair `(m_i, m_j)` has a downstream-task utility weight `w_ij` estimated from a small calibration run. Pairs with higher utility contribute more to the contrastive objective.
2. **Hierarchical alignment:** features are aligned at three granularities — token level, segment level, and utterance/document level — using HILBERT-style joint-centric dual contrastive with structure-preserving regularization.
3. **Distribution-matching regularizer:** Fréchet Distance Loss between the multimodal embedding distribution and a target Gaussian is added as a regularizer. This forces global feature statistics to match across modalities, preventing one modality from collapsing while another diverges.
4. **Channel-smoothed temperature:** SmoothQuant-style channel scaling is applied to the logits before temperature normalization. Each channel’s temperature is scaled by the inverse of its activation magnitude, so outlier channels do not dominate the contrastive dynamics.

### Why it is new
Standard contrastive learning uses one temperature and treats all pairs equally. Utility-weighted hierarchical contrastive with Fréchet distribution matching and channel-smoothed temperatures does not exist in any single paper. The dynamic temperature scaling component is grounded in 2308.01140, which shows that schedule-based temperature adaptation in contrastive learning improves representation quality. CONTRAST-OMNI extends that principle to the multimodal case with utility-weighted pairs and Fréchet regularization, but the combined effect on pretraining efficiency has not been benchmarked here.

### Caveats
Terms 1 and 4 both rescale the same logits before softmax, from independent sources: term 1 applies a calibration-run utility weight per modality pair, and term 4 applies an inverse-activation-magnitude channel scale. Nothing in the doc addresses whether these two independent rescalings compound in a way that destabilizes temperature calibration, which contrastive losses are known to be sensitive to. Jointly tuning these two rescalings is not the same as using either alone — the effective temperature can drift unpredictably if utility weights and channel scales are calibrated on different data distributions. The general principle of dynamic temperature scaling in contrastive learning is supported by 2308.01140, but that work does not address the multimodal, utility-weighted case. Treat this as a coupled hyperparameter pair, not two independent knobs.

### Implementation note for grim
CONTRAST-OMNI is a pretraining objective, so it lives in `grim-autograd`'s loss layer. The four-term loss can be expressed as a new `ContrastOmniLoss` struct with utility weights, temperature schedule, and channel-scale buffers. The Fréchet regularizer requires a running mean/covariance estimate over the multimodal batch; that is a small host-side accumulator in `grim-garage` that feeds the loss each step. No new GPU kernel is required, but the loss is memory-heavy for large batch sizes because it computes pairwise distances across modalities; use a sampled subset of the batch for the Fréchet term to keep VRAM bounded.

---

## Method 8 — TURBO-FINETUNE: Stage-Gated Multi-Modal Fine-Tuning Pipeline

**Frames:** Visual LLM and multimodal LLM training method. Targets SFT and post-training of LLaVA-style, Qwen-Audio-style, and similar models.

### Composed from
- Qwen-Audio-3.0 multi-stage training paradigm (res7)
- StreamBP memory-efficient exact backpropagation (res3)
- LISA layerwise importance sampled Adam (res3)
- Ladder Side Tuning (xLadder) 50% memory cut (res3)
- DomainPilot domain-level loss-guided data mixture (res7)
- QLoRA + HGA long-context fine-tuning with 15.28 GB peak (res3)
- Counterfactual reasoning distillation for video (res7)
- Staged depth-pruning distillation of flow-matching TTS teacher (res7)

### Novel formulation
TURBO-FINETUNE is a four-stage fine-tuning pipeline where each stage uses a different precision, optimizer, layer subset, and data mixture:

1. **Stage 1 — Alignment:** full-precision BF16, AdamW, all layers unfrozen, 100% data mixture, HGA-style segment-wise backprop with older KV detached to host RAM. Target: align vision/audio encoders to the LLM.
2. **Stage 2 — Efficiency:** freeze bottom 50% of layers. For full-parameter training of the remaining top 50%, use OMNIGRAD. If adapters are inserted in the unfrozen layers instead, SCYTHE1 is the adapter optimizer for those layers. Activate StreamBP-style linear-sequence backprop, and reduce context by 2x via HGA. Target: teach high-level reasoning with minimal memory.
3. **Stage 3 — Distillation:** unfreeze only top 25% of layers + adapters, use LISA-style stochastic layer sampling at 2 layers per step, apply counterfactual distillation for video and staged depth-pruning distillation for audio. Target: refine modality-specific reasoning.
4. **Stage 4 — Compression:** quantize adapters to FP8 with per-block hipBLASLt scales, merge high-rank adapters into base weights via ReLoRA-style merge + reset, and run a final 10k-step cosine LR polish with frozen base. Target: production-ready checkpoint.

DomainPilot-style data mixture optimization runs between stages: a small proxy fine-tune over the candidate mixture distribution evaluates per-domain validation loss, and the mixture weights are updated to minimize worst-domain loss.

### Why it is new
Existing multi-stage methods (Qwen-Audio-3.0, CosyVoice 3) are modality-specific and do not switch optimizers or precision between stages. TURBO-FINETUNE is the first design that unifies stage-gated precision, optimizer, layer sampling, backprop strategy, and data mixture into one pipeline with explicit handoff criteria between stages.

### Implementation note for grim
TURBO-FINETUNE is an orchestrator that lives in `grim-garage`. Each stage is a config entry: precision, optimizer variant, frozen layer set, and data mixture weights. The handoff criteria are per-stage validation loss deltas; `grim-garage` already tracks job status and metrics, so stage transitions fit the existing lifecycle model. LISA-style stochastic layer sampling is a per-step random layer mask applied in `grim-autograd`'s backward hook. The distillation-specific losses for video and audio require small host-side models; those can be initialized from `grim-models` once the relevant teacher checkpoints are available. No new GPU kernel is required.

---

## Method 9 — SCALE-ECHO: Compressed-Activation Echo-State Fine-Tuning

**Frames:** Standard LLM training method for consumer GPUs. Targets memory-efficient full-parameter or near-full-parameter fine-tuning without adapters.

### Composed from
- ToSR-LoRA CompoundWord Bridge concept (res2)
- GradLite aggressive activation discarding (res3)
- LOMO fused backprop + immediate weight update (res3)
- MeZO zeroth-order on-device fine-tuning (res3)
- Online Subspace Descent without SVD (res2)
- Adam-mini block-level learning rates (res3)

### Novel formulation
SCALE-ECHO replaces backpropagation with a hybrid echo-state update:

1. **Forward pass only:** activations are computed and immediately discarded after a lightweight low-rank projection. No activation checkpointing is needed because activations are never stored.
2. **Echo-state update:** a random, fixed, low-rank projection matrix `E ∈ R^{d x r}` is applied to the input. The projected echo state `h_t = E^T x_t` is used to estimate the gradient direction via finite differences (MeZO-style), but only in the `r`-dimensional subspace.
3. **Parameter-free online subspace descent:** the optimizer state is eliminated entirely. M+Adam's additive-multiplicative split runs on FP4 master weights with no persistent momentum or variance buffers. The low-rank projected update is applied directly to the weights each step. This replaces SCALE-ECHO's original O(r²) optimizer state with O(1) scalars.
4. **Block-diagonal coupling:** recursive block-diagonal coupling from RBDC is applied to the projection matrix `E`. Each structural block (Q/K/V/O/MLP) gets its own independent rank-r subspace, so the total effective state is `num_blocks * r` scalars instead of one global `d * r` matrix. For a 7B model with ~100 blocks and r=8, this is ~800 scalars total.
5. **Activation compression:** CARE-LoRA-style compressed activations are stored during the forward pass. The reconstruction adapter is itself a tiny rank-r adapter, and it is updated by zeroth-order perturbation (two forward passes, one perturbed, one clean) — no backward pass through the reconstruction error. This keeps the no-tape invariant intact, but compounds MeZO's variance because two parameter sets are now perturbed jointly: the block-diagonal projection matrices `E` and the reconstruction adapters.

### Why it is new
MeZO removes activation storage but is prohibitively slow and still needs an optimizer. LOMO removes optimizer state but still needs activations during backprop. M+Adam enables low-precision training but still stores master weights and activations. RBDC reduces FLOPs but does not address optimizer state. SCALE-ECHO removes all three: it is the first method to combine parameter-free zeroth-order optimization, M+Adam low-precision master weights, block-diagonal subspace projection, and compressed activations in a single full-parameter fine-tuning loop. The result is full-parameter training at ~18 GB VRAM for a 7B model on 24 GB consumer hardware.

### Implementation note for grim
SCALE-ECHO is a separate `echo` training mode in `grim-cli` that bypasses `grim-autograd`'s tape recording. The MeZO-style finite-difference estimator is parameter-free, so no optimizer state is allocated. The rank-r projection matrix `E` is block-diagonal: one small matrix per structural block in `grim-nn`, generated at init from a single host-side RNG seed and never stored in VRAM beyond the per-block scalars. M+Adam's low-precision master weight path reuses the existing FP4 quantization plumbing in `grim-backend-rocm`; the additive-multiplicative split is applied per block by the echo update loop on host. Block-level LR assignment reuses `grim-autograd/src/lr_schedule.rs`. CARE-LoRA-style activation compression is a small rank-r adapter inserted in each block's forward pass. It is updated by zeroth-order perturbation on the reconstruction loss, not by backprop through the tape. Two forward passes per step (clean + perturbed) are required for the ZO estimate. This doubles the forward cost and compounds MeZO variance because both the block-diagonal projection `E` and the reconstruction adapter are perturbed jointly. This is still lowest priority because the convergence guarantee is weaker than adapter-based methods, but the memory ceiling is now hard-fit for 24 GB consumer GPUs.

---

## Method 10 — TRI-MODAL GRAD: Cross-Modal Gradient Surgery With Modality-Aware Noise Filtering

**Frames:** Universal OMNI training optimizer. Targets the gradient-level interaction between text, audio, and visual modalities during joint training.

### Composed from
- Adam under heavy-tailed noise convergence theory (res7)
- PsiLogic chaos-aware active cancellation (res7)
- M+Adam low-precision scale/direction split (res7)
- Utility-Aware Multimodal Contrastive (res7)
- DomainPilot domain-level loss guidance (res7)
- TOPS first-principles token preservation (res7)

### Novel formulation
TRI-MODAL GRAD operates on the joint gradient before any optimizer step:

1. **Modality noise profiling:** the joint gradient `g_joint = g_text + g_audio + g_visual` is split by modality. Each modality gradient’s tail index is estimated. If one modality’s tail index exceeds `tau = 1.5`, its gradient is routed through the additive-multiplicative branch (M+Adam) with high-precision scale tracking; other modalities use standard multiplicative updates.
2. **Gradient surgery:** if cosine similarity between any two modality gradients is negative, the smaller-norm gradient is projected away from the larger-norm gradient (Gradient Surgery / PCGrad). This prevents modalities from pulling the model in opposite directions during early unstable training.
3. **Cross-modal token preservation:** TOPS-style preservation sets are computed per modality. Parameters whose gradients correspond to preservation-set tokens receive a 2x LR boost; parameters outside preservation sets are updated at 0.5x LR.
4. **Domain-aware noise filter:** DomainPilot-style domain loss monitoring detects when one modality’s domain is causing outlier gradients. If domain loss for audio spikes > 3 std above its moving average, audio gradients are clipped to their 90th percentile norm for that step.

### Why it is new
Gradient surgery (PCGrad) exists for multi-task learning, and noise-filtering exists for single-modality training, but no existing method combines per-modality tail-aware routing, cross-modal gradient projection, token-preservation-aware LR scaling, and domain-triggered noise filtering in a single gradient pre-processing step.

### Caveats
TRI-MODAL GRAD inherits PsiLogic's chaos-gating overhead. The source paper reports a 20–80% wall-clock slowdown versus plain Adam from dual-EMA gradient statistics and active-cancellation projection. TRI-MODAL GRAD adds PCGrad projection, domain-loss spike detection, and preservation-set LR computation on top of that baseline. The host-side preprocessing cost is proportional to batch size and modality count. On consumer GPUs where CPU is the bottleneck, this can dominate training step time. Benchmark before scaling to full model size.

The tail-index threshold `tau = 1.5` is applied to a per-step Hill estimator with no smoothing, so it will flap under normal gradient noise. This is the same defect as OMNIGRAD's router, but TRI-MODAL GRAD does not inherit OMNIGRAD's infrastructure — it reimplements the estimator independently, so it carries the same instability without the EMA fix.

Step ordering is also unaddressed: step 2's PCGrad projection alters the "losing" modality's gradient direction, and step 3 then applies a 2x/0.5x LR multiplier based on TOPS preservation sets — but step 3's multiplier is computed from the post-projection gradient, not the original. If a preservation-set token's gradient was already shrunk by step 2, the 2x boost is boosting an already-corrected direction. The doc does not state whether this is intended or an artifact of pipeline order.

Step 4's clip threshold is a single-step 90th percentile norm computed from the very batch being clipped. That is an unstable construction: the clip bound moves with the data it is clipping. A moving-average percentile or a fixed norm bound would be stable; this is not.

### Implementation note for grim
TRI-MODAL GRAD is a gradient pre-processing hook in `grim-garage`'s training loop, not a new optimizer. The tail-index estimator, PCGrad projection, domain-loss spike detector, and preservation-set LR multipliers are all host-side scalars applied to the parameter groups before `grim-autograd`'s optimizer step. The preservation set computation requires attention entropy from the forward pass; if that is not already logged, add a lightweight entropy reduction in `grim-nn`'s attention layer. No new GPU kernel is required, but the preprocessing does add CPU overhead proportional to batch size and modality count.

---

## Method 11 — SPECTRAL-QLORA: Quantized LoRA With Orthogonal Subspace Initialization and Muon Optimizer

**Frames:** Standard LLM and visual LLM training optimizer. Targets 4-bit/8-bit adapter training with faster convergence than QLoRA + AdamW.

### Composed from
- QLoRA frozen 4-bit base + BF16 adapters (res6, res2)
- Muon spectral regularization optimizer (res3)
- SOUL EATER semi-orthogonal basis initialization for adapters (res6)
- LoRA+ different LR for A and B matrices (res6)
- CARE-LoRA compressed activation reconstruction (res2)
- M+Adam low-precision adapter optimizer (res7)
- LoRA-Muon spectral steepest descent on the low-rank manifold (2606.12921)

### Novel formulation
SPECTRAL-QLORA changes three things at once:

1. **Adapter initialization:** A and B matrices are initialized so that the product AB is a semi-orthogonal matrix in the dominant subspace (SOUL EATER’s initialization principle). This eliminates the need for a warmup phase; training starts at the optimal manifold.
2. **Optimizer:** Muon replaces AdamW for adapter training. Muon’s Newton-Schulz orthogonalization step keeps the adapter direction matrix well-conditioned without second-moment storage. Adapter magnitude is updated with a separate sign-SGD step (SOUL EATER’s 1-bit approach), requiring zero moment memory for magnitudes.
3. **Activation compression:** CARE-LoRA-style compressed activations are stored during the forward pass. The reconstruction matrix is itself a tiny LoRA adapter trained jointly, so the compression adapts to the data distribution.

LoRA-Muon (2606.12921) derives the low-rank spectral update from first principles and shows that optimal learning rates transfer across rank, width, and factor-rescaling. SPECTRAL-QLORA should adopt LoRA-Muon's split weight-decay rule and spectral-steepest-descent update rather than inventing an ad hoc `eta_B = gamma * eta_A` constraint. If orthogonal initialization or activation compression is added on top, they must be validated as additive improvements, not assumed to compose for free.

### Why it is new
QLoRA + AdamW is the current standard. SPECTRAL-QLORA replaces AdamW with Muon, initializes adapters on the optimal orthogonal manifold, and compresses activations during adapter training. This specific combination — spectral optimizer + orthogonal init + activation compression in a single LoRA recipe — is not the same as LoRA-Muon (2606.12921), which derives the spectral update but does not add orthogonal initialization or activation compression. The convergence improvement over QLoRA + AdamW is supported by LoRA-Muon's spectral-steepest-descent derivation and its TinyShakespeare benchmark (rank-32 LoRA-Muon beats the dense baseline in seed-averaged loss), but the exact multiplier depends on rank, width, and task, so any quantitative claim should be tied to a specific benchmark rather than stated as a universal 2x.

### Caveats
The circularity in the original formulation (hard `eta_B = gamma * eta_A` constraint stacked on top of Muon's natural behavior) is resolved by adopting LoRA-Muon's (2606.12921) spectral-steepest-descent derivation and split weight-decay rule instead of an ad hoc constraint. The remaining open question is whether orthogonal initialization and activation compression compose additively with LoRA-Muon, or whether they alter the low-rank manifold in ways that require separate validation. Treat those two additions as independent experimental claims, not free extensions of the spectral-update result.

### Implementation note for grim
SPECTRAL-QLORA is an optimizer + init swap in `grim-garage`, similar to SCYTHE1. The orthogonal adapter initialization reuses `grim-quant/src/soul_eater.rs` for Gram-matrix-based orthogonality checks; apply it once at adapter creation rather than per-step. Muon replaces the existing optimizer variants in `grim-autograd/src/adamw.rs` for adapter-only parameter groups. The activation reconstruction adapter is structurally a small rank-r linear projection and can reuse `SoulEaterAdapter`'s shape logic with a lower rank. M+Adam's low-precision update rule is advisory here: SPECTRAL-QLORA trains adapters in BF16 during SFT, so the additive-multiplicative split is deferred to the final production quantization step.

---

## Method 12 — FUSED-QUANT-BWD: Fused Dequant + Backprop + Optimizer Step for Quantized Training

**Frames:** Universal OMNI and standard LLM training kernel. Targets the backward pass of quantized training.

### Composed from
- M+Adam additive-multiplicative split for low-precision training (res7)
- LOMO fused backprop + immediate weight update (res3)
- ToSR-LoRA CompoundWord Bridge concept (res2)
- Unsloth custom backward kernels (res6)
- Dual-Precision FP MAC datapath (res4)
- ELUTQ LUT-GEMM (res4)

### Novel formulation
FUSED-QUANT-BWD is a single HIP kernel that fuses three operations:

1. **Dequantize only the scale:** weights arrive as NF4/MXFP4. The kernel decodes scale factors to FP16 in LDS but keeps the 4-bit mantissas in registers. Matmul is performed in the 4-bit domain using WMMA or custom wave64 dot-product, accumulating to FP16 only at the tile level.
2. **Gradient computation in the compact subspace:** gradients are computed against the quantized forward result, not the dequantized result. This is the same insight as ToSR-LoRA’s CompoundWord Bridge: the gradient flow stays in the lowest common subspace, never expanding to full FP16 for the full tensor.
3. **Optimizer step fused:** M+Adam-style additive-multiplicative update is applied in the same kernel. Scale parameters are updated in BF16; direction parameters in FP8 with per-block scales. The update is applied in-place to the quantized weight storage with scale-bump propagation.

The kernel is launched per tile, not per layer, and supports arbitrary quantization formats via a format descriptor passed at launch time.

### Why it is new
Unsloth fuses forward operations; LOMO fuses backprop and weight update; M+Adam defines the low-precision update rule; ToSR-LoRA defines the compact gradient subspace. None of them combine all four into a single fused backward kernel that never materializes the full dequantized gradient.

### Caveats
The repo already has fused dequant-GEMM plumbing (`fusion.rs`, `FusedDequantGemmConfig`) and an MXFP4 emulation path, but the existing `grim_fused_dequant_gemm_f16` launch path is flagged elsewhere as dead/orphaned code with no corresponding `Storage` variant in `dtype.rs`. Treat the current plumbing as a partial scaffold, not a working foundation — it will need repair or replacement before this kernel can land.

Computing gradients directly against the quantized forward result, without an STE or fake-quantized weight path, means quantization error is not just a forward-path artifact but is differentiated into the gradient estimate. The doc does not discuss whether this introduces bias in the gradient direction versus standard backprop through fake-quantized weights, which is the established approach precisely because it bounds the quantization error to the forward pass. This is a real numerics risk, not a performance concern.

Scale-bump propagation is also mechanically underspecified: if a tile's quantization scale is updated inside the same kernel invocation that computed the gradient against the old scale, the update may be stale by one step. The doc does not clarify whether the scale update is staged (written after all tiles finish) or inline (potentially racing tile-local reads).

### Implementation note for grim
FUSED-QUANT-BWD is a HIP kernel in `grim-backend-rocm`. The repo already has fused dequant-GEMM plumbing (`fusion.rs`, `FusedDequantGemmConfig`) and an MXFP4 emulation path (`QuantMode::MxFp4Emulated`), but the existing `grim_fused_dequant_gemm_f16` launch path is dead/orphaned code with no corresponding `Storage` variant in `dtype.rs`; audit and repair that plumbing first. The WMMA/wave64 dot-product on 4-bit mantissas, scale decoding to FP16 in LDS, and per-tile accumulation match the intended source shape once the launch path is wired. The optimizer-step fusion is new: scale-bump propagation in BF16 and direction updates in FP8 with per-block scales require a small post-GEMM HIP kernel or JIT fragment. The format-descriptor dispatch can reuse `QuantMode` for the current supported types.

---

## Summary Table

|| Method | Frame | Novel combination | Expected gain |
||--------|-------|-------------------|---------------|
|| OMNIGRAD | Universal OMNI | Per-modality noise routing + per-layer LR + phase gating + low-rank state | Convergence stability across modalities; optimizer memory near zero; throughput unvalidated (PsiLogic 20–80% overhead); blocked on unbuilt modality-annotation pipeline |
|| SCYTHE1 | Standard/Visual LLM PEFT | SOUL EATER adapter + Newton-Schulz + inverse-FIM preconditioning in adapter subspace | Faster convergence than SOUL EATER on non-convex adapter losses; same ~1 MB/adapter-layer state |
|| VLLM-OPT | Visual LLM | TOPS training-time pruning + RotateKV/K8V4 inference-time KV compression + ReToken | Up to 60–80% activation reduction + up to 3x KV size reduction; stages trained separately, not end-to-end differentiable |
|| MM-GRPO | Multimodal LLM | GRPO + cross-modal credit assignment + modality-aware mixture | First multimodal RL with per-modality advantage normalization; dominance criterion is asserted, not derived |
|| OMNILO-PRUNE | Universal OMNI | Per-modality rank budget + subspace rotation + activation compression + precision routing | Consumer-GPU omni fine-tuning; VRAM reduction claimed 26–37% (unsourced); depends on OMNIGRAD's noise router |
|| KV-OMNI | Multimodal LLM | RotateKV + KVTuner + RocketKV + JoLT + Dual-Signal + cross-modal eviction | Unified text/audio/video KV compression with persistent on-disk contract |
|| CONTRAST-OMNI | Universal OMNI | Utility-weighted pairs + hierarchical alignment + Fréchet matching + channel-smoothed temp | Better omni representations from the same pretraining compute; two rescaling terms can interact unpredictably |
|| TURBO-FINETUNE | Visual/Multimodal LLM | Stage-gated precision/optimizer/layers/data mixture + distillation | Production-ready multimodal fine-tuning pipeline; value is inherited from its dependencies |
|| SCALE-ECHO | Standard LLM | Echo-state subspace estimation + block-diagonal projection + M+Adam FP4 master weights + parameter-free ZO + activation compression | Near-zero optimizer memory; two forward passes per step and compounded MeZO variance; research curiosity, not roadmap item |
|| TRI-MODAL GRAD | Universal OMNI | Tail-aware routing + gradient surgery + token-preservation LR + domain noise filter | Stable joint training of text+audio+visual without modality dominance; all four mechanisms have identified uncorrected defects |
|| SPECTRAL-QLORA | Standard/Visual LLM | Orthogonal adapter init + Muon + sign-SGD magnitudes + CARE-LoRA activations | 2x faster convergence than QLoRA+AdamW is an unsourced hypothesis, not a result |
|| FUSED-QUANT-BWD | Universal OMNI/LLM kernel | Fused dequant+backprop+optimizer in compact subspace via WMMA | Minimal materialization; conditionally viable once dead launch path repaired and STE added |

---

## Recommended Implementation Order for grim

1. **FUSED-QUANT-BWD** — direct kernel work, immediate VRAM reduction, aligns with existing ROCm backend
2. **SCYTHE1** — adapter optimizer extension in grim-garage; small r×r FIM/preconditioning ops, no new training kernels
3. **KV-OMNI** — extends grim-kvquant with persistent layout + cross-modal policy
4. **OMNIGRAD** — host-side scheduler + GPU-side parameter-group update, no new kernels
5. **SPECTRAL-QLORA** — adapter init + optimizer swap in grim-garage
6. **VLLM-OPT** — training-time token pruning + inference-time KV compression for visual LLMs
7. **TURBO-FINETUNE** — orchestrator/pipeline layer in grim-garage
8. **OMNILO-PRUNE** — rank budget allocator + adapter compression for consumer GPUs
9. **MM-GRPO** — RL post-training layer for multimodal models
10. **CONTRAST-OMNI** — pretraining objective for omni encoders
11. **TRI-MODAL GRAD** — gradient pre-processing hook in training loop
12. **SCALE-ECHO** — experimental zeroth-order subspace training for extreme memory limits

---

## References by Method

### OMNIGRAD
- 2607.10611 M+Adam
- 2607.16268 PsiLogic
- 2607.09266 LionVote
- 2607.27383 Adam heavy-tailed noise
- 2505.16363 AdamS
- 2510.22467 GradLite
- 2509.24406 Muon
- 2408.12857 Online Subspace Descent

### SCYTHE1
- 2509.24406 Muon
- res6 SOUL EATER
- 2510.22467 GradLite
- 2410.16029 Natural GaLore
- 2408.12857 Online Subspace Descent
- 2303.08399 LOMO

### VLLM-OPT
- 2602.14236 Dual-Signal KV-Cache
- 2606.27161 TOPS
- 2607.28627 ReToken
- 2607.13500 Attention-Free Token Reduction
- 2509.23663 HIVTP
- 2603.24680 ReDiPrune
- 2607.12550 JoLT
- 2501.16383 RotateKV

### MM-GRPO
- 2402.03300 GRPO
- 2607.28590 Multimodal OPD
- 2606.07610 LEAF
- 2607.21550 X3-OPD
- 2605.31455 DRIFT
- 2607.22769 DomainPilot

### OMNILO-PRUNE
- res2 ToSR-LoRA
- 2501.04315 RoRA
- 2409.11220 PE-LoRA
- 2307.05695 ReLoRA
- 2402.09353 DoRA
- 2403.17919 LISA
- res2 CARE-LoRA
- 2607.10611 M+Adam

### KV-OMNI
- 2501.16383 RotateKV
- 2502.04420 KVTuner
- 2502.14051 RocketKV
- 2607.12550 JoLT
- 2602.14236 Dual-Signal KV
- 2602.04657 TRIO
- res5 PolyKV

### CONTRAST-OMNI
- 2605.28733 Utility-Aware Contrastive
- 2604.16247 HILBERT
- 2607.06027 Fréchet Distance Loss
- 2211.10438 SmoothQuant
- res7 architectural diversity + foundation scaling

### TURBO-FINETUNE
- 2607.23938 Qwen-Audio-3.0 multi-stage
- 2506.03077 StreamBP
- 2403.17919 LISA
- 2512.14237 Ladder Side Tuning
- 2607.22769 DomainPilot
- 2607.15105 HGA + QLoRA long-context
- 2511.19923 counterfactual distillation
- 2607.18662 staged depth-pruning distillation

### SCALE-ECHO
- res2 ToSR-LoRA CompoundWord Bridge
- 2510.22467 GradLite
- 2303.08399 LOMO
- 2511.11362 MeZO
- 2408.12857 Online Subspace Descent
- 2406.16793 Adam-mini
- 2607.10611 M+Adam
- 2606.14970 AdaNAGED parameter-free ZO
- 2605.23656 RBDC block-diagonal coupling

### TRI-MODAL GRAD
- 2607.27383 Adam heavy-tailed noise
- 2607.16268 PsiLogic
- 2607.10611 M+Adam
- 2605.28733 Utility-Aware Contrastive
- 2607.22769 DomainPilot
- 2606.27161 TOPS

### SPECTRAL-QLORA
- res6 QLoRA baseline
- 2509.24406 Muon
- res6 SOUL EATER basis init
- 2402.12354 LoRA+
- res2 CARE-LoRA
- 2607.10611 M+Adam

### FUSED-QUANT-BWD
- 2607.10611 M+Adam
- 2303.08399 LOMO
- res2 ToSR-LoRA CompoundWord Bridge
- res6 Unsloth custom kernels

---

## Paper Review Update — 2603.15031v1: Attention Residuals (Kimi Team, Mar 2026)

**Review scope:** standalone review of `old/res7/2603.15031v1.pdf`. Not included in the main 12-method synthesis above; kept separate per request.

### Paper summary
Standard residual connections accumulate all prior layer outputs with fixed unit weights, causing hidden-state magnitude growth O(L) with depth and dilution of early-layer contributions. The authors propose **Attention Residuals (AttnRes)**, which replaces fixed accumulation with learned softmax attention over preceding layer outputs. Each layer l has one learned pseudo-query `w_l ∈ R^d`; keys/values are prior layer outputs; attention weights `α_{i→l} = softmax(exp(q_l·k_i) / RMSNorm(k_i))` select which earlier representations matter most for the current layer. Two variants:

- **Full AttnRes:** attend over all L prior layers. O(L²d) compute, O(Ld) memory.
- **Block AttnRes:** group L layers into N blocks (N≈8). Reduce memory/communication from O(Ld) to O(Nd). This is the practical variant.

System innovations: two-phase computation (parallel inter-block attention + sequential intra-block with online softmax merge), cross-stage caching under pipeline parallelism (reduces per-transition comms from O(C) to O(P), V× improvement), sequence-sharded prefilling for long context.

Key empirical results: Block AttnRes with N=8 matches a baseline trained with 1.25× more compute; inference overhead <2%; training overhead <4% under pipeline parallelism; 48B-parameter Kimi Linear pre-training on 1.4T tokens shows bounded output magnitudes and more uniform gradient norms across depth.

### What is viable for grim
- **Block AttnRes as drop-in residual replacement.** One RMSNorm + one d-dimensional pseudo-query vector per layer. Negligible parameter overhead. Maps directly onto grim’s transformer/MLP/Mamba/vision/audio/diffusion layer stacks.
- **Gradient flow improvement.** Mitigates PreNorm dilution. Pairs naturally with SCYTHE1: AttnRes improves the gradient landscape; SCYTHE1 adapts the optimizer to it.
- **Cross-stage caching for pipeline parallelism.** Reduces redundant block-transmission under interleaved 1F1B schedules. grim already has `grim-disagg` and `grim-kvtransport` with `PoolRole::Colocated/Prefill/Decode`. Block-caching maps directly onto that architecture.
- **Two-phase computation on ROCm.** Phase 1 is a single batched query×block-KV matmul per block. Phase 2 is elementwise online-softmax merge, kernel-fusable with RMSNorm. RDNA2/3 batched small-geamm + elementwise-fuse is a known WMMA pattern.
- **Sequence-sharded prefilling.** Shards block representations along sequence dimension across TP devices, merges via reduce-scatter + online softmax. Aligns with grim’s planned TP/PP and KV-quant work.

### Not viable / defer
- **Full AttnRes at scale.** The paper itself identifies O(Ld) activation retention under activation recomputation + pipeline parallelism as the blocker. Block AttnRes exists to solve this. Implement Block first; revisit Full only when pipeline-parallel overhead is fully amortized.
- **Online softmax inference optimization.** The two-phase inference strategy assumes fixed block counts and online-softmax kernels. Baseline is already <2%; premature optimization.

### Implementation plan
**Phase 1 — Model integration (no kernel work)**
1. Add `BlockAttnResConfig { block_size: usize, init_zero: bool }` to `grim-nn` layer config.
2. Implement `block_attn_res(blocks, partial_block, proj, norm)` as a pure Rust fn in `grim-nn/src/block_attn_res.rs`, using existing `grim-nn` matmul + RMSNorm.
3. Inject Block AttnRes into transformer block forward before attn+MLP, maintaining `blocks: Vec<Tensor>` + `partial_block: Option<Tensor>` state across layer calls.
4. Add unit tests: single-block = standard residual parity; N=8 = learned depth attention; init-zero = uniform start.

**Phase 2 — Pipeline parallelism**
5. Implement cross-stage caching in `grim-disagg`/`grim-kvtransport`: cache block summaries at each physical stage, transmit only incremental blocks at stage transitions.
6. Validate V× communication reduction against naive full-history broadcast under simulated 1F1B.

**Phase 3 — Kernel fusion (ROCm)**
7. Fuse Phase 2 online-softmax merge with RMSNorm in `grim-backend-rocm`. Profile WMMA small-geamm for Phase 1 batched queries.
8. Validate inference latency overhead <2% and training overhead <4% against standard residuals on RDNA2/CDNA.

**Phase 4 — Scale validation**
9. Run scaling-law sweep on 5 model sizes (matching paper Table 2) or proxy with 1B–7B checkpoints if 48B is infeasible on consumer GPUs.
10. Pair with SCYTHE1: measure whether gradient-norm uniformity from AttnRes reduces SCYTHE1’s per-layer LR search space.

**Integration note:** SCYTHE1 pairs with Block AttnRes. AttnRes improves the gradient landscape; SCYTHE1’s per-layer subspace + eigendecomposition adapts to it. Implement Block AttnRes first (Phase 1), then SCYTHE1.

**Priority:** implement after SCYTHE1 in the existing order, or parallel — they touch different crates (`grim-nn` vs `grim-autograd`/`grim-garage`).
