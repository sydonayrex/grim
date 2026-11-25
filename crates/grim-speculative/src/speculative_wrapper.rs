//! `SpeculativeCausalLm` - the default-on wrapper that turns a plain `CausalLm` into a speculatively-accelerated one.
//! §5.3.

use std::sync::{Arc, Mutex};

// Per-process env latches: re-reading these via std::env::var happened on every
// decode step on the per-token path. Env is not mutated at runtime; read once.
fn mtp_1pass_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("GRIM_MTP_1PASS").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}
fn decode_graph_m2_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("GRIM_DECODE_GRAPH_M2").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}

fn draft_chain_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("GRIM_MTP_DRAFT_CHAIN").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

fn replay_state_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("GRIM_SPEC_REPLAY_STATE").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

fn spec_tree_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("GRIM_SPEC_TREE").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

use grim_core::error::{Error, Result};
use grim_core::model::AdapterHandle;
use grim_core::session::SessionT;
use grim_core::{CausalLm, Model, ModelConfig};
use grim_tensor::backend::{ElementwiseOps as _, MemoryOps as _};
use grim_tensor::{ArithType, Device, Tensor};

use crate::confidence_head::ConfidenceHead;
use crate::confidence_scheduler::{ConfidenceScheduler, SpeculationConfig, ThroughputProfile};
use crate::depth_tuner::SpeculativeDepthPidController;
use crate::draft_backbone::{DraftBackbone, DraftBlock};
use crate::markov_head::MarkovHead;
use crate::native_mtp::NativeMtp;

/// Strategy choice at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Plain autoregressive fallback — no draft bundle, no MTP heads.
    Plain,
    /// DSpark path: draft + Markov + confidence heads attached.
    DSpark,
    /// Native MTP path: target exposes model-native prediction heads.
    NativeMtp,
}

/// Runtime speculative decoding telemetry (§5.3).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeculativeTelemetry {
    /// Active speculative decoding strategy (`plain`, `dspark`, or `native_mtp`).
    pub strategy: String,
    /// Exponential moving average of acceptance rate across decode steps.
    pub accept_rate_ema: f64,
    /// Total count of observed decode steps.
    pub steps_observed: u64,
    /// Total count of draft tokens proposed across all observed decode steps.
    pub total_drafted_tokens: u64,
    /// Total count of draft tokens accepted by target verification.
    pub total_accepted_tokens: u64,
    /// Minimum acceptance rate threshold before triggering draft adaptation.
    pub min_accept_rate: f64,
    /// Whether acceptance rate has drifted below the threshold warranting draft refresh.
    pub should_adapt: bool,
    /// Current PID-tuned draft depth for the DSpark path (None on other strategies).
    /// T2-4 follow-up: the depth tuner now drives the draft block length instead of the old.
    pub draft_depth_k: Option<u64>,
}

/// The wrapper: holds a target + the chosen strategy + bundle handles.
pub struct SpeculativeCausalLm {
    target: Box<dyn CausalLm>,
    strategy: Strategy,
    /// Native MTP target — None unless `strategy == NativeMtp`.
    mtp_target: Option<Arc<dyn NativeMtp>>,
    /// DSpark pieces — None unless `strategy == DSpark`.
    draft: Option<Arc<dyn DraftBackbone>>,
    markov: Option<Arc<dyn MarkovHead>>,
    confidence: Option<Arc<dyn ConfidenceHead>>,
    /// Confidence scheduler shared across DSpark sessions.
    scheduler: Mutex<ConfidenceScheduler>,
    /// PID depth controller driving the DSpark draft block length K.
    /// Fed by observed (accepted, proposed) counts each DSpark step.
    depth_tuner: Mutex<SpeculativeDepthPidController>,
    /// astrum P5: optional E/cost draft-source policy (output-neutral observation; default OFF).
    astrum_draft_policy: Option<Mutex<crate::draft_policy::DraftPolicy>>,
    /// Trigram suffix prompt-lookup drafter paired with `astrum_draft_policy`.
    astrum_suffix_drafter: Mutex<crate::suffix_drafter::SuffixDrafter>,
    /// Wall-clock cost of the most recently produced draft, used by the
    /// E/cost source policy (only read when that policy is enabled).
    astum_last_draft_ms: Mutex<f64>,
    /// Optional cost-aware verify-width controller (`GRIM_SPEC_WIDTH=1`).
    width_controller: Option<Mutex<crate::verify_width::WidthController>>,
    /// Pointer-commit replay window for recurrent state (`GRIM_SPEC_REPLAY_STATE=1`).
    replay_ring: Option<Mutex<crate::replay_state::ReplayRing<u32>>>,
    /// Optional GPU rejection-sampling verifier (Task 10.6). When set and the
    /// verify logits are ROCm-resident, the ratio test runs on device — no
    /// vocab-sized D2H copy. Errors fall back to the CPU verify loop.
    gpu_verifier: Option<Arc<dyn SpeculativeVerifier>>,
    /// Pending draft block from the previous NativeMtp step, batched with the
    /// next base token into a single trunk forward pass.
    pending_mtp_draft: Mutex<Option<crate::draft_backbone::DraftBlock>>,
    /// GRIM_MTP_DRAFT_CHAIN=1: a chained draft is enqueued device-side; the
    /// next verify consumes its token from pinned staging.
    chained_active: std::sync::atomic::AtomicBool,
    /// Optional m=2 DecodeGraph for 1-pass batched NativeMtp.
    #[allow(dead_code)]
    graph_m2: Mutex<Option<Box<dyn std::any::Any + Send>>>,
    /// Optional adaptive `may_verify` gate (`GRIM_SPEC_MAY_VERIFY=1`, default OFF).
    may_verify: Option<Mutex<crate::may_verify::MayVerifyGate>>,
}

/// Result of one lossless verification pass.
///
/// `accepted_count` is the number of draft tokens accepted (the bonus/residual
/// token is excluded). `emitted_tokens` is the exact token stream to emit:
/// the accepted drafts followed by one residual or bonus token. Empty means
/// the caller must sample from the returned logits itself (greedy path).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerificationOutcome {
    pub accepted_count: usize,
    pub emitted_tokens: Vec<u32>,
}

/// Backend-agnostic GPU verification hook (Task 10.6). Implemented by the
/// engine's ROCm verifier; the wrapper stays free of backend deps.
pub trait SpeculativeVerifier: Send + Sync {
    fn verify(
        &self,
        target_logits: &Tensor,
        draft_logits: &Tensor,
        target_row_offset: usize,
        vocab_size: usize,
        draft_tokens: &[u32],
        uniform_rands: &[f32],
    ) -> Result<VerificationOutcome>;
}

impl SpeculativeCausalLm {
    /// Construct with a plain autoregressive fallback strategy.
    pub fn plain(target: Box<dyn CausalLm>) -> Self {
        Self {
            target,
            strategy: Strategy::Plain,
            mtp_target: None,
            draft: None,
            markov: None,
            confidence: None,
            scheduler: Mutex::new(ConfidenceScheduler::new(
                ThroughputProfile::default(),
                SpeculationConfig::default(),
            )),
            depth_tuner: Mutex::new(SpeculativeDepthPidController::with_default_config()),
            astrum_draft_policy: crate::draft_policy::DraftPolicy::from_env().map(std::sync::Mutex::new),
            astrum_suffix_drafter: Mutex::new(crate::suffix_drafter::SuffixDrafter::new(8192)),
            astum_last_draft_ms: Mutex::new(0.0),
            width_controller: crate::verify_width::WidthController::from_env().map(std::sync::Mutex::new),
            replay_ring: replay_state_enabled()
                .then(|| Mutex::new(crate::replay_state::ReplayRing::new(16))),
            gpu_verifier: None,
            pending_mtp_draft: Mutex::new(None),
            chained_active: std::sync::atomic::AtomicBool::new(false),
            graph_m2: Mutex::new(None),
            may_verify: crate::may_verify::spec_may_verify_enabled()
                .then(|| Mutex::new(crate::may_verify::MayVerifyGate::with_defaults(4))),
        }
    }

    /// Construct wrapped around the DSpark strategy.
    pub fn with_dspark(
        target: Box<dyn CausalLm>,
        draft: Arc<dyn DraftBackbone>,
        markov: Arc<dyn MarkovHead>,
        confidence: Arc<dyn ConfidenceHead>,
        scheduler: ConfidenceScheduler,
    ) -> Self {
        Self::with_dspark_and_tuner(
            target,
            draft,
            markov,
            confidence,
            scheduler,
            SpeculativeDepthPidController::with_default_config(),
        )
    }

    /// Construct wrapped around the DSpark strategy with an explicit depth
    /// tuner (e.g. seeded from a saved `SpeculativeDepthPidConfig`).
    pub fn with_dspark_and_tuner(
        target: Box<dyn CausalLm>,
        draft: Arc<dyn DraftBackbone>,
        markov: Arc<dyn MarkovHead>,
        confidence: Arc<dyn ConfidenceHead>,
        scheduler: ConfidenceScheduler,
        depth_tuner: SpeculativeDepthPidController,
    ) -> Self {
        Self {
            target,
            strategy: Strategy::DSpark,
            mtp_target: None,
            draft: Some(draft),
            markov: Some(markov),
            confidence: Some(confidence),
            scheduler: Mutex::new(scheduler),
            depth_tuner: Mutex::new(depth_tuner),
            astrum_draft_policy: crate::draft_policy::DraftPolicy::from_env().map(std::sync::Mutex::new),
            astrum_suffix_drafter: Mutex::new(crate::suffix_drafter::SuffixDrafter::new(8192)),
            astum_last_draft_ms: Mutex::new(0.0),
            width_controller: crate::verify_width::WidthController::from_env().map(std::sync::Mutex::new),
            replay_ring: replay_state_enabled()
                .then(|| Mutex::new(crate::replay_state::ReplayRing::new(16))),
            gpu_verifier: None,
            pending_mtp_draft: Mutex::new(None),
            chained_active: std::sync::atomic::AtomicBool::new(false),
            graph_m2: Mutex::new(None),
            may_verify: crate::may_verify::spec_may_verify_enabled()
                .then(|| Mutex::new(crate::may_verify::MayVerifyGate::with_defaults(4))),
        }
    }

    /// Construct wrapped around native MTP — the zero-config path.
    pub fn with_native_mtp(target: Box<dyn CausalLm>, mtp_target: Arc<dyn NativeMtp>) -> Self {
        Self {
            target,
            strategy: Strategy::NativeMtp,
            mtp_target: Some(mtp_target),
            draft: None,
            markov: None,
            confidence: None,
            scheduler: Mutex::new(ConfidenceScheduler::new(
                ThroughputProfile::default(),
                SpeculationConfig::default(),
            )),
            depth_tuner: Mutex::new(SpeculativeDepthPidController::with_default_config()),
            astrum_draft_policy: crate::draft_policy::DraftPolicy::from_env().map(std::sync::Mutex::new),
            astrum_suffix_drafter: Mutex::new(crate::suffix_drafter::SuffixDrafter::new(8192)),
            astum_last_draft_ms: Mutex::new(0.0),
            width_controller: crate::verify_width::WidthController::from_env().map(std::sync::Mutex::new),
            replay_ring: replay_state_enabled()
                .then(|| Mutex::new(crate::replay_state::ReplayRing::new(16))),
            gpu_verifier: None,
            pending_mtp_draft: Mutex::new(None),
            chained_active: std::sync::atomic::AtomicBool::new(false),
            graph_m2: Mutex::new(None),
            may_verify: crate::may_verify::spec_may_verify_enabled()
                .then(|| Mutex::new(crate::may_verify::MayVerifyGate::with_defaults(4))),
        }
    }

    /// Construct from a target + optional bundle or native MTP.
    /// Selects strategy automatically.
    pub fn auto_with_native_mtp(
        target: Box<dyn CausalLm>,
        draft: Option<Arc<dyn DraftBackbone>>,
        markov: Option<Arc<dyn MarkovHead>>,
        confidence: Option<Arc<dyn ConfidenceHead>>,
        native_mtp: Option<Arc<dyn NativeMtp>>,
        is_weight_streaming_active: bool,
        available_vram_bytes: Option<usize>,
    ) -> Self {
        if let (Some(draft), Some(markov), Some(confidence)) =
            (draft.as_ref(), markov.as_ref(), confidence.as_ref())
        {
            if is_weight_streaming_active {
                if let Some(available_vram) = available_vram_bytes {
                    let estimated_size = draft.estimated_footprint_bytes();
                    if estimated_size > available_vram {
                        if let Some(mtp) = native_mtp {
                            return Self::with_native_mtp(target, mtp);
                        }
                        return Self::plain(target);
                    }
                }
            }
            Self::with_dspark(
                target,
                draft.clone(),
                markov.clone(),
                confidence.clone(),
                ConfidenceScheduler::new(
                    ThroughputProfile::default(),
                    SpeculationConfig::default(),
                ),
            )
        } else if let Some(mtp) = native_mtp {
            Self::with_native_mtp(target, mtp)
        } else {
            Self::plain(target)
        }
    }

    /// Construct from a target + optional bundle. Selects strategy automatically.
    pub fn auto(
        target: Box<dyn CausalLm>,
        draft: Option<Arc<dyn DraftBackbone>>,
        markov: Option<Arc<dyn MarkovHead>>,
        confidence: Option<Arc<dyn ConfidenceHead>>,
        is_weight_streaming_active: bool,
        available_vram_bytes: Option<usize>,
    ) -> Self {
        Self::auto_with_native_mtp(
            target,
            draft,
            markov,
            confidence,
            None,
            is_weight_streaming_active,
            available_vram_bytes,
        )
    }

    /// Attach a GPU rejection-sampling verifier (Task 10.6). Called by the
    /// engine when the target runs on ROCm; the wrapper itself stays
    /// backend-agnostic. `None` (default) keeps the CPU verify loop.
    pub fn set_gpu_verifier(&mut self, verifier: Arc<dyn SpeculativeVerifier>) {
        self.gpu_verifier = Some(verifier);
    }

    /// Re-rank a drafted block into the highest-mass path of a CAST-style
    /// prefix-closed tree. Greedy-only: the selected tokens are proposals from
    /// the same per-position marginals, and verification still accepts only
    /// target argmax matches. Returns false when no usable tree was built.
    fn apply_tree_ranking(block: &mut DraftBlock, max_depth: usize, enabled: bool) -> bool {
        if !enabled || block.tokens.is_empty() || max_depth == 0 {
            return false;
        }
        let dims = block.base_logits.shape().dims().to_vec();
        if dims.len() != 2 || dims[1] == 0 || dims[0] == 0 {
            return false;
        }
        let vocab = dims[1];
        let depth = max_depth.min(dims[0]).min(block.tokens.len());
        let flat = match block.base_logits.to_vec_f32() {
            Ok(v) => v,
            Err(_) => return false,
        };
        let tree = crate::draft_tree::DraftTree::from_block_logits(
            &flat[..depth * vocab],
            vocab,
            crate::draft_tree::TreeBuildConfig {
                rank_cap: 4,
                max_nodes: (depth * 8).max(8),
                max_depth: depth,
            },
        );
        let Some((best, _)) = tree
            .nodes
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                a.log_mass
                    .total_cmp(&b.log_mass)
                    .then_with(|| ib.cmp(ia))
            })
        else {
            return false;
        };
        let path = tree.path_to(best);
        if path.is_empty() || path.len() > depth {
            return false;
        }
        block.tokens = path;
        block.confidence = vec![1.0; block.tokens.len()];
        true
    }

    /// Request-scoped fallback controller, lazily installed in the session.
    /// A new session starts with `None`, which resets the controller without
    /// any shared per-model state or cross-request contamination.
    fn spec_fallback(session: &mut dyn SessionT) -> Option<&mut crate::may_verify::SpecFallback> {
        let needs_init = session
            .spec_state()
            .and_then(|s| s.downcast_ref::<crate::may_verify::SpecFallback>())
            .is_none();
        if needs_init {
            if let Some(controller) = crate::may_verify::SpecFallback::from_env() {
                session.set_spec_state(Box::new(controller));
            }
        }
        session
            .spec_state_mut()?
            .downcast_mut::<crate::may_verify::SpecFallback>()
    }

    /// Cost-aware width selection; identity when the env gate is off.
    fn choose_verify_width(&self, confidence: &[f32], proposed: usize, ctx_len: usize) -> usize {
        match self.width_controller.as_ref() {
            Some(c) => c
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .choose(confidence, proposed, ctx_len),
            None => proposed,
        }
    }

    /// Feed the online depth tuner and (when enabled) the E/cost source policy
    /// after a native-MTP verification round.
    fn record_native_outcome(
        &self,
        session: &mut dyn SessionT,
        accepted: usize,
        drafted: usize,
        ctx_len: usize,
        used_suffix: bool,
        suffix_cost_ms: f64,
    ) {
        self.depth_tuner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .update(accepted, drafted);
        if let Some(fallback) = Self::spec_fallback(session) {
            fallback.observe(accepted, drafted);
        }
        if let Some(ring) = self.replay_ring.as_ref() {
            ring.lock()
                .unwrap_or_else(|e| e.into_inner())
                .commit(accepted);
        }
        if let Some(width) = self.width_controller.as_ref() {
            width
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .observe(accepted, drafted);
        }
        if let Some(policy) = self.astrum_draft_policy.as_ref() {
            let src = if used_suffix {
                crate::draft_policy::DraftSource::Suffix
            } else {
                crate::draft_policy::DraftSource::Mtp
            };
            let cost = if used_suffix {
                suffix_cost_ms
            } else {
                *self
                    .astum_last_draft_ms
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            };
            // A missing timing means the draft was produced before this step;
            // a positive floor keeps the EMA well-defined.
            let cost = if cost > 0.0 { cost } else { 1.0 };
            policy
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .observe_and_select(src, ctx_len, accepted + 1, cost);
        }
    }

    /// Append committed tokens to the suffix ring and draft from `SuffixDrafter`
    /// when `DraftPolicy::select(ctx_len)` chooses `DraftSource::Suffix`.
    pub fn try_draft_astrum_suffix(
        &self,
        committed_tokens: &[u32],
        ctx_len: usize,
        min_match: usize,
        max_len: usize,
    ) -> Option<crate::draft_backbone::DraftBlock> {
        let mut sfx = self
            .astrum_suffix_drafter
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        sfx.append(committed_tokens);
        let policy_mtx = self.astrum_draft_policy.as_ref()?;
        let mut pol = policy_mtx.lock().unwrap_or_else(|e| e.into_inner());
        if pol.select(ctx_len) == crate::draft_policy::DraftSource::Suffix {
            let blk = sfx.draft(min_match, max_len);
            if !blk.is_empty() {
                return Some(blk);
            }
        }
        None
    }

    /// Ask the E/cost policy for a suffix draft on this decode step and time it.
    /// The suffix drafter has no vocabulary logits, so it is greedy-only; the
    /// caller must not use it for stochastic verification.
    fn astrum_suffix_draft_for_step(
        &self,
        input_ids: &Tensor,
        ctx_len: usize,
        max_len: usize,
    ) -> Option<(crate::draft_backbone::DraftBlock, f64)> {
        let tokens: Vec<u32> = input_ids
            .to_vec_f32()
            .ok()?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let t0 = std::time::Instant::now();
        let block = self.try_draft_astrum_suffix(&tokens, ctx_len, 3, max_len)?;
        let cost_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Some((block, cost_ms))
    }

    /// Shared verify decision (Task 10.6): GPU rejection sampler when a
    /// verifier is attached and both logits tensors are ROCm-resident, CPU
    /// softmax loop otherwise (also the fallback on verifier error). The RNG
    /// is consumed by whichever arm runs; the caller passes a throwaway clone
    /// (session RNG state is unaffected either way).
    fn verify_accepted_count(
        &self,
        target_logits: &Tensor,
        draft_logits: &Tensor,
        target_row_offset: usize,
        vocab_size: usize,
        draft_tokens: &[u32],
        verify_len: usize,
        uniforms: &[f32],
        _rng: &mut grim_core::rng::SimpleRng,
        greedy: bool,
    ) -> Result<VerificationOutcome> {
        let gpu_ready = !greedy
            && self.gpu_verifier.is_some()
            && matches!(target_logits.device(), Device::Rocm(_))
            && matches!(draft_logits.device(), Device::Rocm(_))
            && uniforms.len() >= verify_len + 1;
        if let (true, Some(verifier)) = (gpu_ready, self.gpu_verifier.as_ref()) {
            match verifier.verify(
                target_logits,
                draft_logits,
                target_row_offset,
                vocab_size,
                draft_tokens,
                uniforms,
            ) {
                Ok(outcome) => return Ok(outcome),
                Err(e) => {
                    tracing::debug!("GPU speculative verify failed, falling back to CPU: {e}");
                }
            }
        }
        if greedy && matches!(target_logits.device(), Device::Rocm(_)) {
            let mut accepted_count = 0usize;
            let base_row = target_row_offset;
            // Reuse ONE device scratch row across the verify loop: the per-row
            // path in `argmax_logits_row` allocates a fresh [1, vocab] storage
            // per row (pool churn per iteration).
            let (_guard, dev, scratch) = match target_logits.device() {
                Device::Rocm(o) => {
                    // Guard must outlive the loop below: argmax kernels launch
                    // against the thread-current device.
                    let guard = grim_backend_rocm::DeviceGuard::set(*o as i32);
                    let dev = grim_backend_rocm::RocmDevice::shared(*o);
                    let scratch = grim_tensor::backend::MemoryOps::alloc_storage(
                        dev.as_ref(),
                        &grim_tensor::Shape::new(vec![1, vocab_size]),
                        grim_tensor::DType::F32,
                    )
                    .ok();
                    (guard, dev, scratch)
                }
                _ => unreachable!("greedy rocm branch checked device above"),
            };
            for i in 0..verify_len {
                let best_idx = match scratch.as_ref() {
                    Some(s) => {
                        let offset = (base_row + i) * vocab_size;
                        let mut row_idx = None;
                        if dev
                            .copy_slice_range(
                                s.as_ref(),
                                0,
                                target_logits.storage().as_ref(),
                                offset,
                                vocab_size,
                            )
                            .is_ok()
                        {
                            if let Ok(idx) = dev.argmax(s.as_ref()) {
                                row_idx = Some(idx as usize);
                            }
                        }
                        row_idx.unwrap_or(
                            grim_models_transformer::Qwen35::argmax_logits_row(
                                target_logits,
                                base_row + i,
                                vocab_size,
                            )? as usize,
                        )
                    }
                    None => grim_models_transformer::Qwen35::argmax_logits_row(
                        target_logits,
                        base_row + i,
                        vocab_size,
                    )? as usize,
                };
                if best_idx == draft_tokens[i] as usize {
                    accepted_count += 1;
                } else {
                    break;
                }
            }
            // Greedy verification is exact by construction: the emitted rows are
            // target argmaxes, which equal the accepted draft tokens.
            return Ok(VerificationOutcome {
                accepted_count,
                emitted_tokens: Vec::new(),
            });
        }
        let target_probs = target_logits.to_vec_f32()?;
        let draft_logits_cpu = if greedy {
            Vec::new()
        } else {
            draft_logits.to_vec_f32()?
        };
        let mut accepted_count = 0usize;
        let mut emitted_tokens: Vec<u32> = Vec::with_capacity(verify_len + 1);
        let base_row = target_row_offset;
        let bonus_uniform = uniforms.get(verify_len).copied().unwrap_or(0.5);
        for i in 0..verify_len {
            let draft_tok = draft_tokens[i] as usize;
            let row_start = (base_row + i) * vocab_size;
            let Some(row) = target_probs.get(row_start..row_start + vocab_size) else {
                break;
            };

            if greedy {
                // Greedy/argmax path: accept only when the target's top token equals the draft.
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (j, &v) in row.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = j;
                    }
                }
                if best_idx == draft_tok {
                    accepted_count += 1;
                } else {
                    break;
                }
                continue;
            }

            if draft_tok >= vocab_size {
                break;
            }
            let p = softmax_f32_row(row);
            let draft_row_start = i * vocab_size;
            let q = draft_logits_cpu
                .get(draft_row_start..draft_row_start + vocab_size)
                .map(softmax_f32_row);
            let q_tok = q.as_ref().and_then(|q| q.get(draft_tok).copied()).unwrap_or(0.0);
            // Match the device kernel: q == 0 accepts, and the comparison is `<=`.
            let accept_prob = if q_tok > 0.0 {
                (p[draft_tok] / q_tok).min(1.0)
            } else {
                1.0
            };
            let u = uniforms.get(i).copied().unwrap_or(1.0);
            if u <= accept_prob {
                accepted_count += 1;
                emitted_tokens.push(draft_tokens[i]);
                continue;
            }

            // Rejected at i: the emitted token must come from the residual
            // max(0, p - q), not from p. This is what makes sampling lossless.
            let mut cum = 0.0f32;
            let mut sampled = 0usize;
            let residual_total: f32 = (0..vocab_size)
                .map(|v| {
                    let pv = p[v];
                    let qv = q.as_ref().and_then(|q| q.get(v).copied()).unwrap_or(0.0);
                    (pv - qv).max(0.0)
                })
                .sum();
            let threshold = bonus_uniform * residual_total;
            for v in 0..vocab_size {
                let pv = p[v];
                let qv = q.as_ref().and_then(|q| q.get(v).copied()).unwrap_or(0.0);
                cum += (pv - qv).max(0.0);
                if cum >= threshold {
                    sampled = v;
                    break;
                }
                sampled = v;
            }
            emitted_tokens.push(sampled as u32);
            return Ok(VerificationOutcome {
                accepted_count,
                emitted_tokens,
            });
        }
        if greedy {
            return Ok(VerificationOutcome {
                accepted_count,
                emitted_tokens: Vec::new(),
            });
        }
        // All drafts accepted: the extra emitted token is sampled from the
        // target distribution at the row after the last draft.
        let bonus_start = (base_row + verify_len) * vocab_size;
        if let Some(bonus_row) = target_probs.get(bonus_start..bonus_start + vocab_size) {
            let p = softmax_f32_row(bonus_row);
            let mut cum = 0.0f32;
            let mut sampled = p.len().saturating_sub(1);
            for (v, &pv) in p.iter().enumerate() {
                cum += pv;
                if cum >= bonus_uniform {
                    sampled = v;
                    break;
                }
            }
            emitted_tokens.push(sampled as u32);
        }
        Ok(VerificationOutcome {
            accepted_count,
            emitted_tokens,
        })
    }

    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    pub fn target(&self) -> &dyn CausalLm {
        &*self.target
    }

    pub fn inner_target(&self) -> &dyn CausalLm {
        &*self.target
    }

    /// Pre-capture the 1-pass MTP `M=2` DecodeGraph during load-time prewarm.
    /// Teardown safety for GRIM_MTP_DRAFT_CHAIN: wait out any in-flight
    /// async draft. Cheap (one event wait).
    pub fn drain_chained_draft(&self) {
        if !draft_chain_enabled() {
            return;
        }
        let Some(mtp) = self.mtp_target.as_ref() else {
            return;
        };
        let mut g = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
        mtp.drain_chained_draft(&mut g);
    }

    pub fn prewarm_m2_graph(&self) -> Result<()> {
        if let Some(ref mtp) = self.mtp_target {
            let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
            mtp.prewarm_m2_graph(&mut g_guard)?;
        }
        Ok(())
    }

    /// True when the 1-pass MTP `DecodeGraph` slot holds an active captured graph.
    pub fn has_active_m2_graph(&self) -> bool {
        self.graph_m2
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Query runtime speculative decoding telemetry snapshot.
    pub fn telemetry(&self) -> SpeculativeTelemetry {
        let (state, config) = {
            let sched = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
            (sched.adaptation_state.clone(), sched.adaptation_config)
        };
        let should_adapt = if state.steps_observed < config.min_steps_before_trigger {
            false
        } else {
            state.accept_rate_ema < config.min_accept_rate
        };
        SpeculativeTelemetry {
            strategy: match self.strategy {
                Strategy::Plain => "plain".to_string(),
                Strategy::DSpark => "dspark".to_string(),
                Strategy::NativeMtp => "native_mtp".to_string(),
            },
            accept_rate_ema: state.accept_rate_ema,
            steps_observed: state.steps_observed,
            total_drafted_tokens: state.total_drafted_tokens,
            total_accepted_tokens: state.total_accepted_tokens,
            min_accept_rate: config.min_accept_rate,
            should_adapt,
            draft_depth_k: if matches!(self.strategy, Strategy::DSpark | Strategy::NativeMtp) {
                Some(
                    self.depth_tuner
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .current_depth() as u64,
                )
            } else {
                None
            },
        }
    }

    /// Run one speculative decode step. Returns the verified logits
    /// tensor (same shape as the target's `forward` return).
    pub fn decode_one(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        live_gpu_utilization: f32,
        batch_pressure: usize,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        // Every step starts with "sample from the returned logits"; only a
        // successful stochastic verify installs exact emitted token ids.
        session.set_emitted_token_ids(Vec::new());
        let result = match self.strategy {
            Strategy::Plain => self.target.forward(session, input_ids, positions, adapters),
            Strategy::NativeMtp => Ok(
                if let Some(out) = self.decode_tree(session, input_ids, positions, adapters)? {
                    out
                } else {
                    self.decode_native_mtp(session, input_ids, positions, adapters)?
                },
            ),
            Strategy::DSpark => {
                let draft = self.draft.as_ref().unwrap();
                let markov = self.markov.as_ref().unwrap();
                let confidence = self.confidence.as_ref().unwrap();

                // T2-4 (closed): the draft block length K is PID-tuned online
                // from observed acceptance — not the old hardcoded 3.
                let draft_depth_k = self
                    .depth_tuner
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .current_depth();
                let draft_block = draft.draft_block(session, input_ids, draft_depth_k)?;
                if draft_block.tokens.is_empty() {
                    return self.target.forward(session, input_ids, positions, adapters);
                }

                // Phase 2: Score confidence
                let scores = confidence.score(&draft_block);
                let mut scored = draft_block.clone();
                scored.confidence = scores;

                // Phase 3: Choose verify length dynamically
                let verify_len = self
                    .scheduler
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .choose_verify_len(&scored, live_gpu_utilization, batch_pressure);
                let verify_len = verify_len.min(scored.tokens.len());

                if verify_len == 0 {
                    return self.target.forward(session, input_ids, positions, adapters);
                }

                // Phase 4: Tentative KV Cache append
                if let Some(kv) = session.kv_mut() {
                    kv.tentative_append(verify_len)?;
                }

                // Apply Markov head bias
                let prefix = scored.tokens[..verify_len].to_vec();
                let _bias = markov.bias(&prefix, &scored.base_logits)?;

                // Phase 5: Verification step on Target Causal LM CRIT-3: The target must receive
                // the extended input (original + draft tokens) to produce logits for all draft positions
                let extended_input =
                    self.extend_input_ids(input_ids, &scored.tokens[..verify_len])?;
                let extended_positions = self.extend_positions(positions, verify_len)?;
                let prev_full_logits = session.full_output_logits();
                session.set_full_output_logits(true);
                let target_forward = self
                    .target
                    .forward(session, &extended_input, &extended_positions, adapters);
                session.set_full_output_logits(prev_full_logits);
                let target_logits = target_forward?;
                let vocab_size = scored.base_logits.shape().dims()[1];

                // CRIT-4: Use per-request RNG instead of global rand::random()
                // Task 10.6: draw verify_len+1 uniforms up front so the GPU
                // rejection sampler can run without a vocab-sized D2H copy;
                // the CPU fallback loop consumes the same clone lazily.
                let mut rng = session
                    .request_rng()
                    .cloned()
                    .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15));
                let uniforms: Vec<f32> =
                    (0..=verify_len).map(|_| rng.next_f32()).collect();

                let context_len = input_ids.shape().elem_count();
                let verification = self.verify_accepted_count(
                    &target_logits,
                    &scored.base_logits,
                    context_len.saturating_sub(1),
                    vocab_size,
                    &scored.tokens[..verify_len],
                    verify_len,
                    &uniforms,
                    &mut rng,
                    session.sampling_greedy(),
                )?;
                let accepted_count = verification.accepted_count;

                // CRIT-5: Properly commit/rollback KV cache based on accepted count.
                // tentative_append(verify_len) must have added >= accepted_count slots.
                if accepted_count > verify_len {
                    return Err(Error::Session(format!(
                        "speculative KV contract violation: accepted_count ({accepted_count}) > \
                         verify_len ({verify_len})"
                    )));
                }
                if let Some(kv) = session.kv_mut() {
                    kv.commit(accepted_count)?;
                }

                // Update scheduler and check adaptation gating
                {
                    let mut sched = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
                    sched.record_acceptance(accepted_count, verify_len);

                    if sched.should_adapt_draft() {
                        let mut accepted_mask = vec![false; verify_len];
                        accepted_mask[..accepted_count].fill(true);
                        let target_hidden_states = session
                            .get_last_hidden_state()
                            .and_then(|t| t.to_vec_f32().ok());
                        let refresh_input = crate::distill::DraftRefreshInput {
                            target_hidden_states,
                            draft_tokens: scored.tokens[..verify_len].to_vec(),
                            accepted_mask,
                        };
                        let signal = crate::distill::AdaptationSignal {
                            accept_rate_ema: sched.adaptation_state.accept_rate_ema,
                            steps_observed: sched.adaptation_state.steps_observed,
                            min_accept_rate: sched.adaptation_config.min_accept_rate,
                        };
                        let _outcome =
                            crate::distill::refresh_draft(&signal, &refresh_input, draft.as_ref())?;
                    }
                }

                // Feed the PID depth controller: next step's draft block
                // length follows observed acceptance.
                self.depth_tuner
                    .lock()
                    .unwrap()
                    .update(accepted_count, verify_len);
                if let Some(policy_mtx) = &self.astrum_draft_policy {
                    let mut pol = policy_mtx.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = pol.observe_and_select(
                        crate::draft_policy::DraftSource::Mtp,
                        context_len,
                        accepted_count + 1,
                        1.0,
                    );
                }

                // Emit the accepted drafts plus the target's bonus/rejection
                // row. The KV commit above already kept exactly that prefix;
                // rolling it back here would desync the token stream.
                let emitted = accepted_count + 1;
                session.set_emitted_token_ids(verification.emitted_tokens);
                if context_len == 0 {
                    return Err(Error::Session(
                        "DSpark decode: empty input has no next-token logits row".into(),
                    ));
                }
                // The target's full-output logits may be padded to a wider
                // row than the draft vocab; slice by the real row stride.
                let row_width = target_logits
                    .shape()
                    .dims()
                    .last()
                    .copied()
                    .unwrap_or(vocab_size)
                    .max(vocab_size);
                let flat = target_logits.to_vec_f32()?;
                let mut rows = Vec::with_capacity(emitted * vocab_size);
                for i in 0..emitted {
                    let start = (context_len - 1 + i) * row_width;
                    if start + vocab_size <= flat.len() {
                        rows.extend_from_slice(&flat[start..start + vocab_size]);
                    } else {
                        rows.resize(rows.len() + vocab_size, 0.0);
                    }
                }
                let emitted_logits = grim_backend_cpu::cpu_tensor(
                    rows,
                    grim_tensor::Shape::new(vec![emitted, vocab_size]),
                );
                session.set_last_accepted_tokens(emitted);
                Ok(emitted_logits)
            }
        };
        // ECHO-style bonus recycling: the next draft root must condition on
        // the token actually emitted at this position, including the residual
        // correction, not on a freshly argmaxed logits row.
        if result.is_ok() {
            if let Some(&tok) = session.emitted_token_ids().last() {
                session.set_last_sampled_token(Some(tok));
            }
        }
        result
    }

    /// Optional packed-tree verification path. Returns `Ok(None)` whenever
    /// the model cannot support tree forward (the default), leaving the linear
    /// path untouched. Gated by `GRIM_SPEC_TREE=1`.
    fn decode_tree(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Option<Tensor>> {
        if !spec_tree_enabled() || input_ids.shape().elem_count() != 1 {
            return Ok(None);
        }
        let Some(mtp) = self.mtp_target.as_ref() else {
            return Ok(None);
        };
        if session.skip_output_projection() {
            return Ok(None);
        }
        let Some(block) = self
            .pending_mtp_draft
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        else {
            return Ok(None);
        };
        let dims = block.base_logits.shape().dims().to_vec();
        let vocab = dims.get(1).copied().unwrap_or(0);
        let flat = block.base_logits.to_vec_f32().unwrap_or_default();
        if vocab == 0 || block.tokens.is_empty() || flat.len() < block.tokens.len() * vocab {
            *self
                .pending_mtp_draft
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(block);
            return Ok(None);
        }
        let tree = crate::draft_tree::DraftTree::from_block_logits(
            &flat[..block.tokens.len() * vocab],
            vocab,
            crate::draft_tree::TreeBuildConfig {
                rank_cap: 4,
                max_nodes: (block.tokens.len() * 8).max(8),
                max_depth: block.tokens.len(),
            },
        );
        if tree.nodes.is_empty() || !tree.is_prefix_closed() {
            *self
                .pending_mtp_draft
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(block);
            return Ok(None);
        }
        let tokens = tree.tokens();
        let depths: Vec<usize> = tree.nodes.iter().map(|n| n.depth).collect();
        let parents = tree.parents_u32();
        let base_pos = session.current_pos() as u32;
        let Some(target_logits) =
            mtp.forward_tree(session, &tokens, &depths, &parents, base_pos, adapters)?
        else {
            *self
                .pending_mtp_draft
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(block);
            return Ok(None);
        };
        let tdims = target_logits.shape().dims().to_vec();
        let t_vocab = tdims.last().copied().unwrap_or(vocab);
        let t_flat = target_logits.to_vec_f32()?;
        if t_flat.len() < tree.nodes.len() * t_vocab {
            return Err(Error::Session(format!(
                "tree forward returned {} floats for {} nodes x {t_vocab}",
                t_flat.len(),
                tree.nodes.len()
            )));
        }
        let base_row = &t_flat[..t_vocab];
        let greedy = session.sampling_greedy();
        let uniforms: Vec<f32> = (0..=tree.nodes.len())
            .map(|_| {
                session
                    .request_rng()
                    .cloned()
                    .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15))
                    .next_f32()
            })
            .collect();
        let verification = if greedy {
            crate::draft_tree::verify_tree_greedy(
                &tree,
                &t_flat[..tree.nodes.len() * t_vocab],
                base_row,
                t_vocab,
            )
        } else {
            crate::draft_tree::verify_tree_stochastic(
                &tree,
                &t_flat[..tree.nodes.len() * t_vocab],
                base_row,
                &flat[..block.tokens.len() * vocab],
                vocab,
                &uniforms,
            )
        };
        session.set_emitted_token_ids(verification.emitted_tokens.clone());
        session.set_last_accepted_tokens(verification.emitted_tokens.len().max(1));
        if let Some(kv) = session.kv_mut() {
            kv.tentative_append(tree.nodes.len())?;
            kv.commit(verification.emitted_tokens.len().saturating_sub(1))?;
        }
        if let Some(&last) = verification.emitted_tokens.last() {
            session.set_last_sampled_token(Some(last));
        }
        // Each emitted row comes from the tree node that produced it.
        let mut rows = Vec::new();
        let mut node = verification.best_node;
        let mut path: Vec<usize> = Vec::new();
        while let Some(i) = node {
            path.push(i);
            node = tree.nodes[i].parent;
        }
        path.reverse();
        for i in path {
            let start = i * t_vocab;
            rows.extend_from_slice(&t_flat[start..start + t_vocab]);
        }
        if rows.is_empty() {
            rows.extend_from_slice(base_row);
        }
        Ok(Some(grim_backend_cpu::cpu_tensor(
            rows,
            grim_tensor::Shape::new(vec![1, t_vocab]),
        )))
    }

    fn decode_native_mtp(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let mtp = self.mtp_target.as_ref().unwrap();
        let depth = mtp.mtp_depth();
        let context_len = input_ids.shape().elem_count();
        if let Some(gate) = self.may_verify.as_ref() {
            gate.lock()
                .unwrap_or_else(|e| e.into_inner())
                .ensure_depth(depth);
        }
        let (target_only, just_tripped) = Self::spec_fallback(session)
            .map(|f| {
                let tripped = f.take_tripped();
                (!f.speculating(), tripped)
            })
            .unwrap_or((false, false));
        if target_only {
            if just_tripped {
                grim_core::telemetry::emit_fallback(
                    "grim-speculative/wrapper",
                    grim_core::telemetry::FallbackReason::SpeculativeDraft,
                    "acceptance below threshold; entering target-only window",
                );
            }
            // Target-only window: exactly the plain forward, with no stale
            // pending draft carried into the probe step.
            *self
                .pending_mtp_draft
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
            session.set_last_accepted_tokens(1);
            session.set_emitted_token_ids(Vec::new());
            return self.target.forward(session, input_ids, positions, adapters);
        }

        // M3 correctness: native MTP is a one-token decode-step path. Prefill/multi-token inputs
        // and projection-skipped passes take the plain target forward, and prefill immediately
        // seeds `pending_mtp_draft` from the last hidden state so decode step 0 runs on M=2 graph!
        if depth == 0 || context_len != 1 || session.skip_output_projection() {
            session.set_last_accepted_tokens(1);
            let out = self.target.forward(session, input_ids, positions, adapters)?;
            let mtp_1pass = mtp_1pass_enabled();
            if mtp_1pass
                && depth > 0
                && context_len > 1
                && !session.skip_output_projection()
                && session.get_last_hidden_state().is_some()
            {
                let t_draft = std::time::Instant::now();
                if let Ok(first_draft) =
                    mtp.predict_multi_after_forward(session, input_ids, positions, &out)
                {
                    *self
                        .astum_last_draft_ms
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) =
                        t_draft.elapsed().as_secs_f64() * 1000.0;
                    *self
                        .pending_mtp_draft
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(first_draft);
                }
            }
            return Ok(out);
        }

        let recurrent = self.target.has_recurrent_state() || mtp.has_recurrent_state();

        // 1-pass batched MTP (default-on unless GRIM_MTP_1PASS=0/false/off): if we have a pending draft
        // from the previous step, batch [base_token, draft_tokens...] into a SINGLE trunk forward pass!
        let mtp_1pass = mtp_1pass_enabled();
        let mut pending_draft = if mtp_1pass {
            self.pending_mtp_draft.lock().unwrap_or_else(|e| e.into_inner()).take()
        } else {
            None
        };
        if session.sampling_greedy() && spec_tree_enabled() {
            if let Some(block) = pending_draft.as_mut() {
                Self::apply_tree_ranking(block, depth, true);
            }
        }
        // E/cost policy: when configured, a greedy suffix draft may replace a
        // missing MTP draft. It has no draft-head logits, so it is greedy-only.
        let mut suffix_cost_ms = 0.0f64;
        let mut used_suffix = false;
        if pending_draft.is_none() && session.sampling_greedy() {
            if let Some((block, cost_ms)) =
                self.astrum_suffix_draft_for_step(input_ids, context_len, depth.max(1))
            {
                pending_draft = Some(block);
                suffix_cost_ms = cost_ms;
                used_suffix = true;
            }
        }
        // GRIM_MTP_DRAFT_CHAIN: device-side draft pending → chained verify.
        let chain_pending = draft_chain_enabled()
            && self.chained_active.load(std::sync::atomic::Ordering::Acquire)
            && depth >= 1
            && mtp_1pass;
        if chain_pending {
            self.chained_active.store(false, std::sync::atomic::Ordering::Release);
            let verify_len = 1usize.min(depth);
            if let Some(kv) = session.kv_mut() {
                kv.tentative_append(verify_len)?;
            }
            let batched_input = self.extend_input_ids(input_ids, &[0])?;
            let batched_positions = self.extend_positions(positions, verify_len)?;
            let prev_full = session.full_output_logits();
            session.set_full_output_logits(true);
            let t_verify0 = std::time::Instant::now();
            let base_tok = input_ids.to_vec_f32().ok().and_then(|v| v.first().copied()).unwrap_or(0.0) as u32;
            let base_pos = session.current_pos() as u32;
            let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
            let batched_logits = match mtp.forward_m2_graph_chained(&mut g_guard, session, base_tok, base_pos) {
                Ok(Some(t)) => t,
                Ok(None) => {
                    drop(g_guard);
                    if let Some(kv) = session.kv_mut() {
                        let _ = kv.rollback_to(kv.len().saturating_sub(verify_len));
                    }
                    session.set_full_output_logits(prev_full);
                    let out = self.target.forward(session, &batched_input, &batched_positions, adapters)?;
                    session.set_last_accepted_tokens(1);
                    return Ok(out);
                }
                Err(e) => {
                    drop(g_guard);
                    if let Some(kv) = session.kv_mut() {
                        let _ = kv.rollback_to(kv.len().saturating_sub(verify_len));
                    }
                    session.set_full_output_logits(prev_full);
                    return Err(e);
                }
            };
            session.set_full_output_logits(prev_full);
            let t_verify_ms = t_verify0.elapsed().as_secs_f64() * 1000.0;
            let chained_tok = mtp.read_chained_draft_token(&mut g_guard);
            drop(g_guard);
            let draft_tokens = vec![chained_tok];
            let vocab_size = *batched_logits.shape().dims().last().unwrap_or(&0);
            let mut rng = session
                .request_rng()
                .cloned()
                .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15));
            let uniforms: Vec<f32> = (0..=verify_len).map(|_| rng.next_f32()).collect();
            let greedy = session.sampling_greedy();
            let row0_argmax = if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
                grim_models_transformer::Qwen35::argmax_last_logits(
                    &batched_logits,
                    vocab_size,
                )
                .unwrap_or(u32::MAX)
            } else {
                0
            };
            let verification = self.verify_accepted_count(
                &batched_logits,
                &batched_logits,
                0,
                vocab_size,
                &draft_tokens,
                verify_len,
                &uniforms,
                &mut rng,
                greedy,
            )?;
            let accepted_count = verification.accepted_count;
            session.set_emitted_token_ids(verification.emitted_tokens);
            if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
                let dtok = draft_tokens[0];
                eprintln!(
                    "[chain-debug] accept base={base_tok} draft={dtok} row0argmax={row0_argmax} accepted={accepted_count}"
                );
            }
            {
                let mut sched = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
                sched.record_acceptance(accepted_count, verify_len);
            }
            self.record_native_outcome(session, accepted_count, verify_len, context_len, false, 0.0);
            let emitted = accepted_count + 1;
            let emitted_logits = self.extract_emitted_logits(&batched_logits, emitted, vocab_size, 1)?;
            if accepted_count == verify_len {
                {
                    let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = mtp.commit_m2_graph_spec_state(&mut g_guard);
                    if self.replay_ring.is_some() {
                        let _ = mtp.commit_replay_records(&mut g_guard, accepted_count);
                    }
                }
                session.advance_pos(1 + accepted_count);
                if let Some(kv) = session.kv_mut() {
                    kv.commit(accepted_count)?;
                }
                let enqueued = mtp.enqueue_chained_draft(
                    &mut self.graph_m2.lock().unwrap_or_else(|e| e.into_inner()),
                    session,
                    &batched_input,
                    &batched_positions,
                    &emitted_logits,
                )
                .unwrap_or(false);
                self.chained_active.store(enqueued, std::sync::atomic::Ordering::Release);
            } else {
                // Rejected: publish the accepted row + roll back the trunk
                // graph, then draft SERIALLY (reject-path chaining disabled).
                {
                    let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = mtp.publish_m2_graph_hidden_row(&mut g_guard, session, accepted_count);
                }
                let base_pos_rec = session.current_pos() as u32;
                {
                    let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = mtp.rollback_m2_graph_to_accepted(&mut g_guard, base_pos_rec, 1 + accepted_count);
                }
                session.advance_pos(1 + accepted_count);
                let last_tok = if accepted_count > 0 { draft_tokens[0] } else { base_tok };
                let last_abs_pos = (base_pos_rec as usize).saturating_add(accepted_count);
                let step_input = self.scalar_token_tensor(last_tok);
                let step_pos = self.scalar_position_tensor(positions, last_abs_pos)?;
                if let Ok(next_draft) = mtp.predict_multi_after_forward(session, &step_input, &step_pos, &emitted_logits) {
                    *self.pending_mtp_draft.lock().unwrap_or_else(|e| e.into_inner()) = Some(next_draft);
                }
            }
            if std::env::var_os("GRIM_MTP_TIMING").is_some() {
                eprintln!(
                    "[mtp-timing-chained] accepted={accepted_count} verify={t_verify_ms:.2}ms tok={chained_tok}"
                );
            }
            session.set_last_accepted_tokens(emitted);
            return Ok(emitted_logits);
        }
        let allow_verify = self
            .may_verify
            .as_ref()
            .map(|g| g.lock().unwrap_or_else(|e| e.into_inner()).should_verify())
            .unwrap_or(true);
        if let Some(draft_block) = pending_draft.filter(|_| allow_verify) {
            let verify_len = self.choose_verify_width(
                &draft_block.confidence,
                draft_block.tokens.len().min(depth),
                context_len,
            );
            if verify_len > 0 && mtp_1pass {
                let kv_len_before = session.kv_mut().map(|kv| kv.len());
                // ReplaySSM-style flush decision depends only on the committed
                // cursor and draft length, so it is safe before sampling. When
                // the ring would overflow, use the snapshot/replay path.
                let replay_flush = self
                    .replay_ring
                    .as_ref()
                    .map(|r| {
                        r.lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .needs_flush(verify_len)
                    })
                    .unwrap_or(false);
                let use_m2_graph = verify_len >= 1
                    && decode_graph_m2_enabled()
                    && !replay_flush;
                if let Some(ring) = self.replay_ring.as_ref() {
                    let mut ring = ring.lock().unwrap_or_else(|e| e.into_inner());
                    if replay_flush {
                        ring.flush();
                    }
                    ring.append_draft(&draft_block.tokens[..verify_len]);
                }
                if use_m2_graph {
                    let base = session.current_pos() as u32;
                    let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                    for (i, &tok) in draft_block.tokens[..verify_len].iter().enumerate() {
                        mtp.append_replay_record(&mut g_guard, tok, base + 1 + i as u32)?;
                    }
                }
                let pre_batch_snapshot = if recurrent && !use_m2_graph {
                    session.snapshot_recurrent_state()?
                } else {
                    None
                };
                if let Some(kv) = session.kv_mut() {
                    kv.tentative_append(verify_len)?;
                }
                let batched_input = self.extend_input_ids(input_ids, &draft_block.tokens[..verify_len])?;
                let batched_positions = self.extend_positions(positions, verify_len)?;
                let prev_full = session.full_output_logits();
                session.set_full_output_logits(true);
                let t_verify0 = std::time::Instant::now();
                let batched_res = if verify_len >= 1 {
                    let base_tok = input_ids.to_vec_f32().ok().and_then(|v| v.first().copied()).unwrap_or(0.0) as u32;
                    let base_pos = session.current_pos() as u32;
                    let mut batch_toks = Vec::with_capacity(1 + verify_len);
                    batch_toks.push(base_tok);
                    batch_toks.extend_from_slice(&draft_block.tokens[..verify_len]);
                    let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                    match mtp.forward_m2_graph(&mut g_guard, session, &batch_toks, base_pos) {
                        Ok(Some(t)) => Ok(t),
                        Ok(None) => self.target.forward(session, &batched_input, &batched_positions, adapters),
                        Err(e) => Err(e),
                    }
                } else {
                    self.target.forward(session, &batched_input, &batched_positions, adapters)
                };
                session.set_full_output_logits(prev_full);
                let batched_logits = match batched_res {
                    Ok(t) => t,
                    Err(e) => {
                        if let Some(kv) = session.kv_mut() {
                            let _ = kv.rollback_to(kv.len().saturating_sub(verify_len));
                        }
                        if let Some(snap) = pre_batch_snapshot {
                            let _ = session.restore_recurrent_state(snap);
                        }
                        return Err(e);
                    }
                };
                let t_verify_ms = t_verify0.elapsed().as_secs_f64() * 1000.0;
                let vocab_size = *batched_logits.shape().dims().last().unwrap_or(&0);
                let mut rng = session
                    .request_rng()
                    .cloned()
                    .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15));
                let uniforms: Vec<f32> = (0..=verify_len).map(|_| rng.next_f32()).collect();
                let greedy = session.sampling_greedy();
                let row0_argmax = if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
                    grim_models_transformer::Qwen35::argmax_last_logits(
                        &batched_logits,
                        vocab_size,
                    )
                    .unwrap_or(u32::MAX)
                } else {
                    0
                };
                let verification = self.verify_accepted_count(
                    &batched_logits,
                    &draft_block.base_logits,
                    0,
                    vocab_size,
                    &draft_block.tokens[..verify_len],
                    verify_len,
                    &uniforms,
                    &mut rng,
                    greedy,
                )?;
                let accepted_count = verification.accepted_count;
                session.set_emitted_token_ids(verification.emitted_tokens);
                if let Some(gate_mtx) = &self.may_verify {
                    gate_mtx
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .observe(accepted_count, verify_len);
                }
                if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
                    let dtok0 = draft_block.tokens.first().copied().unwrap_or(0);
                    let btok = input_ids.to_vec_f32().ok().and_then(|v| v.first().copied()).unwrap_or(0.0) as u32;
                    eprintln!(
                        "[serial-debug] base={btok} draft={dtok0} row0argmax={row0_argmax} accepted={accepted_count}"
                    );
                }
                {
                    let mut sched = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
                    sched.record_acceptance(accepted_count, verify_len);
                }
                self.record_native_outcome(
                    session,
                    accepted_count,
                    verify_len,
                    context_len,
                    used_suffix,
                    suffix_cost_ms,
                );
                let emitted = accepted_count + 1;
                if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
                    eprintln!(
                        "[mtp-debug-1pass] depth={} drafted={} accepted={} emitted={} greedy={}",
                        depth, verify_len, accepted_count, emitted, greedy
                    );
                }
                let t_step0 = std::time::Instant::now();
                let mut t_rb_ms = 0.0f64;
                let mut t_draft_ms = 0.0f64;
                let emitted_logits = self.extract_emitted_logits(&batched_logits, emitted, vocab_size, 1)?;
                let t_accept_ms = t_step0.elapsed().as_secs_f64() * 1000.0;
                if accepted_count == verify_len {
                    if use_m2_graph {
                        {
                            let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                            let _ = mtp.publish_m2_graph_hidden_row(&mut g_guard, session, accepted_count);
                            let _ = mtp.commit_m2_graph_spec_state(&mut g_guard);
                            if self.replay_ring.is_some() {
                                let _ = mtp.commit_replay_records(&mut g_guard, accepted_count);
                            }
                        }
                        session.advance_pos(1 + accepted_count);
                    }
                    if let Some(kv) = session.kv_mut() {
                        kv.commit(accepted_count)?;
                    }
                    let t_d0 = std::time::Instant::now();
                    let mut chained = false;
                    if draft_chain_enabled() && session.sampling_greedy() {
                        let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                        chained = mtp
                            .enqueue_chained_draft(&mut g_guard, session, &batched_input, &batched_positions, &emitted_logits)
                            .unwrap_or(false);
                    }
                    self.chained_active.store(chained, std::sync::atomic::Ordering::Release);
                    if !chained {
                        if let Ok(next_draft) = mtp.predict_multi_after_forward(session, &batched_input, &batched_positions, &emitted_logits) {
                            *self.pending_mtp_draft.lock().unwrap_or_else(|e| e.into_inner()) = Some(next_draft);
                        }
                    }
                    t_draft_ms = t_d0.elapsed().as_secs_f64() * 1000.0;
                } else if recurrent {
                    let t_r0 = std::time::Instant::now();
                    if let Some(base_kv_len) = kv_len_before {
                        if let Some(kv) = session.kv_mut() {
                            if kv.len() != base_kv_len {
                                let _ = kv.rollback_to(base_kv_len);
                            }
                        }
                    }
                    if let Some(snap) = pre_batch_snapshot {
                        session.restore_recurrent_state(snap)?;
                    }
                    let base_last_pos = positions.to_vec_f32().ok().and_then(|v| v.last().copied()).unwrap_or(0.0);
                    let base_tok_rec = input_ids.to_vec_f32().ok().and_then(|v| v.first().copied()).unwrap_or(0.0) as u32;
                    let base_pos_rec = session.current_pos() as u32;
                    let mut accepted_toks = Vec::with_capacity(1 + accepted_count);
                    accepted_toks.push(base_tok_rec);
                    accepted_toks.extend_from_slice(&draft_block.tokens[..accepted_count]);
                    // Fast-path (~1 ms, zero GEMVs): publish row `accepted_count` of `head_input`
                    // from the M-token verify pass and roll back only the recurrent/KV state to
                    // `base_pos_rec + 1 + accepted_count` using the already-computed projections!
                    let fast_recovered = if use_m2_graph {
                        let mut g_guard = self.graph_m2.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = mtp.publish_m2_graph_hidden_row(&mut g_guard, session, accepted_count);
                        mtp.rollback_m2_graph_to_accepted(&mut g_guard, base_pos_rec, 1 + accepted_count)
                            .unwrap_or(false)
                    } else {
                        false
                    };
                    let (last_rec_logits, last_rec_input, last_rec_pos) = if fast_recovered {
                        session.advance_pos(1 + accepted_count);
                        let last_tok = *accepted_toks.last().unwrap_or(&base_tok_rec);
                        let last_abs_pos = (base_last_pos as usize).saturating_add(accepted_count);
                        (
                            emitted_logits.clone(),
                            self.scalar_token_tensor(last_tok),
                            self.scalar_position_tensor(positions, last_abs_pos)?,
                        )
                    } else {
                        let mut l_logits = self.target.forward(session, input_ids, positions, adapters)?;
                        let mut l_input = input_ids.clone();
                        let mut l_pos = positions.clone();
                        for i in 0..accepted_count {
                            let tok = draft_block.tokens[i];
                            let step_input = self.scalar_token_tensor(tok);
                            let abs_pos = (base_last_pos as usize).saturating_add(1 + i);
                            let step_pos = self.scalar_position_tensor(positions, abs_pos)?;
                            l_logits = self.target.forward(session, &step_input, &step_pos, adapters)?;
                            l_input = step_input;
                            l_pos = step_pos;
                        }
                        (l_logits, l_input, l_pos)
                    };
                    // Immediately re-draft from recovered state so the next step stays
                    // in 1-pass batched verification instead of dropping to pending_draft=false.
                    t_rb_ms = t_r0.elapsed().as_secs_f64() * 1000.0;
                    let t_d0 = std::time::Instant::now();
                    if let Ok(next_draft) = mtp.predict_multi_after_forward(
                        session,
                        &last_rec_input,
                        &last_rec_pos,
                        &last_rec_logits,
                    ) {
                        *self.pending_mtp_draft.lock().unwrap_or_else(|e| e.into_inner()) = Some(next_draft);
                    }
                    t_draft_ms = t_d0.elapsed().as_secs_f64() * 1000.0;
                } else if let Some(kv) = session.kv_mut() {
                    kv.commit(accepted_count)?;
                }
                if std::env::var_os("GRIM_MTP_TIMING").is_some() {
                    eprintln!(
                        "[mtp-timing] accepted={accepted_count} verify_after_match={t_verify_ms:.2}ms extract={t_accept_ms:.2}ms rollback={t_rb_ms:.2}ms draft={t_draft_ms:.2}ms"
                    );
                }
                session.set_last_accepted_tokens(emitted);
                return Ok(emitted_logits);
            }
        }
        // `mut`: the snapshots are consumed by `restore_recurrent_state`, and
        // `Box<dyn Any + Send>` is not `Clone`, so the reject/partial-accept
        // paths `take()` them rather than cloning.
        let mut base_snapshot: Option<Box<dyn std::any::Any + Send>> = None;

        // 1. Base forward (exactly once). This advances state by the base token and populates
        // the session's last hidden state for the nextn draft head.
        // Run step 0 eagerly so we don't pay ~1.5 s to capture a batch=1 DecodeGraph when all
        // subsequent 1-pass verify steps run at batch=2 (and rejections roll back in-place).
        let prev_full_logits = session.full_output_logits();
        session.set_full_output_logits(false);
        let base_logits_full = self.target.forward(session, input_ids, positions, adapters)?;
        session.set_full_output_logits(prev_full_logits);

        let kv_len_after_base = session.kv_mut().map(|kv| kv.len());
        let base_hidden_state = session.get_last_hidden_state();

        // Snapshot after base for recurrent rollback/replay: avoids replaying base forward on
        // reject/partial-accept and restores exact post-base state (KV/recurrent/pos).
        let tr = std::env::var("GRIM_TRACE_LAYERS").is_ok();
        let tr_stage = |name: &str| {
            if tr {
                eprintln!("[mtp-trace] {}", name);
            }
        };
        tr_stage("decode_native_mtp: enter");
        let mut post_base_snapshot = if recurrent && !mtp_1pass {
            session.snapshot_recurrent_state()?
        } else {
            None
        };

        let base_dims = base_logits_full.shape().dims();
        let vocab_size = *base_dims.last().unwrap_or(&0);
        if vocab_size == 0 {
            if let Some(snap) = base_snapshot {
                let _ = session.restore_recurrent_state(snap);
                // best-effort plain step after rollback
                session.set_last_accepted_tokens(1);
                return self.target.forward(session, input_ids, positions, adapters);
            }
            session.set_last_accepted_tokens(1);
            return Ok(base_logits_full);
        }
        let base_vec = base_logits_full.to_vec_f32()?;
        let base_last = if base_vec.len() >= vocab_size {
            base_vec[base_vec.len() - vocab_size..].to_vec()
        } else {
            base_vec
        };
        if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
            let base_argmax = base_last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            eprintln!("[mtp-debug-base] pos={} argmax={}", session.current_pos(), base_argmax);
        }

        // 2. Draft from the already-run base forward (must not re-run the base input).
        let t_draft = std::time::Instant::now();
        let mut draft_block =
            mtp.predict_multi_after_forward(session, input_ids, positions, &base_logits_full)?;
        *self
            .astum_last_draft_ms
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = t_draft.elapsed().as_secs_f64() * 1000.0;
        if session.sampling_greedy() && spec_tree_enabled() {
            Self::apply_tree_ranking(&mut draft_block, depth, true);
        }
        let verify_len = draft_block.tokens.len().min(depth);
        if verify_len > 0 && mtp_1pass {
            *self.pending_mtp_draft.lock().unwrap_or_else(|e| e.into_inner()) = Some(draft_block);
            session.set_last_accepted_tokens(1);
            return Ok(grim_backend_cpu::cpu_tensor(
                base_last,
                grim_tensor::Shape::new(vec![1, vocab_size]),
            ));
        }

        // Helper: emit only the base row. Recurrent case must restore to post-base snapshot so
        // state is exactly base-advanced (base forward already ran above).
        if verify_len == 0 {
            let had_post = post_base_snapshot.is_some();
            if let Some(snap) = post_base_snapshot.take().or_else(|| base_snapshot.take()) {
                session.restore_recurrent_state(snap)?;
                if !had_post {
                    // Fallback only if post-base snapshot unavailable.
                    let _ = self.target.forward(session, input_ids, positions, adapters)?;
                }
            }
            if let Some(h) = base_hidden_state.clone() {
                session.set_last_hidden_state(h);
            }
            session.set_last_accepted_tokens(1);
            return Ok(grim_backend_cpu::cpu_tensor(
                base_last,
                grim_tensor::Shape::new(vec![1, vocab_size]),
            ));
        }

        // 3. Tentative append draft slots and verify ONLY draft tokens (never the base token).
        if let Some(kv) = session.kv_mut() {
            kv.tentative_append(verify_len)?;
        }
        let draft_input = self.extend_input_ids_ids(&draft_block.tokens[..verify_len])?;
        let base_last_pos = positions
            .to_vec_f32()
            .ok()
            .and_then(|v| v.last().copied())
            .unwrap_or(0.0);
        let draft_positions_vec: Vec<f32> = (0..verify_len)
            .map(|i| base_last_pos + 1.0 + i as f32)
            .collect();
        let draft_positions = grim_backend_cpu::cpu_tensor(
            draft_positions_vec,
            grim_tensor::Shape::new(vec![verify_len]),
        );

        let prev_full2 = session.full_output_logits();
        session.set_full_output_logits(true);
        tr_stage("verify: forward start");
        let verify_forward = self.target.forward(session, &draft_input, &draft_positions, adapters);
        tr_stage("verify: forward done");
        session.set_full_output_logits(prev_full2);
        let verify_logits = match verify_forward {
            Ok(t) => t,
            Err(e) => {
                if let Some(kv) = session.kv_mut() {
                    let _ = kv.rollback_to(kv.len().saturating_sub(verify_len));
                }
                let had_post = post_base_snapshot.is_some();
                if let Some(snap) = post_base_snapshot.take().or_else(|| base_snapshot.take()) {
                    let _ = session.restore_recurrent_state(snap);
                    if !had_post {
                        let _ = self.target.forward(session, input_ids, positions, adapters);
                    }
                }
                return Err(e);
            }
        };

        // 4. Combine [L0 = base_last; verify rows] so row 0 predicts first emitted token and
        // row 1+i predicts token after draft i (context_len=1 for verification/extraction).
        tr_stage("verify: accept-count start");
        let combined = self.prepend_base_row(&base_last, &verify_logits, vocab_size, verify_len)?;

        let mut rng = session
            .request_rng()
            .cloned()
            .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15));
        let uniforms: Vec<f32> = (0..=verify_len).map(|_| rng.next_f32()).collect();
        let greedy = session.sampling_greedy();
        let verification = self.verify_accepted_count(
            &combined,
            &draft_block.base_logits,
            0, // combined row 0 is the base row
            vocab_size,
            &draft_block.tokens[..verify_len],
            verify_len,
            &uniforms,
            &mut rng,
            greedy,
        )?;
        let mut accepted_count = verification.accepted_count;
        session.set_emitted_token_ids(verification.emitted_tokens);
        tr_stage(&format!("accept-count done (accepted={accepted_count})"));
        // Debug/measurement hooks for M2/M3 gate closure (TODO(gpu-verify): remove if undesired).
        if std::env::var_os("GRIM_MTP_FORCE_REJECT").is_some() {
            accepted_count = 0;
        } else if std::env::var_os("GRIM_MTP_FORCE_ACCEPT").is_some() {
            accepted_count = verify_len;
        }

        if accepted_count > verify_len {
            if let Some(kv) = session.kv_mut() {
                let _ = kv.rollback_to(kv.len().saturating_sub(verify_len));
            }
            let had_post = post_base_snapshot.is_some();
            if let Some(snap) = post_base_snapshot.take().or_else(|| base_snapshot.take()) {
                let _ = session.restore_recurrent_state(snap);
                if !had_post {
                    let _ = self.target.forward(session, input_ids, positions, adapters);
                }
            }
            return Err(Error::Session(format!(
                "speculative KV contract violation (NativeMTP): accepted_count ({accepted_count}) > verify_len ({verify_len})"
            )));
        }

        {
            let mut sched = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
            sched.record_acceptance(accepted_count, verify_len);
        }
        self.record_native_outcome(session, accepted_count, verify_len, context_len, false, 0.0);

        let emitted = accepted_count + 1;
        if std::env::var_os("GRIM_MTP_DEBUG").is_some() {
            eprintln!(
                "[mtp-debug] depth={} drafted={} accepted={} emitted={} greedy={} recurrent={} replay={}",
                depth,
                verify_len,
                accepted_count,
                emitted,
                greedy,
                recurrent,
                recurrent && accepted_count < verify_len
            );
        }

        if recurrent {
            if accepted_count < verify_len {
                // Verify forward mutated recurrent/KV state for drafted rows. Drop tentative
                // draft KV slots (if any), then restore to exact post-base and replay only the
                // accepted prefix so rejected rows have zero net effect.
                if let Some(base_kv_len) = kv_len_after_base {
                    if let Some(kv) = session.kv_mut() {
                        if kv.len() != base_kv_len {
                            let _ = kv.rollback_to(base_kv_len);
                        }
                    }
                }
                if let Some(snap) = post_base_snapshot.take() {
                    tr_stage("reject: restore start");
                    session.restore_recurrent_state(snap)?; // also rolls KV to post-base pos
                    tr_stage("reject: restore done");
                    if accepted_count == 0 {
                        if let Some(h) = base_hidden_state.clone() {
                            session.set_last_hidden_state(h);
                        }
                    }
                    for i in 0..accepted_count {
                        let tok = draft_block.tokens[i];
                        let step_input = self.scalar_token_tensor(tok);
                        let abs_pos = (base_last_pos as usize).saturating_add(1 + i);
                        let step_pos = self.scalar_position_tensor(positions, abs_pos)?;
                        let _ = self
                            .target
                            .forward(session, &step_input, &step_pos, adapters)?;
                    }
                } else if let Some(snap) = base_snapshot.take() {
                    session.restore_recurrent_state(snap)?; // also rolls KV to base pos
                    let _ = self.target.forward(session, input_ids, positions, adapters)?;
                    // base forward re-ran: sets post-base hidden state.
                    for i in 0..accepted_count {
                        let tok = draft_block.tokens[i];
                        let step_input = self.scalar_token_tensor(tok);
                        let abs_pos = (base_last_pos as usize).saturating_add(1 + i);
                        let step_pos = self.scalar_position_tensor(positions, abs_pos)?;
                        let _ = self
                            .target
                            .forward(session, &step_input, &step_pos, adapters)?;
                    }
                }
            } else {
                // Full accept: verify forward already applied all drafted rows in order.
                if let Some(kv) = session.kv_mut() {
                    kv.commit(accepted_count)?;
                }
            }
        } else {
            // Non-recurrent: KV tentative slots track drafted rows only.
            if let Some(kv) = session.kv_mut() {
                kv.commit(accepted_count)?;
            }
        }

        session.set_last_accepted_tokens(emitted);
        tr_stage("emit: extract start");
        let accepted_logits = self.extract_emitted_logits(&combined, emitted, vocab_size, 1)?;
        tr_stage("emit: extract done, returning to caller");
        Ok(accepted_logits)
    }

    /// Build input tensor containing only draft token ids (M3 verify forward input).
    fn extend_input_ids_ids(&self, draft_tokens: &[u32]) -> Result<Tensor> {
        let ids: Vec<f32> = draft_tokens.iter().map(|&t| t as f32).collect();
        Ok(grim_backend_cpu::cpu_tensor(
            ids,
            grim_tensor::Shape::new(vec![draft_tokens.len()]),
        ))
    }

    /// Prepend the base forward's last-row logits to verify-pass logits, producing
    /// `[1 + verify_len, vocab]` where row 0 = base (distribution for first emitted token)
    /// and row 1+i = target logits for draft token i.
    fn prepend_base_row(
        &self,
        base_last_row: &[f32],
        verify_logits: &Tensor,
        vocab_size: usize,
        verify_len: usize,
    ) -> Result<Tensor> {
        let v = verify_logits.to_vec_f32()?;
        let need = verify_len * vocab_size;
        let mut out = Vec::with_capacity(vocab_size + need);
        if base_last_row.len() >= vocab_size {
            out.extend_from_slice(&base_last_row[base_last_row.len() - vocab_size..]);
        } else {
            out.extend_from_slice(base_last_row);
            out.resize(vocab_size, 0.0);
        }
        if v.len() >= need {
            out.extend_from_slice(&v[..need]);
        } else {
            out.extend_from_slice(&v);
            out.resize(vocab_size + need, 0.0);
        }
        Ok(grim_backend_cpu::cpu_tensor(
            out,
            grim_tensor::Shape::new(vec![1 + verify_len, vocab_size]),
        ))
    }

    /// Build a single-token input tensor for replay.
    fn scalar_token_tensor(&self, token: u32) -> Tensor {
        grim_backend_cpu::cpu_tensor(vec![token as f32], grim_tensor::Shape::new(vec![1]))
    }

    /// Build a single-token position tensor for replay at absolute position.
    fn scalar_position_tensor(&self, base_positions: &Tensor, abs_pos: usize) -> Result<Tensor> {
        let _ = base_positions; // absolute position preferred for replay correctness
        Ok(grim_backend_cpu::cpu_tensor(
            vec![abs_pos as f32],
            grim_tensor::Shape::new(vec![1]),
        ))
    }

    /// Extend input_ids tensor with draft tokens for verification
    fn extend_input_ids(&self, input_ids: &Tensor, draft_tokens: &[u32]) -> Result<Tensor> {
        let mut ids = input_ids.to_vec_f32()?;
        for tok in draft_tokens {
            ids.push(*tok as f32);
        }
        let shape = grim_tensor::Shape::new(vec![ids.len()]);
        Ok(grim_backend_cpu::cpu_tensor(ids, shape))
    }

    /// Extend positions tensor for the draft tokens
    fn extend_positions(&self, positions: &Tensor, num_new: usize) -> Result<Tensor> {
        let mut pos = positions.to_vec_f32()?;
        let last_pos = pos.last().copied().unwrap_or(-1.0);
        if pos.is_empty() && num_new > 0 {
            // Empty positions: first draft token starts at position 0.
            // last_pos will be -1.0, so we get 0, 1, 2, ...
            for i in 0..num_new {
                pos.push(i as f32);
            }
        } else {
            for i in 0..num_new {
                pos.push(last_pos + 1.0 + i as f32);
            }
        }
        let shape = grim_tensor::Shape::new(vec![pos.len()]);
        Ok(grim_backend_cpu::cpu_tensor(pos, shape))
    }

    /// Extract logits rows for emitted tokens.
    /// `num_rows` is typically `accepted_count + 1` (accepted drafts + bonus).
    /// Rows start at `context_len - 1` in the extended verify logits: row for first emitted
    /// token (or bonus when none accepted), through bonus after last accepted draft.
    fn extract_emitted_logits(
        &self,
        target_logits: &Tensor,
        num_rows: usize,
        vocab_size: usize,
        context_len: usize,
    ) -> Result<Tensor> {
        if num_rows == 0 {
            return Ok(grim_backend_cpu::cpu_tensor(
                Vec::new(),
                grim_tensor::Shape::new(vec![0, vocab_size]),
            ));
        }
        let start_row = context_len.saturating_sub(1);
        grim_models_transformer::Qwen35::slice_logits_rows(
            target_logits,
            start_row,
            num_rows,
            vocab_size,
        )
    }
}

impl Model for SpeculativeCausalLm {
    fn config(&self) -> &dyn ModelConfig {
        self.target.config()
    }
    fn device(&self) -> &Device {
        self.target.device()
    }
    fn param_arith(&self) -> ArithType {
        self.target.param_arith()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for SpeculativeCausalLm {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.target.new_session()
    }

    fn hidden_size_hint(&self) -> Option<usize> {
        self.target.hidden_size_hint()
    }

    fn has_recurrent_state(&self) -> bool {
        self.target.has_recurrent_state()
    }

    fn prepare_session(&self, session: &mut dyn SessionT) {
        self.target.prepare_session(session)
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        // CRIT-6: Derive scheduling params from the session instead of hardcoding.
        let live_gpu_utilization = session.live_gpu_utilization();
        let batch_pressure = session.batch_pressure();
        self.decode_one(
            session,
            input_ids,
            positions,
            live_gpu_utilization,
            batch_pressure,
            adapters,
        )
    }
}

/// Row-wise softmax on an f32 slice in-place.
fn softmax_f32_row(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    let exps: Vec<f32> = logits
        .iter()
        .map(|&x| {
            let e = (x - max_val).exp();
            exp_sum += e;
            e
        })
        .collect();
    exps.into_iter().map(|e| e / exp_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DraftBlock;
    use grim_core::session::Inner;
    use grim_tensor::Shape;

    struct MockCausalLm {
        cfg: grim_models_transformer::LlamaConfig,
        device: Device,
    }

    impl Clone for MockCausalLm {
        fn clone(&self) -> Self {
            Self {
                cfg: self.cfg.clone(),
                device: self.device.clone(),
            }
        }
    }

    impl Model for MockCausalLm {
        fn config(&self) -> &dyn ModelConfig {
            &self.cfg
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn param_arith(&self) -> ArithType {
            ArithType::F32
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl CausalLm for MockCausalLm {
        fn new_session(&self) -> Box<dyn SessionT> {
            Box::new(Inner::new(self.device.clone()))
        }
        fn forward(
            &self,
            session: &mut dyn SessionT,
            input_ids: &Tensor,
            _positions: &Tensor,
            _adapters: &[AdapterHandle],
        ) -> Result<Tensor> {
            let seq_len = input_ids.shape().dims()[0];
            // Mock penultimate hidden states: [1, seq_len, hidden_size]
            let hidden_state = grim_backend_cpu::cpu_tensor(
                vec![0.5f32; seq_len * self.cfg.hidden_size],
                Shape::new(vec![1, seq_len, self.cfg.hidden_size]),
            );
            session.set_last_hidden_state(hidden_state);

            // Mock output logits: return constant values (all accepted)
            let logits = grim_backend_cpu::cpu_tensor(
                vec![0.1f32; seq_len * self.cfg.vocab_size],
                Shape::new(vec![seq_len, self.cfg.vocab_size]),
            );
            Ok(logits)
        }
    }

    #[test]
    fn test_hidden_state_capture_and_adaptation_trigger() {
        let device = Device::Cpu;
        let cfg = grim_models_transformer::LlamaConfig {
            vocab_size: 100,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 2,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,

            partial_rotary_factor: 1.0,
            yarn: None,
        ..Default::default()
        };
        let target = Box::new(MockCausalLm {
            cfg: cfg.clone(),
            device,
        });

        // Mock DSpark components
        let draft = Arc::new(crate::tiny_draft_backbone::TinyDraftBackbone::new(
            100, // vocab_size
            16,  // hidden_size
            5,   // block_len
            42,  // seed
        ));
        let markov = Arc::new(crate::uniform_markov_head::UniformMarkovHead::new(
            100, 5, 42,
        ));
        let confidence = Arc::new(crate::entropy_confidence_head::EntropyConfidenceHead);

        // Create scheduler with high trigger threshold (e.g. 1.5) to always trigger adaptation
        let mut scheduler =
            ConfidenceScheduler::new(ThroughputProfile::default(), SpeculationConfig::default());
        scheduler.adaptation_config.min_steps_before_trigger = 1;
        scheduler.adaptation_config.min_accept_rate = 1.5;
        scheduler.adaptation_config.ema_alpha = 0.5;

        let spec_lm =
            SpeculativeCausalLm::with_dspark(target, draft.clone(), markov, confidence, scheduler);

        let mut session = spec_lm.new_session();
        let input_ids = grim_backend_cpu::cpu_tensor(vec![1f32], Shape::new(vec![1]));
        let positions = grim_backend_cpu::cpu_tensor(vec![0f32], Shape::new(vec![1]));

        // 1. Verify that before forward run, last hidden state is empty
        assert!(session.get_last_hidden_state().is_none());

        // 2. Perform a speculative decode step (this will call MockCausalLm's forward pass)
        let _logits = spec_lm
            .decode_one(session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
            .unwrap();

        // 3. Verify that the penultimate hidden state is successfully captured in the session
        let captured_hidden = session.get_last_hidden_state().unwrap();
        let hidden_shape = captured_hidden.shape();
        // Hidden state shape is [1, 1 + verify_len, hidden_size] because the target receives the extended input (original token + draft tokens).
        // choose_verify_len enforces min_verify_len=1, so verify_len >= 1.
        assert_eq!(hidden_shape.dims(), &[1, 2, 16]); // [1, 1+verify_len, hidden_size]

        // 4. Force weight update (adaptation EMA will drop below 1.5 min threshold after this step)
        let w_head_before = {
            let w = draft.weights.lock().unwrap_or_else(|e| e.into_inner());
            w.w_head.clone()
        };

        let _logits2 = spec_lm
            .decode_one(session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
            .unwrap();

        // Check that the scheduler registered the step and triggered adaptation
        let sched = spec_lm.scheduler.lock().unwrap_or_else(|e| e.into_inner());
        assert!(sched.adaptation_state.steps_observed >= 2);
        assert!(sched.adaptation_state.accept_rate_ema < 1.5);

        // Check that weights were indeed updated (nudge applied)
        let w_head_after = {
            let w = draft.weights.lock().unwrap_or_else(|e| e.into_inner());
            w.w_head.clone()
        };
        assert_ne!(w_head_before, w_head_after);
    }

    /// T2-4 (closed): the DSpark draft depth K is PID-tuned, not hardcoded 3.
    /// With a target that accepts every draft (uniform logits ⇒ ratio 1), the tuner ramps.
    #[test]
    fn test_dspark_depth_ramps_up_with_full_acceptance() {
        let cfg = grim_models_transformer::LlamaConfig {
            vocab_size: 100,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 2,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            partial_rotary_factor: 1.0,
            yarn: None,
        ..Default::default()
        };
        let target = Box::new(MockCausalLm {
            cfg: cfg.clone(),
            device: Device::Cpu,
        });

        let draft = Arc::new(crate::tiny_draft_backbone::TinyDraftBackbone::new(
            100, 16, 5, 42,
        ));
        let markov = Arc::new(crate::uniform_markov_head::UniformMarkovHead::new(
            100, 5, 42,
        ));
        let confidence = Arc::new(crate::entropy_confidence_head::EntropyConfidenceHead);
        let scheduler =
            ConfidenceScheduler::new(ThroughputProfile::default(), SpeculationConfig::default());
        let spec_lm =
            SpeculativeCausalLm::with_dspark(target, draft, markov, confidence, scheduler);

        let mut session = spec_lm.new_session();
        let input_ids = grim_backend_cpu::cpu_tensor(vec![1f32], Shape::new(vec![1]));
        let positions = grim_backend_cpu::cpu_tensor(vec![0f32], Shape::new(vec![1]));

        // PID starts at (min+max)/2 = 3; full acceptance drives it to max 5.
        for _ in 0..4 {
            let _ = spec_lm
                .decode_one(session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
                .unwrap();
        }
        let tel = spec_lm.telemetry();
        assert_eq!(tel.strategy, "dspark");
        assert_eq!(
            tel.draft_depth_k,
            Some(5),
            "full acceptance must ramp PID depth to max"
        );
    }

    /// A drafter whose base logits are one-hot on its drafted token against a uniform target gives p_target/p_draft ≈
    /// 0 - every draft rejected, so the tuner must collapse K to its minimum instead of staying at 3.
    struct AlwaysRejectedDraft;

    impl DraftBackbone for AlwaysRejectedDraft {
        fn draft_block(
            &self,
            _session: &mut dyn grim_core::session::SessionT,
            _context: &Tensor,
            block_len: usize,
        ) -> Result<DraftBlock> {
            let vocab = 100usize;
            let mut logits = vec![0.0f32; block_len * vocab];
            for row in 0..block_len {
                logits[row * vocab + 7] = 10.0; // one-hot on the drafted token
            }
            Ok(DraftBlock {
                tokens: vec![7u32; block_len],
                base_logits: grim_backend_cpu::cpu_tensor(
                    logits,
                    Shape::new(vec![block_len, vocab]),
                ),
                confidence: vec![1.0; block_len],
            })
        }
        fn estimated_footprint_bytes(&self) -> usize {
            0
        }
        fn update_weights(
            &self,
            _target_hidden_states: &[f32],
            _draft_tokens: &[u32],
            _accepted_mask: &[bool],
        ) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_dspark_depth_collapses_under_total_rejection() {
        let cfg = grim_models_transformer::LlamaConfig {
            vocab_size: 100,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 2,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            partial_rotary_factor: 1.0,
            yarn: None,
        ..Default::default()
        };
        let target = Box::new(MockCausalLm {
            cfg: cfg.clone(),
            device: Device::Cpu,
        });

        let draft: Arc<dyn DraftBackbone> = Arc::new(AlwaysRejectedDraft);
        let markov = Arc::new(crate::uniform_markov_head::UniformMarkovHead::new(
            100, 5, 42,
        ));
        let confidence = Arc::new(crate::entropy_confidence_head::EntropyConfidenceHead);
        let scheduler =
            ConfidenceScheduler::new(ThroughputProfile::default(), SpeculationConfig::default());
        let spec_lm =
            SpeculativeCausalLm::with_dspark(target, draft, markov, confidence, scheduler);

        let mut session = spec_lm.new_session();
        let input_ids = grim_backend_cpu::cpu_tensor(vec![1f32], Shape::new(vec![1]));
        let positions = grim_backend_cpu::cpu_tensor(vec![0f32], Shape::new(vec![1]));

        for _ in 0..20 {
            let _ = spec_lm
                .decode_one(session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
                .unwrap();
        }
        let tel = spec_lm.telemetry();
        assert_eq!(
            tel.draft_depth_k,
            Some(1),
            "total rejection must collapse PID depth to min, got {:?}",
            tel.draft_depth_k
        );
        assert!(
            spec_lm
                .scheduler
                .lock()
                .unwrap()
                .adaptation_state
                .accept_rate_ema
                < 0.1,
            "EMA acceptance must reflect the rejections"
        );
    }

    struct EmptyDraft {
        vocab: usize,
    }

    impl crate::draft_backbone::DraftBackbone for EmptyDraft {
        fn draft_block(
            &self,
            _session: &mut dyn SessionT,
            _context: &Tensor,
            _block_len: usize,
        ) -> Result<crate::draft_backbone::DraftBlock> {
            Ok(crate::draft_backbone::DraftBlock {
                tokens: Vec::new(),
                base_logits: grim_backend_cpu::cpu_tensor(
                    vec![0.0f32; self.vocab],
                    Shape::new(vec![1, self.vocab]),
                ),
                confidence: Vec::new(),
            })
        }

        fn estimated_footprint_bytes(&self) -> usize {
            0
        }

        fn update_weights(
            &self,
            _target_hidden_states: &[f32],
            _draft_tokens: &[u32],
            _accepted_mask: &[bool],
        ) -> Result<()> {
            Ok(())
        }
    }

    /// M10: an empty draft block must fall back to the plain target forward
    /// with identical logits. A DSpark path that unconditionally accepts (or
    /// fabricates) drafts would silently diverge here.
    #[test]
    fn test_empty_draft_falls_back_to_plain_target_forward() {
        let cfg = grim_models_transformer::LlamaConfig {
            vocab_size: 100,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 2,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            partial_rotary_factor: 1.0,
            yarn: None,
        ..Default::default()
        };
        let target = Box::new(MockCausalLm {
            cfg: cfg.clone(),
            device: Device::Cpu,
        });
        let draft = Arc::new(EmptyDraft { vocab: 100 });
        let markov = Arc::new(crate::uniform_markov_head::UniformMarkovHead::new(
            100, 5, 42,
        ));
        let confidence = Arc::new(crate::entropy_confidence_head::EntropyConfidenceHead);
        let scheduler =
            ConfidenceScheduler::new(ThroughputProfile::default(), SpeculationConfig::default());
        let spec_lm =
            SpeculativeCausalLm::with_dspark(target, draft, markov, confidence, scheduler);

        let mut session = spec_lm.new_session();
        let input_ids = grim_backend_cpu::cpu_tensor(vec![1f32], Shape::new(vec![1]));
        let positions = grim_backend_cpu::cpu_tensor(vec![0f32], Shape::new(vec![1]));
        let got = spec_lm
            .decode_one(session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();

        // Plain target forward on a fresh session: MockCausalLm emits
        // constant 0.1 logits for [seq, vocab].
        let plain = SpeculativeCausalLm::plain(Box::new(MockCausalLm {
            cfg,
            device: Device::Cpu,
        }));
        let mut plain_session = plain.new_session();
        let want = plain
            .decode_one(plain_session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();

        assert_eq!(got, want, "empty draft must equal plain forward exactly");
    }

    fn tiny_wrapper() -> SpeculativeCausalLm {
        let cfg = grim_models_transformer::LlamaConfig {
            vocab_size: 8,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 2,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            partial_rotary_factor: 1.0,
            yarn: None,
            ..Default::default()
        };
        SpeculativeCausalLm::plain(Box::new(MockCausalLm {
            cfg,
            device: Device::Cpu,
        }))
    }

    fn logits_from_probs(p: &[f32]) -> Tensor {
        grim_backend_cpu::cpu_tensor(
            p.iter().map(|x| x.ln()).collect(),
            Shape::new(vec![1, p.len()]),
        )
    }

    fn stacked_rows(rows: &[Tensor]) -> Tensor {
        let flat: Vec<f32> = rows.iter().flat_map(|t| t.to_vec_f32().unwrap()).collect();
        let vocab = rows[0].shape().dims()[1];
        grim_backend_cpu::cpu_tensor(flat, Shape::new(vec![rows.len(), vocab]))
    }

    /// The second emitted token of a one-token speculative step must follow
    /// the target distribution exactly, whether the draft was accepted (bonus
    /// row) or rejected (residual row). This is the losslessness contract.
    #[test]
    fn stochastic_verify_preserves_target_distribution() {
        let p = [0.30f32, 0.20, 0.15, 0.10, 0.09, 0.07, 0.06, 0.03];
        let q = [0.05f32, 0.05, 0.05, 0.25, 0.25, 0.15, 0.10, 0.10];
        let vocab = p.len();
        let target = stacked_rows(&[logits_from_probs(&p), logits_from_probs(&p)]);
        let draft = logits_from_probs(&q);
        let wrapper = tiny_wrapper();
        let mut rng = grim_core::rng::SimpleRng::new(0x1234_5678_9ABC_DEF0);
        let mut hist = vec![0usize; vocab];
        let trials = 20_000usize;

        for _ in 0..trials {
            // A real drafting step samples the proposal from q; the
            // losslessness proof requires averaging over that draw too.
            let u_draft = rng.next_f32();
            let mut cum = 0.0f32;
            let mut draft_tok = 0u32;
            for (v, &qv) in q.iter().enumerate() {
                cum += qv;
                if cum >= u_draft {
                    draft_tok = v as u32;
                    break;
                }
            }
            let uniforms: Vec<f32> = (0..2).map(|_| rng.next_f32()).collect();
            let mut clone = rng.clone();
            let outcome = wrapper
                .verify_accepted_count(
                    &target,
                    &draft,
                    0,
                    vocab,
                    &[draft_tok],
                    1,
                    &uniforms,
                    &mut clone,
                    false,
                )
                .unwrap();
            assert!(outcome.accepted_count <= 1);
            assert_eq!(outcome.emitted_tokens.len(), outcome.accepted_count + 1);
            hist[outcome.emitted_tokens[0] as usize] += 1;
        }
        for (v, (&want, &got)) in p.iter().zip(hist.iter()).enumerate() {
            let got = got as f32 / trials as f32;
            assert!(
                (got - want).abs() < 0.02,
                "token {v}: speculative emitted {got:.4}, target {want:.4}"
            );
        }
    }

    #[test]
    fn stochastic_verify_uses_pre_drawn_uniforms() {
        let p = [0.30f32, 0.20, 0.15, 0.10, 0.09, 0.07, 0.06, 0.03];
        let q = [0.05f32, 0.05, 0.05, 0.25, 0.25, 0.15, 0.10, 0.10];
        let vocab = p.len();
        let target = stacked_rows(&[logits_from_probs(&p), logits_from_probs(&p)]);
        let draft = logits_from_probs(&q);
        let wrapper = tiny_wrapper();
        let mut rng = grim_core::rng::SimpleRng::new(42);
        let accepted = wrapper
            .verify_accepted_count(
                &target,
                &draft,
                0,
                vocab,
                &[3],
                1,
                &[0.0, 0.999],
                &mut rng,
                false,
            )
            .unwrap();
        // p(3)/q(3) = 0.4; u=0 accepts, then bonus row is sampled from p.
        assert_eq!(accepted.accepted_count, 1);
        assert_eq!(accepted.emitted_tokens[0], 3);
        // u=0.999 on the bonus row lands in the last p bucket.
        assert_eq!(accepted.emitted_tokens[1], 7);
        let rejected = wrapper
            .verify_accepted_count(
                &target,
                &draft,
                0,
                vocab,
                &[3],
                1,
                &[0.999, 0.0],
                &mut rng,
                false,
            )
            .unwrap();
        assert_eq!(rejected.accepted_count, 0);
        assert_eq!(rejected.emitted_tokens.len(), 1);
        assert_eq!(rejected.emitted_tokens[0], 0);
    }

    #[test]
    fn tree_ranking_selects_highest_mass_path() {
        // Position 0 favours token 0, position 1 favours token 1; the
        // best-scoring path is [0, 1], not the placeholder draft [2, 2].
        let logits = [3.0f32, 0.0, 0.0, 0.0, 3.0, 0.0];
        let mut block = DraftBlock {
            tokens: vec![2, 2],
            base_logits: grim_backend_cpu::cpu_tensor(logits.to_vec(), Shape::new(vec![2, 3])),
            confidence: vec![],
        };
        assert!(SpeculativeCausalLm::apply_tree_ranking(&mut block, 2, true));
        assert_eq!(block.tokens, vec![0, 1]);
        assert_eq!(block.confidence.len(), 2);

        let mut untouched = DraftBlock {
            tokens: vec![2, 2],
            base_logits: grim_backend_cpu::cpu_tensor(logits.to_vec(), Shape::new(vec![2, 3])),
            confidence: vec![],
        };
        assert!(!SpeculativeCausalLm::apply_tree_ranking(&mut untouched, 2, false));
        assert_eq!(untouched.tokens, vec![2, 2]);
    }
}
