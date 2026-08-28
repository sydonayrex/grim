//! `SpeculativeCausalLm` — the default-on wrapper that turns a plain
//! `CausalLm` into a speculatively-accelerated one.
//!
//! §5.3. Architecture:
//! - If the target implements `NativeMtp`, use that (zero-config).
//! - Else if a `DraftBackbone` + `MarkovHead` + `ConfidenceHead`
//!   bundle is attached, use the DSpark path.
//! - Else fall back to plain autoregressive decoding.
//!
//! Callers of `CausalLm::forward` never see the wrapper; it's chosen at
//! model-load time based on what the model supports.

use std::sync::{Arc, Mutex};

use grim_core::error::{Error, Result};
use grim_core::model::AdapterHandle;
use grim_core::session::SessionT;
use grim_core::{CausalLm, Model, ModelConfig};
use grim_tensor::{ArithType, Device, Tensor};

use crate::confidence_head::ConfidenceHead;
use crate::confidence_scheduler::{ConfidenceScheduler, SpeculationConfig, ThroughputProfile};
use crate::depth_tuner::SpeculativeDepthPidController;
use crate::draft_backbone::DraftBackbone;
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
    /// Current PID-tuned draft depth for the DSpark path (None on other
    /// strategies). T2-4 follow-up: the depth tuner now drives the draft
    /// block length instead of the old hardcoded K=3.
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
        }
    }

    /// Construct from a target + optional bundle or native MTP. Selects strategy automatically.
    /// Selection priority: DSpark (if bundle attached) > native MTP (if model
    /// implements `NativeMtp`) > plain.
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

    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// Query runtime speculative decoding telemetry snapshot.
    pub fn telemetry(&self) -> SpeculativeTelemetry {
        let (state, config) = {
            let sched = self.scheduler.lock().unwrap();
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
            draft_depth_k: if self.strategy == Strategy::DSpark {
                Some(self.depth_tuner.lock().unwrap().current_depth() as u64)
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
        match self.strategy {
            Strategy::Plain => self.target.forward(session, input_ids, positions, adapters),
            Strategy::NativeMtp => self.decode_native_mtp(session, input_ids, positions, adapters),
            Strategy::DSpark => {
                let draft = self.draft.as_ref().unwrap();
                let markov = self.markov.as_ref().unwrap();
                let confidence = self.confidence.as_ref().unwrap();

                // T2-4 (closed): the draft block length K is PID-tuned online
                // from observed acceptance — not the old hardcoded 3.
                let draft_depth_k = self.depth_tuner.lock().unwrap().current_depth();
                let draft_block = draft.draft_block(session, input_ids, draft_depth_k)?;
                if draft_block.tokens.is_empty() {
                    return self.target.forward(session, input_ids, positions, adapters);
                }

                // Phase 2: Score confidence
                let scores = confidence.score(&draft_block);
                let mut scored = draft_block.clone();
                scored.confidence = scores;

                // Phase 3: Choose verify length dynamically
                let verify_len = self.scheduler.lock().unwrap().choose_verify_len(
                    &scored,
                    live_gpu_utilization,
                    batch_pressure,
                );
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

                // Phase 5: Verification step on Target Causal LM
                // CRIT-3: The target must receive the extended input (original + draft tokens)
                // to produce logits for all draft positions
                let extended_input =
                    self.extend_input_ids(input_ids, &scored.tokens[..verify_len])?;
                let extended_positions = self.extend_positions(positions, verify_len)?;
                let target_logits =
                    self.target
                        .forward(session, &extended_input, &extended_positions, adapters)?;
                let target_probs = target_logits.to_vec_f32()?;
                let vocab_size = scored.base_logits.shape().dims()[1];
                let draft_logits = scored.base_logits.to_vec_f32()?;

                // CRIT-4: Use per-request RNG instead of global rand::random()
                let mut rng = session
                    .request_rng()
                    .cloned()
                    .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15));

                // Rejection-sampling validation loop (§5.3)
                // Correctly index logits as flat [seq, vocab_size] row-major,
                // apply per-row softmax, and use the standard ratio test with per-request randomness.
                // Target logits cover [0..context_len) rows (original input positions).
                // Draft logits cover [0..verify_len) rows (draft positions).
                // Verify draft position i against target position (context_len + i).
                // [P1-18 fix: unified indexing — context_len = original input length.]
                let context_len = input_ids.shape().elem_count();
                let mut accepted_count = 0;
                for i in 0..verify_len {
                    let draft_tok = scored.tokens[i] as usize;

                    let row_start = (context_len + i) * vocab_size;
                    let row_end = row_start + vocab_size;
                    let p_target = softmax_f32_row(&target_probs[row_start..row_end])[draft_tok];
                    let draft_row_start = i * vocab_size;
                    let draft_row_end = draft_row_start + vocab_size;
                    let p_draft =
                        softmax_f32_row(&draft_logits[draft_row_start..draft_row_end])[draft_tok];

                    // Ratio test: accept with min(1, p_target / p_draft).
                    let p_accept = if p_draft > 1e-10 {
                        (p_target / p_draft).min(1.0)
                    } else {
                        0.0
                    };
                    if rng.next_f32() < p_accept {
                        accepted_count += 1;
                    } else {
                        break;
                    }
                }

                // CRIT-5: Properly commit/rollback KV cache based on accepted count.
                // tentative_append(verify_len) must have added >= accepted_count slots.
                // [P1-20 fix: assert verify_len >= accepted_count before commit.]
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
                    let mut sched = self.scheduler.lock().unwrap();
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

                // Return logits for the accepted tokens. Accepted token rows start at
                // context_len (the original input length) since target forward returned
                // [context_len + verify_len, vocab_size] and draft positions map to
                // target positions context_len + i. [P1-18 fix.]
                let accepted_logits = self.extract_accepted_logits(
                    &target_logits,
                    accepted_count,
                    vocab_size,
                    context_len,
                )?;
                session.set_last_accepted_tokens(accepted_count);
                Ok(accepted_logits)
            }
        }
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
        if depth == 0 {
            return self.target.forward(session, input_ids, positions, adapters);
        }

        // 1. Natively predict speculative tokens
        let draft_block = mtp.predict_multi(session, input_ids, positions)?;
        if draft_block.tokens.is_empty() {
            return self.target.forward(session, input_ids, positions, adapters);
        }

        let verify_len = draft_block.tokens.len().min(depth);

        // 2. Tentative append to KV Cache
        if let Some(kv) = session.kv_mut() {
            kv.tentative_append(verify_len)?;
        }

        // 3. Verify
        // CRIT-3: Target must receive extended input
        let extended_input = self.extend_input_ids(input_ids, &draft_block.tokens[..verify_len])?;
        let extended_positions = self.extend_positions(positions, verify_len)?;
        let target_logits =
            self.target
                .forward(session, &extended_input, &extended_positions, adapters)?;

        // 4. Rejection sampling / validation loop (§5.3)
        // The target logits tensor has shape [S + verify_len, vocab_size] where S =
        // len(original_input). Draft token i's target logit row is at offset
        // (S + i) * vocab_size, NOT i * vocab_size.
        // Draft logits have shape [verify_len, vocab_size] — no offset needed.
        let target_probs = target_logits.to_vec_f32()?;
        let vocab_size = draft_block.base_logits.shape().dims()[1];
        let draft_logits = draft_block.base_logits.to_vec_f32()?;
        let context_len = input_ids.shape().elem_count();

        let mut accepted_count = 0;
        let mut rng = session
            .request_rng()
            .cloned()
            .unwrap_or_else(|| grim_core::rng::SimpleRng::new(0x9E37_79B9_7F4A_7C15));
        for i in 0..verify_len {
            let draft_tok = draft_block.tokens[i] as usize;

            // Target row at (context_len + i), draft row at i
            let target_row_start = (context_len + i) * vocab_size;
            let target_row_end = target_row_start + vocab_size;
            let draft_row_start = i * vocab_size;
            let draft_row_end = draft_row_start + vocab_size;
            let p_target =
                softmax_f32_row(&target_probs[target_row_start..target_row_end])[draft_tok];
            let p_draft = softmax_f32_row(&draft_logits[draft_row_start..draft_row_end])[draft_tok];

            let p_accept = if p_draft > 1e-10 {
                (p_target / p_draft).min(1.0)
            } else {
                0.0
            };
            if rng.next_f32() < p_accept {
                accepted_count += 1;
            } else {
                break;
            }
        }

        if accepted_count > verify_len {
            return Err(Error::Session(format!(
                "speculative KV contract violation (NativeMTP): accepted_count ({accepted_count}) > \
                 verify_len ({verify_len})"
            )));
        }
        if let Some(kv) = session.kv_mut() {
            kv.commit(accepted_count)?;
        }

        {
            let mut sched = self.scheduler.lock().unwrap();
            sched.record_acceptance(accepted_count, verify_len);
        }

        // CRIT-7: Return logits for accepted tokens (rows S..S+accepted)
        let accepted_logits =
            self.extract_accepted_logits(&target_logits, accepted_count, vocab_size, context_len)?;
        session.set_last_accepted_tokens(accepted_count);
        Ok(accepted_logits)
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
            // last_pos will be -1.0, so we get 0, 1, 2, ... — correct.
            // Guard: ensure we never emit a negative position.
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

    /// Extract logits for only the accepted tokens.
    /// `context_len` is the length of the original (non-draft) input;
    /// accepted token rows start at offset `context_len * vocab_size`.
    fn extract_accepted_logits(
        &self,
        target_logits: &Tensor,
        accepted_count: usize,
        vocab_size: usize,
        context_len: usize,
    ) -> Result<Tensor> {
        let all_logits = target_logits.to_vec_f32()?;
        let start = context_len * vocab_size;
        let end = start + accepted_count * vocab_size;
        if end > all_logits.len() {
            return Err(grim_core::error::Error::Config(format!(
                "extract_accepted_logits: slice [{start}..{end}] out of bounds (logits len={})",
                all_logits.len()
            )));
        }
        let accepted_logits = all_logits[start..end].to_vec();
        let shape = grim_tensor::Shape::new(vec![accepted_count, vocab_size]);
        Ok(grim_backend_cpu::cpu_tensor(accepted_logits, shape))
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
        // Hidden state shape is [1, 1 + verify_len, hidden_size] because the
        // target receives the extended input (original token + draft tokens).
        // choose_verify_len enforces min_verify_len=1, so verify_len >= 1.
        assert_eq!(hidden_shape.dims(), &[1, 2, 16]); // [1, 1+verify_len, hidden_size]

        // 4. Force weight update (adaptation EMA will drop below 1.5 min threshold after this step)
        let w_head_before = {
            let w = draft.weights.lock().unwrap();
            w.w_head.clone()
        };

        let _logits2 = spec_lm
            .decode_one(session.as_mut(), &input_ids, &positions, 0.0, 0, &[])
            .unwrap();

        // Check that the scheduler registered the step and triggered adaptation
        let sched = spec_lm.scheduler.lock().unwrap();
        assert!(sched.adaptation_state.steps_observed >= 2);
        assert!(sched.adaptation_state.accept_rate_ema < 1.5);

        // Check that weights were indeed updated (nudge applied)
        let w_head_after = {
            let w = draft.weights.lock().unwrap();
            w.w_head.clone()
        };
        assert_ne!(w_head_before, w_head_after);
    }

    /// T2-4 (closed): the DSpark draft depth K is PID-tuned, not hardcoded 3.
    /// With a target that accepts every draft (uniform logits ⇒ ratio 1), the
    /// tuner ramps K to its max and the drafted block grows accordingly.
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

    /// A drafter whose base logits are one-hot on its drafted token against a
    /// uniform target gives p_target/p_draft ≈ 0 — every draft rejected, so
    /// the tuner must collapse K to its minimum instead of staying at 3.
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
}
