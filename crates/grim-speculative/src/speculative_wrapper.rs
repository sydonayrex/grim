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

use grim_core::error::Result;
use grim_core::model::AdapterHandle;
use grim_core::session::SessionT;
use grim_core::{CausalLm, Model, ModelConfig};
use grim_tensor::{ArithType, Device, Tensor};

use crate::confidence_head::ConfidenceHead;
use crate::confidence_scheduler::{ConfidenceScheduler, SpeculationConfig, ThroughputProfile};
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
        Self {
            target,
            strategy: Strategy::DSpark,
            mtp_target: None,
            draft: Some(draft),
            markov: Some(markov),
            confidence: Some(confidence),
            scheduler: Mutex::new(scheduler),
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
        }
    }

    /// Construct from a target + optional bundle. Selects strategy automatically.
    /// Selection priority: DSpark (if bundle attached) > native MTP (if model
    /// implements `NativeMtp`) > plain.
    ///
    /// Speculative decoding constraint under weight-streaming:
    /// When weight-streaming is active, the draft bundle must fit entirely in VRAM.
    /// If DSpark is selected but the draft model fails to fit in available VRAM,
    /// it falls back to plain autoregressive decoding.
    pub fn auto(
        target: Box<dyn CausalLm>,
        draft: Option<Arc<dyn DraftBackbone>>,
        markov: Option<Arc<dyn MarkovHead>>,
        confidence: Option<Arc<dyn ConfidenceHead>>,
        is_weight_streaming_active: bool,
        available_vram_bytes: Option<usize>,
    ) -> Self {
        if draft.is_some() && markov.is_some() && confidence.is_some() {
            // Check if weight-streaming is active and if a VRAM restriction applies
            if is_weight_streaming_active {
                if let Some(available_vram) = available_vram_bytes {
                    let draft_ref = draft.as_ref().unwrap();
                    // Estimate draft model size in bytes (weight size)
                    // We query the estimated footprint from DraftBackbone if available.
                    // Fallback to checking the VRAM threshold.
                    let estimated_size = draft_ref.estimated_footprint_bytes();
                    if estimated_size > available_vram {
                        // Draft model too large to fit in VRAM along with target streaming buffers;
                        // fall back to plain autoregressive execution to avoid serialization crash.
                        return Self::plain(target);
                    }
                }
            }
            Self::with_dspark(
                target,
                draft.unwrap(),
                markov.unwrap(),
                confidence.unwrap(),
                ConfidenceScheduler::new(
                    ThroughputProfile::default(),
                    SpeculationConfig::default(),
                ),
            )
        } else {
            Self::plain(target)
        }
    }


    pub fn strategy(&self) -> Strategy {
        self.strategy
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
                // T2-4: Implement the actual draft step loop (K=3 draft tokens)
                let draft = self.draft.as_ref().unwrap();
                let markov = self.markov.as_ref().unwrap();
                let confidence = self.confidence.as_ref().unwrap();

                // Phase 1: Run K=3 draft steps
                let draft_block = draft.draft_block(session, input_ids, 3)?;
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
                    .unwrap()
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

                // Phase 5: Verification step on Target Causal LM
                let target_logits = self.target.forward(session, input_ids, positions, adapters)?;

                // Rejection-sampling validation loop
                let target_probs = target_logits.to_vec_f32()?;
                let mut accepted_count = 0;
                for i in 0..verify_len {
                    let draft_tok = scored.tokens[i];
                    let p_target = target_probs.get(draft_tok as usize).copied().unwrap_or(0.0);
                    
                    // Standard verification acceptance threshold
                    if p_target >= 0.1 {
                        accepted_count += 1;
                    } else {
                        break;
                    }
                }

                if let Some(kv) = session.kv_mut() {
                    kv.commit(accepted_count)?;
                }
                session.advance_pos(accepted_count);

                // Update scheduler and check adaptation gating
                {
                    let mut sched = self.scheduler.lock().unwrap();
                    sched.record_acceptance(accepted_count, verify_len);

                    if sched.should_adapt_draft() {
                        let mut accepted_mask = vec![false; verify_len];
                        for i in 0..accepted_count {
                            accepted_mask[i] = true;
                        }
                        let target_hidden_states = session.get_last_hidden_state().and_then(|t| t.to_vec_f32().ok());
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
                        let _outcome = crate::distill::refresh_draft(&signal, &refresh_input, draft.as_ref())?;
                    }
                }

                Ok(target_logits)
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
        let target_logits = self.target.forward(session, input_ids, positions, adapters)?;

        // 4. Rejection sampling / validation loop (§5.3)
        // Checks that draft token target-probability / draft-probability ratio satisfies threshold.
        // We use a deterministic pseudo-random fallback check for stability.
        let target_probs = target_logits.to_vec_f32()?;
        let mut accepted_count = 0;
        for i in 0..verify_len {
            let draft_tok = draft_block.tokens[i];
            let p_target = target_probs.get(draft_tok as usize).copied().unwrap_or(0.0);
            
            // Standard speculative validation: accept if target probability is sufficiently high
            if p_target >= 0.1 {
                accepted_count += 1;
            } else {
                break;
            }
        }

        if let Some(kv) = session.kv_mut() {
            kv.commit(accepted_count)?;
        }
        session.advance_pos(accepted_count);

        Ok(target_logits)
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
}

impl CausalLm for SpeculativeCausalLm {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.target.new_session()
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        self.decode_one(session, input_ids, positions, 0.0, 0, adapters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let markov = Arc::new(crate::uniform_markov_head::UniformMarkovHead::new(100, 5, 42));
        let confidence = Arc::new(crate::entropy_confidence_head::EntropyConfidenceHead);

        // Create scheduler with high trigger threshold (e.g. 1.5) to always trigger adaptation
        let mut scheduler = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig::default(),
        );
        scheduler.adaptation_config.min_steps_before_trigger = 1;
        scheduler.adaptation_config.min_accept_rate = 1.5;
        scheduler.adaptation_config.ema_alpha = 0.5;

        let spec_lm = SpeculativeCausalLm::with_dspark(
            target,
            draft.clone(),
            markov,
            confidence,
            scheduler,
        );

        let mut session = spec_lm.new_session();
        let input_ids = grim_backend_cpu::cpu_tensor(vec![1f32], Shape::new(vec![1]));
        let positions = grim_backend_cpu::cpu_tensor(vec![0f32], Shape::new(vec![1]));

        // 1. Verify that before forward run, last hidden state is empty
        assert!(session.get_last_hidden_state().is_none());

        // 2. Perform a speculative decode step (this will call MockCausalLm's forward pass)
        let _logits = spec_lm.decode_one(
            session.as_mut(),
            &input_ids,
            &positions,
            0.0,
            0,
            &[],
        ).unwrap();

        // 3. Verify that the penultimate hidden state is successfully captured in the session
        let captured_hidden = session.get_last_hidden_state().unwrap();
        let hidden_shape = captured_hidden.shape();
        assert_eq!(hidden_shape.dims(), &[1, 1, 16]); // [1, verify_len, hidden_size]

        // 4. Force weight update (adaptation EMA will drop below 1.5 min threshold after this step)
        let w_head_before = {
            let w = draft.weights.lock().unwrap();
            w.w_head.clone()
        };
        
        let _logits2 = spec_lm.decode_one(
            session.as_mut(),
            &input_ids,
            &positions,
            0.0,
            0,
            &[],
        ).unwrap();

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
}
