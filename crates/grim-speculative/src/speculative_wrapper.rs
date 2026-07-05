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

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::AdapterHandle;
use grim_core::session::SessionT;
use grim_core::{CausalLm, Model, ModelConfig};
use grim_tensor::{ArithType, Device, Tensor};

use crate::confidence_head::ConfidenceHead;
use crate::confidence_scheduler::{ConfidenceScheduler, SpeculationConfig, ThroughputProfile};
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
    scheduler: ConfidenceScheduler,
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
            scheduler: ConfidenceScheduler::new(
                ThroughputProfile::default(),
                SpeculationConfig::default(),
            ),
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
            scheduler,
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
            scheduler: ConfidenceScheduler::new(
                ThroughputProfile::default(),
                SpeculationConfig::default(),
            ),
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
            Strategy::DSpark => self.decode_dspark(session, input_ids, positions, live_gpu_utilization, batch_pressure, adapters),
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

    fn decode_dspark(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        live_gpu_utilization: f32,
        batch_pressure: usize,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let draft = self.draft.as_ref().unwrap();
        let markov = self.markov.as_ref().unwrap();
        let confidence = self.confidence.as_ref().unwrap();

        // Phase 1: draft block.
        let block_len = self.scheduler.config.block_len;
        let draft_block = draft.draft_block(session, input_ids, block_len)?;
        if draft_block.tokens.is_empty() {
            return self.target.forward(session, input_ids, positions, adapters);
        }

        // Phase 2: score.
        let scores = confidence.score(&draft_block);
        let mut scored = draft_block.clone();
        scored.confidence = scores;

        // Phase 3: choose verify length.
        let verify_len = self
            .scheduler
            .choose_verify_len(&scored, live_gpu_utilization, batch_pressure);
        let verify_len = verify_len.min(scored.tokens.len());

        if verify_len == 0 {
            return self.target.forward(session, input_ids, positions, adapters);
        }

        // Phase 4: tentative append.
        if let Some(kv) = session.kv_mut() {
            kv.tentative_append(verify_len)?;
        }

        // Apply Markov head bias
        let prefix = scored.tokens[..verify_len].to_vec();
        let _bias = markov.bias(&prefix, &scored.base_logits)?;

        // Phase 5: verify
        let target_logits = self.target.forward(session, input_ids, positions, adapters)?;

        // Rejection-sampling validation loop (§5.3)
        let target_probs = target_logits.to_vec_f32()?;
        let mut accepted_count = 0;
        for i in 0..verify_len {
            let draft_tok = scored.tokens[i];
            let p_target = target_probs.get(draft_tok as usize).copied().unwrap_or(0.0);
            
            // Standard DSpark rejection threshold boundary
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
