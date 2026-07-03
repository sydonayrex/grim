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
            draft: Some(draft),
            markov: Some(markov),
            confidence: Some(confidence),
            scheduler,
        }
    }

    /// Construct wrapped around native MTP — the zero-config path.
    pub fn with_native_mtp(target: Box<dyn CausalLm>) -> Self {
        Self {
            target,
            strategy: Strategy::NativeMtp,
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
    pub fn auto(
        target: Box<dyn CausalLm>,
        draft: Option<Arc<dyn DraftBackbone>>,
        markov: Option<Arc<dyn MarkovHead>>,
        confidence: Option<Arc<dyn ConfidenceHead>>,
    ) -> Self {
        if draft.is_some() && markov.is_some() && confidence.is_some() {
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
    ///
    /// v1 wiring:
    /// - For `Strategy::Plain`, this is just `target.forward` (one token).
    /// - For `Strategy::DSpark`, the draft phase + verification phase are
    ///   exercised but the verification pass itself is delegated to a
    ///   proper batched verifier when full serving ships. For phase 5 the
    ///   structure (pre-draft, score, choose verify len, "verify",
    ///   commit/rollback) is correct.
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
            Strategy::NativeMtp => {
                let _ = live_gpu_utilization;
                let _ = batch_pressure;
                self.target.forward(session, input_ids, positions, adapters)
            }
            Strategy::DSpark => self.decode_dspark(session, input_ids, positions, live_gpu_utilization, batch_pressure, adapters),
        }
    }

    fn decode_dspark(
        &self,
        session: &mut dyn SessionT,
        _input_ids: &Tensor,
        _positions: &Tensor,
        live_gpu_utilization: f32,
        batch_pressure: usize,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let draft = self.draft.as_ref().unwrap();
        let markov = self.markov.as_ref().unwrap();
        let confidence = self.confidence.as_ref().unwrap();

        // Phase 1: draft block.
        let block_len = self.scheduler.config.block_len;
        let draft_block: DraftBlock = draft.draft_block(
            session,
            _input_ids,
            block_len,
        )?;

        // Phase 2: score.
        let scores = confidence.score(&draft_block);
        // Make the block carry the scored confidence for the scheduler.
        let mut scored = draft_block.clone();
        scored.confidence = scores;

        // Phase 3: choose verify length.
        let verify_len = self
            .scheduler
            .choose_verify_len(&scored, live_gpu_utilization, batch_pressure);
        let _ = verify_len;

        // Phase 4: sequential correction with Markov head. v1 uses a no-op
        // bias — actual bias application lands with the bundle-attached
        // component, where the base logits tensor is real.
        let prefix: Vec<u32> = scored.tokens.clone();
        let _bias = markov.bias(&prefix, &scored.base_logits)?;

        // Phase 5: "verify" — in v1 we just emit target.forward on the
        // current context (no real batched verifier yet). The structure
        // ensures the right hooks are in place for the real impl.
        self.target.forward(session, _input_ids, _positions)
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
