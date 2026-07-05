//! `NativeMtp` — model-native Multi-Token Prediction.
//!
//! §5.3.1. A target model trained with its own MTP head(s) shares KV
//! cache and trunk compute with the target. Zero-config speculation: no
//! separate drafter, no distillation, no bundle. Just ask the model for
//! `mtp_depth()` extra speculative positions on every decode step it
//! already runs.

use grim_core::error::Result;
use grim_tensor::Tensor;

use crate::draft_backbone::DraftBlock;

/// Implemented directly by a target model that was trained with its own
/// multi-token-prediction head(s).
pub trait NativeMtp: grim_core::CausalLm {
    /// Returns the causal LM trait reference.
    fn as_causal_lm(&self) -> &dyn grim_core::CausalLm;

    /// How many extra tokens the model can natively predict ahead in one
    /// pass (vLLM's `num_speculative_tokens`). Typically small (1 is a
    /// good default).
    fn mtp_depth(&self) -> usize;

    /// Runs the trunk once and returns predictions for the next
    /// `mtp_depth()` positions, reusing the same KV cache entries the
    /// target forward pass would have written anyway.
    fn predict_multi(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<DraftBlock>;
}
