//! `NativeMtp` - model-native Multi-Token Prediction.
//! §5.3.1.

use grim_core::error::Result;
use grim_tensor::Tensor;

use crate::draft_backbone::DraftBlock;

/// Implemented directly by a target model that was trained with its own
/// multi-token-prediction head(s).
pub trait NativeMtp: grim_core::CausalLm {
    /// Returns the causal LM trait reference.
    fn as_causal_lm(&self) -> &dyn grim_core::CausalLm;

    /// How many extra tokens the model can natively predict ahead in one pass (vLLM's `num_speculative_tokens`).
    /// Typically small (1 is a good default).
    fn mtp_depth(&self) -> usize;

    /// Runs the trunk once and returns predictions for the next `mtp_depth()` positions, reusing
    /// the same KV cache entries the target forward pass would have written anyway.
    fn predict_multi(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<DraftBlock>;

    /// Like [`predict_multi`], but the base/trunk forward for `input_ids` has already been run by
    /// the caller and its logits are supplied as `base_logits`. This must **not** run the base
    /// forward again: re-appending the same input token at the same position double-counts a token
    /// and desynchronizes the KV cache (acceptance collapses to 0 and output garbles).
    ///
    /// The session's last hidden state is expected to be the output of the just-run base forward.
    /// Returns an empty block if the implementation cannot draft without a forward.
    fn predict_multi_after_forward(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        base_logits: &Tensor,
    ) -> Result<DraftBlock>;
}
