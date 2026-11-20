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

    /// Optional m=2 DecodeGraph capture/replay hook for 1-pass batched MTP.
    /// Returns `Ok(Some(logits))` with `[batch, vocab]` logits when handled by the
    /// graph, or `Ok(None)` to fall back to eager `CausalLm::forward`.
    fn forward_m2_graph(
        &self,
        _graph_slot: &mut Option<Box<dyn std::any::Any + Send>>,
        _session: &mut dyn grim_core::session::SessionT,
        _tokens: &[u32],
        _pos: u32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Restore the recurrent state inside `graph_slot` to the pre-step D2D snapshot
    /// when a speculative verification step rejects one or more draft tokens.
    fn restore_m2_graph_recurrent(
        &self,
        _graph_slot: &mut Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<()> {
        Ok(())
    }

    /// Publish row `row` of the graph's `head_input` into `session.last_hidden_state`
    /// so `predict_multi_after_forward` can draft from row `accepted_count` without re-running the trunk.
    fn prewarm_m2_graph(
        &self,
        _graph_slot: &mut Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<()> {
        Ok(())
    }

    fn publish_m2_graph_hidden_row(
        &self,
        _graph_slot: &mut Option<Box<dyn std::any::Any + Send>>,
        _session: &mut dyn grim_core::session::SessionT,
        _row: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Roll back the graph's recurrent + KV state from a verify pass of `1 + verify_len` tokens
    /// at `base_pos` to `base_pos + 1 + accepted_count` using in-place causal state replay over
    /// the already-computed projections (zero GEMVs, zero trunk graph re-run).
    fn rollback_m2_graph_to_accepted(
        &self,
        _graph_slot: &mut Option<Box<dyn std::any::Any + Send>>,
        _base_pos: u32,
        _accepted_tokens: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Commit M=2 speculative KDA state (`S_spec -> S`) when all draft tokens are accepted.
    fn commit_m2_graph_spec_state(
        &self,
        _graph_slot: &mut Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<()> {
        Ok(())
    }
}
