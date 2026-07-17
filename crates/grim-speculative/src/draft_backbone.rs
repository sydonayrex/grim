//! `DraftBackbone` trait — the O(1) parallel drafter.
//!
//! §5.3.2. Drafts a block of candidate tokens in one forward pass rather
//! than autoregressively; the speed is what suppresses the cost of the
//! speculative draft step.

use grim_core::error::Result;
use grim_tensor::Tensor;

/// Parallel drafter backbone. One forward pass over a block-sized draft
/// window, producing base logits for every position simultaneously.
pub trait DraftBackbone: Send + Sync {
    /// Draft a block of `block_len` tokens given a context tensor and an
    /// open session.
    fn draft_block(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        context: &Tensor,
        block_len: usize,
    ) -> Result<DraftBlock>;

    /// Estimate the VRAM footprint of the draft model in bytes.
    fn estimated_footprint_bytes(&self) -> usize;

    /// Update the draft model's weights using target hidden states and acceptance feedback.
    fn update_weights(
        &self,
        target_hidden_states: &[f32],
        draft_tokens: &[u32],
        accepted_mask: &[bool],
    ) -> Result<()>;
}


/// One drafted candidate block.
#[derive(Clone)]
pub struct DraftBlock {
    pub tokens: Vec<u32>,
    /// `(block_len, vocab)` — base logits per position.
    pub base_logits: Tensor,
    /// Per-position acceptance probability estimate (one per token slot).
    pub confidence: Vec<f32>,
}

impl DraftBlock {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
