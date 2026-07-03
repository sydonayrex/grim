//! `ConfidenceHead` trait — predicted acceptance probability per position.
//!
//! §5.3.2. Trained jointly with the drafter against the target model's
//! own acceptance statistics. The score is consumed by `ConfidenceScheduler`
//! to decide how many positions to actually verify against the target on
//! each iteration, given the current load.

use crate::draft_backbone::DraftBlock;

/// Predicts acceptance probability per drafted position.
pub trait ConfidenceHead: Send + Sync {
    fn score(&self, draft_block: &DraftBlock) -> Vec<f32>;
}
