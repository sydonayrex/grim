//! `ConfidenceHead` trait - predicted acceptance probability per position.
//! §5.3.2.

use crate::draft_backbone::DraftBlock;

/// Predicts acceptance probability per drafted position.
pub trait ConfidenceHead: Send + Sync {
    fn score(&self, draft_block: &DraftBlock) -> Vec<f32>;
}
