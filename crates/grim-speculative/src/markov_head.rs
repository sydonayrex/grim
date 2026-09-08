//! `MarkovHead` trait - the lightweight sequential correction.
//! §5.3.2.

use grim_core::error::Result;
use grim_tensor::Tensor;

/// Sequential dependency adjuster. Produces a bias tensor of the same shape
/// as the base logits.
pub trait MarkovHead: Send + Sync {
    fn bias(&self, prefix_within_block: &[u32], base_logits: &Tensor) -> Result<Tensor>;
}
