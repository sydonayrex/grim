//! `MarkovHead` trait — the lightweight sequential correction.
//!
//! §5.3.2. A low-rank (rank-256 in DSpark-style configs) prefix-conditioned
//! bias applied to the base logits BEFORE each in-block token is sampled.
//! This is the "semi" in semi-autoregressive: still one backbone pass, but
//! each position now depends on the tokens already chosen earlier in the
//! same block — suppressing suffix decay.

use grim_core::error::Result;
use grim_tensor::Tensor;

/// Sequential dependency adjuster. Produces a bias tensor of the same shape
/// as the base logits.
pub trait MarkovHead: Send + Sync {
    fn bias(
        &self,
        prefix_within_block: &[u32],
        base_logits: &Tensor,
    ) -> Result<Tensor>;
}
