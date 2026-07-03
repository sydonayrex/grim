//! `Sampler` trait — token selection from logits.
//!
//! Concrete samplers (greedy, top-k, nucleus, mirostat, ...) implement this
//! trait; plugins (§6) provide extensions via either the dylib or WASM path.

use grim_tensor::error::Result;
use grim_tensor::Tensor;

/// History-aware token sampler. The `history` argument carries the most
/// recently emitted tokens (typically the last 64 tokens) for samplers
/// that need repetition context (DRY, mirostat variants, etc.).
pub trait Sampler: Send + Sync {
    /// Sample one token from the logits distribution.
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32>;

    /// Human-readable name for logs / sampler registry.
    fn name(&self) -> &str;
}
