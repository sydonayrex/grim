//! `KvCache` trait — model-agnostic contract for serving inference state.
//!
//! `grim-memory` (§5.1) ships the paged-KV implementation; SSM/Mamba uses
//! a separate `SsmState` trait (in phase 7). The contract here is the
//! common interface every cache implementation honors; speculative-decoding
//! integration (§5.3) uses `tentative_append` / `commit` / `rollback_to`.

use grim_tensor::error::Result;
use grim_tensor::Tensor;

/// Block-addressed KV cache. Backed by a shared pool of physical blocks
/// (§5.1). Sequences address memory through a logical block table; the
/// physical blocks come from a `KvBlockPool`.
///
/// `tentative_append` / `commit` / `rollback_to` support speculative
/// decoding (§5.3): draft tokens are written provisionally, then either
/// committed (accepted prefix) or rolled back off before the next iteration.
pub trait KvCache: Send {
    /// Append a single slot for the next token.
    fn append_slot(&mut self) -> Result<()>;

    /// Tentatively append `n` slots for draft tokens. The slots are
    /// visible to subsequent forward passes but may be rolled back via
    /// `rollback_to` or committed via `commit`.
    fn tentative_append(&mut self, n: usize) -> Result<()>;

    /// After a speculative verification, commit the first `accepted_len`
    /// tentatively-appended slots and drop the tail.
    fn commit(&mut self, accepted_len: usize) -> Result<()>;

    /// Roll back to a previous length (in tokens). Used when the entire
    /// tentative prefix is rejected.
    fn rollback_to(&mut self, len: usize) -> Result<()>;

    /// Current logical length of the cache in tokens.
    fn len(&self) -> usize;

    /// True when the cache holds no tokens.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the keys tensor for the most recently appended slot(s).
    /// Shape: `(num_slots, num_kv_heads, head_dim)`.
    fn current_k(&self) -> Result<Tensor>;

    /// Return the values tensor for the most recently appended slot(s).
    fn current_v(&self) -> Result<Tensor>;
}
