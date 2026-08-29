//! `KvCache` trait — model-agnostic contract for serving inference state.
//!
//! `grim-memory` (§5.1) ships the paged-KV implementation; SSM/Mamba uses
//! a separate `SsmState` trait (in phase 7). The contract here is the
//! common interface every cache implementation honors; speculative-decoding
//! integration (§5.3) uses `tentative_append` / `commit` / `rollback_to`.

use grim_tensor::Tensor;

use crate::error::Result;

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

    /// Store key/value tensors into the most recently allocated slot.
    /// Called by `Session::append_kv` after `append_slot()` to write
    /// the actual K/V data into the block identified by the slot.
    fn store_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()>;

    /// True when this cache is backed by a paged KV store that can be fed
    /// directly to the paged-attention kernel (physical page tensors +
    /// a logical block table). Models consult this to decide whether to
    /// dispatch through `append_kv_layer` + `paged_kv_handles`.
    fn has_paged_kv(&self) -> bool {
        false
    }

    /// Append the key/value tensors for one layer into the paged store.
    /// `k`/`v` carry `seq` tokens (shape `[seq, num_kv_heads, head_dim]` or
    /// `[seq, num_kv_heads * head_dim]`); each token is written into its own
    /// slot in the layer's page tensor. Per-layer page buffers are grown
    /// lazily, so no upfront layer count is required.
    fn append_kv_layer(&mut self, _layer: usize, _k: &Tensor, _v: &Tensor) -> Result<()> {
        Ok(())
    }

    /// Return the logical→physical block table (as u32 physical ids), if a
    /// paged store is active and at least one block has been allocated.
    fn block_table(&self) -> Option<&[u32]> {
        None
    }

    /// Return the physical K/V page tensors for `layer` plus the page size,
    /// if a paged store is active for that layer.
    fn paged_kv_handles(&self, _layer: usize) -> Option<(Tensor, Tensor, usize)> {
        None
    }

    /// Seed the start of the cache with already-computed prefix blocks from
    /// a shared pool (RadixAttention-style prefix reuse, §5.1). The default
    /// is a no-op for caches that don't support cross-sequence sharing.
    fn seed_prefix(&mut self, _blocks: &[usize]) {}

    /// Return the physical block ids currently backing this cache, in token
    /// order. Used by the engine to register a computed prefix into the
    /// shared pool's radix tree. Empty by default.
    fn prefix_physical_ids(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Return the number of active layers in the paged store.
    fn num_layers(&self) -> usize {
        0
    }

    /// Extract key and value slices for a given layer and physical block ID.
    fn layer_block_slice(&self, _layer: usize, _block_id: usize) -> Option<(&[f32], &[f32])> {
        None
    }

    /// Valid token count stored in physical block `block_id` (handoffs must
    /// preserve it; deriving from the zero-padded buffer length would mark
    /// every block fully valid). `None` when the cache does not track
    /// per-block fill state.
    fn block_num_tokens(&self, _block_id: usize) -> Option<usize> {
        None
    }

    /// Write raw key and value slices directly into a physical block for a layer (e.g. for disagg network ingestion).
    fn write_layer_block(
        &mut self,
        _layer: usize,
        _block_id: usize,
        _k: &[f32],
        _v: &[f32],
    ) -> Result<()> {
        Ok(())
    }
}
