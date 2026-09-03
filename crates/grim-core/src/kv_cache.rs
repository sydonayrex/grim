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

    /// Device-resident block table matching the `grim_qkv_attention_paged`
    /// kernel's `BlockTableEntry` ABI, cached across decode steps so the
    /// paged-attention path doesn't re-upload it per layer per token.
    /// Default `None` — implementors without a device buffer fall back to the
    /// host-side upload each call.
    fn block_table_gpu_handle(
        &self,
    ) -> Option<std::sync::Arc<dyn grim_tensor::backend::BackendStorage>> {
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

/// A lightweight in-memory [`KvCache`] for tests (audit gap: callers
/// previously had to pull in the heavyweight grim-memory paged store to test
/// against the trait). Stores per-token K/V rows in flat vectors and honors
/// the speculative `tentative_append`/`commit`/`rollback_to` contract by
/// tracking a tentative length separately from the committed length.
#[derive(Debug, Clone, Default)]
pub struct MockKvCache {
    num_heads: usize,
    head_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    committed: usize,
    tentative: usize,
}

impl MockKvCache {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            k: Vec::new(),
            v: Vec::new(),
            committed: 0,
            tentative: 0,
        }
    }

    pub fn committed_len(&self) -> usize {
        self.committed
    }
}

impl KvCache for MockKvCache {
    fn append_slot(&mut self) -> Result<()> {
        self.committed += 1;
        let row = self.num_heads * self.head_dim;
        self.k.resize(self.committed * row, 0.0);
        self.v.resize(self.committed * row, 0.0);
        Ok(())
    }

    fn tentative_append(&mut self, n_tokens: usize) -> Result<()> {
        self.tentative += n_tokens;
        let total = self.committed + self.tentative;
        let row = self.num_heads * self.head_dim;
        self.k.resize(total * row, 0.0);
        self.v.resize(total * row, 0.0);
        Ok(())
    }

    fn commit(&mut self, accepted_len: usize) -> Result<()> {
        self.committed += accepted_len.min(self.tentative);
        self.tentative = 0;
        let row = self.num_heads * self.head_dim;
        self.k.truncate(self.committed * row);
        self.v.truncate(self.committed * row);
        Ok(())
    }

    fn rollback_to(&mut self, len: usize) -> Result<()> {
        self.committed = self.committed.min(len);
        self.tentative = 0;
        let row = self.num_heads * self.head_dim;
        self.k.truncate(self.committed * row);
        self.v.truncate(self.committed * row);
        Ok(())
    }

    fn len(&self) -> usize {
        self.committed + self.tentative
    }

    fn current_k(&self) -> Result<Tensor> {
        let shape = grim_tensor::Shape::new(vec![self.len(), self.num_heads, self.head_dim]);
        Ok(grim_backend_cpu::cpu_tensor(self.k.clone(), shape))
    }

    fn current_v(&self) -> Result<Tensor> {
        let shape = grim_tensor::Shape::new(vec![self.len(), self.num_heads, self.head_dim]);
        Ok(grim_backend_cpu::cpu_tensor(self.v.clone(), shape))
    }

    fn store_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let row = self.num_heads * self.head_dim;
        let start = (self.len().saturating_sub(1)) * row;
        let kv = k.to_vec_f32()?;
        let vv = v.to_vec_f32()?;
        let take = kv.len().min(self.k.len().saturating_sub(start));
        self.k[start..start + take].copy_from_slice(&kv[..take]);
        let take = vv.len().min(self.v.len().saturating_sub(start));
        self.v[start..start + take].copy_from_slice(&vv[..take]);
        Ok(())
    }
}

#[cfg(test)]
mod mock_kv_cache_tests {
    use super::*;

    /// The mock must honor the speculative contract: tentative slots are
    /// visible in `len` but discarded by rollback, and commit folds only
    /// the accepted count.
    #[test]
    fn mock_kv_cache_speculative_contract() {
        let mut kv = MockKvCache::new(2, 4);
        assert!(kv.is_empty());
        for _ in 0..4 {
            kv.append_slot().unwrap();
        }
        assert_eq!(kv.len(), 4);
        kv.tentative_append(3).unwrap();
        assert_eq!(kv.len(), 7);
        // Commit 1 of 3 drafts.
        kv.commit(1).unwrap();
        assert_eq!(kv.len(), 5);
        assert_eq!(kv.committed_len(), 5);
        // Roll back past a block boundary.
        kv.rollback_to(2).unwrap();
        assert_eq!(kv.len(), 2);
        assert_eq!(kv.current_k().unwrap().to_vec_f32().unwrap().len(), 2 * 2 * 4);
    }

    /// store_kv writes into the newest slot and is readable back.
    #[test]
    fn mock_kv_cache_store_and_read() {
        let mut kv = MockKvCache::new(1, 2);
        kv.append_slot().unwrap();
        let k = grim_backend_cpu::cpu_tensor(vec![1.5f32, -2.0], grim_tensor::Shape::new(vec![1, 2]));
        let v = grim_backend_cpu::cpu_tensor(vec![3.0f32, 4.0], grim_tensor::Shape::new(vec![1, 2]));
        kv.store_kv(&k, &v).unwrap();
        assert_eq!(kv.current_k().unwrap().to_vec_f32().unwrap(), vec![1.5, -2.0]);
        assert_eq!(kv.current_v().unwrap().to_vec_f32().unwrap(), vec![3.0, 4.0]);
    }
}
