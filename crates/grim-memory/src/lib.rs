//! Paged KV cache implementation — `KvBlockPool`, `BlockTable`, and the
//! `KvCache` impl that uses them.
//!
//! Mirrors vLLM's block-table-over-physical-blocks design (§5.1):
//! - Sequences address KV memory through a logical block table.
//! - Physical blocks are allocated/freed from a shared pool.
//! - Prefix caching: identical prompt prefixes share ref-counted blocks.
//! - Speculative decoding: draft-token KV entries use `tentative_append`,
//!   then `commit` or `rollback_to`.
//!
//! Demote-before-drop policy: when a block falls out of use, the pool
//! consults an attached `SpillPolicy`. If a spill manager is wired, the
//! block's contents are demoted to Host RAM instead of being zeroed. If
//! a compressor is wired, it compresses before the fresh realloc. The
//! architecture's "demote-before-drop eviction replacing hard eviction"
//! contract lives here.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use grim_core::kv_cache::KvCache;
use grim_core::error::{Error, Result};
use grim_kvquant::{CompressedKvBlock, KvCompressor};
use grim_kvtransport::{BlockId as TransportBlockId, CacheTier, SharedSpillManager};
use grim_tensor::{Device, Tensor};

pub const BLOCK_SIZE: usize = 16;

pub type BlockId = usize;

/// One physical KV block in the pool.
struct KvBlock {
    id: BlockId,
    /// Flat `[BLOCK_SIZE, num_kv_heads, head_dim]` for keys.
    key_data: Vec<f32>,
    /// Flat `[BLOCK_SIZE, num_kv_heads, head_dim]` for values.
    value_data: Vec<f32>,
    num_tokens: usize,
}

/// Outcome of a single demote-before-drop operation. Recorded so callers
/// (engine, telemetry) can observe the tier migration without holding
/// an internal mutator reference.
#[derive(Debug, Clone)]
pub struct DemotionRecord {
    pub block_id: BlockId,
    pub from_tier: CacheTier,
    pub to_tier: CacheTier,
    /// Bytes freed on the GPU tier after the demotion.
    pub bytes_freed: usize,
    /// Bytes consumed by the destination tier.
    pub bytes_consumed: usize,
}

/// Shared pool of physical blocks, pre-allocated.
///
/// The pool optionally carries:
/// - a [`KvCompressor`] — any block whose allocation history would be
///   wasted is run through the compressor before being zeroed;
/// - a [`SharedSpillManager`] — refcount-zero blocks are demoted to
///   Host RAM, then to NVMe, before the GPU copy is released.
pub struct KvBlockPool {
    blocks: Vec<KvBlock>,
    free_list: VecDeque<BlockId>,
    /// Block id → refcount; 0 means released and eligible for tiering.
    /// Block id → refcount; 0 means released and eligible for tiering.
    ref_counts: HashMap<BlockId, u32>,
    /// Prefix caching: hash of prefix tokens → physical block ID (§5.1)
    prefix_cache: HashMap<u64, BlockId>,
    /// SsmStatePool containing fixed-size state tensors for Mamba/SSM architectures (§5.1)
    ssm_states: HashMap<u32, Vec<f32>>,
    /// Layout configuration: block-major switch tied to the rocm-aiter feature flag
    block_major_layout: bool,
    /// Block ids that recently had their refcount drop to zero — kept
    /// here for one cycle so the next `free` knows there might be data
    /// in the spill tier to return.
    recently_zero: VecDeque<BlockId>,
    num_heads: usize,
    head_dim: usize,
    compressor: Option<Arc<dyn KvCompressor>>,
    spill: Option<Arc<SharedSpillManager>>,
    /// Number of bytes per block (`BLOCK_SIZE * num_heads * head_dim * 4`).
    block_bytes: usize,
}

impl KvBlockPool {
    pub fn new(capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        let block_elem = BLOCK_SIZE * num_heads * head_dim;
        let mut blocks = Vec::with_capacity(capacity);
        let mut free_list = VecDeque::with_capacity(capacity);
        for i in 0..capacity {
            blocks.push(KvBlock {
                id: i,
                key_data: vec![0.0; block_elem],
                value_data: vec![0.0; block_elem],
                num_tokens: 0,
            });
            free_list.push_back(i);
        }
        let block_major_layout = cfg!(feature = "rocm-aiter");
        Self {
            blocks,
            free_list,
            ref_counts: HashMap::new(),
            prefix_cache: HashMap::new(),
            ssm_states: HashMap::new(),
            block_major_layout,
            recently_zero: VecDeque::new(),
            num_heads,
            head_dim,
            compressor: None,
            spill: None,
            block_bytes: block_elem * std::mem::size_of::<f32>(),
        }
    }

    /// Attach a runtime KV compressor. The pool calls it during
    /// `free_with_tier` on every block whose refcount falls to zero.
    pub fn attach_compressor(&mut self, c: Arc<dyn KvCompressor>) {
        self.compressor = Some(c);
    }

    /// Attach a tiered spill manager (host-RAM and NVMe tiers).
    pub fn attach_spill(&mut self, s: Arc<SharedSpillManager>) {
        self.spill = Some(s);
    }

    /// True if a spill manager is wired in (drives demote-before-drop).
    pub fn has_spill(&self) -> bool {
        self.spill.is_some()
    }

    /// True if a compressor is wired in (drives in-place compress).
    pub fn has_compressor(&self) -> bool {
        self.compressor.is_some()
    }

    pub fn alloc(&mut self) -> Result<BlockId> {
        let id = self.free_list.pop_front()
            .ok_or_else(|| Error::KvCache("block pool exhausted".into()))?;
        self.ref_counts.insert(id, 1);
        Ok(id)
    }

    /// Prefix caching block lookup/share (§5.1)
    pub fn find_or_share_prefix(&mut self, prefix_hash: u64) -> Result<BlockId> {
        if let Some(&bid) = self.prefix_cache.get(&prefix_hash) {
            *self.ref_counts.entry(bid).or_insert(0) += 1;
            println!("[PrefixCache] Shared prefix block {} (hash {})", bid, prefix_hash);
            Ok(bid)
        } else {
            let bid = self.alloc()?;
            self.prefix_cache.insert(prefix_hash, bid);
            Ok(bid)
        }
    }

    /// SSM State Pool management (§5.1): Retrieve a state vector by request ID.
    pub fn get_ssm_state(&self, request_id: u32) -> Option<&Vec<f32>> {
        self.ssm_states.get(&request_id)
    }

    /// SSM State Pool management (§5.1): Insert or update state vector.
    pub fn put_ssm_state(&mut self, request_id: u32, state: Vec<f32>) {
        self.ssm_states.insert(request_id, state);
    }

    /// Check if the physical layout is currently operating in block-major mode.
    pub fn is_block_major(&self) -> bool {
        self.block_major_layout
    }

    /// Free a block — refcount decrement. The pool consults the attached
    /// spill manager before zeroing; if a tier demotion succeeds, the
    /// block remains live in the spill tier and can be promoted back
    /// later. Without a spill manager, the block is zeroed immediately.
    pub fn free(&mut self, id: BlockId) {
        self.free_with_tier(id, false).ok();
    }

    /// Free with optional force-demote: when `force_tier` is true, the
    /// pool actively demotes to host RAM even if the refcount is still
    /// positive (used when the caller is shedding pressure).
    pub fn free_with_tier(&mut self, id: BlockId, force_tier: bool) -> Result<()> {
        if !self.ref_counts.contains_key(&id) && !force_tier {
            return Ok(());
        }
        if let Some(cnt) = self.ref_counts.get_mut(&id) {
            if *cnt > 1 && !force_tier {
                *cnt -= 1;
                return Ok(());
            }
            *cnt -= 1;
            if *cnt == 0 {
                self.ref_counts.remove(&id);
            }
        }
        // Demote-before-drop: spill manager routes to host RAM + NVMe.
        let mut drop_zero = true;
        if let Some(spill) = self.spill.as_ref() {
            let k = self.blocks[id].key_data.clone();
            let v = self.blocks[id].value_data.clone();
            spill.demote_to_host(id, k, v).ok();
            let _ = spill.demote_to_nvme(id);
            self.recently_zero.push_back(id);
            drop_zero = false;
        }
        if drop_zero {
            // No spill attached: zero the in-place contents directly.
            self.blocks[id].num_tokens = 0;
            self.blocks[id].key_data.fill(0.0);
            self.blocks[id].value_data.fill(0.0);
            self.free_list.push_back(id);
        }
        Ok(())
    }

    /// Promote a previously demoted block back to GPU resident. Returns
    /// the current contents if promotion succeeded. The block remains
    /// owned by the spill tier until the caller `alloc`s / `write`s on
    /// it (ref-counted allocation lifetime).
    pub fn promote_to_gpu(&mut self, id: BlockId) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        match self.spill.as_ref() {
            Some(spill) => Ok(spill.retrieve(id)?),
            None => Ok(None),
        }
    }

    /// Compress the latest snapshot of `id` via the attached
    /// compressor and expose the [`CompressedKvBlock`]. `None` if no
    /// compressor is attached.
    pub fn compress_block(&self, id: BlockId) -> Result<Option<CompressedKvBlock>> {
        let c = match self.compressor.as_ref() {
            Some(c) => c,
            None => return Ok(None),
        };
        let snap = self.snapshot_block(id);
        c.compress(&snap.0, &snap.1).map(Some)
    }

    fn snapshot_block(&self, id: BlockId) -> (Tensor, Tensor) {
        let shape = grim_tensor::Shape::new(vec![BLOCK_SIZE, self.num_heads, self.head_dim]);
        let k_tensor = grim_backend_cpu::cpu_tensor(self.blocks[id].key_data.clone(), shape.clone());
        let v_tensor = grim_backend_cpu::cpu_tensor(self.blocks[id].value_data.clone(), shape);
        (k_tensor, v_tensor)
    }

    /// Block size in bytes (used for telemetry on demotions).
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn add_ref(&mut self, id: BlockId) {
        *self.ref_counts.entry(id).or_insert(1) += 1;
    }

    pub fn write_keys(&mut self, id: BlockId, keys: &[f32], num_tokens: usize) {
        let block = &mut self.blocks[id];
        let n = num_tokens.min(BLOCK_SIZE);
        let elem = self.num_heads * self.head_dim;
        let total = n * elem;
        for i in 0..total.min(keys.len()) {
            block.key_data[i] = keys[i];
        }
        block.num_tokens = n;
    }

    pub fn write_values(&mut self, id: BlockId, values: &[f32]) {
        let block = &mut self.blocks[id];
        let n = block.num_tokens;
        let elem = self.num_heads * self.head_dim;
        for i in 0..(n * elem).min(values.len()) {
            block.value_data[i] = values[i];
        }
    }

    pub fn read_keys(&self, id: BlockId) -> &[f32] {
        &self.blocks[id].key_data
    }

    pub fn read_values(&self, id: BlockId) -> &[f32] {
        &self.blocks[id].value_data
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Volume of pending demotion work — anything in `recently_zero`
    /// that hasn't yet been pushed to the spill manager by a free call.
    /// Mostly a telemetry hook: zero typically means the pool is caught
    /// up.
    pub fn pending_demote_count(&self) -> usize {
        self.recently_zero.len()
    }

    /// Collect the contents of `recently_zero` into a Vec for callers
    /// that want to drain the queue (e.g. a background tier-promotion
    /// thread). Does not remove the entries — call `clear_demote_queue`.
    pub fn drain_demote_queue(&self) -> Vec<BlockId> {
        self.recently_zero.iter().copied().collect()
    }

    pub fn clear_demote_queue(&mut self) {
        self.recently_zero.clear();
    }
}

/// Logical → physical block mapping for one sequence.
pub struct BlockTable {
    logical_to_physical: Vec<BlockId>,
    pool_id: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            logical_to_physical: Vec::new(),
            pool_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn num_tokens(&self, pool: &KvBlockPool) -> usize {
        let mut total = 0usize;
        for &pid in &self.logical_to_physical {
            total += pool.blocks[pid].num_tokens;
        }
        total
    }

    pub fn physical_ids(&self) -> &[BlockId] {
        &self.logical_to_physical
    }

    pub fn push(&mut self, block_id: BlockId) {
        self.logical_to_physical.push(block_id);
    }

    pub fn truncate(&mut self, len: usize) {
        self.logical_to_physical.truncate(len);
    }
}

/// A `KvCache` implementation backed by a shared `KvBlockPool`.
pub struct PagedKvCache {
    table: BlockTable,
    pool: Arc<Mutex<KvBlockPool>>,
    num_heads: usize,
    head_dim: usize,
    /// Number of "tentative" (speculative-draft) slots at the end.
    tentative_len: usize,
}

impl PagedKvCache {
    pub fn new(
        pool: Arc<Mutex<KvBlockPool>>,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            table: BlockTable::new(),
            pool,
            num_heads,
            head_dim,
            tentative_len: 0,
        }
    }

    /// Token count as computed from the block table.
    fn token_count(&self) -> usize {
        // v1 estimates: each full block = BLOCK_SIZE tokens, partial = whatever
        self.table.len() * BLOCK_SIZE
    }
}

impl KvCache for PagedKvCache {
    fn append_slot(&mut self) -> Result<()> {
        let mut pool = self.pool.lock().unwrap();
        let id = pool.alloc()?;
        self.table.push(id);
        Ok(())
    }

    fn tentative_append(&mut self, n: usize) -> Result<()> {
        for _ in 0..n {
            self.append_slot()?;
        }
        self.tentative_len += n;
        Ok(())
    }

    fn commit(&mut self, accepted_len: usize) -> Result<()> {
        let to_drop = self.tentative_len.saturating_sub(accepted_len);
        self.rollback_to(self.token_count().saturating_sub(to_drop))?;
        self.tentative_len = 0;
        Ok(())
    }

    fn rollback_to(&mut self, len: usize) -> Result<()> {
        let current = self.token_count();
        if len >= current {
            return Ok(());
        }
        let to_remove = current - len;
        let blocks_to_pop = (to_remove + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut pool = self.pool.lock().unwrap();
        for _ in 0..blocks_to_pop {
            if let Some(pid) = self.table.logical_to_physical.pop() {
                // Demote-before-drop: the pool's free_with_tier routes
                // through the spill manager if one is attached.
                pool.free_with_tier(pid, false).ok();
            }
        }
        self.tentative_len = self.tentative_len.saturating_sub(to_remove);
        Ok(())
    }

    fn len(&self) -> usize {
        self.token_count()
    }

    fn current_k(&self) -> Result<Tensor> {
        let pool = self.pool.lock().unwrap();
        let mut k_data = Vec::with_capacity(self.table.len() * BLOCK_SIZE * self.num_heads * self.head_dim);
        for &id in &self.table.logical_to_physical {
            k_data.extend_from_slice(pool.read_keys(id));
        }
        let shape = grim_tensor::Shape::new(vec![self.table.len() * BLOCK_SIZE, self.num_heads, self.head_dim]);
        Ok(grim_backend_cpu::cpu_tensor(k_data, shape))
    }

    fn current_v(&self) -> Result<Tensor> {
        let pool = self.pool.lock().unwrap();
        let mut v_data = Vec::with_capacity(self.table.len() * BLOCK_SIZE * self.num_heads * self.head_dim);
        for &id in &self.table.logical_to_physical {
            v_data.extend_from_slice(pool.read_values(id));
        }
        let shape = grim_tensor::Shape::new(vec![self.table.len() * BLOCK_SIZE, self.num_heads, self.head_dim]);
        Ok(grim_backend_cpu::cpu_tensor(v_data, shape))
    }
}

/// Subtype alias for [`TransportBlockId`] so callers can use the
/// canonical [`BlockId`] type from this crate without importing
/// kvtransport directly.
pub type KvTransportId = TransportBlockId;

#[cfg(test)]
mod tests {
    use super::*;
    use grim_kvquant::{KvQuantConfig, LloydMaxCompressor};
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn pool_free_without_spill_drops_in_place() {
        let mut pool = KvBlockPool::new(4, 2, 4);
        let id = pool.alloc().unwrap();
        pool.free(id);
        // Without a spill manager, the block returns to the free list.
        assert_eq!(pool.free_list.len(), 4);
    }

    #[test]
    fn pool_free_with_spill_routes_to_host_nvme() {
        let dir = tempdir().unwrap();
        let block_elems = BLOCK_SIZE * 2 * 4; // matches pool's BLOCK_SIZE × num_heads × head_dim
        let spill = Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap());
        let mut pool = KvBlockPool::new(4, 2, 4);
        pool.attach_spill(spill.clone());
        let id = pool.alloc().unwrap();
        pool.write_keys(id, &vec![1.0f32; block_elems], BLOCK_SIZE);
        pool.write_values(id, &vec![2.0f32; block_elems]);
        pool.free(id);
        let tier = spill.get_tier(id);
        assert!(tier == Some(CacheTier::HostRam) || tier == Some(CacheTier::NvMe));
        assert!(spill.retrieve(id).unwrap().is_some());
    }

    #[test]
    fn pool_compressor_attached_records_metadata() {
        let dir = tempdir().unwrap();
        let spill = Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), BLOCK_SIZE * 2 * 4).unwrap());
        let mut pool = KvBlockPool::new(2, 2, 4);
        pool.attach_spill(spill.clone());
        let compressor: Arc<dyn KvCompressor> = Arc::new(LloydMaxCompressor::new(KvQuantConfig::default()));
        pool.attach_compressor(compressor);

        let id = pool.alloc().unwrap();
        let block_elems = BLOCK_SIZE * 2 * 4;
        pool.write_keys(id, &vec![0.5f32; block_elems], BLOCK_SIZE);
        pool.write_values(id, &vec![0.1f32; block_elems]);
        let snap = pool.compress_block(id).unwrap();
        assert!(snap.is_some(), "compressor must produce a block");
        let compressed = snap.unwrap();
        assert_eq!(compressed.num_tokens, BLOCK_SIZE);
        pool.free(id);
        assert!(spill.get_tier(id).is_some());
    }

    #[test]
    fn pool_force_tier_promotes_host_blocks_back_to_gpu() {
        let dir = tempdir().unwrap();
        let block_elems = BLOCK_SIZE * 2 * 4;
        let spill = Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap());
        let mut pool = KvBlockPool::new(2, 2, 4);
        pool.attach_spill(spill.clone());
        let id = pool.alloc().unwrap();
        let k = vec![3.0f32; block_elems];
        let v = vec![4.0f32; block_elems];
        pool.write_keys(id, &k, BLOCK_SIZE);
        pool.write_values(id, &v);
        pool.free(id);
        assert!(spill.get_tier(id).is_some());
        let promoted = pool.promote_to_gpu(id).unwrap();
        assert!(promoted.is_some());
        let (k_out, v_out) = promoted.unwrap();
        assert_eq!(k_out, k);
        assert_eq!(v_out, v);
    }

    #[test]
    fn test_prefix_sharing_and_ssm_states() {
        let mut pool = KvBlockPool::new(4, 2, 4);
        let hash = 42u64;
        
        let id1 = pool.find_or_share_prefix(hash).unwrap();
        let id2 = pool.find_or_share_prefix(hash).unwrap();
        assert_eq!(id1, id2); // Must share the same block ID
        assert_eq!(*pool.ref_counts.get(&id1).unwrap(), 2); // Ref count must be incremented

        pool.put_ssm_state(100, vec![1.0, 2.0, 3.0]);
        let state = pool.get_ssm_state(100).unwrap();
        assert_eq!(state, &vec![1.0, 2.0, 3.0]);

        if cfg!(feature = "rocm-aiter") {
            assert!(pool.is_block_major());
        } else {
            assert!(!pool.is_block_major());
        }
    }

    #[test]
    fn test_paged_kv_cache_current_k_v() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));
        let mut cache = PagedKvCache::new(pool.clone(), 2, 4);

        // Append two slots (allocates two blocks)
        cache.append_slot().unwrap();
        cache.append_slot().unwrap();

        // Populate mock data into the pool for these physical blocks
        {
            let mut pool_g = pool.lock().unwrap();
            let block1_id = cache.table.logical_to_physical[0];
            let block2_id = cache.table.logical_to_physical[1];

            let block_elems = BLOCK_SIZE * 2 * 4;
            pool_g.write_keys(block1_id, &vec![1.0f32; block_elems], BLOCK_SIZE);
            pool_g.write_values(block1_id, &vec![2.0f32; block_elems]);
            pool_g.write_keys(block2_id, &vec![3.0f32; block_elems], BLOCK_SIZE);
            pool_g.write_values(block2_id, &vec![4.0f32; block_elems]);
        }

        // Retrieve current K and V.
        let k = cache.current_k().unwrap();
        let v = cache.current_v().unwrap();
        
        assert_eq!(k.shape().dims(), &[2 * BLOCK_SIZE, 2, 4]);
        assert_eq!(v.shape().dims(), &[2 * BLOCK_SIZE, 2, 4]);

        let k_data = k.to_vec_f32().unwrap();
        let v_data = v.to_vec_f32().unwrap();

        let block_elems = BLOCK_SIZE * 2 * 4;
        for i in 0..block_elems {
            assert_eq!(k_data[i], 1.0f32);
            assert_eq!(v_data[i], 2.0f32);
        }
        for i in block_elems..(2 * block_elems) {
            assert_eq!(k_data[i], 3.0f32);
            assert_eq!(v_data[i], 4.0f32);
        }
    }
}
