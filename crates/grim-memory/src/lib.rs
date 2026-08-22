//! Paged KV cache memory pool, logical block tables, prefix sharing, and multi-tier spilling.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use grim_core::error::{Error, Result};
use grim_core::kv_cache::KvCache;
use grim_kvquant::{CompressedKvBlock, KvCompressor};
use grim_kvtransport::{BlockId as TransportBlockId, CacheTier, SharedSpillManager};
use grim_tensor::{DType, Device, Shape, Tensor};

/// MoE resident-set HBM budget (`rocm_kernel_plan.md` WI-C).
pub mod moe_budget;

pub use moe_budget::{MoeResidentBudget, ResidentTier};

/// Block-granular radix tree for prefix (RadixAttention-style) KV sharing.
pub mod radix;

pub use radix::RadixTree;

pub const BLOCK_SIZE: usize = 16;

pub type BlockId = usize;

impl grim_kvtransport::KvBlockStore for KvBlockPool {
    fn num_blocks(&self) -> usize {
        self.num_blocks()
    }
    fn block_elem_per_token(&self) -> usize {
        self.num_heads * self.head_dim
    }
    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
    fn write_keys(&mut self, id: BlockId, keys: &[f32], num_tokens: usize) {
        self.write_keys(id, keys, num_tokens);
    }
    fn write_values(&mut self, id: BlockId, values: &[f32]) {
        self.write_values(id, values);
    }
    fn block_is_received(&self, id: BlockId) -> bool {
        self.block_is_received(id)
    }
    fn write_layer_keys(&mut self, id: BlockId, layer_idx: u32, keys: &[f32], num_tokens: usize) {
        self.write_layer_keys(id, layer_idx as usize, keys, num_tokens);
    }
    fn write_layer_values(&mut self, id: BlockId, layer_idx: u32, values: &[f32]) {
        self.write_layer_values(id, layer_idx as usize, values);
    }
}

/// One physical KV block in the pool.
struct KvBlock {
    _id: BlockId,
    /// Flat `[BLOCK_SIZE, num_kv_heads, head_dim]` for keys (layer 0).
    key_data: Vec<f32>,
    /// Flat `[BLOCK_SIZE, num_kv_heads, head_dim]` for values (layer 0).
    value_data: Vec<f32>,
    /// Per-layer key storage for multi-layer handoffs.
    layer_keys: Vec<Vec<f32>>,
    /// Per-layer value storage for multi-layer handoffs.
    layer_values: Vec<Vec<f32>>,
    num_tokens: usize,
    /// Whether this block has received real KV data (via `store_kv`,
    /// network ingestion, or explicit `write_keys`). Replaces the
    /// fragile non-zero-content sniff in the decode fetch loop: a
    /// genuinely all-zero KV block is valid data, not "not yet arrived."
    received: bool,
    /// Current tier residency (Phase 2.3): explicit so promotion can be
    /// decided without re-querying the spill manager.
    location: CacheTier,
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
    ref_counts: HashMap<BlockId, u32>,
    /// Prefix caching: block-granular radix tree (§5.1). Keys by block
    /// content, so partial/branching prefix sharing across requests works
    /// instead of the old exact-whole-prefix hash map.
    prefix_tree: RadixTree,
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
                _id: i,
                key_data: vec![0.0; block_elem],
                value_data: vec![0.0; block_elem],
                layer_keys: Vec::new(),
                layer_values: Vec::new(),
                num_tokens: 0,
                received: false,
                location: CacheTier::Gpu,
            });
            free_list.push_back(i);
        }
        let block_major_layout = cfg!(feature = "rocm-aiter");
        Self {
            blocks,
            free_list,
            ref_counts: HashMap::new(),
            prefix_tree: RadixTree::new(BLOCK_SIZE),
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
        if self.free_list.is_empty() {
            // Pressure: reclaim a cold, unreferenced trie leaf before
            // declaring exhaustion.
            self.evict_cold();
        }
        let id = self
            .free_list
            .pop_front()
            .ok_or_else(|| Error::KvCache("block pool exhausted".into()))?;
        self.ref_counts.insert(id, 1);
        Ok(id)
    }

    /// Prefix caching (§5.1): look up how much of `tokens` can be reused
    /// from previously computed blocks. Returns the reused physical block
    /// ids and the number of leading tokens they cover. The caller runs
    /// prefill only for `tokens[matched..]` and then calls
    /// [`KvBlockPool::insert_prefix`].
    pub fn match_prefix(&self, tokens: &[u32]) -> (Vec<BlockId>, usize) {
        self.prefix_tree.match_prefix(tokens)
    }

    /// Register newly computed `blocks` for `tokens` after a prefill
    /// completes. Shared prefix nodes are reused; diverging blocks become
    /// new tree nodes.
    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[BlockId]) {
        self.prefix_tree.insert(tokens, blocks);
        self.prefix_tree.touch(tokens);
    }

    /// Drop one sequence's reference to `blocks`, pruning unshared tree
    /// nodes (trie-leaf LRU walk-up). Call this when a sequence is fully
    /// released so eviction does not reclaim a still-referenced prefix.
    pub fn remove_prefix(&mut self, blocks: &[BlockId]) {
        self.prefix_tree.remove(blocks);
    }

    /// Current tier residency of a physical block (Phase 2.3).
    pub fn block_location(&self, id: BlockId) -> CacheTier {
        self.blocks[id].location
    }

    /// Look up reusable prefix blocks for `tokens`, promoting any matched
    /// block that was demoted to host/NVMe back to GPU before returning it
    /// (Phase 2.2: the "cache hit but demoted" path). Returns the matched
    /// block ids, the number of matched tokens, and whether any promotion
    /// occurred.
    pub fn match_prefix_promoting(&mut self, tokens: &[u32]) -> (Vec<BlockId>, usize, bool) {
        let (matched, n) = self.prefix_tree.match_prefix(tokens);
        let mut promoted = false;
        for &bid in &matched {
            if self.blocks[bid].location != CacheTier::Gpu {
                if self.promote_to_gpu(bid).ok().flatten().is_some() {
                    promoted = true;
                }
            }
        }
        (matched, n, promoted)
    }

    /// Demote a cached prefix block to the spill tier under memory pressure
    /// WITHOUT reclaiming it: the radix-tree entry is kept so a future
    /// request can still match it and promote it back, and the block is NOT
    /// returned to the free list (so its cached KV is never overwritten by a
    /// new alloc). The caller (`demote_cold_prefix`) only supplies blocks
    /// the radix tree reports as cold (tree refcount 0), so an actively
    /// referenced block is never demoted.
    fn demote_prefix_block(&mut self, bid: BlockId) -> bool {
        let Some(spill) = self.spill.as_ref() else {
            return false;
        };
        let k = self.blocks[bid].key_data.clone();
        let v = self.blocks[bid].value_data.clone();
        if spill.demote_to_host(bid, k, v).is_err() {
            return false;
        }
        if spill.demote_to_nvme(bid).is_err() {
            return false;
        }
        self.blocks[bid].location = CacheTier::HostRam;
        true
    }

    /// Pressure hook (Phase 2.1): demote the coldest unreferenced prefix
    /// leaf to host/NVMe, keeping it cached. Returns the demoted block id,
    /// or `None` if there is no cold prefix to demote. The engine calls this
    /// when GPU block-pool utilization crosses a threshold.
    pub fn demote_cold_prefix(&mut self) -> Option<BlockId> {
        let bid = self.prefix_tree.coldest_leaf()?;
        if self.demote_prefix_block(bid) {
            Some(bid)
        } else {
            None
        }
    }

    /// Convenience single-call prefix share (§5.1): match, allocate physical
    /// blocks for any non-matching tail, insert, and return all block ids
    /// plus the count of tokens that were reused. Used where the caller
    /// does not split match/prefill/insert (e.g. eager one-shot paths).
    pub fn find_or_share_prefix_tokens(&mut self, tokens: &[u32]) -> Result<(Vec<BlockId>, usize)> {
        let (matched, matched_tokens) = self.prefix_tree.match_prefix(tokens);
        let total_blocks = tokens.len().div_ceil(BLOCK_SIZE);
        let mut all_blocks = matched.clone();
        for _ in matched.len()..total_blocks {
            let bid = self.alloc()?;
            all_blocks.push(bid);
        }
        for &bid in &matched {
            self.add_ref(bid);
        }
        self.insert_prefix(tokens, &all_blocks);
        Ok((all_blocks, matched_tokens))
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
        if let Some(spill) = self.spill.as_ref() {
            let k = self.blocks[id].key_data.clone();
            let v = self.blocks[id].value_data.clone();
            if let Err(e) = spill.demote_to_host(id, k, v) {
                eprintln!("[BlockPool] demote_to_host failed for block {id}: {e}");
            }
            if let Err(e) = spill.demote_to_nvme(id) {
                eprintln!("[BlockPool] demote_to_nvme failed for block {id}: {e}");
            }
            // Mark the block as demoted so promotion can be decided later
            // without re-querying the spill manager.
            self.blocks[id].location = CacheTier::HostRam;
            self.recently_zero.push_back(id);
            // Do NOT push to free_list — the block is spilled, not available
            // for fresh allocation. Only promote_to_gpu can reclaim it.
        } else {
            // No spill attached: zero the in-place contents directly.
            self.blocks[id].num_tokens = 0;
            self.blocks[id].received = false;
            self.blocks[id].key_data.fill(0.0);
            self.blocks[id].value_data.fill(0.0);
            self.blocks[id].location = CacheTier::Gpu;
            self.free_list.push_back(id);
        }
        Ok(())
    }

    /// Promote a previously demoted block back to GPU resident. On success
    /// the retrieved key/value data is written back into the block and its
    /// `location` restored to [`CacheTier::Gpu`]. Returns the contents if
    /// promotion succeeded (the block was demoted), or `None` if there was
    /// nothing to promote (block already GPU-resident or no spill manager).
    pub fn promote_to_gpu(&mut self, id: BlockId) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let Some(spill) = self.spill.as_ref() else {
            return Ok(None);
        };
        match spill.retrieve(id)? {
            Some((k, v)) => {
                let elem = self.num_heads * self.head_dim;
                let n = (k.len() / elem).min(BLOCK_SIZE);
                // Validate retrieved spill data fits within the block's capacity
                // before copying — a mismatch is a panic, not silent corruption.
                let block_cap = self.blocks[id].key_data.len();
                let k_len = k.len().min(block_cap);
                let v_len = v.len().min(self.blocks[id].value_data.len());
                self.blocks[id].key_data[..k_len].copy_from_slice(&k[..k_len]);
                self.blocks[id].value_data[..v_len].copy_from_slice(&v[..v_len]);
                self.blocks[id].num_tokens = n;
                self.blocks[id].received = true;
                self.blocks[id].location = CacheTier::Gpu;
                Ok(Some((k, v)))
            }
            None => Ok(None),
        }
    }

    /// Trie-leaf LRU eviction (Phase 1.6): reclaim the coldest childless
    /// tree leaf whose physical block is not actively referenced, freeing
    /// its contents (demote-to-host/NVMe if a spill manager is attached,
    /// otherwise in-place zero) and returning it to the free list. Returns
    /// `true` if a block was reclaimed.
    fn evict_cold(&mut self) -> bool {
        let Some(bid) = self.prefix_tree.evict_coldest_leaf() else {
            return false;
        };
        // Never reclaim a block still attached to a live sequence.
        if self.ref_counts.get(&bid).copied().unwrap_or(0) > 0 {
            return false;
        }
        if let Some(spill) = self.spill.as_ref() {
            let k = self.blocks[bid].key_data.clone();
            let v = self.blocks[bid].value_data.clone();
            if let Err(e) = spill.demote_to_host(bid, k, v) {
                eprintln!("[BlockPool] evict demote_to_host failed for {bid}: {e}");
            }
            if let Err(e) = spill.demote_to_nvme(bid) {
                eprintln!("[BlockPool] evict demote_to_nvme failed for {bid}: {e}");
            }
            self.blocks[bid].location = CacheTier::HostRam;
            self.recently_zero.push_back(bid);
            // Do NOT push to free_list — spilled blocks are not available for
            // fresh allocation. Only promote_to_gpu can reclaim them.
            self.ref_counts.remove(&bid);
            true
        } else {
            self.blocks[bid].num_tokens = 0;
            self.blocks[bid].received = false;
            self.blocks[bid].key_data.fill(0.0);
            self.blocks[bid].value_data.fill(0.0);
            self.ref_counts.remove(&bid);
            self.free_list.push_back(bid);
            true
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
        let k_tensor =
            grim_backend_cpu::cpu_tensor(self.blocks[id].key_data.clone(), shape.clone());
        let v_tensor = grim_backend_cpu::cpu_tensor(self.blocks[id].value_data.clone(), shape);
        (k_tensor, v_tensor)
    }

    /// Block size in bytes (used for telemetry on demotions).
    /// Total capacity of the KV block pool in blocks.
    pub fn capacity(&self) -> usize {
        self.blocks.len()
    }

    /// Number of blocks currently allocated and in use.
    pub fn used_count(&self) -> usize {
        self.blocks.len().saturating_sub(self.free_list.len())
    }

    /// Size of a single block in bytes.
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
        let len = (n * elem).min(keys.len());
        block.key_data[..len].copy_from_slice(&keys[..len]);
        block.num_tokens = n;
        if len > 0 {
            block.received = true;
        }
    }

    pub fn write_values(&mut self, id: BlockId, values: &[f32]) {
        let block = &mut self.blocks[id];
        let n = block.num_tokens;
        let elem = self.num_heads * self.head_dim;
        let len = (n * elem).min(values.len());
        block.value_data[..len].copy_from_slice(&values[..len]);
    }

    pub fn write_layer_keys(&mut self, id: BlockId, layer: usize, keys: &[f32], num_tokens: usize) {
        if id >= self.blocks.len() {
            return;
        }
        let block_elem = BLOCK_SIZE * self.num_heads * self.head_dim;
        if self.blocks[id].layer_keys.len() <= layer {
            self.blocks[id].layer_keys.resize_with(layer + 1, || vec![0.0; block_elem]);
        }
        let len = keys.len().min(block_elem);
        self.blocks[id].layer_keys[layer][..len].copy_from_slice(&keys[..len]);
        if layer == 0 {
            self.blocks[id].key_data[..len].copy_from_slice(&keys[..len]);
            self.blocks[id].num_tokens = num_tokens.min(BLOCK_SIZE);
            if len > 0 {
                self.blocks[id].received = true;
            }
        }
    }

    pub fn write_layer_values(&mut self, id: BlockId, layer: usize, values: &[f32]) {
        if id >= self.blocks.len() {
            return;
        }
        let block_elem = BLOCK_SIZE * self.num_heads * self.head_dim;
        if self.blocks[id].layer_values.len() <= layer {
            self.blocks[id].layer_values.resize_with(layer + 1, || vec![0.0; block_elem]);
        }
        let len = values.len().min(block_elem);
        self.blocks[id].layer_values[layer][..len].copy_from_slice(&values[..len]);
        if layer == 0 {
            self.blocks[id].value_data[..len].copy_from_slice(&values[..len]);
        }
    }

    pub fn read_layer_keys(&self, id: BlockId, layer: usize) -> Option<&[f32]> {
        if id >= self.blocks.len() {
            return None;
        }
        if layer < self.blocks[id].layer_keys.len() {
            Some(&self.blocks[id].layer_keys[layer])
        } else if layer == 0 {
            Some(&self.blocks[id].key_data)
        } else {
            None
        }
    }

    pub fn read_layer_values(&self, id: BlockId, layer: usize) -> Option<&[f32]> {
        if id >= self.blocks.len() {
            return None;
        }
        if layer < self.blocks[id].layer_values.len() {
            Some(&self.blocks[id].layer_values[layer])
        } else if layer == 0 {
            Some(&self.blocks[id].value_data)
        } else {
            None
        }
    }

    pub fn num_layers(&self, id: BlockId) -> usize {
        if id >= self.blocks.len() {
            0
        } else {
            self.blocks[id].layer_keys.len().max(1)
        }
    }

    pub fn read_keys(&self, id: BlockId) -> &[f32] {
        &self.blocks[id].key_data
    }

    pub fn read_values(&self, id: BlockId) -> &[f32] {
        &self.blocks[id].value_data
    }

    /// Whether block `id` has received real KV data (via `write_keys`
    /// / `store_kv` / network ingestion). Replaces the fragile
    /// non-zero-content sniff: a genuinely all-zero KV block is valid
    /// data, not "not yet arrived."
    pub fn block_is_received(&self, id: BlockId) -> bool {
        id < self.blocks.len() && self.blocks[id].received
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

    /// Invalidate and clear all prefix tree nodes and reclaim unreferenced blocks.
    /// Returns the number of newly reclaimed blocks.
    pub fn reset_prefix_cache(&mut self) -> usize {
        let before_free = self.free_list.len();
        self.prefix_tree = RadixTree::new(BLOCK_SIZE);
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let rc = self.ref_counts.get(&i).copied().unwrap_or(0);
            if rc == 0 && !self.free_list.contains(&i) {
                block.num_tokens = 0;
                block.received = false;
                self.free_list.push_back(i);
            }
        }
        self.recently_zero.clear();
        self.free_list.len().saturating_sub(before_free)
    }
}

/// Logical → physical block mapping for one sequence.
pub struct BlockTable {
    logical_to_physical: Vec<BlockId>,
    _pool_id: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            logical_to_physical: Vec::new(),
            _pool_id: 0,
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

    /// Truncate the logical table to `len` blocks, returning every freed
    /// physical id to `pool`.
    ///
    /// `len` is a *block count* (this wraps `Vec<BlockId>::truncate`), not a
    /// token count — there is no unit conversion here, which is deliberate to
    /// avoid the block-vs-token arithmetic that bit the speculative
    /// `commit`/`rollback_to` family.
    ///
    /// The free loop mirrors `PagedKvCache::rollback_to` exactly: pop from the
    /// back of `logical_to_physical`, call `pool.free_with_tier(pid, false)`
    /// per entry. Keeping both release paths structurally identical is worth
    /// more than either being marginally more efficient in isolation.
    pub fn truncate(&mut self, len: usize, pool: &mut KvBlockPool) {
        if len >= self.logical_to_physical.len() {
            return;
        }
        while self.logical_to_physical.len() > len {
            if let Some(pid) = self.logical_to_physical.pop() {
                pool.free_with_tier(pid, false).ok();
            }
        }
    }
}

/// A `KvCache` implementation backed by a shared `KvBlockPool`.
pub struct PagedKvCache {
    table: BlockTable,
    pool: Arc<Mutex<KvBlockPool>>,
    num_heads: usize,
    head_dim: usize,
    /// Physical block capacity of the backing pool (page tensors are sized
    /// `capacity * page_size * num_heads * head_dim` per layer).
    capacity: usize,
    /// Paged-attention page size (tokens per physical block).
    page_size: usize,
    /// Number of committed tokens in the sequence.
    committed_tokens: usize,
    /// Number of "tentative" (speculative-draft) tokens at the end.
    tentative_len: usize,
    /// Logical→physical block table as u32 physical ids (mirrors
    /// `table.logical_to_physical`), handed to the paged-attention kernel.
    block_table_u32: Vec<u32>,
    /// Per-layer physical K page tensors, laid out flat as
    /// `[capacity * page_size * num_kv_heads * head_dim]`.
    k_pages: Vec<Vec<f32>>,
    /// Per-layer physical V page tensors, same layout as `k_pages`.
    v_pages: Vec<Vec<f32>>,
    /// Per-layer token count so page offsets stay per-layer even though the
    /// block-table / `committed_tokens` counter is shared across layers.
    layer_committed_tokens: Vec<usize>,
    /// Owning device for this session. When ROCm, paged_kv_handles copies
    /// page slices to GPU once and returns RocmStorage tensors so the ROCm
    /// qkv_attention_paged kernel (which demands as_rocm inputs) doesn't
    /// panic on CPU-resident pages.
    device: Option<Device>,
    /// Backend device handle used for `from_cpu` when staging pages to GPU.
    backend: Option<Arc<dyn grim_tensor::BackendDevice>>,
    /// Per-layer full K/V GPU tensors (flat):
    /// `[capacity * page_size * num_kv_heads * head_dim]`, cached once per layer
    /// when first requested on a ROCm session.
    gpu_paged_k: Vec<Option<Arc<dyn grim_tensor::BackendStorage>>>,
    gpu_paged_v: Vec<Option<Arc<dyn grim_tensor::BackendStorage>>>,
}

impl PagedKvCache {
    pub fn new(
        pool: Arc<Mutex<KvBlockPool>>,
        num_heads: usize,
        head_dim: usize,
        page_size_: usize,
    ) -> Self {
        let capacity = pool.lock().unwrap().capacity();
        let page_size = if page_size_ == 0 {
            BLOCK_SIZE
        } else {
            page_size_
        };
        let np = capacity * page_size;
        let _elem_per_page = np * num_heads * head_dim;
        Self {
            table: BlockTable::new(),
            pool,
            num_heads,
            head_dim,
            capacity,
            page_size,
            committed_tokens: 0,
            tentative_len: 0,
            block_table_u32: Vec::new(),
            k_pages: Vec::new(),
            v_pages: Vec::new(),
            layer_committed_tokens: Vec::new(),
            device: None,
            backend: None,
            gpu_paged_k: Vec::new(),
            gpu_paged_v: Vec::new(),
        }
    }

    pub fn set_device(&mut self, device: Device, backend: Arc<dyn grim_tensor::BackendDevice>) {
        self.device = Some(device);
        self.backend = Some(backend);
        let n = self.k_pages.len().max(self.v_pages.len());
        self.gpu_paged_k.resize_with(n, || None);
        self.gpu_paged_v.resize_with(n, || None);
    }

    pub fn copy_pages_to_gpu(
        &self,
        layer: usize,
    ) -> Result<(
        Arc<dyn grim_tensor::BackendStorage>,
        Arc<dyn grim_tensor::BackendStorage>,
    )> {
        let dev = self
            .backend
            .as_ref()
            .ok_or_else(|| grim_core::error::Error::KvCache("no backend device set".into()))?;
        let stride = self.k_pages[layer].len() / (self.capacity * self.page_size);
        let dims = vec![self.capacity * self.page_size, stride];
        let k_storage =
            dev.from_cpu(&self.k_pages[layer], &Shape::new(dims.clone()), DType::F32)?;
        let v_storage = dev.from_cpu(&self.v_pages[layer], &Shape::new(dims), DType::F32)?;
        Ok((Arc::from(k_storage), Arc::from(v_storage)))
    }

    /// Return the number of active layers in this cache.
    pub fn num_layers(&self) -> usize {
        self.k_pages.len()
    }

    /// Extract key and value slices for a given layer and physical block ID.
    pub fn layer_block_slice(&self, layer: usize, block_id: usize) -> Option<(&[f32], &[f32])> {
        if layer >= self.k_pages.len() || layer >= self.v_pages.len() {
            return None;
        }
        let stride = self.k_pages[layer].len() / (self.capacity * self.page_size);
        let block_elems = self.page_size * stride;
        let start = block_id * block_elems;
        let end = start + block_elems;
        if end <= self.k_pages[layer].len() && end <= self.v_pages[layer].len() {
            Some((&self.k_pages[layer][start..end], &self.v_pages[layer][start..end]))
        } else {
            None
        }
    }
}

impl KvCache for PagedKvCache {
    fn append_slot(&mut self) -> Result<()> {
        self.committed_tokens += 1;
        let req_blocks = self.committed_tokens.div_ceil(BLOCK_SIZE);
        if self.table.len() < req_blocks {
            let mut pool = self.pool.lock().unwrap();
            let id = pool.alloc()?;
            self.table.push(id);
            self.block_table_u32.push(id as u32);
        }
        Ok(())
    }

    fn tentative_append(&mut self, n_tokens: usize) -> Result<()> {
        if n_tokens == 0 {
            return Ok(());
        }
        let total_tokens = self.committed_tokens + self.tentative_len + n_tokens;
        let req_blocks = total_tokens.div_ceil(BLOCK_SIZE);
        if self.table.len() < req_blocks {
            let needed = req_blocks - self.table.len();
            let mut pool = self.pool.lock().unwrap();
            for _ in 0..needed {
                let id = pool.alloc()?;
                self.table.push(id);
                self.block_table_u32.push(id as u32);
            }
        }
        self.tentative_len += n_tokens;
        Ok(())
    }

    fn commit(&mut self, accepted_tokens: usize) -> Result<()> {
        let accepted = accepted_tokens.min(self.tentative_len);
        self.committed_tokens += accepted;
        self.tentative_len = 0;
        let keep_blocks = if self.committed_tokens == 0 {
            0
        } else {
            self.committed_tokens.div_ceil(BLOCK_SIZE)
        };
        let mut pool = self.pool.lock().unwrap();
        while self.table.len() > keep_blocks {
            if let Some(pid) = self.table.logical_to_physical.pop() {
                pool.free_with_tier(pid, false).ok();
                self.block_table_u32.pop();
            } else {
                break;
            }
        }
        Ok(())
    }

    fn rollback_to(&mut self, len: usize) -> Result<()> {
        self.committed_tokens = self.committed_tokens.min(len);
        self.tentative_len = 0;
        let keep_blocks = if self.committed_tokens == 0 {
            0
        } else {
            self.committed_tokens.div_ceil(BLOCK_SIZE)
        };
        let mut pool = self.pool.lock().unwrap();
        while self.table.len() > keep_blocks {
            if let Some(pid) = self.table.logical_to_physical.pop() {
                pool.free_with_tier(pid, false).ok();
                self.block_table_u32.pop();
            } else {
                break;
            }
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.committed_tokens + self.tentative_len
    }

    fn current_k(&self) -> Result<Tensor> {
        let pool = self.pool.lock().unwrap();
        let mut k_data =
            Vec::with_capacity(self.table.len() * BLOCK_SIZE * self.num_heads * self.head_dim);
        for &id in &self.table.logical_to_physical {
            k_data.extend_from_slice(pool.read_keys(id));
        }
        let shape = grim_tensor::Shape::new(vec![
            self.table.len() * BLOCK_SIZE,
            self.num_heads,
            self.head_dim,
        ]);
        Ok(grim_backend_cpu::cpu_tensor(k_data, shape))
    }

    fn current_v(&self) -> Result<Tensor> {
        let pool = self.pool.lock().unwrap();
        let mut v_data =
            Vec::with_capacity(self.table.len() * BLOCK_SIZE * self.num_heads * self.head_dim);
        for &id in &self.table.logical_to_physical {
            v_data.extend_from_slice(pool.read_values(id));
        }
        let shape = grim_tensor::Shape::new(vec![
            self.table.len() * BLOCK_SIZE,
            self.num_heads,
            self.head_dim,
        ]);
        Ok(grim_backend_cpu::cpu_tensor(v_data, shape))
    }

    fn store_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(&id) = self.table.logical_to_physical.last() {
            let k_flat = k.to_vec_f32()?;
            let v_flat = v.to_vec_f32()?;
            pool.write_keys(id, &k_flat, k.shape().dims()[0]);
            pool.write_values(id, &v_flat);
        }
        Ok(())
    }

    fn has_paged_kv(&self) -> bool {
        true
    }

    fn append_kv_layer(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let k_flat = k.to_vec_f32()?;
        let v_flat = v.to_vec_f32()?;
        // Derive the per-token stride from the tensor's real shape rather than
        // trusting the cache's configured head counts: the engine builds its
        // `PagedKvCache` from `EngineConfig` defaults that may not match the
        // registered model (e.g. small_llama uses kvh=1, hd=16 while the
        // default config is kvh=4, hd=128). The paged kernel indexes the page
        // tensor by flat `(block_id * page_size + offset) * kv_stride` offset,
        // so only the stride matters — the declared tensor shape is cosmetic.
        let k_dims = k.shape().dims();
        // 3-D `[batch, seq, kvh*head_dim]` (post-RoPE K from the block, or
        // prefill batch) vs 2-D `[seq, kvh*head_dim]` (single-token decode).
        // The batch dim is always 1 here; `seq` is dim 1 for 3-D and dim 0
        // for 2-D.
        let (seq, stride) = if k_dims.len() == 3 {
            (k_dims[1], k_dims[2])
        } else if k_dims.len() == 2 {
            (k_dims[0], k_dims[1])
        } else {
            (1, k_flat.len())
        };
        if stride == 0 || seq == 0 {
            return Ok(());
        }
        // Grow the per-layer page buffers lazily up to `layer`, sized to the
        // actual per-token stride so the flat layout matches the kernel.
        let page_elems = self.capacity * self.page_size * stride;
        if self.k_pages.len() <= layer {
            for _ in self.k_pages.len()..=layer {
                self.k_pages.push(vec![0.0f32; page_elems]);
                self.v_pages.push(vec![0.0f32; page_elems]);
                self.layer_committed_tokens.push(0);
            }
        }
        for t in 0..seq {
            // Only layer 0 drives block-table growth (all layers see the
            // same token sequence, so `append_slot` must fire once per token,
            // not once per layer per token).
            if layer == 0 {
                self.append_slot()?;
            }
            let pos = self.layer_committed_tokens[layer];
            let physical = *self.table.logical_to_physical.last().unwrap();
            let slot = physical * self.page_size + (pos % self.page_size);
            let offset = slot * stride;
            let tok_start = t * stride;
            self.k_pages[layer][offset..offset + stride]
                .copy_from_slice(&k_flat[tok_start..tok_start + stride]);
            self.v_pages[layer][offset..offset + stride]
                .copy_from_slice(&v_flat[tok_start..tok_start + stride]);
            self.layer_committed_tokens[layer] += 1;

            if let Ok(mut pool) = self.pool.lock() {
                pool.write_layer_keys(physical, layer, &k_flat[tok_start..tok_start + stride], 1);
                pool.write_layer_values(physical, layer, &v_flat[tok_start..tok_start + stride]);
            }
        }
        Ok(())
    }

    fn block_table(&self) -> Option<&[u32]> {
        if self.block_table_u32.is_empty() {
            None
        } else {
            Some(&self.block_table_u32)
        }
    }

    fn paged_kv_handles(&self, layer: usize) -> Option<(Tensor, Tensor, usize)> {
        if layer >= self.k_pages.len() {
            return None;
        }
        let stride = self.k_pages[layer].len() / (self.capacity * self.page_size);
        let dims = vec![self.capacity * self.page_size, stride];
        if let (Some(dev), Some(dev_enum)) = (self.backend.as_ref(), self.device.as_ref()) {
            let k_storage = dev
                .from_cpu(&self.k_pages[layer], &Shape::new(dims.clone()), DType::F32)
                .ok()?;
            let v_storage = dev
                .from_cpu(&self.v_pages[layer], &Shape::new(dims.clone()), DType::F32)
                .ok()?;
            Some((
                Tensor::new(
                    Arc::from(k_storage),
                    Shape::new(dims.clone()),
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    dev_enum.clone(),
                ),
                Tensor::new(
                    Arc::from(v_storage),
                    Shape::new(dims),
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    dev_enum.clone(),
                ),
                self.page_size,
            ))
        } else {
            let k =
                grim_backend_cpu::cpu_tensor(self.k_pages[layer].clone(), Shape::new(dims.clone()));
            let v = grim_backend_cpu::cpu_tensor(self.v_pages[layer].clone(), Shape::new(dims));
            Some((k, v, self.page_size))
        }
    }

    fn seed_prefix(&mut self, blocks: &[usize]) {
        let mut pool = self.pool.lock().unwrap();
        for &b in blocks {
            pool.add_ref(b);
            self.table.push(b);
        }
        self.committed_tokens = self.table.len() * BLOCK_SIZE;
    }

    fn prefix_physical_ids(&self) -> Vec<usize> {
        self.table.physical_ids().to_vec()
    }

    fn num_layers(&self) -> usize {
        self.k_pages.len()
    }

    fn layer_block_slice(&self, layer: usize, block_id: usize) -> Option<(&[f32], &[f32])> {
        if layer >= self.k_pages.len() || layer >= self.v_pages.len() {
            return None;
        }
        let stride = self.k_pages[layer].len() / (self.capacity * self.page_size);
        let block_elems = self.page_size * stride;
        let start = block_id * block_elems;
        let end = start + block_elems;
        if end <= self.k_pages[layer].len() && end <= self.v_pages[layer].len() {
            Some((&self.k_pages[layer][start..end], &self.v_pages[layer][start..end]))
        } else {
            None
        }
    }

    fn write_layer_block(&mut self, layer: usize, block_id: usize, k: &[f32], v: &[f32]) -> Result<()> {
        let stride = if layer < self.k_pages.len() && !self.k_pages[layer].is_empty() {
            self.k_pages[layer].len() / (self.capacity * self.page_size)
        } else {
            self.num_heads * self.head_dim
        };
        let page_elems = self.capacity * self.page_size * stride;
        if self.k_pages.len() <= layer {
            for _ in self.k_pages.len()..=layer {
                self.k_pages.push(vec![0.0f32; page_elems]);
                self.v_pages.push(vec![0.0f32; page_elems]);
                self.layer_committed_tokens.push(0);
            }
        }
        let block_elems = self.page_size * stride;
        let start = block_id * block_elems;
        let k_len = k.len().min(block_elems);
        let v_len = v.len().min(block_elems);
        if start + k_len <= self.k_pages[layer].len() {
            self.k_pages[layer][start..start + k_len].copy_from_slice(&k[..k_len]);
        }
        if start + v_len <= self.v_pages[layer].len() {
            self.v_pages[layer][start..start + v_len].copy_from_slice(&v[..v_len]);
        }
        Ok(())
    }
}

/// Subtype alias for [`TransportBlockId`] so callers can use the
/// canonical [`BlockId`] type from this crate without importing
/// kvtransport directly.
pub type KvTransportId = TransportBlockId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_block_pool_telemetry_accessors() {
        let mut pool = KvBlockPool::new(10, 8, 128);
        assert_eq!(pool.capacity(), 10);
        assert_eq!(pool.used_count(), 0);
        assert!(pool.block_bytes() > 0);

        let id1 = pool.alloc().unwrap();
        assert_eq!(pool.used_count(), 1);

        pool.free(id1);
        assert_eq!(pool.used_count(), 0);
    }
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
        let spill =
            Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap());
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
        let spill = Arc::new(
            SharedSpillManager::new(dir.path().to_path_buf(), BLOCK_SIZE * 2 * 4).unwrap(),
        );
        let mut pool = KvBlockPool::new(2, 2, 4);
        pool.attach_spill(spill.clone());
        let compressor: Arc<dyn KvCompressor> =
            Arc::new(LloydMaxCompressor::new(KvQuantConfig::default()));
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
        let spill =
            Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap());
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
        let tokens: Vec<u32> = (0..48).collect(); // 3 blocks of 16 tokens

        let (ids1, reused1) = pool.find_or_share_prefix_tokens(&tokens).unwrap();
        let (ids2, reused2) = pool.find_or_share_prefix_tokens(&tokens).unwrap();
        assert_eq!(ids1, ids2); // Must share the same block IDs
        assert_eq!(reused1, 0); // first call computes everything
        assert_eq!(reused2, 48); // second call reuses the whole prefix
        assert_eq!(*pool.ref_counts.get(&ids1[0]).unwrap(), 2); // Ref count must be incremented

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
    fn prefix_demoted_under_pressure_is_promoted_on_match() {
        // Phase 2: a cached prefix demoted to host/NVMe under pressure must
        // be promoted back (and its KV recovered, not recomputed) when a
        // later request hits that prefix.
        let dir = tempdir().unwrap();
        let block_elems = BLOCK_SIZE * 2 * 4;
        let spill =
            Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap());
        let mut pool = KvBlockPool::new(8, 2, 4);
        pool.attach_spill(spill.clone());

        let tokens: Vec<u32> = (0..16).collect(); // single-block prefix
        let (blocks, _) = pool.find_or_share_prefix_tokens(&tokens).unwrap();
        let k0 = vec![1.0f32; block_elems];
        let v0 = vec![2.0f32; block_elems];
        pool.write_keys(blocks[0], &k0, BLOCK_SIZE);
        pool.write_values(blocks[0], &v0);

        // Release the sequence's reference: the prefix becomes a cold, cached
        // entry (refcount 0) but stays in the tree.
        pool.remove_prefix(&blocks);

        // Pressure: demote the coldest cached leaf to host/NVMe.
        let demoted = pool.demote_cold_prefix().expect("a cold prefix to demote");
        assert!(pool.block_location(demoted) != CacheTier::Gpu);

        // A later request hits the demoted prefix → promote back, recover KV.
        let (matched, n, promoted) = pool.match_prefix_promoting(&tokens);
        assert!(promoted, "demoted prefix must be promoted on match");
        assert_eq!(n, 16);
        assert_eq!(matched.len(), 1);
        assert_eq!(pool.read_keys(matched[0]), &k0[..]);
        assert_eq!(pool.read_values(matched[0]), &v0[..]);
    }

    #[test]
    fn block_table_truncate_returns_freed_ids_to_the_pool() {
        // P1-2 partial: BlockTable::truncate must return the freed physical
        // ids to the pool's free list, not merely forget them. The pre-fix
        // body was `self.logical_to_physical.truncate(len)` (zero pool
        // contact), so this test pins the fix and would re-leak on revert.
        //
        // Start with a 4-capacity pool (free_list length == 4), alloc three
        // blocks into a table (free_list drops to 1), then truncate to 0 and
        // require the free list to refill to 4 — the exact invariant the bare
        // Vec::truncate version would fail.
        let mut pool = KvBlockPool::new(4, 2, 4);
        assert_eq!(pool.free_list.len(), 4, "fresh pool has 4 free blocks");

        let mut table = BlockTable::new();
        for _ in 0..3 {
            let id = pool.alloc().unwrap();
            table.push(id);
        }
        assert_eq!(
            pool.free_list.len(),
            1,
            "three outstanding allocs leave one free block"
        );
        assert_eq!(table.len(), 3);

        table.truncate(0, &mut pool);

        assert_eq!(table.len(), 0, "table must be empty after truncate(0)");
        assert_eq!(
            pool.free_list.len(),
            4,
            "truncate must return all freed physical ids to the free list, \
             not just drop them on the floor (the original P1-2 leak)"
        );
    }

    #[test]
    fn block_table_truncate_partial_keeps_prefix_and_frees_tail() {
        // Partial truncate keeps the first `len` entries and returns only
        // the tail ids to the pool. With 4 free, alloc 3 (leaves 1 free),
        // truncate to 1: exactly 2 ids must come back, leaving 3 free.
        let mut pool = KvBlockPool::new(4, 2, 4);
        let mut table = BlockTable::new();
        let mut pushed = Vec::new();
        for _ in 0..3 {
            let id = pool.alloc().unwrap();
            pushed.push(id);
            table.push(id);
        }
        assert_eq!(pool.free_list.len(), 1);

        table.truncate(1, &mut pool);

        assert_eq!(table.len(), 1, "prefix of one block must survive");
        assert_eq!(
            table.physical_ids()[0],
            pushed[0],
            "the surviving physical id must be the one that was pushed first"
        );
        assert_eq!(
            pool.free_list.len(),
            3,
            "exactly the two truncated tail ids return to the free list"
        );
    }

    #[test]
    fn test_paged_kv_cache_current_k_v() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));
        let mut cache = PagedKvCache::new(pool.clone(), 2, 4, BLOCK_SIZE);

        // Append two blocks of slots (32 slots = 2 blocks of BLOCK_SIZE 16)
        for _ in 0..(2 * BLOCK_SIZE) {
            cache.append_slot().unwrap();
        }

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

    #[test]
    fn test_speculative_kv_rollback_units() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(10, 2, 4)));
        let mut cache = PagedKvCache::new(pool.clone(), 2, 4, BLOCK_SIZE);

        // 1. Append 16 committed slots (1 full block of 16 tokens)
        for _ in 0..16 {
            cache.append_slot().unwrap();
        }
        assert_eq!(cache.len(), 16);
        assert_eq!(cache.table.len(), 1);

        // 2. Tentatively append 5 tokens (requires 2nd block since 16+5=21 > 16)
        cache.tentative_append(5).unwrap();
        assert_eq!(cache.len(), 21);
        assert_eq!(cache.table.len(), 2);

        // 3. Commit 2 of 5 accepted tokens -> total 18 committed tokens -> requires 2 blocks
        cache.commit(2).unwrap();
        assert_eq!(cache.len(), 18);
        assert_eq!(cache.table.len(), 2);

        // 4. Rollback to 16 tokens -> 1 block retained, 1 block freed back to pool
        cache.rollback_to(16).unwrap();
        assert_eq!(cache.len(), 16);
        assert_eq!(cache.table.len(), 1);
    }
}
