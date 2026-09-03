//! Paged KV cache memory pool, logical block tables, prefix sharing, and multi-tier spilling.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use grim_backend_cpu::CpuDevice;
use grim_core::error::{Error, Result};
use grim_core::kv_cache::KvCache;
use grim_kvquant::{CompressedKvBlock, KvCompressor};
use grim_kvtransport::{BlockId as TransportBlockId, CacheTier, SharedSpillManager};
use grim_tensor::{DType, Device, Shape, Tensor};

/// MoE resident-set HBM budget (`rocm_kernel_plan.md` WI-C).
pub mod moe_budget;

pub use moe_budget::{ElasticMoEAllocation, LruResidencyTracker, MoeResidentBudget, ResidentTier};

/// Block-granular radix tree for prefix (RadixAttention-style) KV sharing.
pub mod radix;

pub use radix::RadixTree;

/// Semantic-aware state caching for recurrent/hybrid-attention models.
pub mod semantic_anchor;

pub use semantic_anchor::{
    CheckpointId, RecurrentCheckpointPool, RecurrentLayerState, RecurrentStateCheckpoint,
    SemanticAnchorRegistry,
};

/// Paged KV cache device-resident mirror and asynchronous tiering coordinator (F10).
pub mod kv_mirror;

pub use kv_mirror::{KvDeviceMirror, KvDeviceMirrorConfig, MirrorSyncState};

/// Chunk-aligned intermediate hidden-state cache on host pinned memory.
pub mod hidden_state_store;

pub use hidden_state_store::{
    HiddenStateChunk, HiddenStateKey, HiddenStateStore, SharedHiddenStateStore,
};

/// Multimodal vision and audio dense embedding cache.
pub mod encoder_cache;

pub use encoder_cache::{EncoderCache, EncoderEmbedding, EncoderKey, SharedEncoderCache};

/// Heterogeneous KV layer grouping and layout metadata.
pub mod layer_groups;

pub use layer_groups::{LayerGroupIdentity, LayerGroupRegistry};

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
    fn block_num_tokens(&self, id: BlockId) -> Option<usize> {
        KvBlockPool::block_num_tokens(self, id)
    }
    // F8/F10: pull-mode fetch support. The inherent methods already exist;
    // they were just never exposed through the trait the KV receiver server
    // is generic over, so the server had no way to answer a fetch request.
    // Call via the inherent path (`KvBlockPool::read_keys`) — the shared
    // name would otherwise resolve to the trait method being defined here.
    fn read_keys(&self, id: BlockId) -> Option<Vec<f32>> {
        if id < self.num_blocks() {
            Some(KvBlockPool::read_keys(self, id).to_vec())
        } else {
            None
        }
    }
    fn read_values(&self, id: BlockId) -> Option<Vec<f32>> {
        if id < self.num_blocks() {
            Some(KvBlockPool::read_values(self, id).to_vec())
        } else {
            None
        }
    }
    fn read_layer_keys(&self, id: BlockId, layer_idx: u32) -> Option<Vec<f32>> {
        self.read_layer_keys(id, layer_idx as usize)
            .map(|s| s.to_vec())
    }
    fn read_layer_values(&self, id: BlockId, layer_idx: u32) -> Option<Vec<f32>> {
        self.read_layer_values(id, layer_idx as usize)
            .map(|s| s.to_vec())
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
    /// Checkpoint pool for recurrent/hybrid-attention models anchored at semantic boundaries.
    pub recurrent_checkpoints: RecurrentCheckpointPool,
    /// Registry of semantic boundary token IDs (e.g. <think>, </tool_call>).
    pub anchor_registry: SemanticAnchorRegistry,
    /// Layout configuration: block-major switch tied to the rocm-aiter feature flag
    block_major_layout: bool,
    /// Block ids that recently had their refcount drop to zero — kept
    /// here for one cycle so the next `free` knows there might be data
    /// in the spill tier to return.
    recently_zero: VecDeque<BlockId>,
    /// Set of block IDs whose contents have been modified on the GPU tier
    /// and have not yet been synchronized/flushed to the host device mirror.
    dirty_blocks: HashSet<BlockId>,
    /// Device-resident KV mirror managing dual-tier host replication and watermark eviction.
    pub device_mirror: KvDeviceMirror,
    num_heads: usize,
    head_dim: usize,
    /// Target hardware device ordinal assigned to this pool (default 0).
    device_ordinal: usize,
    /// Optional assigned layer index range [start_layer, end_layer) for pipeline parallelism.
    layer_range: Option<(usize, usize)>,
    compressor: Option<Arc<dyn KvCompressor>>,
    spill: Option<Arc<SharedSpillManager>>,
    /// Number of bytes per block (`BLOCK_SIZE * num_heads * head_dim * 4`).
    block_bytes: usize,
}

impl KvBlockPool {
    pub fn new(capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        Self::new_on_device(capacity, num_heads, head_dim, 0)
    }

    /// Construct a new KV block pool pinned to a specific hardware device ordinal.
    pub fn new_on_device(
        capacity: usize,
        num_heads: usize,
        head_dim: usize,
        device_ordinal: usize,
    ) -> Self {
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
        let device_mirror = KvDeviceMirror::new(
            capacity,
            num_heads,
            head_dim,
            KvDeviceMirrorConfig::default(),
        );
        Self {
            blocks,
            free_list,
            ref_counts: HashMap::new(),
            prefix_tree: RadixTree::new(BLOCK_SIZE),
            ssm_states: HashMap::new(),
            recurrent_checkpoints: RecurrentCheckpointPool::new(64),
            anchor_registry: SemanticAnchorRegistry::new(vec![151644, 151645, 32000, 32001]),
            block_major_layout,
            recently_zero: VecDeque::new(),
            dirty_blocks: HashSet::new(),
            device_mirror,
            num_heads,
            head_dim,
            device_ordinal,
            layer_range: None,
            compressor: None,
            spill: None,
            block_bytes: block_elem * std::mem::size_of::<f32>(),
        }
    }

    /// Pin this KV block pool to a specific pipeline stage layer range [start_layer, end_layer).
    pub fn with_layer_range(mut self, start_layer: usize, end_layer: usize) -> Self {
        self.layer_range = Some((start_layer, end_layer));
        self
    }

    /// Hardware device ordinal assigned to this pool.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Pipeline stage layer range assigned to this pool, if any.
    pub fn layer_range(&self) -> Option<(usize, usize)> {
        self.layer_range
    }

    /// Check if this KV pool manages the specified transformer layer.
    pub fn owns_layer(&self, layer_idx: usize) -> bool {
        match self.layer_range {
            Some((start, end)) => layer_idx >= start && layer_idx < end,
            None => true,
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

    /// Match prefix and return the deepest valid recurrent checkpoint snapshot for hybrid models.
    pub fn match_prefix_with_recurrent(
        &mut self,
        tokens: &[u32],
    ) -> (Vec<BlockId>, usize, Option<Arc<RecurrentStateCheckpoint>>) {
        let (matched, count, state_id) = self.prefix_tree.match_prefix_with_anchor(tokens);
        let checkpoint = state_id.and_then(|id| self.recurrent_checkpoints.get_by_node(id));
        (matched, count, checkpoint)
    }

    /// Register newly computed `blocks` for `tokens` after a prefill
    /// completes. Shared prefix nodes are reused; diverging blocks become
    /// new tree nodes.
    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[BlockId]) {
        self.prefix_tree.insert(tokens, blocks);
        self.prefix_tree.touch(tokens);
    }

    /// Register computed blocks and attach recurrent state snapshots at detected semantic anchors.
    pub fn insert_prefix_with_recurrent_state(
        &mut self,
        tokens: &[u32],
        blocks: &[BlockId],
        layer_states: Vec<RecurrentLayerState>,
    ) {
        self.insert_prefix(tokens, blocks);

        // Find semantic anchors in the prompt
        let anchors = self.anchor_registry.find_anchors(tokens);
        if let Some(&last_anchor_offset) = anchors.last() {
            // Find which block index covers this anchor
            let block_idx = (last_anchor_offset / BLOCK_SIZE).min(blocks.len().saturating_sub(1));
            if block_idx < blocks.len() {
                let target_block_id = blocks[block_idx];
                let _cp = self.recurrent_checkpoints.store_checkpoint(
                    target_block_id,
                    last_anchor_offset,
                    layer_states,
                );
                self.prefix_tree
                    .attach_recurrent_state(target_block_id, target_block_id);
            }
        }
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
            if self.blocks[bid].location != CacheTier::Gpu
                && self.promote_to_gpu(bid).ok().flatten().is_some()
            {
                promoted = true;
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

    /// Match prefix tokens returning matched blocks, full token count, and whether blending is available.
    pub fn match_prefix_blending(&self, tokens: &[u32]) -> (Vec<BlockId>, usize, bool) {
        self.prefix_tree.match_prefix_blending(tokens)
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
    ///
    /// Demotion is tried compressed first (compressor + spill attached),
    /// then raw. If BOTH demotion paths fail, the block falls back to the
    /// in-place release (zero + free list) — a failed demotion must never
    /// strand the block in a fake `HostRam` state with no free-list path,
    /// because nothing else would ever reclaim the slot.
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
            let mut demoted = false;
            // 1. Compressed path: the compressor's serialized block IS the
            // spilled bytes — closes the compression-to-spill loop instead
            // of recording compression as metadata only.
            if let Some(_c) = self.compressor.as_ref() {
                if let Ok(Some(compressed)) = self.compress_block(id) {
                    match spill.demote_compressed(id, compressed.to_bytes()) {
                        Ok(()) => {
                            if let Err(e) = spill.demote_compressed_to_nvme(id) {
                                eprintln!(
                                    "[BlockPool] compressed demote_to_nvme failed for block {id} (host copy retained): {e}"
                                );
                            }
                            demoted = true;
                        }
                        Err(e) => {
                            eprintln!(
                                "[BlockPool] compressed demote_to_host failed for block {id}, falling back to raw: {e}"
                            );
                        }
                    }
                }
            }
            // 2. Raw f32 path (also the fallback when compression failed).
            if !demoted {
                let k = self.blocks[id].key_data.clone();
                let v = self.blocks[id].value_data.clone();
                match spill.demote_to_host(id, k, v) {
                    Ok(()) => {
                        if let Err(e) = spill.demote_to_nvme(id) {
                            eprintln!(
                                "[BlockPool] demote_to_nvme failed for block {id} (host copy retained): {e}"
                            );
                        }
                        demoted = true;
                    }
                    Err(e) => {
                        eprintln!(
                            "[BlockPool] demote_to_host failed for block {id}: {e} — releasing in place"
                        );
                    }
                }
            }
            if demoted {
                // Mark the block as demoted so promotion can be decided later
                // without re-querying the spill manager.
                self.blocks[id].location = CacheTier::HostRam;
                self.recently_zero.push_back(id);
                // Do NOT push to free_list — the block is spilled, not available
                // for fresh allocation. Only promote_to_gpu can reclaim it.
            } else {
                // Both demotion paths failed: the block must stay reclaimable.
                // Zero it in place and return it to the free list rather than
                // stranding GPU capacity in a fake spilled state.
                self.blocks[id].num_tokens = 0;
                self.blocks[id].received = false;
                self.blocks[id].key_data.fill(0.0);
                self.blocks[id].value_data.fill(0.0);
                self.blocks[id].location = CacheTier::Gpu;
                self.free_list.push_back(id);
            }
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

    /// Promote a previously demoted block back to GPU resident. Compressed
    /// spill is decompressed back to f32 K/V first; raw spill is validated
    /// STRICTLY: the retrieved lengths must match the block capacity
    /// exactly, otherwise this is an `Err` — never a silent `.min()`
    /// truncation of cache contents. On success the retrieved key/value
    /// data is written back into the block and its `location` restored to
    /// [`CacheTier::Gpu`]. Returns the contents if promotion succeeded
    /// (the block was demoted), or `None` if there was nothing to promote
    /// (block already GPU-resident or no spill manager).
    pub fn promote_to_gpu(&mut self, id: BlockId) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let Some(spill) = self.spill.as_ref() else {
            return Ok(None);
        };
        // Compressed spill first: decompress back to full f32 K/V.
        let (k, v) = match spill.retrieve_compressed(id)? {
            Some(blob) => {
                let compressor = self.compressor.as_ref().ok_or_else(|| {
                    Error::KvCache(
                        "compressed spill block found but no compressor attached".into(),
                    )
                })?;
                let block = CompressedKvBlock::from_bytes(&blob)?;
                let (keys, values) =
                    compressor.dequantize_for_attention(&block, &CpuDevice::new(), Device::Cpu)?;
                (keys.to_vec_f32()?, values.to_vec_f32()?)
            }
            None => match spill.retrieve(id)? {
                Some((k, v)) => (k, v),
                None => return Ok(None),
            },
        };
        // Strict capacity validation — a mismatch is an error, not silent
        // truncation (the pre-fix `.min()` copies silently dropped cache
        // rows when the spill geometry disagreed with the pool).
        let key_cap = self.blocks[id].key_data.len();
        let val_cap = self.blocks[id].value_data.len();
        if k.len() != key_cap || v.len() != val_cap {
            return Err(Error::KvCache(format!(
                "promote_to_gpu: spill geometry mismatch for block {id} \
                 (k {} vs {key_cap}, v {} vs {val_cap})",
                k.len(),
                v.len()
            )));
        }
        let elem = self.num_heads * self.head_dim;
        let n = (k.len() / elem).min(BLOCK_SIZE);
        self.blocks[id].key_data.copy_from_slice(&k);
        self.blocks[id].value_data.copy_from_slice(&v);
        self.blocks[id].num_tokens = n;
        self.blocks[id].received = true;
        self.blocks[id].location = CacheTier::Gpu;
        self.dirty_blocks.remove(&id);
        Ok(Some((k, v)))
    }

    /// Mark a block as dirty (GPU-modified and needing mirror synchronization).
    pub fn mark_dirty(&mut self, id: BlockId) {
        if id < self.blocks.len() {
            self.dirty_blocks.insert(id);
        }
    }

    /// Check if a block has pending unsynchronized writes.
    pub fn is_dirty(&self, id: BlockId) -> bool {
        self.dirty_blocks.contains(&id)
    }

    /// Number of dirty blocks pending synchronization to the host device mirror.
    pub fn dirty_count(&self) -> usize {
        self.dirty_blocks.len()
    }

    /// Asynchronously flushes all dirty GPU blocks to the host mirror without
    /// deallocating GPU residency or stalling the decode stream.
    pub fn flush_dirty_to_host(&mut self) -> Result<usize> {
        let Some(spill) = self.spill.as_ref() else {
            let count = self.dirty_blocks.len();
            self.dirty_blocks.clear();
            return Ok(count);
        };

        let dirty_ids: Vec<BlockId> = self.dirty_blocks.drain().collect();
        let count = dirty_ids.len();
        for id in dirty_ids {
            if id < self.blocks.len() && self.blocks[id].location == CacheTier::Gpu {
                let k = self.blocks[id].key_data.clone();
                let v = self.blocks[id].value_data.clone();
                spill.demote_to_host(id, k, v)?;
            }
        }
        Ok(count)
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
            let mut demoted = false;
            // Compressed-first demotion (mirrors free_with_tier).
            if self.compressor.is_some() {
                if let Ok(Some(compressed)) = self.compress_block(bid) {
                    if spill.demote_compressed(bid, compressed.to_bytes()).is_ok() {
                        if let Err(e) = spill.demote_compressed_to_nvme(bid) {
                            eprintln!(
                                "[BlockPool] evict compressed demote_to_nvme failed for {bid} (host copy retained): {e}"
                            );
                        }
                        demoted = true;
                    }
                }
            }
            if !demoted {
                let k = self.blocks[bid].key_data.clone();
                let v = self.blocks[bid].value_data.clone();
                if spill.demote_to_host(bid, k, v).is_ok() {
                    if let Err(e) = spill.demote_to_nvme(bid) {
                        eprintln!(
                            "[BlockPool] evict demote_to_nvme failed for {bid} (host copy retained): {e}"
                        );
                    }
                    demoted = true;
                } else {
                    eprintln!(
                        "[BlockPool] evict demote_to_host failed for {bid} — releasing in place"
                    );
                }
            }
            if demoted {
                self.blocks[bid].location = CacheTier::HostRam;
                self.recently_zero.push_back(bid);
                // Do NOT push to free_list — spilled blocks are not available for
                // fresh allocation. Only promote_to_gpu can reclaim them.
                self.ref_counts.remove(&bid);
                return true;
            }
            // Demotion failed: fall through to the in-place release below so
            // the slot is reclaimable instead of stranded in a fake spilled
            // state with no valid backing storage.
        }
        self.blocks[bid].num_tokens = 0;
        self.blocks[bid].received = false;
        self.blocks[bid].key_data.fill(0.0);
        self.blocks[bid].value_data.fill(0.0);
        self.ref_counts.remove(&bid);
        self.free_list.push_back(bid);
        true
    }

    /// Compress the latest snapshot of `id` via the attached
    /// compressor and expose the [`CompressedKvBlock`]. `None` if no
    /// compressor is attached or the block holds no tokens — the snapshot
    /// is taken at the block's ACTUAL `num_tokens` rows, never padded to
    /// [`BLOCK_SIZE`], so compression does no work on padding.
    pub fn compress_block(&self, id: BlockId) -> Result<Option<CompressedKvBlock>> {
        let c = match self.compressor.as_ref() {
            Some(c) => c,
            None => return Ok(None),
        };
        if self.blocks[id].num_tokens == 0 {
            return Ok(None);
        }
        let (k, v) = self.snapshot_block(id);
        c.compress(&k, &v).map(Some)
    }

    fn snapshot_block(&self, id: BlockId) -> (Tensor, Tensor) {
        let rows = self.blocks[id].num_tokens.clamp(1, BLOCK_SIZE);
        let elem = self.num_heads * self.head_dim;
        let shape = grim_tensor::Shape::new(vec![rows, self.num_heads, self.head_dim]);
        let k_tensor = grim_backend_cpu::cpu_tensor(
            self.blocks[id].key_data[..rows * elem].to_vec(),
            shape.clone(),
        );
        let v_tensor = grim_backend_cpu::cpu_tensor(
            self.blocks[id].value_data[..rows * elem].to_vec(),
            shape,
        );
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

    /// Dynamically resize the block pool capacity at a scheduler safe point.
    pub fn resize_capacity(&mut self, new_capacity: usize) {
        let cur_cap = self.blocks.len();
        if new_capacity > cur_cap {
            let block_elem = BLOCK_SIZE * self.num_heads * self.head_dim;
            for i in cur_cap..new_capacity {
                self.blocks.push(KvBlock {
                    _id: i,
                    key_data: vec![0.0; block_elem],
                    value_data: vec![0.0; block_elem],
                    layer_keys: Vec::new(),
                    layer_values: Vec::new(),
                    num_tokens: 0,
                    received: false,
                    location: CacheTier::Gpu,
                });
                self.free_list.push_back(i);
            }
        } else if new_capacity < cur_cap {
            // Reclaim unreferenced blocks from the free list
            while self.blocks.len() > new_capacity {
                if let Some(pos) = self
                    .free_list
                    .iter()
                    .position(|&id| id == self.blocks.len() - 1)
                {
                    self.free_list.remove(pos);
                    self.blocks.pop();
                } else {
                    break;
                }
            }
        }
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
        self.dirty_blocks.insert(id);
    }

    pub fn write_values(&mut self, id: BlockId, values: &[f32]) {
        let block = &mut self.blocks[id];
        let n = block.num_tokens;
        let elem = self.num_heads * self.head_dim;
        let len = (n * elem).min(values.len());
        block.value_data[..len].copy_from_slice(&values[..len]);
        self.dirty_blocks.insert(id);
    }

    pub fn write_layer_keys(&mut self, id: BlockId, layer: usize, keys: &[f32], num_tokens: usize) {
        if id >= self.blocks.len() {
            return;
        }
        let block_elem = BLOCK_SIZE * self.num_heads * self.head_dim;
        if self.blocks[id].layer_keys.len() <= layer {
            self.blocks[id]
                .layer_keys
                .resize_with(layer + 1, || vec![0.0; block_elem]);
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
        self.dirty_blocks.insert(id);
    }

    pub fn write_layer_values(&mut self, id: BlockId, layer: usize, values: &[f32]) {
        if id >= self.blocks.len() {
            return;
        }
        let block_elem = BLOCK_SIZE * self.num_heads * self.head_dim;
        if self.blocks[id].layer_values.len() <= layer {
            self.blocks[id]
                .layer_values
                .resize_with(layer + 1, || vec![0.0; block_elem]);
        }
        let len = values.len().min(block_elem);
        self.blocks[id].layer_values[layer][..len].copy_from_slice(&values[..len]);
        if layer == 0 {
            self.blocks[id].value_data[..len].copy_from_slice(&values[..len]);
        }
        self.dirty_blocks.insert(id);
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

    /// Valid token count stored for block `id` (0 for out-of-range or
    /// never-written blocks). This is the value handoffs must preserve:
    /// deriving the count from the buffer length reports every block as
    /// full because buffers are zero-padded to block capacity.
    pub fn block_num_tokens(&self, id: BlockId) -> Option<usize> {
        self.blocks
            .get(id)
            .filter(|b| b.received)
            .map(|b| b.num_tokens)
    }

    /// Explicitly set whether block `id` holds complete KV data; clearing
    /// the flag also resets `num_tokens` so the block reads as empty.
    ///
    /// NOTE (F8/F10 audit): an earlier comment described a disagg pull path
    /// that clears this flag on partial multi-layer fetch failure "so a
    /// partial transfer is retried next tick." No such retry/tick machinery
    /// exists — `DisaggRouter::fetch_kv_block` is a single blocking
    /// whole-block call. The flag is available for a future per-layer
    /// retry design, but nothing implements it today; do not rely on it.
    pub fn set_received(&mut self, id: BlockId, received: bool) {
        if let Some(b) = self.blocks.get_mut(id) {
            if !received {
                b.num_tokens = 0;
            }
            b.received = received;
        }
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

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn is_empty(&self) -> bool {
        self.logical_to_physical.is_empty()
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
    /// F10: dirty-block device KV mirror state (interior-mutable; the KvCache
    /// trait hands out `&self`).
    mirror_state: std::sync::Mutex<DeviceKvMirror>,
    /// WI-kv: cached device-resident block table (`BlockTableEntry` ABI),
    /// keyed on (len, first id, last id). Skips the per-layer-per-token H2D
    /// upload of the table in the paged-attention decode path.
    gpu_block_table: std::sync::Mutex<Option<GpuBlockTableCache>>,
}

/// Cached device block table + the fingerprint it was built from.
struct GpuBlockTableCache {
    fingerprint: (usize, u32, u32),
    storage: Arc<dyn grim_tensor::BackendStorage>,
}

/// F10: per-block device-resident K/V mirror. Appended KV history is
/// immutable, so each (layer, physical block) uploads to the device exactly
/// once — dirty re-staging only ever touches the ACTIVE tail block, instead
/// of re-pushing the full layer on every append.
pub type DeviceKvMirrorBlocks = HashMap<
    (usize, usize),
    (
        Arc<dyn grim_tensor::BackendStorage>,
        Arc<dyn grim_tensor::BackendStorage>,
    ),
>;

#[derive(Default)]
pub struct DeviceKvMirror {
    /// (layer, physical block) → device-resident (K, V) storage.
    pub blocks: DeviceKvMirrorBlocks,
    /// Blocks whose host pages changed since their last device upload.
    pub dirty: std::collections::BTreeSet<(usize, usize)>,
    /// Total host→device block uploads performed (ITL gate metric).
    pub total_uploads: u64,
    /// Distinct blocks ever uploaded.
    pub uploaded_elems: u64,
    /// F10: persistent full-layer K/V device buffers. Keyed by layer, shared
    /// into every returned Tensor — new K/V land via device-side region writes
    /// of just the dirty tail block instead of whole-layer host re-uploads
    /// (16 MB · 24 layers · every token → KB-scale).
    pub k_full: HashMap<usize, Arc<dyn grim_tensor::BackendStorage>>,
    pub v_full: HashMap<usize, Arc<dyn grim_tensor::BackendStorage>>,
}

impl PagedKvCache {
    /// F10 ITL gate metrics: (total block uploads, distinct blocks uploaded,
    /// uploaded elements, pending dirty count).
    pub fn mirror_stats(&self) -> (u64, u64, u64, usize) {
        let m = self.mirror_state.lock().unwrap_or_else(|e| e.into_inner());
        let unique = m
            .blocks
            .keys()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        (
            m.total_uploads,
            unique as u64,
            m.uploaded_elems,
            m.dirty.len(),
        )
    }

    pub fn new(
        pool: Arc<Mutex<KvBlockPool>>,
        num_heads: usize,
        head_dim: usize,
        page_size_: usize,
    ) -> Self {
        let capacity = pool.lock().unwrap_or_else(|e| e.into_inner()).capacity();
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
            mirror_state: std::sync::Mutex::new(DeviceKvMirror::default()),
            gpu_block_table: std::sync::Mutex::new(None),
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

    /// Valid token count stored in physical block `block_id` of the logical
    /// block table: every block is a full page except the tail, which
    /// carries the `committed_tokens` remainder. `None` when the id is
    /// outside the table. Handoffs use this so a partially-filled tail
    /// block does not arrive marked as fully valid.
    pub fn block_num_tokens(&self, block_id: usize) -> Option<usize> {
        let num_blocks = self.table.len();
        if block_id >= num_blocks {
            return None;
        }
        let full_blocks = self.committed_tokens / self.page_size;
        let tail = self.committed_tokens % self.page_size;
        if block_id < full_blocks {
            Some(self.page_size)
        } else if block_id == full_blocks {
            Some(if tail == 0 { self.page_size } else { tail })
        } else {
            Some(0)
        }
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
            Some((
                &self.k_pages[layer][start..end],
                &self.v_pages[layer][start..end],
            ))
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
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
        let pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
        let pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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

            // WI-perf (decode fast path): when the persistent full-layer device
            // buffer already exists and this append is a single token (decode),
            // push the K/V row device-to-device straight into it. The device
            // copy stays current without the host→device dirty-region upload in
            // `paged_kv_handles`, so steady-state decode issues zero H2D traffic.
            // Prefill (seq > 1) keeps the host-page + dirty-marking path: the
            // first `paged_kv_handles` call materializes `k_full`/`v_full` and
            // back-fills only the touched blocks.
            let mut device_appended = false;
            if seq == 1 {
                let m = self.mirror_state.lock().unwrap_or_else(|e| e.into_inner());
                if let (Some(k_arc), Some(v_arc)) = (m.k_full.get(&layer), m.v_full.get(&layer)) {
                    let (k_arc, v_arc) = (k_arc.clone(), v_arc.clone());
                    drop(m);
                    if let Some(dev) = self.backend.as_ref() {
                        let ok_k = dev
                            .copy_slice_into(
                                &*k_arc,
                                k.storage().as_ref(),
                                offset,
                                stride,
                            )
                            .is_ok();
                        let ok_v = dev
                            .copy_slice_into(
                                &*v_arc,
                                v.storage().as_ref(),
                                offset,
                                stride,
                            )
                            .is_ok();
                        device_appended = ok_k && ok_v;
                        if device_appended {
                            // Count the D2D tail refresh as a block upload for
                            // the ITL gate stats (same semantics as the H2D
                            // dirty-region refresh it replaces).
                            let mut m =
                                self.mirror_state.lock().unwrap_or_else(|e| e.into_inner());
                            m.total_uploads += 1;
                            m.uploaded_elems += (stride * 2) as u64;
                        }
                    }
                }
            }
            if !device_appended {
                // F10: mark the touched (layer, block) dirty. Per-block device
                // upload happens lazily in paged_kv_handles — once per block
                // lifetime for sealed history, only the active tail block while
                // it receives tokens. The old eager path re-uploaded the FULL
                // layer on every single token.
                let mut m = self.mirror_state.lock().unwrap_or_else(|e| e.into_inner());
                m.dirty.insert((layer, physical));
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

    fn block_table_gpu_handle(
        &self,
    ) -> Option<std::sync::Arc<dyn grim_tensor::backend::BackendStorage>> {
        let bt = self.block_table()?;
        let fingerprint = (
            bt.len(),
            bt.first().copied().unwrap_or(0),
            *bt.last().unwrap(),
        );
        let mut g = self.gpu_block_table.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = g.as_ref() {
            if cached.fingerprint == fingerprint {
                return Some(cached.storage.clone());
            }
        }
        // Rebuild: `BlockTableEntry { block_id: u32, page_size: u32 }` pairs,
        // same ABI as the kernel — see block.rs::paged_self_attention.
        let pairs: Vec<f32> = bt
            .iter()
            .flat_map(|&b| [f32::from_bits(b), f32::from_bits(self.page_size as u32)])
            .collect();
        let dev = self.backend.as_ref()?;
        let storage = dev.from_cpu(
            &pairs,
            &Shape::new(vec![pairs.len()]),
            DType::F32,
        )
        .ok()?;
        let arc: Arc<dyn grim_tensor::backend::BackendStorage> = Arc::from(storage);
        *g = Some(GpuBlockTableCache {
            fingerprint,
            storage: arc.clone(),
        });
        Some(arc)
    }

    fn paged_kv_handles(&self, layer: usize) -> Option<(Tensor, Tensor, usize)> {
        if layer >= self.k_pages.len() {
            return None;
        }
        let stride = self.k_pages[layer].len() / (self.capacity * self.page_size);
        let dims = vec![self.capacity * self.page_size, stride];

        // F10: fast path — serve cached layer storage only when this layer
        // has no dirty blocks; otherwise fall through to restaging below.
        let layer_dirty = self
            .mirror_state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .dirty
            .range((layer, 0)..(layer + 1, 0))
            .next()
            .is_some();
        if !layer_dirty {
            let m = self.mirror_state.lock().unwrap_or_else(|e| e.into_inner());
            if let (Some(dev_enum), Some(k_storage), Some(v_storage)) = (
                self.device.as_ref(),
                m.k_full.get(&layer),
                m.v_full.get(&layer),
            ) {
                return Some((
                    Tensor::new(
                        k_storage.clone(),
                        Shape::new(dims.clone()),
                        DType::F32,
                        grim_tensor::QuantProvenance::default(),
                        dev_enum.clone(),
                    ),
                    Tensor::new(
                        v_storage.clone(),
                        Shape::new(dims),
                        DType::F32,
                        grim_tensor::QuantProvenance::default(),
                        dev_enum.clone(),
                    ),
                    self.page_size,
                ));
            }
        } // F10: end !layer_dirty fast path

        if let (Some(dev), Some(dev_enum)) = (self.backend.as_ref(), self.device.as_ref()) {
            let mut m = self.mirror_state.lock().unwrap_or_else(|e| e.into_inner());
            let dirty_here: Vec<usize> = m
                .dirty
                .range((layer, 0)..(layer + 1, 0))
                .map(|&(_, b)| b)
                .collect();

            // F10: persistent full-layer device buffers. First call uploads the
            // whole layer once; later calls push ONLY dirty tail blocks into the
            // same buffer via host→device slice uploads. The old path re-uploaded
            // the full [capacity·page_size, stride] layer (16 MB for 2×128 head
            // config) on every call — that was the per-token H2D flood.
            let k_arc = match m.k_full.get(&layer) {
                Some(a) => a.clone(),
                None => {
                    let k_storage = dev
                        .from_cpu(&self.k_pages[layer], &Shape::new(dims.clone()), DType::F32)
                        .ok()?;
                    let a: Arc<dyn grim_tensor::BackendStorage> = Arc::from(k_storage);
                    m.k_full.insert(layer, a.clone());
                    a
                }
            };
            let v_arc = match m.v_full.get(&layer) {
                Some(a) => a.clone(),
                None => {
                    let v_storage = dev
                        .from_cpu(&self.v_pages[layer], &Shape::new(dims.clone()), DType::F32)
                        .ok()?;
                    let a: Arc<dyn grim_tensor::BackendStorage> = Arc::from(v_storage);
                    m.v_full.insert(layer, a.clone());
                    a
                }
            };

            let block_elems = self.page_size * stride;
            for &b in dirty_here.iter() {
                let off = b * block_elems;
                let k_slice: Vec<f32> = self.k_pages[layer][off..off + block_elems].to_vec();
                let v_slice: Vec<f32> = self.v_pages[layer][off..off + block_elems].to_vec();
                let small_shape = Shape::new(vec![self.page_size, stride]);
                if let (Ok(ks), Ok(vs)) = (
                    dev.from_cpu(&k_slice, &small_shape, DType::F32),
                    dev.from_cpu(&v_slice, &small_shape, DType::F32),
                ) {
                    let _ = dev.copy_slice_into(&*k_arc, ks.as_ref(), off, block_elems);
                    let _ = dev.copy_slice_into(&*v_arc, vs.as_ref(), off, block_elems);
                    m.blocks
                        .insert((layer, b), (Arc::from(ks), Arc::from(vs)));
                    m.total_uploads += 1;
                    m.uploaded_elems += (block_elems * 2) as u64;
                }
            }
            for &b in dirty_here.iter() {
                m.dirty.remove(&(layer, b));
            }
            // Tiering-integrated eviction: drop mirror entries whose
            // physical block left the table (freed/demoted upstream).
            let live: std::collections::HashSet<usize> =
                self.table.physical_ids().iter().copied().collect();
            m.blocks.retain(|(l, b), _| *l != layer || live.contains(b));

            Some((
                Tensor::new(
                    k_arc,
                    Shape::new(dims.clone()),
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    dev_enum.clone(),
                ),
                Tensor::new(
                    v_arc,
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
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
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
        PagedKvCache::layer_block_slice(self, layer, block_id)
    }

    fn block_num_tokens(&self, block_id: usize) -> Option<usize> {
        PagedKvCache::block_num_tokens(self, block_id)
    }

    fn write_layer_block(
        &mut self,
        layer: usize,
        block_id: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
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

    /// F3 regression: a partial (failed-layer) disagg fetch must clear the
    /// received bit so the block is re-fetched instead of attended stale.
    #[test]
    fn set_received_clears_partial_fetch_mark() {
        let mut pool = KvBlockPool::new(4, 2, 4);
        let id = pool.alloc().unwrap();
        // write_keys auto-marks received (layer-0 arrival).
        pool.write_keys(id, &vec![1.0f32; BLOCK_SIZE * 2 * 4], BLOCK_SIZE);
        assert!(pool.block_is_received(id));
        // A later layer failed: pull path clears the mark and token count.
        pool.set_received(id, false);
        assert!(!pool.block_is_received(id));
        // Recovery after a complete retry.
        pool.set_received(id, true);
        assert!(pool.block_is_received(id));
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
            let mut pool_g = pool.lock().unwrap_or_else(|e| e.into_inner());
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

    /// Audit remediation: a failed demotion must NOT strand the block in a
    /// fake HostRam state — the block must fall back to the in-place release
    /// (zero + free list) so the slot stays reclaimable. The mismatched
    /// spill manager (wrong block_elems) makes every demote_to_host fail.
    #[test]
    fn failed_demotion_falls_back_to_in_place_release() {
        let dir = tempdir().unwrap();
        // Spill expects 256-elem blocks; the pool's blocks are
        // BLOCK_SIZE * 2 * 4 = 128 elems → every raw demotion errors.
        let spill = Arc::new(
            SharedSpillManager::new(dir.path().to_path_buf(), 256).unwrap(),
        );
        let mut pool = KvBlockPool::new(2, 2, 4);
        pool.attach_spill(spill.clone());
        let id = pool.alloc().unwrap();
        pool.write_keys(id, &vec![1.0f32; BLOCK_SIZE * 2 * 4], BLOCK_SIZE);
        pool.write_values(id, &vec![2.0f32; BLOCK_SIZE * 2 * 4]);
        pool.free(id);
        // Block must be back on the free list at Gpu tier, NOT stranded as
        // a HostRam block with no backing storage.
        assert_eq!(pool.free_list.len(), 2, "failed demotion must return the slot");
        assert_eq!(pool.block_location(id), CacheTier::Gpu);
        assert!(!pool.block_is_received(id));
    }

    /// Audit remediation: promote_to_gpu must ERROR on a spill-geometry
    /// mismatch, never silently truncate the cache via .min().
    #[test]
    fn promote_geometry_mismatch_is_an_error() {
        let dir = tempdir().unwrap();
        // Spill managed with the WRONG geometry (double the pool's elems):
        // demote a full-size raw block through the manager directly, then
        // promote through the pool.
        let spill = Arc::new(
            SharedSpillManager::new(dir.path().to_path_buf(), BLOCK_SIZE * 2 * 8).unwrap(),
        );
        let mut pool = KvBlockPool::new(2, 2, 4);
        pool.attach_spill(spill.clone());
        let id = pool.alloc().unwrap();
        spill
            .demote_to_host(id, vec![1.0; BLOCK_SIZE * 2 * 8], vec![2.0; BLOCK_SIZE * 2 * 8])
            .unwrap();
        let err = pool.promote_to_gpu(id).unwrap_err();
        assert!(
            err.to_string().contains("geometry mismatch"),
            "mismatch must be reported as an error: {err}"
        );
        // And the block contents were NOT partially overwritten.
        assert!(pool.read_keys(id).iter().all(|&v| v == 0.0));
    }

    /// Audit remediation: snapshot_block must compress the ACTUAL token
    /// count, never the padded BLOCK_SIZE shape.
    #[test]
    fn compress_block_uses_actual_num_tokens() {
        let mut pool = KvBlockPool::new(2, 2, 4);
        let compressor: Arc<dyn KvCompressor> =
            Arc::new(LloydMaxCompressor::new(KvQuantConfig::default()));
        pool.attach_compressor(compressor);
        let id = pool.alloc().unwrap();
        // Only 5 of 16 slots written.
        pool.write_keys(id, &vec![0.5f32; 5 * 2 * 4], 5);
        pool.write_values(id, &vec![0.25f32; 5 * 2 * 4]);
        let compressed = pool.compress_block(id).unwrap().unwrap();
        assert_eq!(compressed.num_tokens, 5, "snapshot must carry real tokens, not padding");
        // Empty block compresses to None instead of quantizing zeros.
        let id2 = pool.alloc().unwrap();
        pool.write_keys(id2, &[], 0);
        assert!(pool.compress_block(id2).unwrap().is_none());
    }

    /// Audit remediation: the compression-to-spill loop must be CLOSED —
    /// freeing a block with compressor + spill attached stores the
    /// COMPRESSED bytes in the spill tier, and promote decompresses back to
    /// f32 K/V at the exact original geometry.
    #[test]
    fn compressor_spill_loop_is_closed_end_to_end() {
        let dir = tempdir().unwrap();
        let block_elems = BLOCK_SIZE * 2 * 4;
        let spill =
            Arc::new(SharedSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap());
        let mut pool = KvBlockPool::new(4, 2, 4);
        pool.attach_spill(spill.clone());
        // Constant data compresses/decompresses EXACTLY under both quantizers
        // (scale of a constant is the constant itself), so the round trip is
        // bit-tight and any scale/geometry bug fails the equality below.
        let k = vec![0.5f32; block_elems];
        let v = vec![-0.25f32; block_elems];
        pool.attach_compressor(Arc::new(LloydMaxCompressor::new(KvQuantConfig {
            key_bits: 3,
            value_bits: 4,
            group_size: 64,
            qk_compute_bits: 8,
        })));
        let id = pool.alloc().unwrap();
        pool.write_keys(id, &k, BLOCK_SIZE);
        pool.write_values(id, &v);

        // Free → the spilled bytes are the COMPRESSED blob (not raw f32).
        pool.free(id);
        assert!(spill.get_tier(id).is_some());
        let blob = spill.retrieve_compressed(id).unwrap().expect(
            "spill must hold the compressed blob when a compressor is attached",
        );
        let block = CompressedKvBlock::from_bytes(&blob).unwrap();
        assert_eq!(block.num_tokens, BLOCK_SIZE);
        // The raw f32 tier must NOT hold this block anymore.
        assert_eq!(spill.retrieve(id).unwrap(), None);

        // Promote → decompress back to f32 at the exact original geometry.
        // The dequantized result must be BIT-IDENTICAL to the compressor's
        // own reference dequantization of the same snapshot (the pipeline is
        // deterministic: same seed → same rotation matrix), isolating pool
        // wiring from the quantizer's inherent lossiness.
        let compressor = LloydMaxCompressor::new(KvQuantConfig {
            key_bits: 3,
            value_bits: 4,
            group_size: 64,
            qk_compute_bits: 8,
        });
        let (k_ref, v_ref) = compressor
            .dequantize_for_attention(&block, &CpuDevice::new(), Device::Cpu)
            .unwrap();
        let promoted = pool.promote_to_gpu(id).unwrap().expect("compressed promote");
        assert_eq!(
            promoted.0,
            k_ref.to_vec_f32().unwrap(),
            "promoted keys must equal the reference dequantization"
        );
        assert_eq!(
            promoted.1,
            v_ref.to_vec_f32().unwrap(),
            "promoted values must equal the reference dequantization"
        );
        // And bounded lossiness vs the original: 3-bit keys (rotated) and
        // 4-bit values must reconstruct within the quantizer granularity.
        let max_k_err = promoted
            .0
            .iter()
            .zip(&k)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_v_err = promoted
            .1
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_k_err < 0.5, "key lossiness bounded, got {max_k_err}");
        assert!(max_v_err < 1e-4, "constant values must be exact, got {max_v_err}");
        // The block now holds the DECOMPRESSED-COMPRESSED data: identical to
        // the reference dequantization, lossy vs the original by exactly the
        // quantizer's granularity (keys rotated + 3-bit; constant values
        // exact under asymmetric min/max quantization).
        assert_eq!(pool.read_keys(id), promoted.0.as_slice());
        assert_eq!(pool.read_values(id), promoted.1.as_slice());
        assert_eq!(pool.block_location(id), CacheTier::Gpu);
        assert!(pool.block_is_received(id));
        assert_eq!(pool.block_num_tokens(id), Some(BLOCK_SIZE));
    }

    #[test]
    fn test_dirty_block_mirror_tracking_and_flushing() {
        let mut pool = KvBlockPool::new(8, 2, 4);
        assert_eq!(pool.dirty_count(), 0);

        let b0 = pool.alloc().unwrap();
        let b1 = pool.alloc().unwrap();
        assert!(!pool.is_dirty(b0));
        assert!(!pool.is_dirty(b1));

        // Writing keys marks block as dirty
        let block_elems = BLOCK_SIZE * 2 * 4;
        pool.write_keys(b0, &vec![1.0f32; block_elems], BLOCK_SIZE);
        assert!(pool.is_dirty(b0));
        assert_eq!(pool.dirty_count(), 1);

        pool.write_values(b1, &vec![2.0f32; block_elems]);
        assert!(pool.is_dirty(b1));
        assert_eq!(pool.dirty_count(), 2);

        // Flushing dirty blocks synchronizes and clears dirty set
        let flushed = pool.flush_dirty_to_host().unwrap();
        assert_eq!(flushed, 2);
        assert_eq!(pool.dirty_count(), 0);
        assert!(!pool.is_dirty(b0));
        assert!(!pool.is_dirty(b1));
    }

    #[test]
    fn test_per_stage_kv_block_pool_on_device() {
        let pool = KvBlockPool::new_on_device(16, 4, 64, 1).with_layer_range(8, 16);
        assert_eq!(pool.device_ordinal(), 1);
        assert_eq!(pool.layer_range(), Some((8, 16)));
        assert!(!pool.owns_layer(7));
        assert!(pool.owns_layer(8));
        assert!(pool.owns_layer(15));
        assert!(!pool.owns_layer(16));
    }
}

#[cfg(test)]
mod f10_mirror_tests {
    use super::*;
    use grim_backend_cpu::CpuDevice;

    /// F10 ITL gate (deterministic proxy): with the dirty-block mirror, a
    /// decode/prefill sequence uploads each physical block exactly once plus
    /// refreshes limited to the ACTIVE tail block — never sealed history, and
    /// never the full layer per token. The hardware ITL measurement itself
    /// requires the gfx1036 runner (see plan.md F10c).
    #[test]
    fn itl_gate_dirty_block_mirror_uploads_each_block_once() {
        let pool = Arc::new(std::sync::Mutex::new(KvBlockPool::new(4, 2, 4)));
        let mut kv = PagedKvCache::new(pool, 2, 4, BLOCK_SIZE);
        kv.set_device(Device::Cpu, Arc::new(CpuDevice::new()));

        let stride = 2 * 4; // num_kv_heads * head_dim
        let append = |kv: &mut PagedKvCache, n: usize, tag: f32| {
            let data: Vec<f32> = (0..n * stride).map(|i| tag + i as f32 * 0.01).collect();
            let k = grim_backend_cpu::cpu_tensor(data.clone(), Shape::new(vec![n, stride]));
            let v = grim_backend_cpu::cpu_tensor(data, Shape::new(vec![n, stride]));
            kv.append_kv_layer(0, &k, &v).unwrap();
        };

        // Step 1: fill block 0 (16 tokens) → 1 upload.
        append(&mut kv, 16, 1.0);
        kv.paged_kv_handles(0).unwrap();
        let (up0, uniq0, _, _) = kv.mirror_stats();
        assert_eq!((up0, uniq0), (1, 1), "block 0 uploads exactly once");

        // Step 2: 4 tokens into block 1 → 1 upload (block 0 NOT re-uploaded).
        append(&mut kv, 4, 2.0);
        kv.paged_kv_handles(0).unwrap();
        let (up1, uniq1, _, _) = kv.mirror_stats();
        assert_eq!(
            (up1, uniq1),
            (2, 2),
            "block 1 uploads once; block 0 untouched"
        );

        // Steps 3-6: single-token decode appends into tail block 1 → only the
        // tail refreshes; sealed block 0 never re-uploads.
        for t in 0..4 {
            append(&mut kv, 1, 3.0 + t as f32);
            kv.paged_kv_handles(0).unwrap();
        }
        let (up, uniq, elems, dirty) = kv.mirror_stats();
        assert_eq!(uniq, 2, "only two distinct blocks exist");
        assert_eq!(up, 6, "2 initial + 4 tail refreshes");
        assert_eq!(dirty, 0, "no dirty blocks survive a staging call");

        // ITL gate: naive per-append FULL-LAYER staging would have uploaded
        // 25 appends × capacity(4 blocks × 16 tokens) rows; the mirror moved
        // 6 block uploads total — a ≥10× traffic reduction at this shape.
        let naive_uploads = 25u64 * 4;
        assert!(
            up * 4 < naive_uploads,
            "mirror uploads must beat naive full-layer staging (up={up})"
        );
        let _ = elems;
    }
}
