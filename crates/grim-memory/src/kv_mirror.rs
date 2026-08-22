//! Paged KV cache device-resident mirror and asynchronous tiering coordinator (F10).
//!
//! Tracks dual-resident KV blocks between Device HBM and Host RAM backing mirrors.
//! Enables asynchronous non-blocking writeback, high-watermark spill staging, and
//! instantaneous promotion without stalling GPU decode kernel execution.

use std::collections::{HashMap, HashSet};

use grim_core::error::Result;

use crate::{BlockId, KvBlockPool};

/// Synchronization lifecycle state of a physical block within the device mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorSyncState {
    /// Data is byte-identical across Device HBM and Host RAM mirror.
    InSync,
    /// Data has been written on Device HBM and needs asynchronous flush to Host mirror.
    DirtyOnDevice,
    /// Asynchronous transfer from Device to Host mirror is currently in-flight.
    Flushing,
    /// Block has been evicted from Device HBM; primary data resides exclusively in Host mirror.
    EvictedToHost,
    /// Block is unallocated or initialized to zeros.
    Clean,
}

/// Operational policy configuration for the device-resident KV mirror.
#[derive(Debug, Clone)]
pub struct KvDeviceMirrorConfig {
    /// Memory pressure ratio (0.0 to 1.0) above which dirty blocks are eagerly flushed to host.
    pub high_watermark_ratio: f32,
    /// Memory pressure ratio below which eagerly flushed blocks may remain unevicted on GPU.
    pub low_watermark_ratio: f32,
    /// Maximum number of blocks to transfer concurrently in a single flush pass.
    pub max_batch_flush_size: usize,
    /// Whether to enable proactive non-blocking background writeback.
    pub enable_proactive_writeback: bool,
}

impl Default for KvDeviceMirrorConfig {
    fn default() -> Self {
        Self {
            high_watermark_ratio: 0.85,
            low_watermark_ratio: 0.70,
            max_batch_flush_size: 64,
            enable_proactive_writeback: true,
        }
    }
}

/// Dual-tier device-resident KV mirror managing device-to-host memory replication.
#[derive(Debug)]
pub struct KvDeviceMirror {
    config: KvDeviceMirrorConfig,
    /// Current sync state per physical block ID.
    states: Vec<MirrorSyncState>,
    /// Host RAM backing mirror storing synchronized key/value float buffers.
    host_mirror: HashMap<BlockId, (Vec<f32>, Vec<f32>)>,
    /// Set of block IDs that require synchronization to host.
    dirty_set: HashSet<BlockId>,
    /// Capacity of the block pool.
    capacity: usize,
    /// Number of elements per token (num_heads * head_dim).
    elem_per_token: usize,
}

impl KvDeviceMirror {
    /// Create a new device-resident KV mirror for a pool of given capacity and geometry.
    pub fn new(
        capacity: usize,
        num_heads: usize,
        head_dim: usize,
        config: KvDeviceMirrorConfig,
    ) -> Self {
        Self {
            config,
            states: vec![MirrorSyncState::Clean; capacity],
            host_mirror: HashMap::with_capacity(capacity / 2),
            dirty_set: HashSet::new(),
            capacity,
            elem_per_token: num_heads * head_dim,
        }
    }

    /// Capacity in physical blocks.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of elements per token.
    pub fn elem_per_token(&self) -> usize {
        self.elem_per_token
    }

    /// Retrieve sync state for a block.
    pub fn state(&self, id: BlockId) -> MirrorSyncState {
        if id < self.states.len() {
            self.states[id]
        } else {
            MirrorSyncState::Clean
        }
    }

    /// Mark a block as modified on Device HBM.
    pub fn mark_dirty(&mut self, id: BlockId) {
        if id < self.states.len() {
            self.states[id] = MirrorSyncState::DirtyOnDevice;
            self.dirty_set.insert(id);
        }
    }

    /// Check if a block has pending unsynchronized writes.
    pub fn is_dirty(&self, id: BlockId) -> bool {
        self.dirty_set.contains(&id)
    }

    /// Number of dirty blocks awaiting synchronization.
    pub fn dirty_count(&self) -> usize {
        self.dirty_set.len()
    }

    /// Number of blocks cached in the host backing mirror.
    pub fn host_mirror_count(&self) -> usize {
        self.host_mirror.len()
    }

    /// Stage a block for asynchronous writeback.
    pub fn stage_flush(&mut self, id: BlockId) -> Option<MirrorSyncState> {
        if id < self.states.len() && self.states[id] == MirrorSyncState::DirtyOnDevice {
            self.states[id] = MirrorSyncState::Flushing;
            Some(MirrorSyncState::Flushing)
        } else {
            None
        }
    }

    /// Commit synchronized key and value buffers into the host mirror and mark InSync.
    pub fn commit_flush(&mut self, id: BlockId, keys: Vec<f32>, values: Vec<f32>) {
        if id < self.states.len() {
            self.host_mirror.insert(id, (keys, values));
            self.states[id] = MirrorSyncState::InSync;
            self.dirty_set.remove(&id);
        }
    }

    /// Evict a block from GPU residency; data is retained in the host backing mirror.
    pub fn mark_evicted(&mut self, id: BlockId) -> bool {
        if id < self.states.len() && self.host_mirror.contains_key(&id) {
            self.states[id] = MirrorSyncState::EvictedToHost;
            self.dirty_set.remove(&id);
            true
        } else {
            false
        }
    }

    /// Reclaim a block from the host backing mirror back into GPU residency.
    pub fn retrieve_mirror(&self, id: BlockId) -> Option<&(Vec<f32>, Vec<f32>)> {
        self.host_mirror.get(&id)
    }

    /// Remove a block from mirror tracking entirely when freed from the pool.
    pub fn release_block(&mut self, id: BlockId) {
        if id < self.states.len() {
            self.states[id] = MirrorSyncState::Clean;
            self.dirty_set.remove(&id);
            self.host_mirror.remove(&id);
        }
    }

    /// Evaluates memory pressure watermark and returns a list of candidate blocks to evict.
    pub fn evaluate_watermark_eviction_candidates(
        &self,
        allocated_blocks: usize,
        total_blocks: usize,
    ) -> Vec<BlockId> {
        if total_blocks == 0 {
            return Vec::new();
        }
        let pressure = (allocated_blocks as f32) / (total_blocks as f32);
        if pressure < self.config.high_watermark_ratio {
            return Vec::new();
        }

        let target_freed =
            ((pressure - self.config.low_watermark_ratio) * (total_blocks as f32)).ceil() as usize;

        // Prioritize blocks that are already InSync with the host mirror (zero transfer cost to evict)
        let mut candidates = Vec::with_capacity(target_freed);
        for id in 0..self.states.len() {
            if self.states[id] == MirrorSyncState::InSync {
                candidates.push(id);
                if candidates.len() >= target_freed {
                    break;
                }
            }
        }
        candidates
    }

    /// Synchronize all pending dirty blocks from a block pool into this host mirror.
    pub fn sync_dirty_from_pool(&mut self, pool: &mut KvBlockPool) -> Result<usize> {
        let dirty_ids: Vec<BlockId> = self.dirty_set.iter().copied().collect();
        let mut synced = 0;

        for id in dirty_ids {
            if id < pool.num_blocks() {
                let keys = pool.read_keys(id).to_vec();
                let values = pool.read_values(id).to_vec();
                self.commit_flush(id, keys, values);
                synced += 1;
            }
        }

        Ok(synced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BLOCK_SIZE;

    #[test]
    fn test_kv_device_mirror_lifecycle() {
        let mut mirror = KvDeviceMirror::new(16, 2, 4, KvDeviceMirrorConfig::default());
        assert_eq!(mirror.state(0), MirrorSyncState::Clean);
        assert_eq!(mirror.dirty_count(), 0);

        // Mark dirty
        mirror.mark_dirty(0);
        assert_eq!(mirror.state(0), MirrorSyncState::DirtyOnDevice);
        assert!(mirror.is_dirty(0));
        assert_eq!(mirror.dirty_count(), 1);

        // Stage flush
        let staged = mirror.stage_flush(0);
        assert_eq!(staged, Some(MirrorSyncState::Flushing));
        assert_eq!(mirror.state(0), MirrorSyncState::Flushing);

        // Commit flush
        let block_elems = BLOCK_SIZE * 2 * 4;
        mirror.commit_flush(0, vec![1.0; block_elems], vec![2.0; block_elems]);
        assert_eq!(mirror.state(0), MirrorSyncState::InSync);
        assert!(!mirror.is_dirty(0));
        assert_eq!(mirror.dirty_count(), 0);
        assert_eq!(mirror.host_mirror_count(), 1);

        // Evict to host
        assert!(mirror.mark_evicted(0));
        assert_eq!(mirror.state(0), MirrorSyncState::EvictedToHost);

        // Retrieve from mirror
        let (k, v) = mirror.retrieve_mirror(0).unwrap();
        assert_eq!(k[0], 1.0);
        assert_eq!(v[0], 2.0);

        // Release
        mirror.release_block(0);
        assert_eq!(mirror.state(0), MirrorSyncState::Clean);
        assert_eq!(mirror.host_mirror_count(), 0);
    }

    #[test]
    fn test_watermark_candidate_selection() {
        let mut mirror = KvDeviceMirror::new(
            100,
            2,
            4,
            KvDeviceMirrorConfig {
                high_watermark_ratio: 0.80,
                low_watermark_ratio: 0.60,
                max_batch_flush_size: 32,
                enable_proactive_writeback: true,
            },
        );

        // Under watermark: 50/100 -> 0 candidates
        let cand = mirror.evaluate_watermark_eviction_candidates(50, 100);
        assert!(cand.is_empty());

        // Above watermark: 90/100 -> need 30 freed blocks
        for id in 0..40 {
            mirror.commit_flush(id, vec![0.0; 128], vec![0.0; 128]);
        }
        let cand = mirror.evaluate_watermark_eviction_candidates(90, 100);
        assert_eq!(cand.len(), 30);
        assert_eq!(cand[0], 0);
        assert_eq!(cand[29], 29);
    }
}
