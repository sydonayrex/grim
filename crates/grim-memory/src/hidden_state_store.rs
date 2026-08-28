//! Standalone chunk-aligned hidden-state cache on pinned host memory pool.
//!
//! Caches intermediate layer hidden states tied to KV cache chunk hashes,
//! allowing instant priming of speculative draft models (e.g. Eagle-3, MTP)
//! without recomputing bottom transformer layers.

use grim_core::error::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Unique identifier for a hidden-state chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HiddenStateKey {
    /// Token sequence hash of the chunk
    pub chunk_hash: u64,
    /// Model layer index whose hidden state is captured
    pub layer_idx: usize,
}

/// A contiguous chunk of intermediate hidden states.
#[derive(Debug, Clone)]
pub struct HiddenStateChunk {
    pub key: HiddenStateKey,
    pub token_count: usize,
    pub hidden_dim: usize,
    /// Raw f32 hidden states: [token_count, hidden_dim]
    pub data: Vec<f32>,
}

/// Manages host-pinned memory allocations and LRU caching for intermediate hidden states.
#[derive(Debug)]
pub struct HiddenStateStore {
    /// Maximum capacity in total f32 elements allowed in the store
    max_elements: usize,
    /// Currently allocated f32 elements
    current_elements: usize,
    /// Chunks indexed by key
    chunks: HashMap<HiddenStateKey, HiddenStateChunk>,
    /// LRU eviction queue of keys
    lru_queue: VecDeque<HiddenStateKey>,
    /// Set of known valid KV chunk hashes for lazy coupled eviction
    valid_kv_chunks: HashMap<u64, usize>,
}

impl HiddenStateStore {
    /// Create a new store with a given element capacity limit.
    /// (e.g. 512MB capacity for hidden_dim 4096 is ~134M f32s).
    pub fn new(max_elements: usize) -> Self {
        Self {
            max_elements,
            current_elements: 0,
            chunks: HashMap::new(),
            lru_queue: VecDeque::new(),
            valid_kv_chunks: HashMap::new(),
        }
    }

    /// Notify store that a KV chunk hash exists or is retained.
    pub fn retain_kv_chunk(&mut self, chunk_hash: u64) {
        let count = self.valid_kv_chunks.entry(chunk_hash).or_insert(0);
        *count += 1;
    }

    /// Notify store that a KV chunk hash was evicted/released.
    /// Lazy coupled eviction: immediately drops any hidden states tied to this chunk.
    pub fn evict_kv_chunk(&mut self, chunk_hash: u64) {
        if let Some(count) = self.valid_kv_chunks.get_mut(&chunk_hash) {
            if *count > 1 {
                *count -= 1;
                return;
            }
        }
        self.valid_kv_chunks.remove(&chunk_hash);

        // Drop all layer hidden states associated with this chunk hash
        let keys_to_remove: Vec<HiddenStateKey> = self
            .chunks
            .keys()
            .filter(|k| k.chunk_hash == chunk_hash)
            .cloned()
            .collect();

        for k in keys_to_remove {
            self.remove_key(&k);
        }
    }

    /// Insert or update a hidden state chunk.
    pub fn insert_chunk(
        &mut self,
        chunk_hash: u64,
        layer_idx: usize,
        token_count: usize,
        hidden_dim: usize,
        data: Vec<f32>,
    ) -> Result<()> {
        let req_elements = data.len();
        if req_elements > self.max_elements {
            return Err(Error::KvCache(format!(
                "Hidden state chunk size {} exceeds maximum capacity {}",
                req_elements, self.max_elements
            )));
        }

        let key = HiddenStateKey {
            chunk_hash,
            layer_idx,
        };

        // If replacing existing entry, subtract old size
        if let Some(old) = self.chunks.get(&key) {
            self.current_elements -= old.data.len();
        }

        // Evict LRU entries until enough space is available
        while self.current_elements + req_elements > self.max_elements {
            if let Some(evict_key) = self.lru_queue.pop_front() {
                self.remove_key(&evict_key);
            } else {
                break;
            }
        }

        self.current_elements += req_elements;
        self.chunks.insert(
            key,
            HiddenStateChunk {
                key,
                token_count,
                hidden_dim,
                data,
            },
        );
        self.touch_lru(key);
        Ok(())
    }

    /// Retrieve a hidden state chunk.
    pub fn get_chunk(&mut self, chunk_hash: u64, layer_idx: usize) -> Option<&HiddenStateChunk> {
        let key = HiddenStateKey {
            chunk_hash,
            layer_idx,
        };
        if self.chunks.contains_key(&key) {
            self.touch_lru(key);
            self.chunks.get(&key)
        } else {
            None
        }
    }

    /// Touch LRU order.
    fn touch_lru(&mut self, key: HiddenStateKey) {
        if let Some(pos) = self.lru_queue.iter().position(|&k| k == key) {
            self.lru_queue.remove(pos);
        }
        self.lru_queue.push_back(key);
    }

    /// Remove a key and reclaim memory.
    fn remove_key(&mut self, key: &HiddenStateKey) {
        if let Some(chunk) = self.chunks.remove(key) {
            self.current_elements = self.current_elements.saturating_sub(chunk.data.len());
        }
        if let Some(pos) = self.lru_queue.iter().position(|k| k == key) {
            self.lru_queue.remove(pos);
        }
    }

    /// Current capacity utilization.
    pub fn allocated_elements(&self) -> usize {
        self.current_elements
    }

    /// Total chunks stored.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }
}

/// Thread-safe shared wrapper for HiddenStateStore.
#[derive(Debug, Clone)]
pub struct SharedHiddenStateStore(pub Arc<Mutex<HiddenStateStore>>);

impl SharedHiddenStateStore {
    pub fn new(max_elements: usize) -> Self {
        Self(Arc::new(Mutex::new(HiddenStateStore::new(max_elements))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hidden_state_store_insert_and_retrieve() {
        let mut store = HiddenStateStore::new(1000);
        let data = vec![0.5f32; 128];
        store
            .insert_chunk(42, 12, 16, 8, data.clone())
            .expect("insert");

        assert_eq!(store.allocated_elements(), 128);
        assert_eq!(store.num_chunks(), 1);

        let chunk = store.get_chunk(42, 12).expect("retrieve");
        assert_eq!(chunk.token_count, 16);
        assert_eq!(chunk.hidden_dim, 8);
        assert_eq!(chunk.data[0], 0.5f32);
    }

    #[test]
    fn test_lazy_coupled_eviction_with_kv() {
        let mut store = HiddenStateStore::new(1000);
        store.retain_kv_chunk(101);
        store.insert_chunk(101, 1, 16, 8, vec![1.0; 128]).unwrap();
        store.insert_chunk(101, 2, 16, 8, vec![2.0; 128]).unwrap();

        assert_eq!(store.num_chunks(), 2);
        assert_eq!(store.allocated_elements(), 256);

        // Evict KV chunk -> should automatically drop both layer hidden states
        store.evict_kv_chunk(101);
        assert_eq!(store.num_chunks(), 0);
        assert_eq!(store.allocated_elements(), 0);
        assert!(store.get_chunk(101, 1).is_none());
    }

    #[test]
    fn test_lru_capacity_eviction() {
        let mut store = HiddenStateStore::new(200);
        store.insert_chunk(1, 0, 10, 10, vec![1.0; 100]).unwrap();
        store.insert_chunk(2, 0, 10, 10, vec![2.0; 100]).unwrap();

        assert_eq!(store.num_chunks(), 2);
        assert_eq!(store.allocated_elements(), 200);

        // Exceeds limit: inserting chunk 3 must evict chunk 1 (least recently used)
        store.insert_chunk(3, 0, 10, 10, vec![3.0; 100]).unwrap();
        assert_eq!(store.num_chunks(), 2);
        assert!(store.get_chunk(1, 0).is_none());
        assert!(store.get_chunk(2, 0).is_some());
        assert!(store.get_chunk(3, 0).is_some());
    }
}
