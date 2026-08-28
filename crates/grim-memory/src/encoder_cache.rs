//! Encoder Cache (EC) for multimodal vision and audio embedding deduplication.
//!
//! Repurposed from LMCache's `ECCacheEngine`.
//! Caches dense output embeddings from vision (ViT/CLIP/SigLIP) and audio encoders
//! keyed by content hashes, bypassing heavy encoder forward passes on repeated
//! image tokens, multi-turn visual conversations, and video streams.

use grim_core::error::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Unique identifier for an encoder output embedding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncoderKey {
    /// 64-bit content hash of raw media (e.g. image bytes / spectrogram)
    pub media_hash: u64,
    /// Modality identifier (e.g. "vision", "audio", "video")
    pub modality: String,
}

/// Cached dense representation from a multimodal encoder.
#[derive(Debug, Clone)]
pub struct EncoderEmbedding {
    pub key: EncoderKey,
    pub num_tokens: usize,
    pub hidden_dim: usize,
    /// Raw f32 embedding tensor [num_tokens, hidden_dim]
    pub data: Vec<f32>,
}

/// LRU-managed cache for heavy multimodal encoder outputs.
#[derive(Debug)]
pub struct EncoderCache {
    /// Maximum capacity in total f32 elements
    max_elements: usize,
    /// Currently allocated f32 elements
    current_elements: usize,
    /// Indexed embeddings
    entries: HashMap<EncoderKey, EncoderEmbedding>,
    /// LRU eviction sequence
    lru_queue: VecDeque<EncoderKey>,
}

impl EncoderCache {
    /// Create a new encoder cache with an element budget.
    pub fn new(max_elements: usize) -> Self {
        Self {
            max_elements,
            current_elements: 0,
            entries: HashMap::new(),
            lru_queue: VecDeque::new(),
        }
    }

    /// Insert or update an encoder embedding.
    pub fn insert(
        &mut self,
        media_hash: u64,
        modality: &str,
        num_tokens: usize,
        hidden_dim: usize,
        data: Vec<f32>,
    ) -> Result<()> {
        let req_elements = data.len();
        if req_elements > self.max_elements {
            return Err(Error::KvCache(format!(
                "Encoder embedding size {} exceeds max capacity {}",
                req_elements, self.max_elements
            )));
        }

        let key = EncoderKey {
            media_hash,
            modality: modality.to_string(),
        };

        if let Some(old) = self.entries.get(&key) {
            self.current_elements -= old.data.len();
        }

        while self.current_elements + req_elements > self.max_elements {
            if let Some(evict_key) = self.lru_queue.pop_front() {
                if let Some(removed) = self.entries.remove(&evict_key) {
                    self.current_elements =
                        self.current_elements.saturating_sub(removed.data.len());
                }
            } else {
                break;
            }
        }

        self.current_elements += req_elements;
        self.entries.insert(
            key.clone(),
            EncoderEmbedding {
                key: key.clone(),
                num_tokens,
                hidden_dim,
                data,
            },
        );
        self.touch_lru(key);
        Ok(())
    }

    /// Lookup cached encoder embedding by media hash and modality.
    pub fn get(&mut self, media_hash: u64, modality: &str) -> Option<&EncoderEmbedding> {
        let key = EncoderKey {
            media_hash,
            modality: modality.to_string(),
        };
        if self.entries.contains_key(&key) {
            self.touch_lru(key.clone());
            self.entries.get(&key)
        } else {
            None
        }
    }

    fn touch_lru(&mut self, key: EncoderKey) {
        if let Some(pos) = self.lru_queue.iter().position(|k| k == &key) {
            self.lru_queue.remove(pos);
        }
        self.lru_queue.push_back(key);
    }

    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn allocated_elements(&self) -> usize {
        self.current_elements
    }
}

/// Shared thread-safe EncoderCache.
#[derive(Debug, Clone)]
pub struct SharedEncoderCache(pub Arc<Mutex<EncoderCache>>);

impl SharedEncoderCache {
    pub fn new(max_elements: usize) -> Self {
        Self(Arc::new(Mutex::new(EncoderCache::new(max_elements))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_cache_insert_and_lru() {
        let mut cache = EncoderCache::new(256);
        let emb1 = vec![1.0; 128];
        let emb2 = vec![2.0; 128];
        let emb3 = vec![3.0; 128];

        cache.insert(1001, "vision", 32, 4, emb1).unwrap();
        cache.insert(1002, "vision", 32, 4, emb2).unwrap();
        assert_eq!(cache.num_entries(), 2);

        // Inserting 3rd must evict 1st
        cache.insert(1003, "vision", 32, 4, emb3).unwrap();
        assert_eq!(cache.num_entries(), 2);
        assert!(cache.get(1001, "vision").is_none());
        assert!(cache.get(1002, "vision").is_some());
        assert!(cache.get(1003, "vision").is_some());
    }
}
