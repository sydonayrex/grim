//! Conversation checkpoint cache at message boundaries (astrum P10).
//!
//! Caches recurrent-state checkpoints (and pinned radix KV block references) at
//! previous turn boundaries so an edited-last-message chat can reuse the long
//! prefix. CPU-manageable cache logic; engine wiring restores on prefix hit.
//! Gated by `GRIM_ASTRUM_CONVERSATION_CKPT=1` at call sites. Default OFF.

use crate::BlockId;

/// Recurrent-state snapshot at a conversation prefix boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationCheckpoint {
    /// Prefix token ids at the boundary (hashed/compared for prefix match).
    pub key: Vec<u32>,
    /// Serialized recurrent state bytes (engine/hybrid-model specific).
    pub state_bytes: Vec<u8>,
    /// Radix-shared KV block ids referenced by this prefix (caller may pin via pool).
    pub kv_block_ids: Vec<BlockId>,
}

/// Restored checkpoint data returned on a prefix hit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredState {
    pub key: Vec<u32>,
    pub state_bytes: Vec<u8>,
    pub kv_block_ids: Vec<BlockId>,
}

/// LRU cache of conversation checkpoints, capped (default 4).
#[derive(Debug)]
pub struct ConversationCheckpointCache {
    cap: usize,
    // most-recently-used at back
    entries: Vec<ConversationCheckpoint>,
}

impl Default for ConversationCheckpointCache {
    fn default() -> Self {
        Self::new(4)
    }
}

impl ConversationCheckpointCache {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            entries: Vec::new(),
        }
    }

    /// Save a checkpoint at `key`. Evicts oldest when over capacity.
    pub fn save(&mut self, key: Vec<u32>, state_bytes: Vec<u8>, kv_block_ids: Vec<BlockId>) {
        // remove existing exact-key entry to refresh LRU position
        self.entries.retain(|e| e.key != key);
        self.entries.push(ConversationCheckpoint {
            key,
            state_bytes,
            kv_block_ids,
        });
        while self.entries.len() > self.cap {
            self.entries.remove(0); // evict oldest
        }
    }

    /// Restore if any cached key is a prefix of `tokens` (or exact match).
    /// Returns the longest matching prefix checkpoint and marks it MRU.
    pub fn restore_if_prefix_matches(&mut self, tokens: &[u32]) -> Option<RestoredState> {
        // find longest matching key
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, e) in self.entries.iter().enumerate() {
            if !e.key.is_empty() && tokens.starts_with(&e.key) && e.key.len() >= best_len {
                best = Some(i);
                best_len = e.key.len();
            }
        }
        let idx = best?;
        let ckpt = self.entries.remove(idx);
        let restored = RestoredState {
            key: ckpt.key.clone(),
            state_bytes: ckpt.state_bytes.clone(),
            kv_block_ids: ckpt.kv_block_ids.clone(),
        };
        // mark MRU
        self.entries.push(ckpt);
        Some(restored)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub fn conversation_ckpt_enabled() -> bool {
    std::env::var("GRIM_ASTRUM_CONVERSATION_CKPT")
        .ok()
        .as_deref()
        == Some("1")
}
