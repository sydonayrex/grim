//! Multi-node distributed cache coherence protocol (LMCache / FreeToken).
//!
//! Synchronizes cache invalidation events across cluster nodes so that when
//! a prefix or KV block is modified or evicted on one node, peer nodes drop
//! their stale cached copies before subsequent accesses.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use grim_core::error::{Error, Result};

/// Compact binary invalidation message for cross-node broadcast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidationMsg {
    /// 64-bit cryptographic/FNV hash of the invalidated token sequence.
    pub prefix_hash: u64,
    /// Origin cluster node ID.
    pub origin_node: u32,
    /// UNIX timestamp in seconds.
    pub timestamp: u64,
}

impl InvalidationMsg {
    /// Encodes the message into exactly 20 bytes: `[prefix_hash (8)][origin_node (4)][timestamp (8)]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        buf.extend_from_slice(&self.prefix_hash.to_le_bytes());
        buf.extend_from_slice(&self.origin_node.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf
    }

    /// Decodes a 20-byte binary invalidation message.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() != 20 {
            return Err(Error::Config(format!(
                "InvalidationMsg::decode: expected 20 bytes, got {}",
                buf.len()
            )));
        }

        let prefix_hash = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let origin_node = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let timestamp = u64::from_le_bytes(buf[12..20].try_into().unwrap());

        Ok(Self {
            prefix_hash,
            origin_node,
            timestamp,
        })
    }
}

/// Cache coherence manager maintaining local prefix bindings and processing peer invalidations.
pub struct CacheCoherenceManager {
    pub node_id: u32,
    pub local_prefixes: HashMap<u64, usize>,
}

impl CacheCoherenceManager {
    /// Constructs a new standalone coherence manager.
    pub fn new_standalone(node_id: u32) -> Self {
        Self {
            node_id,
            local_prefixes: HashMap::new(),
        }
    }

    /// Computes stable 64-bit hash for a token sequence.
    pub fn hash_prefix(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Insert a token prefix into the local cache index.
    pub fn insert_prefix(&mut self, tokens: &[u32], block_id: usize) {
        let hash = Self::hash_prefix(tokens);
        self.local_prefixes.insert(hash, block_id);
    }

    /// Look up a token prefix in the local cache index.
    pub fn lookup_prefix(&self, tokens: &[u32]) -> Option<usize> {
        let hash = Self::hash_prefix(tokens);
        self.local_prefixes.get(&hash).copied()
    }

    /// Invalidate a token prefix locally and construct an `InvalidationMsg` for broadcast.
    pub fn invalidate_prefix(&mut self, tokens: &[u32]) -> InvalidationMsg {
        let hash = Self::hash_prefix(tokens);
        self.local_prefixes.remove(&hash);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        InvalidationMsg {
            prefix_hash: hash,
            origin_node: self.node_id,
            timestamp: now,
        }
    }

    /// Process an incoming invalidation message from a peer node.
    pub fn handle_invalidation(&mut self, msg: &InvalidationMsg) {
        self.local_prefixes.remove(&msg.prefix_hash);
    }
}
