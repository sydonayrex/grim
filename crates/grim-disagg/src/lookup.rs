//! Distributed lookup client with Bloom filter pre-filtering (LMCache).
//!
//! Queries local bloom filter summaries of remote peer KV cache tables before
//! issuing network requests, avoiding expensive TCP round-trips for prefix misses.

use crate::bloom::BloomFilter;
use grim_kvtransport::NetworkKvClient;

/// Client for querying remote disaggregated KV nodes with bloom pre-filtering.
pub struct LookupClient {
    pub remote_bloom: BloomFilter,
    pub peer_addr: String,
    pub inner_client: NetworkKvClient,
}

impl LookupClient {
    /// Create a new lookup client with a remote node's bloom filter snapshot.
    pub fn new(remote_bloom: BloomFilter, peer_addr: String) -> Self {
        Self {
            remote_bloom,
            inner_client: NetworkKvClient::new(peer_addr.clone()),
            peer_addr,
        }
    }

    /// Serialize token IDs into a binary key.
    fn prefix_key(prefix: &[u32]) -> Vec<u8> {
        let mut key = Vec::with_capacity(prefix.len() * 4);
        for &t in prefix {
            key.extend_from_slice(&t.to_le_bytes());
        }
        key
    }

    /// Check if the remote node might contain the given token prefix.
    pub fn might_have_prefix(&self, prefix: &[u32]) -> bool {
        let key = Self::prefix_key(prefix);
        self.remote_bloom.might_contain(&key)
    }

    /// Update the local bloom filter summary from a peer synchronization payload.
    pub fn update_bloom(&mut self, new_bloom: BloomFilter) {
        self.remote_bloom = new_bloom;
    }
}
