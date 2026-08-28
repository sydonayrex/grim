//! High-performance bitmap-backed chunk presence index for multi-tier cache lookups.
//!
//! Tracks cache block presence across L1 (VRAM/GPU), L2 (Host RAM), and L3 (NVMe / Remote Storage)
//! using compact bitmasks and hash tables for O(1) admission checks.

use crate::CacheTier;
use std::collections::HashMap;

/// Bit flags representing presence across cache tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TierMask(pub u8);

impl TierMask {
    pub const NONE: u8 = 0;
    pub const GPU: u8 = 1 << 0;
    pub const HOST: u8 = 1 << 1;
    pub const NVME: u8 = 1 << 2;
    pub const REMOTE: u8 = 1 << 3;

    #[inline]
    pub fn new() -> Self {
        Self(Self::NONE)
    }

    #[inline]
    pub fn with_tier(mut self, tier: CacheTier) -> Self {
        self.set_tier(tier);
        self
    }

    #[inline]
    pub fn set_tier(&mut self, tier: CacheTier) {
        match tier {
            CacheTier::Gpu => self.0 |= Self::GPU,
            CacheTier::HostRam => self.0 |= Self::HOST,
            CacheTier::NvMe | CacheTier::NvMeWeightStream => self.0 |= Self::NVME,
        }
    }

    #[inline]
    pub fn clear_tier(&mut self, tier: CacheTier) {
        match tier {
            CacheTier::Gpu => self.0 &= !Self::GPU,
            CacheTier::HostRam => self.0 &= !Self::HOST,
            CacheTier::NvMe | CacheTier::NvMeWeightStream => self.0 &= !Self::NVME,
        }
    }

    #[inline]
    pub fn has_tier(&self, tier: CacheTier) -> bool {
        match tier {
            CacheTier::Gpu => (self.0 & Self::GPU) != 0,
            CacheTier::HostRam => (self.0 & Self::HOST) != 0,
            CacheTier::NvMe | CacheTier::NvMeWeightStream => (self.0 & Self::NVME) != 0,
        }
    }

    #[inline]
    pub fn highest_tier(&self) -> Option<CacheTier> {
        if self.has_tier(CacheTier::Gpu) {
            Some(CacheTier::Gpu)
        } else if self.has_tier(CacheTier::HostRam) {
            Some(CacheTier::HostRam)
        } else if self.has_tier(CacheTier::NvMe) {
            Some(CacheTier::NvMe)
        } else {
            None
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == Self::NONE
    }
}

/// Chunk metadata containing token sequence hash and block association.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    pub chunk_hash: u64,
    pub block_id: usize,
    pub tier_mask: TierMask,
    pub token_count: usize,
}

/// Multi-tier bitmask chunk presence index.
#[derive(Debug, Default)]
pub struct BitmaskChunkIndex {
    /// Maps chunk hash -> chunk metadata
    entries: HashMap<u64, ChunkEntry>,
    /// Maps block_id -> chunk hash for reverse lookup
    block_to_hash: HashMap<usize, u64>,
}

impl BitmaskChunkIndex {
    /// Create a new empty bitmask chunk index.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            block_to_hash: HashMap::new(),
        }
    }

    /// Record a chunk's presence in a specific cache tier.
    pub fn record_chunk(
        &mut self,
        chunk_hash: u64,
        block_id: usize,
        token_count: usize,
        tier: CacheTier,
    ) {
        let entry = self
            .entries
            .entry(chunk_hash)
            .or_insert_with(|| ChunkEntry {
                chunk_hash,
                block_id,
                tier_mask: TierMask::new(),
                token_count,
            });
        entry.block_id = block_id;
        entry.token_count = token_count;
        entry.tier_mask.set_tier(tier);
        self.block_to_hash.insert(block_id, chunk_hash);
    }

    /// Remove a specific cache tier from a chunk's mask.
    pub fn remove_tier(&mut self, chunk_hash: u64, tier: CacheTier) {
        if let Some(entry) = self.entries.get_mut(&chunk_hash) {
            entry.tier_mask.clear_tier(tier);
            if entry.tier_mask.is_empty() {
                let bid = entry.block_id;
                self.entries.remove(&chunk_hash);
                self.block_to_hash.remove(&bid);
            }
        }
    }

    /// Remove by block ID.
    pub fn remove_block(&mut self, block_id: usize) {
        if let Some(hash) = self.block_to_hash.remove(&block_id) {
            self.entries.remove(&hash);
        }
    }

    /// Lookup a chunk entry by its token sequence hash.
    pub fn lookup(&self, chunk_hash: u64) -> Option<&ChunkEntry> {
        self.entries.get(&chunk_hash)
    }

    /// Check prefix match length across consecutive chunk hashes.
    /// Returns the number of matched tokens and the matched block IDs with tiers.
    pub fn match_prefix_chunks<'a, I>(&self, chunk_hashes: I) -> (usize, Vec<(usize, CacheTier)>)
    where
        I: IntoIterator<Item = &'a u64>,
    {
        let mut total_tokens = 0;
        let mut matches = Vec::new();

        for &hash in chunk_hashes {
            if let Some(entry) = self.entries.get(&hash) {
                if let Some(tier) = entry.tier_mask.highest_tier() {
                    total_tokens += entry.token_count;
                    matches.push((entry.block_id, tier));
                    continue;
                }
            }
            break;
        }

        (total_tokens, matches)
    }

    /// Total indexed chunks.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is index empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_mask_operations() {
        let mut mask = TierMask::new();
        assert!(mask.is_empty());

        mask.set_tier(CacheTier::Gpu);
        assert!(mask.has_tier(CacheTier::Gpu));
        assert!(!mask.has_tier(CacheTier::HostRam));
        assert_eq!(mask.highest_tier(), Some(CacheTier::Gpu));

        mask.set_tier(CacheTier::HostRam);
        assert!(mask.has_tier(CacheTier::HostRam));
        assert_eq!(mask.highest_tier(), Some(CacheTier::Gpu));

        mask.clear_tier(CacheTier::Gpu);
        assert!(!mask.has_tier(CacheTier::Gpu));
        assert_eq!(mask.highest_tier(), Some(CacheTier::HostRam));
    }

    #[test]
    fn test_bitmask_chunk_index_prefix_matching() {
        let mut index = BitmaskChunkIndex::new();
        index.record_chunk(1001, 1, 64, CacheTier::Gpu);
        index.record_chunk(1002, 2, 64, CacheTier::HostRam);
        index.record_chunk(1003, 3, 64, CacheTier::NvMe);

        let query = vec![1001, 1002, 1003, 1004];
        let (tokens, matched) = index.match_prefix_chunks(&query);

        assert_eq!(tokens, 192);
        assert_eq!(matched.len(), 3);
        assert_eq!(matched[0], (1, CacheTier::Gpu));
        assert_eq!(matched[1], (2, CacheTier::HostRam));
        assert_eq!(matched[2], (3, CacheTier::NvMe));
    }
}
