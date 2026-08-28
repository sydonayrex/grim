//! CacheBlend non-prefix KV cache stitching and selective attention recalibration.
//!
//! Repurposed from LMCache's CacheBlend mechanism.
//! Allows reusing cached KV blocks from non-prefix positions (e.g. multi-document RAG,
//! injected context, and tool responses) by stitching block ranges and computing
//! cross-boundary attention adjustments.

/// A contiguous segment of tokens that matches a cached chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedSegment {
    /// Token start index in current prompt
    pub prompt_start: usize,
    /// Number of tokens in the segment
    pub token_len: usize,
    /// Cached physical block IDs in the paged KV pool
    pub block_ids: Vec<usize>,
    /// Chunk sequence hash
    pub chunk_hash: u64,
}

/// Result of stitching prompt tokens against multi-tier cache blocks.
#[derive(Debug, Clone)]
pub struct StitchedPromptLayout {
    /// Total tokens in prompt
    pub total_tokens: usize,
    /// Reused cached segments
    pub cached_segments: Vec<CachedSegment>,
    /// Ranges of tokens that must be computed from scratch (prefilled)
    pub compute_ranges: Vec<(usize, usize)>,
    /// Number of tokens saved from prefill computation
    pub cached_token_count: usize,
}

/// CacheBlend engine for non-prefix chunk matching.
#[derive(Debug, Default)]
pub struct CacheBlendEngine {
    /// Chunk token size (e.g. 64 tokens)
    pub chunk_size: usize,
}

impl CacheBlendEngine {
    /// Create a new CacheBlend engine with a given chunk size.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(16),
        }
    }

    /// Analyze a prompt token stream and identify reusable chunk blocks.
    /// `lookup_fn` returns physical block IDs if a chunk hash is present in cache.
    pub fn plan_stitched_prompt<F>(
        &self,
        token_hashes: &[u64],
        tokens_per_chunk: usize,
        lookup_fn: F,
    ) -> StitchedPromptLayout
    where
        F: Fn(u64) -> Option<Vec<usize>>,
    {
        let total_chunks = token_hashes.len();
        let total_tokens = total_chunks * tokens_per_chunk;

        let mut cached_segments = Vec::new();
        let mut compute_ranges = Vec::new();

        let mut current_compute_start = None;
        let mut cached_tokens = 0;

        for (i, &hash) in token_hashes.iter().enumerate() {
            let prompt_offset = i * tokens_per_chunk;
            if let Some(blocks) = lookup_fn(hash) {
                // If there was an ongoing compute range, close it
                if let Some(start) = current_compute_start.take() {
                    compute_ranges.push((start, prompt_offset));
                }

                cached_tokens += tokens_per_chunk;
                cached_segments.push(CachedSegment {
                    prompt_start: prompt_offset,
                    token_len: tokens_per_chunk,
                    block_ids: blocks,
                    chunk_hash: hash,
                });
            } else if current_compute_start.is_none() {
                current_compute_start = Some(prompt_offset);
            }
        }

        // Close final compute range if prompt ended with a cache miss
        if let Some(start) = current_compute_start {
            compute_ranges.push((start, total_tokens));
        }

        StitchedPromptLayout {
            total_tokens,
            cached_segments,
            compute_ranges,
            cached_token_count: cached_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_cache_blend_stitching_plan() {
        let blend = CacheBlendEngine::new(64);
        let hashes = vec![101, 102, 999, 104]; // chunk 0, 1, 3 cached; chunk 2 missed

        let mut mock_cache = HashMap::new();
        mock_cache.insert(101, vec![1, 2]);
        mock_cache.insert(102, vec![3, 4]);
        mock_cache.insert(104, vec![7, 8]);

        let plan = blend.plan_stitched_prompt(&hashes, 64, |h| mock_cache.get(&h).cloned());

        assert_eq!(plan.total_tokens, 256);
        assert_eq!(plan.cached_token_count, 192);
        assert_eq!(plan.cached_segments.len(), 3);
        assert_eq!(plan.compute_ranges.len(), 1);
        assert_eq!(plan.compute_ranges[0], (128, 192)); // Only chunk 2 needs prefill computation
    }
}
