//! Continuous Batch Reordering & 3-Path Dispatch Orchestrator for ROCm.
//!
//! Reorders mixed-workload batches into contiguous `[Decode : Extend : Prefill]`
//! partitions before attention dispatch, eliminating KV-cache thrashing and
//! memory layout conversions.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestCategory {
    /// Generating 1 token with a populated KV cache.
    Decode,
    /// Generating tokens with partial prefix cache / multi-turn context.
    Extend,
    /// Fresh prompt with zero prior cached tokens.
    Prefill,
}

#[derive(Debug, Clone)]
pub struct SequenceMeta {
    pub seq_id: u64,
    pub num_new_tokens: usize,
    pub num_cached_tokens: usize,
}

impl SequenceMeta {
    pub fn new(seq_id: u64, num_new_tokens: usize, num_cached_tokens: usize) -> Self {
        Self {
            seq_id,
            num_new_tokens,
            num_cached_tokens,
        }
    }

    pub fn category(&self) -> RequestCategory {
        if self.num_new_tokens == 1 && self.num_cached_tokens > 0 {
            RequestCategory::Decode
        } else if self.num_cached_tokens > 0 {
            RequestCategory::Extend
        } else {
            RequestCategory::Prefill
        }
    }
}

/// Contiguous partition descriptor for reordered batch execution.
#[derive(Debug, Clone)]
pub struct ReorderedBatch {
    pub decode_indices: Vec<usize>,
    pub extend_indices: Vec<usize>,
    pub prefill_indices: Vec<usize>,
    /// Forward permutation mapping: original index -> reordered index.
    pub forward_map: Vec<usize>,
    /// Inverse permutation mapping: reordered index -> original index.
    pub inverse_map: Vec<usize>,
}

impl ReorderedBatch {
    pub fn decode_count(&self) -> usize {
        self.decode_indices.len()
    }

    pub fn extend_count(&self) -> usize {
        self.extend_indices.len()
    }

    pub fn prefill_count(&self) -> usize {
        self.prefill_indices.len()
    }

    pub fn is_pure_decode(&self) -> bool {
        self.extend_indices.is_empty() && self.prefill_indices.is_empty()
    }
}

/// Continuous Batch Reorderer for 3-path attention dispatch.
pub struct BatchReorderer;

impl BatchReorderer {
    /// Reorders a list of sequence metadata into `[Decode | Extend | Prefill]` layout.
    pub fn plan(sequences: &[SequenceMeta]) -> ReorderedBatch {
        let mut decode_indices = Vec::new();
        let mut extend_indices = Vec::new();
        let mut prefill_indices = Vec::new();

        for (orig_idx, seq) in sequences.iter().enumerate() {
            match seq.category() {
                RequestCategory::Decode => decode_indices.push(orig_idx),
                RequestCategory::Extend => extend_indices.push(orig_idx),
                RequestCategory::Prefill => prefill_indices.push(orig_idx),
            }
        }

        let n = sequences.len();
        let mut forward_map = vec![0usize; n];
        let mut inverse_map = Vec::with_capacity(n);

        let mut reordered_pos = 0;
        for &orig_idx in &decode_indices {
            forward_map[orig_idx] = reordered_pos;
            inverse_map.push(orig_idx);
            reordered_pos += 1;
        }
        for &orig_idx in &extend_indices {
            forward_map[orig_idx] = reordered_pos;
            inverse_map.push(orig_idx);
            reordered_pos += 1;
        }
        for &orig_idx in &prefill_indices {
            forward_map[orig_idx] = reordered_pos;
            inverse_map.push(orig_idx);
            reordered_pos += 1;
        }

        ReorderedBatch {
            decode_indices,
            extend_indices,
            prefill_indices,
            forward_map,
            inverse_map,
        }
    }

    /// Permutes elements of a slice according to the forward map.
    pub fn permute<T: Clone>(src: &[T], reordered: &ReorderedBatch) -> Vec<T> {
        let mut out = src.to_vec();
        for (orig_idx, &reordered_idx) in reordered.forward_map.iter().enumerate() {
            out[reordered_idx] = src[orig_idx].clone();
        }
        out
    }

    /// Restores elements of a reordered slice back to original request order.
    pub fn restore<T: Clone>(reordered_slice: &[T], reordered: &ReorderedBatch) -> Vec<T> {
        let mut out = reordered_slice.to_vec();
        for (reordered_idx, &orig_idx) in reordered.inverse_map.iter().enumerate() {
            out[orig_idx] = reordered_slice[reordered_idx].clone();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_reordering_partitions_and_restores() {
        let seqs = vec![
            SequenceMeta::new(1, 1, 50),   // Decode
            SequenceMeta::new(2, 64, 0),   // Prefill
            SequenceMeta::new(3, 16, 128), // Extend
            SequenceMeta::new(4, 1, 30),   // Decode
        ];

        let plan = BatchReorderer::plan(&seqs);
        assert_eq!(plan.decode_count(), 2);
        assert_eq!(plan.extend_count(), 1);
        assert_eq!(plan.prefill_count(), 1);

        // Decode: seq 1 (idx 0), seq 4 (idx 3)
        assert_eq!(plan.decode_indices, vec![0, 3]);
        // Extend: seq 3 (idx 2)
        assert_eq!(plan.extend_indices, vec![2]);
        // Prefill: seq 2 (idx 1)
        assert_eq!(plan.prefill_indices, vec![1]);

        let dummy_data = vec!["R1", "R2", "R3", "R4"];
        let permuted = BatchReorderer::permute(&dummy_data, &plan);
        assert_eq!(permuted, vec!["R1", "R4", "R3", "R2"]);

        let restored = BatchReorderer::restore(&permuted, &plan);
        assert_eq!(restored, dummy_data);
    }
}
