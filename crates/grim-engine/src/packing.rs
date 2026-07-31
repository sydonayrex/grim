//! Padding-free batching and varlen sample packing for long-context training.
//!
//! Unsloth's headline technique: concatenate multiple short sequences into a
//! single batch so the GPU processes real tokens only — no padding waste.
//! This module provides the data structures and packing logic.

use grim_tensor::Shape;

/// A batch of token sequences concatenated back-to-back with no padding.
///
/// The `seqlen_offsets` array has length `N+1` for `N` sequences, where
/// `seqlen_offsets[i]` is the starting token index for sequence `i` and
/// `seqlen_offsets[N]` equals the total token count.
#[derive(Debug, Clone)]
pub struct PackedBatch {
    /// All token IDs concatenated back-to-back (no padding tokens).
    pub concatenated_tokens: Vec<u32>,
    /// Starting offset per sequence: `seqlen_offsets[i]` = start index of
    /// sequence `i` in `concatenated_tokens`. Length `num_sequences + 1`.
    pub seqlen_offsets: Vec<usize>,
    /// All-true mask (padding-free has no padding). Exists so downstream
    /// code that expects a boolean mask can call uniformly.
    pub attention_mask: Vec<bool>,
    /// Per-sequence token count used for loss masking at the end.
    pub sequence_lengths: Vec<usize>,
}

impl PackedBatch {
    /// Create an empty packed batch.
    pub fn new() -> Self {
        Self {
            concatenated_tokens: Vec::new(),
            seqlen_offsets: vec![0],
            attention_mask: Vec::new(),
            sequence_lengths: Vec::new(),
        }
    }

    /// Pack samples into batches of at most `max_packed_length` tokens each.
    ///
    /// Uses greedy bin-packing: samples are processed in the order provided
    /// and added to the current batch until the next sample would exceed
    /// `max_packed_length`. Each completed batch is yielded.
    pub fn pack_samples(
        samples: &[(Vec<u32>, Vec<u32>)],
        max_packed_length: usize,
    ) -> Vec<PackedBatch> {
        let mut batches = Vec::new();
        let mut current: PackedBatch = PackedBatch::new();

        for (input_ids, target_ids) in samples {
            let seq_len = input_ids.len().max(target_ids.len());

            if !current.concatenated_tokens.is_empty()
                && current.concatenated_tokens.len() + seq_len > max_packed_length
            {
                // Current batch is full; push it and start a new one.
                batches.push(current);
                current = PackedBatch::new();
            }

            let offset = current.concatenated_tokens.len();
            current.concatenated_tokens.extend_from_slice(input_ids);
            current.concatenated_tokens.extend_from_slice(target_ids);
            current.seqlen_offsets.push(current.concatenated_tokens.len());
            current.sequence_lengths.push(seq_len);
        }

        if !current.concatenated_tokens.is_empty() {
            batches.push(current);
        }

        batches
    }

    /// Returns an all-true attention mask of length equal to the total
    /// number of concatenated tokens — padding-free mode has no padding.
    pub fn packing_attention_mask(&self) -> Vec<bool> {
        vec![true; self.concatenated_tokens.len()]
    }
}

impl Default for PackedBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_single_sample_fits_in_batch() {
        let samples = vec![(vec![1, 2, 3], vec![4, 5, 6])];
        let batches = PackedBatch::pack_samples(&samples, 1024);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].concatenated_tokens, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(batches[0].seqlen_offsets, &[0, 6]);
    }

    #[test]
    fn test_pack_multiple_samples_splits_when_overflow() {
        let samples = vec![
            (vec![1, 2, 3], vec![4, 5, 6]), // 6 tokens
            (vec![7, 8], vec![9, 10]),       // 4 tokens (10 total > 6 limit)
        ];
        let batches = PackedBatch::pack_samples(&samples, 6);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].concatenated_tokens, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(batches[1].concatenated_tokens, &[7, 8, 9, 10]);
    }

    #[test]
    fn test_pack_empty_samples() {
        let batches = PackedBatch::pack_samples(&[], 1024);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_packing_attention_mask_all_true() {
        let samples = vec![(vec![1, 2, 3], vec![4, 5])];
        let batches = PackedBatch::pack_samples(&samples, 1024);
        assert_eq!(batches[0].packing_attention_mask(), &[true; 5]);
    }

    #[test]
    fn test_sequence_lengths_recorded() {
        let samples = vec![
            (vec![1, 2, 3], vec![4]), // input len 3, target len 1 -> max 3
            (vec![5, 6], vec![7, 8]), // max 2
        ];
        let batches = PackedBatch::pack_samples(&samples, 1024);
        assert_eq!(batches[0].sequence_lengths, &[3, 2]);
    }
}
