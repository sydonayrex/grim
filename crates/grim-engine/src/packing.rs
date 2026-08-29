//! Padding-free batching and varlen sample packing for long-context training.
//!
//! Unsloth's headline technique: concatenate multiple short sequences into a
//! single batch so the GPU processes real tokens only — no padding waste.
//! This module provides the data structures and packing logic.

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
    /// Block-diagonal causal attention mask: `mask[i*T + j] == true` iff `i >= j`
    /// and `i, j` belong to the same packed sequence. Computed via
    /// [`PackedBatch::block_diagonal_causal_mask`].
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
            // Both input_ids and target_ids are concatenated into the buffer,
            // so the actual token count per sample is the sum, not the max.
            // Using max here undercounts the overflow check and records an
            // incorrect sequence_length, which breaks packed attention masks
            // when input_ids.len() != target_ids.len().
            let seq_len = input_ids.len() + target_ids.len();

            if !current.concatenated_tokens.is_empty()
                && current.concatenated_tokens.len() + seq_len > max_packed_length
            {
                // Current batch is full; push it and start a new one.
                current.attention_mask = current.block_diagonal_causal_mask();
                batches.push(current);
                current = PackedBatch::new();
            }

            current.concatenated_tokens.extend_from_slice(input_ids);
            current.concatenated_tokens.extend_from_slice(target_ids);
            current
                .seqlen_offsets
                .push(current.concatenated_tokens.len());
            current.sequence_lengths.push(seq_len);
        }

        if !current.concatenated_tokens.is_empty() {
            current.attention_mask = current.block_diagonal_causal_mask();
            batches.push(current);
        }

        batches
    }

    /// Build a `[T, T]` row-major boolean attention mask (T = total tokens)
    /// that is block-diagonal across packed sequences and causal within each.
    /// `mask[i*T + j] == true` iff `i >= j` and `i,j` belong to the same
    /// sequence per `seqlen_offsets`.
    pub fn block_diagonal_causal_mask(&self) -> Vec<bool> {
        let t = self.concatenated_tokens.len();
        if t == 0 {
            return Vec::new();
        }
        let total_elements = t.checked_mul(t).expect("packed batch mask dimensions overflow usize");
        // Cap dense boolean mask allocation to 16M elements (16MB) to prevent OOM
        if total_elements > 16 * 1024 * 1024 {
            panic!("packed sequence length {} exceeds maximum dense mask capacity (4096 tokens)", t);
        }
        let mut mask = vec![false; total_elements];
        let mut seq_of = vec![0usize; t];
        let mut cur = 0usize;
        for (pos, slot) in seq_of.iter_mut().enumerate() {
            while cur + 1 < self.seqlen_offsets.len() && pos >= self.seqlen_offsets[cur + 1] {
                cur += 1;
            }
            *slot = cur;
        }
        for i in 0..t {
            for j in 0..=i {
                if seq_of[i] == seq_of[j] {
                    mask[i * t + j] = true;
                }
            }
        }
        mask
    }

    /// Set the first target token of each packed sequence to `ignore_index`,
    /// because its prediction depends on the previous sequence's last token
    /// (cross-boundary). Mirrors Unsloth `mask_packed_sequence_boundaries`.
    pub fn boundary_loss_mask(&self, targets: &mut [u32], ignore_index: u32) {
        for &start in &self.seqlen_offsets {
            if start < targets.len() {
                targets[start] = ignore_index;
            }
        }
    }

    /// Returns the block-diagonal causal attention mask for this packed batch.
    ///
    /// `mask[i*T + j] == true` iff `i >= j` (causal) and `i, j` belong to the
    /// same packed sequence (block-diagonal — no cross-sequence attention).
    /// This is the correct mask for padding-free packed training.
    pub fn packing_attention_mask(&self) -> Vec<bool> {
        self.block_diagonal_causal_mask()
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
            (vec![7, 8], vec![9, 10]),      // 4 tokens (10 total > 6 limit)
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
    fn test_packing_attention_mask_block_diagonal() {
        let samples = vec![(vec![1, 2, 3], vec![4, 5])];
        let batches = PackedBatch::pack_samples(&samples, 1024);
        // Single sequence of 5 tokens: causal mask (i >= j) should be all-true
        // on the lower triangle, which for a single sequence is all-true.
        let mask = batches[0].packing_attention_mask();
        assert_eq!(mask.len(), 5 * 5);
        // Causal: position i attends to positions 0..=i
        for i in 0..5 {
            for j in 0..=i {
                assert!(mask[i * 5 + j], "expected true at ({}, {})", i, j);
            }
            for j in (i + 1)..5 {
                assert!(!mask[i * 5 + j], "expected false at ({}, {})", i, j);
            }
        }

        // Two-sequence pack: no cross-attention between sequences.
        let samples2 = vec![
            (vec![1, 2], vec![3]), // seq 0: offsets [0, 3]
            (vec![4], vec![5, 6]), // seq 1: offsets [3, 6]
        ];
        let batches2 = PackedBatch::pack_samples(&samples2, 1024);
        assert_eq!(batches2.len(), 1);
        let mask2 = batches2[0].packing_attention_mask();
        let t = 6;
        // seq 0 is positions 0..2 (3 tokens), seq 1 is positions 3..5 (3 tokens)
        // Position 0 in seq 1 (global idx 3) must NOT attend to position 2 (global idx 2)
        assert!(!mask2[3 * t + 2], "seq 1 must not attend to seq 0");
        // But must attend within its own sequence
        assert!(mask2[3 * t + 3], "seq 1 attends to itself (causal)");
        // Position 5 (last) must attend to all of seq 1 (indices 3, 4, 5)
        for j in &[3, 4, 5] {
            assert!(
                mask2[5 * t + *j],
                "last token attends to seq 1 position {}",
                j
            );
        }
        // Position 5 must NOT attend to seq 0 (indices 0, 1, 2)
        for j in &[0, 1, 2] {
            assert!(
                !mask2[5 * t + *j],
                "last token must not attend to seq 0 position {}",
                j
            );
        }
    }

    #[test]
    fn test_sequence_lengths_recorded() {
        let samples = vec![
            (vec![1, 2, 3], vec![4]), // input len 3, target len 1 -> sum 4
            (vec![5, 6], vec![7, 8]), // input len 2, target len 2 -> sum 4
        ];
        let batches = PackedBatch::pack_samples(&samples, 1024);
        assert_eq!(batches[0].sequence_lengths, &[4, 4]);
    }

    #[test]
    fn block_diagonal_mask_separates_packed_sequences() {
        let batch = PackedBatch {
            concatenated_tokens: vec![10, 11, 12, 20, 21],
            seqlen_offsets: vec![0, 3, 5],
            attention_mask: vec![],
            sequence_lengths: vec![3, 2],
        };
        let m = batch.block_diagonal_causal_mask();
        let at = |i: usize, j: usize| m[i * 5 + j];
        assert!(at(0, 0) && !at(0, 1) && !at(0, 2));
        assert!(at(1, 0) && at(1, 1) && !at(1, 2));
        assert!(at(2, 0) && at(2, 1) && at(2, 2));
        assert!(!at(2, 3) && !at(3, 2) && !at(4, 2));
        assert!(at(3, 3) && !at(3, 4));
        assert!(at(4, 3) && at(4, 4));
    }

    #[test]
    fn boundary_loss_mask_marks_sequence_starts() {
        let mut targets = vec![10u32, 11, 12, 20, 21];
        let offsets = vec![0usize, 3, 5];
        let b = PackedBatch {
            concatenated_tokens: vec![],
            seqlen_offsets: offsets,
            attention_mask: vec![],
            sequence_lengths: vec![],
        };
        b.boundary_loss_mask(&mut targets, 0xFFFF_FFFF);
        assert_eq!(targets[0], 0xFFFF_FFFF);
        assert_eq!(targets[3], 0xFFFF_FFFF);
        assert_eq!(targets[1], 11);
    }
}
