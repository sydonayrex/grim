//! Padding-free training and variable-length sequence packing.
//!
//! Enables efficient training by colocating multiple short sequences into
//! single batches, reducing padding waste. Matches Unsloth's varlen
//! training path for long context fine-tuning.

use grim_tensor::{
    BackendDevice, DType, Shape, Tensor,
    error::{Error, Result},
};
use std::sync::Arc;

/// A single token sequence with its length.
#[derive(Debug, Clone)]
pub struct TokenSequence {
    /// Token IDs (stored as f32 for compatibility with the tensor system)
    pub tokens: Vec<f32>,
    /// Optional label IDs (shifted by 1 for next-token prediction)
    pub labels: Vec<f32>,
}

impl TokenSequence {
    /// Create a new token sequence from u32 tokens.
    pub fn new(tokens: Vec<u32>) -> Self {
        let tokens_f32: Vec<f32> = tokens.iter().map(|t| *t as f32).collect();
        let labels: Vec<f32> = tokens_f32.clone();
        Self {
            tokens: tokens_f32,
            labels,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Create from f32 tokens directly (internal use)
    pub fn from_f32(tokens: Vec<f32>) -> Self {
        let labels = tokens.clone();
        Self { tokens, labels }
    }
}

/// Padding-free batch collator that packs variable-length sequences.
#[derive(Debug, Clone)]
pub struct VarLenCollator {
    /// Maximum sequence length in the packed batch
    pub max_seq_len: usize,
    /// Maximum number of sequences to pack
    pub max_seqs: usize,
    /// Padding token ID (as float)
    pub pad_token_id: f32,
}

impl VarLenCollator {
    pub fn new(max_seq_len: usize, max_seqs: usize, pad_token_id: f32) -> Self {
        Self {
            max_seq_len,
            max_seqs,
            pad_token_id,
        }
    }

    /// Create collator with u32 token IDs.
    pub fn with_token_id(max_seq_len: usize, max_seqs: usize, pad_token_id: u32) -> Self {
        Self::new(max_seq_len, max_seqs, pad_token_id as f32)
    }

    /// Collate variable-length sequences into a packed batch.
    /// Returns (input_ids, positions, attention_mask, labels) tensors.
    pub fn collate(&self, sequences: &[TokenSequence]) -> Result<PackedBatch> {
        if sequences.is_empty() {
            return Err(Error::Backend("No sequences to collate".into()));
        }

        // Sort sequences by length (longest first) for efficient packing
        let mut sorted: Vec<_> = sequences.iter().enumerate().collect();
        sorted.sort_by_key(|(_, s)| std::cmp::Reverse(s.len()));

        // Select sequences that fit within max_seq_len
        let mut selected: Vec<_> = Vec::new();
        let mut total_len = 0;

        for (idx, seq) in sorted {
            if selected.len() >= self.max_seqs {
                break;
            }
            if total_len + seq.len() > self.max_seq_len {
                break;
            }
            selected.push((idx, seq));
            total_len += seq.len();
        }

        if selected.is_empty() {
            return Err(Error::Backend("No sequences fit in batch".into()));
        }

        let num_seqs = selected.len();

        // Build packed input_ids tensor [num_seqs, max_len]
        let max_len = selected.iter().map(|(_, s)| s.len()).max().unwrap_or(0);
        let mut input_ids = vec![self.pad_token_id; num_seqs * max_len];
        let mut attention_mask = vec![0.0f32; num_seqs * max_len];
        let mut labels = vec![self.pad_token_id; num_seqs * max_len];

        for (seq_idx, (_, seq)) in selected.iter().enumerate() {
            let seq_len = seq.len();
            for i in 0..seq_len {
                input_ids[seq_idx * max_len + i] = seq.tokens[i];
                attention_mask[seq_idx * max_len + i] = 1.0;
                labels[seq_idx * max_len + i] = seq.labels[i];
            }
        }

        let shape = Shape::new(vec![num_seqs, max_len]);

        let input_tensor = Tensor::new(
            Arc::new(grim_backend_cpu::CpuStorage::new(
                input_ids,
                shape.clone(),
                DType::F32,
            )),
            shape.clone(),
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );

        let attention_tensor = Tensor::new(
            Arc::new(grim_backend_cpu::CpuStorage::new(
                attention_mask,
                shape.clone(),
                DType::F32,
            )),
            shape.clone(),
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );

        // Positions tensor - one position per token in the batch
        let total_tokens = num_seqs * max_len;
        let position_values: Vec<f32> = (0..total_tokens).map(|i| (i / max_len) as f32).collect();
        let pos_shape = Shape::new(vec![total_tokens]);
        let pos_shape_for_storage = pos_shape.clone();
        let positions_tensor = Tensor::new(
            Arc::new(grim_backend_cpu::CpuStorage::new(
                position_values,
                pos_shape_for_storage,
                DType::F32,
            )),
            pos_shape,
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );

        let labels_tensor = Tensor::new(
            Arc::new(grim_backend_cpu::CpuStorage::new(
                labels,
                shape.clone(),
                DType::F32,
            )),
            shape,
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );

        Ok(PackedBatch {
            input_ids: input_tensor,
            positions: positions_tensor,
            attention_mask: attention_tensor,
            labels: labels_tensor,
            cu_seqlens: selected.iter().map(|(_, s)| s.len()).collect(),
        })
    }

    /// Pack variable-length sequences into a single 1D concatenated sequence buffer
    /// with block-diagonal attention mask (P2.1 & P2.2 varlen packing, zero cross-example leakage).
    pub fn collate_1d_packed(&self, sequences: &[TokenSequence]) -> Result<Packed1DBatch> {
        if sequences.is_empty() {
            return Err(Error::Backend("No sequences to collate".into()));
        }

        let mut packed_tokens = Vec::new();
        let mut packed_labels = Vec::new();
        let mut packed_positions = Vec::new();
        let mut cu_seqlens = vec![0usize];
        let mut seq_lens = Vec::new();

        let mut current_offset = 0usize;

        for seq in sequences {
            if packed_tokens.len() + seq.len() > self.max_seq_len {
                break;
            }
            if seq_lens.len() >= self.max_seqs {
                break;
            }

            let len = seq.len();
            for i in 0..len {
                packed_tokens.push(seq.tokens[i]);
                packed_labels.push(seq.labels[i]);
                packed_positions.push(i as f32);
            }
            current_offset += len;
            cu_seqlens.push(current_offset);
            seq_lens.push(len);
        }

        if packed_tokens.is_empty() {
            return Err(Error::Backend("No sequences fit in 1D packed batch".into()));
        }

        let total_tokens = packed_tokens.len();
        // Block diagonal causal mask: [total_tokens, total_tokens]
        // mask[i, j] = 1.0 if token i and token j belong to same sequence AND j <= i, else 0.0
        let mut block_diag_mask = vec![0.0f32; total_tokens * total_tokens];

        let mut seq_start = 0;
        for &len in &seq_lens {
            let seq_end = seq_start + len;
            for i in seq_start..seq_end {
                for j in seq_start..=i {
                    block_diag_mask[i * total_tokens + j] = 1.0;
                }
            }
            seq_start = seq_end;
        }

        let shape_1d = Shape::new(vec![1, total_tokens]);
        let mask_shape = Shape::new(vec![total_tokens, total_tokens]);

        let dev = grim_backend_cpu::CpuDevice::new();
        let input_tensor = Tensor::new(
            Arc::from(dev.from_cpu(&packed_tokens, &shape_1d, DType::F32)?),
            shape_1d.clone(),
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );
        let labels_tensor = Tensor::new(
            Arc::from(dev.from_cpu(&packed_labels, &shape_1d, DType::F32)?),
            shape_1d.clone(),
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );
        let positions_tensor = Tensor::new(
            Arc::from(dev.from_cpu(&packed_positions, &shape_1d, DType::F32)?),
            shape_1d,
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );
        let mask_tensor = Tensor::new(
            Arc::from(dev.from_cpu(&block_diag_mask, &mask_shape, DType::F32)?),
            mask_shape,
            DType::F32,
            Default::default(),
            grim_tensor::Device::Cpu,
        );

        Ok(Packed1DBatch {
            input_ids: input_tensor,
            positions: positions_tensor,
            attention_mask: mask_tensor,
            labels: labels_tensor,
            cu_seqlens,
            total_tokens,
        })
    }
}

/// Packed 1D batch with block-diagonal attention mask for zero-padding SFT.
#[derive(Debug, Clone)]
pub struct Packed1DBatch {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub attention_mask: Tensor,
    pub labels: Tensor,
    pub cu_seqlens: Vec<usize>,
    pub total_tokens: usize,
}

/// Packed batch tensor bundle for varlen training.
#[derive(Debug, Clone)]
pub struct PackedBatch {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub attention_mask: Tensor,
    pub labels: Tensor,
    pub cu_seqlens: Vec<usize>,
}

impl PackedBatch {
    /// Get the number of sequences in this batch.
    pub fn num_sequences(&self) -> usize {
        self.input_ids.shape().dims()[0]
    }

    /// Get the maximum sequence length in this batch.
    pub fn max_seq_len(&self) -> usize {
        self.input_ids.shape().dims()[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varlen_collator_packs_sequences() {
        let collator = VarLenCollator::with_token_id(64, 8, 0);
        let sequences = vec![
            TokenSequence::new(vec![1, 2, 3]),
            TokenSequence::new(vec![4, 5]),
            TokenSequence::new(vec![6, 7, 8, 9]),
        ];
        let batch = collator.collate(&sequences).unwrap();
        assert!(batch.num_sequences() <= 8);
        assert!(batch.max_seq_len() <= 64);
    }

    #[test]
    fn test_varlen_collator_1d_block_diagonal() {
        let collator = VarLenCollator::with_token_id(64, 8, 0);
        let sequences = vec![
            TokenSequence::new(vec![10, 20, 30]),
            TokenSequence::new(vec![40, 50]),
        ];
        let packed = collator.collate_1d_packed(&sequences).unwrap();
        assert_eq!(packed.total_tokens, 5);
        assert_eq!(packed.cu_seqlens, vec![0, 3, 5]);

        let mask = packed.attention_mask.to_vec_f32().unwrap();
        // Token 0 can attend to 0 (seq 0)
        assert_eq!(mask[0 * 5 + 0], 1.0);
        // Token 3 (start of seq 1) CANNOT attend to token 0, 1, 2 (seq 0)
        assert_eq!(mask[3 * 5 + 0], 0.0);
        assert_eq!(mask[3 * 5 + 1], 0.0);
        assert_eq!(mask[3 * 5 + 2], 0.0);
        // Token 3 can attend to token 3
        assert_eq!(mask[3 * 5 + 3], 1.0);
        // Token 4 can attend to token 3 and 4
        assert_eq!(mask[4 * 5 + 3], 1.0);
        assert_eq!(mask[4 * 5 + 4], 1.0);
    }
}
