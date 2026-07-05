//! Concrete `MarkovHead` impl: applies a position-conditioned bias to the
//! base logits using only the prefix within the current block.
//!
//! §5.3.2: the Markov head is *intra-block* — it doesn't see beyond the
//! block, just the prefix within it. A real implementation would learn
//! a low-rank prefix-conditioned bias (typically rank-256 in DSpark-style
//! configs). This structural impl uses a deterministic per-position
//! address into a small learned bank.

use std::sync::Arc;

use grim_core::error::Result;
use grim_tensor::Tensor;

use crate::markov_head::MarkovHead;

/// A small learned bias bank indexed by `(prefix_len, position_within_block)`.
pub struct UniformMarkovHead {
    pub bias_table: Vec<f32>,
    pub vocab_size: usize,
    pub max_block_len: usize,
}

impl UniformMarkovHead {
    /// `bias_table.len() == max_block_len * max_block_len * vocab_size`.
    /// The first axis represents the prefix length observed, the second the
    /// query position within the block — typically the first index (prefix
    /// length 0) applies a uniform bias; later positions bias later vocab
    /// ids more strongly.
    pub fn new(vocab_size: usize, max_block_len: usize, seed: u64) -> Self {
        let mut rng = crate::test_rng::SimpleRng::new(seed);
        let total = max_block_len * max_block_len * vocab_size;
        let bias_table = (0..total).map(|_| (rng.next_f32() - 0.5) * 0.05).collect();
        Self {
            bias_table,
            vocab_size,
            max_block_len,
        }
    }
}

impl MarkovHead for UniformMarkovHead {
    fn bias(
        &self,
        prefix_within_block: &[u32],
        base_logits: &Tensor,
    ) -> Result<Tensor> {
        let prefix_len = prefix_within_block.len().min(self.max_block_len);
        let shape = base_logits.shape().dims().to_vec();
        if shape.len() != 2 || shape[1] != self.vocab_size {
            return Err(grim_core::error::Error::Shape(format!(
                "UniformMarkovHead expects (T, vocab={}), got {:?}",
                self.vocab_size, shape
            )));
        }
        let logits = base_logits.to_vec_f32()?;
        let tokens = shape[0];
        let mut biased = vec![0.0f32; logits.len()];
        for t in 0..tokens {
            // position_within_block ranges over the bank's second axis.
            let pos_in_block = t.min(self.max_block_len - 1);
            for v in 0..self.vocab_size {
                let idx = (pos_in_block * self.max_block_len + prefix_len)
                    * self.vocab_size
                    + v;
                biased[t * self.vocab_size + v] = logits[t * self.vocab_size + v]
                    + self.bias_table[idx.min(self.bias_table.len() - 1)];
            }
        }
        Ok(grim_backend_cpu::cpu_tensor(
            biased,
            grim_tensor::Shape::new(shape),
        ))
    }
}

impl From<UniformMarkovHead> for Arc<dyn MarkovHead> {
    fn from(m: UniformMarkovHead) -> Self {
        Arc::new(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::Shape;

    #[test]
    fn markov_head_returns_same_shape() {
        let head = UniformMarkovHead::new(32, 5, 0xCAFE_BABE);
        let logits = grim_backend_cpu::cpu_tensor(vec![0.0f32; 5 * 32], Shape::new(vec![5, 32]));
        let bias = head.bias(&[1, 2, 3], &logits).unwrap();
        assert_eq!(bias.shape().dims(), &[5, 32]);
    }

    #[test]
    fn markov_head_changes_logit_distribution() {
        let head = UniformMarkovHead::new(8, 4, 0x1234);
        let logits = grim_backend_cpu::cpu_tensor(vec![1.0f32; 2 * 8], Shape::new(vec![2, 8]));
        let baseline = logits.to_vec_f32().unwrap();
        let biased = head.bias(&[], &logits).unwrap().to_vec_f32().unwrap();
        // The biased output must differ at least somewhere from the
        // baseline (Markov bias should never be a no-op on a real
        // weight initialization).
        assert_ne!(baseline, biased);
    }
}
