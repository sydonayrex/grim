//! TOPS-style visual token pruning for training-time KV compression.
//!
//! Token dropping based on attention entropy: low-entropy (focused-attention)
//! tokens are deemed important and preserved; high-entropy tokens are pruned.
//!
//! Reference: TOPS (Token Pruning via Attention Entropy) — keep top
//! `preservation_ratio` tokens by importance = 1 / (entropy + eps).


use grim_tensor::{Shape, Tensor};

/// Configuration for TOPS token pruning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopsConfig {
    /// Fraction of tokens to preserve (keep top-k by importance).
    /// Defaults to `0.2` (keep top 20%).
    pub preservation_ratio: f32,
    /// Window size used when computing entropy from raw attention weights.
    /// Defaults to `64`.
    pub entropy_window: usize,
}

impl Default for TopsConfig {
    fn default() -> Self {
        Self {
            preservation_ratio: 0.2,
            entropy_window: 64,
        }
    }
}

/// Host-side TOPS pruner. Operates on CPU tensors for now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopsPruner {
    pub config: TopsConfig,
}

impl TopsPruner {
    /// Create a new pruner with the given configuration.
    pub fn new(config: TopsConfig) -> Self {
        Self { config }
    }

    /// Create a pruner with default configuration (`preservation_ratio = 0.2`).
    pub fn default_pruner() -> Self {
        Self {
            config: TopsConfig::default(),
        }
    }

    /// Prune input tokens by per-token attention entropy.
    ///
    /// # Arguments
    /// - `input_tensor`: 2D tensor of shape `[seq_len, hidden]` (CPU data).
    /// - `attention_entropy`: per-token scalar entropy values, length `seq_len`.
    ///
    /// # Returns
    /// `(pruned_tensor, preserved_indices)` where:
    /// - `pruned_tensor` has shape `[k, hidden]` containing the selected tokens.
    /// - `preserved_indices` gives the original token positions kept, length `k`.
    ///
    /// # Panics
    /// Panics if `input_tensor` is not 2D, or if `attention_entropy` length
    /// does not match `input_tensor`'s first dimension.
    pub fn prune(&self, input_tensor: &Tensor, attention_entropy: &[f32]) -> (Tensor, Vec<usize>) {
        let dims = input_tensor.shape().dims();
        assert!(
            dims.len() == 2,
            "prune expects 2D tensor [seq_len, hidden], got shape {:?}",
            dims
        );
        let seq_len = dims[0];
        let hidden = dims[1];
        assert_eq!(
            attention_entropy.len(),
            seq_len,
            "attention_entropy length ({}) must equal seq_len ({})",
            attention_entropy.len(),
            seq_len
        );

        let data = input_tensor
            .to_vec_f32()
            .expect("prune currently requires CPU tensor data");

        // (a) importance = 1 / (entropy + eps)
        let eps = 1e-8f32;
        let mut indexed: Vec<(usize, f32)> = attention_entropy
            .iter()
            .enumerate()
            .map(|(idx, &ent)| (idx, 1.0 / (ent + eps)))
            .collect();

        // (b) select top-k by importance
        let k = ((self.config.preservation_ratio * seq_len as f32).floor() as usize)
            .max(1)
            .min(seq_len);
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let preserved: Vec<usize> = indexed.iter().take(k).map(|(idx, _)| *idx).collect();

        // (c) build pruned tensor [k, hidden]
        let mut pruned_data = Vec::with_capacity(k * hidden);
        for &idx in &preserved {
            let start = idx * hidden;
            let end = start + hidden;
            pruned_data.extend_from_slice(&data[start..end]);
        }

        let pruned_shape = Shape::new(vec![k, hidden]);
        let pruned_tensor = cpu_tensor(pruned_data, pruned_shape);

        (pruned_tensor, preserved)
    }
}

/// Compute average attention entropy per token from raw attention weights.
///
/// # Arguments
/// - `attention_weights`: flattened attention weight matrix with length
///   `num_heads * seq_len * seq_len` in row-major order.
/// - `seq_len`: sequence length.
/// - `num_heads`: number of attention heads.
///
/// # Returns
/// Per-token entropy values of length `seq_len`, averaged across heads.
///
/// # Panics
/// Panics if `attention_weights.len() != num_heads * seq_len * seq_len`.
pub fn compute_entropy(attention_weights: &[f32], seq_len: usize, num_heads: usize) -> Vec<f32> {
    assert_eq!(
        attention_weights.len(),
        num_heads * seq_len * seq_len,
        "compute_entropy expects num_heads * seq_len^2 weights; got {}",
        attention_weights.len()
    );

    let mut entropy_per_token = vec![0.0f32; seq_len];
    let row_stride = seq_len;

    for head in 0..num_heads {
        let head_offset = head * seq_len * seq_len;
        for token_idx in 0..seq_len {
            let row_start = head_offset + token_idx * row_stride;
            let row = &attention_weights[row_start..row_start + row_stride];

            // Entropy of the attention distribution over the key/value tokens.
            let mut ent = 0.0f32;
            for &w in row {
                if w > 0.0 {
                    ent -= w * w.ln();
                }
            }
            entropy_per_token[token_idx] += ent;
        }
    }

    if num_heads > 0 {
        for val in entropy_per_token.iter_mut() {
            *val /= num_heads as f32;
        }
    }

    entropy_per_token
}

/// Build a CPU tensor via the shared backend helper.
fn cpu_tensor(data: Vec<f32>, shape: Shape) -> Tensor {
    use grim_backend_cpu::cpu_tensor;
    cpu_tensor(data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tops_entropy_computation() {
        // Uniform attention: entropy should be ln(seq_len)
        let seq_len = 4usize;
        let num_heads = 2usize;
        let uniform = 1.0f32 / seq_len as f32;
        let mut weights = vec![0.0f32; num_heads * seq_len * seq_len];
        for head in 0..num_heads {
            let off = head * seq_len * seq_len;
            for i in 0..seq_len {
                weights[off + i * seq_len + i] = uniform;
            }
        }
        let ent = compute_entropy(&weights, seq_len, num_heads);
        assert_eq!(ent.len(), seq_len);
        // Per-head entropy of a single non-zero weight p=uniform: -p*ln(p).
        // Both heads are identical, so the averaged value equals the per-head value.
        let expected = -uniform * uniform.ln();
        for v in &ent {
            assert!(
                (v - expected).abs() < 1e-5,
                "uniform entropy expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_tops_prune_keeps_top_tokens() {
        let pruner = TopsPruner::new(TopsConfig {
            preservation_ratio: 0.5,
            entropy_window: 64,
        });
        // 4 tokens, hidden=2.
        let input = cpu_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![4, 2]),
        );
        // entropy low for tokens 0 and 1 -> high importance -> keep
        let entropy = vec![0.1, 0.2, 10.0, 20.0];
        let (pruned, indices) = pruner.prune(&input, &entropy);
        // k = floor(0.5 * 4) = 2; lowest entropy tokens are idx 0,1
        assert_eq!(indices, vec![0, 1]);
        // pruned should be rows 0 and 1: [1,2] and [3,4]
        let out = pruned.to_vec_f32().unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tops_prune_preserves_shape() {
        let pruner = TopsPruner::default_pruner();
        // seq_len=10, hidden=3, preservation_ratio=0.2 => k=2
        let mut data = vec![0.0f32; 10 * 3];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = i as f32;
        }
        let input = cpu_tensor(data.clone(), Shape::new(vec![10, 3]));
        let entropy: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (pruned, indices) = pruner.prune(&input, &entropy);
        let out = pruned.shape().dims();
        assert_eq!(out, &[2, 3], "expected [2, 3], got {:?}", out);
        assert_eq!(indices.len(), 2);
        // Highest importance = lowest entropy: idx 0,1
        assert_eq!(indices, vec![0, 1]);
        let out_vec = pruned.to_vec_f32().unwrap();
        assert_eq!(out_vec, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
