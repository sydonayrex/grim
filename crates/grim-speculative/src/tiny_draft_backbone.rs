//! Concrete `DraftBackbone` impl: a tiny single-layer transformer
//! designed for unit testability of the speculative-decoding pipeline.
//!
//! On the CPU side this is a structural forward pass — small enough
//! that learning the weight by autograd is out of scope, but large
//! enough to produce non-trivial base logits for the verifier and
//! Markov-corrected confidence head to operate on.

use std::sync::{Arc, Mutex};

use grim_core::error::Result;
use grim_tensor::{Shape, Tensor};

use crate::draft_backbone::{DraftBackbone, DraftBlock};

/// Thread-safe interior mutability container for weights.
pub struct DraftWeights {
    pub(crate) pos_emb: Vec<f32>,
    pub(crate) w_q: Vec<f32>,
    pub(crate) w_k: Vec<f32>,
    pub(crate) w_v: Vec<f32>,
    pub(crate) w_proj: Vec<f32>,
    pub(crate) w_head: Vec<f32>,
}

/// Tiny draft transformer.
///
/// Holds a single linear projection (`emb → hidden`), positional
/// embeddings, an attention-style score against the query, and a
/// decoder head that maps to vocab logits.
///
/// Constructed from a seed for determinism in tests.
pub struct TinyDraftBackbone {
    pub vocab_size: usize,
    pub hidden: usize,
    pub block_len: usize,
    pub(crate) weights: Mutex<DraftWeights>,
}

impl TinyDraftBackbone {
    pub fn new(vocab_size: usize, hidden: usize, block_len: usize, seed: u64) -> Self {
        let mut rng = crate::test_rng::SimpleRng::new(seed);
        let mut rand_vec = |size: usize, std: f32| -> Vec<f32> {
            (0..size).map(|_| (rng.next_f32() - 0.5) * std).collect()
        };
        Self {
            vocab_size,
            hidden,
            block_len,
            weights: Mutex::new(DraftWeights {
                pos_emb: rand_vec(block_len * hidden, 0.05),
                w_q: rand_vec(hidden * hidden, 0.05),
                w_k: rand_vec(hidden * hidden, 0.05),
                w_v: rand_vec(hidden * hidden, 0.05),
                w_proj: rand_vec(vocab_size * hidden, 0.05),
                w_head: rand_vec(vocab_size * hidden, 0.05),
            }),
        }
    }
}

impl DraftBackbone for TinyDraftBackbone {
    fn draft_block(
        &self,
        _session: &mut dyn grim_core::session::SessionT,
        context: &Tensor,
        block_len: usize,
    ) -> Result<DraftBlock> {
        let block_len = block_len.min(self.block_len);
        let vocab = self.vocab_size;
        let hidden = self.hidden;

        // Lock weights for the forward pass
        let weights = self.weights.lock().unwrap();

        // 1. Pull the context representation (a 1-D F32 tensor of size `hidden`
        //    proxying the embedding of the last prompt token; real impl would
        //    embed the actual token and project to hidden).
        let ctx_data = context.to_vec_f32()?;
        let ctx_vec = if ctx_data.len() >= hidden {
            ctx_data[..hidden].to_vec()
        } else {
            let mut padded = vec![0.0f32; hidden];
            for (i, v) in ctx_data.iter().enumerate() {
                padded[i] = *v;
            }
            padded
        };

        // 2. Build the draft window: `block_len` queries, each = ctx + pos_emb.
        let mut queries = vec![0.0f32; block_len * hidden];
        for pos in 0..block_len {
            for h in 0..hidden {
                queries[pos * hidden + h] = ctx_vec[h] + weights.pos_emb[pos * hidden + h];
            }
        }

        // 3. Compute scores q · k (self-attention, treating every position
        //    as both q and k, simplified to position-conditioned vectors).
        //    For v1 we drop the k/v attention and just project the queries
        //    straight to vocab logits.
        let attn_out = matmul(&queries, &weights.w_q, block_len, hidden, hidden);

        // 4. Decode to vocab.
        let logits = matmul(&attn_out, &weights.w_head, block_len, hidden, vocab);

        // 5. Argmax sample per position.
        let mut tokens = Vec::with_capacity(block_len);
        let mut conf = Vec::with_capacity(block_len);
        let scale = 1.0 / (hidden as f32).sqrt();
        for pos in 0..block_len {
            // Softmax then max.
            let mut max = f32::NEG_INFINITY;
            for v in 0..vocab {
                if logits[pos * vocab + v] > max {
                    max = logits[pos * vocab + v];
                }
            }
            let mut sum = 0.0;
            let mut probs = vec![0.0f32; vocab];
            for v in 0..vocab {
                probs[v] = ((logits[pos * vocab + v] - max) * scale).exp();
                sum += probs[v];
            }
            for v in 0..vocab {
                probs[v] /= sum;
            }
            let mut best = 0;
            let mut best_p = f32::NEG_INFINITY;
            for v in 0..vocab {
                if probs[v] > best_p {
                    best_p = probs[v];
                    best = v;
                }
            }
            tokens.push(best as u32);
            conf.push(best_p);
        }

        let shape = Shape::new(vec![block_len, vocab]);
        let base_logits = grim_backend_cpu::cpu_tensor(logits, shape);

        Ok(DraftBlock {
            tokens,
            base_logits,
            confidence: conf,
        })
    }

    fn estimated_footprint_bytes(&self) -> usize {
        let weights = self.weights.lock().unwrap();
        // Calculate raw size of all the parameter arrays
        let params_len = weights.pos_emb.len()
            + weights.w_q.len()
            + weights.w_k.len()
            + weights.w_v.len()
            + weights.w_proj.len()
            + weights.w_head.len();
        params_len * std::mem::size_of::<f32>()
    }

    fn update_weights(
        &self,
        target_hidden_states: &[f32],
        draft_tokens: &[u32],
        accepted_mask: &[bool],
    ) -> Result<()> {
        let verify_len = draft_tokens.len();
        let hidden_size = self.hidden;
        let vocab_size = self.vocab_size;
        let mut weights = self.weights.lock().unwrap();
        
        let lr = 0.01f32;
        // Penultimate layer target hidden state mapping update:
        // Adjust the linear classification head (w_head) columns for accepted draft tokens
        // to associate stronger with the target's computed hidden states.
        for pos in 0..verify_len {
            if pos < accepted_mask.len() && accepted_mask[pos] {
                let t = draft_tokens[pos] as usize;
                if t < vocab_size {
                    let h_start = pos * hidden_size;
                    if h_start + hidden_size <= target_hidden_states.len() {
                        for d in 0..hidden_size {
                            weights.w_head[t * hidden_size + d] += lr * target_hidden_states[h_start + d];
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn matmul(input: &[f32], weight: &[f32], rows: usize, inner: usize, out: usize) -> Vec<f32> {
    // output[i, j] = Σ_k input[i, k] * weight[k, j]
    let mut out_buf = vec![0.0f32; rows * out];
    for i in 0..rows {
        for j in 0..out {
            let mut acc = 0.0f32;
            for k in 0..inner {
                acc += input[i * inner + k] * weight[k * out + j];
            }
            out_buf[i * out + j] = acc;
        }
    }
    out_buf
}

/// `Arc` convenience impl so callers can wrap [`TinyDraftBackbone`] for
/// the speculative wrapper's `DraftBackbone` trait object.
impl From<TinyDraftBackbone> for Arc<dyn DraftBackbone> {
    fn from(t: TinyDraftBackbone) -> Self {
        Arc::new(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::Shape;

    #[test]
    fn tiny_backbone_drafts_a_block_of_expected_shape() {
        let bd = TinyDraftBackbone::new(64, 16, 4, 0xDEAD_BEEF_CAFE_F00D_u64);
        let ctx = grim_backend_cpu::cpu_tensor(
            (0..16).map(|i| (i as f32 + 1.0) * 0.01).collect(),
            Shape::new(vec![16]),
        );
        let mut sess = grim_core::session::Inner::new(grim_tensor::Device::Cpu);
        let block = bd.draft_block(&mut sess, &ctx, 4).unwrap();
        assert_eq!(block.tokens.len(), 4);
        assert_eq!(block.base_logits.shape().dims(), &[4, 64]);
        assert!(block.confidence.iter().all(|c| *c > 0.0 && *c <= 1.0));
    }

    #[test]
    fn tiny_backbone_handles_shorter_block_request() {
        let bd = TinyDraftBackbone::new(8, 4, 8, 0x42);
        let ctx = grim_backend_cpu::cpu_tensor(vec![0.0f32; 4], Shape::new(vec![4]));
        let mut sess = grim_core::session::Inner::new(grim_tensor::Device::Cpu);
        let block = bd.draft_block(&mut sess, &ctx, 3).unwrap();
        assert_eq!(block.tokens.len(), 3);
    }
}
