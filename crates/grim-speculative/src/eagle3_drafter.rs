//! EAGLE-3 integration as an accelerated speculative `DraftBackbone`.
//!
//! Exposes [`grim_models_transformer::Eagle3`] with multi-layer target feature fusion
//! and autoregressive draft rollout.

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_models_transformer::Eagle3;
use grim_tensor::{Device, Shape, Tensor};

use crate::draft_backbone::{DraftBackbone, DraftBlock};

/// Drafter backbone wrapping an `Eagle3` multi-layer feature fusion transformer model.
pub struct Eagle3Drafter {
    model: Arc<Eagle3>,
    _device: Device,
}

impl Eagle3Drafter {
    /// Create a new drafter backbone from an `Eagle3` model instance.
    pub fn new(model: Arc<Eagle3>) -> Self {
        let device = model.device.clone();
        Self {
            model,
            _device: device,
        }
    }

    /// Access the underlying `Eagle3` model.
    pub fn model(&self) -> &Eagle3 {
        &self.model
    }

    /// Perform multi-step speculative draft rollout fused with multi-layer target hidden features.
    ///
    /// Real EAGLE-3 takes the target model's intermediate layer representations (e.g. low, mid, high),
    /// projects them through `fc: 3 * D_target -> D_draft` to form initial hidden state $H_0$,
    /// and expands $K$ draft tokens autoregressively by concatenating $[E_t, H_{t-1}]$ at each step.
    pub fn draft_block_with_fusion(
        &self,
        target_hiddens: &[&Tensor],
        start_token: u32,
        block_len: usize,
    ) -> Result<DraftBlock> {
        let mut cur_h = self.model.fuse_target_layers(target_hiddens)?;
        let mut cur_token = start_token;

        let mut drafted_tokens = Vec::with_capacity(block_len);
        let mut confidences = Vec::with_capacity(block_len);
        let mut last_logits = None;

        let vocab_size = self.model.cfg.vocab_size;

        for step in 0..block_len {
            let embed =
                self.model
                    .tok_embeddings
                    .forward(&[cur_token], 1, self.model.cfg.hidden_size)?;

            let (logits, next_h) = self.model.decode_step(&embed, &cur_h, &[step as u32])?;

            let logits_vec = logits.to_vec_f32()?;
            let row_slice = if logits_vec.len() >= vocab_size {
                &logits_vec[logits_vec.len() - vocab_size..]
            } else {
                &logits_vec[..]
            };

            let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut top1 = f32::NEG_INFINITY;
            let mut top1_idx = 0u32;
            let mut top2 = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            for (i, &l) in row_slice.iter().enumerate() {
                let exp = (l - max_val).exp();
                sum_exp += exp;
                if l > top1 {
                    top2 = top1;
                    top1 = l;
                    top1_idx = i as u32;
                } else if l > top2 {
                    top2 = l;
                }
            }

            let top1_prob = if sum_exp > 0.0 {
                (top1 - max_val).exp() / sum_exp
            } else {
                0.5
            };
            let top2_prob = if sum_exp > 0.0 && top2 > f32::NEG_INFINITY {
                (top2 - max_val).exp() / sum_exp
            } else {
                0.0
            };
            let margin_conf = (top1_prob - top2_prob).clamp(0.05, 0.99);

            drafted_tokens.push(top1_idx);
            confidences.push(margin_conf);

            cur_token = top1_idx;
            cur_h = next_h;
            last_logits = Some(logits);
        }

        let base_logits = last_logits.unwrap_or_else(|| {
            grim_backend_cpu::cpu_tensor(vec![0.0f32; vocab_size], Shape::new(vec![1, vocab_size]))
        });

        Ok(DraftBlock {
            tokens: drafted_tokens,
            base_logits,
            confidence: confidences,
        })
    }
}

impl DraftBackbone for Eagle3Drafter {
    fn draft_block(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        context: &Tensor,
        block_len: usize,
    ) -> Result<DraftBlock> {
        let n_tokens = context.shape().dims().first().copied().unwrap_or(1);
        let pos_vec: Vec<f32> = (0..n_tokens).map(|i| i as f32).collect();
        let positions = grim_backend_cpu::cpu_tensor(pos_vec, Shape::new(vec![n_tokens]));

        let logits = self.model.forward(session, context, &positions, &[])?;
        let vocab_size = self.model.cfg.vocab_size;

        let logits_vec = logits.to_vec_f32()?;
        let mut drafted_tokens = Vec::with_capacity(block_len);
        let mut confidences = Vec::with_capacity(block_len);

        let rows = logits_vec.len() / vocab_size.max(1);
        let start_row = rows.saturating_sub(block_len);

        for r in start_row..rows {
            let row_slice = &logits_vec[r * vocab_size..(r + 1) * vocab_size];
            let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut top1 = f32::NEG_INFINITY;
            let mut top1_idx = 0u32;
            let mut top2 = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            for (i, &l) in row_slice.iter().enumerate() {
                let exp = (l - max_val).exp();
                sum_exp += exp;
                if l > top1 {
                    top2 = top1;
                    top1 = l;
                    top1_idx = i as u32;
                } else if l > top2 {
                    top2 = l;
                }
            }

            let top1_prob = if sum_exp > 0.0 {
                (top1 - max_val).exp() / sum_exp
            } else {
                0.5
            };
            let top2_prob = if sum_exp > 0.0 && top2 > f32::NEG_INFINITY {
                (top2 - max_val).exp() / sum_exp
            } else {
                0.0
            };
            let margin_conf = (top1_prob - top2_prob).clamp(0.05, 0.99);

            drafted_tokens.push(top1_idx);
            confidences.push(margin_conf);
        }

        Ok(DraftBlock {
            tokens: drafted_tokens,
            base_logits: logits,
            confidence: confidences,
        })
    }

    fn estimated_footprint_bytes(&self) -> usize {
        self.model.cfg.num_layers * self.model.cfg.hidden_size * self.model.cfg.hidden_size * 4
    }

    fn update_weights(
        &self,
        _target_hidden_states: &[f32],
        _draft_tokens: &[u32],
        _accepted_mask: &[bool],
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_models_transformer::Eagle3Config;

    #[test]
    fn test_eagle3_drafter_fusion_rollout() {
        let cfg = Eagle3Config {
            vocab_size: 50,
            hidden_size: 16,
            target_hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            num_layers: 1,
            intermediate_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 64,
            num_target_fusion_layers: 3,
        };

        let model = Arc::new(Eagle3::random(Device::Cpu, cfg));
        let drafter = Eagle3Drafter::new(model);

        // 3 target layer hidden states of dim 32
        let h1 = cpu_tensor(vec![0.5; 32], Shape::new(vec![1, 32]));
        let h2 = cpu_tensor(vec![1.0; 32], Shape::new(vec![1, 32]));
        let h3 = cpu_tensor(vec![1.5; 32], Shape::new(vec![1, 32]));

        // Roll out 3 speculative draft tokens with feature fusion
        let block = drafter
            .draft_block_with_fusion(&[&h1, &h2, &h3], 10, 3)
            .expect("draft_block_with_fusion should succeed");

        assert_eq!(block.tokens.len(), 3);
        assert_eq!(block.confidence.len(), 3);
        for conf in &block.confidence {
            assert!(*conf >= 0.05 && *conf <= 0.99);
        }
    }
}
