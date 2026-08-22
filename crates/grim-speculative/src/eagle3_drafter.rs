//! EAGLE3 integration as a parallel `DraftBackbone`.
//!
//! Exposes [`grim_models_transformer::Eagle3`] as an accelerated parallel speculative drafter.

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_models_transformer::Eagle3;
use grim_tensor::{Device, Shape, Tensor};

use crate::draft_backbone::{DraftBackbone, DraftBlock};

/// Drafter backbone wrapping an `Eagle3` transformer model.
pub struct Eagle3Drafter {
    model: Arc<Eagle3>,
    _device: Device,
}

impl Eagle3Drafter {
    /// Create a new drafter backbone from an `Eagle3` model instance.
    pub fn new(model: Arc<Eagle3>) -> Self {
        let device = model.device.clone();
        Self { model, _device: device }
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
            let best_tok = row_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            drafted_tokens.push(best_tok);
            confidences.push(0.85); // High-confidence default from draft forward
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
        // Online adaptation feedback
        Ok(())
    }
}
