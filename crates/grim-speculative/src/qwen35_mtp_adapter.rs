//! Adapter exposing [`grim_models_transformer::Qwen35`] nextn/MTP draft capability
//! through [`NativeMtp`] (M1/M3/M4 wiring).

use std::sync::Arc;

use grim_core::error::{Error, Result};
use grim_core::model::CausalLm;
use grim_core::session::SessionT;
use grim_models_transformer::Qwen35;
use grim_tensor::{DType, Device, QuantProvenance, Shape, Tensor};

use crate::draft_backbone::DraftBlock;
use crate::native_mtp::NativeMtp;

/// M1+ adapter wrapping a Qwen3.5/3.8 model that has a loaded nextn block.
pub struct Qwen35MtpAdapter {
    inner: Arc<dyn CausalLm>,
}

impl Qwen35MtpAdapter {
    pub fn new(inner: Arc<dyn CausalLm>) -> Self {
        Self { inner }
    }

    pub fn from_qwen35(inner: Arc<Qwen35>) -> Self {
        Self {
            inner: inner as Arc<dyn CausalLm>,
        }
    }

    fn qwen(&self) -> Result<&Qwen35> {
        self.inner
            .as_any()
            .downcast_ref::<Qwen35>()
            .ok_or_else(|| Error::Config("Qwen35MtpAdapter: inner model is not Qwen35".into()))
    }

    pub fn inner(&self) -> &Arc<dyn CausalLm> {
        &self.inner
    }
}

impl grim_core::Model for Qwen35MtpAdapter {
    fn config(&self) -> &dyn grim_core::ModelConfig {
        self.inner.config()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device(&self) -> &Device {
        self.inner.device()
    }
    fn param_arith(&self) -> grim_tensor::ArithType {
        self.inner.param_arith()
    }
}

impl CausalLm for Qwen35MtpAdapter {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.inner.new_session()
    }

    fn has_recurrent_state(&self) -> bool {
        self.inner.has_recurrent_state()
    }

    fn hidden_size_hint(&self) -> Option<usize> {
        self.inner.hidden_size_hint()
    }

    fn num_layers_hint(&self) -> Option<usize> {
        self.inner.num_layers_hint()
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[grim_core::model::AdapterHandle],
    ) -> Result<Tensor> {
        self.inner.forward(session, input_ids, positions, adapters)
    }
}

impl NativeMtp for Qwen35MtpAdapter {
    fn as_causal_lm(&self) -> &dyn CausalLm {
        self.inner.as_ref()
    }

    fn mtp_depth(&self) -> usize {
        self.qwen().map(|q| q.mtp_depth()).unwrap_or(0)
    }

    fn predict_multi(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<DraftBlock> {
        let qwen = self.qwen()?;
        let depth = qwen.mtp_depth();
        if depth == 0 {
            return Ok(empty_block(grim_core::Model::device(qwen)));
        }

        // Run trunk forward to obtain base logits and populate last hidden state.
        let base_logits = self.inner.forward(session, input_ids, positions, &[])?;
        self.predict_multi_after_forward(session, input_ids, positions, &base_logits)
    }

    /// M3: draft from the *already-run* base forward. Never re-runs the base input; doing so
    /// would append the same token at the same position twice and desync the KV cache.
    fn predict_multi_after_forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _base_logits: &Tensor,
    ) -> Result<DraftBlock> {
        let qwen = self.qwen()?;
        let depth = qwen.mtp_depth();
        if depth == 0 {
            return Ok(empty_block(grim_core::Model::device(qwen)));
        }

        // The trunk hidden state saved by the base forward is the nextn block's input. Without
        // it we cannot draft; return empty and let the caller fall back to a plain step.
        let Some(h_prev) = session.get_last_hidden_state() else {
            return Ok(empty_block(grim_core::Model::device(qwen)));
        };

        let input_vec = input_ids.to_vec_f32().unwrap_or_default();
        let pos_vec = positions.to_vec_f32().unwrap_or_default();
        let bonus_token = input_vec.last().copied().unwrap_or(0.0) as u32;
        let bonus_pos = pos_vec.last().copied().unwrap_or(0.0) as u32;

        // Trunk seed is post-output_norm hidden (`h_nextn`-equivalent, llama.cpp qwen3next /
        // Strata normed-seed contract); nextn applies its own enorm/hnorm in forward_step.
        // TODO(gpu-verify): confirm acceptance/identity on clean sequential GPUs.
        let (tokens, step_logits) = qwen.predict_nextn_draft(
            bonus_token,
            &h_prev,
            bonus_pos.saturating_add(1),
            depth,
        )?;

        if tokens.is_empty() {
            return Ok(DraftBlock {
                tokens: vec![],
                base_logits: empty_block(grim_core::Model::device(qwen)).base_logits,
                confidence: vec![],
            });
        }

        // Stack per-step logits [1, vocab] -> [depth, vocab] on CPU for DraftBlock.
        let vocab_size = qwen.cfg.vocab_size;
        let mut flat: Vec<f32> = Vec::with_capacity(tokens.len() * vocab_size);
        for l in &step_logits {
            let v = l.to_vec_f32()?;
            if v.len() == vocab_size {
                flat.extend_from_slice(&v);
            } else if v.len() > vocab_size {
                flat.extend_from_slice(&v[v.len() - vocab_size..]);
            } else {
                flat.extend(std::iter::repeat_n(0.0, vocab_size));
            }
        }
        let shape = Shape::new(vec![tokens.len(), vocab_size]);
        let base_logits = grim_backend_cpu::cpu_tensor(flat, shape);

        Ok(DraftBlock {
            confidence: vec![1.0; tokens.len()],
            tokens,
            base_logits,
        })
    }
}

fn empty_block(device: &Device) -> DraftBlock {
    let storage = std::sync::Arc::new(grim_backend_cpu::CpuStorage::new(
        vec![0.0f32],
        Shape::new(vec![1, 1]),
        DType::F32,
    )) as std::sync::Arc<dyn grim_tensor::BackendStorage>;
    DraftBlock {
        tokens: vec![],
        base_logits: Tensor::new(
            storage,
            Shape::new(vec![1, 1]),
            DType::F32,
            QuantProvenance::GrimNative,
            device.clone(),
        ),
        confidence: vec![],
    }
}
