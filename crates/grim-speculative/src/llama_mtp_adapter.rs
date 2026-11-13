//! Adapter that exposes [`grim_models_transformer::LlamaMtp`] (a Mamba/SSM model with native multi-token prediction head) through the [`NativeMtp`] speculative-decoding trait.
//! §5.3.1: One model class registered to the speculative wrapper.

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::SessionT;
use grim_models_transformer::{LlamaMtp, MtpDepthProvider};
use grim_tensor::Tensor;

use crate::draft_backbone::DraftBlock;
use crate::native_mtp::NativeMtp;

/// Adapter wrapper that exposes `LlamaMtp` as a `NativeMtp`.
pub struct LlamaMtpAdapter {
    inner: Arc<LlamaMtp>,
}

impl LlamaMtpAdapter {
    pub fn new(inner: Arc<LlamaMtp>) -> Self {
        Self { inner }
    }
}

impl grim_core::Model for LlamaMtpAdapter {
    fn config(&self) -> &dyn grim_core::ModelConfig {
        self.inner.config()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device(&self) -> &grim_tensor::Device {
        self.inner.device()
    }
    fn param_arith(&self) -> grim_tensor::ArithType {
        self.inner.param_arith()
    }
}

impl CausalLm for LlamaMtpAdapter {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.inner.new_session()
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

impl NativeMtp for LlamaMtpAdapter {
    fn as_causal_lm(&self) -> &dyn CausalLm {
        self.inner.as_ref()
    }

    fn mtp_depth(&self) -> usize {
        self.inner.mtp_depth()
    }

    fn predict_multi(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<DraftBlock> {
        let base_logits = self.inner.forward(session, input_ids, positions, &[])?;
        self.predict_multi_after_forward(session, input_ids, positions, &base_logits)
    }

    /// M3: draft from an already-run base forward. Must not re-run the base input.
    fn predict_multi_after_forward(
        &self,
        session: &mut dyn SessionT,
        _input_ids: &Tensor,
        _positions: &Tensor,
        base_logits: &Tensor,
    ) -> Result<DraftBlock> {
        let tokens = self.inner.predict_mtp_tokens_after_forward(session, base_logits)?;

        if tokens.is_empty() {
            return Ok(DraftBlock {
                tokens: vec![],
                base_logits: empty_logits(),
                confidence: vec![],
            });
        }

        // Llama MTP layers currently expose draft token ids but not per-step draft-head
        // distributions as a DraftBlock tensor; use a sentinel block shape [1,vocab] is not
        // required here since we provide per-draft rows only if available. Fall back to empty
        // draft logits tensor with correct depth rows filled by caller? Instead, synthesize
        // zero rows per draft to keep tensor shape [depth, 1] minimal; verification will rely on
        // target logits and greedy/ratio paths as before. TODO(gpu-verify): provide true p_draft.
        let base_row = base_logits.to_vec_f32().unwrap_or_default();
        let vocab_size = base_logits.shape().dims().get(1).copied().unwrap_or(base_row.len()).max(1);
        let mut flat = Vec::with_capacity(tokens.len() * vocab_size);
        for _ in 0..tokens.len() {
            flat.extend(std::iter::repeat_n(0.0f32, vocab_size));
        }
        let shape = grim_tensor::Shape::new(vec![tokens.len(), vocab_size]);
        let draft_logits = grim_backend_cpu::cpu_tensor(flat, shape);

        Ok(DraftBlock {
            tokens,
            base_logits: draft_logits,
            confidence: vec![1.0; self.mtp_depth()],
        })
    }
}

fn empty_logits() -> Tensor {
    use grim_tensor::{DType, Device, QuantProvenance, Shape};
    let storage = Arc::new(grim_backend_cpu::CpuStorage::new(
        vec![0.0f32],
        Shape::new(vec![1, 1]),
        DType::F32,
    )) as Arc<dyn grim_tensor::BackendStorage>;
    Tensor::new(
        storage,
        Shape::new(vec![1, 1]),
        DType::F32,
        QuantProvenance::GrimNative,
        Device::Cpu,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_mtp_adapter_exists() {
        // Verify the adapter type is sized and non-zero
        assert!(std::mem::size_of::<LlamaMtpAdapter>() > 0);

        // Verify empty_logits() produces the correct shape and values
        let logits = empty_logits();
        assert_eq!(logits.shape().dims(), &[1, 1]);
        let data = logits.to_vec_f32().unwrap();
        assert_eq!(data, vec![0.0f32]);
    }

    #[test]
    fn test_llama_mtp_adapter_implements_traits() {
        // Verify the adapter implements all required traits by constructing
        // trait objects. This catches mutations that remove trait impls.
        fn _assert_native_mtp<T: NativeMtp + CausalLm + grim_core::Model>() {}
        _assert_native_mtp::<LlamaMtpAdapter>();
    }
}
