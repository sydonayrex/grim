//! Adapter that exposes [`grim_models_transformer::LlamaMtp`] (a Mamba/SSM
//! model with native multi-token prediction head) through the
//! [`NativeMtp`] speculative-decoding trait.
//!
//! §5.3.1: One model class registered to the speculative wrapper.

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::SessionT;
use grim_tensor::Tensor;
use grim_models_transformer::{LlamaMtp, MtpDepthProvider};

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
        // Get MTP tokens from the base provider
        let tokens = self.inner.predict_mtp_tokens(session, input_ids, positions)?;
        
        if tokens.is_empty() {
            return Ok(DraftBlock {
                tokens: vec![],
                base_logits: empty_logits(),
                confidence: vec![],
            });
        }

        // Run base forward to get base_logits
        let base_logits = self.inner.forward(session, input_ids, positions, &[])?;
        
        Ok(DraftBlock {
            tokens,
            base_logits,
            confidence: vec![1.0; self.mtp_depth()],
        })
    }
}

fn empty_logits() -> Tensor {
    use grim_tensor::{Shape, DType, Device, QuantProvenance};
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
        // Smoke test to ensure the adapter compiles and Linked correctly
        let _ = std::mem::size_of::<LlamaMtpAdapter>();
    }
}
