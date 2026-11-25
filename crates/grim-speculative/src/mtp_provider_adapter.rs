//! Generic `NativeMtp` adapter for any model implementing `CausalLm + MtpDepthProvider`.

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::SessionT;
use grim_models_transformer::MtpDepthProvider;
use grim_tensor::{DType, Device, QuantProvenance, Shape, Tensor};

use crate::draft_backbone::DraftBlock;
use crate::native_mtp::NativeMtp;

/// Model-agnostic adapter exposing any `CausalLm + MtpDepthProvider`
/// (e.g. `Qwen38FlashNextMtp`, `LlamaMtp`, or future HC-MTP models) through `NativeMtp`.
pub struct MtpProviderAdapter<M: CausalLm + MtpDepthProvider + 'static> {
    inner: Arc<M>,
}

impl<M: CausalLm + MtpDepthProvider + 'static> MtpProviderAdapter<M> {
    pub fn new(inner: Arc<M>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &Arc<M> {
        &self.inner
    }
}

impl<M: CausalLm + MtpDepthProvider + 'static> grim_core::Model for MtpProviderAdapter<M> {
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

impl<M: CausalLm + MtpDepthProvider + 'static> CausalLm for MtpProviderAdapter<M> {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.inner.new_session()
    }
    fn has_recurrent_state(&self) -> bool {
        self.inner.has_recurrent_state()
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

impl<M: CausalLm + MtpDepthProvider + 'static> NativeMtp for MtpProviderAdapter<M> {
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

    fn predict_multi_after_forward(
        &self,
        session: &mut dyn SessionT,
        _input_ids: &Tensor,
        _positions: &Tensor,
        base_logits: &Tensor,
    ) -> Result<DraftBlock> {
        // MtpDepthProvider exposes ids but not per-step draft distributions.
        // Stochastic speculation without real q cannot be lossless; decline
        // and let the wrapper use the plain target forward instead.
        if !session.sampling_greedy() {
            return Ok(DraftBlock {
                tokens: Vec::new(),
                base_logits: empty_logits(),
                confidence: Vec::new(),
            });
        }
        let tokens = self
            .inner
            .predict_mtp_tokens_after_forward(session, base_logits)?;
        if tokens.is_empty() {
            return Ok(DraftBlock {
                tokens: Vec::new(),
                base_logits: empty_logits(),
                confidence: Vec::new(),
            });
        }
        let n = tokens.len();
        let draft_logits = if session.sampling_greedy() {
            empty_logits()
        } else {
            let vocab = base_logits.shape().dims().last().copied().unwrap_or(1).max(1);
            grim_backend_cpu::cpu_tensor(vec![0.0f32; n * vocab], Shape::new(vec![n, vocab]))
        };
        Ok(DraftBlock {
            tokens,
            base_logits: draft_logits,
            confidence: vec![1.0; n],
        })
    }
}

fn empty_logits() -> Tensor {
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
