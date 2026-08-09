//! Thin wrapper around `Llama` for qwen3next uses a Llama-style transformer.

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, Model, ModelConfig, ModalityHint};
use grim_core::session::SessionT;
use grim_nn::TensorParallelConfig;
use grim_tensor::{ArithType, Device, Tensor};

use crate::model::{Llama, LlamaConfig};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Qwen3NextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for Qwen3NextConfig {
    fn name(&self) -> &str {
        "qwen3next"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Model — thin wrapper around Llama
// ---------------------------------------------------------------------------

pub struct Qwen3Next {
    pub cfg: Qwen3NextConfig,
    pub device: Device,
    pub inner: Llama,
}

impl Qwen3Next {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: Qwen3NextConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen3NextConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let llama_cfg = LlamaConfig {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_layers,
            intermediate_size: cfg.intermediate_size,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            max_seq_len: cfg.max_seq_len,
        };
        let inner = Llama::load_tp(device.clone(), ws, llama_cfg, tp)?;
        Ok(Self { cfg, device: inner.device.clone(), inner })
    }
}

impl Model for Qwen3Next {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for Qwen3Next {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.inner.new_session()
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        self.inner.forward(session, input_ids, positions, adapters)
    }
}
