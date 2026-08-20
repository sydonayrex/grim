//! Thin wrapper around `Llama` for glmdsa uses a Llama-style transformer.

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::TensorParallelConfig;
use grim_tensor::{ArithType, Device, Tensor};

use crate::model::{Llama, LlamaConfig};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GlmDsaConfig {
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
    /// WI-P1 — sparse-attention (lightning-indexer) fields. Names match the
    /// DeepSeek-V3.2-Exp checkpoint header keys (`index_head_dim`,
    /// `index_n_heads`, `index_topk`), verified 2026-08-19 from the real
    /// header. All zero = dense (the historic default); any non-zero enables
    /// the selection core at load time.
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
}

impl Default for GlmDsaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            num_layers: 0,
            intermediate_size: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 0,
            index_head_dim: 0,
            index_n_heads: 0,
            index_topk: 0,
        }
    }
}

impl ModelConfig for GlmDsaConfig {
    fn name(&self) -> &str {
        "glmdsa"
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

pub struct GlmDsa {
    pub cfg: GlmDsaConfig,
    pub device: Device,
    pub inner: Llama,
    /// WI-P1 — constructed at load time when the config carries sparse-
    /// attention fields; `None` = dense fallback.
    pub selector: Option<grim_nn::sparse_attention::SparseAttentionSelector>,
}

impl GlmDsa {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: GlmDsaConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: GlmDsaConfig,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let inner = Llama::load_tp(device.clone(), ws, llama_cfg, tp)?;

        // WI-P1: construct the sparse-attention selection core when the config
        // carries indexer fields; otherwise serve dense (the historic
        // behavior) with a warning so the fallback is not silent.
        let selector = if cfg.index_head_dim > 0 || cfg.index_n_heads > 0 || cfg.index_topk > 0 {
            let sel = grim_nn::sparse_attention::SparseAttentionSelector::new(
                grim_nn::sparse_attention::SparseAttentionConfig {
                    index_head_dim: cfg.index_head_dim,
                    index_n_heads: cfg.index_n_heads,
                    index_topk: cfg.index_topk,
                },
            )?;
            eprintln!(
                "[glmdsa] WI-P1: sparse-attention selection enabled (index_head_dim={}, index_n_heads={}, index_topk={}).                  Applying the sparse mask inside Llama's attention is the checkpoint-gated follow-up: it requires loading the                  trained indexer weight tensors from a real GLM-DSA / DeepSeek-V3.2 checkpoint (tensor names unverified in this build);                  until then, serving remains dense.",
                cfg.index_head_dim, cfg.index_n_heads, cfg.index_topk
            );
            Some(sel)
        } else {
            eprintln!(
                "[glmdsa] warning: 'glmdsa' is serving as a dense Llama (no sparse-attention indexer fields in config);                  the DeepSeek Sparse Attention mechanism is not enabled."
            );
            None
        };
        Ok(Self {
            cfg,
            device: inner.device.clone(),
            inner,
            selector,
        })
    }
}

impl Model for GlmDsa {
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

impl CausalLm for GlmDsa {
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
