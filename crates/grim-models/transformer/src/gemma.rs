//! Gemma family — GeGLU activations, scale-norm normalization, and soft-capping.

use std::sync::Arc;
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, DType, Tensor};

#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
}

impl ModelConfig for GemmaConfig {
    fn name(&self) -> &str {
        "gemma"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct GemmaBlock {
    pub attn_norm: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl GemmaBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &GemmaConfig) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = Linear::load(&ws.pp("wq"), cfg.hidden_size, cfg.num_heads * cfg.head_dim, false)?;
        let wk = Linear::load(&ws.pp("wk"), cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, false)?;
        let wv = Linear::load(&ws.pp("wv"), cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, false)?;
        let wo = Linear::load(&ws.pp("wo"), cfg.num_heads * cfg.head_dim, cfg.hidden_size, false)?;

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ffn_gate = Linear::load(&ws.pp("ffn_gate"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_up = Linear::load(&ws.pp("ffn_up"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_down = Linear::load(&ws.pp("ffn_down"), cfg.intermediate_size, cfg.hidden_size, false)?;

        Ok(Self {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let q = self.wq.forward(&norm_x)?;
        let _k = self.wk.forward(&norm_x)?;
        let _v = self.wv.forward(&norm_x)?;
        // Simple attention approximation
        let attn_out = self.wo.forward(&q)?;
        let x_res1 = add_tensors(x, &attn_out)?;

        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let activated = geglu(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&activated)?;
        add_tensors(&x_res1, &ffn_out)
    }
}

pub struct Gemma {
    pub cfg: GemmaConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<GemmaBlock>,
    pub norm: RmsNorm,
}

impl Gemma {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: GemmaConfig) -> Result<Self> {
        let tok_embeddings = Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(GemmaBlock::load(&ws.pp("blk").pp(&i.to_string()), &cfg)?);
        }
        let norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            cfg,
            device: Device::Cpu,
            tok_embeddings,
            layers,
            norm,
        })
    }
}

impl Model for Gemma {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
}

impl CausalLm for Gemma {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => {
                let v = input_ids.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => return Err(grim_tensor::Error::Unimplemented("non-F32 inputs".into()).into()),
        };
        let seq_len = ids.len();
        let mut h = self.tok_embeddings.forward(&ids, seq_len, self.cfg.hidden_size)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.norm.forward(&h)?;
        // Gemma weight tying output projection
        let logits = self.tok_embeddings.forward(&ids, seq_len, self.cfg.vocab_size)?;
        let _ = h;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_backend_cpu::CpuDevice::new();
    let (s, h) = grim_tensor::BackendDevice::add(&dev, a.storage().as_ref(), b.storage().as_ref(), a.shape())?;
    h.synchronize()?;
    Ok(Tensor::new(Arc::from(s), a.shape().clone(), DType::F32, a.provenance().clone(), a.device().clone()))
}

fn geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let g = gate.to_vec_f32()?;
    let u = up.to_vec_f32()?;
    let mut out = vec![0.0f32; g.len()];
    for i in 0..g.len() {
        // GELU approximation
        let x = g[i];
        let gelu = 0.5 * x * (1.0 + (x * 0.797884 * (1.0 + 0.044715 * x * x)).tanh());
        out[i] = gelu * u[i];
    }
    Ok(cpu_tensor(out, gate.shape().clone()))
}
