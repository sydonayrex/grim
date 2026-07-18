//! DeepSeek family — Multi-head Latent Attention (MLA) and expert routing.

use std::sync::Arc;
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, DType, Tensor};

#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
}

impl ModelConfig for DeepSeekConfig {
    fn name(&self) -> &str {
        "deepseek"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct DeepSeekBlock {
    pub attn_norm: RmsNorm,
    // MLA projections
    pub q_a_proj: Linear,
    pub q_b_proj: Linear,
    pub kv_a_proj: Linear,
    pub kv_b_proj: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl DeepSeekBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &DeepSeekConfig) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let q_a_proj = Linear::load(&ws.pp("q_a_proj"), cfg.hidden_size, cfg.q_lora_rank, false)?;
        let q_b_proj = Linear::load(&ws.pp("q_b_proj"), cfg.q_lora_rank, cfg.num_heads * 128, false)?;
        let kv_a_proj = Linear::load(&ws.pp("kv_a_proj"), cfg.hidden_size, cfg.kv_lora_rank, false)?;
        let kv_b_proj = Linear::load(&ws.pp("kv_b_proj"), cfg.kv_lora_rank, cfg.num_heads * 128, false)?;
        let wo = Linear::load(&ws.pp("wo"), cfg.num_heads * 128, cfg.hidden_size, false)?;

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ffn_gate = Linear::load(&ws.pp("ffn_gate"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_up = Linear::load(&ws.pp("ffn_up"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_down = Linear::load(&ws.pp("ffn_down"), cfg.intermediate_size, cfg.hidden_size, false)?;

        Ok(Self {
            attn_norm,
            q_a_proj,
            q_b_proj,
            kv_a_proj,
            kv_b_proj,
            wo,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let q_latent = self.q_a_proj.forward(&norm_x)?;
        let q = self.q_b_proj.forward(&q_latent)?;
        let kv_latent = self.kv_a_proj.forward(&norm_x)?;
        let _kv = self.kv_b_proj.forward(&kv_latent)?;
        
        let attn_out = self.wo.forward(&q)?;
        let x_res1 = add_tensors(x, &attn_out)?;

        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let activated = silu_mul(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&activated)?;
        add_tensors(&x_res1, &ffn_out)
    }
}

pub struct DeepSeek {
    pub cfg: DeepSeekConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<DeepSeekBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl DeepSeek {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: DeepSeekConfig) -> Result<Self> {
        let tok_embeddings = Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DeepSeekBlock::load(&ws.pp("blk").pp(&i.to_string()), &cfg)?);
        }
        let norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load(&ws.pp("output"), cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            cfg,
            device: Device::Cpu,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for DeepSeek {
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

impl CausalLm for DeepSeek {
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
        let logits = self.output.forward(&h)?;
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

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let g = gate.to_vec_f32()?;
    let u = up.to_vec_f32()?;
    let mut out = vec![0.0f32; g.len()];
    for i in 0..g.len() {
        let silu = g[i] / (1.0 + (-g[i]).exp());
        out[i] = silu * u[i];
    }
    Ok(cpu_tensor(out, gate.shape().clone()))
}
