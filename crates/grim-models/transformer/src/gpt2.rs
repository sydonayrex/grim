//! GPT2 & GPT-NeoX family — standard LayerNorm + absolute positional embeddings.

use std::sync::Arc;
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear};
use grim_tensor::{ArithType, Device, DType, Tensor};

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for Gpt2Config {
    fn name(&self) -> &str {
        "gpt2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(ws: &grim_nn::WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        let bias = ws.get([dim], "bias")?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xv = x.to_vec_f32()?;
        let dim = x.shape().dims().last().copied().unwrap_or(1);
        let mut out = vec![0.0f32; xv.len()];
        for chunk in xv.chunks(dim).enumerate() {
            let (i, c) = chunk;
            let mean = c.iter().sum::<f32>() / dim as f32;
            let variance = c.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (variance + self.eps).sqrt();
            let w = self.weight.to_vec_f32()?;
            let b = self.bias.to_vec_f32()?;
            for j in 0..dim {
                out[i * dim + j] = ((c[j] - mean) * inv_std) * w[j] + b[j];
            }
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

pub struct Gpt2Block {
    pub ln_1: LayerNorm,
    pub wqkv: Linear,
    pub c_proj: Linear,
    pub ln_2: LayerNorm,
    pub ffn_gate: Linear,
    pub ffn_down: Linear,
}

impl Gpt2Block {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &Gpt2Config) -> Result<Self> {
        let ln_1 = LayerNorm::load(&ws.pp("ln_1"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let wqkv = Linear::load(&ws.pp("attn.wqkv"), cfg.hidden_size, 3 * cfg.hidden_size, true)?;
        let c_proj = Linear::load(&ws.pp("attn.c_proj"), cfg.hidden_size, cfg.hidden_size, true)?;
        let ln_2 = LayerNorm::load(&ws.pp("ln_2"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let ffn_gate = Linear::load(&ws.pp("mlp.c_fc"), cfg.hidden_size, cfg.intermediate_size, true)?;
        let ffn_down = Linear::load(&ws.pp("mlp.c_proj"), cfg.intermediate_size, cfg.hidden_size, true)?;

        Ok(Self {
            ln_1,
            wqkv,
            c_proj,
            ln_2,
            ffn_gate,
            ffn_down,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.ln_1.forward(x)?;
        let qkv = self.wqkv.forward(&norm_x)?;
        let attn_out = self.c_proj.forward(&qkv)?;
        let x_res1 = add_tensors(x, &attn_out)?;

        let norm_x2 = self.ln_2.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let ffn_out = self.ffn_down.forward(&gate)?;
        add_tensors(&x_res1, &ffn_out)
    }
}

pub struct Gpt2 {
    pub cfg: Gpt2Config,
    pub device: Device,
    pub wte: Embedding,
    pub wpe: Embedding,
    pub layers: Vec<Gpt2Block>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}

impl Gpt2 {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: Gpt2Config) -> Result<Self> {
        let wte = Embedding::load(&ws.pp("wte"), cfg.vocab_size, cfg.hidden_size)?;
        let wpe = Embedding::load(&ws.pp("wpe"), cfg.max_seq_len, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Gpt2Block::load(&ws.pp("h").pp(&i.to_string()), &cfg)?);
        }
        let ln_f = LayerNorm::load(&ws.pp("ln_f"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let lm_head = Linear::load(&ws.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            cfg,
            device: Device::Cpu,
            wte,
            wpe,
            layers,
            ln_f,
            lm_head,
        })
    }
}

impl Model for Gpt2 {
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

impl CausalLm for Gpt2 {
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
        let tok_emb = self.wte.forward(&ids, seq_len, self.cfg.hidden_size)?;
        let pos_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let pos_emb = self.wpe.forward(&pos_ids, seq_len, self.cfg.hidden_size)?;

        let mut h = add_tensors(&tok_emb, &pos_emb)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.ln_f.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
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
