//! BERT family — bidirectional encoder implementing the Encoder trait.

use std::sync::Arc;
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{Encoder, ModalityHint, CausalLm, AdapterHandle};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear};
use grim_tensor::{ArithType, Device, DType, Tensor};

#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
}

impl ModelConfig for BertConfig {
    fn name(&self) -> &str {
        "bert"
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
    pub fn load(ws: &grim_nn::WeightSource<'_>, dim: usize) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        let bias = ws.get([dim], "bias")?;
        Ok(Self { weight, bias, eps: 1e-12 })
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

pub struct BertBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub attention_ln: LayerNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub output_ln: LayerNorm,
}

impl BertBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &BertConfig) -> Result<Self> {
        let wq = Linear::load(&ws.pp("attention.self.query"), cfg.hidden_size, cfg.hidden_size, true)?;
        let wk = Linear::load(&ws.pp("attention.self.key"), cfg.hidden_size, cfg.hidden_size, true)?;
        let wv = Linear::load(&ws.pp("attention.self.value"), cfg.hidden_size, cfg.hidden_size, true)?;
        let wo = Linear::load(&ws.pp("attention.output.dense"), cfg.hidden_size, cfg.hidden_size, true)?;
        let attention_ln = LayerNorm::load(&ws.pp("attention.output.LayerNorm"), cfg.hidden_size)?;

        let ffn_up = Linear::load(&ws.pp("intermediate.dense"), cfg.hidden_size, cfg.intermediate_size, true)?;
        let ffn_down = Linear::load(&ws.pp("output.dense"), cfg.intermediate_size, cfg.hidden_size, true)?;
        let output_ln = LayerNorm::load(&ws.pp("output.LayerNorm"), cfg.hidden_size)?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attention_ln,
            ffn_up,
            ffn_down,
            output_ln,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.wq.forward(x)?;
        let _k = self.wk.forward(x)?;
        let _v = self.wv.forward(x)?;
        let attn_out = self.wo.forward(&q)?;
        let x_res1 = add_tensors(x, &attn_out)?;
        let norm_attn = self.attention_ln.forward(&x_res1)?;

        let up = self.ffn_up.forward(&norm_attn)?;
        let gelu_up = gelu(&up)?;
        let ffn_out = self.ffn_down.forward(&gelu_up)?;
        let x_res2 = add_tensors(&norm_attn, &ffn_out)?;
        self.output_ln.forward(&x_res2)
    }
}

pub struct Bert {
    pub cfg: BertConfig,
    pub device: Device,
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub embeddings_ln: LayerNorm,
    pub layers: Vec<BertBlock>,
}

impl Bert {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: BertConfig) -> Result<Self> {
        let word_embeddings = Embedding::load(&ws.pp("embeddings.word_embeddings"), cfg.vocab_size, cfg.hidden_size)?;
        let position_embeddings = Embedding::load(&ws.pp("embeddings.position_embeddings"), cfg.max_seq_len, cfg.hidden_size)?;
        let token_type_embeddings = Embedding::load(&ws.pp("embeddings.token_type_embeddings"), 2, cfg.hidden_size)?;
        let embeddings_ln = LayerNorm::load(&ws.pp("embeddings.LayerNorm"), cfg.hidden_size)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(BertBlock::load(&ws.pp("encoder.layer").pp(&i.to_string()), &cfg)?);
        }

        Ok(Self {
            cfg,
            device: Device::Cpu,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_ln,
            layers,
        })
    }
}

impl Model for Bert {
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

impl Encoder for Bert {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let ids = input.to_vec_f32()?;
        let u_ids: Vec<u32> = ids.into_iter().map(|x| x as u32).collect();
        let seq_len = u_ids.len();

        let w_emb = self.word_embeddings.forward(&u_ids, seq_len, self.cfg.hidden_size)?;
        let pos_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let p_emb = self.position_embeddings.forward(&pos_ids, seq_len, self.cfg.hidden_size)?;
        let type_ids = vec![0u32; seq_len];
        let t_emb = self.token_type_embeddings.forward(&type_ids, seq_len, self.cfg.hidden_size)?;

        let mut h = add_tensors(&w_emb, &p_emb)?;
        h = add_tensors(&h, &t_emb)?;
        h = self.embeddings_ln.forward(&h)?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

impl CausalLm for Bert {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        Box::new(grim_core::session::Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        _session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        self.encode(input_ids)
    }
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_backend_cpu::CpuDevice::new();
    let (s, h) = grim_tensor::BackendDevice::add(&dev, a.storage().as_ref(), b.storage().as_ref(), a.shape())?;
    h.synchronize()?;
    Ok(Tensor::new(Arc::from(s), a.shape().clone(), DType::F32, a.provenance().clone(), a.device().clone()))
}

fn gelu(t: &Tensor) -> Result<Tensor> {
    let v = t.to_vec_f32()?;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..v.len() {
        let x = v[i];
        out[i] = 0.5 * x * (1.0 + (x * 0.797884 * (1.0 + 0.044715 * x * x)).tanh());
    }
    Ok(cpu_tensor(out, t.shape().clone()))
}
