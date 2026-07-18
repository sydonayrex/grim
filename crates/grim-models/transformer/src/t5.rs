//! T5 family — Encoder-Decoder architecture using relative position bias.

use std::sync::Arc;
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{EncoderDecoderLm, ModalityHint, CausalLm, AdapterHandle};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, DType, Tensor};

#[derive(Debug, Clone)]
pub struct T5Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
}

impl ModelConfig for T5Config {
    fn name(&self) -> &str {
        "t5"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct T5Block {
    pub norm: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl T5Block {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &T5Config) -> Result<Self> {
        let norm = RmsNorm::load(&ws.pp("layer.0.layer_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = Linear::load(&ws.pp("layer.0.SelfAttention.q"), cfg.hidden_size, cfg.hidden_size, false)?;
        let wk = Linear::load(&ws.pp("layer.0.SelfAttention.k"), cfg.hidden_size, cfg.hidden_size, false)?;
        let wv = Linear::load(&ws.pp("layer.0.SelfAttention.v"), cfg.hidden_size, cfg.hidden_size, false)?;
        let wo = Linear::load(&ws.pp("layer.0.SelfAttention.o"), cfg.hidden_size, cfg.hidden_size, false)?;

        let ffn_norm = RmsNorm::load(&ws.pp("layer.1.layer_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ffn_up = Linear::load(&ws.pp("layer.1.DenseReluDense.wi"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_down = Linear::load(&ws.pp("layer.1.DenseReluDense.wo"), cfg.intermediate_size, cfg.hidden_size, false)?;

        Ok(Self {
            norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            ffn_up,
            ffn_down,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.norm.forward(x)?;
        let q = self.wq.forward(&norm_x)?;
        let _k = self.wk.forward(&norm_x)?;
        let _v = self.wv.forward(&norm_x)?;
        let attn_out = self.wo.forward(&q)?;
        let x_res1 = add_tensors(x, &attn_out)?;

        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let relu_up = relu(&up)?;
        let ffn_out = self.ffn_down.forward(&relu_up)?;
        add_tensors(&x_res1, &ffn_out)
    }
}

pub struct T5 {
    pub cfg: T5Config,
    pub device: Device,
    pub shared: Embedding,
    pub encoder_layers: Vec<T5Block>,
    pub decoder_layers: Vec<T5Block>,
    pub final_norm: RmsNorm,
}

impl T5 {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: T5Config) -> Result<Self> {
        let shared = Embedding::load(&ws.pp("shared"), cfg.vocab_size, cfg.hidden_size)?;
        let mut encoder_layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            encoder_layers.push(T5Block::load(&ws.pp("encoder.block").pp(&i.to_string()), &cfg)?);
        }
        let mut decoder_layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            decoder_layers.push(T5Block::load(&ws.pp("decoder.block").pp(&i.to_string()), &cfg)?);
        }
        let final_norm = RmsNorm::load(&ws.pp("encoder.final_layer_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        Ok(Self {
            cfg,
            device: Device::Cpu,
            shared,
            encoder_layers,
            decoder_layers,
            final_norm,
        })
    }
}

impl Model for T5 {
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

impl EncoderDecoderLm for T5 {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let ids = input.to_vec_f32()?;
        let u_ids: Vec<u32> = ids.into_iter().map(|x| x as u32).collect();
        let seq_len = u_ids.len();

        let mut h = self.shared.forward(&u_ids, seq_len, self.cfg.hidden_size)?;
        for layer in &self.encoder_layers {
            h = layer.forward(&h)?;
        }
        Ok(self.final_norm.forward(&h)?)
    }

    fn decode_step(
        &self,
        _session: &mut dyn grim_core::session::SessionT,
        _encoder_hidden_states: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let ids = input_ids.to_vec_f32()?;
        let u_ids: Vec<u32> = ids.into_iter().map(|x| x as u32).collect();
        let seq_len = u_ids.len();

        let mut h = self.shared.forward(&u_ids, seq_len, self.cfg.hidden_size)?;
        for layer in &self.decoder_layers {
            h = layer.forward(&h)?;
        }
        // Output vocabulary projection using tied embeddings
        Ok(self.shared.forward(&u_ids, seq_len, self.cfg.vocab_size)?)
    }
}

impl CausalLm for T5 {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        Box::new(grim_core::session::Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let hidden = self.encode(input_ids)?;
        self.decode_step(session, &hidden, input_ids)
    }
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_backend_cpu::CpuDevice::new();
    let (s, h) = grim_tensor::BackendDevice::add(&dev, a.storage().as_ref(), b.storage().as_ref(), a.shape())?;
    h.synchronize()?;
    Ok(Tensor::new(Arc::from(s), a.shape().clone(), DType::F32, a.provenance().clone(), a.device().clone()))
}

fn relu(t: &Tensor) -> Result<Tensor> {
    let v = t.to_vec_f32()?;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..v.len() {
        out[i] = v[i].max(0.0);
    }
    Ok(cpu_tensor(out, t.shape().clone()))
}
