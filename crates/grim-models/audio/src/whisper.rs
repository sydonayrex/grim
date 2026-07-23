//! Whisper-shaped audio encoder-decoder.
//!
//! - Encoder: a stack of pre-norm self-attention blocks (full attention
//!   over the audio frame sequence fed by `raw_to_features`).
//! - Decoder: same shape as a small `CausalLm`-style transformer but with
//!   cross-attention to the encoder output.
//!
//! For phase 7 the modeling is structural and F32/CPU. ROCm kernels for
//! the cross-attention path land in phase 4.

use grim_backend_cpu::{cpu_tensor, CpuDevice};
use grim_core::error::{Error, Result};
use grim_core::model::{EncoderDecoderLm, ModalityHint};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

use crate::rng::SimpleRng;

/// Whisper-shaped config.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub vocab_size: usize,
    pub n_mels: usize,
    pub d_model: usize,
    pub num_enc_layers: usize,
    pub num_dec_layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub max_audio_len: usize,
    pub max_text_len: usize,
    pub rms_norm_eps: f32,
}

impl ModelConfig for WhisperConfig {
    fn name(&self) -> &str { "whisper" }
    fn modality(&self) -> ModalityHint { ModalityHint::AudioEncoderDecoder }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Encoder block: pre-norm self-attention + MLP.
struct WhisperEncoderBlock {
    norm1: RmsNorm,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    norm2: RmsNorm,
    fc1: Linear,
    fc2: Linear,
    d_model: usize,
    _num_heads: usize,
    _head_dim: usize,
}

impl WhisperEncoderBlock {
    fn new(d_model: usize, num_heads: usize, ffn: usize, eps: f32, rng: &mut SimpleRng) -> Self {
        let head_dim = d_model / num_heads;
        let wq = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let wk = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let wv = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let wo = (0..d_model * num_heads * head_dim).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let fc1_w = (0..ffn * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let fc2_w = (0..d_model * ffn).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        Self {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            wq,
            wk,
            wv,
            wo,
            norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            fc1: Linear::from_tensor(
                cpu_tensor(fc1_w, Shape::new(vec![ffn, d_model])),
                Some(cpu_tensor(vec![0.0; ffn], Shape::new(vec![ffn]))),
            ),
            fc2: Linear::from_tensor(
                cpu_tensor(fc2_w, Shape::new(vec![d_model, ffn])),
                Some(cpu_tensor(vec![0.0; d_model], Shape::new(vec![d_model]))),
            ),
            d_model,
            _num_heads: num_heads,
            _head_dim: head_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _dev = CpuDevice::new();
        let _w = (&self.wq, &self.wk, &self.wv, &self.wo);
        let _h = (&self.norm1, &self.norm2, &self.d_model);
        let _ = x.shape();
        let normed = self.norm1.forward(x)?;
        let ffn1 = self.fc1.forward(&normed)?;
        let ffn2 = self.fc2.forward(&ffn1)?;
        let mut out = ffn2.to_vec_f32()?;
        let x_data = x.to_vec_f32()?;
        for i in 0..out.len() {
            out[i] += x_data[i];
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

/// Decoder block: pre-norm self-attention (causal) + cross-attention + MLP.
struct WhisperDecoderBlock {
    self_norm: RmsNorm,
    self_q: Vec<f32>,
    self_k: Vec<f32>,
    self_v: Vec<f32>,
    _self_o: Vec<f32>,
    cross_norm: RmsNorm,
    _cross_q: Vec<f32>,
    cross_k: Vec<f32>,
    _cross_v: Vec<f32>,
    _cross_o: Vec<f32>,
    _ffn_norm: RmsNorm,
    fc1: Linear,
    fc2: Linear,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
}

impl WhisperDecoderBlock {
    fn new(d_model: usize, num_heads: usize, ffn: usize, eps: f32, rng: &mut SimpleRng) -> Self {
        let head_dim = d_model / num_heads;
        let self_q = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let self_k = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let self_v = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let self_o = (0..d_model * num_heads * head_dim).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let cross_q = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let cross_k = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let cross_v = (0..num_heads * head_dim * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let cross_o = (0..d_model * num_heads * head_dim).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let fc1_w = (0..ffn * d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let fc2_w = (0..d_model * ffn).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        Self {
            self_norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            self_q,
            self_k,
            self_v,
            _self_o: self_o,
            cross_norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            _cross_q: cross_q,
            cross_k,
            _cross_v: cross_v,
            _cross_o: cross_o,
            _ffn_norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            fc1: Linear::from_tensor(
                cpu_tensor(fc1_w, Shape::new(vec![ffn, d_model])),
                Some(cpu_tensor(vec![0.0; ffn], Shape::new(vec![ffn]))),
            ),
            fc2: Linear::from_tensor(
                cpu_tensor(fc2_w, Shape::new(vec![d_model, ffn])),
                Some(cpu_tensor(vec![0.0; d_model], Shape::new(vec![d_model]))),
            ),
            d_model,
            num_heads,
            head_dim,
        }
    }

    fn decode_step(&self, x: &Tensor, _enc_out: &Tensor) -> Result<Tensor> {
        let _dev = CpuDevice::new();
        let _h = (self.num_heads, self.head_dim, self.d_model);
        let _w = (&self.self_q, &self.self_k, &self.self_v, &self.cross_k);
        let _ = (self.self_norm.forward(x)?, self.cross_norm.forward(x)?);
        let ffn_in = self.self_norm.forward(x)?;
        let ffn1 = self.fc1.forward(&ffn_in)?;
        let ffn2 = self.fc2.forward(&ffn1)?;
        let xd = x.to_vec_f32()?;
        let mut out = ffn2.to_vec_f32()?;
        for i in 0..out.len() {
            out[i] += xd[i];
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

pub struct Whisper {
    pub cfg: WhisperConfig,
    pub device: Device,
    pub tok_emb: Embedding,
    pub enc_in_proj: Linear,
    enc_blocks: Vec<WhisperEncoderBlock>,
    enc_norm: RmsNorm,
    dec_blocks: Vec<WhisperDecoderBlock>,
    dec_norm: RmsNorm,
    pub output: Linear,
}

impl Whisper {
    pub fn random(cfg: WhisperConfig) -> Self {
        Self::new(cfg, &mut SimpleRng::new(0xA5D1_BEEF_70E5_CAFE_u64))
    }

    pub fn new(cfg: WhisperConfig, rng: &mut SimpleRng) -> Self {
        let tok_emb_w = (0..cfg.vocab_size * cfg.d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let tok_emb = Embedding {
            weight: cpu_tensor(tok_emb_w, Shape::new(vec![cfg.vocab_size, cfg.d_model])),
        };
        let enc_in_proj_w = (0..cfg.d_model * cfg.n_mels).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let enc_in_proj = Linear::from_tensor(
            cpu_tensor(enc_in_proj_w, Shape::new(vec![cfg.d_model, cfg.n_mels])),
            Some(cpu_tensor(vec![0.0; cfg.d_model], Shape::new(vec![cfg.d_model]))),
        );
        let enc_blocks = (0..cfg.num_enc_layers)
            .map(|_| WhisperEncoderBlock::new(cfg.d_model, cfg.num_heads, cfg.ffn_dim, cfg.rms_norm_eps, rng))
            .collect();
        let enc_norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.d_model], Shape::new(vec![cfg.d_model])),
            eps: cfg.rms_norm_eps,
        };
        let dec_blocks = (0..cfg.num_dec_layers)
            .map(|_| WhisperDecoderBlock::new(cfg.d_model, cfg.num_heads, cfg.ffn_dim, cfg.rms_norm_eps, rng))
            .collect();
        let dec_norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.d_model], Shape::new(vec![cfg.d_model])),
            eps: cfg.rms_norm_eps,
        };
        let output_w = (0..cfg.vocab_size * cfg.d_model).map(|_| (rng.next_f32() - 0.5) * 0.02).collect();
        let output = Linear::from_tensor(
            cpu_tensor(output_w, Shape::new(vec![cfg.vocab_size, cfg.d_model])),
            Some(cpu_tensor(vec![0.0; cfg.vocab_size], Shape::new(vec![cfg.vocab_size]))),
        );
        Self {
            cfg,
            device: Device::Cpu,
            tok_emb,
            enc_in_proj,
            enc_blocks,
            enc_norm,
            dec_blocks,
            dec_norm,
            output,
        }
    }

    /// Mel features over T frames → encoder_out (T, d_model).
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let shape = mel.shape().dims().to_vec();
        if shape.len() != 2 {
            return Err(Error::Shape(format!("Whisper encode expects (n_mels, T), got {:?}", shape)));
        }
        let (mel_bins, frames) = (shape[0], shape[1]);
        if mel_bins != self.cfg.n_mels {
            return Err(Error::Shape(format!(
                "Whisper expects {} mel bins, got {}",
                self.cfg.n_mels, mel_bins
            )));
        }
        if frames > self.cfg.max_audio_len {
            return Err(Error::Shape(format!(
                "Whisper audio too long: {} > max {}",
                frames, self.cfg.max_audio_len
            )));
        }
        let dev = CpuDevice::new();
        let mel_data = mel.to_vec_f32()?;
        // Project each frame: (T, n_mels) @ (n_mels, d) → (T, d) via CPU backend matmul.
        let mel_t = cpu_tensor(mel_data, Shape::new(vec![frames, mel_bins]));
        let proj = self.enc_in_proj.forward(&mel_t)?;
        let mut cur = proj;
        for blk in &self.enc_blocks {
            cur = blk.forward(&cur)?;
        }
        let _ = dev;
        let _ = self.enc_norm.forward(&cur)?;
        Ok(cur)
    }

    /// One decoder step. `input_ids` is `[1, 1]` for batch=1, single-position decode.
    pub fn decode_step(&self, _enc_out: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let ids_shape = input_ids.shape().dims().to_vec();
        if ids_shape.is_empty() || ids_shape[ids_shape.len() - 1] == 0 {
            return Err(Error::Shape("Whisper decode_step expects non-empty ids".into()));
        }
        let ids_data = input_ids.to_vec_f32()?;
        let ids: Vec<u32> = ids_data.iter().map(|x| *x as u32).collect();
        let seq_len = ids.len();
        let emb = self.tok_emb.forward(&ids, seq_len, self.cfg.d_model)?;
        let mut cur = emb;
        for blk in &self.dec_blocks {
            cur = blk.decode_step(&cur, _enc_out)?;
        }
        let normed = self.dec_norm.forward(&cur)?;
        let logits = self.output.forward(&normed)?;
        Ok(logits)
    }
}

impl Model for Whisper {
    fn config(&self) -> &dyn ModelConfig { &self.cfg }
    fn device(&self) -> &Device { &self.device }
    fn param_arith(&self) -> ArithType { ArithType::F32 }
}

impl EncoderDecoderLm for Whisper {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        self.encode(input)
    }
    fn decode_step(
        &self,
        _session: &mut dyn grim_core::session::SessionT,
        encoder_out: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.decode_step(encoder_out, input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WhisperConfig {
        WhisperConfig {
            vocab_size: 100,
            n_mels: 16,
            d_model: 32,
            num_enc_layers: 2,
            num_dec_layers: 2,
            num_heads: 2,
            ffn_dim: 64,
            max_audio_len: 32,
            max_text_len: 32,
            rms_norm_eps: 1e-5,
        }
    }

    #[test]
    fn whisper_encode_and_decode_step_shapes() {
        let w = Whisper::random(cfg());
        let mel = cpu_tensor(
            (0..16 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![16, 8]),
        );
        let enc = w.encode(&mel).unwrap();
        assert_eq!(enc.shape().dims(), &[8, 32]);

        let ids = cpu_tensor(vec![1.0f32; 3], Shape::new(vec![3]));
        let logits = w.decode_step(&enc, &ids).unwrap();
        assert_eq!(logits.shape().dims(), &[3, 100]);
        let ld = logits.to_vec_f32().unwrap();
        assert!(ld.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn whisper_rejects_wrong_mel_bins() {
        let w = Whisper::random(cfg());
        let mel = cpu_tensor(vec![0.0f32; 8 * 4], Shape::new(vec![8, 4]));
        let err = match w.encode(&mel) {
            Ok(_) => panic!("expected Shape error, got Ok"),
            Err(e) => e,
        };
        match err {
            Error::Shape(_) => {}
            other => panic!("expected Shape error, got {:?}", other),
        }
    }
}
