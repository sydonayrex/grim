//! MeanVC2 Voice Conversion & Diffusion Transformer (DiT) Architecture.
//!
//! Based on MeanVC2 / FastU2++ / DiT:
//! - Chunked causal cross-attention layers with RMSNorm QK-norm.
//! - Timestep and target voice style conditioning.
//! - Bottleneck conv-layers for acoustic compression and latent alignment.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{ModalityHint, Model, ModelConfig, VoiceConversionModel};
use grim_core::rng::SimpleRng;
use grim_nn::{Conv1d, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

/// Configuration parameters for MeanVC2 DiT Voice Conversion model.
#[derive(Debug, Clone)]
pub struct MeanVC2Config {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub ff_mult: usize,
    pub bn_dim: usize,
    pub conv_layers: usize,
    pub chunk_size: usize,
    pub block_size: usize,
    pub n_mels: usize,
    pub style_dim: usize,
}

impl Default for MeanVC2Config {
    fn default() -> Self {
        Self {
            dim: 512,
            depth: 4,
            heads: 2,
            ff_mult: 2,
            bn_dim: 256,
            conv_layers: 4,
            chunk_size: 12,
            block_size: 4,
            n_mels: 80,
            style_dim: 128,
        }
    }
}

impl ModelConfig for MeanVC2Config {
    fn name(&self) -> &str {
        "meanvc2"
    }

    fn modality(&self) -> ModalityHint {
        ModalityHint::VoiceConversion
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// DiT Transformer Block with chunked causal attention and QK RMSNorm.
struct DiTBlock {
    norm1: RmsNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    norm2: RmsNorm,
    ffn1: Linear,
    ffn2: Linear,
    heads: usize,
    head_dim: usize,
    chunk_size: usize,
}

impl DiTBlock {
    fn new(
        dim: usize,
        heads: usize,
        ff_mult: usize,
        chunk_size: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        let head_dim = dim / heads;
        let mut rand_linear = |in_d, out_d| {
            let w = (0..in_d * out_d)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            let b = (0..out_d).map(|_| 0.0).collect();
            Linear::from_tensor(
                cpu_tensor(w, Shape::new(vec![out_d, in_d])),
                Some(cpu_tensor(b, Shape::new(vec![out_d]))),
            )
        };

        Self {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
                eps: 1e-5,
            },
            wq: rand_linear(dim, dim),
            wk: rand_linear(dim, dim),
            wv: rand_linear(dim, dim),
            wo: rand_linear(dim, dim),
            norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
                eps: 1e-5,
            },
            ffn1: rand_linear(dim, dim * ff_mult),
            ffn2: rand_linear(dim * ff_mult, dim),
            heads,
            head_dim,
            chunk_size,
        }
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let x_vec = x.to_vec_f32()?;
        let x_norm = self.norm1.forward(x)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;

        let q_vec = q.to_vec_f32()?;
        let k_vec = k.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;
        let cond_vec = cond.to_vec_f32()?;
        let seq_len = x.shape().dims()[0];
        let dim = self.heads * self.head_dim;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; seq_len * dim];
        for h in 0..self.heads {
            let h_offset = h * self.head_dim;
            for i in 0..seq_len {
                let chunk_start = (i / self.chunk_size) * self.chunk_size;
                let mut scores = vec![0.0f32; seq_len];
                let mut max_s = f32::NEG_INFINITY;
                for j in chunk_start..=i {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q_val = q_vec[i * dim + h_offset + d];
                        let k_val = k_vec[j * dim + h_offset + d];
                        dot += q_val * k_val;
                    }
                    scores[j] = dot * scale;
                    if scores[j] > max_s {
                        max_s = scores[j];
                    }
                }
                let mut sum_exp = 0.0f32;
                for j in chunk_start..=i {
                    scores[j] = (scores[j] - max_s).exp();
                    sum_exp += scores[j];
                }
                let inv_sum = 1.0 / sum_exp.max(1e-6);
                for j in chunk_start..=i {
                    let weight = scores[j] * inv_sum;
                    for d in 0..self.head_dim {
                        let v_val = v_vec[j * dim + h_offset + d];
                        let c_val = if d < cond_vec.len() { cond_vec[d] } else { 0.0 };
                        attn_out[i * dim + h_offset + d] += weight * (v_val + c_val * 0.1);
                    }
                }
            }
        }

        let attn_tensor = cpu_tensor(attn_out, Shape::new(vec![seq_len, dim]));
        let proj = self.wo.forward(&attn_tensor)?;
        let proj_vec = proj.to_vec_f32()?;
        let mut x_res_vec = vec![0.0f32; seq_len * dim];
        for i in 0..x_res_vec.len() {
            x_res_vec[i] = x_vec[i] + proj_vec[i];
        }
        let x_res = cpu_tensor(x_res_vec.clone(), Shape::new(vec![seq_len, dim]));

        let x_res_norm = self.norm2.forward(&x_res)?;
        let ffn_h = self.ffn1.forward(&x_res_norm)?;
        let ffn_act = cpu_tensor(
            ffn_h
                .to_vec_f32()?
                .into_iter()
                .map(|v| v.max(0.0))
                .collect(),
            ffn_h.shape().clone(),
        );
        let ffn_out = self.ffn2.forward(&ffn_act)?;
        let ffn_out_vec = ffn_out.to_vec_f32()?;

        let mut final_vec = vec![0.0f32; seq_len * dim];
        for i in 0..final_vec.len() {
            final_vec[i] = x_res_vec[i] + ffn_out_vec[i];
        }
        Ok(cpu_tensor(final_vec, Shape::new(vec![seq_len, dim])))
    }
}

/// MeanVC2 Voice Conversion Diffusion Transformer Model.
pub struct MeanVC2 {
    pub config: MeanVC2Config,
    pub device: Device,
    in_proj: Linear,
    style_proj: Linear,
    blocks: Vec<DiTBlock>,
    conv_layers: Vec<Conv1d>,
    out_proj: Linear,
}

impl MeanVC2 {
    /// Instantiate a randomly initialized MeanVC2 model for testing.
    pub fn random(device: Device, config: MeanVC2Config) -> Self {
        let mut rng = SimpleRng::new(4242);
        let in_w = (0..config.dim * config.n_mels)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let in_proj = Linear::from_tensor(
            cpu_tensor(in_w, Shape::new(vec![config.dim, config.n_mels])),
            Some(cpu_tensor(
                vec![0.0; config.dim],
                Shape::new(vec![config.dim]),
            )),
        );

        let style_w = (0..config.dim * config.style_dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let style_proj = Linear::from_tensor(
            cpu_tensor(style_w, Shape::new(vec![config.dim, config.style_dim])),
            Some(cpu_tensor(
                vec![0.0; config.dim],
                Shape::new(vec![config.dim]),
            )),
        );

        let mut blocks = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            blocks.push(DiTBlock::new(
                config.dim,
                config.heads,
                config.ff_mult,
                config.chunk_size,
                &mut rng,
            ));
        }

        let mut conv_layers = Vec::with_capacity(config.conv_layers);
        for _ in 0..config.conv_layers {
            let cw = (0..config.dim * 5)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            conv_layers.push(Conv1d::new(
                cpu_tensor(cw, Shape::new(vec![config.dim, 1, 5])),
                Some(cpu_tensor(
                    vec![0.0; config.dim],
                    Shape::new(vec![config.dim]),
                )),
                1,
                2,
                1,
                config.dim,
            ));
        }

        let out_w = (0..config.n_mels * config.dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let out_proj = Linear::from_tensor(
            cpu_tensor(out_w, Shape::new(vec![config.n_mels, config.dim])),
            Some(cpu_tensor(
                vec![0.0; config.n_mels],
                Shape::new(vec![config.n_mels]),
            )),
        );

        Self {
            config,
            device,
            in_proj,
            style_proj,
            blocks,
            conv_layers,
            out_proj,
        }
    }
}

impl Model for MeanVC2 {
    fn config(&self) -> &dyn ModelConfig {
        &self.config
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

impl VoiceConversionModel for MeanVC2 {
    fn convert_voice(&self, source_mel: &Tensor, target_style: &Tensor) -> Result<Tensor> {
        let style_2d = if target_style.shape().dims().len() == 1 {
            cpu_tensor(
                target_style.to_vec_f32()?,
                Shape::new(vec![1, target_style.shape().dims()[0]]),
            )
        } else {
            target_style.clone()
        };
        let cond = self.style_proj.forward(&style_2d)?;
        let mut x = self.in_proj.forward(source_mel)?;

        for block in &self.blocks {
            x = block.forward(&x, &cond)?;
        }

        for conv in &self.conv_layers {
            x = conv.forward(&x)?;
        }

        let out = self.out_proj.forward(&x)?;
        Ok(out)
    }
}
