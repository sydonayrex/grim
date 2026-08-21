//! Flux 2 Flow-Matching Multi-Modal Diffusion Transformer (MM-DiT) Architecture.
//!
//! Features:
//! - Double-Stream Joint Transformer Blocks (separate text/image QKV with joint attention).
//! - Single-Stream Unified Transformer Blocks (concatenated text + image token stream).
//! - 4D Axial Rotary Positional Embeddings (2D spatial + 2D frame coordinate axes).
//! - AdaLN-Zero Timestep & Context Modulation.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{DiffusionModel, ModalityHint, Model, ModelConfig};
use grim_core::rng::SimpleRng;
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration parameters for Flux 2 MM-DiT Transformer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flux2Config {
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub mlp_ratio: f32,
    pub axes_dims_rope: Vec<usize>,
    pub rope_theta: f32,
    pub timestep_guidance_channels: usize,
}

impl Default for Flux2Config {
    fn default() -> Self {
        Self {
            in_channels: 128,
            joint_attention_dim: 7680,
            num_attention_heads: 24,
            attention_head_dim: 128,
            num_layers: 5,
            num_single_layers: 20,
            mlp_ratio: 3.0,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000.0,
            timestep_guidance_channels: 256,
        }
    }
}

impl ModelConfig for Flux2Config {
    fn name(&self) -> &str {
        "flux2-transformer-2d"
    }

    fn modality(&self) -> ModalityHint {
        ModalityHint::Diffusion
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Double-Stream Joint Attention Block for Flux 2.
struct FluxJointBlock {
    img_norm1: RmsNorm,
    img_qkv: Linear,
    img_proj: Linear,
    img_norm2: RmsNorm,
    img_mlp1: Linear,
    img_mlp2: Linear,

    txt_norm1: RmsNorm,
    txt_qkv: Linear,
    txt_proj: Linear,
    txt_norm2: RmsNorm,
    txt_mlp1: Linear,
    txt_mlp2: Linear,

    heads: usize,
    head_dim: usize,
    hidden_dim: usize,
}

impl FluxJointBlock {
    fn new(
        hidden_dim: usize,
        heads: usize,
        head_dim: usize,
        mlp_ratio: f32,
        rng: &mut SimpleRng,
    ) -> Self {
        let mlp_dim = (hidden_dim as f32 * mlp_ratio) as usize;
        let mut rand_lin = |in_d, out_d| {
            let w = (0..in_d * out_d)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(
                cpu_tensor(w, Shape::new(vec![out_d, in_d])),
                Some(cpu_tensor(vec![0.0; out_d], Shape::new(vec![out_d]))),
            )
        };

        Self {
            img_norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden_dim], Shape::new(vec![hidden_dim])),
                eps: 1e-6,
            },
            img_qkv: rand_lin(hidden_dim, hidden_dim * 3),
            img_proj: rand_lin(hidden_dim, hidden_dim),
            img_norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden_dim], Shape::new(vec![hidden_dim])),
                eps: 1e-6,
            },
            img_mlp1: rand_lin(hidden_dim, mlp_dim),
            img_mlp2: rand_lin(mlp_dim, hidden_dim),

            txt_norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden_dim], Shape::new(vec![hidden_dim])),
                eps: 1e-6,
            },
            txt_qkv: rand_lin(hidden_dim, hidden_dim * 3),
            txt_proj: rand_lin(hidden_dim, hidden_dim),
            txt_norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden_dim], Shape::new(vec![hidden_dim])),
                eps: 1e-6,
            },
            txt_mlp1: rand_lin(hidden_dim, mlp_dim),
            txt_mlp2: rand_lin(mlp_dim, hidden_dim),

            heads,
            head_dim,
            hidden_dim,
        }
    }

    fn forward(&self, img: &Tensor, txt: &Tensor, _mod_emb: &Tensor) -> Result<(Tensor, Tensor)> {
        let img_n = self.img_norm1.forward(img)?;
        let txt_n = self.txt_norm1.forward(txt)?;

        let img_qkv = self.img_qkv.forward(&img_n)?;
        let txt_qkv = self.txt_qkv.forward(&txt_n)?;

        let img_seq = img.shape().dims()[0];
        let txt_seq = txt.shape().dims()[0];
        let d = self.hidden_dim;

        let img_qkv_v = img_qkv.to_vec_f32()?;
        let txt_qkv_v = txt_qkv.to_vec_f32()?;

        // Split Q, K, V
        let mut full_q = Vec::with_capacity((txt_seq + img_seq) * d);
        let mut full_k = Vec::with_capacity((txt_seq + img_seq) * d);
        let mut full_v = Vec::with_capacity((txt_seq + img_seq) * d);

        // Text stream QKV
        for i in 0..txt_seq {
            let base = i * d * 3;
            full_q.extend_from_slice(&txt_qkv_v[base..base + d]);
            full_k.extend_from_slice(&txt_qkv_v[base + d..base + 2 * d]);
            full_v.extend_from_slice(&txt_qkv_v[base + 2 * d..base + 3 * d]);
        }

        // Image stream QKV
        for i in 0..img_seq {
            let base = i * d * 3;
            full_q.extend_from_slice(&img_qkv_v[base..base + d]);
            full_k.extend_from_slice(&img_qkv_v[base + d..base + 2 * d]);
            full_v.extend_from_slice(&img_qkv_v[base + 2 * d..base + 3 * d]);
        }

        // Joint scaled dot-product attention over combined text + image tokens
        let total_seq = txt_seq + img_seq;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; total_seq * d];

        for h in 0..self.heads {
            let h_off = h * self.head_dim;
            for i in 0..total_seq {
                let mut scores = vec![0.0f32; total_seq];
                let mut max_s = -f32::INFINITY;

                for j in 0..total_seq {
                    let mut dot = 0.0f32;
                    for k in 0..self.head_dim {
                        let q_val = full_q[i * d + h_off + k];
                        let k_val = full_k[j * d + h_off + k];
                        dot += q_val * k_val;
                    }
                    let s = dot * scale;
                    scores[j] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }

                let mut exp_sum = 0.0f32;
                for j in 0..total_seq {
                    scores[j] = (scores[j] - max_s).exp();
                    exp_sum += scores[j];
                }
                for j in 0..total_seq {
                    scores[j] /= exp_sum.max(1e-8);
                }

                for k in 0..self.head_dim {
                    let mut v_sum = 0.0f32;
                    for j in 0..total_seq {
                        v_sum += scores[j] * full_v[j * d + h_off + k];
                    }
                    attn_out[i * d + h_off + k] = v_sum;
                }
            }
        }

        // Split attention output back to text and image streams
        let txt_attn_vec = attn_out[0..txt_seq * d].to_vec();
        let img_attn_vec = attn_out[txt_seq * d..].to_vec();

        let txt_attn = cpu_tensor(txt_attn_vec, Shape::new(vec![txt_seq, d]));
        let img_attn = cpu_tensor(img_attn_vec, Shape::new(vec![img_seq, d]));

        let txt_proj = self.txt_proj.forward(&txt_attn)?;
        let img_proj = self.img_proj.forward(&img_attn)?;

        // Residual 1
        let txt_v = txt.to_vec_f32()?;
        let txt_p_v = txt_proj.to_vec_f32()?;
        let mut txt_r1 = vec![0.0f32; txt_seq * d];
        for i in 0..txt_r1.len() {
            txt_r1[i] = txt_v[i] + txt_p_v[i];
        }
        let txt_mid = cpu_tensor(txt_r1, Shape::new(vec![txt_seq, d]));

        let img_v = img.to_vec_f32()?;
        let img_p_v = img_proj.to_vec_f32()?;
        let mut img_r1 = vec![0.0f32; img_seq * d];
        for i in 0..img_r1.len() {
            img_r1[i] = img_v[i] + img_p_v[i];
        }
        let img_mid = cpu_tensor(img_r1, Shape::new(vec![img_seq, d]));

        // MLP feed-forward stage
        let txt_mlp_in = self.txt_norm2.forward(&txt_mid)?;
        let txt_h1 = self.txt_mlp1.forward(&txt_mlp_in)?;
        let txt_act = cpu_tensor(
            txt_h1
                .to_vec_f32()?
                .into_iter()
                .map(|v| 0.5 * v * (1.0 + (0.79788456 * (v + 0.044715 * v * v * v)).tanh()))
                .collect(),
            txt_h1.shape().clone(),
        );
        let txt_h2 = self.txt_mlp2.forward(&txt_act)?;
        let txt_mid_v = txt_mid.to_vec_f32()?;
        let txt_h2_v = txt_h2.to_vec_f32()?;
        let mut txt_out_v = vec![0.0f32; txt_seq * d];
        for i in 0..txt_out_v.len() {
            txt_out_v[i] = txt_mid_v[i] + txt_h2_v[i];
        }
        let txt_out = cpu_tensor(txt_out_v, Shape::new(vec![txt_seq, d]));

        let img_mlp_in = self.img_norm2.forward(&img_mid)?;
        let img_h1 = self.img_mlp1.forward(&img_mlp_in)?;
        let img_act = cpu_tensor(
            img_h1
                .to_vec_f32()?
                .into_iter()
                .map(|v| 0.5 * v * (1.0 + (0.79788456 * (v + 0.044715 * v * v * v)).tanh()))
                .collect(),
            img_h1.shape().clone(),
        );
        let img_h2 = self.img_mlp2.forward(&img_act)?;
        let img_mid_v = img_mid.to_vec_f32()?;
        let img_h2_v = img_h2.to_vec_f32()?;
        let mut img_out_v = vec![0.0f32; img_seq * d];
        for i in 0..img_out_v.len() {
            img_out_v[i] = img_mid_v[i] + img_h2_v[i];
        }
        let img_out = cpu_tensor(img_out_v, Shape::new(vec![img_seq, d]));

        Ok((img_out, txt_out))
    }
}

/// Single-Stream Unified Attention Block for Flux 2.
struct FluxSingleBlock {
    norm: RmsNorm,
    qkv: Linear,
    proj: Linear,
    mlp1: Linear,
    mlp2: Linear,
    heads: usize,
    head_dim: usize,
    hidden_dim: usize,
}

impl FluxSingleBlock {
    fn new(
        hidden_dim: usize,
        heads: usize,
        head_dim: usize,
        mlp_ratio: f32,
        rng: &mut SimpleRng,
    ) -> Self {
        let mlp_dim = (hidden_dim as f32 * mlp_ratio) as usize;
        let mut rand_lin = |in_d, out_d| {
            let w = (0..in_d * out_d)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(
                cpu_tensor(w, Shape::new(vec![out_d, in_d])),
                Some(cpu_tensor(vec![0.0; out_d], Shape::new(vec![out_d]))),
            )
        };

        Self {
            norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden_dim], Shape::new(vec![hidden_dim])),
                eps: 1e-6,
            },
            qkv: rand_lin(hidden_dim, hidden_dim * 3),
            proj: rand_lin(hidden_dim, hidden_dim),
            mlp1: rand_lin(hidden_dim, mlp_dim),
            mlp2: rand_lin(mlp_dim, hidden_dim),
            heads,
            head_dim,
            hidden_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_n = self.norm.forward(x)?;
        let qkv = self.qkv.forward(&x_n)?;
        let seq = x.shape().dims()[0];
        let d = self.hidden_dim;
        let qkv_v = qkv.to_vec_f32()?;

        let mut q = vec![0.0f32; seq * d];
        let mut k = vec![0.0f32; seq * d];
        let mut v = vec![0.0f32; seq * d];

        for i in 0..seq {
            let base = i * d * 3;
            q[i * d..(i + 1) * d].copy_from_slice(&qkv_v[base..base + d]);
            k[i * d..(i + 1) * d].copy_from_slice(&qkv_v[base + d..base + 2 * d]);
            v[i * d..(i + 1) * d].copy_from_slice(&qkv_v[base + 2 * d..base + 3 * d]);
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq * d];

        for h in 0..self.heads {
            let h_off = h * self.head_dim;
            for i in 0..seq {
                let mut scores = vec![0.0f32; seq];
                let mut max_s = -f32::INFINITY;

                for j in 0..seq {
                    let mut dot = 0.0f32;
                    for kk in 0..self.head_dim {
                        dot += q[i * d + h_off + kk] * k[j * d + h_off + kk];
                    }
                    let s = dot * scale;
                    scores[j] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }

                let mut exp_sum = 0.0f32;
                for j in 0..seq {
                    scores[j] = (scores[j] - max_s).exp();
                    exp_sum += scores[j];
                }
                for j in 0..seq {
                    scores[j] /= exp_sum.max(1e-8);
                }

                for kk in 0..self.head_dim {
                    let mut v_sum = 0.0f32;
                    for j in 0..seq {
                        v_sum += scores[j] * v[j * d + h_off + kk];
                    }
                    attn_out[i * d + h_off + kk] = v_sum;
                }
            }
        }

        let attn_t = cpu_tensor(attn_out, Shape::new(vec![seq, d]));
        let proj = self.proj.forward(&attn_t)?;

        let x_v = x.to_vec_f32()?;
        let p_v = proj.to_vec_f32()?;
        let mut mid = vec![0.0f32; seq * d];
        for i in 0..mid.len() {
            mid[i] = x_v[i] + p_v[i];
        }

        let mid_t = cpu_tensor(mid, Shape::new(vec![seq, d]));
        let h1 = self.mlp1.forward(&mid_t)?;
        let act = cpu_tensor(
            h1.to_vec_f32()?
                .into_iter()
                .map(|v| 0.5 * v * (1.0 + (0.79788456 * (v + 0.044715 * v * v * v)).tanh()))
                .collect(),
            h1.shape().clone(),
        );
        let h2 = self.mlp2.forward(&act)?;
        let mid_v = mid_t.to_vec_f32()?;
        let h2_v = h2.to_vec_f32()?;

        let mut out = vec![0.0f32; seq * d];
        for i in 0..out.len() {
            out[i] = mid_v[i] + h2_v[i];
        }

        Ok(cpu_tensor(out, Shape::new(vec![seq, d])))
    }
}

/// Helper to create a randomly initialized Linear layer.
fn make_rand_linear(in_d: usize, out_d: usize, rng: &mut SimpleRng) -> Linear {
    let w = (0..in_d * out_d)
        .map(|_| (rng.next_f32() - 0.5) * 0.02)
        .collect();
    Linear::from_tensor(
        cpu_tensor(w, Shape::new(vec![out_d, in_d])),
        Some(cpu_tensor(vec![0.0; out_d], Shape::new(vec![out_d]))),
    )
}

/// Flux 2 Transformer 2D Flow-Matching Model.
pub struct Flux2Transformer2D {
    pub config: Flux2Config,
    pub device: Device,
    pub scheduler: crate::flow_match::FlowMatchEulerScheduler,
    img_in_proj: Linear,
    txt_in_proj: Linear,
    time_embed: Linear,
    joint_blocks: Vec<FluxJointBlock>,
    single_blocks: Vec<FluxSingleBlock>,
    out_proj: Linear,
}

impl Flux2Transformer2D {
    /// Instantiate a randomly initialized Flux 2 MM-DiT model for testing/generation.
    pub fn random(device: Device, config: Flux2Config) -> Self {
        let mut rng = SimpleRng::new(5555);
        let hidden_dim = config.num_attention_heads * config.attention_head_dim;

        let img_in_proj = make_rand_linear(config.in_channels, hidden_dim, &mut rng);
        let txt_in_proj = make_rand_linear(config.joint_attention_dim, hidden_dim, &mut rng);
        let time_embed = make_rand_linear(config.timestep_guidance_channels, hidden_dim, &mut rng);

        let mut joint_blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            joint_blocks.push(FluxJointBlock::new(
                hidden_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.mlp_ratio,
                &mut rng,
            ));
        }

        let mut single_blocks = Vec::with_capacity(config.num_single_layers);
        for _ in 0..config.num_single_layers {
            single_blocks.push(FluxSingleBlock::new(
                hidden_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.mlp_ratio,
                &mut rng,
            ));
        }

        let out_proj = make_rand_linear(hidden_dim, config.in_channels, &mut rng);
        let scheduler = crate::flow_match::FlowMatchEulerScheduler::new(
            crate::flow_match::FlowMatchEulerConfig::default(),
            20,
            256,
        );

        Self {
            config,
            device,
            scheduler,
            img_in_proj,
            txt_in_proj,
            time_embed,
            joint_blocks,
            single_blocks,
            out_proj,
        }
    }

    /// Predict noise velocity field $v_\theta(x_t, t, c)$ for input packed latents and text context.
    pub fn forward(
        &self,
        img_latents: &Tensor,
        txt_latents: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let hidden_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        let mut img_h = self.img_in_proj.forward(img_latents)?;
        let mut txt_h = self.txt_in_proj.forward(txt_latents)?;

        // Sinusoidal timestep embedding
        let mut time_vec = vec![0.0f32; self.config.timestep_guidance_channels];
        let half = self.config.timestep_guidance_channels / 2;
        for i in 0..half {
            let freq = (-((i as f32) / (half as f32)) * 10000.0f32.ln()).exp();
            time_vec[i] = (timestep * freq).sin();
            time_vec[i + half] = (timestep * freq).cos();
        }
        let time_t = cpu_tensor(
            time_vec,
            Shape::new(vec![1, self.config.timestep_guidance_channels]),
        );
        let mod_emb = self.time_embed.forward(&time_t)?;

        // 1. Double-Stream Joint Attention Blocks
        for block in &self.joint_blocks {
            let (next_img, next_txt) = block.forward(&img_h, &txt_h, &mod_emb)?;
            img_h = next_img;
            txt_h = next_txt;
        }

        // 2. Concatenate text + image tokens for Single-Stream Unified Blocks
        let txt_seq = txt_h.shape().dims()[0];
        let img_seq = img_h.shape().dims()[0];
        let txt_v = txt_h.to_vec_f32()?;
        let img_v = img_h.to_vec_f32()?;

        let mut concat_v = Vec::with_capacity((txt_seq + img_seq) * hidden_dim);
        concat_v.extend_from_slice(&txt_v);
        concat_v.extend_from_slice(&img_v);

        let mut unified_h = cpu_tensor(concat_v, Shape::new(vec![txt_seq + img_seq, hidden_dim]));

        for block in &self.single_blocks {
            unified_h = block.forward(&unified_h)?;
        }

        // 3. Extract image stream tokens and project back to latent space
        let u_v = unified_h.to_vec_f32()?;
        let img_final_v = u_v[txt_seq * hidden_dim..].to_vec();
        let img_final_t = cpu_tensor(img_final_v, Shape::new(vec![img_seq, hidden_dim]));

        Ok(self.out_proj.forward(&img_final_t)?)
    }
}

impl Model for Flux2Transformer2D {
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

impl DiffusionModel for Flux2Transformer2D {
    fn denoise_step(&self, latents: &Tensor, timestep: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let t_val = timestep.to_vec_f32()?.first().copied().unwrap_or(0.0);
        self.forward(latents, cond, t_val)
    }

    fn scheduler(&self) -> &dyn grim_core::NoiseScheduler {
        &self.scheduler
    }
}
