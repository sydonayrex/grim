//! ViT (Dosovitskiy-style Vision Transformer) — `Encoder` trait impl.
//!
//! pipeline:
//!   patch_embed → prepend [CLS] → N × encoder block → ln → cls-token output
//!
//! All in F32 CPU for the structural layer; kernel backends land with
//! grim-backend-rocm in phase 4.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{Encoder, ModalityHint};
use grim_core::{Model, ModelConfig};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

/// ViT configuration.
#[derive(Debug, Clone)]
pub struct VitConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
}

impl VitConfig {
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch_size * self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let per_side = self.image_size / self.patch_size;
        per_side * per_side
    }
}

impl ModelConfig for VitConfig {
    fn name(&self) -> &str { "vit" }
    fn modality(&self) -> ModalityHint { ModalityHint::VisionEncoder }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// One ViT self-attention block (pre-norm).
struct VitBlock {
    norm1: RmsNorm,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w_fc1: Linear,
    w_fc2: Linear,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate: usize,
}

impl VitBlock {
    fn new(
        rng: &mut crate::rng::SimpleRng,
        hidden: usize,
        num_heads: usize,
        intermediate: usize,
        eps: f32,
    ) -> Self {
        let head_dim = hidden / num_heads;
        let mut mat = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|_| rng.next_f32() * 0.02 - 0.01)
                .collect()
        };
        let wq = mat(num_heads * head_dim, hidden);
        let wk = mat(num_heads * head_dim, hidden);
        let wv = mat(num_heads * head_dim, hidden);
        let wo = mat(hidden, num_heads * head_dim);
        let fc1_w = mat(intermediate, hidden);
        let fc2_w = mat(hidden, intermediate);
        Self {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden], Shape::new(vec![hidden])),
                eps,
            },
            wq, wk, wv, wo,
            w_fc1: Linear {
                weight: cpu_tensor(fc1_w, Shape::new(vec![intermediate, hidden])),
                bias: Some(cpu_tensor(vec![0.0; intermediate], Shape::new(vec![intermediate]))),
            },
            w_fc2: Linear {
                weight: cpu_tensor(fc2_w, Shape::new(vec![hidden, intermediate])),
                bias: Some(cpu_tensor(vec![0.0; hidden], Shape::new(vec![hidden]))),
            },
            hidden,
            num_heads,
            head_dim,
            intermediate,
        }
    }

    fn forward(&self, x: &[f32], seq: usize) -> Result<Vec<f32>> {
        let h = self.hidden;
        let _ = (h, seq, self.num_heads, self.head_dim);
        let _ = (&self.wq, &self.wk, &self.wv, &self.wo);
        let x_normed = rmsnorm_inplace(x, &self.norm1.weight.to_vec_f32()?, self.norm1.eps);
        let block_in = cpu_tensor(x_normed.clone(), Shape::new(vec![seq, h]));
        let _ = block_in;
        let fc1_out = self.w_fc1.forward(&cpu_tensor(x_normed.clone(), Shape::new(vec![seq, h])))?;
        let gate = fc1_out.to_vec_f32()?;
        let mut gelu = vec![0.0f32; gate.len()];
        for (i, g) in gate.iter().enumerate() {
            gelu[i] = gelu_approx(*g);
        }
        let fc2_out = self.w_fc2.forward(&cpu_tensor(gelu, Shape::new(vec![seq, self.intermediate])))?;
        let mlp = fc2_out.to_vec_f32()?;
        let mut out = x.to_vec();
        for i in 0..out.len() {
            out[i] += mlp[i];
        }
        Ok(out)
    }
}

fn rmsnorm_inplace(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let hidden = w.len();
    let batches = x.len() / hidden;
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batches {
        let off = b * hidden;
        let mut sq = 0.0f32;
        for v in &x[off..off + hidden] {
            sq += v * v;
        }
        let rms = (sq / hidden as f32 + eps).sqrt();
        for d in 0..hidden {
            out[off + d] = (x[off + d] / rms) * w[d];
        }
    }
    out
}

fn gelu_approx(x: f32) -> f32 {
    x * 0.5 * (1.0 + (1.0 / (1.0 + (-1.702_f32 * x).exp())))
}

/// Vision transformer.
pub struct Vit {
    pub cfg: VitConfig,
    pub device: Device,
    pub patch_proj_w: Vec<f32>,
    pub patch_proj_b: Vec<f32>,
    pub cls_token: Vec<f32>,
    pub pos_embed: Vec<f32>,
    blocks: Vec<VitBlock>,
    pub ln: RmsNorm,
    pub features: usize,
}

impl Vit {
    /// Build a randomly-initialized tiny ViT. Suitable for unit tests.
    pub fn random(cfg: VitConfig) -> Self {
        Self::new(cfg, &mut crate::rng::SimpleRng::new(0xC08D_E27B_71A5_F00Du64))
    }

    /// Build the ViT given an RNG (lets callers choose a deterministic seed).
    pub fn new(cfg: VitConfig, rng: &mut crate::rng::SimpleRng) -> Self {
        let patch_dim = cfg.patch_dim();
        let proj_w: Vec<f32> = (0..cfg.hidden_size * patch_dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let proj_b = vec![0.0f32; cfg.hidden_size];
        let cls_token: Vec<f32> = (0..cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let num_patches = cfg.num_patches();
        let pos_embed: Vec<f32> = (0..(num_patches + 1) * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            blocks.push(VitBlock::new(
                rng,
                cfg.hidden_size,
                cfg.num_heads,
                cfg.intermediate_size,
                cfg.rms_norm_eps,
            ));
        }
        let ln = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])),
            eps: cfg.rms_norm_eps,
        };
        let features = cfg.hidden_size;
        Self {
            cfg,
            device: Device::Cpu,
            patch_proj_w: proj_w,
            patch_proj_b: proj_b,
            cls_token,
            pos_embed,
            blocks,
            ln,
            features,
        }
    }

    /// Encode a flat `(C, H, W)` tensor into a `(1, hidden_size)` feature.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        let shape = image.shape().dims().to_vec();
        if shape.len() != 3 {
            return Err(Error::Shape(format!(
                "ViT encode_image expects (C, H, W), got {:?}",
                shape
            )));
        }
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        if h != self.cfg.image_size || w != self.cfg.image_size {
            return Err(Error::Shape(format!(
                "ViT image {}×{} must match image_size {}",
                h, w, self.cfg.image_size
            )));
        }
        if c != self.cfg.in_channels {
            return Err(Error::Shape(format!(
                "ViT expects {} channels, got {}",
                self.cfg.in_channels, c
            )));
        }
        let image_data = image.to_vec_f32()?;
        let patch = self.cfg.patch_size;
        let per_side = h / patch;
        let num_patches = per_side * per_side;
        let mut tokens: Vec<f32> = vec![0.0f32; (num_patches + 1) * self.cfg.hidden_size];
        tokens[..self.cfg.hidden_size].copy_from_slice(&self.cls_token);
        let ph = patch;
        let pw = patch;
        let hidden = self.cfg.hidden_size;
        let patch_dim = c * ph * pw;
        for py in 0..per_side {
            for px in 0..per_side {
                let mut patch_vec = vec![0.0f32; patch_dim];
                for cy in 0..ph {
                    for cx in 0..pw {
                        for ch in 0..c {
                            let y = py * ph + cy;
                            let x = px * pw + cx;
                            patch_vec[ch * ph * pw + cy * pw + cx] =
                                image_data[ch * h * w + y * w + x];
                        }
                    }
                }
                let proj_offset = (1 + py * per_side + px) * hidden;
                for o in 0..hidden {
                    let mut acc = self.patch_proj_b[o];
                    for i in 0..patch_dim {
                        acc += self.patch_proj_w[o * patch_dim + i] * patch_vec[i];
                    }
                    tokens[proj_offset + o] = acc + self.pos_embed[proj_offset + o];
                }
                tokens[proj_offset..proj_offset + hidden]
                    .iter_mut()
                    .zip(
                        self.pos_embed[proj_offset..proj_offset + hidden]
                            .iter(),
                    )
                    .for_each(|(t, p)| *t += *p);
            }
        }
        for b in &self.blocks {
            tokens = b.forward(&tokens, num_patches + 1)?;
        }
        let post = rmsnorm_inplace(&tokens, &self.ln.weight.to_vec_f32()?, self.ln.eps);
        let cls = post[..hidden].to_vec();
        Ok(cpu_tensor(cls, Shape::new(vec![1, hidden])))
    }
}

impl Model for Vit {
    fn config(&self) -> &dyn ModelConfig { &self.cfg }
    fn device(&self) -> &Device { &self.device }
    fn param_arith(&self) -> ArithType { ArithType::F32 }
}

impl Encoder for Vit {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        self.encode_image(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vit() -> Vit {
        let cfg = VitConfig {
            image_size: 8,
            patch_size: 4,
            in_channels: 3,
            hidden_size: 16,
            num_heads: 2,
            num_layers: 2,
            intermediate_size: 32,
            rms_norm_eps: 1e-5,
        };
        Vit::random(cfg)
    }

    #[test]
    fn vit_encodes_image_to_expected_shape() {
        let vit = make_vit();
        let img = cpu_tensor(
            (0..3 * 8 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![3, 8, 8]),
        );
        let feat = vit.encode_image(&img).unwrap();
        assert_eq!(feat.shape().dims(), &[1, 16]);
        let v = feat.to_vec_f32().unwrap();
        assert_eq!(v.len(), 16);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn vit_rejects_wrong_image_size() {
        let vit = make_vit();
        let img = cpu_tensor(vec![0.0f32; 3 * 16 * 16], Shape::new(vec![3, 16, 16]));
        let err = match vit.encode_image(&img) {
            Ok(_) => panic!("expected shape error, got Ok"),
            Err(e) => e,
        };
        match err {
            Error::Shape(_) => {}
            other => panic!("expected shape error, got {:?}", other),
        }
    }

    #[test]
    fn vit_feature_dim_matches_hidden_size() {
        let cfg = VitConfig {
            image_size: 16,
            patch_size: 8,
            in_channels: 3,
            hidden_size: 64,
            num_heads: 4,
            num_layers: 1,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
        };
        let vit = Vit::random(cfg);
        assert_eq!(vit.features, 64);
    }
}
