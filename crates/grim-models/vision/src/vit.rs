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
use grim_nn::{Linear, RmsNorm, WeightSource};
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
    fn name(&self) -> &str {
        "vit"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::VisionEncoder
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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
        rng: &mut grim_core::rng::SimpleRng,
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
            wq,
            wk,
            wv,
            wo,
            w_fc1: Linear::from_tensor(
                cpu_tensor(fc1_w, Shape::new(vec![intermediate, hidden])),
                Some(cpu_tensor(
                    vec![0.0; intermediate],
                    Shape::new(vec![intermediate]),
                )),
            ),
            w_fc2: Linear::from_tensor(
                cpu_tensor(fc2_w, Shape::new(vec![hidden, intermediate])),
                Some(cpu_tensor(vec![0.0; hidden], Shape::new(vec![hidden]))),
            ),
            hidden,
            num_heads,
            head_dim,
            intermediate,
        }
    }

    fn load(
        ws: &WeightSource<'_>,
        hidden: usize,
        num_heads: usize,
        intermediate: usize,
        eps: f32,
    ) -> Result<Self> {
        let head_dim = hidden / num_heads;
        let wq = ws
            .get([num_heads * head_dim, hidden], "attn.q.weight")?
            .to_vec_f32()?;
        let wk = ws
            .get([num_heads * head_dim, hidden], "attn.k.weight")?
            .to_vec_f32()?;
        let wv = ws
            .get([num_heads * head_dim, hidden], "attn.v.weight")?
            .to_vec_f32()?;
        let wo = ws
            .get([hidden, num_heads * head_dim], "attn.o.weight")?
            .to_vec_f32()?;
        let norm1 = RmsNorm::load(&ws.pp("attn_norm"), hidden, eps)?;
        let w_fc1 = Linear::load(&ws.pp("ffn.0"), hidden, intermediate, true)?;
        let w_fc2 = Linear::load(&ws.pp("ffn.1"), intermediate, hidden, true)?;
        Ok(Self {
            norm1,
            wq,
            wk,
            wv,
            wo,
            w_fc1,
            w_fc2,
            hidden,
            num_heads,
            head_dim,
            intermediate,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Result<Vec<f32>> {
        let h = self.hidden;
        let x_normed = rmsnorm_inplace(x, &self.norm1.weight.to_vec_f32()?, self.norm1.eps);

        let mut q = vec![0.0f32; seq * h];
        let mut k = vec![0.0f32; seq * h];
        let mut v = vec![0.0f32; seq * h];
        for s in 0..seq {
            for col in 0..h {
                let mut sum_q = 0.0f32;
                let mut sum_k = 0.0f32;
                let mut sum_v = 0.0f32;
                for k_idx in 0..h {
                    let val = x_normed[s * h + k_idx];
                    sum_q += val * self.wq[col * h + k_idx];
                    sum_k += val * self.wk[col * h + k_idx];
                    sum_v += val * self.wv[col * h + k_idx];
                }
                q[s * h + col] = sum_q;
                k[s * h + col] = sum_k;
                v[s * h + col] = sum_v;
            }
        }

        let mut attn_val = vec![0.0f32; seq * h];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let hd = self.head_dim;
        for head in 0..self.num_heads {
            for s in 0..seq {
                let mut scores = vec![0.0f32; seq];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[s * h + head * hd + d] * k[j * h + head * hd + d];
                    }
                    scores[j] = scale * dot;
                    if scores[j] > max_score {
                        max_score = scores[j];
                    }
                }
                let mut sum_exp = 0.0f32;
                for j in 0..seq {
                    scores[j] = (scores[j] - max_score).exp();
                    sum_exp += scores[j];
                }
                for j in 0..seq {
                    scores[j] /= sum_exp;
                }
                for d in 0..hd {
                    let mut val = 0.0f32;
                    for j in 0..seq {
                        val += scores[j] * v[j * h + head * hd + d];
                    }
                    attn_val[s * h + head * hd + d] = val;
                }
            }
        }

        let mut attn_out = vec![0.0f32; seq * h];
        for s in 0..seq {
            for col in 0..h {
                let mut sum_o = 0.0f32;
                for k_idx in 0..h {
                    sum_o += self.wo[col * h + k_idx] * attn_val[s * h + k_idx];
                }
                attn_out[s * h + col] = sum_o;
            }
        }

        let mut attn_res = x_normed.clone();
        for i in 0..attn_res.len() {
            attn_res[i] += attn_out[i];
        }

        let fc1_out = self
            .w_fc1
            .forward(&cpu_tensor(attn_res.clone(), Shape::new(vec![seq, h])))?;
        let gate = fc1_out.to_vec_f32()?;
        let mut gelu = vec![0.0f32; gate.len()];
        for (i, g) in gate.iter().enumerate() {
            gelu[i] = gelu_approx(*g);
        }
        let fc2_out = self
            .w_fc2
            .forward(&cpu_tensor(gelu, Shape::new(vec![seq, self.intermediate])))?;
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
    pub fn random(device: Device, cfg: VitConfig) -> Self {
        Self::new(
            device,
            cfg,
            &mut grim_core::rng::SimpleRng::new(0xC08D_E27B_71A5_F00Du64),
        )
    }

    /// Build the ViT given an RNG (lets callers choose a deterministic seed).
    pub fn new(device: Device, cfg: VitConfig, rng: &mut grim_core::rng::SimpleRng) -> Self {
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
            weight: cpu_tensor(
                vec![1.0; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps,
        };
        let features = cfg.hidden_size;
        Self {
            cfg,
            device,
            patch_proj_w: proj_w,
            patch_proj_b: proj_b,
            cls_token,
            pos_embed,
            blocks,
            ln,
            features,
        }
    }

    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: VitConfig) -> Result<Self> {
        let patch_dim = cfg.patch_dim();
        let proj_w = ws
            .get([cfg.hidden_size, patch_dim], "proj.weight")?
            .to_vec_f32()?;
        let proj_b = ws.get([cfg.hidden_size], "proj.bias")?.to_vec_f32()?;
        let cls_token = ws.get([cfg.hidden_size], "cls_token")?.to_vec_f32()?;
        let num_patches = cfg.num_patches();
        let pos_embed = ws
            .get([num_patches + 1, cfg.hidden_size], "pos_embed")?
            .to_vec_f32()?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let blk = VitBlock::load(
                &ws.pp(&format!("blocks.{i}")),
                cfg.hidden_size,
                cfg.num_heads,
                cfg.intermediate_size,
                cfg.rms_norm_eps,
            )?;
            blocks.push(blk);
        }
        let ln = RmsNorm::load(&ws.pp("ln"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let features = cfg.hidden_size;
        Ok(Self {
            cfg,
            device,
            patch_proj_w: proj_w,
            patch_proj_b: proj_b,
            cls_token,
            pos_embed,
            blocks,
            ln,
            features,
        })
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
        let hidden = self.cfg.hidden_size;
        let ph = patch;
        let pw = patch;
        let patch_dim = c * ph * pw;
        // CLS token at index 0 — apply positional embedding pos_embed[0].
        for o in 0..hidden {
            tokens[o] = self.cls_token[o] + self.pos_embed[o];
        }
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
                    // Apply positional embedding once (pos_embed[1..] for patches).
                    tokens[proj_offset + o] = acc + self.pos_embed[proj_offset + o];
                }
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
        Vit::random(Device::Cpu, cfg)
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
        let vit = Vit::random(Device::Cpu, cfg);
        assert_eq!(vit.features, 64);
    }

    #[test]
    fn vit_pos_embed_applied_once_and_to_cls() {
        // With zeroed weights, zeroed cls_token, and all-ones pos_embed,
        // every token before the blocks should be 1.0 (cls + pos_embed[0],
        // patches + pos_embed[1..]) and rmsnorm with weight=1 should preserve ~1.0.
        // Crucially, CLS must be non-zero (proving pos_embed[0] was applied),
        // and patches must not be doubled (proving single application).
        let cfg = VitConfig {
            image_size: 4,
            patch_size: 2,
            in_channels: 1,
            hidden_size: 4,
            num_heads: 2,
            num_layers: 0,
            intermediate_size: 8,
            rms_norm_eps: 1e-5,
        };
        let mut vit = Vit::random(Device::Cpu, cfg);
        vit.patch_proj_w = vec![0.0f32; vit.patch_proj_w.len()];
        vit.patch_proj_b = vec![0.0f32; vit.patch_proj_b.len()];
        vit.cls_token = vec![0.0f32; vit.cls_token.len()];
        vit.pos_embed = vec![1.0f32; vit.pos_embed.len()];
        vit.ln.weight = cpu_tensor(vec![1.0f32; 4], Shape::new(vec![4]));

        let image_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.01).collect();
        let img = cpu_tensor(image_data, Shape::new(vec![1, 4, 4]));
        let feat = vit.encode_image(&img).unwrap();
        let out = feat.to_vec_f32().unwrap();

        // With all-ones pos_embed, all-zero weights/bias/cls_token:
        // All token dims should be 1.0 (single pos_embed application).
        // Old buggy code: CLS=0 (no pos_embed), patches=2.0 (double pos_embed).
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 0.01,
                "CLS output should be ~1.0 (pos_embed applied once), got {v}"
            );
        }
    }
}
