//! Muse-Glimmer temporal-patch ViT vision encoder — `Encoder` trait impl.
//!
//! pipeline:
//!   temporal patch_embed (patch_temporal × patch_size²) → N × encoder block
//!   (with token-merging every `merge_size` blocks) → optional final norm
//!
//! Muse-Glimmer's vision tower (from `MuseVisionConfig`): a temporal ViT with
//! `hidden_size=768`, `patch_size=16`, `patch_temporal=1`, `merge_size=2`,
//! `use_vision_norm=true`, `n_layers=24`. `merge_size` controls how often a
//! deterministic adjacent-token merge halves the token count, mirroring the
//! token-merging graph in the reference implementation.
//!
//! All in F32 CPU for the structural layer; kernel backends land with
//! grim-backend-rocm in phase 4.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{Encoder, ModalityHint};
use grim_core::{Model, ModelConfig};
use grim_nn::{Linear, RmsNorm, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

/// Glimmer temporal-ViT configuration.
#[derive(Debug, Clone)]
pub struct GlimmerVisionConfig {
    pub image_temporal: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    /// Merge adjacent tokens every `merge_size` encoder blocks
    /// (1 = merge after every block; 0 disables merging).
    pub merge_size: usize,
    /// Apply the final LayerNorm after the encoder stack.
    pub use_vision_norm: bool,
}

impl GlimmerVisionConfig {
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let t = self.image_temporal / self.temporal_patch_size;
        let per_side = self.image_size / self.patch_size;
        t * per_side * per_side
    }
}

impl ModelConfig for GlimmerVisionConfig {
    fn name(&self) -> &str {
        "glimmer-vision"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::VisionEncoder
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// One Glimmer ViT self-attention block (pre-norm, bidirectional).
struct GlimmerVisionBlock {
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

impl GlimmerVisionBlock {
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
        let w_fc1 = Linear::load(&ws.pp("ffn.fc1"), hidden, intermediate, true)?;
        let w_fc2 = Linear::load(&ws.pp("ffn.fc2"), intermediate, hidden, true)?;
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

/// Merge adjacent tokens by averaging pairs, halving the sequence length.
/// Deterministic stand-in for the reference token-merging graph.
fn merge_adjacent(tokens: &mut Vec<f32>, hidden: usize) {
    let seq = tokens.len() / hidden;
    if seq < 2 {
        return;
    }
    let merged_seq = seq / 2;
    let mut out = vec![0.0f32; merged_seq * hidden];
    for i in 0..merged_seq {
        for d in 0..hidden {
            out[i * hidden + d] =
                0.5 * (tokens[2 * i * hidden + d] + tokens[(2 * i + 1) * hidden + d]);
        }
    }
    *tokens = out;
}

/// Muse-Glimmer temporal-patch vision transformer.
pub struct GlimmerVision {
    pub cfg: GlimmerVisionConfig,
    pub device: Device,
    pub patch_proj_w: Vec<f32>,
    pub patch_proj_b: Vec<f32>,
    blocks: Vec<GlimmerVisionBlock>,
    pub ln: Option<RmsNorm>,
    pub features: usize,
}

impl GlimmerVision {
    /// Build a randomly-initialized tiny Glimmer vision encoder.
    pub fn random(device: Device, cfg: GlimmerVisionConfig) -> Self {
        Self::new(
            device,
            cfg,
            &mut grim_core::rng::SimpleRng::new(0x6C71_6D0A_2B1E_5F21u64),
        )
    }

    /// Build given an RNG (deterministic seed control for tests).
    pub fn new(
        device: Device,
        cfg: GlimmerVisionConfig,
        rng: &mut grim_core::rng::SimpleRng,
    ) -> Self {
        let patch_dim = cfg.patch_dim();
        let proj_w: Vec<f32> = (0..cfg.hidden_size * patch_dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let proj_b = vec![0.0f32; cfg.hidden_size];
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            blocks.push(GlimmerVisionBlock::new(
                rng,
                cfg.hidden_size,
                cfg.num_heads,
                cfg.intermediate_size,
                cfg.rms_norm_eps,
            ));
        }
        let ln = if cfg.use_vision_norm {
            Some(RmsNorm {
                weight: cpu_tensor(
                    vec![1.0; cfg.hidden_size],
                    Shape::new(vec![cfg.hidden_size]),
                ),
                eps: cfg.rms_norm_eps,
            })
        } else {
            None
        };
        let features = cfg.hidden_size;
        Self {
            cfg,
            device,
            patch_proj_w: proj_w,
            patch_proj_b: proj_b,
            blocks,
            ln,
            features,
        }
    }

    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: GlimmerVisionConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry. Vision encoders do not run on the serving
    /// engine's text-out path; TP here is refused until an encoder consumer
    /// with an all-reduce hook arrives (mirrors `Vit`).
    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: GlimmerVisionConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "GlimmerVision",
            "vision encoder GlimmerVisionBlock::forward calls plain Linear::forward \
             with no all-reduce hook",
        )
        .map_err(Error::Unimplemented)?;
        let patch_dim = cfg.patch_dim();
        let proj_w = ws
            .get([cfg.hidden_size, patch_dim], "patch_embed.weight")?
            .to_vec_f32()?;
        let proj_b = ws
            .get([cfg.hidden_size], "patch_embed.bias")?
            .to_vec_f32()?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let blk = GlimmerVisionBlock::load(
                &ws.pp(&format!("blocks.{i}")),
                cfg.hidden_size,
                cfg.num_heads,
                cfg.intermediate_size,
                cfg.rms_norm_eps,
            )?;
            blocks.push(blk);
        }
        let ln = if cfg.use_vision_norm {
            Some(RmsNorm::load(
                &ws.pp("ln"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?)
        } else {
            None
        };
        let features = cfg.hidden_size;
        Ok(Self {
            cfg,
            device,
            patch_proj_w: proj_w,
            patch_proj_b: proj_b,
            blocks,
            ln,
            features,
        })
    }

    /// Encode a flat `(C, T, H, W)` tensor into `(num_tokens, hidden_size)`
    /// patch features (post final norm if `use_vision_norm`).
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        let shape = image.shape().dims().to_vec();
        if shape.len() != 4 {
            return Err(Error::Shape(format!(
                "GlimmerVision encode_image expects (C, T, H, W), got {:?}",
                shape
            )));
        }
        let (c, t, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if h != self.cfg.image_size || w != self.cfg.image_size {
            return Err(Error::Shape(format!(
                "GlimmerVision image {}×{} must match image_size {}",
                h, w, self.cfg.image_size
            )));
        }
        if t != self.cfg.image_temporal {
            return Err(Error::Shape(format!(
                "GlimmerVision expects {} temporal frames, got {}",
                self.cfg.image_temporal, t
            )));
        }
        if c != self.cfg.in_channels {
            return Err(Error::Shape(format!(
                "GlimmerVision expects {} channels, got {}",
                self.cfg.in_channels, c
            )));
        }
        let image_data = image.to_vec_f32()?;
        let ps = self.cfg.patch_size;
        let pt = self.cfg.temporal_patch_size;
        let per_t = t / pt;
        let per_side = h / ps;
        let num_patches = per_t * per_side * per_side;
        let hidden = self.cfg.hidden_size;
        let patch_dim = self.cfg.patch_dim();
        let mut tokens: Vec<f32> = vec![0.0f32; num_patches * hidden];
        for pt_idx in 0..per_t {
            for py in 0..per_side {
                for px in 0..per_side {
                    let patch_idx = (pt_idx * per_side + py) * per_side + px;
                    let mut patch_vec = vec![0.0f32; patch_dim];
                    for cti in 0..pt {
                        for cy in 0..ps {
                            for cx in 0..ps {
                                for ch in 0..c {
                                    let y = py * ps + cy;
                                    let x = px * ps + cx;
                                    let ti = pt_idx * pt + cti;
                                    let flat = ch * (t * h * w) + ti * (h * w) + y * w + x;
                                    let slot = (ch * pt + cti) * (ps * ps) + cy * ps + cx;
                                    patch_vec[slot] = image_data[flat];
                                }
                            }
                        }
                    }
                    let proj_offset = patch_idx * hidden;
                    for o in 0..hidden {
                        let mut acc = self.patch_proj_b[o];
                        for i in 0..patch_dim {
                            acc += self.patch_proj_w[o * patch_dim + i] * patch_vec[i];
                        }
                        tokens[proj_offset + o] = acc;
                    }
                }
            }
        }
        let mut seq = num_patches;
        for (i, b) in self.blocks.iter().enumerate() {
            tokens = b.forward(&tokens, seq)?;
            if self.cfg.merge_size > 0 && (i + 1) % self.cfg.merge_size == 0 {
                merge_adjacent(&mut tokens, hidden);
                seq = tokens.len() / hidden;
            }
        }
        if let Some(ln) = &self.ln {
            tokens = rmsnorm_inplace(&tokens, &ln.weight.to_vec_f32()?, ln.eps);
        }
        Ok(cpu_tensor(tokens, Shape::new(vec![seq, hidden])))
    }
}

impl Model for GlimmerVision {
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

impl Encoder for GlimmerVision {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        self.encode_image(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_glimmer_vision() -> GlimmerVision {
        let cfg = GlimmerVisionConfig {
            image_temporal: 2,
            image_size: 8,
            patch_size: 4,
            temporal_patch_size: 1,
            in_channels: 3,
            hidden_size: 16,
            num_heads: 2,
            num_layers: 2,
            intermediate_size: 32,
            rms_norm_eps: 1e-5,
            merge_size: 2,
            use_vision_norm: true,
        };
        GlimmerVision::random(Device::Cpu, cfg)
    }

    #[test]
    fn encodes_image_to_expected_shape_with_merge() {
        // 2 temporal frames × 8×8 image, patch 4 → 2*2*2 = 8 patches.
        // merge_size=2 merges after block 2: 8 → 4 tokens.
        let vision = make_glimmer_vision();
        let img = cpu_tensor(
            (0..3 * 2 * 8 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![3, 2, 8, 8]),
        );
        let feat = vision.encode_image(&img).unwrap();
        assert_eq!(feat.shape().dims(), &[4, 16]);
        let v = feat.to_vec_f32().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn rejects_wrong_temporal() {
        let vision = make_glimmer_vision();
        let img = cpu_tensor(vec![0.0f32; 3 * 1 * 8 * 8], Shape::new(vec![3, 1, 8, 8]));
        match vision.encode_image(&img) {
            Ok(_) => panic!("expected shape error, got Ok"),
            Err(Error::Shape(_)) => {}
            Err(other) => panic!("expected shape error, got {:?}", other),
        }
    }

    #[test]
    fn no_merge_keeps_all_patches() {
        let cfg = GlimmerVisionConfig {
            merge_size: 0,
            ..make_glimmer_vision().cfg
        };
        let vision = GlimmerVision::random(Device::Cpu, cfg);
        let img = cpu_tensor(
            (0..3 * 2 * 8 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![3, 2, 8, 8]),
        );
        let feat = vision.encode_image(&img).unwrap();
        assert_eq!(feat.shape().dims(), &[8, 16]);
    }
}
