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
use grim_nn::{Linear, WeightSource};
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

    pub fn from_hf(value: &serde_json::Value) -> Self {
        let vision_cfg = value.get("vision_config").unwrap_or(value);
        let u = |k: &str| vision_cfg.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| vision_cfg.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        let img_sz = if u("image_size") > 0 {
            u("image_size")
        } else {
            224
        };
        let patch_sz = if u("patch_size") > 0 {
            u("patch_size")
        } else {
            14
        };
        let in_ch = if u("num_channels") > 0 {
            u("num_channels")
        } else if u("in_channels") > 0 {
            u("in_channels")
        } else {
            3
        };
        let hidden = if u("hidden_size") > 0 {
            u("hidden_size")
        } else {
            768
        };
        let heads = if u("num_attention_heads") > 0 {
            u("num_attention_heads")
        } else if u("num_heads") > 0 {
            u("num_heads")
        } else {
            12
        };
        let layers = if u("num_hidden_layers") > 0 {
            u("num_hidden_layers")
        } else if u("num_layers") > 0 {
            u("num_layers")
        } else {
            12
        };
        let inter = if u("intermediate_size") > 0 {
            u("intermediate_size")
        } else {
            3072
        };
        let eps = if f("layer_norm_eps") > 0.0 {
            f("layer_norm_eps")
        } else if f("rms_norm_eps") > 0.0 {
            f("rms_norm_eps")
        } else {
            1e-5
        };

        VitConfig {
            image_size: img_sz,
            patch_size: patch_sz,
            in_channels: in_ch,
            hidden_size: hidden,
            num_heads: heads,
            num_layers: layers,
            intermediate_size: inter,
            rms_norm_eps: eps,
        }
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

/// True LayerNorm (mean-subtracted, learnable bias) — distinct from `RmsNorm`.
///
/// Real ViT checkpoints are trained with LayerNorm, so loading their `weight`/
/// `bias` into an RmsNorm (no mean subtraction, no bias) is numerically wrong.
/// [P1-34 fix: ViT norms are LayerNorm, not RmsNorm.]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    fn ones(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            bias: vec![0.0; dim],
            eps,
        }
    }

    /// Load `weight` and `bias` under `ws`. A missing `bias` tensor is treated
    /// as zeros (some exports omit it); a missing `weight` is a hard error.
    fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?.to_vec_f32()?;
        let bias = match ws.get([dim], "bias") {
            Ok(t) => t.to_vec_f32()?,
            Err(_) => vec![0.0; dim],
        };
        Ok(Self { weight, bias, eps })
    }

    /// Row-wise LayerNorm over the last dimension of a `[n, dim]` buffer.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.weight.len();
        let rows = x.len() / dim;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let off = r * dim;
            let row = &x[off..off + dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + self.eps).sqrt();
            for d in 0..dim {
                out[off + d] = (row[d] - mean) * inv * self.weight[d] + self.bias[d];
            }
        }
        out
    }
}

/// One ViT self-attention block (pre-norm).
struct VitBlock {
    norm1: LayerNorm,
    /// Second LayerNorm, applied before the MLP (pre-norm ViT has two norms
    /// per block). [P1-34 fix: pre-MLP norm was missing entirely.]
    norm2: LayerNorm,
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
            norm1: LayerNorm::ones(hidden, eps),
            norm2: LayerNorm::ones(hidden, eps),
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
        let norm1 = LayerNorm::load(&ws.pp("attn_norm"), hidden, eps)?;
        let norm2 = LayerNorm::load(&ws.pp("ffn_norm"), hidden, eps)?;
        let w_fc1 = Linear::load(&ws.pp("ffn.0"), hidden, intermediate, true)?;
        let w_fc2 = Linear::load(&ws.pp("ffn.1"), intermediate, hidden, true)?;
        Ok(Self {
            norm1,
            norm2,
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
        let x_normed = self.norm1.forward(x);

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
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                for score in &mut scores {
                    *score /= sum_exp;
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

        // Pre-norm residual: skip connection uses the original input x, not x_normed.
        // attn_res = x + attn(norm(x)). [P1-34 fix: residual uses x, not x_normed.]
        let mut attn_res: Vec<f32> = x.to_vec();
        for i in 0..attn_res.len() {
            attn_res[i] += attn_out[i];
        }
        // Pre-MLP norm: mlp operates on norm2(attn_res), and the second
        // residual adds onto attn_res (not the block input x).
        // [P1-34 fix: missing pre-MLP norm + wrong second residual base.]
        let normed2 = self.norm2.forward(&attn_res);
        let fc1_out = self
            .w_fc1
            .forward(&cpu_tensor(normed2, Shape::new(vec![seq, h])))?;
        let gate = fc1_out.to_vec_f32()?;
        let mut gelu = vec![0.0f32; gate.len()];
        for (i, g) in gate.iter().enumerate() {
            gelu[i] = gelu_approx(*g);
        }
        let fc2_out = self
            .w_fc2
            .forward(&cpu_tensor(gelu, Shape::new(vec![seq, self.intermediate])))?;
        let mlp = fc2_out.to_vec_f32()?;
        let mut out = attn_res;
        for i in 0..out.len() {
            out[i] += mlp[i];
        }
        Ok(out)
    }
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
    pub ln: LayerNorm,
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
        let ln = LayerNorm::ones(cfg.hidden_size, cfg.rms_norm_eps);
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
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for ViT. ViT is a vision encoder (`Model`/
    /// `Encoder`, not `CausalLm`); `VitBlock::forward` calls plain
    /// `Linear::forward` with no all-reduce hook, and TP for vision encoders is
    /// low-leverage since they don't run on the serving engine's text-out
    /// path. Refused until a `forward` rework + an encoder consumer arrive.
    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: VitConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "ViT",
            "vision encoder VitBlock::forward calls plain Linear::forward with no \
             all-reduce hook",
        )
        .map_err(Error::Unimplemented)?;
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
        let ln = LayerNorm::load(&ws.pp("ln"), cfg.hidden_size, cfg.rms_norm_eps)?;
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
        for (o, tok) in tokens[..hidden].iter_mut().enumerate() {
            *tok = self.cls_token[o] + self.pos_embed[o];
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
                    for (i, &pv) in patch_vec.iter().enumerate() {
                        acc += self.patch_proj_w[o * patch_dim + i] * pv;
                    }
                    // Apply positional embedding once (pos_embed[1..] for patches).
                    tokens[proj_offset + o] = acc + self.pos_embed[proj_offset + o];
                }
            }
        }
        for b in &self.blocks {
            tokens = b.forward(&tokens, num_patches + 1)?;
        }
        let post = self.ln.forward(&tokens);
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
        // With zeroed projection weights and a zeroed cls_token, the CLS row
        // entering the block stack is exactly pos_embed[0..hidden] — so the
        // final LayerNorm output must equal LayerNorm(pos_embed[0..hidden]).
        // Double application (2x) or a missing CLS pos_embed (zeros) both
        // produce a different result, since LayerNorm is not scale-invariant
        // once the row is non-constant vs. all-zero.
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
        // Non-constant per-dim pattern, repeated for every token.
        let pattern = [1.0f32, 2.0, 3.0, 4.0];
        vit.pos_embed = (0..vit.pos_embed.len()).map(|i| pattern[i % 4]).collect();
        vit.ln = LayerNorm::ones(4, 1e-5);

        let image_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.01).collect();
        let img = cpu_tensor(image_data, Shape::new(vec![1, 4, 4]));
        let feat = vit.encode_image(&img).unwrap();
        let out = feat.to_vec_f32().unwrap();

        let expect = LayerNorm::ones(4, 1e-5).forward(&pattern);
        assert_eq!(out.len(), 4);
        for (i, (&got, &want)) in out.iter().zip(expect.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "dim {i}: expected LayerNorm(pos_embed) {want}, got {got} \
                 (pos_embed applied zero or twice?)"
            );
        }
        // Sanity: the row is genuinely non-trivial, so this is a real check.
        assert!(expect.iter().any(|v| v.abs() > 0.5));
    }

    #[test]
    fn vit_config_from_hf_parsing() {
        let json_str = r#"{
            "vision_config": {
                "image_size": 336,
                "patch_size": 14,
                "num_channels": 3,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "intermediate_size": 4096,
                "layer_norm_eps": 1e-5
            }
        }"#;
        let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let cfg = VitConfig::from_hf(&v);
        assert_eq!(cfg.image_size, 336);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.name(), "vit");
    }
}
