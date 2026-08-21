//! Chameleon multimodal-compatible transformer with per-head Q/K normalization.
//!
//! # Architecture Details
//! - **Per-Head Q/K Normalization (`swin_norm`)**: LayerNorm / RMSNorm applied to each attention head's Q and K independently prior to RoPE.
//! - **GQA Attention & SwiGLU MLP**: Standard Grouped Query Attention with RoPE and SwiGLU feed-forward networks.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

use crate::falcon::LayerNorm;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Chameleon model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChameleonConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub swin_norm: bool,
}

impl Default for ChameleonConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65536,
            hidden_size: 8192,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 48,
            intermediate_size: 22016,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            swin_norm: true,
        }
    }
}

impl ModelConfig for ChameleonConfig {
    fn name(&self) -> &str {
        "chameleon"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Chameleon Block
// ---------------------------------------------------------------------------

/// A transformer block with per-head normalized attention and SwiGLU feed-forward.
pub struct ChameleonBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: Option<LayerNorm>,
    pub k_norm: Option<LayerNorm>,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub swin_norm: bool,
}

impl ChameleonBlock {
    /// Loads Chameleon transformer layer weights with optional per-head Q/K LayerNorms.
    pub fn load(ws: &WeightSource<'_>, cfg: &ChameleonConfig) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let (q_norm, k_norm) = if cfg.swin_norm {
            let qn =
                LayerNorm::load(&attn_ws.scoped("q_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
            let kn =
                LayerNorm::load(&attn_ws.scoped("k_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
            (qn, kn)
        } else {
            (None, None)
        };

        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let mlp_ws = ws.scoped("mlp");
        let w_gate = Linear::load_shape(
            &mlp_ws.scoped("gate_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w_up = Linear::load_shape(
            &mlp_ws.scoped("up_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w_down = Linear::load_shape(
            &mlp_ws.scoped("down_proj"),
            [cfg.intermediate_size, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            attn_norm,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            swin_norm: cfg.swin_norm,
        })
    }

    /// Evaluates one transformer block: Pre-RMSNorm -> Q/K norm -> RoPE -> GQA -> Post-RMSNorm -> SwiGLU.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.attn_norm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let q_dim = self.num_heads * self.head_dim;
        let k_dim = self.num_kv_heads * self.head_dim;

        let mut q_vec = if let Some(ref qn) = self.q_norm {
            let q_v = q.to_vec_f32()?;
            let mut q_normed = vec![0.0f32; seq_len * q_dim];
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    let slice =
                        &q_v[s * q_dim + h * self.head_dim..s * q_dim + (h + 1) * self.head_dim];
                    let head_t = cpu_tensor(slice.to_vec(), Shape::new(vec![1, self.head_dim]));
                    let head_out = qn.forward(&head_t)?.to_vec_f32()?;
                    q_normed[s * q_dim + h * self.head_dim..s * q_dim + (h + 1) * self.head_dim]
                        .copy_from_slice(&head_out);
                }
            }
            q_normed
        } else {
            q.to_vec_f32()?
        };

        let mut k_vec = if let Some(ref kn) = self.k_norm {
            let kv_vec = k.to_vec_f32()?;
            let mut k_normed = vec![0.0f32; seq_len * k_dim];
            for s in 0..seq_len {
                for h in 0..self.num_kv_heads {
                    let slice =
                        &kv_vec[s * k_dim + h * self.head_dim..s * k_dim + (h + 1) * self.head_dim];
                    let head_t = cpu_tensor(slice.to_vec(), Shape::new(vec![1, self.head_dim]));
                    let head_out = kn.forward(&head_t)?.to_vec_f32()?;
                    k_normed[s * k_dim + h * self.head_dim..s * k_dim + (h + 1) * self.head_dim]
                        .copy_from_slice(&head_out);
                }
            }
            k_normed
        } else {
            k.to_vec_f32()?
        };

        crate::qwen35::apply_rope_neox(
            &mut q_vec,
            positions,
            self.num_heads,
            self.head_dim,
            10000.0,
        );
        crate::qwen35::apply_rope_neox(
            &mut k_vec,
            positions,
            self.num_kv_heads,
            self.head_dim,
            10000.0,
        );

        let q_rot = cpu_tensor(q_vec, Shape::new(vec![seq_len, q_dim]));
        let k_rot = cpu_tensor(k_vec, Shape::new(vec![seq_len, k_dim]));

        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            new_k.extend(k_rot.to_vec_f32()?);
            new_v.extend(v.to_vec_f32()?);
            let total_seq = new_k.len() / k_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, k_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, k_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k_rot.clone(), v.clone()));
            (k_rot, v)
        };

        let total_kv_len = k_all.shape().dims()[0];
        let q_heads = q_rot.to_vec_f32()?;
        let k_heads = k_all.to_vec_f32()?;
        let v_heads = v_all.to_vec_f32()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let kv_group_size = (self.num_heads / self.num_kv_heads).max(1);

        let mut attn_out = vec![0.0f32; seq_len * q_dim];

        for s in 0..seq_len {
            for h in 0..self.num_heads {
                let kv_h = h / kv_group_size;
                let q_slice =
                    &q_heads[s * q_dim + h * self.head_dim..s * q_dim + (h + 1) * self.head_dim];

                let mut scores = vec![0.0f32; total_kv_len];
                for t in 0..total_kv_len {
                    let k_slice = &k_heads
                        [t * k_dim + kv_h * self.head_dim..t * k_dim + (kv_h + 1) * self.head_dim];
                    let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
                    scores[t] = dot * scale;
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / (sum_exp + 1e-12)).collect();

                for d in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for t in 0..total_kv_len {
                        let v_val = v_heads[t * k_dim + kv_h * self.head_dim + d];
                        acc += weights[t] * v_val;
                    }
                    attn_out[s * q_dim + h * self.head_dim + d] = acc;
                }
            }
        }

        let attn_tensor = cpu_tensor(attn_out, Shape::new(vec![seq_len, q_dim]));
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let xv = x.to_vec_f32()?;
        let av = attn_proj.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;
        let gate = self.w_gate.forward(&normed_ffn)?;
        let up = self.w_up.forward(&normed_ffn)?;

        let g_v = gate.to_vec_f32()?;
        let u_v = up.to_vec_f32()?;
        let swiglu: Vec<f32> = g_v
            .iter()
            .zip(u_v.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let swiglu_t = cpu_tensor(swiglu, gate.shape().clone());
        let mlp_out = self.w_down.forward(&swiglu_t)?;

        let r1v = res1_t.to_vec_f32()?;
        let mv = mlp_out.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct Chameleon {
    pub cfg: ChameleonConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<ChameleonBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Chameleon {
    /// Loads complete Chameleon causal LM weights from the weight source with standard TP config.
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: ChameleonConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Loads complete Chameleon causal LM weights with tensor parallelism support.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: ChameleonConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = ChameleonBlock::load(&layer_ws, &cfg)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| tok_embeddings.clone());

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for Chameleon {
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

impl CausalLm for Chameleon {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids = input_ids.to_vec_f32()?;
        let seq_len = ids.len();
        let pos_v: Vec<u32> = positions
            .to_vec_f32()
            .map(|v| v.into_iter().map(|p| p as u32).collect())
            .unwrap_or_else(|_| (0..seq_len as u32).collect());

        let mut hidden = vec![0.0f32; seq_len * self.cfg.hidden_size];

        let embed_w = self.tok_embeddings.weight.to_vec_f32()?;
        for (i, &tok_f) in ids.iter().enumerate() {
            let tok = tok_f as usize;
            if tok < self.cfg.vocab_size {
                hidden[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size].copy_from_slice(
                    &embed_w[tok * self.cfg.hidden_size..(tok + 1) * self.cfg.hidden_size],
                );
            }
        }

        let mut x = cpu_tensor(hidden, Shape::new(vec![seq_len, self.cfg.hidden_size]));
        let mut kv_caches = vec![None; self.layers.len()];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, &mut kv_caches[layer_idx])?;
        }

        let normed = self.norm.forward(&x)?;
        let logits = self.output.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chameleon_config() {
        let cfg = ChameleonConfig::default();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.swin_norm, true);
    }
}
