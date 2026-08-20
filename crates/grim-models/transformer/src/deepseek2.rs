//! DeepSeek V2 architecture with Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE).
//!
//! # Architecture Details
//! - **Multi-Head Latent Attention (MLA)**: Compresses KV cache into low-rank latent representations with `kv_a_proj_with_mqa` and `kv_a_layernorm`, then expands to decoupled non-rotary and rotary components (`qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`).
//! - **DeepSeek MoE**: Top-k routed experts combined with dedicated shared experts (`first_k_dense_replace: 1` dense base layers).

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for DeepSeek V2 model architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeepSeek2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
}

impl Default for DeepSeek2Config {
    fn default() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 192,
            num_layers: 27,
            intermediate_size: 10944,
            kv_lora_rank: 512,
            q_lora_rank: None,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 163840,
            moe_intermediate_size: 1408,
            n_routed_experts: 64,
            n_shared_experts: 2,
            num_experts_per_tok: 6,
            first_k_dense_replace: 1,
            routed_scaling_factor: 1.0,
        }
    }
}

impl ModelConfig for DeepSeek2Config {
    fn name(&self) -> &str {
        "deepseek2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MLA Attention Block
// ---------------------------------------------------------------------------

/// Multi-Head Latent Attention layer for DeepSeek V2.
pub struct DeepSeek2Mla {
    pub q_proj: Linear,
    pub kv_a_proj: Linear,
    pub kv_a_layernorm: RmsNorm,
    pub kv_b_proj: Linear,
    pub o_proj: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl DeepSeek2Mla {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek2Config) -> Result<Self> {
        let q_dim = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
        let q_proj = Linear::load_shape(&ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;

        let kv_a_proj = Linear::load_shape(
            &ws.scoped("kv_a_proj_with_mqa"),
            [cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim],
        )?;
        let kv_a_layernorm = RmsNorm::load(&ws.scoped("kv_a_layernorm"), cfg.kv_lora_rank, cfg.rms_norm_eps)?;

        let kv_b_proj = Linear::load_shape(
            &ws.scoped("kv_b_proj"),
            [cfg.kv_lora_rank, cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)],
        )?;
        let o_proj = Linear::load_shape(&ws.scoped("o_proj"), [cfg.num_heads * cfg.v_head_dim, cfg.hidden_size])?;

        let rope = Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta);

        Ok(Self {
            q_proj,
            kv_a_proj,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rope,
            num_heads: cfg.num_heads,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];

        // 1. Q projection
        let q_full = self.q_proj.forward(x)?;
        let q_full_v = q_full.to_vec_f32()?;
        let total_q_head = self.qk_nope_head_dim + self.qk_rope_head_dim;

        let mut q_nope_v = vec![0.0f32; seq_len * self.num_heads * self.qk_nope_head_dim];
        let mut q_rope_v = vec![0.0f32; seq_len * self.num_heads * self.qk_rope_head_dim];

        for s in 0..seq_len {
            for h in 0..self.num_heads {
                let in_off = s * self.num_heads * total_q_head + h * total_q_head;
                let nope_off = s * self.num_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim;
                let rope_off = s * self.num_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim;

                q_nope_v[nope_off..nope_off + self.qk_nope_head_dim]
                    .copy_from_slice(&q_full_v[in_off..in_off + self.qk_nope_head_dim]);
                q_rope_v[rope_off..rope_off + self.qk_rope_head_dim]
                    .copy_from_slice(&q_full_v[in_off + self.qk_nope_head_dim..in_off + total_q_head]);
            }
        }

        crate::qwen35::apply_rope_neox(&mut q_rope_v, positions, self.num_heads, self.qk_rope_head_dim, 10000.0);

        // 2. KV latent projection
        let kv_latent = self.kv_a_proj.forward(x)?;
        let kv_latent_v = kv_latent.to_vec_f32()?;
        let kv_rank = self.kv_a_layernorm.weight.shape().dims()[0];

        let mut kv_a_v = vec![0.0f32; seq_len * kv_rank];
        let mut k_rope_v = vec![0.0f32; seq_len * self.qk_rope_head_dim];

        for s in 0..seq_len {
            let in_off = s * (kv_rank + self.qk_rope_head_dim);
            kv_a_v[s * kv_rank..(s + 1) * kv_rank]
                .copy_from_slice(&kv_latent_v[in_off..in_off + kv_rank]);
            k_rope_v[s * self.qk_rope_head_dim..(s + 1) * self.qk_rope_head_dim]
                .copy_from_slice(&kv_latent_v[in_off + kv_rank..in_off + kv_rank + self.qk_rope_head_dim]);
        }

        let kv_a_t = cpu_tensor(kv_a_v, Shape::new(vec![seq_len, kv_rank]));
        let kv_a_normed = self.kv_a_layernorm.forward(&kv_a_t)?;

        crate::qwen35::apply_rope_neox(&mut k_rope_v, positions, 1, self.qk_rope_head_dim, 10000.0);

        // Uncompress KV from latent
        let kv_b = self.kv_b_proj.forward(&kv_a_normed)?;
        let kv_b_v = kv_b.to_vec_f32()?;
        let kv_b_head = self.qk_nope_head_dim + self.v_head_dim;

        let mut k_nope_v = vec![0.0f32; seq_len * self.num_heads * self.qk_nope_head_dim];
        let mut v_v = vec![0.0f32; seq_len * self.num_heads * self.v_head_dim];

        for s in 0..seq_len {
            for h in 0..self.num_heads {
                let in_off = s * self.num_heads * kv_b_head + h * kv_b_head;
                let k_off = s * self.num_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim;
                let v_off = s * self.num_heads * self.v_head_dim + h * self.v_head_dim;

                k_nope_v[k_off..k_off + self.qk_nope_head_dim]
                    .copy_from_slice(&kv_b_v[in_off..in_off + self.qk_nope_head_dim]);
                v_v[v_off..v_off + self.v_head_dim]
                    .copy_from_slice(&kv_b_v[in_off + self.qk_nope_head_dim..in_off + kv_b_head]);
            }
        }

        // Cache update
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            new_k.extend(k_nope_v);
            new_v.extend(v_v);
            let total_k_dim = self.num_heads * self.qk_nope_head_dim;
            let total_v_dim = self.num_heads * self.v_head_dim;
            let total_seq = new_k.len() / total_k_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, total_k_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, total_v_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            let total_k_dim = self.num_heads * self.qk_nope_head_dim;
            let total_v_dim = self.num_heads * self.v_head_dim;
            let full_k = cpu_tensor(k_nope_v, Shape::new(vec![seq_len, total_k_dim]));
            let full_v = cpu_tensor(v_v, Shape::new(vec![seq_len, total_v_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        };

        let total_kv_len = k_all.shape().dims()[0];
        let k_all_v = k_all.to_vec_f32()?;
        let v_all_v = v_all.to_vec_f32()?;

        let scale = 1.0 / ((self.qk_nope_head_dim + self.qk_rope_head_dim) as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * self.num_heads * self.v_head_dim];

        for s in 0..seq_len {
            for h in 0..self.num_heads {
                let q_nope_slice = &q_nope_v[s * self.num_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim
                    ..s * self.num_heads * self.qk_nope_head_dim + (h + 1) * self.qk_nope_head_dim];
                let q_rope_slice = &q_rope_v[s * self.num_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim
                    ..s * self.num_heads * self.qk_rope_head_dim + (h + 1) * self.qk_rope_head_dim];

                let mut scores = vec![0.0f32; total_kv_len];
                for t in 0..total_kv_len {
                    let k_nope_slice = &k_all_v[t * self.num_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim
                        ..t * self.num_heads * self.qk_nope_head_dim + (h + 1) * self.qk_nope_head_dim];
                    let k_rope_slice = if t < seq_len {
                        &k_rope_v[t * self.qk_rope_head_dim..(t + 1) * self.qk_rope_head_dim]
                    } else {
                        &k_rope_v[0..self.qk_rope_head_dim]
                    };

                    let dot_nope: f32 = q_nope_slice.iter().zip(k_nope_slice.iter()).map(|(a, b)| a * b).sum();
                    let dot_rope: f32 = q_rope_slice.iter().zip(k_rope_slice.iter()).map(|(a, b)| a * b).sum();
                    scores[t] = (dot_nope + dot_rope) * scale;
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / (sum_exp + 1e-12)).collect();

                for d in 0..self.v_head_dim {
                    let mut acc = 0.0f32;
                    for t in 0..total_kv_len {
                        let v_val = v_all_v[t * self.num_heads * self.v_head_dim + h * self.v_head_dim + d];
                        acc += weights[t] * v_val;
                    }
                    attn_out[s * self.num_heads * self.v_head_dim + h * self.v_head_dim + d] = acc;
                }
            }
        }

        let attn_tensor = cpu_tensor(attn_out, Shape::new(vec![seq_len, self.num_heads * self.v_head_dim]));
        Ok(self.o_proj.forward(&attn_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// MoE Feed-Forward Layer
// ---------------------------------------------------------------------------

pub struct DeepSeek2Expert {
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
}

impl DeepSeek2Expert {
    pub fn load(ws: &WeightSource<'_>, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let w1 = Linear::load_shape(&ws.scoped("w1"), [hidden_size, intermediate_size])?;
        let w3 = Linear::load_shape(&ws.scoped("w3"), [hidden_size, intermediate_size])?;
        let w2 = Linear::load_shape(&ws.scoped("w2"), [intermediate_size, hidden_size])?;
        Ok(Self { w1, w3, w2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let gv = gate.to_vec_f32()?;
        let uv = up.to_vec_f32()?;
        let swiglu: Vec<f32> = gv
            .iter()
            .zip(uv.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let swiglu_t = cpu_tensor(swiglu, gate.shape().clone());
        Ok(self.w2.forward(&swiglu_t)?)
    }
}

pub struct DeepSeek2Moe {
    pub gate: Linear,
    pub experts: Vec<DeepSeek2Expert>,
    pub shared_experts: Option<DeepSeek2Expert>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
}

impl DeepSeek2Moe {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek2Config) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.n_routed_experts])?;

        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        let exp_ws = ws.scoped("experts");
        for e in 0..cfg.n_routed_experts {
            let exp = DeepSeek2Expert::load(&exp_ws.scoped(&e.to_string()), cfg.hidden_size, cfg.moe_intermediate_size)?;
            experts.push(exp);
        }

        let shared_experts = if cfg.n_shared_experts > 0 {
            let shared_ws = ws.scoped("shared_experts");
            let exp = DeepSeek2Expert::load(&shared_ws, cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts)?;
            Some(exp)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let hidden_dim = x.shape().dims()[1];
        let logits = self.gate.forward(x)?;
        let logits_v = logits.to_vec_f32()?;
        let num_exp = self.experts.len();

        let xv = x.to_vec_f32()?;
        let mut out = vec![0.0f32; seq_len * hidden_dim];

        for s in 0..seq_len {
            let row = &logits_v[s * num_exp..(s + 1) * num_exp];
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &indexed[..self.num_experts_per_tok.min(num_exp)];

            let max_l = topk.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| (e / (sum_e + 1e-12)) * self.routed_scaling_factor).collect();

            let token_x = cpu_tensor(xv[s * hidden_dim..(s + 1) * hidden_dim].to_vec(), Shape::new(vec![1, hidden_dim]));

            for (i, (exp_idx, _)) in topk.iter().enumerate() {
                let w = weights[i];
                let exp_out = self.experts[*exp_idx].forward(&token_x)?.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out[s * hidden_dim + d] += w * exp_out[d];
                }
            }
        }

        let mut out_t = cpu_tensor(out, x.shape().clone());

        if let Some(ref shared) = self.shared_experts {
            let sh_out = shared.forward(x)?;
            let ov = out_t.to_vec_f32()?;
            let sv = sh_out.to_vec_f32()?;
            let combined: Vec<f32> = ov.iter().zip(sv.iter()).map(|(&a, &b)| a + b).collect();
            out_t = cpu_tensor(combined, x.shape().clone());
        }

        Ok(out_t)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct DeepSeek2Block {
    pub attn_norm: RmsNorm,
    pub self_attn: DeepSeek2Mla,
    pub ffn_norm: RmsNorm,
    pub mlp: Option<DeepSeek2Expert>,
    pub moe: Option<DeepSeek2Moe>,
}

impl DeepSeek2Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek2Config, is_dense: bool) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.scoped("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let self_attn = DeepSeek2Mla::load(&ws.scoped("self_attn"), cfg)?;
        let ffn_norm = RmsNorm::load(&ws.scoped("post_attention_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        let (mlp, moe) = if is_dense {
            let mlp = DeepSeek2Expert::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
            (Some(mlp), None)
        } else {
            let moe = DeepSeek2Moe::load(&ws.scoped("mlp"), cfg)?;
            (None, Some(moe))
        };

        Ok(Self {
            attn_norm,
            self_attn,
            ffn_norm,
            mlp,
            moe,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let normed_attn = self.attn_norm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed_attn, positions, kv_cache)?;

        let xv = x.to_vec_f32()?;
        let av = attn_out.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;
        let mlp_out = if let Some(ref mlp) = self.mlp {
            mlp.forward(&normed_ffn)?
        } else if let Some(ref moe) = self.moe {
            moe.forward(&normed_ffn)?
        } else {
            normed_ffn.clone()
        };

        let r1v = res1_t.to_vec_f32()?;
        let mv = mlp_out.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct DeepSeek2 {
    pub cfg: DeepSeek2Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<DeepSeek2Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl DeepSeek2 {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: DeepSeek2Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: DeepSeek2Config,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(&root.scoped("embed_tokens"), [cfg.vocab_size, cfg.hidden_size])?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let is_dense = i < cfg.first_k_dense_replace;
            let block = DeepSeek2Block::load(&layer_ws, &cfg, is_dense)?;
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

impl Model for DeepSeek2 {
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

impl CausalLm for DeepSeek2 {
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
                hidden[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size]
                    .copy_from_slice(&embed_w[tok * self.cfg.hidden_size..(tok + 1) * self.cfg.hidden_size]);
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
    fn test_deepseek2_config() {
        let cfg = DeepSeek2Config::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.n_routed_experts, 64);
        assert_eq!(cfg.n_shared_experts, 2);
    }
}

