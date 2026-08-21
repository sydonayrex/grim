//! DeepSeek V2 architecture with Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE).
//!
//! # Architecture Details
//! - **Multi-Head Latent Attention (MLA)**: Compresses KV cache into low-rank latent representations with `kv_a_proj_with_mqa` and `kv_a_layernorm`, then expands to decoupled non-rotary and rotary components (`qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`).
//! - **DeepSeek MoE**: Top-k routed experts combined with dedicated shared experts (`first_k_dense_replace: 1` dense base layers).

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, DType, QuantProvenance, Shape, Tensor};
use std::sync::Arc;

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
    /// Absorbed per-head key up-projection `w_kc[h]`, row-major
    /// `[num_heads, qk_nope_head_dim, kv_lora_rank]` (rows of `kv_b_proj.weight`).
    pub w_kc: Vec<f32>,
    /// Per-head value up-projection `w_vc[h]`, row-major
    /// `[num_heads, v_head_dim, kv_lora_rank]`.
    pub w_vc: Vec<f32>,
}

impl DeepSeek2Mla {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek2Config) -> Result<Self> {
        let q_dim = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
        let q_proj = Linear::load_shape(&ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;

        let kv_a_proj = Linear::load_shape(
            &ws.scoped("kv_a_proj_with_mqa"),
            [cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim],
        )?;
        let kv_a_layernorm = RmsNorm::load(
            &ws.scoped("kv_a_layernorm"),
            cfg.kv_lora_rank,
            cfg.rms_norm_eps,
        )?;

        let kv_b_proj = Linear::load_shape(
            &ws.scoped("kv_b_proj"),
            [
                cfg.kv_lora_rank,
                cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            ],
        )?;
        let o_proj = Linear::load_shape(
            &ws.scoped("o_proj"),
            [cfg.num_heads * cfg.v_head_dim, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta);

        // Extract the per-head key/value up-projections from kv_b_proj's
        // weight ([num_heads * (nope + v), rank], GGUF row-major) so queries
        // can absorb w_kc and attention can run in latent space.
        let kv_b_w = kv_b_proj.weight.to_vec_f32()?;
        let kv_b_head = cfg.qk_nope_head_dim + cfg.v_head_dim;
        let rank = cfg.kv_lora_rank;
        let mut w_kc = vec![0.0f32; cfg.num_heads * cfg.qk_nope_head_dim * rank];
        let mut w_vc = vec![0.0f32; cfg.num_heads * cfg.v_head_dim * rank];
        for h in 0..cfg.num_heads {
            let hb = h * kv_b_head;
            for d in 0..cfg.qk_nope_head_dim {
                let src = (hb + d) * rank;
                let dst = (h * cfg.qk_nope_head_dim + d) * rank;
                w_kc[dst..dst + rank].copy_from_slice(&kv_b_w[src..src + rank]);
            }
            for d in 0..cfg.v_head_dim {
                let src = (hb + cfg.qk_nope_head_dim + d) * rank;
                let dst = (h * cfg.v_head_dim + d) * rank;
                w_vc[dst..dst + rank].copy_from_slice(&kv_b_w[src..src + rank]);
            }
        }

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
            w_kc,
            w_vc,
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
                let nope_off =
                    s * self.num_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim;
                let rope_off =
                    s * self.num_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim;

                q_nope_v[nope_off..nope_off + self.qk_nope_head_dim]
                    .copy_from_slice(&q_full_v[in_off..in_off + self.qk_nope_head_dim]);
                q_rope_v[rope_off..rope_off + self.qk_rope_head_dim].copy_from_slice(
                    &q_full_v[in_off + self.qk_nope_head_dim..in_off + total_q_head],
                );
            }
        }

        crate::qwen35::apply_rope_neox(
            &mut q_rope_v,
            positions,
            self.num_heads,
            self.qk_rope_head_dim,
            10000.0,
        );

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
            k_rope_v[s * self.qk_rope_head_dim..(s + 1) * self.qk_rope_head_dim].copy_from_slice(
                &kv_latent_v[in_off + kv_rank..in_off + kv_rank + self.qk_rope_head_dim],
            );
        }

        let kv_a_t = cpu_tensor(kv_a_v, Shape::new(vec![seq_len, kv_rank]));
        let kv_a_normed = self.kv_a_layernorm.forward(&kv_a_t)?;

        crate::qwen35::apply_rope_neox(&mut k_rope_v, positions, 1, self.qk_rope_head_dim, 10000.0);

        // 3. Absorb the per-head key up-projection (w_kc) into the query so
        //    attention runs entirely in latent space:
        //    q_absorbed[s,h] = q_nope[s,h] @ w_kc[h]^T
        //    (q_nope · (w_kc c) == (q_nope w_kc) · c.)
        let kv_a_normed_v = kv_a_normed.to_vec_f32()?;
        let rank = kv_rank;
        let nope = self.qk_nope_head_dim;
        let rope_d = self.qk_rope_head_dim;
        let vd = self.v_head_dim;
        let nh = self.num_heads;

        let mut q_absorbed = vec![0.0f32; seq_len * nh * rank];
        for s in 0..seq_len {
            for h in 0..nh {
                for d in 0..nope {
                    let qv = q_nope_v[(s * nh + h) * nope + d];
                    let wrow = &self.w_kc[(h * nope + d) * rank..(h * nope + d + 1) * rank];
                    let dst = &mut q_absorbed[(s * nh + h) * rank..(s * nh + h + 1) * rank];
                    for (o, w) in dst.iter_mut().zip(wrow.iter()) {
                        *o += qv * w;
                    }
                }
            }
        }

        // 4. Compressed latent rows for this step: [normed c_kv || roped k_pe].
        let mut latent_new = vec![0.0f32; seq_len * (rank + rope_d)];
        for s in 0..seq_len {
            let dst = s * (rank + rope_d);
            latent_new[dst..dst + rank]
                .copy_from_slice(&kv_a_normed_v[s * rank..(s + 1) * rank]);
            latent_new[dst + rank..dst + rank + rope_d]
                .copy_from_slice(&k_rope_v[s * rope_d..(s + 1) * rope_d]);
        }

        // 5. Append to the latent KV cache. Format: `.0` holds the compressed
        //    latent `[total_kv, rank + rope_d]`; `.1` is unused (kept only for
        //    the `(Tensor, Tensor)` cache shape the surrounding plumbing uses).
        let latent_all_v = match kv_cache.as_ref() {
            Some((prev_latent, _unused)) => {
                let mut v = prev_latent.to_vec_f32()?;
                v.extend(latent_new);
                v
            }
            None => latent_new,
        };
        let total_kv_len = latent_all_v.len() / (rank + rope_d);
        *kv_cache = Some((
            cpu_tensor(latent_all_v.clone(), Shape::new(vec![total_kv_len, rank + rope_d])),
            cpu_tensor(Vec::new(), Shape::new(vec![0, 0])),
        ));

        let scale = 1.0 / ((nope + rope_d) as f32).sqrt();

        // 6a. GPU decode fast path (decode-only kernel: grid = num_heads).
        if seq_len == 1 && x.device() != &Device::Cpu {
            if let Some(out_v) = self.gpu_absorbed_decode(
                &q_absorbed,
                &q_rope_v,
                &latent_all_v,
                rank,
                total_kv_len,
                scale,
                x.device(),
            ) {
                let attn_tensor = cpu_tensor(out_v, Shape::new(vec![seq_len, nh * vd]));
                return Ok(self.o_proj.forward(&attn_tensor)?);
            }
        }

        // 6b. Scalar latent-space reference path with causal masking. Query at
        // absolute position cache_offset + s attends only to t <= cache_offset + s.
        let cache_offset = total_kv_len - seq_len;
        let row = rank + rope_d;
        let mut attn_out = vec![0.0f32; seq_len * nh * vd];

        for s in 0..seq_len {
            let causal_limit = cache_offset + s;
            for h in 0..nh {
                let q_abs = &q_absorbed[(s * nh + h) * rank..(s * nh + h + 1) * rank];
                let q_rp = &q_rope_v[(s * nh + h) * rope_d..(s * nh + h + 1) * rope_d];

                let mut scores = vec![0.0f32; causal_limit + 1];
                for t in 0..=causal_limit {
                    let lb = t * row;
                    let dot_c: f32 = q_abs
                        .iter()
                        .zip(&latent_all_v[lb..lb + rank])
                        .map(|(a, b)| a * b)
                        .sum();
                    let dot_r: f32 = q_rp
                        .iter()
                        .zip(&latent_all_v[lb + rank..lb + row])
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[t] = (dot_c + dot_r) * scale;
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
                let weights: Vec<f32> = scores
                    .iter()
                    .map(|e| (e - max_score).exp() / (sum_exp + 1e-12))
                    .collect();

                let mut attn_latent = vec![0.0f32; rank];
                for t in 0..=causal_limit {
                    let w = weights[t];
                    for (o, l) in attn_latent
                        .iter_mut()
                        .zip(&latent_all_v[t * row..t * row + rank])
                    {
                        *o += w * l;
                    }
                }

                for d in 0..vd {
                    let wrow = &self.w_vc[(h * vd + d) * rank..(h * vd + d + 1) * rank];
                    attn_out[(s * nh + h) * vd + d] =
                        attn_latent.iter().zip(wrow.iter()).map(|(a, b)| a * b).sum();
                }
            }
        }

        let attn_tensor = cpu_tensor(
            attn_out,
            Shape::new(vec![seq_len, self.num_heads * self.v_head_dim]),
        );
        Ok(self.o_proj.forward(&attn_tensor)?)
    }

    /// GPU decode path via `BackendDevice::mla_absorbed_decode` (decode-only,
    /// `seq_len == 1`). Returns `None` when the backend cannot run the kernel;
    /// the caller then uses the scalar latent loop.
    ///
    /// `w_uv` is deliberately not passed: the kernel indexes `w_uv` without a
    /// per-head offset (`w_uv[v * kv_lora_rank + c]` for every head), so a
    /// full multi-head `w_uv` would repeat one head's rows for all heads.
    /// Instead the softmax-normalized latent is read back and projected per
    /// head through `w_vc` here.
    fn gpu_absorbed_decode(
        &self,
        q_absorbed: &[f32],
        q_rope: &[f32],
        latent_all: &[f32],
        rank: usize,
        total_kv_len: usize,
        scale: f32,
        device: &Device,
    ) -> Option<Vec<f32>> {
        use grim_nn::modules::pick_device_for_storage_device;

        let nh = self.num_heads;
        let rope_d = self.qk_rope_head_dim;
        let vd = self.v_head_dim;
        let dev = pick_device_for_storage_device(device);

        // The kernel applies a fixed 1/sqrt(rank + rope_d); pre-scale q so the
        // effective softmax scale stays the model's 1/sqrt(nope + rope_d).
        let kernel_scale = 1.0f32 / ((rank + rope_d) as f32).sqrt();
        let ratio = scale / kernel_scale;
        let q_abs_scaled: Vec<f32> = q_absorbed.iter().map(|v| v * ratio).collect();
        let q_rope_scaled: Vec<f32> = q_rope.iter().map(|v| v * ratio).collect();

        let qa_shape = Shape::new(vec![1, nh, rank]);
        let qr_shape = Shape::new(vec![1, nh, rope_d]);
        let kv_shape = Shape::new(vec![total_kv_len, 1, rank + rope_d]);
        let out_shape = Shape::new(vec![1, nh, rank]);

        let qa_st = dev.from_cpu(&q_abs_scaled, &qa_shape, DType::F32).ok()?;
        let qr_st = dev.from_cpu(&q_rope_scaled, &qr_shape, DType::F32).ok()?;
        let kv_st = dev.from_cpu(latent_all, &kv_shape, DType::F32).ok()?;
        let out_st = dev.zeros(&out_shape, DType::F32).ok()?;

        let handle = dev
            .mla_absorbed_decode(
                qa_st.as_ref(),
                qr_st.as_ref(),
                kv_st.as_ref(),
                None,
                out_st.as_ref(),
                nh,
                rank,
                rope_d,
                vd,
                total_kv_len,
            )
            .ok()?;
        handle.synchronize().ok()?;

        let out_t = Tensor::new(
            Arc::from(out_st),
            out_shape,
            DType::F32,
            QuantProvenance::default(),
            device.clone(),
        );
        let lat = out_t.to_vec_f32().ok()?;

        // Project the softmax-normalized latent per head through w_vc.
        let mut out = vec![0.0f32; nh * vd];
        for h in 0..nh {
            for d in 0..vd {
                let wrow = &self.w_vc[(h * vd + d) * rank..(h * vd + d + 1) * rank];
                out[h * vd + d] = wrow
                    .iter()
                    .zip(&lat[h * rank..(h + 1) * rank])
                    .map(|(w, l)| w * l)
                    .sum();
            }
        }
        Some(out)
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
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
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
            let exp = DeepSeek2Expert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
            )?;
            experts.push(exp);
        }

        let shared_experts = if cfg.n_shared_experts > 0 {
            let shared_ws = ws.scoped("shared_experts");
            let exp = DeepSeek2Expert::load(
                &shared_ws,
                cfg.hidden_size,
                cfg.moe_intermediate_size * cfg.n_shared_experts,
            )?;
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

            let max_l = topk
                .iter()
                .map(|(_, l)| *l)
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps
                .iter()
                .map(|e| (e / (sum_e + 1e-12)) * self.routed_scaling_factor)
                .collect();

            let token_x = cpu_tensor(
                xv[s * hidden_dim..(s + 1) * hidden_dim].to_vec(),
                Shape::new(vec![1, hidden_dim]),
            );

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
        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let self_attn = DeepSeek2Mla::load(&ws.scoped("self_attn"), cfg)?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let (mlp, moe) = if is_dense {
            let mlp =
                DeepSeek2Expert::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
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

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

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
    fn test_deepseek2_config() {
        let cfg = DeepSeek2Config::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.n_routed_experts, 64);
        assert_eq!(cfg.n_shared_experts, 2);
    }

    /// Latent-space (absorbed) MLA must match the OLD uncompressed per-head
    /// math: `q_nope · (w_kc c) == (q_nope w_kc) · c`. Covers seq_len=3 over
    /// kv_len=7 with causal masking (cache_offset = 4).
    #[test]
    fn test_mla_latent_attention_matches_uncompressed() {
        let seq = 3usize;
        let kv = 7usize;
        let nh = 4usize;
        let nope = 6usize;
        let rope_d = 4usize;
        let vd = 5usize;
        let rank = 8usize;
        let head = nope + vd;
        let scale = 1.0f32 / ((nope + rope_d) as f32).sqrt();

        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut q_nope = vec![0.0f32; seq * nh * nope];
        let mut q_rope = vec![0.0f32; seq * nh * rope_d];
        let mut w = vec![0.0f32; nh * head * rank]; // kv_b_proj weight layout
        let mut c_kv = vec![0.0f32; kv * rank]; // normed latent
        let mut k_rope = vec![0.0f32; kv * rope_d];
        for x in &mut q_nope {
            *x = rand();
        }
        for x in &mut q_rope {
            *x = rand();
        }
        for x in &mut w {
            *x = rand();
        }
        for x in &mut c_kv {
            *x = rand();
        }
        for x in &mut k_rope {
            *x = rand();
        }

        // w_kc[h] = weight rows [h*head .. +nope), w_vc[h] = rows [+nope .. +head).
        let w_row = |h: usize, d: usize| &w[(h * head + d) * rank..(h * head + d + 1) * rank];

        let cache_offset = kv - seq;
        let mut out_latent = vec![0.0f32; seq * nh * vd];
        let mut out_old = vec![0.0f32; seq * nh * vd];

        for s in 0..seq {
            let causal_limit = cache_offset + s;
            for h in 0..nh {
                // --- new (latent-space) path ---
                let q_abs: Vec<f32> = (0..rank)
                    .map(|c| {
                        (0..nope)
                            .map(|d| q_nope[(s * nh + h) * nope + d] * w_row(h, d)[c])
                            .sum()
                    })
                    .collect();
                let mut scores = vec![0.0f32; causal_limit + 1];
                for t in 0..=causal_limit {
                    let dot_c: f32 = q_abs
                        .iter()
                        .zip(&c_kv[t * rank..(t + 1) * rank])
                        .map(|(a, b)| a * b)
                        .sum();
                    let dot_r: f32 = q_rope[(s * nh + h) * rope_d..(s * nh + h + 1) * rope_d]
                        .iter()
                        .zip(&k_rope[t * rope_d..(t + 1) * rope_d])
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[t] = (dot_c + dot_r) * scale;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|v| (v - mx).exp()).sum();
                let weights: Vec<f32> = scores.iter().map(|v| (v - mx).exp() / sum).collect();

                let mut attn_latent = vec![0.0f32; rank];
                for t in 0..=causal_limit {
                    for (o, l) in attn_latent
                        .iter_mut()
                        .zip(&c_kv[t * rank..(t + 1) * rank])
                    {
                        *o += weights[t] * l;
                    }
                }
                for d in 0..vd {
                    out_latent[(s * nh + h) * vd + d] = attn_latent
                        .iter()
                        .zip(w_row(h, nope + d).iter())
                        .map(|(a, b)| a * b)
                        .sum();
                }

                // --- old (uncompressed) path: uncompress K/V, then attend ---
                let mut scores_old = vec![0.0f32; causal_limit + 1];
                for t in 0..=causal_limit {
                    let dot_nope: f32 = (0..nope)
                        .map(|d| {
                            let k_d: f32 = w_row(h, d)
                                .iter()
                                .zip(&c_kv[t * rank..(t + 1) * rank])
                                .map(|(a, b)| a * b)
                                .sum();
                            q_nope[(s * nh + h) * nope + d] * k_d
                        })
                        .sum();
                    let dot_r: f32 = q_rope[(s * nh + h) * rope_d..(s * nh + h + 1) * rope_d]
                        .iter()
                        .zip(&k_rope[t * rope_d..(t + 1) * rope_d])
                        .map(|(a, b)| a * b)
                        .sum();
                    scores_old[t] = (dot_nope + dot_r) * scale;
                }
                let mx_old = scores_old.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_old: f32 = scores_old.iter().map(|v| (v - mx_old).exp()).sum();
                let weights_old: Vec<f32> =
                    scores_old.iter().map(|v| (v - mx_old).exp() / sum_old).collect();

                for d in 0..vd {
                    let mut acc = 0.0f32;
                    for t in 0..=causal_limit {
                        let v_d: f32 = w_row(h, nope + d)
                            .iter()
                            .zip(&c_kv[t * rank..(t + 1) * rank])
                            .map(|(a, b)| a * b)
                            .sum();
                        acc += weights_old[t] * v_d;
                    }
                    out_old[(s * nh + h) * vd + d] = acc;
                }
            }
        }

        for (i, (&a, &b)) in out_latent.iter().zip(out_old.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "latent vs uncompressed mismatch at {i}: {a} vs {b}"
            );
        }
    }
}
