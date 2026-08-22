//! Compatibility loader and native implementation for `moonshotai/Kimi-K3`.
//!
//! # Architecture Details
//! - **Multi-Head Latent Attention (MLA)**: Dual Q-LoRA (`q_a_proj -> q_b_proj`) and KV-LoRA (`kv_a_proj_with_mqa -> kv_b_proj`) latent projections.
//! - **Kimi MoE**: 64-expert Mixture of Experts with top-6 routing and scaling factor.
//! - **SwiGLU Experts**: $w_1$ (gate), $w_3$ (up), and $w_2$ (down) feed-forward branches.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Kimi-K3 architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiK3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for KimiK3Config {
    fn default() -> Self {
        Self {
            vocab_size: 163840,
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 128,
            num_hidden_layers: 28,
            q_lora_rank: 256,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_experts: 64,
            num_experts_per_tok: 6,
            intermediate_size: 1408,
            routed_scaling_factor: 2.0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

impl ModelConfig for KimiK3Config {
    fn name(&self) -> &str {
        "kimi_k3"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl KimiK3Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        KimiK3Config {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_hidden_layers: u("num_hidden_layers"),
            q_lora_rank: u("q_lora_rank"),
            kv_lora_rank: u("kv_lora_rank"),
            qk_nope_head_dim: u("qk_nope_head_dim"),
            qk_rope_head_dim: u("qk_rope_head_dim"),
            v_head_dim: u("v_head_dim"),
            num_experts: u("num_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            intermediate_size: if u("intermediate_size") > 0 {
                u("intermediate_size")
            } else {
                1408
            },
            routed_scaling_factor: if f("routed_scaling_factor") > 0.0 {
                f("routed_scaling_factor")
            } else {
                2.0
            },
            rms_norm_eps: if f("rms_norm_eps") > 0.0 {
                f("rms_norm_eps")
            } else {
                1e-6
            },
            rope_theta: if f("rope_theta") > 0.0 {
                f("rope_theta")
            } else {
                10000.0
            },
        }
    }
}

// ---------------------------------------------------------------------------
// MLA Attention Block
// ---------------------------------------------------------------------------

pub struct KimiK3Mla {
    pub q_a_proj: Linear,
    pub q_b_proj: Linear,
    pub kv_a_proj: Linear,
    pub kv_b_proj: Linear,
    pub o_proj: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
}

impl KimiK3Mla {
    pub fn load(ws: &WeightSource<'_>, cfg: &KimiK3Config) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
        let q_a_proj =
            Linear::load_shape(&ws.scoped("q_a_proj"), [cfg.hidden_size, cfg.q_lora_rank])?;
        let q_b_proj = Linear::load_shape(&ws.scoped("q_b_proj"), [cfg.q_lora_rank, q_dim])?;

        let kv_a_proj = Linear::load_shape(
            &ws.scoped("kv_a_proj_with_mqa"),
            [cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim],
        )?;
        let kv_b_proj = Linear::load_shape(
            &ws.scoped("kv_b_proj"),
            [
                cfg.kv_lora_rank,
                cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            ],
        )?;
        let o_proj = Linear::load_shape(
            &ws.scoped("o_proj"),
            [cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta);

        Ok(Self {
            q_a_proj,
            q_b_proj,
            kv_a_proj,
            kv_b_proj,
            o_proj,
            rope,
            num_heads: cfg.num_attention_heads,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
            q_lora_rank: cfg.q_lora_rank,
            kv_lora_rank: cfg.kv_lora_rank,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];

        // 1. Q projection
        let q_lat = self.q_a_proj.forward(x)?;
        let q_full = self.q_b_proj.forward(&q_lat)?;
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

        let mut kv_a_v = vec![0.0f32; seq_len * self.kv_lora_rank];
        let mut k_rope_v = vec![0.0f32; seq_len * self.qk_rope_head_dim];

        for s in 0..seq_len {
            let in_off = s * (self.kv_lora_rank + self.qk_rope_head_dim);
            kv_a_v[s * self.kv_lora_rank..(s + 1) * self.kv_lora_rank]
                .copy_from_slice(&kv_latent_v[in_off..in_off + self.kv_lora_rank]);
            k_rope_v[s * self.qk_rope_head_dim..(s + 1) * self.qk_rope_head_dim].copy_from_slice(
                &kv_latent_v[in_off + self.kv_lora_rank
                    ..in_off + self.kv_lora_rank + self.qk_rope_head_dim],
            );
        }

        let kv_a_t = cpu_tensor(kv_a_v, Shape::new(vec![seq_len, self.kv_lora_rank]));
        crate::qwen35::apply_rope_neox(&mut k_rope_v, positions, 1, self.qk_rope_head_dim, 10000.0);

        let kv_b = self.kv_b_proj.forward(&kv_a_t)?;
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

        let (k_all, v_all, k_rope_all) = if let Some((prev_k, prev_v, prev_rope)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            let mut new_rope = prev_rope.to_vec_f32()?;
            new_k.extend(k_nope_v);
            new_v.extend(v_v);
            new_rope.extend(k_rope_v);
            let total_k_dim = self.num_heads * self.qk_nope_head_dim;
            let total_v_dim = self.num_heads * self.v_head_dim;
            let total_seq = new_k.len() / total_k_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, total_k_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, total_v_dim]));
            let full_rope =
                cpu_tensor(new_rope, Shape::new(vec![total_seq, self.qk_rope_head_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone(), full_rope.clone()));
            (full_k, full_v, full_rope)
        } else {
            let total_k_dim = self.num_heads * self.qk_nope_head_dim;
            let total_v_dim = self.num_heads * self.v_head_dim;
            let full_k = cpu_tensor(k_nope_v, Shape::new(vec![seq_len, total_k_dim]));
            let full_v = cpu_tensor(v_v, Shape::new(vec![seq_len, total_v_dim]));
            let full_rope = cpu_tensor(k_rope_v, Shape::new(vec![seq_len, self.qk_rope_head_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone(), full_rope.clone()));
            (full_k, full_v, full_rope)
        };

        let total_kv_len = k_all.shape().dims()[0];
        let k_all_v = k_all.to_vec_f32()?;
        let v_all_v = v_all.to_vec_f32()?;
        let k_rope_all_v = k_rope_all.to_vec_f32()?;
        // Causal mask: query s (absolute position cache_offset + s) must not
        // see future in-chunk keys.
        let cache_offset = total_kv_len.saturating_sub(seq_len);

        let scale = 1.0 / ((self.qk_nope_head_dim + self.qk_rope_head_dim) as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * self.num_heads * self.v_head_dim];

        for s in 0..seq_len {
            let causal_limit = cache_offset + s;
            for h in 0..self.num_heads {
                let q_nope_slice = &q_nope_v[s * self.num_heads * self.qk_nope_head_dim
                    + h * self.qk_nope_head_dim
                    ..s * self.num_heads * self.qk_nope_head_dim + (h + 1) * self.qk_nope_head_dim];
                let q_rope_slice = &q_rope_v[s * self.num_heads * self.qk_rope_head_dim
                    + h * self.qk_rope_head_dim
                    ..s * self.num_heads * self.qk_rope_head_dim + (h + 1) * self.qk_rope_head_dim];

                let mut scores = vec![f32::NEG_INFINITY; total_kv_len];
                for t in 0..=causal_limit {
                    let k_nope_slice = &k_all_v[t * self.num_heads * self.qk_nope_head_dim
                        + h * self.qk_nope_head_dim
                        ..t * self.num_heads * self.qk_nope_head_dim
                            + (h + 1) * self.qk_nope_head_dim];
                    let k_rope_slice =
                        &k_rope_all_v[t * self.qk_rope_head_dim..(t + 1) * self.qk_rope_head_dim];

                    let dot_nope: f32 = q_nope_slice
                        .iter()
                        .zip(k_nope_slice.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let dot_rope: f32 = q_rope_slice
                        .iter()
                        .zip(k_rope_slice.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[t] = (dot_nope + dot_rope) * scale;
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / (sum_exp + 1e-12)).collect();

                for d in 0..self.v_head_dim {
                    let mut acc = 0.0f32;
                    for t in 0..total_kv_len {
                        let v_val =
                            v_all_v[t * self.num_heads * self.v_head_dim + h * self.v_head_dim + d];
                        acc += weights[t] * v_val;
                    }
                    attn_out[s * self.num_heads * self.v_head_dim + h * self.v_head_dim + d] = acc;
                }
            }
        }

        let attn_tensor = cpu_tensor(
            attn_out,
            Shape::new(vec![seq_len, self.num_heads * self.v_head_dim]),
        );
        Ok(self.o_proj.forward(&attn_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// MoE Feed-Forward Layer
// ---------------------------------------------------------------------------

pub struct KimiK3Expert {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl KimiK3Expert {
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let gate_proj =
            Linear::load_shape(&ws.scoped("gate_proj"), [hidden_size, intermediate_size])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [hidden_size, intermediate_size])?;
        let down_proj =
            Linear::load_shape(&ws.scoped("down_proj"), [intermediate_size, hidden_size])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gv = gate.to_vec_f32()?;
        let uv = up.to_vec_f32()?;
        let swiglu: Vec<f32> = gv
            .iter()
            .zip(uv.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let swiglu_t = cpu_tensor(swiglu, gate.shape().clone());
        Ok(self.down_proj.forward(&swiglu_t)?)
    }
}

pub struct KimiK3Moe {
    pub gate: Linear,
    pub experts: Vec<KimiK3Expert>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
}

impl KimiK3Moe {
    pub fn load(ws: &WeightSource<'_>, cfg: &KimiK3Config) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;

        let intermediate_size = (cfg.hidden_size * 8 / 3) / 8 * 8;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        let exp_ws = ws.scoped("experts");
        for e in 0..cfg.num_experts {
            let exp = KimiK3Expert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                intermediate_size,
            )?;
            experts.push(exp);
        }

        Ok(Self {
            gate,
            experts,
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

        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct KimiK3Block {
    pub attn_norm: RmsNorm,
    pub self_attn: KimiK3Mla,
    pub ffn_norm: RmsNorm,
    pub moe: KimiK3Moe,
}

impl KimiK3Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &KimiK3Config) -> Result<Self> {
        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let self_attn = KimiK3Mla::load(&ws.scoped("self_attn"), cfg)?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let moe = KimiK3Moe::load(&ws.scoped("block_sparse_moe"), cfg)?;

        Ok(Self {
            attn_norm,
            self_attn,
            ffn_norm,
            moe,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let normed_attn = self.attn_norm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed_attn, positions, kv_cache)?;

        let xv = x.to_vec_f32()?;
        let av = attn_out.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;
        let mlp_out = self.moe.forward(&normed_ffn)?;

        let r1v = res1_t.to_vec_f32()?;
        let mv = mlp_out.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct KimiK3 {
    pub cfg: KimiK3Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<KimiK3Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl KimiK3 {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: KimiK3Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: KimiK3Config,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = KimiK3Block::load(&layer_ws, &cfg)?;
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

impl Model for KimiK3 {
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

impl CausalLm for KimiK3 {
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
    use grim_core::architecture::ModelArchitecture;

    const KIMI_K3_CONFIG: &str = r#"{
        "architectures": ["KimiK3ForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 6,
        "routed_scaling_factor": 2.0,
        "rms_norm_eps": 1e-06,
        "vocab_size": 163840
    }"#;

    #[test]
    fn parses_kimi_k3_config() {
        let v: serde_json::Value = serde_json::from_str(KIMI_K3_CONFIG).unwrap();
        let cfg = KimiK3Config::from_hf(&v);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.name(), "kimi_k3");
    }

    #[test]
    fn dispatches_kimi_k3_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("kimi_k3"),
            ModelArchitecture::KimiK3
        );
    }

    // --- WI-X11 regression probes (tiny synthetic MLA block) ---

    fn tiny_mla() -> KimiK3Mla {
        use grim_backend_cpu::cpu_tensor;
        use grim_nn::Linear;
        let (hidden, ql, rank, nh, nope, rope_d, vd) =
            (8usize, 4usize, 6usize, 2usize, 4usize, 2usize, 3usize);
        let q_dim = nh * (nope + rope_d);
        let kv_b_out = nh * (nope + vd);
        let mut seed = 0xC0FFEEu64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut w = |rows: usize, cols: usize| {
            let data: Vec<f32> = (0..rows * cols).map(|_| rand()).collect();
            Linear::from_tensor(cpu_tensor(data, Shape::new(vec![rows, cols])), None)
        };
        KimiK3Mla {
            q_a_proj: w(ql, hidden),
            q_b_proj: w(q_dim, ql),
            kv_a_proj: w(rank + rope_d, hidden),
            kv_b_proj: w(kv_b_out, rank),
            o_proj: w(hidden, nh * vd),
            rope: Rope::new(rope_d, 10000.0),
            num_heads: nh,
            qk_nope_head_dim: nope,
            qk_rope_head_dim: rope_d,
            v_head_dim: vd,
            q_lora_rank: ql,
            kv_lora_rank: rank,
        }
    }

    fn cpu_x(rows: usize, cols: usize) -> Tensor {
        let mut seed = 0xBEEFu64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let data: Vec<f32> = (0..rows * cols).map(|_| rand()).collect();
        cpu_tensor(data, Shape::new(vec![rows, cols]))
    }

    /// WI-X11a: decode must attend with the CACHED rope keys, not the current
    /// call's buffer. Probe: poison every cached rope row after prefill; a
    /// correct decode output changes. (The old bug read `k_rope_v[0..]` from
    /// the current call and was blind to the cache.)
    #[test]
    fn decode_uses_cached_rope_keys() {
        let mla = tiny_mla();
        let mut cache: Option<(Tensor, Tensor, Tensor)> = None;
        // Prefill 3 tokens at positions 0..2.
        let _ = mla.forward(&cpu_x(3, 8), &[0, 1, 2], &mut cache).unwrap();
        let (k0, v0, r0) = cache.as_ref().unwrap();
        let baseline = {
            let mut c = Some((
                cpu_tensor(k0.to_vec_f32().unwrap(), k0.shape().clone()),
                cpu_tensor(v0.to_vec_f32().unwrap(), v0.shape().clone()),
                cpu_tensor(r0.to_vec_f32().unwrap(), r0.shape().clone()),
            ));
            mla.forward(&cpu_x(1, 8), &[3], &mut c)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        };
        // Poison the cached rope rows only.
        let poisoned = vec![7.5f32; r0.to_vec_f32().unwrap().len()];
        let mut c = Some((
            cpu_tensor(k0.to_vec_f32().unwrap(), k0.shape().clone()),
            cpu_tensor(v0.to_vec_f32().unwrap(), v0.shape().clone()),
            cpu_tensor(poisoned, r0.shape().clone()),
        ));
        let perturbed = mla
            .forward(&cpu_x(1, 8), &[3], &mut c)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert!(
            baseline
                .iter()
                .zip(&perturbed)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "decode output ignored cached rope keys — rope-history bug regressed"
        );
    }

    /// WI-X11b: causal mask — the FIRST query of a multi-token call must not
    /// see later tokens of the same call. Probe: change only the second
    /// token's input; query 0's output must be bit-identical, query 1's must
    /// differ (it attends to itself).
    #[test]
    fn causal_mask_blocks_future_keys() {
        let mla = tiny_mla();
        let mut cache: Option<(Tensor, Tensor, Tensor)> = None;
        // Seed 2 past tokens (positions 0,1).
        let _ = mla.forward(&cpu_x(2, 8), &[0, 1], &mut cache).unwrap();
        let (k0, v0, r0) = cache.as_ref().unwrap();
        let clone_cache = || {
            (
                cpu_tensor(k0.to_vec_f32().unwrap(), k0.shape().clone()),
                cpu_tensor(v0.to_vec_f32().unwrap(), v0.shape().clone()),
                cpu_tensor(r0.to_vec_f32().unwrap(), r0.shape().clone()),
            )
        };
        // Two new tokens at absolute positions 2,3. `xb` differs from `xa`
        // ONLY in the second token's hidden vector (row 1).
        let xa = cpu_x(2, 8);
        let mut xb = xa.to_vec_f32().unwrap();
        for e in xb[8..].iter_mut() {
            *e *= -1.0;
        }
        let xb = cpu_tensor(xb, Shape::new(vec![2, 8]));
        let base = {
            let mut c = Some(clone_cache());
            mla.forward(&xa, &[2, 3], &mut c)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        };
        let perturbed = {
            let mut c = Some(clone_cache());
            mla.forward(&xb, &[2, 3], &mut c)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        };
        let head_dims = mla.num_heads * mla.v_head_dim;
        assert!(
            base[..head_dims]
                .iter()
                .zip(&perturbed[..head_dims])
                .all(|(a, b)| (a - b).abs() < 1e-5),
            "query at abs pos 2 changed when the LATER token changed — future leak"
        );
        assert!(
            base[head_dims..]
                .iter()
                .zip(&perturbed[head_dims..])
                .any(|(a, b)| (a - b).abs() > 1e-5),
            "second query ignored its own token — probe inert"
        );
    }
}
