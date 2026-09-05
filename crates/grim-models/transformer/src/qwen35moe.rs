//! Qwen3.5-MoE architecture with YaRN RoPE support and fine-grained routed/shared experts.
//!
//! # Architecture Details
//! - **YaRN Frequency Scaling**: Decoupled high/low frequency interpolation on RoPE positional encodings.
//! - **Fine-Grained MoE**: Top-k softmax routing across $N$ routed experts plus dedicated shared expert pathways.
//! - **GQA Attention**: Grouped Query Attention with RMSNorm pre/post attention normalizations.

use grim_core::error::Result;
use grim_backend_cpu::cpu_tensor;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::moe::{ExpertBank, ExpertTriple, MoeFfn, MoeRouter, RouterKind};
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Qwen3.5-MoE transformer architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen35MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub routed_scaling_factor: f32,
    pub layer_types: Vec<String>,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub full_yarn: Option<YaRNParams>,
}

impl Default for Qwen35MoeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            num_layers: 24,
            intermediate_size: 1408,
            num_experts: 64,
            num_experts_per_tok: 8,
            shared_expert_intermediate_size: Some(1408),
            routed_scaling_factor: 1.0,
            layer_types: vec!["moe".into(); 24],
            linear_key_head_dim: 128,
            linear_num_key_heads: 4,
            linear_value_head_dim: 128,
            linear_num_value_heads: 4,
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_seq_len: 32768,
            full_yarn: None,
        }
    }
}

impl ModelConfig for Qwen35MoeConfig {
    fn name(&self) -> &str {
        "qwen35moe"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MoE Feed-Forward Layer
// ---------------------------------------------------------------------------

pub struct Qwen35MoeExpert {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Qwen35MoeExpert {
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
        let swiglu = grim_nn::modules::silu_mul_on_device(&gate, &up)
            .map_err(grim_core::error::Error::from)?;
        Ok(self.down_proj.forward(&swiglu)?)
    }
}

/// Qwen3.5-MoE feed-forward: routes through the shared `MoeFfn` so ROCm
/// serving gets the fused Charon dispatch (top-k router + routed experts +
/// shared expert) instead of a per-token host loop. Weight layout is
/// unchanged (`mlp.gate`, `mlp.experts.{e}.gate_proj|up_proj|down_proj`,
/// `mlp.shared_expert.*`).
pub struct Qwen35MoeLayer {
    pub ffn: MoeFfn,
}

impl Qwen35MoeLayer {
    pub fn load(ws: &WeightSource<'_>, cfg: &Qwen35MoeConfig) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;

        let shared_expert = if let Some(shared_dim) = cfg.shared_expert_intermediate_size {
            Some(Qwen35MoeExpert::load(
                &ws.scoped("shared_expert"),
                cfg.hidden_size,
                shared_dim,
            )?)
        } else {
            None
        };

        let mut gates = Vec::with_capacity(cfg.num_experts);
        let mut ups = Vec::with_capacity(cfg.num_experts);
        let mut downs = Vec::with_capacity(cfg.num_experts);
        let exp_ws = ws.scoped("experts");
        for e in 0..cfg.num_experts {
            let exp = Qwen35MoeExpert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            let Qwen35MoeExpert {
                gate_proj,
                up_proj,
                down_proj,
            } = exp;
            gates.push(gate_proj);
            ups.push(up_proj);
            downs.push(down_proj);
        }

        let router = MoeRouter::new(
            gate,
            RouterKind::SoftmaxTopK,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            None,
        );
        let shared = shared_expert.map(|s| ExpertTriple {
            gate: s.gate_proj,
            up: s.up_proj,
            down: s.down_proj,
            inter: cfg
                .shared_expert_intermediate_size
                .unwrap_or(cfg.intermediate_size),
            hidden: cfg.hidden_size,
        });

        Ok(Self {
            ffn: MoeFfn::new(
                router,
                ExpertBank::from_linears(gates, ups, downs),
                shared,
                cfg.routed_scaling_factor,
            ),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.ffn.forward(x).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct Qwen35MoeBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub moe: Qwen35MoeLayer,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Qwen35MoeBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Qwen35MoeConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

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

        let moe = Qwen35MoeLayer::load(&ws.scoped("mlp"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attn_norm,
            ffn_norm,
            moe,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first forward: Q/K RoPE, KV-cache concat, attention and the
    /// residual adds run on the tensor's device. Host paths are only reached
    /// through the fused-kernel fallback guards and the (host-side) MoE
    /// routing pull.
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

        let q = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &q,
            self.num_heads,
            positions,
        )?;
        let k = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &k,
            self.num_kv_heads,
            positions,
        )?;

        // Device-side history: prev rows stay resident, only the new rows
        // are appended (D2D arena copy when the backend supports it).
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let full_k = crate::shared_attention::concat_rows_on_device(prev_k, &k)?;
            let full_v = crate::shared_attention::concat_rows_on_device(prev_v, &v)?;
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k.clone(), v.clone()));
            (k.clone(), v.clone())
        };
        let kv_len = k_all.shape().dims()[0];

        // Shared helper applies the causal mask at cache_offset + s (fixes
        // future-token leakage during multi-token prefill).
        let attn_tensor = crate::shared_attention::fused_attention_tensors(
            &q,
            &k_all,
            &v_all,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            kv_len,
            None,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.ffn_norm.forward(&res1)?;
        let mlp_out = self.moe.forward(&normed_ffn)?;
        // Routing stays host-side, so the MoE output lands on the host; stage
        // it back next to `res1` before the residual add.
        let mlp_out = grim_nn::modules::move_to_device(&mlp_out, x.device())?;

        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct Qwen35Moe {
    pub cfg: Qwen35MoeConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<Qwen35MoeBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Qwen35Moe {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen35MoeConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen35MoeConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = Qwen35MoeBlock::load(&layer_ws, &cfg, tp)?;
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

impl Model for Qwen35Moe {
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

impl CausalLm for Qwen35Moe {
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
        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let ids: Vec<u32> = ids_f32.iter().map(|&t| t as u32).collect();
        let pos_v: Vec<u32> = positions
            .to_vec_f32()
            .map(|v| v.into_iter().map(|p| p as u32).collect())
            .unwrap_or_else(|_| (0..seq_len as u32).collect());

        // GPU-first embedding gather: rows land on the weight's device; the
        // vocab×hidden table never crosses to host.
        let mut x = grim_nn::embedding_gather_on_device(
            &self.tok_embeddings.weight,
            &ids,
            seq_len,
            self.cfg.hidden_size,
        )?;

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
    fn test_qwen35moe_config() {
        let cfg = Qwen35MoeConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
    }

    /// Parity gate for the MoeFfn migration: the layer must reproduce the
    /// original per-token host algorithm (softmax top-k routing, weighted
    /// expert sum scaled by routed_scaling_factor, shared expert added
    /// unconditionally) — that algorithm is what the fused Charon kernel
    /// matches bit-for-bit on ROCm.
    #[test]
    fn test_qwen35moe_layer_matches_host_reference() {
        let hidden = 4usize;
        let inter = 4usize;
        let num_experts = 3usize;
        let top_k = 2usize;
        let scaling = 1.7f32;

        let lin = |rows: usize, cols: usize, seed: f32| {
            Linear::from_tensor(
                cpu_tensor(
                    (0..rows * cols)
                        .map(|i| ((i as f32 + seed).sin()) * 0.4)
                        .collect(),
                    Shape::new(vec![rows, cols]),
                ),
                None,
            )
        };

        let router = MoeRouter::new(
            lin(num_experts, hidden, 11.0),
            RouterKind::SoftmaxTopK,
            top_k,
            num_experts,
            None,
        );
        let gates: Vec<Linear> = (0..num_experts).map(|e| lin(hidden, inter, e as f32)).collect();
        let ups: Vec<Linear> = (0..num_experts).map(|e| lin(hidden, inter, e as f32 + 0.3)).collect();
        let downs: Vec<Linear> = (0..num_experts).map(|e| lin(inter, hidden, e as f32 + 0.6)).collect();
        let shared = ExpertTriple {
            gate: lin(hidden, inter, 9.1),
            up: lin(hidden, inter, 9.4),
            down: lin(inter, hidden, 9.7),
            inter,
            hidden,
        };
        let layer = Qwen35MoeLayer {
            ffn: MoeFfn::new(
                router,
                ExpertBank::from_linears(gates.clone(), ups.clone(), downs.clone()),
                Some(shared),
                scaling,
            ),
        };

        let swiglu = |g: &[f32], u: &[f32]| -> Vec<f32> {
            g.iter()
                .zip(u.iter())
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect()
        };
        let forward_expert = |g: &Linear, u: &Linear, d: &Linear, x: &[f32]| -> Vec<f32> {
            let mm = |w: &[f32], r: usize, c: usize, v: &[f32]| -> Vec<f32> {
                (0..r)
                    .map(|o| (0..c).map(|k| v[k] * w[o * c + k]).sum::<f32>())
                    .collect()
            };
            // Linear::from_tensor stores [out, in]; forward = x @ w_t.
            let g_out = mm(&g.weight().to_vec_f32().unwrap(), inter, hidden, x);
            let u_out = mm(&u.weight().to_vec_f32().unwrap(), inter, hidden, x);
            let act = swiglu(&g_out, &u_out);
            mm(&d.weight().to_vec_f32().unwrap(), hidden, inter, &act)
        };

        let x: Vec<f32> = vec![0.5f32, -0.3, 0.9, -0.7];
        let rows = 1usize;
        let x_t = cpu_tensor(x.clone(), Shape::new(vec![rows, hidden]));

        // Reference algorithm (the pre-migration host loop). Router gate was
        // stored [num_experts, hidden], so logit_e = x . w_row_e.
        let gw: Vec<f32> = (0..num_experts * hidden)
            .map(|i| ((i as f32 + 11.0).sin()) * 0.4)
            .collect();
        let shared_triple = (
            lin(hidden, inter, 9.1),
            lin(hidden, inter, 9.4),
            lin(inter, hidden, 9.7),
        );
        let mut expected = vec![0.0f32; rows * hidden];
        for s in 0..rows {
            let token = &x[s * hidden..(s + 1) * hidden];
            let mut row_logits: Vec<(usize, f32)> = (0..num_experts)
                .map(|e| {
                    let dot: f32 = (0..hidden).map(|k| token[k] * gw[e * hidden + k]).sum::<f32>();
                    (e, dot)
                })
                .collect();
            row_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top = &row_logits[..top_k];
            let max_l = top.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = top.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (_i, ((e, _), ex)) in top.iter().zip(exps.iter()).enumerate() {
                let w = ex / sum * scaling;
                let eo = forward_expert(&gates[*e], &ups[*e], &downs[*e], token);
                for dd in 0..hidden {
                    expected[s * hidden + dd] += w * eo[dd];
                }
            }
            let so = forward_expert(&shared_triple.0, &shared_triple.1, &shared_triple.2, token);
            for dd in 0..hidden {
                expected[s * hidden + dd] += so[dd];
            }
        }

        let out = layer.forward(&x_t).unwrap().to_vec_f32().unwrap();
        for (i, (&act, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (act - exp).abs() < 1e-4,
                "qwen35moe layer diverges from host reference at [{i}]: {act} vs {exp}"
            );
        }
    }
}
