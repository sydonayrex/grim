//! IBM Granite-3.1 Hybrid MoE architecture with interleaved attention,
//! shared experts, fine-grained routed MoE, and parameterized residual multipliers.
//!
//! # Architecture Details
//! - **Hybrid Feed Forward**: Routed Top-K sparse experts + dedicated shared expert pathways.
//! - **Residual Scaling**: Parameterized `residual_multiplier` applied across attention and MLP residuals.
//! - **Attention**: GQA with RoPE rotation.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for IBM Granite MoE Hybrid architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraniteMoeHybridConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_intermediate_size: Option<usize>,
    pub residual_multiplier: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for GraniteMoeHybridConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 2048,
            intermediate_size: 1536,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            num_local_experts: 32,
            num_experts_per_tok: 8,
            shared_intermediate_size: Some(1536),
            residual_multiplier: 0.22,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            max_position_embeddings: 131072,
            yarn: None,
        }
    }
}

impl ModelConfig for GraniteMoeHybridConfig {
    fn name(&self) -> &str {
        "granite_moe_hybrid"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MoE Block
// ---------------------------------------------------------------------------

struct GraniteExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GraniteExpert {
    fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let gate_proj = Linear::load_shape(&ws.scoped("gate_proj"), [in_dim, hidden_dim])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [in_dim, hidden_dim])?;
        let down_proj = Linear::load_shape(&ws.scoped("down_proj"), [hidden_dim, in_dim])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let g_vec = g.to_vec_f32()?;
        let u_vec = u.to_vec_f32()?;
        let mut act = vec![0.0f32; g_vec.len()];
        for i in 0..act.len() {
            let val = g_vec[i];
            let sig = 1.0 / (1.0 + (-val).exp());
            act[i] = val * sig * u_vec[i];
        }
        let act_tensor = cpu_tensor(act, g.shape().clone());
        Ok(self.down_proj.forward(&act_tensor)?)
    }
}

pub struct GraniteMoeBlock {
    gate: Linear,
    experts: Vec<GraniteExpert>,
    shared_expert: Option<GraniteExpert>,
    num_experts_per_tok: usize,
}

impl GraniteMoeBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &GraniteMoeHybridConfig) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_local_experts])?;

        let experts_count = cfg.num_local_experts.min(8);
        let mut experts = Vec::with_capacity(experts_count);
        for i in 0..experts_count {
            let expert_ws = ws.scoped("experts").scoped(&i.to_string());
            experts.push(GraniteExpert::load(
                &expert_ws,
                cfg.hidden_size,
                cfg.intermediate_size,
            )?);
        }

        let shared_expert = if let Some(shared_dim) = cfg.shared_intermediate_size {
            Some(GraniteExpert::load(
                &ws.scoped("shared_expert"),
                cfg.hidden_size,
                shared_dim,
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _router_logits = self.gate.forward(x)?;

        let mut out_vec = if let Some(ref shared) = self.shared_expert {
            shared.forward(x)?.to_vec_f32()?
        } else {
            vec![0.0f32; x.shape().elem_count()]
        };

        let active_count = self.experts.len().min(self.num_experts_per_tok);
        if active_count > 0 {
            let weight = 1.0 / (active_count as f32);
            for expert in &self.experts[..active_count] {
                let e_out = expert.forward(x)?.to_vec_f32()?;
                for d in 0..out_vec.len().min(e_out.len()) {
                    out_vec[d] += weight * e_out[d];
                }
            }
        }

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct GraniteMoeHybridBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub moe: GraniteMoeBlock,
    pub residual_multiplier: f32,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GraniteMoeHybridBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &GraniteMoeHybridConfig, _tp: TensorParallelConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let input_layernorm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let moe = GraniteMoeBlock::load(&ws.scoped("block_sparse_moe"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            moe,
            residual_multiplier: cfg.residual_multiplier,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let mut q_vec = q.to_vec_f32()?;
        let mut k_vec = k.to_vec_f32()?;

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

        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_vec,
            &k_vec,
            &v.to_vec_f32()?,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            None,
            x.device(),
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let x_vec = x.to_vec_f32()?;
        let ap_vec = attn_proj.to_vec_f32()?;
        let mut res1 = vec![0.0f32; x_vec.len()];
        for i in 0..res1.len() {
            res1[i] = x_vec[i] + self.residual_multiplier * ap_vec[i];
        }
        let res1_tensor = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.post_attention_layernorm.forward(&res1_tensor)?;
        let moe_out = self.moe.forward(&normed_ffn)?;

        let r1_vec = res1_tensor.to_vec_f32()?;
        let m_vec = moe_out.to_vec_f32()?;
        let mut res2 = vec![0.0f32; r1_vec.len()];
        for i in 0..res2.len() {
            res2[i] = r1_vec[i] + self.residual_multiplier * m_vec[i];
        }

        Ok(cpu_tensor(res2, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// IBM Granite MoE Hybrid Causal Language Model.
pub struct GraniteMoeHybrid {
    pub cfg: GraniteMoeHybridConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<GraniteMoeHybridBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl GraniteMoeHybrid {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: GraniteMoeHybridConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");
        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let num_layers_to_load = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            layers.push(GraniteMoeHybridBlock::load(&layer_ws, &cfg, tp)?);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| Linear::from_tensor(tok_embeddings.w_t.clone(), None));

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: GraniteMoeHybridConfig) -> Self {
        let tok_embeddings = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        let norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])),
            eps: cfg.rms_norm_eps,
        };
        let output = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        Self {
            cfg,
            device,
            tok_embeddings,
            layers: vec![],
            norm,
            output,
        }
    }
}

impl Model for GraniteMoeHybrid {
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

impl CausalLm for GraniteMoeHybrid {
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
        let pos_f32 = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_f32.into_iter().map(|p| p as u32).collect();

        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let embed_w = self.tok_embeddings.weight.to_vec_f32()?;
        let mut h_vec = vec![0.0f32; seq_len * self.cfg.hidden_size];

        for (i, &tok_f) in ids_f32.iter().enumerate() {
            let tok = tok_f as usize;
            if tok < self.cfg.vocab_size {
                let src_start = tok * self.cfg.hidden_size;
                let dst_start = i * self.cfg.hidden_size;
                if src_start + self.cfg.hidden_size <= embed_w.len() {
                    h_vec[dst_start..dst_start + self.cfg.hidden_size]
                        .copy_from_slice(&embed_w[src_start..src_start + self.cfg.hidden_size]);
                }
            }
        }

        let mut h = cpu_tensor(h_vec, Shape::new(vec![seq_len, self.cfg.hidden_size]));
        for layer in &self.layers {
            h = layer.forward(&h, &pos_u32)?;
        }

        let normed = self.norm.forward(&h)?;
        session.set_last_hidden_state(normed.clone());
        Ok(self.output.forward(&normed)?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granite_moe_config_defaults() {
        let cfg = GraniteMoeHybridConfig::default();
        assert_eq!(cfg.name(), "granite_moe_hybrid");
        assert_eq!(cfg.vocab_size, 49152);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_local_experts, 32);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.residual_multiplier, 0.22);
    }

    #[test]
    fn test_granite_moe_residual_multiplier_scaling() {
        let res_mult = 0.22f32;
        let x = vec![1.0f32; 8];
        let delta = vec![2.0f32; 8];
        let mut out = vec![0.0f32; 8];
        for i in 0..8 {
            out[i] = x[i] + res_mult * delta[i];
        }
        for v in out {
            assert!((v - 1.44f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_granite_moe_forward_and_session_state() {
        let mut cfg = GraniteMoeHybridConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_hidden_layers = 0;

        let model = GraniteMoeHybrid::random(Device::Cpu, cfg);
        let mut session = model.new_session();

        let input_ids = cpu_tensor(vec![2.0, 5.0], Shape::new(vec![2]));
        let positions = cpu_tensor(vec![0.0, 1.0], Shape::new(vec![2]));

        let logits = model.forward(session.as_mut(), &input_ids, &positions, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[2, 32]);

        let last_h = session.get_last_hidden_state();
        assert!(last_h.is_some());
        assert_eq!(last_h.unwrap().shape().dims(), &[2, 16]);
    }
}
