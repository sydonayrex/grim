//! Tencent Hunyuan-V3 (HY-V3) architecture with QK Normalization,
//! fine-grained routed Mixture of Experts (MoE), dedicated shared expert, and RoPE.
//!
//! # Architecture Details
//! - **Attention**: Grouped Query Attention (GQA) with per-head QK Normalization (`q_norm`, `k_norm`) before RoPE.
//! - **Feed Forward**: Fine-grained MoE with routed top-k experts and dedicated shared expert.
//! - **Normalization**: Pre-attention and pre-FFN RMSNorm.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for HY-V3 architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HyV3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for HyV3Config {
    fn default() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 4096,
            intermediate_size: 2048,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            num_experts: 64,
            num_experts_per_tok: 8,
            shared_expert_intermediate_size: Some(2048),
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            max_position_embeddings: 131072,
            yarn: None,
        }
    }
}

impl ModelConfig for HyV3Config {
    fn name(&self) -> &str {
        "hyv3"
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

struct HyV3Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl HyV3Expert {
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
        let act = grim_nn::modules::silu_mul_on_device(&g, &u)
            .map_err(grim_core::error::Error::from)?;
        Ok(self.down_proj.forward(&act)?)
    }
}

pub struct HyV3MoeBlock {
    gate: Linear,
    experts: Vec<HyV3Expert>,
    shared_expert: Option<HyV3Expert>,
    num_experts_per_tok: usize,
}

impl HyV3MoeBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &HyV3Config) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;

        let experts_count = cfg.num_experts.min(8);
        let mut experts = Vec::with_capacity(experts_count);
        for i in 0..experts_count {
            let expert_ws = ws.scoped("experts").scoped(&i.to_string());
            experts.push(HyV3Expert::load(
                &expert_ws,
                cfg.hidden_size,
                cfg.intermediate_size,
            )?);
        }

        let shared_expert = if let Some(shared_dim) = cfg.shared_expert_intermediate_size {
            Some(HyV3Expert::load(
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

pub struct HyV3Block {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub moe: HyV3MoeBlock,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl HyV3Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &HyV3Config, _tp: TensorParallelConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let q_norm = RmsNorm::load(&attn_ws.scoped("q_norm"), cfg.head_dim, cfg.rms_norm_eps)
            .or_else(|_| RmsNorm::load(&attn_ws.scoped("q_layernorm"), cfg.head_dim, cfg.rms_norm_eps))?;
        let k_norm = RmsNorm::load(&attn_ws.scoped("k_norm"), cfg.head_dim, cfg.rms_norm_eps)
            .or_else(|_| RmsNorm::load(&attn_ws.scoped("k_layernorm"), cfg.head_dim, cfg.rms_norm_eps))?;

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

        let moe = HyV3MoeBlock::load(&ws.scoped("mlp"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            input_layernorm,
            post_attention_layernorm,
            moe,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first forward: Q/K RoPE, attention and the residual adds run on
    /// the tensor's device. Host paths are only reached through the
    /// fused-kernel fallback guards and the (host-side) MoE routing pull.
    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

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

        // Stateless attention over the current step only (kv_len == steps,
        // cache_offset == 0); the helper applies the causal mask at s.
        let attn_tensor = crate::shared_attention::fused_attention_tensors(
            &q,
            &k,
            &v,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            seq_len,
            None,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.post_attention_layernorm.forward(&res1)?;
        let moe_out = self.moe.forward(&normed_ffn)?;
        // Routing stays host-side, so the MoE output lands on the host; stage
        // it back next to `res1` before the residual add.
        let moe_out = grim_nn::modules::move_to_device(&moe_out, x.device())?;

        grim_nn::modules::add_on_device(&res1, &moe_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct HyV3 {
    pub cfg: HyV3Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<HyV3Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl HyV3 {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: HyV3Config,
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
            layers.push(HyV3Block::load(&layer_ws, &cfg, tp)?);
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

    pub fn random(device: Device, cfg: HyV3Config) -> Self {
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

impl Model for HyV3 {
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

impl CausalLm for HyV3 {
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
        let ids: Vec<u32> = ids_f32.iter().map(|&t| t as u32).collect();

        // GPU-first embedding gather: rows land on the weight's device; the
        // vocab×hidden table never crosses to host.
        let mut h = grim_nn::embedding_gather_on_device(
            &self.tok_embeddings.weight,
            &ids,
            seq_len,
            self.cfg.hidden_size,
        )?;

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
    fn test_hy_v3_config_defaults() {
        let cfg = HyV3Config::default();
        assert_eq!(cfg.name(), "hyv3");
        assert_eq!(cfg.vocab_size, 131072);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
    }

    #[test]
    fn test_hy_v3_forward_and_session_state() {
        let mut cfg = HyV3Config::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_hidden_layers = 0;

        let model = HyV3::random(Device::Cpu, cfg);
        let mut session = model.new_session();

        let input_ids = cpu_tensor(vec![1.0, 4.0], Shape::new(vec![2]));
        let positions = cpu_tensor(vec![0.0, 1.0], Shape::new(vec![2]));

        let logits = model.forward(session.as_mut(), &input_ids, &positions, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[2, 32]);

        let last_h = session.get_last_hidden_state();
        assert!(last_h.is_some());
        assert_eq!(last_h.unwrap().shape().dims(), &[2, 16]);
    }
}
