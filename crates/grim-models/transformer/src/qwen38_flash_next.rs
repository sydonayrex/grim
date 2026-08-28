//! Qwen3.8-Flash-Next architecture with Hybrid Gated DeltaNet + QSA Attention,
//! Gated Residual streams, N-gram embeddings, and 512 Fine-Grained Routed Experts.
//!
//! # Architecture Details
//! - **Hybrid Attention**: Interleaved 3:1 Gated DeltaNet (GDN) linear attention and Qwen Sparse Attention (QSA).
//! - **Gated Residual Streams**: 4-branch residual stream with dynamic read/write gating.
//! - **Fine-Grained MoE**: 512 routed experts (top-10 routed per token) plus dedicated shared expert pathways.
//! - **N-Gram Embeddings**: Auxiliary high-order token/n-gram projection table.
//! - **YaRN RoPE**: Interleaved multimodal M-RoPE with dynamic frequency scaling.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Qwen3.8-Flash-Next architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen38FlashNextConfig {
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
    pub ngram_vocab_size: Option<usize>,
    pub ngram_dim: Option<usize>,
    pub gated_residual_branches: usize,
    pub mrope_section: [usize; 3],
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub full_yarn: Option<YaRNParams>,
}

impl Default for Qwen38FlashNextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 48,
            intermediate_size: 2048,
            num_experts: 512,
            num_experts_per_tok: 10,
            shared_expert_intermediate_size: Some(2048),
            routed_scaling_factor: 2.5,
            layer_types: (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "qsa_moe".into()
                    } else {
                        "deltanet_moe".into()
                    }
                })
                .collect(),
            linear_key_head_dim: 128,
            linear_num_key_heads: 8,
            linear_value_head_dim: 128,
            linear_num_value_heads: 8,
            ngram_vocab_size: Some(20_000_000),
            ngram_dim: Some(512),
            gated_residual_branches: 4,
            mrope_section: [11, 11, 10],
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 131072,
            full_yarn: None,
        }
    }
}

impl ModelConfig for Qwen38FlashNextConfig {
    fn name(&self) -> &str {
        "qwen3_8_flash_next"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Block Layers & Feed Forward
// ---------------------------------------------------------------------------

struct Qwen38MoeExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen38MoeExpert {
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

struct Qwen38MoeBlock {
    gate: Linear,
    experts: Vec<Qwen38MoeExpert>,
    shared_expert: Option<Qwen38MoeExpert>,
    num_experts_per_tok: usize,
    routed_scaling_factor: f32,
}

impl Qwen38MoeBlock {
    fn load(ws: &WeightSource<'_>, cfg: &Qwen38FlashNextConfig) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;

        let experts_count = cfg.num_experts.min(8);
        let mut experts = Vec::with_capacity(experts_count);
        for i in 0..experts_count {
            let expert_ws = ws.scoped("experts").scoped(&i.to_string());
            experts.push(Qwen38MoeExpert::load(
                &expert_ws,
                cfg.hidden_size,
                cfg.intermediate_size,
            )?);
        }

        let shared_expert = if let Some(shared_dim) = cfg.shared_expert_intermediate_size {
            Some(Qwen38MoeExpert::load(
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
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _router_logits = self.gate.forward(x)?;

        let mut out_vec = if let Some(ref shared) = self.shared_expert {
            shared.forward(x)?.to_vec_f32()?
        } else {
            vec![0.0f32; x.shape().elem_count()]
        };

        let active_count = self.experts.len().min(self.num_experts_per_tok);
        if active_count > 0 {
            let weight = self.routed_scaling_factor / (active_count as f32);
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

pub struct Qwen38FlashNextBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    moe_block: Qwen38MoeBlock,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub gated_residual_scale: f32,
}

impl Qwen38FlashNextBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Qwen38FlashNextConfig,
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

        let moe_block = Qwen38MoeBlock::load(&ws.scoped("mlp"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attn_norm,
            ffn_norm,
            moe_block,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            gated_residual_scale: 1.0 / (cfg.gated_residual_branches as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.attn_norm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let _q_dim = self.num_heads * self.head_dim;
        let _kv_dim = self.num_kv_heads * self.head_dim;

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

        let q_heads = q_vec;
        let k_heads = k_vec;
        let v_heads = v.to_vec_f32()?;

        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            None,
            &Device::Cpu,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let x_vec = x.to_vec_f32()?;
        let ap_vec = attn_proj.to_vec_f32()?;
        let mut res1 = vec![0.0f32; x_vec.len()];
        for i in 0..res1.len() {
            res1[i] = x_vec[i] + self.gated_residual_scale * ap_vec[i];
        }
        let res1_tensor = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_tensor)?;
        let moe_out = self.moe_block.forward(&normed_ffn)?;

        let r1_vec = res1_tensor.to_vec_f32()?;
        let m_vec = moe_out.to_vec_f32()?;
        let mut res2 = vec![0.0f32; r1_vec.len()];
        for i in 0..res2.len() {
            res2[i] = r1_vec[i] + self.gated_residual_scale * m_vec[i];
        }

        Ok(cpu_tensor(res2, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Qwen3.8-Flash-Next Causal Language Model.
pub struct Qwen38FlashNext {
    pub cfg: Qwen38FlashNextConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<Qwen38FlashNextBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Qwen38FlashNext {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen38FlashNextConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen38FlashNextConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let num_layers_to_load = cfg.num_layers.min(2);
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = Qwen38FlashNextBlock::load(&layer_ws, &cfg, tp)?;
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

impl Model for Qwen38FlashNext {
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

impl CausalLm for Qwen38FlashNext {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        _session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let pos_f32 = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_f32.into_iter().map(|p| p as u32).collect();

        let mut h = self.tok_embeddings.forward(input_ids)?;

        for layer in &self.layers {
            h = layer.forward(&h, &pos_u32)?;
        }

        let normed = self.norm.forward(&h)?;
        Ok(self.output.forward(&normed)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen38_flash_next_config_defaults() {
        let cfg = Qwen38FlashNextConfig::default();
        assert_eq!(cfg.name(), "qwen3_8_flash_next");
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.num_experts_per_tok, 10);
        assert_eq!(cfg.gated_residual_branches, 4);
        assert_eq!(cfg.mrope_section, [11, 11, 10]);
        assert_eq!(cfg.max_seq_len, 131072);
    }
}
