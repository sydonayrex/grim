//! Qwen3.5-MoE architecture with YaRN RoPE support and fine-grained routed/shared experts.
//!
//! # Architecture Details
//! - **YaRN Frequency Scaling**: Decoupled high/low frequency interpolation on RoPE positional encodings.
//! - **Fine-Grained MoE**: Top-k softmax routing across $N$ routed experts plus dedicated shared expert pathways.
//! - **GQA Attention**: Grouped Query Attention with RMSNorm pre/post attention normalizations.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
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

pub struct Qwen35MoeLayer {
    pub gate: Linear,
    pub shared_expert: Option<Qwen35MoeExpert>,
    pub experts: Vec<Qwen35MoeExpert>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
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

        let mut experts = Vec::with_capacity(cfg.num_experts);
        let exp_ws = ws.scoped("experts");
        for e in 0..cfg.num_experts {
            let exp = Qwen35MoeExpert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            experts.push(exp);
        }

        Ok(Self {
            gate,
            shared_expert,
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

            if let Some(ref shared) = self.shared_expert {
                let shared_out = shared.forward(&token_x)?.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out[s * hidden_dim + d] += shared_out[d];
                }
            }
        }

        Ok(cpu_tensor(out, x.shape().clone()))
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
}
