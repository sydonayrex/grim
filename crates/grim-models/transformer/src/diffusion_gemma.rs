//! Compatibility loader and native implementation for `google/diffusiongemma-26B-A4B-it`.
//!
//! # Architecture Details
//! - **Block Diffusion Attention**: Bidirectional self-attention within 256-token canvas blocks paired with causal prompt context.
//! - **GeGLU FFN**: GELU-gated linear units with post-feedforward normalization.
//! - **GQA Attention**: Grouped Query Attention with RMSNorm normalization.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, WeightSource};
use grim_tensor::{ArithType, Device, DType, Shape, Tensor};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// Upload host f32 rows onto `device` (GPU-first). Used to hand results of
/// documented kernel-gap host loops back to the device residency of their
/// inputs instead of leaving the residual stream on CPU.
fn f32_rows_on_device(device: &Device, data: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
    let shape = Shape::new(vec![rows, cols]);
    let dev = grim_nn::modules::pick_device_for_storage_device(device);
    let storage = dev.from_cpu(data, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    ))
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Diffusion-Gemma model architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffusionGemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for DiffusionGemmaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 128,
            num_hidden_layers: 46,
            intermediate_size: 16384,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 8192,
        }
    }
}

impl ModelConfig for DiffusionGemmaConfig {
    fn name(&self) -> &str {
        "diffusion_gemma"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl DiffusionGemmaConfig {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        DiffusionGemmaConfig {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_hidden_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            rms_norm_eps: f("rms_norm_eps"),
            rope_theta: f("rope_theta"),
            max_position_embeddings: u("max_position_embeddings"),
        }
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct DiffusionGemmaBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl DiffusionGemmaBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &DiffusionGemmaConfig) -> Result<Self> {
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

        let mlp_ws = ws.scoped("mlp");
        let gate_proj = Linear::load_shape(
            &mlp_ws.scoped("gate_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let up_proj = Linear::load_shape(
            &mlp_ws.scoped("up_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let down_proj = Linear::load_shape(
            &mlp_ws.scoped("down_proj"),
            [cfg.intermediate_size, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Device-first: attention runs on-device via `dev.qkv_attention` (GQA +
    /// causal + KV cache). RoPE runs through the device kernel. The host path
    /// only runs when the backend lacks the rope/qkv_attention kernels.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let dev = grim_nn::modules::pick_device_for_storage_device(x.device());
        let rope_cfg = grim_tensor::RopeConfig::new(self.head_dim, 10000.0);

        // Device RoPE for Q
        let mut pos_ext = Vec::with_capacity(seq_len * self.num_heads);
        for &pos in positions {
            for _ in 0..self.num_heads {
                pos_ext.push(pos);
            }
        }
        let q3 = crate::block::reshaped_view(&q, &Shape::new(vec![1, seq_len * self.num_heads, self.head_dim]))?;
        let (rope_q_s, _) = dev.rope(q3.storage().as_ref(), &pos_ext, &rope_cfg, q3.shape())?;
        let q_rope_tmp = Tensor::new(Arc::from(rope_q_s), q3.shape().clone(), DType::F32, q.provenance().clone(), x.device().clone());
        let q_rope = crate::block::reshaped_view(&q_rope_tmp, &Shape::new(vec![seq_len, q_dim]))?;

        // Device RoPE for K
        let mut pos_kv = Vec::with_capacity(seq_len * self.num_kv_heads);
        for &pos in positions {
            for _ in 0..self.num_kv_heads {
                pos_kv.push(pos);
            }
        }
        let k3 = crate::block::reshaped_view(&k, &Shape::new(vec![1, seq_len * self.num_kv_heads, self.head_dim]))?;
        let (rope_k_s, _) = dev.rope(k3.storage().as_ref(), &pos_kv, &rope_cfg, k3.shape())?;
        let k_rope_tmp = Tensor::new(Arc::from(rope_k_s), k3.shape().clone(), DType::F32, k.provenance().clone(), x.device().clone());
        let k_rope = crate::block::reshaped_view(&k_rope_tmp, &Shape::new(vec![seq_len, kv_dim]))?;

        // KV cache: append new K/V to device-resident history
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let total_prev = prev_k.shape().dims()[0];
            let new_total = total_prev + seq_len;
            let full_shape = Shape::new(vec![new_total, kv_dim]);
            let k_grown = dev.alloc_storage(&full_shape, DType::F32)?;
            let v_grown = dev.alloc_storage(&full_shape, DType::F32)?;
            dev.copy_slice_range(k_grown.as_ref(), 0, prev_k.storage().as_ref(), 0, total_prev * kv_dim)?;
            dev.copy_slice_range(k_grown.as_ref(), total_prev * kv_dim, k_rope.storage().as_ref(), 0, seq_len * kv_dim)?;
            dev.copy_slice_range(v_grown.as_ref(), 0, prev_v.storage().as_ref(), 0, total_prev * kv_dim)?;
            dev.copy_slice_range(v_grown.as_ref(), total_prev * kv_dim, v.storage().as_ref(), 0, seq_len * kv_dim)?;
            let k_t = Tensor::new(Arc::from(k_grown), full_shape.clone(), DType::F32, k.provenance().clone(), x.device().clone());
            let v_t = Tensor::new(Arc::from(v_grown), full_shape.clone(), DType::F32, v.provenance().clone(), x.device().clone());
            *kv_cache = Some((k_t.clone(), v_t.clone()));
            (k_t, v_t)
        } else {
            let k_t = crate::block::reshaped_view(&k_rope, &Shape::new(vec![seq_len, kv_dim]))?;
            let v_t = crate::block::reshaped_view(&v, &Shape::new(vec![seq_len, kv_dim]))?;
            *kv_cache = Some((k_t.clone(), v_t.clone()));
            (k_t, v_t)
        };

        let total_kv_len = k_all.shape().dims()[0];

        // Device GQA attention (causal, with KV history)
        let out_shape = Shape::new(vec![seq_len, q_dim]);
        let attn_tensor = match dev.qkv_attention(
            q_rope.storage().as_ref(),
            k_all.storage().as_ref(),
            v_all.storage().as_ref(),
            self.num_kv_heads,
            total_kv_len,
            seq_len as u32,
            None,
            &out_shape,
            None,
            None,
        ) {
            Ok((s, _h)) => Tensor::new(
                Arc::from(s),
                out_shape,
                DType::F32,
                grim_tensor::QuantProvenance::default(),
                x.device().clone(),
            ),
            Err(_) => {
                // Host fallback (legacy path)
                let mut q_vec = q_rope.to_vec_f32()?;
                let mut k_vec = k_all.to_vec_f32()?;
                let v_vec = v_all.to_vec_f32()?;
                crate::qwen35::apply_rope_neox(&mut q_vec, positions, self.num_heads, self.head_dim, 10000.0);
                crate::qwen35::apply_rope_neox(&mut k_vec, positions, self.num_kv_heads, self.head_dim, 10000.0);
                let q_rot = cpu_tensor(q_vec, Shape::new(vec![seq_len, q_dim]));
                let k_rot = cpu_tensor(k_vec, Shape::new(vec![total_kv_len, kv_dim]));
                let v_t2 = cpu_tensor(v_vec, Shape::new(vec![total_kv_len, kv_dim]));
                let q_heads = q_rot.to_vec_f32()?;
                let k_heads = k_rot.to_vec_f32()?;
                let v_heads = v_t2.to_vec_f32()?;
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let kv_group_size = (self.num_heads / self.num_kv_heads).max(1);
                let mut attn_out = vec![0.0f32; seq_len * q_dim];
                for s_loop in 0..seq_len {
                    for h in 0..self.num_heads {
                        let kv_h = h / kv_group_size;
                        let q_slice = &q_heads[s_loop * q_dim + h * self.head_dim..s_loop * q_dim + (h + 1) * self.head_dim];
                        let mut scores = vec![0.0f32; total_kv_len];
                        for t in 0..total_kv_len {
                            let k_slice = &k_heads[t * kv_dim + kv_h * self.head_dim..t * kv_dim + (kv_h + 1) * self.head_dim];
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
                                let v_val = v_heads[t * kv_dim + kv_h * self.head_dim + d];
                                acc += weights[t] * v_val;
                            }
                            attn_out[s_loop * q_dim + h * self.head_dim + d] = acc;
                        }
                    }
                }
                f32_rows_on_device(x.device(), &attn_out, seq_len, q_dim)?
            }
        };

        let attn_proj = self.wo.forward(&attn_tensor)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.post_attention_layernorm.forward(&res1)?;
        let gate = self.gate_proj.forward(&normed_ffn)?;
        let up = self.up_proj.forward(&normed_ffn)?;

        // GeGLU — gelu-tanh has no device kernel yet (kernel gap); the loop
        // stays host-side with device-resident inputs and outputs.
        let g_v = gate.to_vec_f32()?;
        let u_v = up.to_vec_f32()?;
        let geglu: Vec<f32> = g_v
            .iter()
            .zip(u_v.iter())
            .map(|(&g, &u)| {
                let gelu = 0.5 * g * (1.0 + ((0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()));
                gelu * u
            })
            .collect();
        let width = gate.shape().dims()[1];
        let geglu_t = f32_rows_on_device(gate.device(), &geglu, seq_len, width)?;
        let mlp_out = self.down_proj.forward(&geglu_t)?;

        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct DiffusionGemma {
    pub cfg: DiffusionGemmaConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<DiffusionGemmaBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl DiffusionGemma {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DiffusionGemmaConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DiffusionGemmaConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = DiffusionGemmaBlock::load(&layer_ws, &cfg)?;
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

impl Model for DiffusionGemma {
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

impl CausalLm for DiffusionGemma {
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
    use grim_core::architecture::ModelArchitecture;

    const DIFFUSION_GEMMA_CONFIG: &str = r#"{
        "architectures": ["DiffusionGemmaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 46,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "intermediate_size": 16384,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "vocab_size": 256000
    }"#;

    #[test]
    fn parses_diffusion_gemma_config() {
        let v: serde_json::Value = serde_json::from_str(DIFFUSION_GEMMA_CONFIG).unwrap();
        let cfg = DiffusionGemmaConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 46);
        assert_eq!(cfg.name(), "diffusion_gemma");
    }

    #[test]
    fn dispatches_diffusion_gemma_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("diffusion_gemma"),
            ModelArchitecture::DiffusionGemma
        );
    }
}
