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
use grim_tensor::{ArithType, BackendDevice, Device, DType, Shape, Tensor};

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

    /// Forward pass. NOTE: attention here is intentionally BIDIRECTIONAL
    /// (block-diffusion canvases attend over the whole context, no causal
    /// mask, epsilon-weighted softmax), which the causal-only device
    /// `qkv_attention` kernel cannot express — so the attention core, RoPE
    /// and the KV history stay host-side (documented kernel gap). Embedding,
    /// residual adds and the MLP run device-first around it.
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

        // Kernel gap: bidirectional attention runs host-side, so Q/K/V cross
        // to the host here and RoPE keeps its host NeoX loop.
        let mut q_vec = q.to_vec_f32()?;
        let mut k_vec = k.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;

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
        let k_rot = cpu_tensor(k_vec, Shape::new(vec![seq_len, kv_dim]));
        let v_t = cpu_tensor(v_vec, Shape::new(vec![seq_len, kv_dim]));

        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            new_k.extend(k_rot.to_vec_f32()?);
            new_v.extend(v_t.to_vec_f32()?);
            let total_seq = new_k.len() / kv_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, kv_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, kv_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k_rot.clone(), v_t.clone()));
            (k_rot, v_t)
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
                    let k_slice = &k_heads[t * kv_dim + kv_h * self.head_dim
                        ..t * kv_dim + (kv_h + 1) * self.head_dim];
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
                    attn_out[s * q_dim + h * self.head_dim + d] = acc;
                }
            }
        }

        // Return the attention output to the device residency of x.
        let attn_tensor = f32_rows_on_device(x.device(), &attn_out, seq_len, q_dim)?;
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
