//! Compatibility loader and native implementation for `internlm/Intern-S2-Mobius`.
//!
//! # Architecture Details
//! - **Fused Wqkv Projection**: Single weight matrix `attention.wqkv` projecting query, key, and value vectors.
//! - **GQA Attention**: Grouped Query Attention with RoPE rotary embeddings.
//! - **SwiGLU FFN**: $w_1$ (gate), $w_3$ (up), and $w_2$ (down) feed-forward projections with RMSNorm normalization.

use std::sync::Arc;

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

/// Configuration for Intern-S2-Mobius architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InternS2MobiusConfig {
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

impl Default for InternS2MobiusConfig {
    fn default() -> Self {
        Self {
            vocab_size: 92544,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            num_hidden_layers: 32,
            intermediate_size: 14336,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            max_position_embeddings: 32768,
        }
    }
}

impl ModelConfig for InternS2MobiusConfig {
    fn name(&self) -> &str {
        "interns2_mobius"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl InternS2MobiusConfig {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        InternS2MobiusConfig {
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

pub struct InternS2MobiusBlock {
    pub wqkv: Linear,
    pub wo: Linear,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl InternS2MobiusBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &InternS2MobiusConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        let attn_ws = ws.scoped("attention");
        let wqkv = Linear::load_shape(&attn_ws.scoped("wqkv"), [cfg.hidden_size, qkv_dim])
            .or_else(|_| {
                Linear::load_shape(&attn_ws.scoped("wqkv_proj"), [cfg.hidden_size, qkv_dim])
            })?;
        let wo = Linear::load_shape(&attn_ws.scoped("wo"), [q_dim, cfg.hidden_size])?;

        let attention_norm = RmsNorm::load(
            &ws.scoped("attention_norm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let ffn_norm = RmsNorm::load(&ws.scoped("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        let ffn_ws = ws.scoped("feed_forward");
        let w1 = Linear::load_shape(
            &ffn_ws.scoped("w1"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w3 = Linear::load_shape(
            &ffn_ws.scoped("w3"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w2 = Linear::load_shape(
            &ffn_ws.scoped("w2"),
            [cfg.intermediate_size, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wqkv,
            wo,
            attention_norm,
            ffn_norm,
            w1,
            w3,
            w2,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first forward. NOTE: the fused `wqkv` projection interleaves
    /// Q/K/V column-wise per row and there is no device-side column-split
    /// kernel yet, so the split stays host-side (documented kernel gap —
    /// one D2H of the qkv activation). The split Q/K/V are uploaded once;
    /// RoPE, the KV-cache concat, causal attention and the SwiGLU MLP all
    /// run device-first.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.attention_norm.forward(x)?;

        let qkv = self.wqkv.forward(&normed_attn)?;
        let qkv_v = qkv.to_vec_f32()?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        let mut q_v = vec![0.0f32; seq_len * q_dim];
        let mut k_v = vec![0.0f32; seq_len * kv_dim];
        let mut v_v = vec![0.0f32; seq_len * kv_dim];

        for s in 0..seq_len {
            let row_off = s * qkv_dim;
            q_v[s * q_dim..(s + 1) * q_dim].copy_from_slice(&qkv_v[row_off..row_off + q_dim]);
            k_v[s * kv_dim..(s + 1) * kv_dim]
                .copy_from_slice(&qkv_v[row_off + q_dim..row_off + q_dim + kv_dim]);
            v_v[s * kv_dim..(s + 1) * kv_dim]
                .copy_from_slice(&qkv_v[row_off + q_dim + kv_dim..row_off + qkv_dim]);
        }

        // Upload the split projections once; everything downstream stays on
        // the device.
        let q = f32_rows_on_device(x.device(), &q_v, seq_len, q_dim)?;
        let k = f32_rows_on_device(x.device(), &k_v, seq_len, kv_dim)?;
        let v = f32_rows_on_device(x.device(), &v_v, seq_len, kv_dim)?;

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
        let w1_out = self.w1.forward(&normed_ffn)?;
        let w3_out = self.w3.forward(&normed_ffn)?;
        let act = grim_nn::modules::silu_mul_on_device(&w1_out, &w3_out)?;
        let mlp_out = self.w2.forward(&act)?;

        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct InternS2Mobius {
    pub cfg: InternS2MobiusConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<InternS2MobiusBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl InternS2Mobius {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: InternS2MobiusConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: InternS2MobiusConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("tok_embeddings"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = InternS2MobiusBlock::load(&layer_ws, &cfg)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("output"), [cfg.hidden_size, cfg.vocab_size])
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

impl Model for InternS2Mobius {
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

impl CausalLm for InternS2Mobius {
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

    const INTERN_S2_CONFIG: &str = r#"{
        "architectures": ["InternS2MobiusForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "vocab_size": 92544
    }"#;

    #[test]
    fn parses_intern_s2_mobius_config() {
        let v: serde_json::Value = serde_json::from_str(INTERN_S2_CONFIG).unwrap();
        let cfg = InternS2MobiusConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.name(), "interns2_mobius");
    }

    #[test]
    fn dispatches_intern_s2_mobius_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("interns2_mobius"),
            ModelArchitecture::InternS2Mobius
        );
    }
}
