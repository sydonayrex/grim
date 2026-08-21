//! Compatibility loader and native implementation for `internlm/Intern-S2-Mobius`.
//!
//! # Architecture Details
//! - **Fused Wqkv Projection**: Single weight matrix `attention.wqkv` projecting query, key, and value vectors.
//! - **GQA Attention**: Grouped Query Attention with RoPE rotary embeddings.
//! - **SwiGLU FFN**: $w_1$ (gate), $w_3$ (up), and $w_2$ (down) feed-forward projections with RMSNorm normalization.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

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

        crate::qwen35::apply_rope_neox(
            &mut q_v,
            positions,
            self.num_heads,
            self.head_dim,
            1000000.0,
        );
        crate::qwen35::apply_rope_neox(
            &mut k_v,
            positions,
            self.num_kv_heads,
            self.head_dim,
            1000000.0,
        );

        let q_rot = cpu_tensor(q_v, Shape::new(vec![seq_len, q_dim]));
        let k_rot = cpu_tensor(k_v, Shape::new(vec![seq_len, kv_dim]));

        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            new_k.extend(k_rot.to_vec_f32()?);
            new_v.extend(v_v);
            let total_seq = new_k.len() / kv_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, kv_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, kv_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            let full_k = k_rot.clone();
            let full_v = cpu_tensor(v_v, Shape::new(vec![seq_len, kv_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        };

        let q_heads = q_rot.to_vec_f32()?;
        let k_heads = k_all.to_vec_f32()?;
        let v_heads = v_all.to_vec_f32()?;

        // Shared helper applies the causal mask at cache_offset + s (fixes
        // future-token leakage during multi-token prefill).
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

        let xv = x.to_vec_f32()?;
        let av = attn_proj.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;
        let w1_out = self.w1.forward(&normed_ffn)?;
        let w3_out = self.w3.forward(&normed_ffn)?;

        let g_v = w1_out.to_vec_f32()?;
        let u_v = w3_out.to_vec_f32()?;
        let swiglu: Vec<f32> = g_v
            .iter()
            .zip(u_v.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let swiglu_t = cpu_tensor(swiglu, w1_out.shape().clone());
        let mlp_out = self.w2.forward(&swiglu_t)?;

        let r1v = res1_t.to_vec_f32()?;
        let mv = mlp_out.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
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
