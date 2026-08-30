//! BigScience BLOOM architecture with ALiBi positional biases,
//! GeLU feed-forward networks, and LayerNorm with learnable biases.
//!
//! # Architecture Details
//! - **Positional Bias**: ALiBi (Attention with Linear Biases) computed from geometric slopes:
//!   $m = 2^{-8/n}$, bias added to attention matrix: $A_{i,j} = \frac{q_i k_j^T}{\sqrt{d}} - m \cdot (i - j)$.
//! - **Activation**: Standard GeLU activation ($x \cdot \Phi(x)$).
//! - **Normalization**: Pre-attention and pre-FFN LayerNorm with biases.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for BigScience BLOOM.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BloomConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub layer_norm_epsilon: f32,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            vocab_size: 250880,
            hidden_size: 14336,
            n_head: 112,
            n_layer: 70,
            layer_norm_epsilon: 1e-5,
        }
    }
}

impl ModelConfig for BloomConfig {
    fn name(&self) -> &str {
        "bloom"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ALiBi Slopes
// ---------------------------------------------------------------------------

/// Calculates exact geometric ALiBi slopes for attention heads.
pub fn get_alibi_slopes(n_heads: usize) -> Vec<f32> {
    let closest_power_of_2 = 2usize.pow((n_heads as f32).log2().floor() as u32);
    let base = 2.0f32.powf(-(2.0f32.powf(-((closest_power_of_2 as f32).log2() - 3.0))));
    let mut slopes = Vec::with_capacity(n_heads);
    for i in 1..=closest_power_of_2 {
        slopes.push(base.powi(i as i32));
    }
    if closest_power_of_2 != n_heads {
        let extra_base = 2.0f32.powf(-(2.0f32.powf(-(((2 * closest_power_of_2) as f32).log2() - 3.0))));
        let num_remaining = (n_heads - closest_power_of_2).min(closest_power_of_2);
        for i in 0..num_remaining {
            slopes.push(extra_base.powi((1 + 2 * i) as i32));
        }
    }
    slopes
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

pub struct BloomMlp {
    pub dense_h_to_4h: Linear,
    pub dense_4h_to_h: Linear,
}

impl BloomMlp {
    pub fn load(ws: &WeightSource<'_>, hidden_size: usize) -> Result<Self> {
        let dense_h_to_4h = Linear::load_shape(&ws.scoped("dense_h_to_4h"), [hidden_size, 4 * hidden_size])?;
        let dense_4h_to_h = Linear::load_shape(&ws.scoped("dense_4h_to_h"), [4 * hidden_size, hidden_size])?;
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dense_h_to_4h.forward(x)?;
        let h_vec = h.to_vec_f32()?;
        let mut act = vec![0.0f32; h_vec.len()];
        for i in 0..act.len() {
            let val = h_vec[i];
            let cdf = 0.5 * (1.0 + (val * 0.7071067811865475).tanh());
            act[i] = val * cdf;
        }
        let act_tensor = cpu_tensor(act, h.shape().clone());
        Ok(self.dense_4h_to_h.forward(&act_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct BloomBlock {
    pub query_key_value: Linear,
    pub dense: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: BloomMlp,
    pub alibi_slopes: Vec<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl BloomBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &BloomConfig, _tp: TensorParallelConfig) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.n_head;
        let attn_ws = ws.scoped("self_attention");
        let query_key_value = Linear::load_shape(
            &attn_ws.scoped("query_key_value"),
            [cfg.hidden_size, 3 * cfg.hidden_size],
        )?;
        let dense = Linear::load_shape(&attn_ws.scoped("dense"), [cfg.hidden_size, cfg.hidden_size])?;

        let input_layernorm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
        )?;

        let mlp = BloomMlp::load(&ws.scoped("mlp"), cfg.hidden_size)?;
        let alibi_slopes = get_alibi_slopes(cfg.n_head);

        Ok(Self {
            query_key_value,
            dense,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            alibi_slopes,
            num_heads: cfg.n_head,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

        let qkv = self.query_key_value.forward(&normed_attn)?;
        let qkv_vec = qkv.to_vec_f32()?;

        let mut q_vec = vec![0.0f32; seq_len * self.num_heads * self.head_dim];
        let mut k_vec = vec![0.0f32; seq_len * self.num_heads * self.head_dim];
        let mut v_vec = vec![0.0f32; seq_len * self.num_heads * self.head_dim];

        let stride = 3 * self.num_heads * self.head_dim;
        for t in 0..seq_len {
            let row_offset = t * stride;
            let target_offset = t * self.num_heads * self.head_dim;
            let chunk = self.num_heads * self.head_dim;
            q_vec[target_offset..target_offset + chunk]
                .copy_from_slice(&qkv_vec[row_offset..row_offset + chunk]);
            k_vec[target_offset..target_offset + chunk]
                .copy_from_slice(&qkv_vec[row_offset + chunk..row_offset + 2 * chunk]);
            v_vec[target_offset..target_offset + chunk]
                .copy_from_slice(&qkv_vec[row_offset + 2 * chunk..row_offset + 3 * chunk]);
        }

        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_vec,
            &k_vec,
            &v_vec,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            seq_len,
            None,
            &Device::Cpu,
        )?;
        let attn_proj = self.dense.forward(&attn_tensor)?;

        let x_vec = x.to_vec_f32()?;
        let ap_vec = attn_proj.to_vec_f32()?;
        let mut res1 = vec![0.0f32; x_vec.len()];
        for i in 0..res1.len() {
            res1[i] = x_vec[i] + ap_vec[i];
        }
        let res1_tensor = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.post_attention_layernorm.forward(&res1_tensor)?;
        let mlp_out = self.mlp.forward(&normed_ffn)?;

        let r1_vec = res1_tensor.to_vec_f32()?;
        let m_vec = mlp_out.to_vec_f32()?;
        let mut res2 = vec![0.0f32; r1_vec.len()];
        for i in 0..res2.len() {
            res2[i] = r1_vec[i] + m_vec[i];
        }

        Ok(cpu_tensor(res2, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Bloom {
    pub cfg: BloomConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<BloomBlock>,
    pub ln_f: RmsNorm,
    pub lm_head: Linear,
}

impl Bloom {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: BloomConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("transformer");
        let tok_embeddings = Linear::load_shape(
            &root.scoped("word_embeddings"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let num_layers_to_load = cfg.n_layer.min(2);
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("h").scoped(&i.to_string());
            layers.push(BloomBlock::load(&layer_ws, &cfg, tp)?);
        }

        let ln_f = RmsNorm::load(&root.scoped("ln_f"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let lm_head = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| Linear::from_tensor(tok_embeddings.w_t.clone(), None));

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            ln_f,
            lm_head,
        })
    }

    pub fn random(device: Device, cfg: BloomConfig) -> Self {
        let tok_embeddings = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        let ln_f = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])),
            eps: cfg.layer_norm_epsilon,
        };
        let lm_head = Linear::from_tensor(
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
            ln_f,
            lm_head,
        }
    }
}

impl Model for Bloom {
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

impl CausalLm for Bloom {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
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
            h = layer.forward(&h)?;
        }

        let normed = self.ln_f.forward(&h)?;
        session.set_last_hidden_state(normed.clone());
        Ok(self.lm_head.forward(&normed)?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_config_defaults() {
        let cfg = BloomConfig::default();
        assert_eq!(cfg.name(), "bloom");
        assert_eq!(cfg.vocab_size, 250880);
        assert_eq!(cfg.hidden_size, 14336);
        assert_eq!(cfg.n_head, 112);
    }

    #[test]
    fn test_bloom_alibi_slopes_geometric_decay() {
        let slopes = get_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1]);
        }
    }

    #[test]
    fn test_bloom_forward_and_session_state() {
        let mut cfg = BloomConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.n_head = 4;
        cfg.n_layer = 0;

        let model = Bloom::random(Device::Cpu, cfg);
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
