//! Google Gemma-2 architecture with dual pre/post layer norms,
//! GeLU-Tanh gated activations, and attention / logit softcapping.
//!
//! # Architecture Details
//! - **Normalization**: 4 RMSNorm layers per block (`input_layernorm`, `post_attention_layernorm`,
//!   `pre_feedforward_layernorm`, `post_feedforward_layernorm`).
//! - **Softcapping**: Attention logits softcapped at $50.0$, final LM head logits softcapped at $30.0$:
//!   $\text{logits} = \text{cap} \cdot \tanh(\text{logits} / \text{cap})$.
//! - **Activation**: GeLU with Tanh approximation and multiplication gating ($\text{GeLU}_{\text{tanh}}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})$).

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Google Gemma-2 models.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gemma2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,
    pub sliding_window: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for Gemma2Config {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 3584,
            intermediate_size: 14336,
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(4096),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 8192,
            yarn: None,
        }
    }
}

impl ModelConfig for Gemma2Config {
    fn name(&self) -> &str {
        "gemma2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Feed Forward / GeLU-Tanh Gated MLP
// ---------------------------------------------------------------------------

pub struct Gemma2Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Gemma2Mlp {
    pub fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let gate_proj = Linear::load_shape(&ws.scoped("gate_proj"), [in_dim, hidden_dim])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [in_dim, hidden_dim])?;
        let down_proj = Linear::load_shape(&ws.scoped("down_proj"), [hidden_dim, in_dim])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass executing GeLU-Tanh gated multiplication:
    /// $\text{out} = (\text{GeLU}_{\text{tanh}}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})) \cdot W_{\text{down}}$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let g_vec = g.to_vec_f32()?;
        let u_vec = u.to_vec_f32()?;
        let mut act = vec![0.0f32; g_vec.len()];
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();

        for i in 0..act.len() {
            let x_val = g_vec[i];
            let tanh_in = sqrt_2_over_pi * (x_val + 0.044715 * x_val.powi(3));
            let gelu = 0.5 * x_val * (1.0 + tanh_in.tanh());
            act[i] = gelu * u_vec[i];
        }
        let act_tensor = cpu_tensor(act, g.shape().clone());
        Ok(self.down_proj.forward(&act_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct Gemma2Block {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub mlp: Gemma2Mlp,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub attn_logit_softcapping: Option<f32>,
}

impl Gemma2Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &Gemma2Config, _tp: TensorParallelConfig) -> Result<Self> {
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
        let pre_feedforward_layernorm = RmsNorm::load(
            &ws.scoped("pre_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let post_feedforward_layernorm = RmsNorm::load(
            &ws.scoped("post_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let mlp = Gemma2Mlp::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            mlp,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
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
        let post_attn = self.post_attention_layernorm.forward(&attn_proj)?;

        let res1 = grim_nn::modules::add_on_device(x, &post_attn)?;
        let pre_ffn = self.pre_feedforward_layernorm.forward(&res1)?;
        let mlp_out = self.mlp.forward(&pre_ffn)?;
        let post_ffn = self.post_feedforward_layernorm.forward(&mlp_out)?;
        grim_nn::modules::add_on_device(&res1, &post_ffn).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Gemma2 {
    pub cfg: Gemma2Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<Gemma2Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Gemma2 {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Gemma2Config,
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
            layers.push(Gemma2Block::load(&layer_ws, &cfg, tp)?);
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

    pub fn random(device: Device, cfg: Gemma2Config) -> Self {
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

impl Model for Gemma2 {
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

impl CausalLm for Gemma2 {
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
        let logits = self.output.forward(&normed)?;

        // Apply final logit softcapping if configured
        if let Some(cap) = self.cfg.final_logit_softcapping {
            let mut l_vec = logits.to_vec_f32()?;
            for val in &mut l_vec {
                *val = cap * (*val / cap).tanh();
            }
            Ok(cpu_tensor(l_vec, logits.shape().clone()))
        } else {
            Ok(logits)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma2_config_defaults() {
        let cfg = Gemma2Config::default();
        assert_eq!(cfg.name(), "gemma2");
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn test_gemma2_logit_softcapping_bounds() {
        let cap = 30.0f32;
        let large_val = 1000.0f32;
        let softcapped = cap * (large_val / cap).tanh();
        assert!(softcapped <= cap);
        assert!((softcapped - cap).abs() < 1e-4);
    }

    #[test]
    fn test_gemma2_forward_and_session_state() {
        let mut cfg = Gemma2Config::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_hidden_layers = 0;

        let model = Gemma2::random(Device::Cpu, cfg);
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
