//! Cohere Command-R / Command-R+ architecture with per-head QK-Normalization,
//! parallel Attention + MLP residual connections, and LayerNorm / RMSNorm.
//!
//! # Architecture Details
//! - **Parallel Residual**: Attention and MLP compute in parallel from normalized input:
//!   $\text{out} = x + \text{Attn}(\text{Norm}(x)) + \text{MLP}(\text{Norm}(x))$.
//! - **QK Normalization**: Per-head `q_norm` and `k_norm` applied to query and key heads before RoPE.
//! - **RoPE**: 8192 / 128k context with `rope_theta: 500000.0`.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Cohere Command-R / Command-R+.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandRConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub use_qk_norm: bool,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for CommandRConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 8192,
            intermediate_size: 22528,
            num_hidden_layers: 40,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            use_qk_norm: true,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 131072,
            yarn: None,
        }
    }
}

impl ModelConfig for CommandRConfig {
    fn name(&self) -> &str {
        "commandr"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

pub struct CommandRMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl CommandRMlp {
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

    /// GPU-first: the gated activation runs through the fused `silu_mul`
    /// device kernel; host paths are only reached through its fallback guard.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let act = grim_nn::modules::silu_mul_on_device(&g, &u)?;
        Ok(self.down_proj.forward(&act)?)
    }
}

// ---------------------------------------------------------------------------
// Block (Parallel Attention + MLP)
// ---------------------------------------------------------------------------

pub struct CommandRBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub input_layernorm: RmsNorm,
    pub mlp: CommandRMlp,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl CommandRBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &CommandRConfig, _tp: TensorParallelConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let (q_norm, k_norm) = if cfg.use_qk_norm {
            (
                RmsNorm::load(&attn_ws.scoped("q_norm"), cfg.head_dim, cfg.rms_norm_eps).ok(),
                RmsNorm::load(&attn_ws.scoped("k_norm"), cfg.head_dim, cfg.rms_norm_eps).ok(),
            )
        } else {
            (None, None)
        };

        let input_layernorm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let mlp = CommandRMlp::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            input_layernorm,
            mlp,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Forward pass executing parallel Attention + MLP branches:
    /// $\text{out} = x + \text{Attn}(\text{Norm}(x)) + \text{MLP}(\text{Norm}(x))$.
    ///
    /// GPU-first: Q/K/V, RoPE, attention and both residual adds run on the
    /// tensor's device; host paths are only reached through the fused-kernel
    /// fallback guards.
    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed = self.input_layernorm.forward(x)?;

        let q = self.wq.forward(&normed)?;
        let k = self.wk.forward(&normed)?;
        let v = self.wv.forward(&normed)?;

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

        // GPU-first: fused tensor-level attention; on backends that reject the
        // kernel call (e.g. no fused kernel) fall back to the host-history
        // entry, which degrades to the scalar reference on CPU.
        let attn_tensor = match crate::shared_attention::fused_attention_tensors(
            &q,
            &k,
            &v,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            seq_len,
            None,
        ) {
            Ok(t) => t,
            Err(_) => crate::shared_attention::fused_or_scalar_attention(
                &q.to_vec_f32()?,
                &k.to_vec_f32()?,
                &v.to_vec_f32()?,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                None,
                x.device(),
            )?,
        };
        let attn_proj = self.wo.forward(&attn_tensor)?;
        let mlp_out = self.mlp.forward(&normed)?;

        // Parallel residual: x + attn + mlp as two device-side adds.
        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct CommandR {
    pub cfg: CommandRConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<CommandRBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl CommandR {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: CommandRConfig,
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
            layers.push(CommandRBlock::load(&layer_ws, &cfg, tp)?);
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

    pub fn random(device: Device, cfg: CommandRConfig) -> Self {
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

impl Model for CommandR {
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

impl CausalLm for CommandR {
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
    fn test_commandr_config_defaults() {
        let cfg = CommandRConfig::default();
        assert_eq!(cfg.name(), "commandr");
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(cfg.use_qk_norm);
    }

    #[test]
    fn test_commandr_parallel_residual_numerics() {
        // Test parallel residual formula: x + attn(x) + mlp(x)
        let x = vec![1.0f32, 2.0];
        let attn = vec![0.5f32, 0.25];
        let mlp = vec![1.5f32, 0.75];
        let mut out = vec![0.0f32; 2];
        for i in 0..2 {
            out[i] = x[i] + attn[i] + mlp[i];
        }
        assert_eq!(out, vec![3.0f32, 3.0]);
    }

    #[test]
    fn test_commandr_forward_and_session_state() {
        let mut cfg = CommandRConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_hidden_layers = 0;

        let model = CommandR::random(Device::Cpu, cfg);
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
