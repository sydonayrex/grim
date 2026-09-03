//! Dots-3 Note architecture with specialized reasoning/note token attention,
//! RoPE positional embeddings, SwiGLU feed-forward networks, and RMSNorm.
//!
//! # Architecture Details
//! - **Attention**: GQA with RoPE rotation.
//! - **Feed Forward**: SwiGLU activation ($x \cdot \text{SiLU}(x \cdot W_{\text{gate}}) \cdot W_{\text{up}} \cdot W_{\text{down}}$).
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

/// Configuration for Dots-3 Note.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dots3NoteConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for Dots3NoteConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            max_position_embeddings: 65536,
            yarn: None,
        }
    }
}

impl ModelConfig for Dots3NoteConfig {
    fn name(&self) -> &str {
        "dots3_note"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Feed Forward
// ---------------------------------------------------------------------------

pub struct Dots3NoteMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Dots3NoteMlp {
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let act = grim_nn::modules::silu_mul_on_device(&g, &u)?;
        Ok(self.down_proj.forward(&act)?)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct Dots3NoteBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Dots3NoteMlp,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Dots3NoteBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &Dots3NoteConfig, _tp: TensorParallelConfig) -> Result<Self> {
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

        let mlp = Dots3NoteMlp::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first: Q/K/V, RoPE and attention run on the tensor's device; the
    /// host path is only reached through the fused-kernel fallback guard.
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

        // GPU-first; on backends that reject the kernel call fall back to the
        // host-history entry (scalar reference on CPU).
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

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.post_attention_layernorm.forward(&res1)?;
        let mlp_out = self.mlp.forward(&normed_ffn)?;
        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Dots3Note {
    pub cfg: Dots3NoteConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<Dots3NoteBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Dots3Note {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Dots3NoteConfig,
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
            layers.push(Dots3NoteBlock::load(&layer_ws, &cfg, tp)?);
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

    pub fn random(device: Device, cfg: Dots3NoteConfig) -> Self {
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

impl Model for Dots3Note {
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

impl CausalLm for Dots3Note {
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

        // GPU-first embedding gather: only the gathered rows exist as a new
        // device tensor; the vocab×hidden table never crosses to host.
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
    fn test_dots3_note_config_defaults() {
        let cfg = Dots3NoteConfig::default();
        assert_eq!(cfg.name(), "dots3_note");
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_dots3_note_forward_and_session_state() {
        let mut cfg = Dots3NoteConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_hidden_layers = 0;

        let model = Dots3Note::random(Device::Cpu, cfg);
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
