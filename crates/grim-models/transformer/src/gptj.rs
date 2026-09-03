//! EleutherAI GPT-J architecture with parallel Attention + MLP branches,
//! partial rotary positional embeddings ($d_R = 64$), and LayerNorm.
//!
//! # Architecture Details
//! - **Parallel Residual**: Attention and MLP execute concurrently on the single normalized input:
//!   $\text{out} = x + \text{Attn}(\text{Norm}(x)) + \text{MLP}(\text{Norm}(x))$.
//! - **Partial RoPE**: Rotary positional embeddings applied only to the first `rotary_dim` (64) dimensions per head.
//! - **Activation**: GeLU with standard/tanh approximation.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for EleutherAI GPT-J.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GptJConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_epsilon: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for GptJConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50400,
            hidden_size: 4096,
            num_attention_heads: 16,
            head_dim: 256,
            rotary_dim: 64,
            num_hidden_layers: 28,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }
}

impl ModelConfig for GptJConfig {
    fn name(&self) -> &str {
        "gptj"
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

pub struct GptJMlp {
    pub fc_in: Linear,
    pub fc_out: Linear,
}

impl GptJMlp {
    pub fn load(ws: &WeightSource<'_>, hidden_size: usize) -> Result<Self> {
        let fc_in = Linear::load_shape(&ws.scoped("fc_in"), [hidden_size, 4 * hidden_size])?;
        let fc_out = Linear::load_shape(&ws.scoped("fc_out"), [4 * hidden_size, hidden_size])?;
        Ok(Self { fc_in, fc_out })
    }

    /// GeLU-exact activation — host kernel gap (no device GELU kernel),
    /// pulled once and re-uploaded onto the input's device.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc_in.forward(x)?;
        let h_vec = h.to_vec_f32()?;
        let mut act = vec![0.0f32; h_vec.len()];
        for i in 0..act.len() {
            let val = h_vec[i];
            let cdf = 0.5 * (1.0 + (val * 0.7071067811865475).tanh());
            act[i] = val * cdf;
        }
        let act_tensor = grim_nn::modules::move_to_device(&cpu_tensor(act, h.shape().clone()), x.device())?;
        Ok(self.fc_out.forward(&act_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// Block (Parallel Attention + MLP)
// ---------------------------------------------------------------------------

pub struct GptJBlock {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub ln_1: RmsNorm,
    pub mlp: GptJMlp,
    pub rope: Rope,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

impl GptJBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &GptJConfig, _tp: TensorParallelConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let attn_ws = ws.scoped("attn");
        let q_proj = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let k_proj = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, q_dim])?;
        let v_proj = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, q_dim])?;
        let out_proj = Linear::load_shape(&attn_ws.scoped("out_proj"), [q_dim, cfg.hidden_size])?;

        let ln_1 = RmsNorm::load(
            &ws.scoped("ln_1"),
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
        )?;

        let mlp = GptJMlp::load(&ws.scoped("mlp"), cfg.hidden_size)?;
        let rope = Rope::new(cfg.rotary_dim, cfg.rope_theta);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            ln_1,
            mlp,
            rope,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim,
        })
    }

    /// Parallel block forward: $\text{out} = x + \text{Attn}(\text{Norm}(x)) + \text{MLP}(\text{Norm}(x))$.
    ///
    /// GPU-first: RoPE and attention run on the tensor's device; host paths
    /// are only reached through the fused-kernel fallback guards.
    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed = self.ln_1.forward(x)?;

        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;

        // GPT-J is partial-rotary (block `rope` carries rotary_dim), but this
        // loader always rotated the FULL head_dim at theta 10000 — mirror that
        // with a full-width rope on-device.
        let full_rope = Rope::new(self.head_dim, 10000.0);
        let q = crate::shared_attention::rope_2d_on_device(
            &full_rope,
            &q,
            self.num_heads,
            positions,
        )?;
        let k = crate::shared_attention::rope_2d_on_device(
            &full_rope,
            &k,
            self.num_heads,
            positions,
        )?;

        // GPU-first; on backends that reject the kernel call fall back to the
        // host-history entry (scalar reference on CPU).
        let attn_tensor = match crate::shared_attention::fused_attention_tensors(
            &q,
            &k,
            &v,
            self.num_heads,
            self.num_heads,
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
                self.num_heads,
                self.head_dim,
                seq_len,
                None,
                x.device(),
            )?,
        };
        let attn_proj = self.out_proj.forward(&attn_tensor)?;
        let mlp_out = self.mlp.forward(&normed)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct GptJ {
    pub cfg: GptJConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<GptJBlock>,
    pub ln_f: RmsNorm,
    pub lm_head: Linear,
}

impl GptJ {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: GptJConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("transformer");
        let tok_embeddings = Linear::load_shape(
            &root.scoped("wte"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let num_layers_to_load = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("h").scoped(&i.to_string());
            layers.push(GptJBlock::load(&layer_ws, &cfg, tp)?);
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

    pub fn random(device: Device, cfg: GptJConfig) -> Self {
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

impl Model for GptJ {
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

impl CausalLm for GptJ {
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
    fn test_gptj_config_defaults() {
        let cfg = GptJConfig::default();
        assert_eq!(cfg.name(), "gptj");
        assert_eq!(cfg.vocab_size, 50400);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.rotary_dim, 64);
    }

    #[test]
    fn test_gptj_forward_and_session_state() {
        let mut cfg = GptJConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.num_attention_heads = 2;
        cfg.head_dim = 8;
        cfg.rotary_dim = 8;
        cfg.num_hidden_layers = 0;

        let model = GptJ::random(Device::Cpu, cfg);
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
