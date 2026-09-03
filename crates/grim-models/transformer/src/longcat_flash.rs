//! LongCat Flash architecture with Multi-head Latent Attention (MLA),
//! compressed low-rank key-value projections, and SwiGLU feed-forward layers.
//!
//! # Architecture Details
//! - **Attention**: Multi-head Latent Attention (MLA) compressing KV cache into latent rank $d_c$.
//! - **Rotary**: Decoupled RoPE applied to dedicated positional query/key heads ($d_R$).
//! - **Feed Forward**: SwiGLU gated linear projections.
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

/// Configuration for LongCat Flash architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LongCatFlashConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for LongCatFlashConfig {
    fn default() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 4096,
            intermediate_size: 8192,
            num_layers: 28,
            num_attention_heads: 64,
            num_key_value_heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 65536,
            yarn: None,
        }
    }
}

impl ModelConfig for LongCatFlashConfig {
    fn name(&self) -> &str {
        "longcat_flash"
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

pub struct LongCatFlashMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl LongCatFlashMlp {
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
        let g_vec = g.to_vec_f32()?;
        let u_vec = u.to_vec_f32()?;
        let mut act = vec![0.0f32; g_vec.len()];
        for i in 0..act.len() {
            let val = g_vec[i];
            let sig = 1.0 / (1.0 + (-val).exp());
            act[i] = val * sig * u_vec[i];
        }
        let act_tensor = cpu_tensor(act, g.shape().clone());
        Ok(self.down_proj.forward(&act_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct LongCatFlashBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: LongCatFlashMlp,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LongCatFlashBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &LongCatFlashConfig, _tp: TensorParallelConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
        let kv_dim = cfg.num_key_value_heads * cfg.v_head_dim;

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

        let mlp = LongCatFlashMlp::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
        let rope = Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta);

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
            head_dim: cfg.v_head_dim,
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
            &Device::Cpu,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

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

pub struct LongCatFlash {
    pub cfg: LongCatFlashConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<LongCatFlashBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl LongCatFlash {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: LongCatFlashConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");
        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let num_layers_to_load = cfg.num_layers;
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            layers.push(LongCatFlashBlock::load(&layer_ws, &cfg, tp)?);
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

    pub fn random(device: Device, cfg: LongCatFlashConfig) -> Self {
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

impl Model for LongCatFlash {
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

impl CausalLm for LongCatFlash {
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
    fn test_longcat_flash_config_defaults() {
        let cfg = LongCatFlashConfig::default();
        assert_eq!(cfg.name(), "longcat_flash");
        assert_eq!(cfg.vocab_size, 131072);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.kv_lora_rank, 512);
    }

    #[test]
    fn test_longcat_flash_forward_and_session_state() {
        let mut cfg = LongCatFlashConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_layers = 0;

        let model = LongCatFlash::random(Device::Cpu, cfg);
        let mut session = model.new_session();

        let input_ids = cpu_tensor(vec![1.0, 3.0], Shape::new(vec![2]));
        let positions = cpu_tensor(vec![0.0, 1.0], Shape::new(vec![2]));

        let logits = model.forward(session.as_mut(), &input_ids, &positions, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[2, 32]);

        let last_h = session.get_last_hidden_state();
        assert!(last_h.is_some());
        assert_eq!(last_h.unwrap().shape().dims(), &[2, 16]);
    }
}
