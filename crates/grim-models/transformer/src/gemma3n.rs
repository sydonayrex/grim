//! Gemma 3n transformer architecture with GeGLU activations and 3-norm layer normalization sandwich.
//!
//! # Architecture Details
//! - **GeGLU Activation**: Feed-forward network uses GELU-gated linear units: $\text{MLP}(x) = (W_{\text{gate}} x \cdot \text{GELU}(W_{\text{gate}} x)) \odot (W_{\text{up}} x) \cdot W_{\text{down}}$.
//! - **Triple LayerNorm Sandwich**: Input RMSNorm, Post-Attention RMSNorm, and Post-FeedForward RMSNorm.
//! - **Embedding Scaling**: Token embeddings are scaled by $\sqrt{d_{\text{model}}}$.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Gemma 3n architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gemma3nConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub sliding_window_size: usize,
    pub query_pre_attn_scalar: f32,
}

impl Default for Gemma3nConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 2048,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            num_layers: 18,
            intermediate_size: 16384,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 8192,
            sliding_window_size: 1024,
            query_pre_attn_scalar: 256.0,
        }
    }
}

impl ModelConfig for Gemma3nConfig {
    fn name(&self) -> &str {
        "gemma3n"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// Gemma 3n transformer block with 3 RMSNorms and GeGLU MLP.
pub struct Gemma3nBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub post_ffw_layernorm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub query_scale: f32,
    pub sliding_window: usize,
}

impl Gemma3nBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Gemma3nConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

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
        let post_ffw_layernorm = RmsNorm::load(
            &ws.scoped("post_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .or_else(|_| {
            RmsNorm::load(
                &ws.scoped("post_attention_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )
        })?;

        let mlp_ws = ws.scoped("mlp");
        let w_gate = Linear::load_shape(
            &mlp_ws.scoped("gate_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w_up = Linear::load_shape(
            &mlp_ws.scoped("up_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w_down = Linear::load_shape(
            &mlp_ws.scoped("down_proj"),
            [cfg.intermediate_size, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);
        let query_scale = 1.0 / (cfg.query_pre_attn_scalar.sqrt());

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            post_ffw_layernorm,
            w_gate,
            w_up,
            w_down,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            query_scale,
            sliding_window: cfg.sliding_window_size,
        })
    }

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

        let q_rot = cpu_tensor(q_vec, Shape::new(vec![seq_len, q_dim]));
        let k_rot = cpu_tensor(k_vec, Shape::new(vec![seq_len, kv_dim]));

        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            new_k.extend(k_rot.to_vec_f32()?);
            new_v.extend(v.to_vec_f32()?);
            let total_seq = new_k.len() / kv_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, kv_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, kv_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k_rot.clone(), v.clone()));
            (k_rot, v)
        };

        let q_heads = q_rot.to_vec_f32()?;
        let k_heads = k_all.to_vec_f32()?;
        let v_heads = v_all.to_vec_f32()?;

        let window = if self.sliding_window > 0 {
            Some(self.sliding_window)
        } else {
            None
        };
        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            window,
            x.device(),
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;
        let post_attn = self.post_attention_layernorm.forward(&attn_proj)?;

        let xv = x.to_vec_f32()?;
        let av = post_attn.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        // GeGLU MLP
        let gate = self.w_gate.forward(&res1_t)?;
        let up = self.w_up.forward(&res1_t)?;
        let gv = gate.to_vec_f32()?;
        let uv = up.to_vec_f32()?;
        let geglu: Vec<f32> = gv
            .iter()
            .zip(uv.iter())
            .map(|(&g, &u)| {
                // approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let gelu = 0.5 * g * (1.0 + ((0.7978845608 * (g + 0.044715 * g * g * g)).tanh()));
                gelu * u
            })
            .collect();
        let geglu_t = cpu_tensor(geglu, gate.shape().clone());
        let mlp_out = self.w_down.forward(&geglu_t)?;
        let post_mlp = self.post_ffw_layernorm.forward(&mlp_out)?;

        let r1v = res1_t.to_vec_f32()?;
        let mv = post_mlp.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct Gemma3n {
    pub cfg: Gemma3nConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<Gemma3nBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
    pub embed_scale: f32,
}

impl Gemma3n {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Gemma3nConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Gemma3nConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = Gemma3nBlock::load(&layer_ws, &cfg, tp)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| tok_embeddings.clone());

        let embed_scale = (cfg.hidden_size as f32).sqrt();

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
            embed_scale,
        })
    }
}

impl Model for Gemma3n {
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

impl CausalLm for Gemma3n {
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
                let slice = &embed_w[tok * self.cfg.hidden_size..(tok + 1) * self.cfg.hidden_size];
                for (d, &val) in slice.iter().enumerate() {
                    hidden[i * self.cfg.hidden_size + d] = val * self.embed_scale;
                }
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

    #[test]
    fn test_gemma3n_config() {
        let cfg = Gemma3nConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.head_dim, 256);
    }
}
