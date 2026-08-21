//! DeltaNet linear attention transformer architecture with delta rule state recurrence.
//!
//! # Architecture Details
//! - **Delta Rule Recurrence**: Recurrent state matrix $S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$ computed per head.
//! - **Linear Attention**: Query readout $o_t = q_t S_t$ with linear $O(1)$ memory complexity per step.
//! - **SwiGLU FFN**: Feed-forward projection with RMSNorm normalization.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for DeltaNet linear attention architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaNetBaseConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub chunk_size: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl Default for DeltaNetBaseConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_heads: 16,
            head_dim: 128,
            num_layers: 24,
            intermediate_size: 5632,
            chunk_size: 64,
            rms_norm_eps: 1e-5,
            max_seq_len: 8192,
        }
    }
}

impl ModelConfig for DeltaNetBaseConfig {
    fn name(&self) -> &str {
        "delta-net-base"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Delta Attention Layer
// ---------------------------------------------------------------------------

/// DeltaNet recurrent linear attention layer.
pub struct DeltaNetAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub beta_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl DeltaNetAttention {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeltaNetBaseConfig) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let q_proj = Linear::load_shape(&ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let k_proj = Linear::load_shape(&ws.scoped("k_proj"), [cfg.hidden_size, q_dim])?;
        let v_proj = Linear::load_shape(&ws.scoped("v_proj"), [cfg.hidden_size, q_dim])?;
        let beta_proj =
            Linear::load_shape(&ws.scoped("beta_proj"), [cfg.hidden_size, cfg.num_heads])?;
        let o_proj = Linear::load_shape(&ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            beta_proj,
            o_proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Forward pass updating per-head state matrix $S \in \mathbb{R}^{H \times D \times D}$.
    pub fn forward(&self, x: &Tensor, state: &mut Option<Vec<f32>>) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let beta = self.beta_proj.forward(x)?;

        let q_v = q.to_vec_f32()?;
        let k_v = k.to_vec_f32()?;
        let v_v = v.to_vec_f32()?;
        let b_v = beta.to_vec_f32()?;

        let q_dim = self.num_heads * self.head_dim;
        let d = self.head_dim;
        let state_size = self.num_heads * d * d;

        let mut s_mat = state.clone().unwrap_or_else(|| vec![0.0f32; state_size]);
        let mut out = vec![0.0f32; seq_len * q_dim];

        for t in 0..seq_len {
            for h in 0..self.num_heads {
                let q_slice = &q_v[t * q_dim + h * d..t * q_dim + (h + 1) * d];
                let k_slice = &k_v[t * q_dim + h * d..t * q_dim + (h + 1) * d];
                let v_slice = &v_v[t * q_dim + h * d..t * q_dim + (h + 1) * d];
                let b_val = 1.0 / (1.0 + (-b_v[t * self.num_heads + h]).exp()); // sigmoid

                let s_head_off = h * d * d;

                // Compute Sk = S * k
                let mut sk = vec![0.0f32; d];
                for i in 0..d {
                    let mut sum = 0.0f32;
                    for j in 0..d {
                        sum += s_mat[s_head_off + i * d + j] * k_slice[j];
                    }
                    sk[i] = sum;
                }

                // Delta error: delta_v = beta * (v - Sk)
                let mut delta_v = vec![0.0f32; d];
                for i in 0..d {
                    delta_v[i] = b_val * (v_slice[i] - sk[i]);
                }

                // Update S: S += delta_v * k^T
                for i in 0..d {
                    for j in 0..d {
                        s_mat[s_head_off + i * d + j] += delta_v[i] * k_slice[j];
                    }
                }

                // Query output: o = q * S^T
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..d {
                        acc += q_slice[j] * s_mat[s_head_off + i * d + j];
                    }
                    out[t * q_dim + h * d + i] = acc;
                }
            }
        }

        *state = Some(s_mat);
        let out_t = cpu_tensor(out, Shape::new(vec![seq_len, q_dim]));
        Ok(self.o_proj.forward(&out_t)?)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct DeltaNetBlock {
    pub attn_norm: RmsNorm,
    pub attn: DeltaNetAttention,
    pub ffn_norm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
}

impl DeltaNetBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeltaNetBaseConfig) -> Result<Self> {
        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let attn = DeltaNetAttention::load(&ws.scoped("self_attn"), cfg)?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

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

        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        })
    }

    pub fn forward(&self, x: &Tensor, state: &mut Option<Vec<f32>>) -> Result<Tensor> {
        let normed_attn = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&normed_attn, state)?;

        let xv = x.to_vec_f32()?;
        let av = attn_out.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;
        let gate = self.w_gate.forward(&normed_ffn)?;
        let up = self.w_up.forward(&normed_ffn)?;

        let gv = gate.to_vec_f32()?;
        let uv = up.to_vec_f32()?;
        let swiglu: Vec<f32> = gv
            .iter()
            .zip(uv.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let swiglu_t = cpu_tensor(swiglu, gate.shape().clone());
        let mlp_out = self.w_down.forward(&swiglu_t)?;

        let r1v = res1_t.to_vec_f32()?;
        let mv = mlp_out.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct DeltaNetBase {
    pub cfg: DeltaNetBaseConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<DeltaNetBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl DeltaNetBase {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DeltaNetBaseConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DeltaNetBaseConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = DeltaNetBlock::load(&layer_ws, &cfg)?;
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

impl Model for DeltaNetBase {
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

impl CausalLm for DeltaNetBase {
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
        let ids = input_ids.to_vec_f32()?;
        let seq_len = ids.len();
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
        let mut states = vec![None; self.layers.len()];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mut states[layer_idx])?;
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
    fn test_delta_net_config() {
        let cfg = DeltaNetBaseConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.chunk_size, 64);
    }
}
