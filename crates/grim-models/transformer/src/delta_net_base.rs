//! DeltaNet linear attention transformer architecture with delta rule state recurrence.
//!
//! # Architecture Details
//! - **Delta Rule Recurrence**: Recurrent state matrix $S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$ computed per head.
//! - **Linear Attention**: Query readout $o_t = q_t S_t$ with linear $O(1)$ memory complexity per step.
//! - **SwiGLU FFN**: Feed-forward projection with RMSNorm normalization.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
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
            // An out-of-vocab token id is a tokenizer contract violation:
            // silently embedding a zero row would feed garbage through every
            // downstream layer with no signal — error instead.
            if tok >= self.cfg.vocab_size {
                return Err(Error::Session(format!(
                    "delta_net: token id {tok} at position {i} is out of vocab (vocab_size = {})",
                    self.cfg.vocab_size
                )));
            }
            hidden[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size].copy_from_slice(
                &embed_w[tok * self.cfg.hidden_size..(tok + 1) * self.cfg.hidden_size],
            );
        }

        let mut x = cpu_tensor(hidden, Shape::new(vec![seq_len, self.cfg.hidden_size]));

        // Audit fix (grim-models): the per-layer delta-rule states used to be
        // a LOCAL vec — every forward started from zeroed delta-state, so
        // decode was context-free after the first token (the same bug class
        // the Mamba session-state fix addressed). The states now live on the
        // session and advance across calls, mirroring the KV-cache contract.
        if session.model_state().is_none() {
            session.set_model_state(Box::new(Vec::<Option<Vec<f32>>>::new()));
        }
        let states_cell = session
            .model_state_mut()
            .ok_or_else(|| Error::Session("delta_net: session model_state vanished".into()))?;
        let states = states_cell
            .downcast_mut::<Vec<Option<Vec<f32>>>>()
            .ok_or_else(|| {
                Error::Session("delta_net: session model_state holds another model's state".into())
            })?;
        if states.len() != self.layers.len() {
            states.resize_with(self.layers.len(), || None);
        }

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

// ---------------------------------------------------------------------------
// Numeric reference + session-state gates (audit follow-up).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod delta_numeric_reference_tests {
    use super::*;
    use grim_core::session::Inner;
    use grim_nn::Linear;

    fn lin(weight: Vec<f32>, out_dim: usize, in_dim: usize) -> Linear {
        Linear::from_tensor(
            cpu_tensor(weight, Shape::new(vec![out_dim, in_dim])),
            None,
        )
    }

    /// Deterministic pseudo-random weights (LCG) so the reference test
    /// exercises non-trivial projections.
    fn weights(seed: u64, n: usize) -> Vec<f32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((st >> 33) % 2000) as f32 - 1000.0) / 1000.0 * 0.3
            })
            .collect()
    }

    fn test_attention() -> DeltaNetAttention {
        let d = 4usize;
        DeltaNetAttention {
            q_proj: lin(weights(1, d * d), d, d),
            k_proj: lin(weights(2, d * d), d, d),
            v_proj: lin(weights(3, d * d), d, d),
            beta_proj: lin(weights(4, d), 1, d),
            o_proj: lin(weights(5, d * d), d, d),
            num_heads: 1,
            head_dim: d,
        }
    }

    /// Delta rule vs an independent f64 recomputation:
    ///   S' = S + β(v − S·k)kᵀ   (β = sigmoid(β_proj·x))
    ///   o  = q·S'ᵀ
    /// run over a 2-token sequence WITH carried state (the second token's
    /// output must see the state the first token wrote).
    #[test]
    fn delta_rule_matches_f64_reference() {
        let attn = test_attention();
        let d = 4usize;
        let x_data = vec![0.4f32, -0.2, 0.6, 0.1, 0.9, 0.0, -0.5, 0.3]; // 2 tokens
        let x = cpu_tensor(x_data.clone(), Shape::new(vec![2, d]));
        let mut state = None;
        let got = attn.forward(&x, &mut state).unwrap().to_vec_f32().unwrap();
        assert!(state.is_some(), "attention must persist the delta state");

        let proj = |w: &[f32], v: &[f64]| -> Vec<f64> {
            (0..v.len())
                .map(|o| (0..v.len()).map(|i| w[o * v.len() + i] as f64 * v[i]).sum())
                .collect()
        };
        // f64 per-token projections (weight [out,in] row-major).
        let mut s = vec![0.0f64; d * d];
        for t in 0..2 {
            let xt: Vec<f64> = x_data[t * d..(t + 1) * d].iter().map(|&v| v as f64).collect();
            let q = proj(&attn.q_proj.weight.to_vec_f32().unwrap(), &xt);
            let k = proj(&attn.k_proj.weight.to_vec_f32().unwrap(), &xt);
            let v = proj(&attn.v_proj.weight.to_vec_f32().unwrap(), &xt);
            let b_logit = attn.beta_proj.weight.to_vec_f32().unwrap();
            let bl: f64 = (0..d).map(|i| b_logit[i] as f64 * xt[i]).sum();
            let beta = 1.0 / (1.0 + (-bl).exp());
            // Sk, delta update, query output.
            let mut sk = vec![0.0f64; d];
            for i in 0..d {
                sk[i] = (0..d).map(|j| s[i * d + j] * k[j]).sum();
            }
            let mut o = vec![0.0f64; d];
            for i in 0..d {
                for j in 0..d {
                    s[i * d + j] += beta * (v[i] - sk[i]) * k[j];
                }
                o[i] = (0..d).map(|j| q[j] * s[i * d + j]).sum();
            }
            let out = proj(&attn.o_proj.weight.to_vec_f32().unwrap(), &o);
            for (r, g) in out.iter().zip(&got[t * d..(t + 1) * d]) {
                assert!((r - *g as f64).abs() < 1e-4, "token {t}: reference {r} vs impl {g}");
            }
        }
    }

    fn test_model() -> DeltaNetBase {
        let cfg = DeltaNetBaseConfig {
            vocab_size: 32,
            hidden_size: 4,
            num_heads: 1,
            head_dim: 4,
            num_layers: 2,
            intermediate_size: 4,
            chunk_size: 64,
            rms_norm_eps: 1e-5,
            max_seq_len: 256,
        };
        let unit_norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; 4], Shape::new(vec![4])),
            eps: 1e-5,
        };
        let layer = |seed: u64| DeltaNetBlock {
            attn_norm: unit_norm.clone(),
            attn: DeltaNetAttention {
                q_proj: lin(weights(seed, 16), 4, 4),
                k_proj: lin(weights(seed + 1, 16), 4, 4),
                v_proj: lin(weights(seed + 2, 16), 4, 4),
                beta_proj: lin(weights(seed + 3, 4), 1, 4),
                o_proj: lin(weights(seed + 4, 16), 4, 4),
                num_heads: 1,
                head_dim: 4,
            },
            ffn_norm: unit_norm.clone(),
            w_gate: lin(weights(seed + 5, 16), 4, 4),
            w_up: lin(weights(seed + 6, 16), 4, 4),
            w_down: lin(weights(seed + 7, 16), 4, 4),
        };
        DeltaNetBase {
            cfg,
            device: Device::Cpu,
            tok_embeddings: lin(weights(50, 4 * 32), 4, 32),
            layers: vec![layer(10), layer(20)],
            norm: unit_norm,
            output: lin(weights(60, 32 * 4), 32, 4),
        }
    }

    fn tok(v: f32) -> Tensor {
        cpu_tensor(vec![v], Shape::new(vec![1]))
    }

    /// Audit fix gate (bug #1): the delta-rule states used to be a LOCAL vec,
    /// making every forward context-free. Through ONE session, sequential
    /// single-token forwards must produce the same second-token logits as a
    /// single batched 2-token forward (per-token independence of norm/FFN
    /// means the only cross-token coupling is the delta state).
    #[test]
    fn deltanet_session_state_makes_decode_context_aware() {
        let model = test_model();
        let mut sess = Inner::new(Device::Cpu);

        let first = CausalLm::forward(&model, &mut sess, &tok(3.0), &tok(0.0), &[]).unwrap();
        let second_sequential = CausalLm::forward(&model, &mut sess, &tok(7.0), &tok(0.0), &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();

        let batched_input = cpu_tensor(vec![3.0, 7.0], Shape::new(vec![2]));
        let mut fresh_sess = Inner::new(Device::Cpu);
        let batched = CausalLm::forward(&model, &mut fresh_sess, &batched_input, &tok(0.0), &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let vocab = model.cfg.vocab_size;
        let batched_last = &batched[vocab..2 * vocab]; // second token's logits

        let max_diff = second_sequential
            .iter()
            .zip(batched_last)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "sequential decode must match batched prefill last token (state must persist): max_diff={max_diff}"
        );
        // The context-free (pre-fix) behavior would NOT match: the stateful
        // second call sees token 3's delta state, the batched run sees both.
        let _ = first;
    }

    /// Audit fix gate (bug #7): an out-of-vocab token id must be an error,
    /// never a silently-zeroed embedding row.
    #[test]
    fn deltanet_rejects_out_of_vocab_tokens() {
        let model = test_model();
        let mut sess = Inner::new(Device::Cpu);
        let oob = cpu_tensor(vec![32.0], Shape::new(vec![1])); // vocab_size = 32
        let result = CausalLm::forward(&model, &mut sess, &oob, &tok(0.0), &[]);
        assert!(result.is_err(), "OOV token must error");
    }
}
