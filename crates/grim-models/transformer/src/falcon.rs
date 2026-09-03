//! Falcon causal language model architecture with parallel attention and fused QKV.
//!
//! # Architecture Details
//! - **Fused QKV**: A single linear layer projects the normalized input into concatenated Q, K, and V matrices.
//! - **Parallel Attention & MLP**: Attention and MLP branches operate concurrently on normalized inputs:
//!   `x_out = x + Attn(LN_attn(x)) + MLP(LN_mlp(x))`.
//! - **Rotary Embedding (RoPE)**: Applied to queries and keys per attention head.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Standard LayerNorm with learnable scale and bias.
///
/// Host-side kernel gap: no backend currently exposes a LayerNorm device
/// kernel (grim-nn's own `LayerNorm` is host-bound too), so `forward` pulls
/// the input and weight/bias once per call and returns a host tensor.
/// Callers re-upload the result onto the activation's device.
#[derive(Clone)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f32,
}

impl LayerNorm {
    /// Loads LayerNorm weights and optional bias from a weight source.
    pub fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        let bias = ws.get([dim], "bias").ok();
        Ok(Self { weight, bias, eps })
    }

    /// Normalizes input across the innermost dimension with mean and variance scaling.
    /// Weight/bias are pulled once per call (never per row).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xv = x.to_vec_f32()?;
        let dim = x.shape().dims().last().copied().unwrap_or(1);
        let mut out = vec![0.0f32; xv.len()];
        let w = self.weight.to_vec_f32()?;
        let b = self.bias.as_ref().map(|b| b.to_vec_f32()).transpose()?;

        for (i, c) in xv.chunks(dim).enumerate() {
            let mean = c.iter().sum::<f32>() / dim as f32;
            let variance = c.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (variance + self.eps).sqrt();

            for j in 0..dim {
                let mut val = (c[j] - mean) * inv_std * w[j];
                if let Some(ref bias_vec) = b {
                    val += bias_vec[j];
                }
                out[i * dim + j] = val;
            }
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Falcon model architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FalconConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub parallel_attn: bool,
    pub new_decoder_architecture: bool,
    pub multi_query: bool,
}

impl Default for FalconConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_heads: 71,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 32,
            intermediate_size: 18176,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            parallel_attn: true,
            new_decoder_architecture: true,
            multi_query: true,
        }
    }
}

impl ModelConfig for FalconConfig {
    fn name(&self) -> &str {
        "falcon"
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

/// A single Falcon transformer layer featuring fused QKV and parallel residual branches.
pub struct FalconBlock {
    pub fused_qkv: Linear,
    pub dense: Linear,
    pub ln_attn: LayerNorm,
    pub ln_mlp: Option<LayerNorm>,
    pub dense_h_to_4h: Linear,
    pub dense_4h_to_h: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub parallel_attn: bool,
}

impl FalconBlock {
    /// Loads a Falcon transformer block from weight sources.
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &FalconConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let qkv_out_dim = (cfg.num_heads + 2 * cfg.num_kv_heads) * cfg.head_dim;
        let fused_qkv = Linear::load_shape(
            &ws.scoped("self_attention").scoped("query_key_value"),
            [cfg.hidden_size, qkv_out_dim],
        )?;
        let dense = Linear::load_shape(
            &ws.scoped("self_attention").scoped("dense"),
            [cfg.num_heads * cfg.head_dim, cfg.hidden_size],
        )?;

        let ln_attn = LayerNorm::load(
            &ws.scoped("ln_attn"),
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
        )?;
        let ln_mlp = if !cfg.new_decoder_architecture {
            Some(LayerNorm::load(
                &ws.scoped("ln_mlp"),
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
            )?)
        } else {
            None
        };

        let mlp_ws = ws.scoped("mlp");
        let dense_h_to_4h = Linear::load_shape(
            &mlp_ws.scoped("dense_h_to_4h"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let dense_4h_to_h = Linear::load_shape(
            &mlp_ws.scoped("dense_4h_to_h"),
            [cfg.intermediate_size, cfg.hidden_size],
        )?;

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            fused_qkv,
            dense,
            ln_attn,
            ln_mlp,
            dense_h_to_4h,
            dense_4h_to_h,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            parallel_attn: cfg.parallel_attn,
        })
    }

    /// Evaluates forward pass over input hidden states, returning the output activations.
    ///
    /// GPU-first: RoPE, KV-cache concat and attention run on the tensor's
    /// device. The fused-QKV split and the GELU activation stay host-side
    /// (no device column-slice / GELU kernels); each pulls once per call and
    /// re-uploads only the split pieces.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let attn_normed = self.ln_attn.forward(x)?;
        let mlp_normed = if let Some(ref ln) = self.ln_mlp {
            grim_nn::modules::move_to_device(&ln.forward(x)?, x.device())?
        } else {
            attn_normed.clone()
        };

        // 1. QKV projection. The fused column split has no device kernel, so
        // pull once and upload only the q/k/v pieces (kernel gap).
        let qkv = self.fused_qkv.forward(&attn_normed)?;
        let qkv_vec = qkv.to_vec_f32()?;

        let q_dim = self.num_heads * self.head_dim;
        let k_dim = self.num_kv_heads * self.head_dim;
        let v_dim = self.num_kv_heads * self.head_dim;
        let total_qkv = q_dim + k_dim + v_dim;

        let mut q_data = vec![0.0f32; seq_len * q_dim];
        let mut k_data = vec![0.0f32; seq_len * k_dim];
        let mut v_data = vec![0.0f32; seq_len * v_dim];

        for s in 0..seq_len {
            let row_offset = s * total_qkv;
            q_data[s * q_dim..(s + 1) * q_dim]
                .copy_from_slice(&qkv_vec[row_offset..row_offset + q_dim]);
            k_data[s * k_dim..(s + 1) * k_dim]
                .copy_from_slice(&qkv_vec[row_offset + q_dim..row_offset + q_dim + k_dim]);
            v_data[s * v_dim..(s + 1) * v_dim]
                .copy_from_slice(&qkv_vec[row_offset + q_dim + k_dim..row_offset + total_qkv]);
        }

        let q_rot = grim_nn::modules::move_to_device(
            &cpu_tensor(q_data, Shape::new(vec![seq_len, q_dim])),
            x.device(),
        )?;
        let k_rot = grim_nn::modules::move_to_device(
            &cpu_tensor(k_data, Shape::new(vec![seq_len, k_dim])),
            x.device(),
        )?;
        let v_tensor = grim_nn::modules::move_to_device(
            &cpu_tensor(v_data, Shape::new(vec![seq_len, v_dim])),
            x.device(),
        )?;

        // Apply RoPE on-device.
        let q_rot = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &q_rot,
            self.num_heads,
            positions,
        )?;
        let k_rot = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &k_rot,
            self.num_kv_heads,
            positions,
        )?;

        // Device-side cache update: prev rows stay resident, only the new
        // rows are appended (D2D arena copy when the backend supports it).
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let full_k = crate::shared_attention::concat_rows_on_device(prev_k, &k_rot)?;
            let full_v = crate::shared_attention::concat_rows_on_device(prev_v, &v_tensor)?;
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k_rot.clone(), v_tensor.clone()));
            (k_rot, v_tensor)
        };
        let kv_len = k_all.shape().dims()[0];

        // GQA attention on-device; the shared helper applies the causal mask
        // at cache_offset + s. On backends that reject the kernel call fall
        // back to the host-history entry (scalar reference on CPU).
        let attn_tensor = match crate::shared_attention::fused_attention_tensors(
            &q_rot,
            &k_all,
            &v_all,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            kv_len,
            None,
        ) {
            Ok(t) => t,
            Err(_) => crate::shared_attention::fused_or_scalar_attention(
                &q_rot.to_vec_f32()?,
                &k_all.to_vec_f32()?,
                &v_all.to_vec_f32()?,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                None,
                x.device(),
            )?,
        };
        let attn_proj = self.dense.forward(&attn_tensor)?;

        // 2. MLP branch (GELU — host kernel gap, pulled once, re-uploaded).
        let mlp_mid = self.dense_h_to_4h.forward(&mlp_normed)?;
        let mlp_mid_v = mlp_mid.to_vec_f32()?;
        let gelu_v: Vec<f32> = mlp_mid_v
            .iter()
            .map(|&v| {
                let c = 0.797_884_6 * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + c.tanh())
            })
            .collect();
        let mlp_act = grim_nn::modules::move_to_device(
            &cpu_tensor(gelu_v, mlp_mid.shape().clone()),
            x.device(),
        )?;
        let mlp_proj = self.dense_4h_to_h.forward(&mlp_act)?;

        // 3. Parallel residual combination
        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        grim_nn::modules::add_on_device(&res1, &mlp_proj).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct Falcon {
    pub cfg: FalconConfig,
    pub device: Device,
    pub word_embeddings: Linear,
    pub layers: Vec<FalconBlock>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}

impl Falcon {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: FalconConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: FalconConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = if ws.has_tensor("transformer.h.0.self_attention.query_key_value.weight") {
            ws.scoped("transformer")
        } else {
            ws.scoped("model")
        };

        let word_embeddings = Linear::load_shape(
            &root.scoped("word_embeddings"),
            [cfg.vocab_size, cfg.hidden_size],
        )
        .or_else(|_| {
            Linear::load_shape(
                &root.scoped("embed_tokens"),
                [cfg.vocab_size, cfg.hidden_size],
            )
        })?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("h").scoped(&i.to_string());
            let block = FalconBlock::load(&layer_ws, &cfg, tp)?;
            layers.push(block);
        }

        let ln_f = LayerNorm::load(
            &root.scoped("ln_f"),
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
        )
        .or_else(|_| {
            LayerNorm::load(
                &root.scoped("norm"),
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
            )
        })?;

        let lm_head = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| word_embeddings.clone());

        Ok(Self {
            cfg,
            device,
            word_embeddings,
            layers,
            ln_f,
            lm_head,
        })
    }
}

impl Model for Falcon {
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

impl CausalLm for Falcon {
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

        let ids_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();

        // GPU-first embedding gather: rows land on the weight's device; the
        // vocab×hidden table never crosses to host.
        let mut x = grim_nn::embedding_gather_on_device(
            &self.word_embeddings.weight,
            &ids_u32,
            seq_len,
            self.cfg.hidden_size,
        )?;
        if session.model_state().is_none() {
            let fresh: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
            session.set_model_state(Box::new(fresh));
        }

        let kv_caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<(Tensor, Tensor)>>>())
            .expect("Falcon::forward: model_state must be Vec<Option<(Tensor, Tensor)>>");

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, &mut kv_caches[layer_idx])?;
        }

        let normed = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_falcon_config_serialization() {
        let cfg = FalconConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: FalconConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hidden_size, cfg.hidden_size);
        assert_eq!(parsed.num_heads, cfg.num_heads);
    }

    #[test]
    fn test_falcon_session_kv_cache_persistence() {
        let mut cfg = FalconConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_layers = 1;
        cfg.num_heads = 2;
        cfg.num_kv_heads = 1;
        cfg.head_dim = 8;
        cfg.parallel_attn = true;
        cfg.new_decoder_architecture = true;

        let word_embeddings = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 32 * 16], Shape::new(vec![32, 16])),
            None,
        );
        let ln = LayerNorm {
            weight: cpu_tensor(vec![1.0f32; 16], Shape::new(vec![16])),
            bias: Some(cpu_tensor(vec![0.0f32; 16], Shape::new(vec![16]))),
            eps: 1e-5,
        };
        let lm_head = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 32 * 16], Shape::new(vec![32, 16])),
            None,
        );
        let qkv_out_dim = (2 + 2 * 1) * 8; // 32
        let fused_qkv = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 16 * qkv_out_dim], Shape::new(vec![qkv_out_dim, 16])),
            None,
        );
        let dense = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 16 * 16], Shape::new(vec![16, 16])),
            None,
        );
        let dense_h_to_4h = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 16 * 32], Shape::new(vec![32, 16])),
            None,
        );
        let dense_4h_to_h = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 32 * 16], Shape::new(vec![16, 32])),
            None,
        );

        let block = FalconBlock {
            fused_qkv,
            dense,
            ln_attn: ln.clone(),
            ln_mlp: None,
            dense_h_to_4h,
            dense_4h_to_h,
            rope: Rope::new(8, 10000.0),
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            parallel_attn: true,
        };

        let model = Falcon {
            cfg,
            device: Device::Cpu,
            word_embeddings,
            layers: vec![block],
            ln_f: ln,
            lm_head,
        };

        let mut session = model.new_session();
        let tok0 = cpu_tensor(vec![1.0f32], Shape::new(vec![1]));
        let pos0 = cpu_tensor(vec![0.0f32], Shape::new(vec![1]));
        let _ = model.forward(session.as_mut(), &tok0, &pos0, &[]).unwrap();

        let caches = session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Option<(Tensor, Tensor)>>>())
            .expect("session model_state holds kv caches");
        assert!(caches[0].is_some(), "Layer 0 cache must be populated");
        let (k, _v) = caches[0].as_ref().unwrap();
        assert_eq!(k.shape().dims()[0], 1, "Cache length after 1 token is 1");

        let tok1 = cpu_tensor(vec![2.0f32], Shape::new(vec![1]));
        let pos1 = cpu_tensor(vec![1.0f32], Shape::new(vec![1]));
        let _ = model.forward(session.as_mut(), &tok1, &pos1, &[]).unwrap();

        let caches2 = session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Option<(Tensor, Tensor)>>>())
            .unwrap();
        let (k2, _v2) = caches2[0].as_ref().unwrap();
        assert_eq!(k2.shape().dims()[0], 2, "Cache length after 2 tokens is 2");
    }
}
