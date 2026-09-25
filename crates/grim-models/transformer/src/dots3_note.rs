//! Dots-3 Note architecture with specialized reasoning/note token attention, RoPE positional embeddings, SwiGLU feed-forward networks, and RMSNorm.
//! # Architecture Details - **Attention**: GQA with RoPE rotation.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{
    ArithType, CoreTensorOps, DType, Device, QuantProvenance, Shape, Storage, Tensor, YaRNParams,
};

// Config

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

// Feed Forward

pub struct Dots3NoteMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    /// Fused Q8_0 Gate+Up projection blob on ROCm for single-token decode.
    pub w_gate_up_q80_fused: Option<Arc<grim_backend_rocm::FusedGateUpWeights>>,
}

impl Dots3NoteMlp {
    pub fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let gate_proj = Linear::load_shape(&ws.scoped("gate_proj"), [in_dim, hidden_dim])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [in_dim, hidden_dim])?;
        let down_proj = Linear::load_shape(&ws.scoped("down_proj"), [hidden_dim, in_dim])?;
        let device = gate_proj.weight.device().clone();
        let is_q80 = |weight: &Tensor| {
            matches!(
                weight.dtype().storage,
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80)
            )
        };
        let w_gate_up_q80_fused = if matches!(&device, Device::Rocm(_))
            && std::env::var("GRIM_FUSED_FFN").as_deref() != Ok("0")
            && is_q80(&gate_proj.weight)
            && is_q80(&up_proj.weight)
        {
            let ordinal = match &device {
                Device::Rocm(ordinal) => *ordinal,
                _ => 0,
            };
            grim_backend_rocm::RocmDevice::try_new(ordinal)
                .ok()
                .and_then(|dev| {
                    dev.build_fused_gate_up_q80(
                        gate_proj.weight.storage().as_ref(),
                        up_proj.weight.storage().as_ref(),
                    )
                    .ok()
                })
                .map(Arc::new)
        } else {
            None
        };
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            w_gate_up_q80_fused,
        })
    }

    fn fused_gate_up_dot4_decode(
        &self,
        norm_x: &Tensor,
        fused: &grim_backend_rocm::FusedGateUpWeights,
    ) -> Result<(Tensor, Tensor)> {
        let dev = grim_backend_rocm::RocmDevice::shared(match norm_x.device() {
            Device::Rocm(ordinal) => *ordinal,
            _ => 0,
        });
        let hidden = norm_x.shape().dims().last().copied().unwrap_or(0);
        let q81_bytes = (hidden / 32) * 36;
        let act_q81 = Tensor::new(
            Arc::from(dev.zeros(
                &Shape::new(vec![q81_bytes]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?),
            Shape::new(vec![q81_bytes]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
            QuantProvenance::GrimNative,
            norm_x.device().clone(),
        );
        let x_rocm = grim_backend_rocm::as_rocm(norm_x.storage().as_ref())?;
        let act_rocm = grim_backend_rocm::as_rocm(act_q81.storage().as_ref())?;
        dev.launch_quantize_q8_1(x_rocm, act_rocm, 1, hidden)?;
        let out = dev.launch_fused_gate_up_dot4(
            act_rocm,
            &fused.storage,
            fused.n_gate,
            fused.n_up,
            hidden,
        )?;
        let out_arc: Arc<dyn grim_tensor::BackendStorage> = Arc::from(out);
        let gate_view = grim_backend_rocm::RocmStorageView::from_offset(
            out_arc.clone(),
            0,
            fused.n_gate * 4,
            Shape::new(vec![1, fused.n_gate]),
        )?;
        let up_view = grim_backend_rocm::RocmStorageView::from_offset(
            out_arc,
            fused.n_gate * 4,
            fused.n_up * 4,
            Shape::new(vec![1, fused.n_up]),
        )?;
        Ok((
            Tensor::new(
                Arc::from(gate_view),
                Shape::new(vec![1, fused.n_gate]),
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            ),
            Tensor::new(
                Arc::from(up_view),
                Shape::new(vec![1, fused.n_up]),
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            ),
        ))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (g, u) = match self.w_gate_up_q80_fused.as_ref() {
            Some(fused) if x.shape().dims().first().copied() == Some(1) => {
                self.fused_gate_up_dot4_decode(x, fused)?
            }
            _ => (self.gate_proj.forward(x)?, self.up_proj.forward(x)?),
        };
        let act = grim_nn::modules::silu_mul_on_device(&g, &u)?;
        Ok(self.down_proj.forward(&act)?)
    }
}

// Block

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
    /// Fused Q8_0 QKV projection blob on ROCm (Phase 2b). Issues 1 dot4 GEMV
    /// instead of 3 when single-token decoding.
    pub wqkv_q80_fused: Option<std::sync::Arc<grim_backend_rocm::FusedQkvWeights>>,
}

impl Dots3NoteBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Dots3NoteConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
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

        // Phase 2b: build a fused Q8_0 QKV projection blob when all three
        // projections are Q8_0 on ROCm. Falls back to 3 separate GEMVs otherwise.
        let wqkv_q80_fused =
            crate::shared_attention::build_fused_qkv_q80(&wq, &wk, &wv).map(std::sync::Arc::new);

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
            wqkv_q80_fused,
        })
    }

    /// GPU-first: Q/K/V, RoPE and attention run on the tensor's device; the
    /// host path is only reached through the fused-kernel fallback guard.
    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

        // Phase 2b: single-token decode issues ONE fused Q8_0 QKV GEMV
        // (quantize + fused dot4 + RoPE) instead of 3 separate GEMV + RoPE.
        let (q, k, v) = match self.wqkv_q80_fused.as_ref() {
            Some(fused) if seq_len == 1 => crate::shared_attention::fused_qkv_project(
                &normed_attn,
                fused,
                &self.rope,
                self.num_heads,
                self.num_kv_heads,
                positions,
            )?,
            _ => {
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
                (q, k, v)
            }
        };

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

// Model

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
            weight: cpu_tensor(
                vec![1.0; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
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

// Tests

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
    #[allow(clippy::field_reassign_with_default)]
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

        let logits = model
            .forward(session.as_mut(), &input_ids, &positions, &[])
            .unwrap();
        assert_eq!(logits.shape().dims(), &[2, 32]);

        let last_h = session.get_last_hidden_state();
        assert!(last_h.is_some());
        assert_eq!(last_h.unwrap().shape().dims(), &[2, 16]);
    }
}
