//! Qwen3.5 / Qwen3.8 hybrid SSM (Mamba) + GQA Attention + SwiGLU FFN.
//! Supports `Qwen3.8-27B` and related hybrid GGUF checkpoints with fused `attn_qkv`, `attn_gate`, 1D short-convolution SSM layers,.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::{Inner, SessionT};
use grim_nn::modules::{Embedding, Linear, RmsNorm, pick_device_for_storage_device};
use grim_nn::{TensorParallelConfig, WeightSource};
use grim_tensor::{
    ArithType, CoreTensorOps, DType, Device, QuantProvenance, Shape, Storage, Tensor,
};

// Config

#[derive(Debug, Clone)]
pub struct Qwen35Config {
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

    // Hybrid SSM parameters
    pub full_attention_interval: usize,
    pub ssm_d_state: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_conv: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,

    // Multi-device pipeline distribution
    pub devices: Vec<Device>,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self {
            vocab_size: 248320,
            hidden_size: 5120,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            num_layers: 65,
            intermediate_size: 17408,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 262144,
            full_attention_interval: 4,
            ssm_d_state: 128,
            ssm_d_inner: 6144,
            ssm_d_conv: 4,
            ssm_dt_rank: 48,
            ssm_n_group: 16,
            devices: Vec::new(),
        }
    }
}

impl ModelConfig for Qwen35Config {
    fn name(&self) -> &str {
        "qwen35"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Layer Cache

pub struct Qwen35LayerCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub current_pos: usize,
    /// WI-kv (qwen35): device-resident K/V arenas for full-attention layers.
    /// History stays on the GPU so decode uploads only the current step's rows instead of.
    #[doc(hidden)]
    pub k_device: Option<Box<dyn grim_tensor::BackendStorage>>,
    #[doc(hidden)]
    pub v_device: Option<Box<dyn grim_tensor::BackendStorage>>,
}

// `BackendStorage` doesn't implement Debug — hand-roll one that prints the
// host fields and omits the device arenas.
impl std::fmt::Debug for Qwen35LayerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35LayerCache")
            .field("k_cache", &self.k_cache.len())
            .field("v_cache", &self.v_cache.len())
            .field("conv_state", &self.conv_state.len())
            .field("ssm_state", &self.ssm_state.len())
            .field("current_pos", &self.current_pos)
            .field("k_device", &self.k_device.is_some())
            .field("v_device", &self.v_device.is_some())
            .finish()
    }
}

// Hand-rolled Clone: Box<dyn BackendStorage> isn't Clone; cloned caches start
// without the device arenas and rebuild them on the next forward.
impl Clone for Qwen35LayerCache {
    fn clone(&self) -> Self {
        Self {
            k_cache: self.k_cache.clone(),
            v_cache: self.v_cache.clone(),
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
            current_pos: self.current_pos,
            k_device: None,
            v_device: None,
        }
    }
}

impl Qwen35LayerCache {
    pub fn new(cfg: &Qwen35Config) -> Self {
        let conv_dim = cfg.hidden_size.max(cfg.ssm_d_inner) * 2;
        let conv_size = (cfg.ssm_d_conv.max(1) - 1) * conv_dim;
        let ssm_size = cfg.ssm_n_group.max(1)
            * cfg.ssm_d_state.max(1)
            * (cfg.ssm_d_inner / cfg.ssm_n_group.max(1));
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            conv_state: vec![0.0; conv_size.max(1)],
            ssm_state: vec![0.0; ssm_size.max(1)],
            current_pos: 0,
            k_device: None,
            v_device: None,
        }
    }
}

// Block

pub struct Qwen35Block {
    pub device: Device,
    pub attn_norm: RmsNorm,

    // Attention path tensors (for full attention layers)
    pub wq: Option<Linear>,
    pub wk: Option<Linear>,
    pub wv: Option<Linear>,
    pub wo: Option<Linear>,
    pub attn_q_norm: Option<RmsNorm>,
    pub attn_k_norm: Option<RmsNorm>,

    // SSM path tensors (for recurrent layers)
    pub attn_qkv: Option<Linear>,
    pub attn_gate: Option<Linear>,
    pub ssm_out: Option<Linear>,
    pub ssm_conv1d: Option<Tensor>,
    pub ssm_conv_vec: Option<Vec<f32>>,
    pub ssm_a: Option<Vec<f32>>,
    pub ssm_alpha: Option<Linear>,
    pub ssm_beta: Option<Linear>,
    pub ssm_dt_bias: Option<Vec<f32>>,
    pub ssm_norm: Option<Vec<f32>>,

    // Feed-forward Network
    pub post_attention_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,

    pub is_full_attention: bool,
    pub layer_idx: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Fused Q8_0 QKV projection blob on ROCm (Phase 2b). Only built when all
    /// three projections are present AND row-exact (no TP padding); issues
    /// 1 dot4 GEMV instead of 3 on single-token decode.
    pub wqkv_q80_fused: Option<std::sync::Arc<grim_backend_rocm::FusedQkvWeights>>,
    /// Concatenated Q4_K Gate+Up weights for the local tensor-parallel shard.
    /// This uses the safe Q4_K fused-dequant/WMMA path, not the RDNA4-incompatible
    /// dot4 Q4_K kernel.
    pub w_gate_up_q4k_fused: Option<std::sync::Arc<grim_backend_rocm::FusedGateUpQ4KWeights>>,
}

impl Qwen35Block {
    pub fn load_tp(
        ws: &WeightSource<'_>,
        cfg: &Qwen35Config,
        layer_idx: usize,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let device = ws.device();
        let is_full_attention = (layer_idx + 1) % cfg.full_attention_interval.max(1) == 0;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm, attn_qkv, attn_gate, ssm_out) =
            if is_full_attention {
                let wq = Linear::load_column_parallel(
                    &ws.pp("attn_q"),
                    cfg.hidden_size,
                    q_dim.max(12288),
                    false,
                    tp,
                )
                .ok();
                let wk = Linear::load_column_parallel(
                    &ws.pp("attn_k"),
                    cfg.hidden_size,
                    kv_dim,
                    false,
                    tp,
                )
                .ok();
                let wv = Linear::load_column_parallel(
                    &ws.pp("attn_v"),
                    cfg.hidden_size,
                    kv_dim,
                    false,
                    tp,
                )
                .ok();
                let wo = Linear::load_row_parallel(
                    &ws.pp("attn_output"),
                    q_dim,
                    cfg.hidden_size,
                    false,
                    tp,
                )
                .ok();
                let attn_q_norm =
                    RmsNorm::load(&ws.pp("attn_q_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
                let attn_k_norm =
                    RmsNorm::load(&ws.pp("attn_k_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
                (wq, wk, wv, wo, attn_q_norm, attn_k_norm, None, None, None)
            } else {
                let attn_qkv = Linear::load_column_parallel(
                    &ws.pp("attn_qkv"),
                    cfg.hidden_size,
                    qkv_dim.max(10240),
                    false,
                    tp,
                )
                .ok();
                let attn_gate = Linear::load_column_parallel(
                    &ws.pp("attn_gate"),
                    cfg.hidden_size,
                    q_dim.max(6144),
                    false,
                    tp,
                )
                .ok();
                let ssm_out = Linear::load_row_parallel(
                    &ws.pp("ssm_out"),
                    q_dim.max(6144),
                    cfg.hidden_size,
                    false,
                    tp,
                )
                .ok();
                (
                    None, None, None, None, None, None, attn_qkv, attn_gate, ssm_out,
                )
            };

        let (ssm_conv1d, ssm_conv_vec) = if let Ok(t) = ws.get_unconstrained("ssm_conv1d.weight") {
            let vec = t.to_vec_f32().ok();
            (Some(t), vec)
        } else {
            (None, None)
        };

        let ssm_a = ws
            .get_unconstrained("ssm_a")
            .ok()
            .and_then(|t| t.to_vec_f32().ok());
        let ssm_alpha = Linear::load_column_parallel(
            &ws.pp("ssm_alpha"),
            cfg.hidden_size,
            cfg.ssm_dt_rank,
            false,
            tp,
        )
        .ok();
        let ssm_beta = Linear::load_column_parallel(
            &ws.pp("ssm_beta"),
            cfg.hidden_size,
            cfg.ssm_dt_rank,
            false,
            tp,
        )
        .ok();
        let ssm_dt_bias = ws
            .get_unconstrained("ssm_dt.bias")
            .ok()
            .and_then(|t| t.to_vec_f32().ok());
        let ssm_norm = ws
            .get_unconstrained("ssm_norm.weight")
            .ok()
            .and_then(|t| t.to_vec_f32().ok());

        let post_attention_norm = if let Ok(m) = RmsNorm::load(
            &ws.pp("post_attention_norm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        ) {
            m
        } else {
            RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?
        };

        let ffn_gate = Linear::load_column_parallel(
            &ws.pp("ffn_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
            tp,
        )?;
        let ffn_up = Linear::load_column_parallel(
            &ws.pp("ffn_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
            tp,
        )?;
        let ffn_down = Linear::load_row_parallel(
            &ws.pp("ffn_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
            tp,
        )?;

        let w_gate_up_q4k_fused = if matches!(&device, Device::Rocm(_))
            && matches!(
                std::env::var("GRIM_Q4K_FUSED_GATEUP").as_deref(),
                Ok("1" | "true" | "on" | "yes")
            )
            && matches!(
                ffn_gate.weight().dtype().storage,
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K)
            )
            && matches!(
                ffn_up.weight().dtype().storage,
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K)
            ) {
            let ordinal = match &device {
                Device::Rocm(ordinal) => *ordinal,
                _ => 0,
            };
            grim_backend_rocm::RocmDevice::try_new(ordinal)
                .ok()
                .and_then(|dev| {
                    dev.build_fused_gate_up_q4k(
                        ffn_gate.weight().storage().as_ref(),
                        ffn_up.weight().storage().as_ref(),
                    )
                    .ok()
                })
                .map(std::sync::Arc::new)
        } else {
            None
        };

        // Phase 2b: fused Q8_0 QKV blob — only when all three projections are
        // present AND row-exact (some TP shards pad rows to a minimum width;
        // the stock path cuts them via `exact()` but the fused GEMV cannot).
        let wqkv_q80_fused = if is_full_attention {
            let row_exact = |w: Option<&Linear>, want_rows: usize| {
                w.map(|l| l.weight.shape().dims().first().copied().unwrap_or(0) == want_rows)
                    .unwrap_or(false)
            };
            if row_exact(wq.as_ref(), q_dim)
                && row_exact(wk.as_ref(), kv_dim)
                && row_exact(wv.as_ref(), kv_dim)
            {
                crate::shared_attention::build_fused_qkv_q80_opt(
                    wq.as_ref(),
                    wk.as_ref(),
                    wv.as_ref(),
                )
                .map(std::sync::Arc::new)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            device,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            attn_q_norm,
            attn_k_norm,
            attn_qkv,
            attn_gate,
            ssm_out,
            ssm_conv1d,
            ssm_conv_vec,
            ssm_a,
            ssm_alpha,
            ssm_beta,
            ssm_dt_bias,
            ssm_norm,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            is_full_attention,
            layer_idx,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            wqkv_q80_fused,
            w_gate_up_q4k_fused,
        })
    }

    fn fused_gate_up_q4k_decode(
        &self,
        norm_x: &Tensor,
        fused: &grim_backend_rocm::FusedGateUpQ4KWeights,
    ) -> Result<(Tensor, Tensor)> {
        let dev = grim_backend_rocm::RocmDevice::shared(match norm_x.device() {
            Device::Rocm(ordinal) => *ordinal,
            _ => 0,
        });
        let act = grim_backend_rocm::as_rocm(norm_x.storage().as_ref())?;
        let out = dev.launch_fused_gate_up_q4k(act, fused)?;
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

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        cache: &mut Qwen35LayerCache,
    ) -> Result<Tensor> {
        let seq_len = positions.len();
        let device = x.device().clone();

        // 1. Pre-norm
        let x_normed = self.attn_norm.forward(x)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let mut out_branch = vec![0.0f32; seq_len * q_dim];

        if self.is_full_attention {
            // Attention path with separated wq, wk, wv.
            // GPU-first: projections stay on-device; RoPE runs through the device kernel; K/V are appended into the.
            let dev = pick_device_for_storage_device(&device);
            // Some TP-sharded projections emit padded rows — cut to the
            // exact [seq, width] extent via a D2D staging copy when needed.
            let exact = |t: Tensor, rows: usize, width: usize| -> Result<Tensor> {
                let want = Shape::new(vec![rows, width]);
                if t.shape().elem_count() == rows * width {
                    return crate::block::reshaped_view(&t, &want);
                }
                let scratch = dev.alloc_storage(&want, DType::F32)?;
                dev.copy_slice_range(scratch.as_ref(), 0, t.storage().as_ref(), 0, rows * width)?;
                Ok(Tensor::new(
                    scratch.into(),
                    want,
                    DType::F32,
                    t.provenance().clone(),
                    t.device().clone(),
                ))
            };

            let rope_cfg = grim_tensor::RopeConfig::new(self.head_dim, self.rope_theta);
            let rope_ext = |t: &Tensor, heads: usize| -> Result<Tensor> {
                let mut pos_ext = Vec::with_capacity(seq_len * heads);
                for &pos in positions {
                    for _ in 0..heads {
                        pos_ext.push(pos);
                    }
                }
                let t3 = crate::block::reshaped_view(
                    t,
                    &Shape::new(vec![1, seq_len * heads, self.head_dim]),
                )?;
                let (rope_s, _) =
                    dev.rope(t3.storage().as_ref(), &pos_ext, &rope_cfg, t3.shape())?;
                let roped = Tensor::new(
                    rope_s.into(),
                    t3.shape().clone(),
                    DType::F32,
                    t.provenance().clone(),
                    t.device().clone(),
                );
                crate::block::reshaped_view(
                    &roped,
                    &Shape::new(vec![seq_len, heads * self.head_dim]),
                )
            };

            // Phase 2b: single-token decode issues ONE fused Q8_0 QKV GEMV
            // (pre-rope) instead of 3 separate GEMVs; the same rope_ext and
            // arena-attention path follow unchanged.
            let (q_dev, k_dev_t, v_dev_t) = match self.wqkv_q80_fused.as_ref() {
                Some(fused) if seq_len == 1 => {
                    crate::shared_attention::fused_qkv_project_raw(&x_normed, fused)?
                }
                _ => {
                    let q_dev = match self.wq.as_ref() {
                        Some(wq) => exact(wq.forward(&x_normed)?, seq_len, q_dim)?,
                        None => Tensor::new(
                            dev.zeros(&Shape::new(vec![seq_len, q_dim]), DType::F32)?
                                .into(),
                            Shape::new(vec![seq_len, q_dim]),
                            DType::F32,
                            x_normed.provenance().clone(),
                            x_normed.device().clone(),
                        ),
                    };
                    let k_dev_t = match self.wk.as_ref() {
                        Some(wk) => exact(wk.forward(&x_normed)?, seq_len, kv_dim)?,
                        None => Tensor::new(
                            dev.zeros(&Shape::new(vec![seq_len, kv_dim]), DType::F32)?
                                .into(),
                            Shape::new(vec![seq_len, kv_dim]),
                            DType::F32,
                            x_normed.provenance().clone(),
                            x_normed.device().clone(),
                        ),
                    };
                    let v_dev_t = match self.wv.as_ref() {
                        Some(wv) => exact(wv.forward(&x_normed)?, seq_len, kv_dim)?,
                        None => Tensor::new(
                            dev.zeros(&Shape::new(vec![seq_len, kv_dim]), DType::F32)?
                                .into(),
                            Shape::new(vec![seq_len, kv_dim]),
                            DType::F32,
                            x_normed.provenance().clone(),
                            x_normed.device().clone(),
                        ),
                    };
                    (q_dev, k_dev_t, v_dev_t)
                }
            };

            let q_rope = rope_ext(&q_dev, self.num_heads)?;
            let k_rope = rope_ext(&k_dev_t, self.num_kv_heads)?;

            // Per-step Q crosses to host for the arena attention entry
            // point; K/V history never leaves the device.
            let q_all = q_rope.to_vec_f32()?;

            // Append to the device arena via copy_slice_range (device-side).
            // K/V history stays on the GPU — never round-trips to host.
            let k_new_rows = seq_len * self.num_kv_heads;
            let kv_elems = k_new_rows * self.head_dim;
            let k_cap_rows = cache
                .k_device
                .as_ref()
                .map(|s| s.shape().dims()[0])
                .unwrap_or(0);
            let v_cap_rows = cache
                .v_device
                .as_ref()
                .map(|s| s.shape().dims()[0])
                .unwrap_or(0);
            let need_rows = cache.current_pos + k_new_rows;
            let _ = v_cap_rows;

            if k_cap_rows >= need_rows {
                let k_dev = cache.k_device.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("cache.k_device missing".into())
                })?;
                let v_dev = cache.v_device.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("cache.v_device missing".into())
                })?;
                dev.copy_slice_range(
                    &**k_dev,
                    cache.current_pos * self.num_kv_heads * self.head_dim,
                    k_rope.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                dev.copy_slice_range(
                    &**v_dev,
                    cache.current_pos * self.num_kv_heads * self.head_dim,
                    v_dev_t.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
            } else {
                // Arena full — grow geometrically and re-copy via D2D.
                let new_rows = ((need_rows * 2) + 64).next_power_of_two();
                let k_idx = self.num_kv_heads * self.head_dim;
                let full_shape = Shape::new(vec![new_rows, self.num_kv_heads, self.head_dim]);
                let k_grown = dev.alloc_storage(&full_shape, DType::F32)?;
                let v_grown = dev.alloc_storage(&full_shape, DType::F32)?;
                if let Some(ref old_k) = cache.k_device {
                    dev.copy_slice_range(
                        k_grown.as_ref(),
                        0,
                        old_k.as_ref(),
                        0,
                        cache.current_pos * self.num_kv_heads * self.head_dim,
                    )?;
                }
                if let Some(ref old_v) = cache.v_device {
                    dev.copy_slice_range(
                        v_grown.as_ref(),
                        0,
                        old_v.as_ref(),
                        0,
                        cache.current_pos * self.num_kv_heads * self.head_dim,
                    )?;
                }
                dev.copy_slice_range(
                    k_grown.as_ref(),
                    cache.current_pos * k_idx,
                    k_rope.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                dev.copy_slice_range(
                    v_grown.as_ref(),
                    cache.current_pos * k_idx,
                    v_dev_t.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                cache.k_device = Some(k_grown);
                cache.v_device = Some(v_grown);
            }

            let total_kv = cache.current_pos + seq_len;
            let k_dev = cache
                .k_device
                .as_ref()
                .ok_or_else(|| grim_core::error::Error::Backend("cache.k_device missing".into()))?;
            let v_dev = cache
                .v_device
                .as_ref()
                .ok_or_else(|| grim_core::error::Error::Backend("cache.v_device missing".into()))?;
            let attn_tensor = crate::shared_attention::fused_or_scalar_attention_arena(
                &q_all,
                k_dev.as_ref(),
                v_dev.as_ref(),
                total_kv,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                None,
                &device,
            )?;
            out_branch = attn_tensor.to_vec_f32()?;
        } else {
            // SSM short-conv path with fused attn_qkv.
            // GPU-first: conv runs on device via short_conv1d wrapper; state is uploaded/downloaded around the device call.
            if let Some(ref qkv_lin) = self.attn_qkv {
                let qkv = qkv_lin.forward(&x_normed)?;
                let qkv_vec = qkv.to_vec_f32()?;
                let qkv_total_per_tok = qkv_vec.len() / seq_len;
                let conv_w = self.ssm_conv_vec.as_deref();
                let dev = pick_device_for_storage_device(&device);

                for t in 0..seq_len {
                    let base = t * qkv_total_per_tok;
                    let num_feats = q_dim.min(qkv_total_per_tok);

                    if let Some(w) = conv_w {
                        let l_conv = (w.len() / num_feats.max(1)).max(1);
                        let state_len_per_feat = l_conv.saturating_sub(1);

                        // Build per-token input [1, 1, num_feats] for device conv
                        let tok_data = &qkv_vec[base..base + num_feats];
                        let x_storage =
                            dev.from_cpu(tok_data, &Shape::new(vec![1, 1, num_feats]), DType::F32)?;
                        let x_tok = Tensor::new(
                            Arc::from(x_storage),
                            Shape::new(vec![1, 1, num_feats]),
                            DType::F32,
                            qkv.provenance().clone(),
                            device.clone(),
                        );

                        // Build conv weight [num_feats, l_conv] (depthwise)
                        let w_storage = dev.from_cpu(
                            &w[..num_feats * l_conv],
                            &Shape::new(vec![num_feats, l_conv]),
                            DType::F32,
                        )?;
                        let w_tensor = Tensor::new(
                            Arc::from(w_storage),
                            Shape::new(vec![num_feats, l_conv]),
                            DType::F32,
                            qkv.provenance().clone(),
                            device.clone(),
                        );

                        // Upload conv state to device
                        let state_data = if state_len_per_feat > 0
                            && cache.conv_state.len() >= num_feats * state_len_per_feat
                        {
                            cache.conv_state[..num_feats * state_len_per_feat].to_vec()
                        } else {
                            vec![0.0f32; num_feats * state_len_per_feat.max(1)]
                        };
                        let state_storage = dev.from_cpu(
                            &state_data,
                            &Shape::new(vec![1, state_len_per_feat.max(1), num_feats]),
                            DType::F32,
                        )?;
                        let mut state_tensor = Tensor::new(
                            Arc::from(state_storage),
                            Shape::new(vec![1, state_len_per_feat.max(1), num_feats]),
                            DType::F32,
                            qkv.provenance().clone(),
                            device.clone(),
                        );

                        // Device short-conv: updates state_tensor in-place
                        let conv_result = grim_nn::modules::short_conv1d(
                            &x_tok,
                            &w_tensor,
                            None,
                            Some(&mut state_tensor),
                        )?;

                        // Download result and updated state
                        let conv_vec = conv_result.to_vec_f32()?;
                        let state_new = state_tensor.to_vec_f32()?;
                        if state_len_per_feat > 0
                            && cache.conv_state.len() >= num_feats * state_len_per_feat
                        {
                            cache.conv_state[..num_feats * state_len_per_feat]
                                .copy_from_slice(&state_new[..num_feats * state_len_per_feat]);
                        }

                        // Apply silu (elementwise on CPU — data already downloaded)
                        for d in 0..num_feats.min(conv_vec.len()) {
                            let v = conv_vec[d];
                            out_branch[t * q_dim + d] = v / (1.0 + (-v).exp());
                        }
                    } else {
                        for d in 0..num_feats {
                            out_branch[t * q_dim + d] = silu(qkv_vec[base + d]);
                        }
                    }
                }
            }
        }

        // Apply attention gate if present (aligned per token across seq_len)
        if let Some(ref gate_lin) = self.attn_gate {
            let gate_tensor = gate_lin.forward(&x_normed)?;
            let gate_vec = gate_tensor.to_vec_f32()?;
            let gate_len_per_tok = gate_vec.len() / seq_len.max(1);
            for t in 0..seq_len {
                let gate_base = t * gate_len_per_tok;
                let out_base = t * q_dim;
                for d in 0..q_dim.min(gate_len_per_tok) {
                    let g = gate_vec[gate_base + d];
                    out_branch[out_base + d] *= 1.0 / (1.0 + (-g).exp()); // sigmoid gate
                }
            }
        }

        let branch_tensor = device_tensor(out_branch, Shape::new(vec![seq_len, q_dim]), &device)?;

        let proj_out = if let Some(ref wo) = self.wo {
            wo.forward(&branch_tensor)?
        } else if let Some(ref out_proj) = self.ssm_out {
            out_proj.forward(&branch_tensor)?
        } else {
            branch_tensor
        };

        // Residual 1
        let h = grim_nn::modules::add_on_device(x, &proj_out)?;

        // 3. Post-attention norm
        let h_normed = self.post_attention_norm.forward(&h)?;

        // 4. SwiGLU FFN
        let (gate, up) = match self.w_gate_up_q4k_fused.as_ref() {
            Some(fused) if seq_len == 1 => self.fused_gate_up_q4k_decode(&h_normed, fused)?,
            _ => (
                self.ffn_gate.forward(&h_normed)?,
                self.ffn_up.forward(&h_normed)?,
            ),
        };
        let act = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&act)?;

        // Residual 2
        let out = grim_nn::modules::add_on_device(&h, &ffn_out)?;
        Ok(out)
    }
}

// Full Model

pub struct Qwen35 {
    pub cfg: Qwen35Config,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub blocks: Vec<Qwen35Block>,
    pub output_norm: RmsNorm,
    pub output: Linear,
}

impl Qwen35 {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: Qwen35Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: Qwen35Config,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let available_devices = if !cfg.devices.is_empty() {
            cfg.devices.clone()
        } else {
            vec![device.clone()]
        };

        eprintln!(
            "[grim] Initializing Qwen3.5/3.8 hybrid model: layers={}, hidden={}, vocab={}, interval={}, devices={:?}",
            cfg.num_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.full_attention_interval,
            available_devices
        );

        let first_device = available_devices[0].clone();
        let tok_embeddings = Embedding::load(
            &ws.with_device(first_device.clone()).pp("token_embd"),
            cfg.vocab_size,
            cfg.hidden_size,
        )
        .or_else(|e| {
            fallback_on_missing(e, || {
                Embedding::load(
                    &ws.with_device(first_device).pp("tok_embeddings"),
                    cfg.vocab_size,
                    cfg.hidden_size,
                )
            })
        })?;

        let layer_devices = plan_layer_devices(
            ws,
            &available_devices,
            cfg.num_layers,
            cfg.max_seq_len,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.full_attention_interval,
        );

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_device = layer_devices[i].clone();
            if i % 10 == 0 || i + 1 == cfg.num_layers {
                eprintln!(
                    "[grim] Loading layer {}/{} on {}...",
                    i + 1,
                    cfg.num_layers,
                    layer_device
                );
            }
            let layer_ws = ws.with_device(layer_device).pp("blk").pp(&i.to_string());
            blocks.push(Qwen35Block::load_tp(&layer_ws, &cfg, i, tp)?);
        }

        let last_device = available_devices.last().unwrap_or(&device).clone();
        let output_norm = RmsNorm::load(
            &ws.with_device(last_device.clone()).pp("output_norm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .or_else(|e| {
            fallback_on_missing(e, || {
                RmsNorm::load(
                    &ws.with_device(last_device.clone()).pp("norm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )
            })
        })?;

        let output = Linear::load_column_parallel(
            &ws.with_device(last_device.clone()).pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
            tp,
        )
        .or_else(|e| {
            fallback_on_missing(e, || {
                Linear::load(
                    &ws.with_device(last_device).pp("output"),
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                )
            })
        })?;

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            blocks,
            output_norm,
            output,
        })
    }
}

impl Model for Qwen35 {
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

impl CausalLm for Qwen35 {
    fn new_session(&self) -> Box<dyn SessionT> {
        let caches: Vec<Qwen35LayerCache> = (0..self.blocks.len())
            .map(|_| Qwen35LayerCache::new(&self.cfg))
            .collect();
        let mut session = Inner::new(self.device.clone());
        session.set_model_state(Box::new(caches));
        Box::new(session)
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => {
                let v = input_ids.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => return Err(grim_tensor::Error::Unimplemented("non-F32 inputs".into()).into()),
        };
        let positions_vec: Vec<u32> = match positions.dtype() {
            d if d == DType::F32 => {
                let v = positions.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => return Err(grim_tensor::Error::Unimplemented("non-F32 positions".into()).into()),
        };

        let seq_len = ids.len();
        let mut h = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?;

        if session.model_state().is_none() {
            let fresh: Vec<Qwen35LayerCache> = (0..self.blocks.len())
                .map(|_| Qwen35LayerCache::new(&self.cfg))
                .collect();
            session.set_model_state(Box::new(fresh));
        }

        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Qwen35LayerCache>>())
            .ok_or_else(|| {
                grim_core::error::Error::Backend(
                    "Qwen35::forward: session.model_state must be Vec<Qwen35LayerCache>".into(),
                )
            })?;

        for (i, block) in self.blocks.iter().enumerate() {
            if h.device() != &block.device {
                h = grim_nn::modules::move_to_device(&h, &block.device)?;
            }
            h = block.forward(&h, &positions_vec, &mut caches[i])?;
            caches[i].current_pos += seq_len;
        }

        if h.device() != self.output_norm.weight.device() {
            h = grim_nn::modules::move_to_device(&h, self.output_norm.weight.device())?;
        }

        let normed = self.output_norm.forward(&h)?;
        let logits = self.output.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

// Helpers

/// Fraction of a device's free VRAM the loader is allowed to plan against.
///
/// The remainder absorbs activations, the KV cache, the decode graph's
/// scratch buffers, and fragmentation. Planning to 100% of free VRAM is what
/// pushes the allocator into HIP managed (host-backed) memory, which then
/// thrashes under oversubscription.
const VRAM_PLAN_FRACTION: f64 = 0.90;

/// Assign each transformer block to a device by measured capacity.
///
/// The previous placement was index arithmetic —
/// `devices[i * devices.len() / num_layers]` — which is a static round-robin
/// that ignores both per-layer weight size and how much VRAM each device
/// actually has. On a mixed pair (9070 XT + 9060 XT, 17 GB each) loading the
/// 27B Q4_K checkpoint it drove device 0 to 99.9% of VRAM while device 1 sat
/// near half full, which forced the managed-memory fallback and then failed
/// the first kernel module load.
///
/// This reserves each device's share of the decode-graph KV arenas, then walks
/// the layers in order giving each one to the device with the most remaining
/// headroom, so a large early layer cannot monopolize one card. Falls back to
/// the old round-robin whenever capacity cannot be queried (a CPU device, or a
/// backend without `hipMemGetInfo`), keeping the old behavior as the
/// conservative default.
#[allow(clippy::too_many_arguments)]
fn plan_layer_devices(
    ws: &WeightSource<'_>,
    devices: &[Device],
    num_layers: usize,
    ctx: usize,
    kv_heads: usize,
    head_dim: usize,
    interval: usize,
) -> Vec<Device> {
    // Single-device or unknown layout: nothing to plan.
    if devices.len() <= 1 {
        return vec![devices[0].clone(); num_layers];
    }

    let round_robin = |i: usize| devices[i * devices.len() / num_layers.max(1)].clone();

    // Reserve the decode-graph KV arenas BEFORE handing out weight budget.
    //
    // The arenas are allocated at full context for every full-attention layer:
    // `ctx * num_kv_heads * head_dim * 4 bytes` for K and again for V. For this
    // 27B Q4_K model at a 228k context that is ~1.87 GB per attention layer
    // across 17 such layers = ~31.8 GB, which alone exceeds the 34 GB the pair
    // has. Planning weights against the *whole* free budget therefore drives
    // both cards to ~99% and leaves the arenas nowhere to go, which is what
    // pushed the run into HIP managed memory. Reserving first lets weights and
    // KV share the budget honestly, and makes an oversized context visible as a
    // warning rather than a silent spill.
    let kv_bytes_per_attn_layer = |ctx: usize| -> u64 {
        (ctx as u64)
            .saturating_mul(kv_heads as u64)
            .saturating_mul(head_dim as u64)
            .saturating_mul(4) // f32
            .saturating_mul(2) // K and V
    };
    let num_attn_layers = (0..num_layers).filter(|i| i % interval.max(1) == 0).count();
    let kv_total = kv_bytes_per_attn_layer(ctx.max(1)).saturating_mul(num_attn_layers as u64);
    if kv_total > 0 {
        eprintln!(
            "[qwen35] reserving KV arenas: ctx={ctx}, kv_heads={kv_heads}, \
             head_dim={head_dim}, attn_layers={num_attn_layers} \
             -> {:.1} GB total",
            kv_total as f64 / 1e9
        );
    }

    // Remaining plannable bytes per device, or None when VRAM is unknowable.
    let kv_share = (kv_total as f64 / devices.len() as f64) as u64;
    let per_device: Vec<Option<u64>> = devices
        .iter()
        .map(|d| match d {
            Device::Rocm(ord) => {
                let (free, total) = grim_backend_rocm::vram_info(*ord);
                // A device reporting 0 total is not queryable; treat the whole
                // set as unplannable rather than guessing.
                if total == 0 {
                    return None;
                }
                let budget = ((free as f64) * VRAM_PLAN_FRACTION) as u64;
                Some(budget.saturating_sub(kv_share.min(budget)))
            }
            _ => None,
        })
        .collect();

    let all_known = per_device.iter().all(Option::is_some);
    let Some(mut remaining): Option<Vec<u64>> =
        all_known.then(|| per_device.into_iter().map(|v| v.unwrap_or(0)).collect())
    else {
        eprintln!(
            "[qwen35] VRAM capacity unavailable; using static layer round-robin \
             across {} devices",
            devices.len()
        );
        return (0..num_layers).map(round_robin).collect();
    };

    let mut assignment = Vec::with_capacity(num_layers);
    // Size each block from checkpoint metadata before committing it.
    let layer_bytes: Vec<u64> = (0..num_layers)
        .map(|i| ws.pp("blk").pp(&i.to_string()).prefix_bytes())
        .collect();
    let (targets, unplaced) = assign_by_headroom(&layer_bytes, &mut remaining);
    for t in targets {
        assignment.push(devices[t].clone());
    }

    if unplaced > 0 {
        // Do NOT fail here: managed memory is slow but correct, and the loader
        // already warns when it engages. Surface it loudly and continue.
        eprintln!(
            "[qwen35] WARNING: {unplaced}/{num_layers} layers exceed aggregate \
             plannable VRAM across {} devices; those will fall back to HIP \
             managed memory, which degrades throughput under oversubscription",
            devices.len()
        );
    }
    let mut per_device: Vec<(String, usize)> = Vec::new();
    for d in &assignment {
        let key = d.to_string();
        match per_device.iter_mut().find(|(k, _)| *k == key) {
            Some((_, count)) => *count += 1,
            None => per_device.push((key, 1)),
        }
    }
    eprintln!(
        "[qwen35] capacity-aware placement: {}",
        per_device
            .iter()
            .map(|(k, c)| format!("{k}={c} layers"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assignment
}

/// Greedy first-fit-decreasing-free placement: each layer goes to the device
/// with the most remaining headroom, and `remaining` is decremented in place.
///
/// Returns the chosen device index per layer plus the count of layers that did
/// not fit anywhere (those are placed on device 0 and will fall back to managed
/// memory). Pure function so placement can be unit-tested without a GPU.
fn assign_by_headroom(layer_bytes: &[u64], remaining: &mut [u64]) -> (Vec<usize>, usize) {
    let mut targets = Vec::with_capacity(layer_bytes.len());
    let mut unplaced = 0usize;
    for &bytes in layer_bytes {
        let max_remaining = remaining.iter().copied().max().unwrap_or(0);
        let target = if bytes <= max_remaining {
            remaining
                .iter()
                .enumerate()
                .max_by_key(|(_, rem)| **rem)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            unplaced += 1;
            0
        };
        remaining[target] = remaining[target].saturating_sub(bytes);
        targets.push(target);
    }
    (targets, unplaced)
}

/// Retry a load under an alternate tensor-name prefix ONLY when the first
/// attempt failed because the tensor is absent.
///
/// The loader tries several naming conventions (`output_norm` vs `norm`,
/// `token_embd` vs `tok_embeddings`). A blanket `.or_else(|_| ...)` cannot tell
/// "this prefix does not exist, try the next one" apart from "the tensor exists
/// but has the wrong shape", so the second case was silently discarded and the
/// run continued with a corrupt model. A `ShapeMismatch` here is a hard error.
fn fallback_on_missing<T, F>(err: grim_tensor::Error, alt: F) -> grim_tensor::Result<T>
where
    F: FnOnce() -> grim_tensor::Result<T>,
{
    match err {
        grim_tensor::Error::Backend(msg) if msg.contains("not found") => alt(),
        other => Err(other),
    }
}

fn device_tensor(data: Vec<f32>, shape: Shape, device: &Device) -> Result<Tensor> {
    if device == &Device::Cpu {
        Ok(cpu_tensor(data, shape))
    } else {
        let dev = pick_device_for_storage_device(device);
        let storage = dev.from_cpu(&data, &shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            device.clone(),
        ))
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn apply_rope_neox(
    v: &mut [f32],
    positions: &[u32],
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half = head_dim / 2;
    let seq_len = positions.len();

    for (t, &pos_raw) in positions.iter().enumerate().take(seq_len) {
        let pos = pos_raw as f32;
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for i in 0..half {
                let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos * freq;
                let (sin, cos) = theta.sin_cos();

                let x0 = v[base + i];
                let x1 = v[base + i + half];

                v[base + i] = x0 * cos - x1 * sin;
                v[base + i + half] = x0 * sin + x1 * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing tensor is the one case where trying the alternate name prefix
    /// is correct, so the fallback must run.
    #[test]
    fn fallback_on_missing_retries_when_tensor_absent() {
        let got = fallback_on_missing(
            grim_tensor::Error::Backend("tensor 'blk.0.norm.weight' not found in GGUF file".into()),
            || Ok::<u8, grim_tensor::Error>(7),
        );
        assert_eq!(got.expect("absent tensor should retry"), 7);
    }

    /// Regression for the Qwen3.8 vocab bug: a ShapeMismatch used to be
    /// swallowed by a blanket `.or_else(|_| ...)`, so a 248320-row
    /// `token_embd.weight` was accepted under a bogus 32000 config and the run
    /// continued against a corrupt model. It must surface instead.
    #[test]
    fn fallback_on_missing_propagates_shape_mismatch() {
        let err = fallback_on_missing(
            grim_tensor::Error::ShapeMismatch {
                expected: vec![32000, 5120],
                got: vec![248320, 5120],
            },
            || -> grim_tensor::Result<u8> { panic!("fallback must not run for a shape mismatch") },
        );
        match err {
            Err(grim_tensor::Error::ShapeMismatch { expected, got }) => {
                assert_eq!(expected, vec![32000, 5120]);
                assert_eq!(got, vec![248320, 5120]);
            }
            other => panic!("expected ShapeMismatch to propagate, got {other:?}"),
        }
    }

    /// A backend failure that is not "not found" is also a real error and must
    /// not trigger a silent retry under a different prefix.
    #[test]
    fn fallback_on_missing_propagates_other_backend_errors() {
        let err = fallback_on_missing(
            grim_tensor::Error::Backend("corrupt q4_k block".into()),
            || -> grim_tensor::Result<u8> { panic!("must not retry") },
        );
        assert!(matches!(err, Err(grim_tensor::Error::Backend(_))));
    }

    /// Even layers of equal size must land on both devices rather than piling
    /// onto one. The old static round-robin split 40/25 for 65 layers over two
    /// devices, which combined with a 248320-row vocab drove device 0 to 99.9%
    /// of VRAM.
    #[test]
    fn placement_spreads_equal_layers_across_devices() {
        let layers = vec![100u64; 8];
        let mut remaining = vec![1000u64, 1000];
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining);
        assert_eq!(unplaced, 0);
        let on0 = targets.iter().filter(|&&t| t == 0).count();
        let on1 = targets.iter().filter(|&&t| t == 1).count();
        assert_eq!(on0, 4, "equal layers should split evenly");
        assert_eq!(on1, 4);
    }

    /// A layer too large for the roomiest device is reported as unplaced and
    /// sent to device 0, so the loader can warn instead of silently thrashing.
    #[test]
    fn placement_flags_layers_that_exceed_all_headroom() {
        let layers = vec![100u64, 5000u64, 100u64];
        let mut remaining = vec![300u64, 300];
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining);
        assert_eq!(unplaced, 1, "the 5000-byte layer fits nowhere");
        assert_eq!(targets[1], 0, "unplaced layers fall back to device 0");
    }

    /// Unequal devices: the bigger card should absorb proportionally more work
    /// instead of the 50/50 static split.
    #[test]
    fn placement_favors_the_device_with_more_headroom() {
        let layers = vec![100u64; 6];
        // Device 1 has 3x the room; it should take strictly more layers.
        let mut remaining = vec![500u64, 1500];
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining);
        assert_eq!(unplaced, 0);
        let on0 = targets.iter().filter(|&&t| t == 0).count();
        let on1 = targets.iter().filter(|&&t| t == 1).count();
        assert!(
            on1 > on0,
            "expected the roomier device to take more, got {on1} vs {on0}"
        );
    }

    /// No device may be driven negative; remaining headroom saturates at zero.
    #[test]
    fn placement_never_underflows_budget() {
        let layers = vec![400u64; 4];
        let mut remaining = vec![500u64, 10];
        let (_targets, unplaced) = assign_by_headroom(&layers, &mut remaining);
        assert!(remaining.iter().all(|&r| r <= 500), "budgets must not wrap");
        let _ = unplaced;
    }

    #[allow(clippy::field_reassign_with_default)]
    #[test]
    fn test_qwen35_shortconv_recurrent_state_advances() {
        let mut cfg = Qwen35Config::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.num_heads = 2;
        cfg.num_kv_heads = 1;
        cfg.head_dim = 8;
        cfg.num_layers = 2;
        cfg.full_attention_interval = 4; // Layer 0 is recurrent SSM
        cfg.ssm_d_conv = 4;
        cfg.ssm_d_inner = 16;

        let mut cache = Qwen35LayerCache::new(&cfg);
        let conv_initial = cache.conv_state.clone();

        // Create synthetic block with short-conv kernel
        let q_dim = cfg.num_heads * cfg.head_dim;
        let l_conv = 4;
        let conv_w = vec![0.25f32; q_dim * l_conv];

        let block = Qwen35Block {
            device: Device::Cpu,
            attn_norm: RmsNorm::new(
                cpu_tensor(
                    vec![1.0; cfg.hidden_size],
                    Shape::new(vec![cfg.hidden_size]),
                ),
                1e-6,
            ),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            attn_qkv: Some(Linear::from_tensor(
                cpu_tensor(
                    vec![0.1; (q_dim + 2 * cfg.num_kv_heads * cfg.head_dim) * cfg.hidden_size],
                    Shape::new(vec![
                        q_dim + 2 * cfg.num_kv_heads * cfg.head_dim,
                        cfg.hidden_size,
                    ]),
                ),
                None,
            )),
            attn_gate: Some(Linear::from_tensor(
                cpu_tensor(
                    vec![0.1; q_dim * cfg.hidden_size],
                    Shape::new(vec![q_dim, cfg.hidden_size]),
                ),
                None,
            )),
            ssm_out: Some(Linear::from_tensor(
                cpu_tensor(
                    vec![0.1; cfg.hidden_size * q_dim],
                    Shape::new(vec![cfg.hidden_size, q_dim]),
                ),
                None,
            )),
            ssm_conv1d: None,
            ssm_conv_vec: Some(conv_w),
            ssm_a: None,
            ssm_alpha: None,
            ssm_beta: None,
            ssm_dt_bias: None,
            ssm_norm: None,
            post_attention_norm: RmsNorm::new(
                cpu_tensor(
                    vec![1.0; cfg.hidden_size],
                    Shape::new(vec![cfg.hidden_size]),
                ),
                1e-6,
            ),
            ffn_gate: Linear::from_tensor(
                cpu_tensor(
                    vec![0.1; cfg.intermediate_size * cfg.hidden_size],
                    Shape::new(vec![cfg.intermediate_size, cfg.hidden_size]),
                ),
                None,
            ),
            ffn_up: Linear::from_tensor(
                cpu_tensor(
                    vec![0.1; cfg.intermediate_size * cfg.hidden_size],
                    Shape::new(vec![cfg.intermediate_size, cfg.hidden_size]),
                ),
                None,
            ),
            ffn_down: Linear::from_tensor(
                cpu_tensor(
                    vec![0.1; cfg.hidden_size * cfg.intermediate_size],
                    Shape::new(vec![cfg.hidden_size, cfg.intermediate_size]),
                ),
                None,
            ),
            is_full_attention: false,
            layer_idx: 0,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            wqkv_q80_fused: None,
            w_gate_up_q4k_fused: None,
        };

        let x = cpu_tensor(
            vec![1.0; 2 * cfg.hidden_size],
            Shape::new(vec![2, cfg.hidden_size]),
        );
        let out = block
            .forward(&x, &[0, 1], &mut cache)
            .expect("forward recurrent layer");
        assert_eq!(out.shape().dims(), &[2, cfg.hidden_size]);

        // State must not be all zeroes after forward pass with non-zero inputs
        assert_ne!(
            cache.conv_state, conv_initial,
            "conv_state must be updated across steps"
        );
    }
}
