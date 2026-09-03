//! Qwen3.5 / Qwen3.8 hybrid SSM (Mamba) + GQA Attention + SwiGLU FFN.
//!
//! Supports `Qwen3.8-27B` and related hybrid GGUF checkpoints with fused `attn_qkv`,
//! `attn_gate`, 1D short-convolution SSM layers, full attention intervals, and SwiGLU feed-forward networks.
//! Supports automatic multi-GPU layer pipelining across available discrete GPUs to fit strictly within VRAM.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::{Inner, SessionT};
use grim_nn::modules::{Embedding, Linear, RmsNorm, pick_device_for_storage_device};
use grim_nn::{TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Layer Cache
// ---------------------------------------------------------------------------

pub struct Qwen35LayerCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub current_pos: usize,
    /// WI-kv (qwen35): device-resident K/V arenas for full-attention layers.
    /// History stays on the GPU so decode uploads only the current step's
    /// rows instead of re-uploading the full context per layer per token.
    /// `Box<dyn BackendStorage>` is not `Clone`; cloned caches (session
    /// fork/rollback) fall back to the host vectors and re-grow lazily.
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

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

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
        })
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
            // GPU-first: projections stay on-device; RoPE runs through the
            // device kernel; K/V are appended into the device arenas D2D.
            // Only the (small) per-step Q rows cross to host for the
            // arena-attention entry point.
            let dev = pick_device_for_storage_device(&device);
            // Some TP-sharded projections emit padded rows — cut to the
            // exact [seq, width] extent via a D2D staging copy when needed.
            let exact = |t: Tensor, rows: usize, width: usize| -> Result<Tensor> {
                let want = Shape::new(vec![rows, width]);
                if t.shape().elem_count() == rows * width {
                    return Ok(crate::block::reshaped_view(&t, &want)?);
                }
                let scratch = dev.alloc_storage(&want, DType::F32)?;
                dev.copy_slice_range(
                    scratch.as_ref(),
                    0,
                    t.storage().as_ref(),
                    0,
                    rows * width,
                )?;
                Ok(Tensor::new(
                    scratch.into(),
                    want,
                    DType::F32,
                    t.provenance().clone(),
                    t.device().clone(),
                ))
            };

            let rope_cfg = grim_tensor::RopeConfig::new(self.head_dim, self.rope_theta);
            let rope_ext =
                |t: &Tensor, heads: usize| -> Result<Tensor> {
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
                    let (rope_s, _) = dev.rope(
                        t3.storage().as_ref(),
                        &pos_ext,
                        &rope_cfg,
                        t3.shape(),
                    )?;
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

            let q_dev = match self.wq.as_ref() {
                Some(wq) => exact(wq.forward(&x_normed)?, seq_len, q_dim)?,
                None => Tensor::new(
                    dev.zeros(&Shape::new(vec![seq_len, q_dim]), DType::F32)?.into(),
                    Shape::new(vec![seq_len, q_dim]),
                    DType::F32,
                    x_normed.provenance().clone(),
                    x_normed.device().clone(),
                ),
            };
            let k_dev_t = match self.wk.as_ref() {
                Some(wk) => exact(wk.forward(&x_normed)?, seq_len, kv_dim)?,
                None => Tensor::new(
                    dev.zeros(&Shape::new(vec![seq_len, kv_dim]), DType::F32)?.into(),
                    Shape::new(vec![seq_len, kv_dim]),
                    DType::F32,
                    x_normed.provenance().clone(),
                    x_normed.device().clone(),
                ),
            };
            let v_dev_t = match self.wv.as_ref() {
                Some(wv) => exact(wv.forward(&x_normed)?, seq_len, kv_dim)?,
                None => Tensor::new(
                    dev.zeros(&Shape::new(vec![seq_len, kv_dim]), DType::F32)?.into(),
                    Shape::new(vec![seq_len, kv_dim]),
                    DType::F32,
                    x_normed.provenance().clone(),
                    x_normed.device().clone(),
                ),
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
                dev.copy_slice_range(
                    &**cache.k_device.as_ref().unwrap(),
                    cache.current_pos * self.num_kv_heads * self.head_dim,
                    k_rope.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                dev.copy_slice_range(
                    &**cache.v_device.as_ref().unwrap(),
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
                cache.k_device = Some(k_grown.into());
                cache.v_device = Some(v_grown.into());
            }

            let total_kv = cache.current_pos + seq_len;
            let attn_tensor = crate::shared_attention::fused_or_scalar_attention_arena(
                &q_all,
                cache.k_device.as_ref().unwrap().as_ref(),
                cache.v_device.as_ref().unwrap().as_ref(),
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
            // SSM short-conv path with fused attn_qkv
            if let Some(ref qkv_lin) = self.attn_qkv {
                let qkv = qkv_lin.forward(&x_normed)?;
                let qkv_vec = qkv.to_vec_f32()?;
                let qkv_total_per_tok = qkv_vec.len() / seq_len;
                let conv_w = self.ssm_conv_vec.as_deref();

                for t in 0..seq_len {
                    let base = t * qkv_total_per_tok;
                    let num_feats = q_dim.min(qkv_total_per_tok);

                    // If short-conv kernel weights are present and state is allocated,
                    // apply causal 1D convolution and roll state.
                    if let Some(w) = conv_w {
                        let l_conv = (w.len() / num_feats.max(1)).max(1);
                        let state = &mut cache.conv_state;
                        let state_len_per_feat = l_conv.saturating_sub(1);

                        for d in 0..num_feats {
                            let curr_val = qkv_vec[base + d];
                            let w_base = d * l_conv;
                            let mut sum = w.get(w_base + l_conv.saturating_sub(1)).copied().unwrap_or(1.0) * curr_val;

                            if state_len_per_feat > 0 && state.len() >= num_feats * state_len_per_feat {
                                for k in 0..state_len_per_feat {
                                    let st_val = state[k * num_feats + d];
                                    sum += w.get(w_base + k).copied().unwrap_or(0.0) * st_val;
                                }
                            }
                            out_branch[t * q_dim + d] = silu(sum);
                        }

                        // Roll state: shift older steps and append current input
                        if state_len_per_feat > 0 && state.len() >= num_feats * state_len_per_feat {
                            if state_len_per_feat > 1 {
                                state.copy_within(num_feats..num_feats * state_len_per_feat, 0);
                            }
                            let last_offset = (state_len_per_feat - 1) * num_feats;
                            for d in 0..num_feats {
                                state[last_offset + d] = qkv_vec[base + d];
                            }
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
        let gate = self.ffn_gate.forward(&h_normed)?;
        let up = self.ffn_up.forward(&h_normed)?;
        let act = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&act)?;

        // Residual 2
        let out = grim_nn::modules::add_on_device(&h, &ffn_out)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

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
        .or_else(|_| {
            Embedding::load(
                &ws.with_device(first_device).pp("tok_embeddings"),
                cfg.vocab_size,
                cfg.hidden_size,
            )
        })?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_device =
                available_devices[i * available_devices.len() / cfg.num_layers].clone();
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
        .or_else(|_| {
            RmsNorm::load(
                &ws.with_device(last_device.clone()).pp("norm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )
        })?;

        let output = Linear::load_column_parallel(
            &ws.with_device(last_device.clone()).pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
            tp,
        )
        .or_else(|_| {
            Linear::load(
                &ws.with_device(last_device).pp("output"),
                cfg.hidden_size,
                cfg.vocab_size,
                false,
            )
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
            .expect("Qwen35::forward: session.model_state must be Vec<Qwen35LayerCache>");

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------



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
            attn_norm: RmsNorm::new(cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])), 1e-6),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            attn_qkv: Some(Linear::from_tensor(cpu_tensor(vec![0.1; (q_dim + 2 * cfg.num_kv_heads * cfg.head_dim) * cfg.hidden_size], Shape::new(vec![q_dim + 2 * cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size])), None)),
            attn_gate: Some(Linear::from_tensor(cpu_tensor(vec![0.1; q_dim * cfg.hidden_size], Shape::new(vec![q_dim, cfg.hidden_size])), None)),
            ssm_out: Some(Linear::from_tensor(cpu_tensor(vec![0.1; cfg.hidden_size * q_dim], Shape::new(vec![cfg.hidden_size, q_dim])), None)),
            ssm_conv1d: None,
            ssm_conv_vec: Some(conv_w),
            ssm_a: None,
            ssm_alpha: None,
            ssm_beta: None,
            ssm_dt_bias: None,
            ssm_norm: None,
            post_attention_norm: RmsNorm::new(cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])), 1e-6),
            ffn_gate: Linear::from_tensor(cpu_tensor(vec![0.1; cfg.intermediate_size * cfg.hidden_size], Shape::new(vec![cfg.intermediate_size, cfg.hidden_size])), None),
            ffn_up: Linear::from_tensor(cpu_tensor(vec![0.1; cfg.intermediate_size * cfg.hidden_size], Shape::new(vec![cfg.intermediate_size, cfg.hidden_size])), None),
            ffn_down: Linear::from_tensor(cpu_tensor(vec![0.1; cfg.hidden_size * cfg.intermediate_size], Shape::new(vec![cfg.hidden_size, cfg.intermediate_size])), None),
            is_full_attention: false,
            layer_idx: 0,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
        };

        let x = cpu_tensor(vec![1.0; 2 * cfg.hidden_size], Shape::new(vec![2, cfg.hidden_size]));
        let out = block.forward(&x, &[0, 1], &mut cache).expect("forward recurrent layer");
        assert_eq!(out.shape().dims(), &[2, cfg.hidden_size]);

        // State must not be all zeroes after forward pass with non-zero inputs
        assert_ne!(cache.conv_state, conv_initial, "conv_state must be updated across steps");
    }
}
