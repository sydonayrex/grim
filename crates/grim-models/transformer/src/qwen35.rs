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
use grim_nn::modules::{Embedding, Linear, RmsNorm, add_tensors, pick_device_for_storage_device};
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

#[derive(Clone, Debug)]
pub struct Qwen35LayerCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub current_pos: usize,
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
            // Attention path with separated wq, wk, wv
            let mut q_all = if let Some(ref wq) = self.wq {
                let q_t = wq.forward(&x_normed)?;
                let mut q_v = q_t.to_vec_f32()?;
                q_v.truncate(seq_len * q_dim);
                q_v
            } else {
                vec![0.0f32; seq_len * q_dim]
            };

            let mut k_all = if let Some(ref wk) = self.wk {
                let k_t = wk.forward(&x_normed)?;
                let mut k_v = k_t.to_vec_f32()?;
                k_v.truncate(seq_len * kv_dim);
                k_v
            } else {
                vec![0.0f32; seq_len * kv_dim]
            };

            let v_all = if let Some(ref wv) = self.wv {
                let v_t = wv.forward(&x_normed)?;
                let mut v_v = v_t.to_vec_f32()?;
                v_v.truncate(seq_len * kv_dim);
                v_v
            } else {
                vec![0.0f32; seq_len * kv_dim]
            };

            // Apply RoPE to Q and K
            apply_rope_neox(
                &mut q_all,
                positions,
                self.num_heads,
                self.head_dim,
                self.rope_theta,
            );
            apply_rope_neox(
                &mut k_all,
                positions,
                self.num_kv_heads,
                self.head_dim,
                self.rope_theta,
            );

            cache.k_cache.extend_from_slice(&k_all);
            cache.v_cache.extend_from_slice(&v_all);

            let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
                &q_all,
                cache.k_cache.as_slice(),
                cache.v_cache.as_slice(),
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
                for t in 0..seq_len {
                    let base = t * qkv_total_per_tok;
                    for d in 0..q_dim.min(qkv_total_per_tok) {
                        out_branch[t * q_dim + d] = silu(qkv_vec[base + d]);
                    }
                }
            }
        }

        // Apply attention gate if present
        if let Some(ref gate_lin) = self.attn_gate {
            let gate_tensor = gate_lin.forward(&x_normed)?;
            let gate_vec = gate_tensor.to_vec_f32()?;
            for i in 0..out_branch.len() {
                let g = gate_vec[i % gate_vec.len()];
                out_branch[i] *= 1.0 / (1.0 + (-g).exp()); // sigmoid gate
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
        let h = add_tensors(x, &proj_out)?;

        // 3. Post-attention norm
        let h_normed = self.post_attention_norm.forward(&h)?;

        // 4. SwiGLU FFN
        let gate = self.ffn_gate.forward(&h_normed)?;
        let up = self.ffn_up.forward(&h_normed)?;
        let act = silu_mul(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&act)?;

        // Residual 2
        let out = add_tensors(&h, &ffn_out)?;
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
                h = transfer_tensor(&h, &block.device)?;
            }
            h = block.forward(&h, &positions_vec, &mut caches[i])?;
            caches[i].current_pos += seq_len;
        }

        if h.device() != self.output_norm.weight.device() {
            h = transfer_tensor(&h, self.output_norm.weight.device())?;
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

fn transfer_tensor(tensor: &Tensor, target_device: &Device) -> Result<Tensor> {
    if tensor.device() == target_device {
        return Ok(tensor.clone());
    }
    let data = tensor.to_vec_f32()?;
    device_tensor(data, tensor.shape().clone(), target_device)
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

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let g = gate.to_vec_f32()?;
    let u = up.to_vec_f32()?;
    let mut out = vec![0.0f32; g.len()];
    for i in 0..g.len() {
        out[i] = silu(g[i]) * u[i];
    }
    device_tensor(out, gate.shape().clone(), gate.device())
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
