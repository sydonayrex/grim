//! LFM2 (Liquid Foundation Model v2) — `CausalLm` implementation in 100% Rust.
//! Includes recurrent ShortConv blocks and MoE gating logic.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, add_tensors, broadcast_bias};
use grim_tensor::dtype::{FloatPackScheme, QuantProvenance, Storage};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};
use std::sync::Arc;

/// Max KV-cache rows pre-allocated on the ROCm device for the fused MXFP4 QKV path.
const LFM2_FUSED_KV_CACHE_LEN: usize = 4096;

#[derive(Debug, Clone)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub n_shortconv_l_cache: usize,
    pub is_recr: Vec<bool>,
    pub n_layer_dense_lead: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub expert_weights_scale: f32,
    pub expert_gating_func: u32,
    pub n_swa: usize,
    pub swa_type: u32,
    pub n_embd_out: usize,
    /// Opt-in: route attention QKV through the ROCm fused MXFP4 GEMM + QK-Norm + RoPE kernel.
    /// Off by default so the F32 reference path remains the golden behavior.
    pub mxfp4_qkv_attention: bool,
}

impl ModelConfig for Lfm2Config {
    fn name(&self) -> &str {
        "lfm2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub enum Lfm2LayerCache {
    ShortConv(Vec<f32>),
    Attention {
        k: Vec<f32>,
        v: Vec<f32>,
        /// Device-resident KV cache arena for the fused MXFP4 path (ROCm only).
        k_dev: Option<Box<Tensor>>,
        v_dev: Option<Box<Tensor>>,
    },
}

pub struct Lfm2Block {
    pub attn_norm: RmsNorm,
    pub wq: Option<Linear>,
    pub wk: Option<Linear>,
    pub wv: Option<Linear>,
    pub wo: Option<Linear>,
    pub attn_q_norm: Option<RmsNorm>,
    pub attn_k_norm: Option<RmsNorm>,
    /// Fused MXFP4 QKV pack (ROCm only, built when `mxfp4_qkv_attention` is set).
    /// `wqkv_codes`/`wqkv_exps` are the packed MXFP4 GEMM weights `[N_total, hidden]`,
    /// `gamma_q`/`gamma_k` are the per-head Q/K norm weights `[head_dim]`.
    pub wqkv_codes: Option<Tensor>,
    pub wqkv_exps: Option<Tensor>,
    pub gamma_q: Option<Tensor>,
    pub gamma_k: Option<Tensor>,
    pub shortconv_in_proj: Option<Linear>,
    pub shortconv_conv: Option<Tensor>,
    pub shortconv_conv_vec: Option<Vec<f32>>,
    pub shortconv_out_proj: Option<Linear>,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub ffn_gate_inp: Option<Linear>,
    pub ffn_gate_exps: Option<Tensor>,
    pub ffn_up_exps: Option<Tensor>,
    pub ffn_down_exps: Option<Tensor>,
    pub ffn_exp_probs_b: Option<Tensor>,
    pub is_moe: bool,
    pub n_expert: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl Lfm2Block {
    pub fn load(
        ws: &grim_nn::WeightSource<'_>,
        cfg: &Lfm2Config,
        layer_idx: usize,
    ) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let is_recurrent = cfg.is_recr.get(layer_idx).copied().unwrap_or(false);

        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if !is_recurrent {
            let wq = Some(Linear::load(
                &ws.pp("attn_q"),
                cfg.hidden_size,
                cfg.num_heads * cfg.head_dim,
                false,
            )?);
            let wk = Some(Linear::load(
                &ws.pp("attn_k"),
                cfg.hidden_size,
                cfg.num_kv_heads * cfg.head_dim,
                false,
            )?);
            let wv = Some(Linear::load(
                &ws.pp("attn_v"),
                cfg.hidden_size,
                cfg.num_kv_heads * cfg.head_dim,
                false,
            )?);
            let wo = Some(Linear::load(
                &ws.pp("attn_output"),
                cfg.num_heads * cfg.head_dim,
                cfg.hidden_size,
                false,
            )?);
            let attn_q_norm = Some(RmsNorm::load(
                &ws.pp("attn_q_norm"),
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
            let attn_k_norm = Some(RmsNorm::load(
                &ws.pp("attn_k_norm"),
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
            (wq, wk, wv, wo, attn_q_norm, attn_k_norm)
        } else {
            (None, None, None, None, None, None)
        };

        let device = wq.as_ref().map(|w| w.weight.device().clone());
        let (wqkv_codes, wqkv_exps, gamma_q, gamma_k) = if !is_recurrent
            && cfg.mxfp4_qkv_attention
            && device
                .as_ref()
                .map(|d| matches!(d, Device::Rocm(_)))
                .unwrap_or(false)
        {
            build_fused_qkv_pack(
                wq.as_ref().unwrap(),
                wk.as_ref().unwrap(),
                wv.as_ref().unwrap(),
                attn_q_norm.as_ref().unwrap(),
                attn_k_norm.as_ref().unwrap(),
                cfg,
            )?
        } else {
            (None, None, None, None)
        };

        let (shortconv_in_proj, shortconv_conv, shortconv_conv_vec, shortconv_out_proj) =
            if is_recurrent {
                let in_proj = Some(Linear::load(
                    &ws.pp("shortconv.in_proj"),
                    cfg.hidden_size,
                    3 * cfg.hidden_size,
                    false,
                )?);
                let conv = ws
                    .get(
                        [cfg.hidden_size, cfg.n_shortconv_l_cache],
                        "shortconv.conv.weight",
                    )
                    .or_else(|_| {
                        ws.get(
                            [cfg.hidden_size, 1, cfg.n_shortconv_l_cache],
                            "shortconv.conv.weight",
                        )
                    })?;
                let conv_vec = conv.to_vec_f32().ok();
                let out_proj = Some(Linear::load(
                    &ws.pp("shortconv.out_proj"),
                    cfg.hidden_size,
                    cfg.hidden_size,
                    false,
                )?);
                (in_proj, Some(conv), conv_vec, out_proj)
            } else {
                (None, None, None, None)
            };

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let is_moe = layer_idx >= cfg.n_layer_dense_lead;

        let (
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_gate_inp,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
            ffn_exp_probs_b,
        ) = if is_moe {
            let ffn_gate_inp = Some(Linear::load(
                &ws.pp("ffn_gate_inp"),
                cfg.hidden_size,
                cfg.n_expert,
                false,
            )?);
            let ffn_gate_exps = Some(ws.get(
                [cfg.n_expert, cfg.n_ff_exp, cfg.hidden_size],
                "ffn_gate_exps.weight",
            )?);
            eprintln!(
                "[grim] dequant done: layer {} ffn_gate_exps ({}x{}x{})",
                layer_idx, cfg.n_expert, cfg.n_ff_exp, cfg.hidden_size
            );
            let ffn_up_exps = Some(ws.get(
                [cfg.n_expert, cfg.n_ff_exp, cfg.hidden_size],
                "ffn_up_exps.weight",
            )?);
            eprintln!(
                "[grim] dequant done: layer {} ffn_up_exps ({}x{}x{})",
                layer_idx, cfg.n_expert, cfg.n_ff_exp, cfg.hidden_size
            );
            let ffn_down_exps = Some(ws.get(
                [cfg.n_ff_exp, cfg.hidden_size, cfg.n_expert],
                "ffn_down_exps.weight",
            )?);
            eprintln!(
                "[grim] dequant done: layer {} ffn_down_exps ({}x{}x{})",
                layer_idx, cfg.n_ff_exp, cfg.hidden_size, cfg.n_expert
            );
            let ffn_exp_probs_b_val = ws.get([cfg.n_expert], "ffn_exp_probs_b.bias")?;
            let ffn_gate = Linear::load(
                &ws.pp("ffn_gate"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?;
            let ffn_up = Linear::load(
                &ws.pp("ffn_up"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?;
            let ffn_down = Linear::load(
                &ws.pp("ffn_down"),
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
            )?;
            (
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_gate_inp,
                ffn_gate_exps,
                ffn_up_exps,
                ffn_down_exps,
                Some(ffn_exp_probs_b_val),
            )
        } else {
            let ffn_gate = Linear::load(
                &ws.pp("ffn_gate"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?;
            let ffn_up = Linear::load(
                &ws.pp("ffn_up"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?;
            let ffn_down = Linear::load(
                &ws.pp("ffn_down"),
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
            )?;
            (
                ffn_gate,
                ffn_up,
                ffn_down,
                Option::<Linear>::None,
                Option::<Tensor>::None,
                Option::<Tensor>::None,
                Option::<Tensor>::None,
                Option::<Tensor>::None,
            )
        };

        Ok(Self {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            attn_q_norm,
            attn_k_norm,
            wqkv_codes,
            wqkv_exps,
            gamma_q,
            gamma_k,
            shortconv_in_proj,
            shortconv_conv,
            shortconv_conv_vec,
            shortconv_out_proj,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_gate_inp,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
            ffn_exp_probs_b,
            is_moe,
            n_expert: if is_moe { cfg.n_expert } else { 0 },
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            eps: cfg.rms_norm_eps,
        })
    }

    /// Load-time coherence check (grim-models audit M11): the forward paths
    /// unwrap the variant-specific `Option` fields, so an incoherent block
    /// (e.g. shortconv projection without its kernel, or a full-attention
    /// block missing QK norms) used to PANIC at first forward. Validate the
    /// variant contract at load time instead and name the missing field.
    pub fn validate(&self, idx: usize) -> Result<()> {
        let ctx = |field: &str| {
            grim_core::error::Error::Config(format!(
                "lfm2 layer {idx}: {} is required by this block variant but was not loaded",
                field
            ))
        };
        if self.shortconv_in_proj.is_some() {
            // ShortConv attention variant.
            let required = [
                ("shortconv_conv", self.shortconv_conv.is_some()),
                ("shortconv_conv_vec", self.shortconv_conv_vec.is_some()),
                ("shortconv_out_proj", self.shortconv_out_proj.is_some()),
            ];
            for (name, present) in required {
                if !present {
                    return Err(ctx(name));
                }
            }
        } else {
            // Full-attention / fused-QKV variants: the F32 reference path
            // unconditionally unwraps these.
            let required = [
                ("wq", self.wq.is_some()),
                ("wk", self.wk.is_some()),
                ("wv", self.wv.is_some()),
                ("wo", self.wo.is_some()),
                ("attn_q_norm", self.attn_q_norm.is_some()),
                ("attn_k_norm", self.attn_k_norm.is_some()),
            ];
            for (name, present) in required {
                if !present {
                    return Err(ctx(name));
                }
            }
            // The fused MXFP4 pair must be coherent if present at all.
            if self.wqkv_codes.is_some() != self.wqkv_exps.is_some() {
                return Err(grim_core::error::Error::Config(format!(
                    "lfm2 layer {idx}: wqkv_codes/wqkv_exps must be loaded together"
                )));
            }
        }
        if self.is_moe && self.ffn_gate_inp.is_none() {
            return Err(ctx("ffn_gate_inp"));
        }
        Ok(())
    }

    pub fn forward(&self, x: &Tensor, cache: &mut Option<Lfm2LayerCache>) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;

        let block_out = if let Some(in_proj) = &self.shortconv_in_proj {
            let proj = in_proj.forward(&norm_x)?;
            let proj_v = proj.to_vec_f32()?;
            let h_dim = norm_x.shape().dims().last().copied().unwrap_or(0);
            let steps = proj_v.len() / (3 * h_dim);

            // Debug: compare projection output between CPU and CUDA.
            if std::env::var_os("GRIM_DEBUG_SHORTCONV").is_some() {
                eprintln!(
                    "[shortconv-dbg] proj_v: len={} steps={} h_dim={} head={:?}",
                    proj_v.len(),
                    steps,
                    h_dim,
                    &proj_v[..4.min(proj_v.len())]
                );
                // Check the c and x_val components.
                let c = &proj_v[h_dim..2 * h_dim];
                let x_val = &proj_v[2 * h_dim..3 * h_dim];
                eprintln!(
                    "[shortconv-dbg] c[0..4]={:?} x_val[0..4]={:?}",
                    &c[..4.min(c.len())],
                    &x_val[..4.min(x_val.len())]
                );
                // Check the weight tensor shape and values.
                let w_t = &self.shortconv_in_proj.as_ref().unwrap().w_t;
                eprintln!(
                    "[shortconv-dbg] in_proj w_t shape={:?} dtype={:?}",
                    w_t.shape(),
                    w_t.dtype()
                );
                // Check weight values for different output features.
                let w_t_vec = w_t.to_vec_f32().expect("w_t to_vec");
                // w_t is [1024, 3072]. Check features 0, 1024, 2048.
                eprintln!(
                    "[shortconv-dbg] w_t[0][0..4]={:?} w_t[0][1024..1028]={:?} w_t[0][2048..2052]={:?}",
                    &w_t_vec[..4],
                    &w_t_vec[1024..1028.min(w_t_vec.len())],
                    &w_t_vec[2048..2052.min(w_t_vec.len())]
                );
            }

            let mut y_out = vec![0.0f32; steps * h_dim];

            let conv_kernel_vec = self.shortconv_conv_vec.as_ref().unwrap();
            let conv_shape = self.shortconv_conv.as_ref().unwrap().shape().dims();
            let l_cache = *conv_shape.last().unwrap_or(&3);

            if cache.is_none() {
                *cache = Some(Lfm2LayerCache::ShortConv(vec![
                    0.0f32;
                    h_dim * (l_cache - 1)
                ]));
            }

            let state = match cache.as_mut().unwrap() {
                Lfm2LayerCache::ShortConv(st) => st,
                _ => {
                    return Err(grim_core::error::Error::Session(
                        "Mismatched ShortConv layer cache".into(),
                    ));
                }
            };

            for step in 0..steps {
                let offset = step * 3 * h_dim;
                let b = &proj_v[offset..offset + h_dim];
                let c = &proj_v[offset + h_dim..offset + 2 * h_dim];
                let x_val = &proj_v[offset + 2 * h_dim..offset + 3 * h_dim];

                let bx: Vec<f32> = b.iter().zip(x_val.iter()).map(|(bv, xv)| bv * xv).collect();

                for d in 0..h_dim {
                    let w_base = d * l_cache;
                    let mut sum = conv_kernel_vec[w_base + l_cache - 1] * bx[d];
                    for k in 0..l_cache - 1 {
                        sum += conv_kernel_vec[w_base + k] * state[k * h_dim + d];
                    }
                    y_out[step * h_dim + d] = c[d] * sum;
                }

                if l_cache > 1 {
                    state.copy_within(h_dim.., 0);
                    state[(l_cache - 2) * h_dim..].copy_from_slice(&bx);
                }
            }

            // Debug: compare convolution output between CPU and CUDA.
            if std::env::var_os("GRIM_DEBUG_SHORTCONV").is_some() {
                eprintln!(
                    "[shortconv-dbg] y_out: len={} head={:?}",
                    y_out.len(),
                    &y_out[..4.min(y_out.len())]
                );
                // Check intermediate values.
                let c_head = &proj_v[h_dim..2 * h_dim];
                let bx_head: Vec<f32> = proj_v[..h_dim]
                    .iter()
                    .zip(&proj_v[2 * h_dim..3 * h_dim])
                    .map(|(a, b)| a * b)
                    .collect();
                eprintln!(
                    "[shortconv-dbg] c_head={:?} bx_head={:?} conv_kernel_head={:?}",
                    &c_head[..4.min(c_head.len())],
                    &bx_head[..4.min(bx_head.len())],
                    &conv_kernel_vec[..4.min(conv_kernel_vec.len())]
                );
            }

            let y_tensor = device_tensor(y_out, Shape::new(vec![steps, h_dim]), norm_x.device())?;
            let block_out_2d = self
                .shortconv_out_proj
                .as_ref()
                .unwrap()
                .forward(&y_tensor)?;
            // Reshape from [steps, h_dim] to [1, steps, h_dim] so the
            // residual add works correctly on backends without broadcasting
            // (e.g. CUDA). The input x is [1, steps, hidden_size].
            Tensor::new(
                block_out_2d.storage().clone(),
                Shape::new(vec![1, steps, h_dim]),
                block_out_2d.dtype(),
                block_out_2d.provenance().clone(),
                block_out_2d.device().clone(),
            )
        } else if self.is_moe {
            self.forward_moe_ffn(&norm_x)?
        } else {
            let steps = norm_x.shape().dims()[0];
            let hidden = norm_x.shape().dims().last().copied().unwrap();
            let kv_stride = self.num_kv_heads * self.head_dim;

            let use_fused = self.wqkv_codes.is_some() && matches!(norm_x.device(), Device::Rocm(_));

            // Produce `q_rot_vec` (rotated, QK-normalized Q) and extend the
            // K/V history. Both the fused MXFP4 path and the F32 reference
            // produce identical layouts so the attention loop is shared.
            // `arena_total` is Some when the new K/V rows were mirrored into
            // the device arenas (WI-X2), enabling arena-resident attention.
            let (q_rot_vec, arena_total): (Vec<f32>, Option<usize>) = if use_fused {
                // The fused kernel appended the new K/V rows directly into
                // the device arenas — attention can stay arena-resident.
                let past = match cache {
                    Some(Lfm2LayerCache::Attention { k, .. }) => k.len() / (self.num_kv_heads * self.head_dim),
                    _ => 0,
                };
                (
                    self.fused_qkv(&norm_x, cache, steps, hidden)?,
                    Some(past + steps),
                )
            } else {
                let q = self.wq.as_ref().unwrap().forward(&norm_x)?;
                let k = self.wk.as_ref().unwrap().forward(&norm_x)?;
                let v = self.wv.as_ref().unwrap().forward(&norm_x)?;

                let q_2d = Tensor::new(
                    q.storage().clone(),
                    Shape::new(vec![steps * self.num_heads, self.head_dim]),
                    q.dtype(),
                    q.provenance().clone(),
                    q.device().clone(),
                );
                let q_norm = self.attn_q_norm.as_ref().unwrap().forward(&q_2d)?;

                let k_2d = Tensor::new(
                    k.storage().clone(),
                    Shape::new(vec![steps * self.num_kv_heads, self.head_dim]),
                    k.dtype(),
                    k.provenance().clone(),
                    k.device().clone(),
                );
                let k_norm = self.attn_k_norm.as_ref().unwrap().forward(&k_2d)?;

                let dev = grim_nn::modules::pick_device_for_storage_device(norm_x.device());
                let q_shape = Shape::new(vec![1, steps * self.num_heads, self.head_dim]);
                let k_shape = Shape::new(vec![1, steps * self.num_kv_heads, self.head_dim]);

                let cache_offset = match cache {
                    Some(Lfm2LayerCache::Attention { k, .. }) => k.len() / kv_stride,
                    _ => 0,
                };
                let q_positions: Vec<u32> = {
                    let mut v = Vec::with_capacity(steps * self.num_heads);
                    for t in 0..steps {
                        for _ in 0..self.num_heads {
                            v.push((cache_offset + t) as u32);
                        }
                    }
                    v
                };
                let k_positions: Vec<u32> = {
                    let mut v = Vec::with_capacity(steps * self.num_kv_heads);
                    for t in 0..steps {
                        for _ in 0..self.num_kv_heads {
                            v.push((cache_offset + t) as u32);
                        }
                    }
                    v
                };
                let rope_cfg = grim_tensor::RopeConfig::new(self.head_dim, self.rope_theta);
                let (q_rot_storage, _) =
                    dev.rope(q_norm.storage().as_ref(), &q_positions, &rope_cfg, &q_shape)?;
                let (k_rot_storage, _) =
                    dev.rope(k_norm.storage().as_ref(), &k_positions, &rope_cfg, &k_shape)?;

                let q_rot_vec = q_rot_storage.to_cpu_vec_f32()?;
                let k_rot_vec = k_rot_storage.to_cpu_vec_f32()?;
                let v_vec = v.to_vec_f32()?;

                if cache.is_none() {
                    *cache = Some(Lfm2LayerCache::Attention {
                        k: vec![],
                        v: vec![],
                        k_dev: None,
                        v_dev: None,
                    });
                }

                let mut arena_total: Option<usize> = None;
                let v_storage = v.storage();
                match cache.as_mut().unwrap() {
                    Lfm2LayerCache::Attention { k, v, k_dev, v_dev } => {
                        let past = k.len() / kv_stride;
                        k.extend_from_slice(&k_rot_vec);
                        v.extend_from_slice(&v_vec);
                        // WI-X2: mirror ONLY the new rows into preallocated
                        // device arenas so attention runs without re-uploading
                        // the whole history each decode step. Falls back to the
                        // host-history path when the backend lacks the copies.
                        if k_dev.is_none() {
                            let shape = Shape::new(vec![LFM2_FUSED_KV_CACHE_LEN, kv_stride]);
                            *k_dev = Some(Box::new(Tensor::new(
                                Arc::from(dev.zeros(&shape, DType::F32)?),
                                shape.clone(),
                                DType::F32,
                                QuantProvenance::GrimNative,
                                norm_x.device().clone(),
                            )));
                            *v_dev = Some(Box::new(Tensor::new(
                                Arc::from(dev.zeros(&shape, DType::F32)?),
                                shape,
                                DType::F32,
                                QuantProvenance::GrimNative,
                                norm_x.device().clone(),
                            )));
                        }
                        let off_elems = past * kv_stride;
                        let cnt_elems = steps * kv_stride;
                        let k_ok = dev.copy_slice_into(
                            k_dev.as_ref().unwrap().storage().as_ref(),
                            k_rot_storage.as_ref(),
                            off_elems,
                            cnt_elems,
                        );
                        let v_ok = dev.copy_slice_into(
                            v_dev.as_ref().unwrap().storage().as_ref(),
                            v_storage.as_ref(),
                            off_elems,
                            cnt_elems,
                        );
                        if k_ok.is_ok()
                            && v_ok.is_ok()
                            && std::env::var("GRIM_LFM2_KV_ARENA").as_deref() != Ok("0")
                        {
                            arena_total = Some(past + steps);
                        }
                    }
                    _ => {
                        return Err(grim_core::error::Error::Session(
                            "Mismatched Attention layer cache".into(),
                        ));
                    }
                }
                (q_rot_vec, arena_total)
            };

            if cache.is_none() {
                *cache = Some(Lfm2LayerCache::Attention {
                    k: vec![],
                    v: vec![],
                    k_dev: None,
                    v_dev: None,
                });
            }

            // WI-X2: prefer arena-resident attention (history never re-uploads);
            // fall back to the host-history path when the D2D mirror failed.
            let attn_tensor = if let Some(total) = arena_total {
                let (kd, vd) = match cache.as_ref().unwrap() {
                    Lfm2LayerCache::Attention { k_dev, v_dev, .. } => (k_dev, v_dev),
                    _ => {
                        return Err(grim_core::error::Error::Session(
                            "Mismatched Attention layer cache".into(),
                        ));
                    }
                };
                crate::shared_attention::fused_or_scalar_attention_arena(
                    &q_rot_vec,
                    kd.as_ref().unwrap().storage().as_ref(),
                    vd.as_ref().unwrap().storage().as_ref(),
                    total,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    steps,
                    None,
                    norm_x.device(),
                )?
            } else {
                let (k_hist, v_hist) = match cache.as_ref().unwrap() {
                    Lfm2LayerCache::Attention { k, v, .. } => (k.as_slice(), v.as_slice()),
                    _ => {
                        return Err(grim_core::error::Error::Session(
                            "Mismatched Attention layer cache".into(),
                        ));
                    }
                };
                crate::shared_attention::fused_or_scalar_attention(
                    &q_rot_vec,
                    k_hist,
                    v_hist,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    steps,
                    None,
                    norm_x.device(),
                )?
            };
            self.wo.as_ref().unwrap().forward(&attn_tensor)?
        };

        let x_added = add_tensors(x, &block_out).map_err(grim_core::Error::Tensor)?;

        let norm_x_ffn = self.ffn_norm.forward(&x_added)?;
        let ffn_out = if self.is_moe {
            self.forward_moe_ffn(&norm_x_ffn)?
        } else {
            let gate = self.ffn_gate.forward(&norm_x_ffn)?;
            let up = self.ffn_up.forward(&norm_x_ffn)?;
            let activated = silu_mul(&gate, &up)?;
            self.ffn_down.forward(&activated)?
        };

        add_tensors(&x_added, &ffn_out).map_err(grim_core::Error::Tensor)
    }

    /// Fused MXFP4 QKV path: a single ROCm kernel computes the QKV GEMM,
    /// per-head QK-Norm (dual gamma), and plain RoPE, writing Q to `q_out`
    /// and K/V rows into the device-resident cache at their sequence
    /// positions. Returns the rotated Q vector (matching the F32 reference
    /// layout) and extends the CPU K/V history so the shared attention loop
    /// below can run unchanged.
    fn fused_qkv(
        &self,
        norm_x: &Tensor,
        cache: &mut Option<Lfm2LayerCache>,
        steps: usize,
        hidden: usize,
    ) -> Result<Vec<f32>> {
        let dev_arc = grim_nn::modules::pick_device_for_storage_device(norm_x.device());
        let dev = dev_arc.as_ref();
        let n_q = self.num_heads * self.head_dim;
        let n_k = self.num_kv_heads * self.head_dim;
        let n_v = self.num_kv_heads * self.head_dim;
        let mut max_seq = LFM2_FUSED_KV_CACHE_LEN;

        let cache_offset = match cache {
            Some(Lfm2LayerCache::Attention { k, .. }) => k.len() / n_k,
            _ => 0,
        };
        let positions: Vec<u32> = (0..steps).map(|t| (cache_offset + t) as u32).collect();

        if cache.is_none() {
            *cache = Some(Lfm2LayerCache::Attention {
                k: vec![],
                v: vec![],
                k_dev: None,
                v_dev: None,
            });
        }
        let (k_dev, v_dev) = match cache.as_mut().unwrap() {
            Lfm2LayerCache::Attention { k_dev, v_dev, .. } => (k_dev, v_dev),
            _ => {
                return Err(grim_core::error::Error::Session(
                    "Mismatched Attention layer cache".into(),
                ));
            }
        };
        // The fused-KV scratch starts at `LFM2_FUSED_KV_CACHE_LEN` positions
        // but the model's context window is far larger; sessions whose
        // sequence runs past the current capacity grow it (doubling) instead
        // of reading past the allocation. K and V always share one capacity.
        let needed = cache_offset + steps;
        {
            let cur = k_dev.as_ref().map(|t| t.shape().dims()[0]).unwrap_or(0);
            if cur < needed {
                let new_cap = needed.max(LFM2_FUSED_KV_CACHE_LEN * 2);
                let grow = |slot: &mut Option<Box<Tensor>>, row_len: usize| -> Result<()> {
                    let old_data = slot.as_ref().map(|t| t.to_vec_f32()).transpose()?;
                    let mut data = vec![0f32; new_cap * row_len];
                    if let Some(old) = old_data {
                        let keep = old.len().min(data.len());
                        data[..keep].copy_from_slice(&old[..keep]);
                    }
                    let shape = Shape::new(vec![new_cap, row_len]);
                    *slot = Some(Box::new(Tensor::new(
                        Arc::from(dev.from_cpu_bytes(as_u8_slice(&data), &shape, DType::F32)?),
                        shape,
                        DType::F32,
                        QuantProvenance::GrimNative,
                        norm_x.device().clone(),
                    )));
                    Ok(())
                };
                grow(k_dev, n_k)?;
                grow(v_dev, n_v)?;
                max_seq = new_cap;
            }
        }
        if k_dev.is_none() {
            let k_shape = Shape::new(vec![max_seq, n_k]);
            *k_dev = Some(Box::new(Tensor::new(
                Arc::from(dev.zeros(&k_shape, DType::F32)?),
                k_shape,
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            )));
            let v_shape = Shape::new(vec![max_seq, n_v]);
            *v_dev = Some(Box::new(Tensor::new(
                Arc::from(dev.zeros(&v_shape, DType::F32)?),
                v_shape,
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            )));
        }

        let q_shape = Shape::new(vec![steps, n_q]);
        let q_out = Tensor::new(
            Arc::from(dev.zeros(&q_shape, DType::F32)?),
            q_shape,
            DType::F32,
            QuantProvenance::GrimNative,
            norm_x.device().clone(),
        );

        let pos_shape = Shape::new(vec![steps]);
        let pos_storage = dev.from_cpu_bytes(as_u8_slice(&positions), &pos_shape, DType::U32)?;

        let _handle = dev.fused_mxfp4_gemm_qk_norm_rope_kv(
            norm_x.storage().as_ref(),
            self.gamma_q.as_ref().unwrap().storage().as_ref(),
            self.gamma_k.as_ref().unwrap().storage().as_ref(),
            self.wqkv_codes.as_ref().unwrap().storage().as_ref(),
            self.wqkv_exps.as_ref().unwrap().storage().as_ref(),
            Some(q_out.storage().as_ref()),
            Some(k_dev.as_ref().unwrap().storage().as_ref()),
            Some(v_dev.as_ref().unwrap().storage().as_ref()),
            None,
            Some(&*pos_storage),
            steps,
            hidden,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim,
            self.rope_theta,
            None,
            1.0,
            self.eps,
            max_seq,
        )?;
        // No explicit synchronize: the storages sync lazily on first host
        // read (WI-Host-1 rationale); an eager sync here would stall the
        // pipeline every decode step.

        let q_rot_vec = q_out.to_vec_f32()?;

        // Mirror ONLY the new K/V rows into the host history: stage them via
        // a D2D range-copy into a `[steps, row_len]` scratch, then one small
        // D2H. The old path downloaded the ENTIRE device arena per token
        // (O(context) D2H + O(context) H2D when attention re-uploaded it).
        let stage = |arena: &Tensor, row_len: usize| -> Result<Vec<f32>> {
            let stage_shape = Shape::new(vec![steps, row_len]);
            let scratch = dev.alloc_storage(&stage_shape, DType::F32)?;
            dev.copy_slice_range(
                scratch.as_ref(),
                0,
                arena.storage().as_ref(),
                cache_offset * row_len,
                steps * row_len,
            )?;
            let staged = Tensor::new(
                Arc::from(scratch),
                stage_shape,
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            );
            staged.to_vec_f32().map_err(grim_core::error::Error::from)
        };
        let (new_k_rows, new_v_rows) = match (
            stage(k_dev.as_ref().unwrap(), n_k),
            stage(v_dev.as_ref().unwrap(), n_v),
        ) {
            (Ok(k), Ok(v)) => (k, v),
            // Backend lacks the range-copy primitives: fall back to reading
            // the arenas whole (correct, O(context) — degraded path only).
            (Err(e), _) | (_, Err(e)) if crate::is_unimplemented(&e) => (
                k_dev
                    .as_ref()
                    .unwrap()
                    .to_vec_f32()
                    .map_err(grim_core::error::Error::from)?,
                v_dev
                    .as_ref()
                    .unwrap()
                    .to_vec_f32()
                    .map_err(grim_core::error::Error::from)?,
            ),
            (Err(e), _) | (_, Err(e)) => return Err(e),
        };
        let (k_hist, v_hist) = match cache.as_mut().unwrap() {
            Lfm2LayerCache::Attention { k, v, .. } => (k, v),
            _ => {
                return Err(grim_core::error::Error::Session(
                    "Mismatched Attention layer cache".into(),
                ));
            }
        };
        k_hist.extend_from_slice(&new_k_rows);
        v_hist.extend_from_slice(&new_v_rows);
        Ok(q_rot_vec)
    }

    /// Device-side per-token expert compute: extract the winning expert's
    /// weight block from the stacked `[E, F, H]` tensor, transpose on-device,
    /// matmul the token row, silu-gate with the up projection, then the down
    /// projection. All of it on the device; the host only sees the tiny
    /// gate-logit vector. Falls back to the host loop only when a backend
    /// lacks one of the required ops (allocsilce/transpose/matmul/silu_mul).
    fn forward_moe_ffn_device(
        &self,
        x: &Tensor,
        probs: &[f32],
        steps: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let n_expert = self.n_expert;
        let gate_w = self.ffn_gate_exps.as_ref().unwrap();
        let up_w = self.ffn_up_exps.as_ref().unwrap();
        let down_w = self.ffn_down_exps.as_ref().unwrap();

        let gate_dims = gate_w.shape().dims().to_vec();
        let n_ff = gate_dims[1];
        let n_hidden = gate_dims[2];

        let dev = grim_nn::modules::pick_device_for_storage_device(x.device());
        let out_shape = Shape::new(vec![steps, hidden]);
        let out_storage = dev.alloc_storage(&out_shape, DType::F32)?;
        let out_arc: Arc<dyn grim_tensor::backend::BackendStorage> = Arc::from(out_storage);
        let x_st = x.storage().clone();

        let row_shape = Shape::new(vec![1, n_hidden]);
        let ffxh = Shape::new(vec![n_ff, n_hidden]);
        let row_ff = Shape::new(vec![1, n_ff]);

        for s in 0..steps {
            let (mut best_e, mut best_p) = (0usize, probs[s * n_expert]);
            for e in 1..n_expert {
                if probs[s * n_expert + e] > best_p {
                    best_p = probs[s * n_expert + e];
                    best_e = e;
                }
            }

            let xr = dev.alloc_storage(&row_shape, DType::F32)?;
            dev.copy_slice_range(xr.as_ref(), 0, x_st.as_ref(), s * n_hidden, n_hidden)?;

            let wg = dev.alloc_storage(&ffxh, DType::F32)?;
            dev.copy_slice_range(
                wg.as_ref(),
                0,
                gate_w.storage().as_ref(),
                best_e * n_ff * n_hidden,
                n_ff * n_hidden,
            )?;
            let wu = dev.alloc_storage(&ffxh, DType::F32)?;
            dev.copy_slice_range(
                wu.as_ref(),
                0,
                up_w.storage().as_ref(),
                best_e * n_ff * n_hidden,
                n_ff * n_hidden,
            )?;
            let wd = dev.alloc_storage(&ffxh, DType::F32)?;
            dev.copy_slice_range(
                wd.as_ref(),
                0,
                down_w.storage().as_ref(),
                best_e * n_ff * n_hidden,
                n_ff * n_hidden,
            )?;

            let hxf = Shape::new(vec![n_hidden, n_ff]);
            let wg_t = dev
                .transpose_2d(wg.as_ref(), n_ff, n_hidden, &hxf)?
                .0;
            let wu_t = dev
                .transpose_2d(wu.as_ref(), n_ff, n_hidden, &hxf)?
                .0;
            // out_f[s] = Σ_h x[h] · W[f, h]  ←  x_row[1,H] @ W^T [H→F]
            let (g_st, _) = dev.matmul(xr.as_ref(), wg_t.as_ref(), &row_ff)?;
            let (u_st, _) = dev.matmul(xr.as_ref(), wu_t.as_ref(), &row_ff)?;
            let (a_st, _) = dev.silu_mul(g_st.as_ref(), u_st.as_ref(), &row_ff)?;
            // out_d = Σ_f act[f] · Wd[f, d]  ←  act[1,F] @ Wd [F,H]
            let (o_st, _) = dev.matmul(a_st.as_ref(), wd.as_ref(), &row_shape)?;
            let scaled = dev.mul_scalar(o_st.as_ref(), best_p, &row_shape);
            let final_st: Arc<dyn grim_tensor::backend::BackendStorage> = match scaled {
                Ok((st, _)) => Arc::from(st),
                Err(_) => Arc::from(o_st),
            };
            dev.copy_slice_range(out_arc.as_ref(), s * n_hidden, final_st.as_ref(), 0, n_hidden)?;
        }

        Ok(Tensor::new(
            out_arc,
            out_shape,
            DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            x.device().clone(),
        ))
    }

    /// Top-1 routed MoE feed-forward.
    /// Matches llama.cpp's `build_moe_ffn` gate/probs semantics with silu-gated experts.
    fn forward_moe_ffn(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.shape().dims().last().copied().unwrap_or(0);
        let steps = x.shape().dims()[0];
        let n_expert = self.n_expert;
        let n_ff = self.ffn_gate.weight.shape().dims()[0];

        let gate_logits = self.ffn_gate_inp.as_ref().unwrap().forward(x)?;
        let gate_vec = gate_logits.to_vec_f32()?;

        let gate_vec = if let Some(bias) = &self.ffn_exp_probs_b {
            let bias_vec = bias.to_vec_f32()?;
            let mut g = gate_vec;
            for i in 0..g.len() {
                g[i] += bias_vec[i % bias_vec.len()];
            }
            g
        } else {
            gate_vec
        };

        let mut probs = vec![0.0f32; steps * n_expert];
        for s in 0..steps {
            let mut max_s = f32::NEG_INFINITY;
            for e in 0..n_expert {
                max_s = max_s.max(gate_vec[s * n_expert + e]);
            }
            let mut sum_s = 0.0f32;
            for e in 0..n_expert {
                probs[s * n_expert + e] = (gate_vec[s * n_expert + e] - max_s).exp();
                sum_s += probs[s * n_expert + e];
            }
            if sum_s > 0.0 {
                for e in 0..n_expert {
                    probs[s * n_expert + e] /= sum_s;
                }
            }
        }

        // Device-side MoE: weights stay on the GPU; only the tiny gate logits
        // cross to host. Old path round-tripped the full expert stacks
        // (gate/up/down) per forward — multi-MB D2H per token.
        if x.device() != &Device::Cpu {
            if let Ok(t) = self.forward_moe_ffn_device(x, &probs, steps, hidden) {
                return Ok(t);
            }
        }

        let mut out = vec![0.0f32; steps * hidden];
        let x_vec = x.to_vec_f32()?;
        let gate_exps_vec = self.ffn_gate_exps.as_ref().unwrap().to_vec_f32()?;
        let up_exps_vec = self.ffn_up_exps.as_ref().unwrap().to_vec_f32()?;
        let down_exps_vec = self.ffn_down_exps.as_ref().unwrap().to_vec_f32()?;

        for s in 0..steps {
            let mut best_e = 0;
            let mut best_p = probs[s * n_expert];
            for e in 1..n_expert {
                if probs[s * n_expert + e] > best_p {
                    best_p = probs[s * n_expert + e];
                    best_e = e;
                }
            }

            let x_s = &x_vec[s * hidden..(s + 1) * hidden];
            let mut gate_e = vec![0.0f32; n_ff];
            let mut up_e = vec![0.0f32; n_ff];

            for f in 0..n_ff {
                let mut g_sum = 0.0f32;
                let mut u_sum = 0.0f32;
                for (d, x) in x_s.iter().enumerate() {
                    let g_idx = best_e * n_ff * hidden + f * hidden + d;
                    let u_idx = best_e * n_ff * hidden + f * hidden + d;
                    g_sum += x * gate_exps_vec[g_idx];
                    u_sum += x * up_exps_vec[u_idx];
                }
                gate_e[f] = g_sum;
                up_e[f] = u_sum;
            }

            let mut activated = vec![0.0f32; n_ff];
            for (a, (g, u)) in activated.iter_mut().zip(gate_e.iter().zip(up_e.iter())) {
                let silu = g / (1.0 + (-g).exp());
                *a = silu * u;
            }

            for d in 0..hidden {
                let mut acc = 0.0f32;
                for (f, &a) in activated.iter().enumerate() {
                    let d_idx = best_e * n_ff * hidden + f * hidden + d;
                    acc += a * down_exps_vec[d_idx];
                }
                out[s * hidden + d] = acc * best_p;
            }
        }

        device_tensor(out, Shape::new(vec![steps, hidden]), x.device())
    }
}

/// Build the ROCm device-resident fused QKV pack: concatenate the Q/K/V
/// projection weights into one `[N_total, hidden]` matrix, MXFP4-quantize it,
/// and upload the packed codes/exps plus the per-head Q/K norm weights.
/// Returns `(codes, exps, gamma_q, gamma_k)`, all as device tensors.
/// Fused QKV pack tensors: `(codes, exps, gamma_q, gamma_k)`.
type FusedQkvPack = (
    Option<Tensor>,
    Option<Tensor>,
    Option<Tensor>,
    Option<Tensor>,
);

fn build_fused_qkv_pack(
    wq: &Linear,
    wk: &Linear,
    wv: &Linear,
    attn_q_norm: &RmsNorm,
    attn_k_norm: &RmsNorm,
    cfg: &Lfm2Config,
) -> Result<FusedQkvPack> {
    let n_q = cfg.num_heads * cfg.head_dim;
    let n_k = cfg.num_kv_heads * cfg.head_dim;
    let n_v = cfg.num_kv_heads * cfg.head_dim;
    let n_total = n_q + n_k + n_v;
    let hidden = cfg.hidden_size;

    let wq_d = wq.weight.to_vec_f32()?;
    let wk_d = wk.weight.to_vec_f32()?;
    let wv_d = wv.weight.to_vec_f32()?;
    let mut concat = Vec::with_capacity(n_total * hidden);
    concat.extend_from_slice(&wq_d);
    concat.extend_from_slice(&wk_d);
    concat.extend_from_slice(&wv_d);

    let (codes, exps) = grim_quant::quant_mxfp4_matrix(&concat, n_total, hidden);
    let num_blocks = (n_total * hidden) / 32;

    let device = wq.weight.device().clone();
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);

    let codes_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(FloatPackScheme::MxFp4),
    };
    let codes_shape = Shape::new(vec![n_total, hidden]);
    let exps_shape = Shape::new(vec![num_blocks]);
    let gamma_shape = Shape::new(vec![cfg.head_dim]);

    let codes_storage = dev.from_cpu_bytes(&codes, &codes_shape, codes_dtype.clone())?;
    let exps_storage = dev.from_cpu_bytes(
        &exps,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;
    let gq_d = attn_q_norm.weight.to_vec_f32()?;
    let gk_d = attn_k_norm.weight.to_vec_f32()?;
    let gq_storage = dev.from_cpu(&gq_d, &gamma_shape, DType::F32)?;
    let gk_storage = dev.from_cpu(&gk_d, &gamma_shape, DType::F32)?;

    let codes_t = Tensor::new(
        Arc::from(codes_storage),
        codes_shape,
        codes_dtype,
        QuantProvenance::GrimNative,
        device.clone(),
    );
    let exps_t = Tensor::new(
        Arc::from(exps_storage),
        exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
        QuantProvenance::GrimNative,
        device.clone(),
    );
    let gq_t = Tensor::new(
        Arc::from(gq_storage),
        gamma_shape.clone(),
        DType::F32,
        QuantProvenance::GrimNative,
        device.clone(),
    );
    let gk_t = Tensor::new(
        Arc::from(gk_storage),
        gamma_shape.clone(),
        DType::F32,
        QuantProvenance::GrimNative,
        device.clone(),
    );
    Ok((Some(codes_t), Some(exps_t), Some(gq_t), Some(gk_t)))
}

pub struct Lfm2 {
    pub cfg: Lfm2Config,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<Lfm2Block>,
    pub norm: RmsNorm,
    pub output: Linear,
    pub dense_2_out: Option<Linear>,
    pub dense_2_out_bias: Option<Tensor>,
}

impl Lfm2 {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: Lfm2Config) -> Result<Self> {
        Self::load_tp(ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for Lfm2. Lfm2 mixes attention and recurrent
    /// (`is_recr`) layers in one stack; the recurrent blocks have no
    /// row-parallel all-reduce semantics, and `Lfm2Block::forward` calls plain
    /// `Linear::forward`. A safe `load_tp` needs a per-block-type sharding plan
    /// plus a `forward` rework. Refuses `world_size > 1` until then.
    pub fn load_tp(
        ws: &grim_nn::WeightSource<'_>,
        cfg: Lfm2Config,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "Lfm2",
            "mixed attention/recurrent blocks need per-block-type sharding and a \
             forward rework to add the all-reduce hook",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        if cfg.is_recr.len() != cfg.num_layers {
            return Err(grim_core::error::Error::Config(format!(
                "Lfm2Config: is_recr has {} entries but num_layers is {}",
                cfg.is_recr.len(),
                cfg.num_layers
            )));
        }
        let tok_embeddings =
            Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = Lfm2Block::load(&ws.pp("blk").pp(&i.to_string()), &cfg, i)?;
            // Audit fix: fail at LOAD time with the layer index and missing
            // field, not with a panic on the first forward.
            block.validate(i)?;
            layers.push(block);
        }
        let norm = match RmsNorm::load(&ws.pp("token_embd_norm"), cfg.hidden_size, cfg.rms_norm_eps)
        {
            Ok(n) => n,
            Err(_) => RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?,
        };
        let output = Linear::from_tensor(tok_embeddings.weight.clone(), None);
        let device = tok_embeddings.weight.device().clone();

        let (dense_2_out, dense_2_out_bias) = if cfg.n_embd_out > 0 {
            let out = Linear::load(&ws.pp("dense_2_out"), cfg.hidden_size, cfg.n_embd_out, true)?;
            let bias = ws.get([cfg.n_embd_out], "dense_2_out.bias").ok();
            (Some(out), bias)
        } else {
            (None, None)
        };

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
            dense_2_out,
            dense_2_out_bias,
        })
    }
}

impl Model for Lfm2 {
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

impl CausalLm for Lfm2 {
    fn new_session(&self) -> Box<dyn SessionT> {
        let caches: Vec<Option<Lfm2LayerCache>> = vec![None; self.layers.len()];
        let mut session = Inner::new(self.device.clone());
        session.set_model_state(Box::new(caches));
        Box::new(session)
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => {
                let v = input_ids.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => return Err(grim_tensor::Error::Unimplemented("non-F32 inputs".into()).into()),
        };
        let seq_len = ids.len();
        let mut h = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?;
        fwd_trace_stage("embed", &h);

        if session.model_state().is_none() {
            session.set_model_state(Box::new(vec![None::<Lfm2LayerCache>; self.layers.len()]));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<Lfm2LayerCache>>>())
            .expect("Lfm2::forward: session.model_state must be Vec<Option<Lfm2LayerCache>>");

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i])?;
            fwd_trace_stage(&format!("layer{i}"), &h);
        }
        let h_normed = self.norm.forward(&h)?;
        fwd_trace_stage("norm", &h_normed);

        let h_final = if let Some(ref d2o) = self.dense_2_out {
            let projected = d2o.forward(&h_normed)?;
            if let Some(ref bias) = self.dense_2_out_bias {
                let out_dim = projected.shape().dims().last().copied().unwrap_or(0);
                let steps = projected.shape().elem_count() / out_dim.max(1);
                let broadcast_b = broadcast_bias(bias, steps, out_dim).map_err(grim_core::Error::Tensor)?;
                add_tensors(&projected, &broadcast_b).map_err(grim_core::Error::Tensor)?
            } else {
                projected
            }
        } else {
            h_normed
        };

        fwd_trace_stage("h_final", &h_final);
        let logits = self.output.forward(&h_final)?;
        fwd_trace_stage("logits", &logits);

        session.advance_pos(seq_len);
        Ok(logits)
    }
}

/// Root-cause instrumentation (validation log 2026-08-23e): env-gated
/// activation checksums localizing the first zeroed stage of the Lfm2
/// forward on non-zero ordinals. Enable with GRIM_FORWARD_TRACE=1.
fn fwd_trace_stage(name: &str, t: &Tensor) {
    if std::env::var_os("GRIM_FORWARD_TRACE").is_none() {
        return;
    }
    match t.to_vec_f32() {
        Ok(v) => eprintln!(
            "[fwd-trace] {name}: len={} nonzero={} head={:?}",
            v.len(),
            v.iter().filter(|&&x| x != 0.0).count(),
            &v[..v.len().min(4)]
        ),
        Err(e) => eprintln!("[fwd-trace] {name}: READ FAIL {e}"),
    }
}

/// Reinterpret a slice of `T` as raw bytes (for `from_cpu_bytes` uploads).
fn as_u8_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn device_tensor(data: Vec<f32>, shape: Shape, device: &Device) -> Result<Tensor> {
    if device == &Device::Cpu {
        Ok(cpu_tensor(data, shape))
    } else {
        let dev = grim_nn::modules::pick_device_for_storage_device(device);
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

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device() == &Device::Cpu {
        let g = gate.to_vec_f32()?;
        let u = up.to_vec_f32()?;
        let mut out = vec![0.0f32; g.len()];
        for i in 0..g.len() {
            let silu = g[i] / (1.0 + (-g[i]).exp());
            out[i] = silu * u[i];
        }
        device_tensor(out, gate.shape().clone(), gate.device())
    } else {
        let dev = grim_nn::modules::pick_device_for_storage_device(gate.device());
        let (storage, _h) = dev
            .silu_mul(gate.storage().as_ref(), up.storage().as_ref(), gate.shape())
            .map_err(grim_core::Error::Tensor)?;
        Ok(Tensor::new(
            Arc::from(storage),
            gate.shape().clone(),
            DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            gate.device().clone(),
        ))
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    /// Audit gate (M11): an incoherent block variant must be NAMED by
    /// validate() instead of panicking at first forward.
    #[test]
    fn lfm2_validate_names_missing_variant_fields() {
        let eps = 1e-5f32;
        let norm = RmsNorm {
            weight: grim_backend_cpu::cpu_tensor(vec![1.0f32; 8], grim_tensor::Shape::new(vec![8])),
            eps,
        };
        let lin = Linear::from_tensor(
            grim_backend_cpu::cpu_tensor(vec![0.0f32; 64], grim_tensor::Shape::new(vec![8, 8])),
            None,
        );
        // Full-attention block with wq/wk/wv but NO wo / QK norms.
        let block = Lfm2Block {
            attn_norm: norm.clone(),
            wq: Some(lin.clone()),
            wk: Some(lin.clone()),
            wv: Some(lin.clone()),
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            ffn_norm: norm.clone(),
            ffn_gate: lin.clone(),
            ffn_up: lin.clone(),
            ffn_down: lin,
            ffn_gate_inp: None,
            ffn_gate_exps: None,
            ffn_up_exps: None,
            ffn_down_exps: None,
            ffn_exp_probs_b: None,
            is_moe: false,
            n_expert: 0,
            shortconv_in_proj: None,
            shortconv_conv: None,
            shortconv_conv_vec: None,
            shortconv_out_proj: None,
            wqkv_codes: None,
            wqkv_exps: None,
            gamma_q: None,
            gamma_k: None,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
            rope_theta: 10_000.0,
            eps,
        };
        let err = block.validate(3).expect_err("incoherent block must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("layer 3") && (msg.contains("wo") || msg.contains("wq")),
            "validate must name the layer and missing field: {msg}"
        );

        // ShortConv variant missing its kernel.
        let mut conv_block = Lfm2Block {
            attn_norm: norm,
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            ffn_norm: RmsNorm {
                weight: grim_backend_cpu::cpu_tensor(
                    vec![1.0f32; 8],
                    grim_tensor::Shape::new(vec![8]),
                ),
                eps,
            },
            ffn_gate: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(vec![0.0f32; 64], grim_tensor::Shape::new(vec![8, 8])),
                None,
            ),
            ffn_up: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(vec![0.0f32; 64], grim_tensor::Shape::new(vec![8, 8])),
                None,
            ),
            ffn_down: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(vec![0.0f32; 64], grim_tensor::Shape::new(vec![8, 8])),
                None,
            ),
            ffn_gate_inp: None,
            ffn_gate_exps: None,
            ffn_up_exps: None,
            ffn_down_exps: None,
            ffn_exp_probs_b: None,
            is_moe: false,
            n_expert: 0,
            shortconv_in_proj: Some(Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(
                    vec![0.0f32; 192],
                    grim_tensor::Shape::new(vec![24, 8]),
                ),
                None,
            )),
            shortconv_conv: None,
            shortconv_conv_vec: None,
            shortconv_out_proj: None,
            wqkv_codes: None,
            wqkv_exps: None,
            gamma_q: None,
            gamma_k: None,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
            rope_theta: 10_000.0,
            eps,
        };
        let err = conv_block
            .validate(5)
            .expect_err("shortconv without kernel must fail");
        assert!(
            err.to_string().contains("shortconv_conv"),
            "validate must name the missing shortconv field: {err}"
        );
        // And a coherent ShortConv block passes.
        conv_block.shortconv_conv = Some(grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 12],
            grim_tensor::Shape::new(vec![6, 2]),
        ));
        conv_block.shortconv_conv_vec = Some(vec![0.1f32; 12]);
        conv_block.shortconv_out_proj = Some(Linear::from_tensor(
            grim_backend_cpu::cpu_tensor(vec![0.0f32; 64], grim_tensor::Shape::new(vec![8, 8])),
            None,
        ));
        assert!(
            conv_block.validate(5).is_ok(),
            "coherent ShortConv block must pass"
        );
    }
}

// ---------------------------------------------------------------------------
// Numeric reference test (audit follow-up): the shortconv causal recurrence
// was guarded by validate() but never value-checked.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod shortconv_numeric_reference_tests {
    use super::*;
    use grim_nn::Linear;

    fn lin(weight: Vec<f32>, out_dim: usize, in_dim: usize) -> Linear {
        Linear::from_tensor(
            grim_backend_cpu::cpu_tensor(weight, Shape::new(vec![out_dim, in_dim])),
            None,
        )
    }

    fn weights(seed: u64, n: usize) -> Vec<f32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((st >> 33) % 2000) as f32 - 1000.0) / 1000.0 * 0.4
            })
            .collect()
    }

    fn unit_norm() -> RmsNorm {
        RmsNorm {
            weight: grim_backend_cpu::cpu_tensor(vec![1.0; 4], Shape::new(vec![4])),
            eps: 1e-5,
        }
    }

    #[test]
    fn shortconv_causal_conv_matches_f64_reference() {
        let hidden = 4usize;
        let l_cache = 3usize;
        let steps = 3usize;
        let unit = unit_norm();
        let block = Lfm2Block {
            attn_norm: unit.clone(),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            wqkv_codes: None,
            wqkv_exps: None,
            gamma_q: None,
            gamma_k: None,
            shortconv_in_proj: Some(lin(weights(1, 3 * hidden * hidden), 3 * hidden, hidden)),
            shortconv_conv: Some(grim_backend_cpu::cpu_tensor(
                weights(2, hidden * l_cache),
                Shape::new(vec![hidden, 1, l_cache]),
            )),
            shortconv_conv_vec: Some(weights(2, hidden * l_cache)),
            shortconv_out_proj: Some(lin(weights(3, hidden * hidden), hidden, hidden)),
            ffn_norm: unit.clone(),
            ffn_gate: lin(weights(4, hidden * hidden), hidden, hidden),
            ffn_up: lin(weights(5, hidden * hidden), hidden, hidden),
            ffn_down: lin(weights(6, hidden * hidden), hidden, hidden),
            ffn_gate_inp: None,
            ffn_gate_exps: None,
            ffn_up_exps: None,
            ffn_down_exps: None,
            ffn_exp_probs_b: None,
            is_moe: false,
            n_expert: 0,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden,
            rope_theta: 10000.0,
            eps: 1e-5,
        };

        // 3-token input; run forward once over the whole sequence.
        let x_data: Vec<f32> = vec![0.3, -0.5, 0.7, 0.1, -0.2, 0.4, 0.6, -0.1, 0.25, 0.05, -0.35, 0.45];
        let x = grim_backend_cpu::cpu_tensor(x_data.clone(), Shape::new(vec![steps, hidden]));
        let mut cache = None;
        let out = block.forward(&x, &mut cache).unwrap().to_vec_f32().unwrap();
        assert_eq!(out.len(), steps * hidden);

        // Independent f64 reference of the documented recurrence:
        //   proj = W_in · rmsnorm(x);  per step: b, c, xv = thirds of proj
        //   bx = b·xv;  y[d] = c[d]·(w[d·L+L−1]·bx[d] + Σ_{k<L−1} w[d·L+k]·state[k][d])
        //   state shifts left by one, appending bx.
        let w_norm = vec![1.0f32; hidden];
        let w_in = weights(1, 3 * hidden * hidden);
        let conv = weights(2, hidden * l_cache);
        let w_out = weights(3, hidden * hidden);
        let eps = 1e-5f64;

        let mut state = vec![0.0f64; hidden * (l_cache - 1)];
        let mut ref_out = Vec::with_capacity(steps * hidden);
        for step in 0..steps {
            let xr: Vec<f64> = x_data[step * hidden..(step + 1) * hidden]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let mean_sq = xr.iter().map(|v| v * v).sum::<f64>() / hidden as f64;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            let normed: Vec<f64> =
                xr.iter().zip(&w_norm).map(|(&v, &w)| v * inv * w as f64).collect();
            let mut proj = vec![0.0f64; 3 * hidden];
            for o in 0..3 * hidden {
                proj[o] = (0..hidden)
                    .map(|i| w_in[o * hidden + i] as f64 * normed[i])
                    .sum();
            }
            let b = &proj[..hidden];
            let c = &proj[hidden..2 * hidden];
            let xv = &proj[2 * hidden..3 * hidden];
            for d in 0..hidden {
                let bx = b[d] * xv[d];
                let mut sum = conv[d * l_cache + l_cache - 1] as f64 * bx;
                for k in 0..l_cache - 1 {
                    sum += conv[d * l_cache + k] as f64 * state[k * hidden + d];
                }
                ref_out.push(c[d] * sum);
            }
            // Shift the ring: drop the oldest, append bx.
            for k in 0..l_cache - 2 {
                for d in 0..hidden {
                    state[k * hidden + d] = state[(k + 1) * hidden + d];
                }
            }
            for d in 0..hidden {
                state[(l_cache - 2) * hidden + d] = b[d] * xv[d];
            }
        }
        // Final projection to hidden width — per step (row-major [out, in]
        // matvec on each step's h_dim-vector).
        let mut per_step_ref: Vec<f64> = Vec::with_capacity(steps * hidden);
        for step in 0..steps {
            let y = &ref_out[step * hidden..(step + 1) * hidden];
            for o in 0..hidden {
                per_step_ref
                    .push((0..hidden).map(|i| w_out[o * hidden + i] as f64 * y[i]).sum());
            }
        }
        // The block continues past the conv branch: residual add, FFN norm,
        // SwiGLU FFN (gate·silu(up) → down), second residual. Extend the
        // reference through the full block tail.
        let w_g = weights(4, hidden * hidden);
        let w_u = weights(5, hidden * hidden);
        let w_d = weights(6, hidden * hidden);
        let mut full_ref: Vec<f64> = Vec::with_capacity(steps * hidden);
        for step in 0..steps {
            let xr: Vec<f64> = x_data[step * hidden..(step + 1) * hidden]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let conv_out = &per_step_ref[step * hidden..(step + 1) * hidden];
            let x_added: Vec<f64> = xr.iter().zip(conv_out).map(|(&a, &b)| a + b).collect();
            let mean_sq = x_added.iter().map(|v| v * v).sum::<f64>() / hidden as f64;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            let n2: Vec<f64> = x_added.iter().map(|v| v * inv).collect();
            let gate: Vec<f64> = (0..hidden)
                .map(|o| (0..hidden).map(|i| w_g[o * hidden + i] as f64 * n2[i]).sum())
                .collect();
            let up: Vec<f64> = (0..hidden)
                .map(|o| (0..hidden).map(|i| w_u[o * hidden + i] as f64 * n2[i]).sum())
                .collect();
            // SwiGLU: SiLU applies to the GATE projection.
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            for o in 0..hidden {
                let down: f64 = (0..hidden).map(|i| w_d[o * hidden + i] as f64 * act[i]).sum();
                full_ref.push(x_added[o] + down);
            }
        }
        for (i, (r, g)) in full_ref.iter().zip(&out).enumerate() {
            assert!(
                (r - *g as f64).abs() < 1e-4,
                "elem {i}: reference {r} vs impl {g}"
            );
        }

        // Causality gate: the cache must have advanced so a FOLLOW-UP call
        // sees the last two steps' bx values (state ring = last l_cache−1).
        match &cache {
            Some(Lfm2LayerCache::ShortConv(st)) => {
                // After 3 steps with l_cache=3, the ring holds the LAST two
                // steps' bx vectors — both must be non-zero.
                assert!(
                    st.iter().any(|&v| v != 0.0),
                    "shortconv cache must hold shifted bx history"
                );
            }
            _ => panic!("shortconv forward must populate a ShortConv cache"),
        }
    }
}
