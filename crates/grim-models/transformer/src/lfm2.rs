//! LFM2 (Liquid Foundation Model v2) — `CausalLm` implementation in 100% Rust.
//! Includes recurrent ShortConv blocks and MoE gating logic.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, add_tensors};
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
        k_dev: Option<Tensor>,
        v_dev: Option<Tensor>,
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
                let conv_vec = conv.to_vec_f32().ok().map(|raw| raw);
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

            let y_tensor = device_tensor(y_out, Shape::new(vec![steps, h_dim]), norm_x.device())?;
            self.shortconv_out_proj
                .as_ref()
                .unwrap()
                .forward(&y_tensor)?
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
                (self.fused_qkv(&norm_x, cache, steps, hidden)?, None)
            } else {
                let q = self.wq.as_ref().unwrap().forward(&norm_x)?;
                let k = self.wk.as_ref().unwrap().forward(&norm_x)?;
                let v = self.wv.as_ref().unwrap().forward(&norm_x)?;

                let q_2d = device_tensor(
                    q.to_vec_f32()?,
                    Shape::new(vec![steps * self.num_heads, self.head_dim]),
                    norm_x.device(),
                )?;
                let q_norm = self.attn_q_norm.as_ref().unwrap().forward(&q_2d)?;

                let k_2d = device_tensor(
                    k.to_vec_f32()?,
                    Shape::new(vec![steps * self.num_kv_heads, self.head_dim]),
                    norm_x.device(),
                )?;
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
                            *k_dev = Some(Tensor::new(
                                Arc::from(dev.zeros(&shape, DType::F32)?),
                                shape.clone(),
                                DType::F32,
                                QuantProvenance::GrimNative.into(),
                                norm_x.device().clone(),
                            ));
                            *v_dev = Some(Tensor::new(
                                Arc::from(dev.zeros(&shape, DType::F32)?),
                                shape,
                                DType::F32,
                                QuantProvenance::GrimNative.into(),
                                norm_x.device().clone(),
                            ));
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
                let grow =
                    |slot: &mut Option<Tensor>, row_len: usize| -> Result<()> {
                        let old_data = slot.as_ref().map(|t| t.to_vec_f32()).transpose()?;
                        let mut data = vec![0f32; new_cap * row_len];
                        if let Some(old) = old_data {
                            let keep = old.len().min(data.len());
                            data[..keep].copy_from_slice(&old[..keep]);
                        }
                        let shape = Shape::new(vec![new_cap, row_len]);
                        *slot = Some(Tensor::new(
                            Arc::from(dev.from_cpu_bytes(
                                as_u8_slice(&data),
                                &shape,
                                DType::F32,
                            )?),
                            shape,
                            DType::F32,
                            QuantProvenance::GrimNative.into(),
                            norm_x.device().clone(),
                        ));
                        Ok(())
                    };
                grow(k_dev, n_k)?;
                grow(v_dev, n_v)?;
                max_seq = new_cap;
            }
        }
        if k_dev.is_none() {
            let k_shape = Shape::new(vec![max_seq, n_k]);
            *k_dev = Some(Tensor::new(
                Arc::from(dev.zeros(&k_shape, DType::F32)?),
                k_shape,
                DType::F32,
                QuantProvenance::GrimNative.into(),
                norm_x.device().clone(),
            ));
            let v_shape = Shape::new(vec![max_seq, n_v]);
            *v_dev = Some(Tensor::new(
                Arc::from(dev.zeros(&v_shape, DType::F32)?),
                v_shape,
                DType::F32,
                QuantProvenance::GrimNative.into(),
                norm_x.device().clone(),
            ));
        }

        let q_shape = Shape::new(vec![steps, n_q]);
        let q_out = Tensor::new(
            Arc::from(dev.zeros(&q_shape, DType::F32)?),
            q_shape,
            DType::F32,
            QuantProvenance::GrimNative.into(),
            norm_x.device().clone(),
        );

        let pos_shape = Shape::new(vec![steps]);
        let pos_storage = dev.from_cpu_bytes(as_u8_slice(&positions), &pos_shape, DType::U32)?;

        let handle = dev.fused_mxfp4_gemm_qk_norm_rope_kv(
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
        handle.synchronize()?;

        let q_rot_vec = q_out.to_vec_f32()?;
        let k_cache_vec = k_dev.as_ref().unwrap().to_vec_f32()?;
        let v_cache_vec = v_dev.as_ref().unwrap().to_vec_f32()?;

        let (k_hist, v_hist) = match cache.as_mut().unwrap() {
            Lfm2LayerCache::Attention { k, v, .. } => (k, v),
            _ => {
                return Err(grim_core::error::Error::Session(
                    "Mismatched Attention layer cache".into(),
                ));
            }
        };
        for t in 0..steps {
            let pos = cache_offset + t;
            let koff = pos * n_k;
            k_hist.extend_from_slice(&k_cache_vec[koff..koff + n_k]);
            let voff = pos * n_v;
            v_hist.extend_from_slice(&v_cache_vec[voff..voff + n_v]);
        }
        Ok(q_rot_vec)
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
                for d in 0..hidden {
                    let g_idx = best_e * n_ff * hidden + f * hidden + d;
                    let u_idx = best_e * n_ff * hidden + f * hidden + d;
                    g_sum += x_s[d] * gate_exps_vec[g_idx];
                    u_sum += x_s[d] * up_exps_vec[u_idx];
                }
                gate_e[f] = g_sum;
                up_e[f] = u_sum;
            }

            let mut activated = vec![0.0f32; n_ff];
            for f in 0..n_ff {
                let silu = gate_e[f] / (1.0 + (-gate_e[f]).exp());
                activated[f] = silu * up_e[f];
            }

            for d in 0..hidden {
                let mut acc = 0.0f32;
                for f in 0..n_ff {
                    let d_idx = best_e * n_ff * hidden + f * hidden + d;
                    acc += activated[f] * down_exps_vec[d_idx];
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
fn build_fused_qkv_pack(
    wq: &Linear,
    wk: &Linear,
    wv: &Linear,
    attn_q_norm: &RmsNorm,
    attn_k_norm: &RmsNorm,
    cfg: &Lfm2Config,
) -> Result<(
    Option<Tensor>,
    Option<Tensor>,
    Option<Tensor>,
    Option<Tensor>,
)> {
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
        QuantProvenance::GrimNative.into(),
        device.clone(),
    );
    let exps_t = Tensor::new(
        Arc::from(exps_storage),
        exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
        QuantProvenance::GrimNative.into(),
        device.clone(),
    );
    let gq_t = Tensor::new(
        Arc::from(gq_storage),
        gamma_shape.clone(),
        DType::F32,
        QuantProvenance::GrimNative.into(),
        device.clone(),
    );
    let gk_t = Tensor::new(
        Arc::from(gk_storage),
        gamma_shape.clone(),
        DType::F32,
        QuantProvenance::GrimNative.into(),
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
                let bias_vec = bias.to_vec_f32()?;
                let proj_vec = projected.to_vec_f32()?;
                let mut out = proj_vec;
                for i in 0..out.len() {
                    out[i] += bias_vec[i % bias_vec.len()];
                }
                device_tensor(out, projected.shape().clone(), projected.device())?
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
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
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
            grim_tensor::QuantProvenance::GrimNative.into(),
            device.clone(),
        ))
    }
}

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let g = gate.to_vec_f32()?;
    let u = up.to_vec_f32()?;
    let mut out = vec![0.0f32; g.len()];
    for i in 0..g.len() {
        let silu = g[i] / (1.0 + (-g[i]).exp());
        out[i] = silu * u[i];
    }
    device_tensor(out, gate.shape().clone(), gate.device())
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
            grim_backend_cpu::cpu_tensor(
                vec![0.0f32; 64],
                grim_tensor::Shape::new(vec![8, 8]),
            ),
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
                grim_backend_cpu::cpu_tensor(
                    vec![0.0f32; 64],
                    grim_tensor::Shape::new(vec![8, 8]),
                ),
                None,
            ),
            ffn_up: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(
                    vec![0.0f32; 64],
                    grim_tensor::Shape::new(vec![8, 8]),
                ),
                None,
            ),
            ffn_down: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(
                    vec![0.0f32; 64],
                    grim_tensor::Shape::new(vec![8, 8]),
                ),
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
        let err = conv_block.validate(5).expect_err("shortconv without kernel must fail");
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
        assert!(conv_block.validate(5).is_ok(), "coherent ShortConv block must pass");
    }
}
