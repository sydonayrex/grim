//! LFM2 (Liquid Foundation Model v2) — `CausalLm` implementation in 100% Rust.
//! Includes recurrent ShortConv blocks and MoE gating logic.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, add_tensors};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};
use std::sync::Arc;

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
    Attention { k: Vec<f32>, v: Vec<f32> },
}

pub struct Lfm2Block {
    pub attn_norm: RmsNorm,
    pub wq: Option<Linear>,
    pub wk: Option<Linear>,
    pub wv: Option<Linear>,
    pub wo: Option<Linear>,
    pub attn_q_norm: Option<RmsNorm>,
    pub attn_k_norm: Option<RmsNorm>,
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
            let ffn_up_exps = Some(ws.get(
                [cfg.n_expert, cfg.n_ff_exp, cfg.hidden_size],
                "ffn_up_exps.weight",
            )?);
            let ffn_down_exps = Some(ws.get(
                [cfg.n_ff_exp, cfg.hidden_size, cfg.n_expert],
                "ffn_down_exps.weight",
            )?);
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
        })
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
            let q = self.wq.as_ref().unwrap().forward(&norm_x)?;
            let k = self.wk.as_ref().unwrap().forward(&norm_x)?;
            let v = self.wv.as_ref().unwrap().forward(&norm_x)?;

            let steps = q.shape().dims()[0];

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

            let kv_stride = self.num_kv_heads * self.head_dim;
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
            let (q_rot_storage, _) = dev.rope(
                q_norm.storage().as_ref(),
                &q_positions,
                &rope_cfg,
                &q_shape,
            )?;
            let (k_rot_storage, _) = dev.rope(
                k_norm.storage().as_ref(),
                &k_positions,
                &rope_cfg,
                &k_shape,
            )?;

            let q_rot_vec = q_rot_storage.to_cpu_vec_f32()?;
            let k_rot_vec = k_rot_storage.to_cpu_vec_f32()?;
            let v_vec = v.to_vec_f32()?;

            if cache.is_none() {
                *cache = Some(Lfm2LayerCache::Attention {
                    k: vec![],
                    v: vec![],
                });
            }

            let (k_hist, v_hist) = match cache.as_mut().unwrap() {
                Lfm2LayerCache::Attention { k, v } => (k, v),
                _ => {
                    return Err(grim_core::error::Error::Session(
                        "Mismatched Attention layer cache".into(),
                    ));
                }
            };

            k_hist.extend_from_slice(&k_rot_vec);
            v_hist.extend_from_slice(&v_vec);

            let total_kv_tokens = k_hist.len() / kv_stride;
            let num_head_dims = self.num_heads * self.head_dim;
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            let mut attn_out_vec = vec![0.0f32; steps * num_head_dims];

            for t in 0..steps {
                let past_tokens = (total_kv_tokens - steps) + t;
                for h in 0..self.num_heads {
                    let kvh = (h * self.num_kv_heads) / self.num_heads;
                    let mut scores = vec![0.0f32; past_tokens + 1];
                    for t2 in 0..=past_tokens {
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            dot += q_rot_vec[t * num_head_dims + h * self.head_dim + d]
                                * k_hist[t2 * kv_stride + kvh * self.head_dim + d];
                        }
                        scores[t2] = dot * scale;
                    }

                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_s = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_s += *s;
                    }
                    for s in &mut scores {
                        *s /= sum_s;
                    }

                    for d in 0..self.head_dim {
                        let mut acc = 0.0f32;
                        for t2 in 0..=past_tokens {
                            acc += scores[t2] * v_hist[t2 * kv_stride + kvh * self.head_dim + d];
                        }
                        attn_out_vec[t * num_head_dims + h * self.head_dim + d] = acc;
                    }
                }
            }

            let attn_tensor = device_tensor(
                attn_out_vec,
                Shape::new(vec![steps, num_head_dims]),
                norm_x.device(),
            )?;
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
            layers.push(Lfm2Block::load(&ws.pp("blk").pp(&i.to_string()), &cfg, i)?);
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

        if session.model_state().is_none() {
            session.set_model_state(Box::new(vec![None::<Lfm2LayerCache>; self.layers.len()]));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<Lfm2LayerCache>>>())
            .expect("Lfm2::forward: session.model_state must be Vec<Option<Lfm2LayerCache>>");

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i])?;
        }
        let h_normed = self.norm.forward(&h)?;

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

        let logits = self.output.forward(&h_final)?;

        session.advance_pos(seq_len);
        Ok(logits)
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
