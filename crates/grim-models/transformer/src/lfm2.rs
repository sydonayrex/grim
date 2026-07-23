//! LFM2 (Liquid Foundation Model v2) — `CausalLm` implementation in 100% Rust.
//! Includes recurrent ShortConv blocks and MoE gating logic.

use std::sync::{Arc, Mutex};
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, DType, Shape, Tensor};

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
    pub is_recr: Vec<bool>, // Whether each layer is recurrent
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
    },
}

pub struct Lfm2Block {
    pub attn_norm: RmsNorm,
    // Attention path
    pub wq: Option<Linear>,
    pub wk: Option<Linear>,
    pub wv: Option<Linear>,
    pub wo: Option<Linear>,
    pub attn_q_norm: Option<RmsNorm>,
    pub attn_k_norm: Option<RmsNorm>,
    // Recurrent ShortConv path
    pub shortconv_in_proj: Option<Linear>,
    pub shortconv_conv: Option<Tensor>,
    pub shortconv_conv_vec: Option<Vec<f32>>,
    pub shortconv_out_proj: Option<Linear>,
    // Feed Forward
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

impl Lfm2Block {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &Lfm2Config, layer_idx: usize) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let is_recurrent = cfg.is_recr.get(layer_idx).copied().unwrap_or(false);

        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if !is_recurrent {
            let wq = Some(Linear::load(&ws.pp("attn_q"), cfg.hidden_size, cfg.num_heads * cfg.head_dim, false)?);
            let wk = Some(Linear::load(&ws.pp("attn_k"), cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, false)?);
            let wv = Some(Linear::load(&ws.pp("attn_v"), cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, false)?);
            let wo = Some(Linear::load(&ws.pp("attn_output"), cfg.num_heads * cfg.head_dim, cfg.hidden_size, false)?);
            let attn_q_norm = Some(RmsNorm::load(&ws.pp("attn_q_norm"), cfg.head_dim, cfg.rms_norm_eps)?);
            let attn_k_norm = Some(RmsNorm::load(&ws.pp("attn_k_norm"), cfg.head_dim, cfg.rms_norm_eps)?);
            (wq, wk, wv, wo, attn_q_norm, attn_k_norm)
        } else {
            (None, None, None, None, None, None)
        };

        let (shortconv_in_proj, shortconv_conv, shortconv_conv_vec, shortconv_out_proj) = if is_recurrent {
            let in_proj = Some(Linear::load(&ws.pp("shortconv.in_proj"), cfg.hidden_size, 3 * cfg.hidden_size, false)?);
            // Canonical LFM2 conv.weight is stored as [hidden, *, kernel] row-major:
            //   - safetensors: 3D [hidden, 1, kernel] — the middle dim is in_channels=1.
            //   - GGUF:        2D [hidden, kernel] — the converter squeezes the in_channels
            //     axis, and the raw flat bytes remain in [hidden, kernel] row-major order.
            //     Although the GGUF tensor dims are written as [3, 1024] (kernel, hidden),
            //     the actual data layout is [hidden, kernel] row-major — verified by direct
            //     byte-for-byte comparison with the safetensors source. So no transpose is
            //     ever needed; the dequantized flat data is already in the canonical layout
            //     (weight[d * kernel + k] = conv weight for channel d, kernel tap k).
            let conv = ws.get([cfg.hidden_size, cfg.n_shortconv_l_cache], "shortconv.conv.weight")
                .or_else(|_| ws.get([cfg.hidden_size, 1, cfg.n_shortconv_l_cache], "shortconv.conv.weight"))?;
            let conv_vec = conv.to_vec_f32().ok().map(|raw| {
                // The flat data is already [hidden, kernel] row-major for both formats.
                // No transpose required (previous code incorrectly transposed the GGUF
                // path, scrambling the conv weight and producing gibberish output).
                raw
            });
            let out_proj = Some(Linear::load(&ws.pp("shortconv.out_proj"), cfg.hidden_size, cfg.hidden_size, false)?);
            (in_proj, Some(conv), conv_vec, out_proj)
        } else {
            (None, None, None, None)
        };

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ffn_gate = Linear::load(&ws.pp("ffn_gate"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_up = Linear::load(&ws.pp("ffn_up"), cfg.hidden_size, cfg.intermediate_size, false)?;
        let ffn_down = Linear::load(&ws.pp("ffn_down"), cfg.intermediate_size, cfg.hidden_size, false)?;

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
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    pub fn forward(&self, x: &Tensor, cache: &mut Option<Lfm2LayerCache>) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;

        let block_out = if let Some(in_proj) = &self.shortconv_in_proj {
            // ShortConv recurrent block (gated depthwise causal 1D convolution).
            // State: only the *previous* (l_cache-1) Bx columns (oldest first, channels each).
            // Conv: the LAST kernel tap weights the *current* Bx; earlier taps weight history.
            let proj = in_proj.forward(&norm_x)?;
            let proj_v = proj.to_vec_f32()?;
            let h_dim = norm_x.shape().dims().last().copied().unwrap_or(0);
            let steps = proj_v.len() / (3 * h_dim);

            let mut y_out = vec![0.0f32; steps * h_dim];

            let conv_kernel_vec = self.shortconv_conv_vec.as_ref().unwrap();
            // Shape may be 2D [hidden, kernel] (GGUF) or 3D [hidden, 1, kernel] (safetensors).
            // Use the LAST dim as the kernel size regardless of rank.
            let conv_shape = self.shortconv_conv.as_ref().unwrap().shape().dims();
            let l_cache = *conv_shape.last().unwrap_or(&3);

            if cache.is_none() {
                *cache = Some(Lfm2LayerCache::ShortConv(vec![0.0f32; h_dim * (l_cache - 1)]));
            }

            let state = match cache.as_mut().unwrap() {
                Lfm2LayerCache::ShortConv(st) => st,
                _ => return Err(grim_core::error::Error::Session("Mismatched ShortConv layer cache".into())),
            };

            for step in 0..steps {
                let offset = step * 3 * h_dim;
                let b = &proj_v[offset..offset + h_dim];
                let c = &proj_v[offset + h_dim..offset + 2 * h_dim];
                let x_val = &proj_v[offset + 2 * h_dim..offset + 3 * h_dim];

                // Element-wise mul: B ⊙ h̃ → bx
                let bx: Vec<f32> = b.iter().zip(x_val.iter()).map(|(bv, xv)| bv * xv).collect();

                // Causal depthwise conv_step (per bebelm-main kernels/conv.rs).
                // out[d] = weight[d*l_cache + l_cache-1]*bx[d] + sum_{k=0}^{l_cache-2} weight[d*l_cache + k]*state[k*h_dim + d]
                for d in 0..h_dim {
                    let w_base = d * l_cache;
                    let mut sum = conv_kernel_vec[w_base + l_cache - 1] * bx[d];
                    for k in 0..l_cache - 1 {
                        sum += conv_kernel_vec[w_base + k] * state[k * h_dim + d];
                    }
                    y_out[step * h_dim + d] = c[d] * sum;
                }

                // Slide state: shift left, append current bx (per bebelm conv_step_op state update).
                if l_cache > 1 {
                    state.copy_within(h_dim.., 0);
                    state[(l_cache - 2) * h_dim..].copy_from_slice(&bx);
                }
            }

            let y_tensor = device_tensor(y_out, Shape::new(vec![steps, h_dim]), norm_x.device())?;
            self.shortconv_out_proj.as_ref().unwrap().forward(&y_tensor)?
        } else {
            // Full Causal Scaled Dot-Product Attention with GQA & Per-Head RMSNorm
            let q = self.wq.as_ref().unwrap().forward(&norm_x)?;
            let k = self.wk.as_ref().unwrap().forward(&norm_x)?;
            let v = self.wv.as_ref().unwrap().forward(&norm_x)?;

            let steps = q.shape().dims()[0];

            // Apply per-head RMSNorm on Q and K
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
            let mut k_norm_vec = k_norm.to_vec_f32()?;
            let mut q_norm_vec = q_norm.to_vec_f32()?;
            let v_vec = v.to_vec_f32()?;

            if cache.is_none() {
                *cache = Some(Lfm2LayerCache::Attention { k: vec![], v: vec![] });
            }

            let (k_hist, v_hist) = match cache.as_mut().unwrap() {
                Lfm2LayerCache::Attention { k, v } => (k, v),
                _ => return Err(grim_core::error::Error::Session("Mismatched Attention layer cache".into())),
            };

            let kv_stride = self.num_kv_heads * self.head_dim;
            let current_total = k_hist.len() / kv_stride;
            let half = self.head_dim / 2;

            // Apply RoPE (Rotary Position Embedding) to Q and K at current token positions
            for t in 0..steps {
                let pos = (current_total + t) as f32;
                // RoPE for Q
                for h in 0..self.num_heads {
                    let head_offset = t * (self.num_heads * self.head_dim) + h * self.head_dim;
                    for i in 0..half {
                        let freq = 1.0 / self.rope_theta.powf((2 * i) as f32 / self.head_dim as f32);
                        let angle = pos * freq;
                        let (cos, sin) = (angle.cos(), angle.sin());
                        let q0 = q_norm_vec[head_offset + 2 * i];
                        let q1 = q_norm_vec[head_offset + 2 * i + 1];
                        q_norm_vec[head_offset + 2 * i] = q0 * cos - q1 * sin;
                        q_norm_vec[head_offset + 2 * i + 1] = q0 * sin + q1 * cos;
                    }
                }
                // RoPE for K
                for kvh in 0..self.num_kv_heads {
                    let head_offset = t * (self.num_kv_heads * self.head_dim) + kvh * self.head_dim;
                    for i in 0..half {
                        let freq = 1.0 / self.rope_theta.powf((2 * i) as f32 / self.head_dim as f32);
                        let angle = pos * freq;
                        let (cos, sin) = (angle.cos(), angle.sin());
                        let k0 = k_norm_vec[head_offset + 2 * i];
                        let k1 = k_norm_vec[head_offset + 2 * i + 1];
                        k_norm_vec[head_offset + 2 * i] = k0 * cos - k1 * sin;
                        k_norm_vec[head_offset + 2 * i + 1] = k0 * sin + k1 * cos;
                    }
                }
            }

            k_hist.extend_from_slice(&k_norm_vec);
            v_hist.extend_from_slice(&v_vec);

            let kv_stride = self.num_kv_heads * self.head_dim;
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
                            dot += q_norm_vec[t * num_head_dims + h * self.head_dim + d]
                                * k_hist[t2 * kv_stride + kvh * self.head_dim + d];
                        }
                        scores[t2] = dot * scale;
                    }

                    // Softmax
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_s = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_s += *s;
                    }
                    for s in &mut scores {
                        *s /= sum_s;
                    }

                    // Weighted sum of V
                    for d in 0..self.head_dim {
                        let mut acc = 0.0f32;
                        for t2 in 0..=past_tokens {
                            acc += scores[t2] * v_hist[t2 * kv_stride + kvh * self.head_dim + d];
                        }
                        attn_out_vec[t * num_head_dims + h * self.head_dim + d] = acc;
                    }
                }
            }

            let attn_tensor = device_tensor(attn_out_vec, Shape::new(vec![steps, num_head_dims]), norm_x.device())?;
            self.wo.as_ref().unwrap().forward(&attn_tensor)?
        };

        // Residual connection
        let x_added = add_tensors(x, &block_out)?;

        // FFN block
        let norm_x_ffn = self.ffn_norm.forward(&x_added)?;
        let gate = self.ffn_gate.forward(&norm_x_ffn)?;
        let up = self.ffn_up.forward(&norm_x_ffn)?;
        let activated = silu_mul(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&activated)?;

        add_tensors(&x_added, &ffn_out)
    }
}

pub struct Lfm2 {
    pub cfg: Lfm2Config,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<Lfm2Block>,
    pub norm: RmsNorm,
    pub output: Linear,
    pub caches: Mutex<Vec<Option<Lfm2LayerCache>>>,
}

impl Lfm2 {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: Lfm2Config) -> Result<Self> {
        let tok_embeddings = Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Lfm2Block::load(&ws.pp("blk").pp(&i.to_string()), &cfg, i)?);
        }
        // canonical LFM2: post-norm before lm_head is fused into `token_embd_norm`
        // (an embedding pre-norm), and lm_head is tied to `token_embd`.
        let norm = match RmsNorm::load(&ws.pp("token_embd_norm"), cfg.hidden_size, cfg.rms_norm_eps) {
            Ok(n) => n,
            Err(_) => RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?,
        };
        // LFM2 uses tied embeddings (lm_head = token_embd^T). After `Embedding::load`
        // — which transposes GGUF's [hidden, vocab] to [vocab, hidden] —
        // the embedding weight is row-major [vocab, hidden], so it can be reused
        // directly as the Linear lm_head weight.
        let output = Linear::from_tensor(tok_embeddings.weight.clone(), None);
        let device = tok_embeddings.weight.device().clone();
        let caches = Mutex::new(vec![None; cfg.num_layers]);

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
            caches,
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
}

impl CausalLm for Lfm2 {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
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
        let mut h = self.tok_embeddings.forward(&ids, seq_len, self.cfg.hidden_size)?;

        let mut caches_guard = self.caches.lock().unwrap();
        if session.current_pos() == 0 {
            *caches_guard = vec![None; self.layers.len()];
        }
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches_guard[i])?;
        }

        let h_normed = self.norm.forward(&h)?;
        let logits = self.output.forward(&h_normed)?;

        session.advance_pos(seq_len);
        Ok(logits)
    }
}

// Helpers
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

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_nn::modules::pick_device_for_tensor(a);
    let (s, h) = grim_tensor::BackendDevice::add(&*dev, a.storage().as_ref(), b.storage().as_ref(), a.shape())?;
    h.synchronize()?;
    Ok(Tensor::new(Arc::from(s), a.shape().clone(), DType::F32, a.provenance().clone(), a.device().clone()))
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
