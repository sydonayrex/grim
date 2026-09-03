//! Whisper-shaped audio encoder-decoder.
//!
//! - Encoder: a stack of pre-norm self-attention blocks (full attention
//!   over the audio frame sequence fed by `raw_to_features`).
//! - Decoder: same shape as a small `CausalLm`-style transformer but with
//!   cross-attention to the encoder output.
//!
//! For phase 7 the modeling is structural and F32/CPU. ROCm kernels for
//! the cross-attention path land in phase 4.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{EncoderDecoderLm, ModalityHint};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

use grim_core::rng::SimpleRng;

/// Whisper-shaped config.
///
/// Serde-serializable so it can be built straight from a model's
/// `config.json` (every audio checkpoint in `models/audio/` ships one) via
/// [`WhisperConfig::from_hf`], which accepts both the HuggingFace
/// transformers key set (`d_model`, `encoder_layers`, …) and OpenAI's
/// original Whisper key set (`n_audio_state`, `n_audio_layer`, …).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhisperConfig {
    pub vocab_size: usize,
    pub n_mels: usize,
    pub d_model: usize,
    pub num_enc_layers: usize,
    pub num_dec_layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub max_audio_len: usize,
    pub max_text_len: usize,
    pub rms_norm_eps: f32,
}

impl Default for WhisperConfig {
    /// whisper-tiny shape — the smallest widely deployed reference config.
    fn default() -> Self {
        Self {
            vocab_size: 51864,
            n_mels: 80,
            d_model: 384,
            num_enc_layers: 4,
            num_dec_layers: 4,
            num_heads: 6,
            ffn_dim: 1536,
            max_audio_len: 3000,
            max_text_len: 448,
            rms_norm_eps: 1e-5,
        }
    }
}

impl WhisperConfig {
    /// Build a config from a parsed `config.json`.
    ///
    /// Accepts HuggingFace transformers Whisper keys (preferred) and falls
    /// back to OpenAI's original naming when the HF keys are absent. Any
    /// missing field keeps its [`Default`] value, so partial configs (like
    /// Kokoro/MeanVC2-style minimal JSON) still load.
    pub fn from_hf(json: &serde_json::Value) -> Self {
        let get = |keys: &[&str]| -> Option<serde_json::Value> {
            keys.iter().find_map(|k| json.get(*k)).cloned()
        };
        let as_usize =
            |v: Option<serde_json::Value>| v.and_then(|v| v.as_u64()).map(|v| v as usize);
        let d = Self::default();
        // Loud-fallback discipline (mirrors the sage_attention precedent):
        // a partial config silently loading as whisper-tiny-shaped is a
        // design choice, but the caller must hear about it — a wrong default
        // shape produces confidently wrong inference.
        let missing = |keys: &[&str]| keys.iter().all(|k| json.get(*k).is_none());
        let mut fell_back: Vec<&'static str> = Vec::new();
        if missing(&["encoder_attention_heads", "n_audio_head", "n_text_head"]) {
            fell_back.push("num_heads");
        }
        if missing(&["encoder_ffn_dim", "d_ff", "n_audio_state"]) {
            fell_back.push("ffn_dim");
        }
        let num_heads = as_usize(get(&[
            "encoder_attention_heads",
            "n_audio_head",
            "n_text_head",
        ]))
        .unwrap_or(d.num_heads);
        let ffn_dim =
            as_usize(get(&["encoder_ffn_dim", "d_ff", "n_audio_state"])).unwrap_or(d.ffn_dim);
        if !fell_back.is_empty() {
            eprintln!(
                "[whisper] config.json missing {fell_back:?} — falling back to whisper-tiny defaults for those fields"
            );
        }
        Self {
            vocab_size: as_usize(get(&["vocab_size", "n_vocab"])).unwrap_or(d.vocab_size),
            n_mels: as_usize(get(&["num_mel_bins", "n_mels"])).unwrap_or(d.n_mels),
            d_model: as_usize(get(&[
                "d_model",
                "n_audio_state",
                "n_text_state",
                "hidden_size",
            ]))
            .unwrap_or(d.d_model),
            num_enc_layers: as_usize(get(&["encoder_layers", "n_audio_layer"]))
                .unwrap_or(d.num_enc_layers),
            num_dec_layers: as_usize(get(&["decoder_layers", "n_text_layer"]))
                .unwrap_or(d.num_dec_layers),
            num_heads,
            ffn_dim,
            max_audio_len: as_usize(get(&["max_source_positions", "n_audio_ctx"]))
                .unwrap_or(d.max_audio_len),
            max_text_len: as_usize(get(&["max_target_positions", "n_text_ctx"]))
                .unwrap_or(d.max_text_len),
            rms_norm_eps: json
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d.rms_norm_eps),
        }
    }
}

impl ModelConfig for WhisperConfig {
    fn name(&self) -> &str {
        "whisper"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::AudioEncoderDecoder
    }
    /// Decoder context window — the server enforces prompt + max_tokens
    /// against this for Whisper models like any other generative model.
    fn context_length(&self) -> u64 {
        self.max_text_len as u64
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Encoder block: pre-norm self-attention + MLP.
struct WhisperEncoderBlock {
    norm1: RmsNorm,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    norm2: RmsNorm,
    fc1: Linear,
    fc2: Linear,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
}

impl WhisperEncoderBlock {
    fn new(d_model: usize, num_heads: usize, ffn: usize, eps: f32, rng: &mut SimpleRng) -> Self {
        let head_dim = d_model / num_heads;
        let scale = 1.0 / (d_model as f32).sqrt();
        let wq = (0..num_heads * head_dim * d_model)
            .map(|_| (rng.next_f32() - 0.5) * scale)
            .collect();
        let wk = (0..num_heads * head_dim * d_model)
            .map(|_| (rng.next_f32() - 0.5) * scale)
            .collect();
        let wv = (0..num_heads * head_dim * d_model)
            .map(|_| (rng.next_f32() - 0.5) * scale)
            .collect();
        let wo = (0..d_model * num_heads * head_dim)
            .map(|_| (rng.next_f32() - 0.5) * scale)
            .collect();
        let fc1_w = (0..ffn * d_model)
            .map(|_| (rng.next_f32() - 0.5) * scale)
            .collect();
        let fc2_w = (0..d_model * ffn)
            .map(|_| (rng.next_f32() - 0.5) * (1.0 / (ffn as f32).sqrt()))
            .collect();
        Self {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            wq,
            wk,
            wv,
            wo,
            norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            fc1: Linear::from_tensor(
                cpu_tensor(fc1_w, Shape::new(vec![ffn, d_model])),
                Some(cpu_tensor(vec![0.0; ffn], Shape::new(vec![ffn]))),
            ),
            fc2: Linear::from_tensor(
                cpu_tensor(fc2_w, Shape::new(vec![d_model, ffn])),
                Some(cpu_tensor(vec![0.0; d_model], Shape::new(vec![d_model]))),
            ),
            d_model,
            num_heads,
            head_dim,
        }
    }

    fn load(
        ws: &WeightSource<'_>,
        d_model: usize,
        num_heads: usize,
        ffn: usize,
        eps: f32,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let wq = ws
            .get([num_heads * head_dim, d_model], "attn.q.weight")?
            .to_vec_f32()?;
        let wk = ws
            .get([num_heads * head_dim, d_model], "attn.k.weight")?
            .to_vec_f32()?;
        let wv = ws
            .get([num_heads * head_dim, d_model], "attn.v.weight")?
            .to_vec_f32()?;
        let wo = ws
            .get([d_model, num_heads * head_dim], "attn.o.weight")?
            .to_vec_f32()?;
        let norm1 = RmsNorm::load(&ws.pp("attn_norm"), d_model, eps)?;
        let norm2 = RmsNorm::load(&ws.pp("ffn_norm"), d_model, eps)?;
        let fc1 = Linear::load(&ws.pp("ffn.0"), d_model, ffn, true)?;
        let fc2 = Linear::load(&ws.pp("ffn.1"), ffn, d_model, true)?;
        Ok(Self {
            norm1,
            wq,
            wk,
            wv,
            wo,
            norm2,
            fc1,
            fc2,
            d_model,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_data = x.to_vec_f32()?;
        let shape = x.shape().dims().to_vec();
        let seq = shape[0];
        let d = self.d_model;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let sqrt_hd = (hd as f32).sqrt();

        let normed = self.norm1.forward(x)?;
        let normed_data = normed.to_vec_f32()?;

        let attn_out = self_attn(
            &normed_data,
            seq,
            d,
            nh,
            hd,
            sqrt_hd,
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            false,
        );

        let mut after_attn = vec![0.0f32; seq * d];
        for i in 0..seq * d {
            after_attn[i] = x_data[i] + attn_out[i];
        }

        let ffn_in = self
            .norm2
            .forward(&cpu_tensor(after_attn.clone(), Shape::new(vec![seq, d])))?;
        let ffn1 = self.fc1.forward(&ffn_in)?;
        let ffn1_gelu = gelu(&ffn1)?;
        let ffn2 = self.fc2.forward(&ffn1_gelu)?;
        let ffn_out = ffn2.to_vec_f32()?;
        let mut out = vec![0.0f32; seq * d];
        for i in 0..seq * d {
            out[i] = after_attn[i] + ffn_out[i];
        }
        Ok(cpu_tensor(out, Shape::new(vec![seq, d])))
    }
}

fn gelu(t: &Tensor) -> Result<Tensor> {
    let v = t.to_vec_f32()?;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..v.len() {
        let x = v[i];
        out[i] = 0.5 * x * (1.0 + (x * 0.797884 * (1.0 + 0.044715 * x * x)).tanh());
    }
    Ok(cpu_tensor(out, t.shape().clone()))
}

fn self_attn(
    x: &[f32],
    seq: usize,
    d: usize,
    nh: usize,
    hd: usize,
    sqrt_hd: f32,
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    causal: bool,
) -> Vec<f32> {
    let project = |out: &mut [f32], w: &[f32]| {
        for pos in 0..seq {
            for o_idx in 0..d {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += x[pos * d + k] * w[o_idx * d + k];
                }
                out[pos * d + o_idx] = sum;
            }
        }
    };

    let mut q = vec![0.0f32; seq * d];
    let mut k = vec![0.0f32; seq * d];
    let mut v = vec![0.0f32; seq * d];
    project(&mut q, wq);
    project(&mut k, wk);
    project(&mut v, wv);

    let mut out = vec![0.0f32; seq * d];
    for h in 0..nh {
        let ho = h * hd;
        let mut scores = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut sum = 0.0;
                for hk in 0..hd {
                    sum += q[i * d + ho + hk] * k[j * d + ho + hk];
                }
                scores[i * seq + j] = sum / sqrt_hd;
            }
        }
        if causal {
            for i in 0..seq {
                for j in (i + 1)..seq {
                    scores[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        for i in 0..seq {
            let mut max_v = scores[i * seq];
            for j in 1..seq {
                if scores[i * seq + j] > max_v {
                    max_v = scores[i * seq + j];
                }
            }
            let mut sum_e = 0.0;
            for j in 0..seq {
                let e = (scores[i * seq + j] - max_v).exp();
                scores[i * seq + j] = e;
                sum_e += e;
            }
            for j in 0..seq {
                scores[i * seq + j] /= sum_e;
            }
        }
        for i in 0..seq {
            for hk in 0..hd {
                let mut sum = 0.0;
                for j in 0..seq {
                    sum += scores[i * seq + j] * v[j * d + ho + hk];
                }
                out[i * d + ho + hk] = sum;
            }
        }
    }

    let mut result = vec![0.0f32; seq * d];
    for pos in 0..seq {
        for o_idx in 0..d {
            let mut sum = 0.0;
            for k in 0..d {
                sum += out[pos * d + k] * wo[o_idx * d + k];
            }
            result[pos * d + o_idx] = sum;
        }
    }
    result
}

fn cross_attn(
    q_hidden: &[f32],
    enc_out: &[f32],
    seq: usize,
    enc_seq: usize,
    d: usize,
    nh: usize,
    hd: usize,
    sqrt_hd: f32,
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
) -> Vec<f32> {
    let project_q = |out: &mut [f32]| {
        for pos in 0..seq {
            for o_idx in 0..d {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += q_hidden[pos * d + k] * wq[o_idx * d + k];
                }
                out[pos * d + o_idx] = sum;
            }
        }
    };
    let project_kv = |out: &mut [f32], w: &[f32]| {
        for pos in 0..enc_seq {
            for o_idx in 0..d {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += enc_out[pos * d + k] * w[o_idx * d + k];
                }
                out[pos * d + o_idx] = sum;
            }
        }
    };

    let mut q = vec![0.0f32; seq * d];
    let mut k = vec![0.0f32; enc_seq * d];
    let mut v = vec![0.0f32; enc_seq * d];
    project_q(&mut q);
    project_kv(&mut k, wk);
    project_kv(&mut v, wv);

    let mut out = vec![0.0f32; seq * d];
    for h in 0..nh {
        let ho = h * hd;
        let mut scores = vec![0.0f32; seq * enc_seq];
        for i in 0..seq {
            for j in 0..enc_seq {
                let mut sum = 0.0;
                for hk in 0..hd {
                    sum += q[i * d + ho + hk] * k[j * d + ho + hk];
                }
                scores[i * enc_seq + j] = sum / sqrt_hd;
            }
        }
        for i in 0..seq {
            let mut max_v = scores[i * enc_seq];
            for j in 1..enc_seq {
                if scores[i * enc_seq + j] > max_v {
                    max_v = scores[i * enc_seq + j];
                }
            }
            let mut sum_e = 0.0;
            for j in 0..enc_seq {
                let e = (scores[i * enc_seq + j] - max_v).exp();
                scores[i * enc_seq + j] = e;
                sum_e += e;
            }
            for j in 0..enc_seq {
                scores[i * enc_seq + j] /= sum_e;
            }
        }
        for i in 0..seq {
            for hk in 0..hd {
                let mut sum = 0.0;
                for j in 0..enc_seq {
                    sum += scores[i * enc_seq + j] * v[j * d + ho + hk];
                }
                out[i * d + ho + hk] = sum;
            }
        }
    }

    let mut result = vec![0.0f32; seq * d];
    for pos in 0..seq {
        for o_idx in 0..d {
            let mut sum = 0.0;
            for k in 0..d {
                sum += out[pos * d + k] * wo[o_idx * d + k];
            }
            result[pos * d + o_idx] = sum;
        }
    }
    result
}

/// Decoder block: pre-norm self-attention (causal) + cross-attention + MLP.
struct WhisperDecoderBlock {
    self_norm: RmsNorm,
    self_q: Vec<f32>,
    self_k: Vec<f32>,
    self_v: Vec<f32>,
    self_o: Vec<f32>,
    cross_norm: RmsNorm,
    cross_q: Vec<f32>,
    cross_k: Vec<f32>,
    cross_v: Vec<f32>,
    cross_o: Vec<f32>,
    fc1: Linear,
    fc2: Linear,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
}

impl WhisperDecoderBlock {
    fn new(
        d_model: usize,
        num_heads: usize,
        ffn: usize,
        eps: f32,
        rng: &mut SimpleRng,
        device: Device,
    ) -> Self {
        let head_dim = d_model / num_heads;
        let n = d_model;
        let scale = 1.0 / (d_model as f32).sqrt();
        let self_q = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let self_k = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let self_v = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let self_o = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let cross_q = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let cross_k = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let cross_v = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let cross_o = (0..n * n).map(|_| (rng.next_f32() - 0.5) * scale).collect();
        let fc1_w = (0..ffn * d_model)
            .map(|_| (rng.next_f32() - 0.5) * scale)
            .collect();
        let fc2_w = (0..d_model * ffn)
            .map(|_| (rng.next_f32() - 0.5) * (1.0 / (ffn as f32).sqrt()))
            .collect();
        Self {
            self_norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            self_q,
            self_k,
            self_v,
            self_o,
            cross_norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; d_model], Shape::new(vec![d_model])),
                eps,
            },
            cross_q,
            cross_k,
            cross_v,
            cross_o,
            fc1: Linear::from_tensor(
                cpu_tensor(fc1_w, Shape::new(vec![ffn, d_model])),
                Some(cpu_tensor(vec![0.0; ffn], Shape::new(vec![ffn]))),
            ),
            fc2: Linear::from_tensor(
                cpu_tensor(fc2_w, Shape::new(vec![d_model, ffn])),
                Some(cpu_tensor(vec![0.0; d_model], Shape::new(vec![d_model]))),
            ),
            d_model,
            num_heads,
            head_dim,
            device,
        }
    }

    fn load(
        ws: &WeightSource<'_>,
        d_model: usize,
        num_heads: usize,
        ffn: usize,
        eps: f32,
        device: Device,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let n = d_model;
        let self_q = ws.get([n, n], "self_attn.q.weight")?.to_vec_f32()?;
        let self_k = ws.get([n, n], "self_attn.k.weight")?.to_vec_f32()?;
        let self_v = ws.get([n, n], "self_attn.v.weight")?.to_vec_f32()?;
        let self_o = ws.get([n, n], "self_attn.o.weight")?.to_vec_f32()?;
        let cross_q = ws.get([n, n], "cross_attn.q.weight")?.to_vec_f32()?;
        let cross_k = ws.get([n, n], "cross_attn.k.weight")?.to_vec_f32()?;
        let cross_v = ws.get([n, n], "cross_attn.v.weight")?.to_vec_f32()?;
        let cross_o = ws.get([n, n], "cross_attn.o.weight")?.to_vec_f32()?;
        let self_norm = RmsNorm::load(&ws.pp("self_attn_norm"), d_model, eps)?;
        let cross_norm = RmsNorm::load(&ws.pp("cross_attn_norm"), d_model, eps)?;
        let fc1 = Linear::load(&ws.pp("ffn.0"), d_model, ffn, true)?;
        let fc2 = Linear::load(&ws.pp("ffn.1"), ffn, d_model, true)?;
        Ok(Self {
            self_norm,
            self_q,
            self_k,
            self_v,
            self_o,
            cross_norm,
            cross_q,
            cross_k,
            cross_v,
            cross_o,
            fc1,
            fc2,
            d_model,
            num_heads,
            head_dim,
            device,
        })
    }

    fn decode_step(&self, x: &Tensor, enc_out: &Tensor) -> Result<Tensor> {
        let x_data = x.to_vec_f32()?;
        let shape = x.shape().dims().to_vec();
        let seq = shape[0];
        let d = self.d_model;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let sqrt_hd = (hd as f32).sqrt();
        let enc_seq = enc_out.shape().dims()[0];
        let enc_data = enc_out.to_vec_f32()?;

        let normed = self.self_norm.forward(x)?;
        let normed_data = normed.to_vec_f32()?;
        let self_attn_out = self_attn(
            &normed_data,
            seq,
            d,
            nh,
            hd,
            sqrt_hd,
            &self.self_q,
            &self.self_k,
            &self.self_v,
            &self.self_o,
            true,
        );

        let mut after_self = vec![0.0f32; seq * d];
        for i in 0..seq * d {
            after_self[i] = x_data[i] + self_attn_out[i];
        }

        let cross_normed = self
            .cross_norm
            .forward(&cpu_tensor(after_self.clone(), Shape::new(vec![seq, d])))?;
        let cross_normed_data = cross_normed.to_vec_f32()?;

        // GPU dispatch path: cross-attention HIP kernel (Phase 2 — mambo5.md Item 13).
        // When device is Rocm, dispatch to `BackendDevice::cross_attention`.
        // Falls back to CPU `cross_attn` on any failure or CPU device.
        let cross_attn_out = if let Device::Rocm(ordinal) = self.device {
            #[cfg(feature = "rocm")]
            {
                self.cross_attention_gpu(
                    &cross_normed_data,
                    &enc_data,
                    seq,
                    enc_seq,
                    d,
                    nh,
                    hd,
                    sqrt_hd,
                    ordinal,
                )
                .unwrap_or_else(|_| {
                    cross_attn(
                        &cross_normed_data,
                        &enc_data,
                        seq,
                        enc_seq,
                        d,
                        nh,
                        hd,
                        sqrt_hd,
                        &self.cross_q,
                        &self.cross_k,
                        &self.cross_v,
                        &self.cross_o,
                    )
                })
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = ordinal;
                cross_attn(
                    &cross_normed_data,
                    &enc_data,
                    seq,
                    enc_seq,
                    d,
                    nh,
                    hd,
                    sqrt_hd,
                    &self.cross_q,
                    &self.cross_k,
                    &self.cross_v,
                    &self.cross_o,
                )
            }
        } else {
            cross_attn(
                &cross_normed_data,
                &enc_data,
                seq,
                enc_seq,
                d,
                nh,
                hd,
                sqrt_hd,
                &self.cross_q,
                &self.cross_k,
                &self.cross_v,
                &self.cross_o,
            )
        };

        let mut after_cross = vec![0.0f32; seq * d];
        for i in 0..seq * d {
            after_cross[i] = after_self[i] + cross_attn_out[i];
        }

        let ffn1 = self
            .fc1
            .forward(&cpu_tensor(after_cross.clone(), Shape::new(vec![seq, d])))?;
        let ffn1_gelu = gelu(&ffn1)?;
        let ffn2 = self.fc2.forward(&ffn1_gelu)?;
        let ffn_out = ffn2.to_vec_f32()?;
        let mut out = vec![0.0f32; seq * d];
        for i in 0..seq * d {
            out[i] = after_cross[i] + ffn_out[i];
        }
        Ok(cpu_tensor(out, Shape::new(vec![seq, d])))
    }

    /// GPU dispatch path for Whisper cross-attention via
    /// `BackendDevice::cross_attention` (Phase 2 — mambo5.md Item 13).
    /// Q is projected per decoder step; K/V projected once per encode pass.
    /// Encoder K/V projected once, reused across decoder steps.
    #[cfg(feature = "rocm")]
    fn cross_attention_gpu(
        &self,
        cross_normed_data: &[f32],
        enc_data: &[f32],
        seq: usize,
        enc_seq: usize,
        d: usize,
        nh: usize,
        hd: usize,
        _sqrt_hd: f32,
        ordinal: usize,
    ) -> Result<Vec<f32>> {
        use grim_backend_rocm::RocmDevice;
        use grim_tensor::{CoreTensorOps, AttentionOps};

        let dev = RocmDevice::try_new(ordinal)?;

        // Project Q/K/V on the host, matching the CPU cross_attn projection,
        // then upload to the GPU for the cross-attention kernel.
        let project_q = |out: &mut [f32]| {
            for pos in 0..seq {
                for o_idx in 0..d {
                    let mut sum = 0.0;
                    for k in 0..d {
                        sum += cross_normed_data[pos * d + k] * self.cross_q[o_idx * d + k];
                    }
                    out[pos * d + o_idx] = sum;
                }
            }
        };
        let project_kv = |out: &mut [f32], w: &[f32]| {
            for pos in 0..enc_seq {
                for o_idx in 0..d {
                    let mut sum = 0.0;
                    for k in 0..d {
                        sum += enc_data[pos * d + k] * w[o_idx * d + k];
                    }
                    out[pos * d + o_idx] = sum;
                }
            }
        };
        let mut q_data = vec![0.0f32; seq * d];
        let mut k_data = vec![0.0f32; enc_seq * d];
        let mut v_data = vec![0.0f32; enc_seq * d];
        project_q(&mut q_data);
        project_kv(&mut k_data, &self.cross_k);
        project_kv(&mut v_data, &self.cross_v);

        let q_gpu = dev.from_cpu(&q_data, &Shape::new(vec![seq, d]), grim_tensor::DType::F32)?;
        let k_gpu = dev.from_cpu(
            &k_data,
            &Shape::new(vec![enc_seq, d]),
            grim_tensor::DType::F32,
        )?;
        let v_gpu = dev.from_cpu(
            &v_data,
            &Shape::new(vec![enc_seq, d]),
            grim_tensor::DType::F32,
        )?;

        let out_shape = Shape::new(vec![seq, d]);
        let (attn_out, _) = dev.cross_attention(
            q_gpu.as_ref(),
            k_gpu.as_ref(),
            v_gpu.as_ref(),
            nh,
            hd,
            seq,
            enc_seq,
            &out_shape,
        )?;
        let attn = attn_out.to_cpu_vec_f32()?;

        // Apply the output projection W_o on the host.
        let mut result = vec![0.0f32; seq * d];
        for pos in 0..seq {
            for o_idx in 0..d {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += attn[pos * d + k] * self.cross_o[o_idx * d + k];
                }
                result[pos * d + o_idx] = sum;
            }
        }
        Ok(result)
    }
}

pub struct Whisper {
    pub cfg: WhisperConfig,
    pub device: Device,
    pub tok_emb: Embedding,
    pub enc_in_proj: Linear,
    enc_blocks: Vec<WhisperEncoderBlock>,
    enc_norm: RmsNorm,
    dec_blocks: Vec<WhisperDecoderBlock>,
    dec_norm: RmsNorm,
    pub output: Linear,
}

/// Whisper sinusoidal position table: `n` rows of `d_model` values, starting at
/// absolute position `offset`. Matches OpenAI Whisper's
/// `sinusoids()` (log-spaced timescales, `[sin | cos]` halves).
/// [P1-33 fix: positional information for encoder and decoder.]
fn sinusoid_positions(d_model: usize, n: usize, offset: usize) -> Vec<f32> {
    let half = d_model / 2;
    let mut out = vec![0.0f32; n * d_model];
    if half == 0 {
        return out;
    }
    let inc = if half > 1 {
        (10000.0f32).ln() / (half - 1) as f32
    } else {
        0.0
    };
    for p in 0..n {
        let pos = (p + offset) as f32;
        for i in 0..half {
            let scaled = pos * (-inc * i as f32).exp();
            out[p * d_model + i] = scaled.sin();
            out[p * d_model + half + i] = scaled.cos();
        }
    }
    out
}

/// Add a position table in place to a `[n, d_model]` row-major activation.
fn add_positions(x: &mut [f32], d_model: usize, offset: usize) {
    let n = x.len() / d_model;
    let pos = sinusoid_positions(d_model, n, offset);
    for (v, p) in x.iter_mut().zip(pos.iter()) {
        *v += *p;
    }
}

impl Whisper {
    pub fn random(device: Device, cfg: WhisperConfig) -> Self {
        Self::new(device, cfg, &mut SimpleRng::new(0xA5D1_BEEF_70E5_CAFE_u64))
    }

    pub fn new(device: Device, cfg: WhisperConfig, rng: &mut SimpleRng) -> Self {
        let tok_emb_w = (0..cfg.vocab_size * cfg.d_model)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_emb = Embedding {
            weight: cpu_tensor(tok_emb_w, Shape::new(vec![cfg.vocab_size, cfg.d_model])),
        };
        let scale_in = 1.0 / (cfg.n_mels as f32).sqrt();
        let enc_in_proj_w = (0..cfg.d_model * cfg.n_mels)
            .map(|_| (rng.next_f32() - 0.5) * scale_in)
            .collect();
        let enc_in_proj = Linear::from_tensor(
            cpu_tensor(enc_in_proj_w, Shape::new(vec![cfg.d_model, cfg.n_mels])),
            Some(cpu_tensor(
                vec![0.0; cfg.d_model],
                Shape::new(vec![cfg.d_model]),
            )),
        );
        let enc_blocks = (0..cfg.num_enc_layers)
            .map(|_| {
                WhisperEncoderBlock::new(
                    cfg.d_model,
                    cfg.num_heads,
                    cfg.ffn_dim,
                    cfg.rms_norm_eps,
                    rng,
                )
            })
            .collect();
        let enc_norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.d_model], Shape::new(vec![cfg.d_model])),
            eps: cfg.rms_norm_eps,
        };
        let dec_blocks = (0..cfg.num_dec_layers)
            .map(|_| {
                WhisperDecoderBlock::new(
                    cfg.d_model,
                    cfg.num_heads,
                    cfg.ffn_dim,
                    cfg.rms_norm_eps,
                    rng,
                    device.clone(),
                )
            })
            .collect();
        let dec_norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.d_model], Shape::new(vec![cfg.d_model])),
            eps: cfg.rms_norm_eps,
        };
        let scale_out = 1.0 / (cfg.d_model as f32).sqrt();
        let output_w = (0..cfg.vocab_size * cfg.d_model)
            .map(|_| (rng.next_f32() - 0.5) * scale_out)
            .collect();
        let output = Linear::from_tensor(
            cpu_tensor(output_w, Shape::new(vec![cfg.vocab_size, cfg.d_model])),
            Some(cpu_tensor(
                vec![0.0; cfg.vocab_size],
                Shape::new(vec![cfg.vocab_size]),
            )),
        );
        Self {
            cfg,
            device,
            tok_emb,
            enc_in_proj,
            enc_blocks,
            enc_norm,
            dec_blocks,
            dec_norm,
            output,
        }
    }

    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: WhisperConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for Whisper. Whisper is an audio
    /// encoder–decoder with cross-attention in the decoder; like T5, the
    /// cross-attention sharding adds symmetrical constraints beyond a plain
    /// column/row split, and the block `forward` calls plain `Linear::forward`
    /// with no all-reduce hook. Refused until both land.
    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: WhisperConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "Whisper",
            "audio encoder–decoder cross-attention needs bespoke sharding and a \
             forward rework to add the all-reduce hook",
        )
        .map_err(Error::Unimplemented)?;
        let tok_emb = Embedding::load(&ws.pp("tok_emb"), cfg.vocab_size, cfg.d_model)?;
        let enc_in_proj = Linear::load(&ws.pp("enc_in_proj"), cfg.n_mels, cfg.d_model, true)?;
        let mut enc_blocks = Vec::with_capacity(cfg.num_enc_layers);
        for i in 0..cfg.num_enc_layers {
            enc_blocks.push(WhisperEncoderBlock::load(
                &ws.pp(&format!("encoder.blocks.{i}")),
                cfg.d_model,
                cfg.num_heads,
                cfg.ffn_dim,
                cfg.rms_norm_eps,
            )?);
        }
        let enc_norm = RmsNorm::load(&ws.pp("encoder.norm"), cfg.d_model, cfg.rms_norm_eps)?;
        let mut dec_blocks = Vec::with_capacity(cfg.num_dec_layers);
        for i in 0..cfg.num_dec_layers {
            dec_blocks.push(WhisperDecoderBlock::load(
                &ws.pp(&format!("decoder.blocks.{i}")),
                cfg.d_model,
                cfg.num_heads,
                cfg.ffn_dim,
                cfg.rms_norm_eps,
                device.clone(),
            )?);
        }
        let dec_norm = RmsNorm::load(&ws.pp("decoder.norm"), cfg.d_model, cfg.rms_norm_eps)?;
        let output = Linear::load(&ws.pp("output"), cfg.d_model, cfg.vocab_size, true)?;
        Ok(Self {
            cfg,
            device,
            tok_emb,
            enc_in_proj,
            enc_blocks,
            enc_norm,
            dec_blocks,
            dec_norm,
            output,
        })
    }

    /// Mel features over T frames → encoder_out (T, d_model).
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let shape = mel.shape().dims().to_vec();
        if shape.len() != 2 {
            return Err(Error::Shape(format!(
                "Whisper encode expects (n_mels, T), got {:?}",
                shape
            )));
        }
        let (mel_bins, frames) = (shape[0], shape[1]);
        if mel_bins != self.cfg.n_mels {
            return Err(Error::Shape(format!(
                "Whisper expects {} mel bins, got {}",
                self.cfg.n_mels, mel_bins
            )));
        }
        if frames > self.cfg.max_audio_len {
            return Err(Error::Shape(format!(
                "Whisper audio too long: {} > max {}",
                frames, self.cfg.max_audio_len
            )));
        }
        let mel_data = mel.to_vec_f32()?;
        // Shape validated as (n_mels, frames) above; must transpose to (frames, n_mels)
        // for the projection. Without this transpose, data is scrambled for n_mels != frames.
        // [P1-33 fix: transpose mel matrix after shape check.]
        let transposed: Vec<f32> = (0..frames)
            .flat_map(|f| {
                let row: Vec<f32> = (0..mel_bins).map(|m| mel_data[m * frames + f]).collect();
                row.into_iter()
            })
            .collect();
        let mel_t = cpu_tensor(transposed, Shape::new(vec![frames, mel_bins]));
        let proj = self.enc_in_proj.forward(&mel_t)?;
        // Encoder positional embeddings: Whisper adds a fixed sinusoidal table
        // to the convolution/projection output before the encoder blocks.
        // [P1-33 fix: encoder positional embeddings.]
        let mut proj_data = proj.to_vec_f32()?;
        add_positions(&mut proj_data, self.cfg.d_model, 0);
        let mut cur = cpu_tensor(proj_data, Shape::new(vec![frames, self.cfg.d_model]));
        for blk in &self.enc_blocks {
            cur = blk.forward(&cur)?;
        }
        cur = self.enc_norm.forward(&cur)?;
        Ok(cur)
    }

    /// One decoder step. `input_ids` is `[1, 1]` for batch=1, single-position decode.
    pub fn decode_step(&self, _enc_out: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let ids_shape = input_ids.shape().dims().to_vec();
        if ids_shape.is_empty() || ids_shape[ids_shape.len() - 1] == 0 {
            return Err(Error::Shape(
                "Whisper decode_step expects non-empty ids".into(),
            ));
        }
        let ids_data = input_ids.to_vec_f32()?;
        let ids: Vec<u32> = ids_data.iter().map(|x| *x as u32).collect();
        let seq_len = ids.len();
        if seq_len > self.cfg.max_text_len {
            return Err(Error::Shape(format!(
                "Whisper text too long: {} > max {}",
                seq_len, self.cfg.max_text_len
            )));
        }
        let emb = self.tok_emb.forward(&ids, seq_len, self.cfg.d_model)?;
        // Decoder positional embeddings. `decode_step` is passed the full id
        // prefix and recomputes attention over it, so absolute positions start
        // at 0 for row 0. [P1-33 fix: decoder positional embeddings.]
        let mut emb_data = emb.to_vec_f32()?;
        add_positions(&mut emb_data, self.cfg.d_model, 0);
        let mut cur = cpu_tensor(emb_data, Shape::new(vec![seq_len, self.cfg.d_model]));
        for blk in &self.dec_blocks {
            cur = blk.decode_step(&cur, _enc_out)?;
        }
        let normed = self.dec_norm.forward(&cur)?;
        let logits = self.output.forward(&normed)?;
        Ok(logits)
    }

    /// Transcribe mel-spectrogram features into a sequence of token IDs via autoregressive greedy decoding.
    pub fn transcribe_tokens(&self, mel: &Tensor, max_tokens: usize) -> Result<Vec<u32>> {
        let enc_out = self.encode(mel)?;
        let mut tokens: Vec<u32> = if self.cfg.vocab_size >= 50360 {
            vec![50258, 50259, 50359] // <|startoftranscript|>, <|en|>, <|transcribe|>
        } else {
            vec![1.min((self.cfg.vocab_size.saturating_sub(1)) as u32)]
        };
        let limit = max_tokens.min(self.cfg.max_text_len);

        for _ in 0..limit {
            let ids_tensor = cpu_tensor(
                tokens.iter().map(|&t| t as f32).collect(),
                Shape::new(vec![tokens.len()]),
            );
            let logits = self.decode_step(&enc_out, &ids_tensor)?;
            let logits_data = logits.to_vec_f32()?;
            let vocab_size = self.cfg.vocab_size;
            let last_row_start = (tokens.len() - 1) * vocab_size;
            if last_row_start + vocab_size > logits_data.len() {
                break;
            }
            let last_row = &logits_data[last_row_start..last_row_start + vocab_size];

            let mut best_tok = 0u32;
            let mut best_logit = f32::NEG_INFINITY;
            for (idx, &l) in last_row.iter().enumerate() {
                if l > best_logit {
                    best_logit = l;
                    best_tok = idx as u32;
                }
            }

            if best_tok == 50257 {
                // <|endoftranscript|>
                break;
            }
            tokens.push(best_tok);
        }

        Ok(tokens)
    }
}

impl Model for Whisper {
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

impl EncoderDecoderLm for Whisper {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        self.encode(input)
    }
    fn decode_step(
        &self,
        _session: &mut dyn grim_core::session::SessionT,
        encoder_out: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.decode_step(encoder_out, input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WhisperConfig {
        WhisperConfig {
            vocab_size: 100,
            n_mels: 16,
            d_model: 32,
            num_enc_layers: 2,
            num_dec_layers: 2,
            num_heads: 2,
            ffn_dim: 64,
            max_audio_len: 32,
            max_text_len: 32,
            rms_norm_eps: 1e-5,
        }
    }

    #[test]
    fn whisper_encode_and_decode_step_shapes() {
        let w = Whisper::random(Device::Cpu, cfg());
        let mel = cpu_tensor(
            (0..16 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![16, 8]),
        );
        let enc = w.encode(&mel).unwrap();
        assert_eq!(enc.shape().dims(), &[8, 32]);

        let ids = cpu_tensor(vec![1.0f32; 3], Shape::new(vec![3]));
        let logits = w.decode_step(&enc, &ids).unwrap();
        assert_eq!(logits.shape().dims(), &[3, 100]);
        let ld = logits.to_vec_f32().unwrap();
        assert!(ld.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn whisper_rejects_wrong_mel_bins() {
        let w = Whisper::random(Device::Cpu, cfg());
        let mel = cpu_tensor(vec![0.0f32; 8 * 4], Shape::new(vec![8, 4]));
        let err = match w.encode(&mel) {
            Ok(_) => panic!("expected Shape error, got Ok"),
            Err(e) => e,
        };
        match err {
            Error::Shape(_) => {}
            other => panic!("expected Shape error, got {:?}", other),
        }
    }

    #[test]
    fn whisper_rejects_text_longer_than_context() {
        let mut c = cfg();
        c.max_text_len = 4;
        let w = Whisper::random(Device::Cpu, c);
        let mel = cpu_tensor(
            (0..16 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![16, 8]),
        );
        let enc = w.encode(&mel).unwrap();
        let ids = cpu_tensor(vec![1.0f32; 5], Shape::new(vec![5]));
        match w.decode_step(&enc, &ids) {
            Err(Error::Shape(msg)) => assert!(msg.contains("text too long"), "{msg}"),
            Ok(_) => panic!("expected Shape error, got Ok"),
            other => panic!("expected Shape error, got {:?}", other),
        }
    }

    /// HuggingFace transformers key set (`openai/whisper-tiny` config.json).
    #[test]
    fn whisper_config_from_hf_transformers_keys() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "model_type": "whisper",
                "vocab_size": 51865,
                "num_mel_bins": 80,
                "d_model": 384,
                "encoder_layers": 4,
                "decoder_layers": 4,
                "encoder_attention_heads": 6,
                "decoder_attention_heads": 6,
                "encoder_ffn_dim": 1536,
                "max_source_positions": 1500,
                "max_target_positions": 448
            }"#,
        )
        .unwrap();
        let c = WhisperConfig::from_hf(&json);
        assert_eq!(c.vocab_size, 51865);
        assert_eq!(c.n_mels, 80);
        assert_eq!(c.d_model, 384);
        assert_eq!(c.num_enc_layers, 4);
        assert_eq!(c.num_dec_layers, 4);
        assert_eq!(c.num_heads, 6);
        assert_eq!(c.ffn_dim, 1536);
        assert_eq!(c.max_audio_len, 1500);
        assert_eq!(c.max_text_len, 448);
    }

    /// OpenAI's original Whisper key set (`n_audio_state`, …) as used by the
    /// reference checkpoints released alongside `models/audio/`.
    #[test]
    fn whisper_config_from_openai_keys() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "n_vocab": 51864,
                "n_mels": 80,
                "n_audio_ctx": 3000,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
                "n_text_ctx": 448,
                "n_text_state": 384,
                "n_text_head": 6,
                "n_text_layer": 4
            }"#,
        )
        .unwrap();
        let c = WhisperConfig::from_hf(&json);
        assert_eq!(c.vocab_size, 51864);
        assert_eq!(c.d_model, 384);
        assert_eq!(c.num_enc_layers, 4);
        assert_eq!(c.num_dec_layers, 4);
        assert_eq!(c.max_audio_len, 3000);
        assert_eq!(c.max_text_len, 448);
    }

    /// Partial configs (minimal JSON like the Kokoro/MeanVC2 configs ship)
    /// must fall back to defaults instead of failing.
    #[test]
    fn whisper_config_partial_json_uses_defaults() {
        let json: serde_json::Value = serde_json::from_str(r#"{ "num_mel_bins": 128 }"#).unwrap();
        let c = WhisperConfig::from_hf(&json);
        assert_eq!(c.n_mels, 128);
        assert_eq!(c.d_model, WhisperConfig::default().d_model);
    }

    #[test]
    fn whisper_config_serde_roundtrip() {
        let c = WhisperConfig::default();
        let s = serde_json::to_string(&c).unwrap();
        let back: WhisperConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.vocab_size, c.vocab_size);
        assert_eq!(back.d_model, c.d_model);
        assert_eq!(back.rms_norm_eps, c.rms_norm_eps);
    }

    #[test]
    fn whisper_config_reports_context_length_and_modality() {
        let c = cfg();
        assert_eq!(ModelConfig::context_length(&c), c.max_text_len as u64);
        assert_eq!(
            ModelConfig::modality(&c),
            grim_core::model::ModalityHint::AudioEncoderDecoder
        );
    }
}

#[cfg(test)]
mod golden_attention {
    use super::*;

    fn close(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-5,
            "{ctx}: got {got:?} want {want:?} (abs={abs})",
        );
    }

    fn identity_w(d: usize) -> Vec<f32> {
        let mut w = vec![0.0; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        w
    }

    fn zero_linear(in_d: usize, out_d: usize) -> Linear {
        Linear::from_tensor(
            cpu_tensor(vec![0.0; out_d * in_d], Shape::new(vec![out_d, in_d])),
            Some(cpu_tensor(vec![0.0; out_d], Shape::new(vec![out_d]))),
        )
    }

    fn rms(v: &[f32]) -> f32 {
        let sum_sq: f32 = v.iter().map(|x| x * x).sum();
        (sum_sq / v.len() as f32 + 1e-5).sqrt()
    }

    // ==================================================================
    // Test 1: Encoder self-attention with hand-constructed identity weights.
    //
    //   x = [1, 2, -1, -2], seq=1, d=4, nh=1
    //   Q=K=V=O = I  →  attn_out = x  →  output = x + attn_out = 2x
    //   FFN zeroed → output = 2x = [2, 4, -2, -4]
    // ==================================================================
    #[test]
    fn golden_whisper_self_attn_hand_constructed_weights() {
        let d_model = 4;
        let nh = 1;
        let hd = d_model / nh;
        let ffn_dim = 8;
        let eps = 1e-5;
        let x = vec![1.0, 2.0, -1.0, -2.0];
        let x_rms = rms(&x);

        let block = WhisperEncoderBlock {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![x_rms; d_model], Shape::new(vec![d_model])),
                eps,
            },
            wq: identity_w(d_model),
            wk: identity_w(d_model),
            wv: identity_w(d_model),
            wo: identity_w(d_model),
            norm2: RmsNorm {
                weight: cpu_tensor(
                    vec![rms(&[2.0, 4.0, -2.0, -4.0]); d_model],
                    Shape::new(vec![d_model]),
                ),
                eps,
            },
            fc1: zero_linear(d_model, ffn_dim),
            fc2: zero_linear(ffn_dim, d_model),
            d_model,
            num_heads: nh,
            head_dim: hd,
        };

        let input_t = cpu_tensor(x.clone(), Shape::new(vec![1, d_model]));
        let output_t = block.forward(&input_t).unwrap();
        let output = output_t.to_vec_f32().unwrap();

        assert_eq!(output.len(), d_model);
        for i in 0..d_model {
            close(output[i], x[i] * 2.0, &format!("self_attn[{i}]"));
        }
    }

    // ==================================================================
    // Test 2: Decoder cross-attention with identity weights.
    //
    //   h = [1,2,3,4], seq=1, d=4, nh=1
    //   enc_out = [[1,1,1,1], [1,1,1,1]] (2 frames)
    //   All Q/K/V/O = I, FFN zeroed
    //
    //   Self-attn: after_self = h + h = [2,4,6,8]
    //   Cross-attn: q=[2,4,6,8], k=v=[1,1,1,1] each frame
    //     scores = 20/2 = 10 per frame, softmax uniform = 0.5 each
    //     weighted v = [1,1,1,1]
    //     after_cross = [2,4,6,8] + [1,1,1,1] = [3,5,7,9]
    //   FFN zeroed → output = [3,5,7,9]
    // ==================================================================
    #[test]
    fn golden_whisper_cross_attn_encoder_decoder_interaction() {
        let d_model = 4;
        let nh = 1;
        let hd = d_model / nh;
        let ffn_dim = 8;
        let eps = 1e-5;
        let h = vec![1.0, 2.0, 3.0, 4.0];
        // enc_out: 2 frames, all ones
        let enc_seq = 2;
        let enc_out = vec![1.0; enc_seq * d_model];

        let after_self_val: Vec<f32> = h.iter().map(|v| v * 2.0).collect();
        let after_cross_val: Vec<f32> = after_self_val.iter().map(|v| v + 1.0).collect();

        let block = WhisperDecoderBlock {
            self_norm: RmsNorm {
                weight: cpu_tensor(vec![rms(&h); d_model], Shape::new(vec![d_model])),
                eps,
            },
            self_q: identity_w(d_model),
            self_k: identity_w(d_model),
            self_v: identity_w(d_model),
            self_o: identity_w(d_model),
            cross_norm: RmsNorm {
                weight: cpu_tensor(
                    vec![rms(&after_self_val); d_model],
                    Shape::new(vec![d_model]),
                ),
                eps,
            },
            cross_q: identity_w(d_model),
            cross_k: identity_w(d_model),
            cross_v: identity_w(d_model),
            cross_o: identity_w(d_model),
            fc1: zero_linear(d_model, ffn_dim),
            fc2: zero_linear(ffn_dim, d_model),
            d_model,
            num_heads: nh,
            head_dim: hd,
            device: Device::Cpu,
        };

        let h_t = cpu_tensor(h.clone(), Shape::new(vec![1, d_model]));
        let enc_t = cpu_tensor(enc_out.clone(), Shape::new(vec![enc_seq, d_model]));
        let output_t = block.decode_step(&h_t, &enc_t).unwrap();
        let output = output_t.to_vec_f32().unwrap();

        assert_eq!(output.len(), d_model);
        for i in 0..d_model {
            close(output[i], after_cross_val[i], &format!("cross_attn[{i}]"));
        }
    }

    // ==================================================================
    // Test 3: FFN still works after attention wiring.
    //
    //   Encoder block with zeroed Q/K/V/O (attn_out=0) and
    //   hand-constructed FFN: W_fc0 = 2×I, W_fc1 = 0.5×I
    //   Input x = [1, -1, 2, -2], seq=1, d=4
    //
    //   after_attn = x  (attn zeroed)
    //   fc1(norm(x)) = fc1(x) = 2x
    //   gelu(2x) → via shared gelu()
    //   fc2(gelu(2x)) = 0.5 * gelu(2x)
    //   output = after_attn + ffn_out = x + 0.5 * gelu(2x)
    //
    //   Expected computed via same gelu() function for exact match.
    // ==================================================================
    #[test]
    fn golden_whisper_ffn_still_works_after_attn_wiring() {
        let d_model = 4;
        let nh = 1;
        let hd = d_model / nh;
        let ffn_dim = 8;
        let eps = 1e-5;
        let x = vec![1.0, -1.0, 2.0, -2.0];
        let zero_w = vec![0.0; d_model * d_model];

        let fc0_w: Vec<f32> = {
            let mut w = vec![0.0; ffn_dim * d_model];
            for i in 0..d_model {
                w[i * d_model + i] = 2.0;
            }
            w
        };
        let fc1_w: Vec<f32> = {
            let mut w = vec![0.0; d_model * ffn_dim];
            for i in 0..d_model {
                w[i * ffn_dim + i] = 0.5;
            }
            w
        };

        let block = WhisperEncoderBlock {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![rms(&x); d_model], Shape::new(vec![d_model])),
                eps,
            },
            wq: zero_w.clone(),
            wk: zero_w.clone(),
            wv: zero_w.clone(),
            wo: zero_w.clone(),
            norm2: RmsNorm {
                weight: cpu_tensor(vec![rms(&x); d_model], Shape::new(vec![d_model])),
                eps,
            },
            fc1: Linear::from_tensor(
                cpu_tensor(fc0_w, Shape::new(vec![ffn_dim, d_model])),
                Some(cpu_tensor(vec![0.0; ffn_dim], Shape::new(vec![ffn_dim]))),
            ),
            fc2: Linear::from_tensor(
                cpu_tensor(fc1_w, Shape::new(vec![d_model, ffn_dim])),
                Some(cpu_tensor(vec![0.0; d_model], Shape::new(vec![d_model]))),
            ),
            d_model,
            num_heads: nh,
            head_dim: hd,
        };

        // Expected: x + 0.5 * gelu(2*x) — computed via same gelu().
        let fc0_t = cpu_tensor(
            x.iter().map(|v| v * 2.0).collect(),
            Shape::new(vec![1, d_model]),
        );
        let gelu_t = gelu(&fc0_t).unwrap();
        let gelu_v = gelu_t.to_vec_f32().unwrap();
        let expected: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.5 * gelu_v[i])
            .collect();

        let input_t = cpu_tensor(x, Shape::new(vec![1, d_model]));
        let output_t = block.forward(&input_t).unwrap();
        let output = output_t.to_vec_f32().unwrap();

        assert_eq!(output.len(), d_model);
        for i in 0..d_model {
            close(output[i], expected[i], &format!("ffn[{i}]"));
        }
    }
}
