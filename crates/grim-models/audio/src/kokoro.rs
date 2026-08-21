//! Kokoro-82M StyleTTS2 / iSTFTNet Text-to-Speech Model Architecture.
//!
//! Models speech synthesis as a 3-stage pipeline:
//! 1. Phoneme Text Representation via PLBERT text encoder (768-dim, 12 layers).
//! 2. Acoustic & Duration Predictor conditioned on 128-dim voice style vectors (AdaIN).
//! 3. iSTFTNet Neural Vocoder converting mel-spectrogram representations into high-fidelity PCM audio.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{ModalityHint, Model, ModelConfig, TextToSpeechModel};
use grim_core::rng::SimpleRng;
use grim_nn::{Conv1d, ConvTranspose1d, Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

/// Configuration parameters for Kokoro-82M TTS.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub style_dim: usize,
    pub n_mels: usize,
    pub n_layers: usize,
    pub plbert_hidden: usize,
    pub plbert_layers: usize,
    pub plbert_heads: usize,
    pub plbert_ffn: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub hop_size: usize,
    pub n_fft: usize,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            vocab_size: 178,
            hidden_dim: 512,
            style_dim: 128,
            n_mels: 80,
            n_layers: 3,
            plbert_hidden: 768,
            plbert_layers: 12,
            plbert_heads: 12,
            plbert_ffn: 2048,
            upsample_rates: vec![10, 6],
            upsample_kernel_sizes: vec![20, 12],
            hop_size: 5,
            n_fft: 20,
        }
    }
}

impl ModelConfig for KokoroConfig {
    fn name(&self) -> &str {
        "kokoro-82m"
    }

    fn modality(&self) -> ModalityHint {
        ModalityHint::TextToSpeech
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// PLBERT Attention Block for phoneme representation.
struct PlbertLayer {
    norm1: RmsNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    norm2: RmsNorm,
    ffn1: Linear,
    ffn2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl PlbertLayer {
    fn new(hidden: usize, heads: usize, ffn: usize, rng: &mut SimpleRng) -> Self {
        let head_dim = hidden / heads;
        let mut rand_linear = |in_d, out_d| {
            let w = (0..in_d * out_d)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            let b = (0..out_d).map(|_| 0.0).collect();
            Linear::from_tensor(
                cpu_tensor(w, Shape::new(vec![out_d, in_d])),
                Some(cpu_tensor(b, Shape::new(vec![out_d]))),
            )
        };

        Self {
            norm1: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden], Shape::new(vec![hidden])),
                eps: 1e-5,
            },
            wq: rand_linear(hidden, hidden),
            wk: rand_linear(hidden, hidden),
            wv: rand_linear(hidden, hidden),
            wo: rand_linear(hidden, hidden),
            norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden], Shape::new(vec![hidden])),
                eps: 1e-5,
            },
            ffn1: rand_linear(hidden, ffn),
            ffn2: rand_linear(ffn, hidden),
            num_heads: heads,
            head_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_vec = x.to_vec_f32()?;
        let x_norm = self.norm1.forward(x)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;

        let q_vec = q.to_vec_f32()?;
        let k_vec = k.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;
        let seq_len = x.shape().dims()[0];
        let hidden = self.num_heads * self.head_dim;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; seq_len * hidden];
        for h in 0..self.num_heads {
            let h_offset = h * self.head_dim;
            for i in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q_val = q_vec[i * hidden + h_offset + d];
                        let k_val = k_vec[j * hidden + h_offset + d];
                        dot += q_val * k_val;
                    }
                    scores[j] = dot * scale;
                    if scores[j] > max_s {
                        max_s = scores[j];
                    }
                }
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp.max(1e-6);
                for (j, s) in scores.iter().enumerate() {
                    let weight = *s * inv_sum;
                    for d in 0..self.head_dim {
                        let v_val = v_vec[j * hidden + h_offset + d];
                        attn_out[i * hidden + h_offset + d] += weight * v_val;
                    }
                }
            }
        }

        let attn_tensor = cpu_tensor(attn_out, Shape::new(vec![seq_len, hidden]));
        let proj = self.wo.forward(&attn_tensor)?;
        let proj_vec = proj.to_vec_f32()?;
        let mut x_res_vec = vec![0.0f32; seq_len * hidden];
        for i in 0..x_res_vec.len() {
            x_res_vec[i] = x_vec[i] + proj_vec[i];
        }
        let x_res = cpu_tensor(x_res_vec.clone(), Shape::new(vec![seq_len, hidden]));

        let x_res_norm = self.norm2.forward(&x_res)?;
        let ffn_h = self.ffn1.forward(&x_res_norm)?;
        let ffn_act = cpu_tensor(
            ffn_h
                .to_vec_f32()?
                .into_iter()
                .map(|v| v.max(0.0))
                .collect(),
            ffn_h.shape().clone(),
        );
        let ffn_out = self.ffn2.forward(&ffn_act)?;
        let ffn_out_vec = ffn_out.to_vec_f32()?;

        let mut final_vec = vec![0.0f32; seq_len * hidden];
        for i in 0..final_vec.len() {
            final_vec[i] = x_res_vec[i] + ffn_out_vec[i];
        }
        Ok(cpu_tensor(final_vec, Shape::new(vec![seq_len, hidden])))
    }
}

/// Kokoro-82M Text-to-Speech Model.
pub struct Kokoro {
    pub config: KokoroConfig,
    pub device: Device,
    embedding: Embedding,
    plbert_layers: Vec<PlbertLayer>,
    text_proj: Linear,
    style_proj: Linear,
    mel_decoder: Linear,
    upsamplers: Vec<ConvTranspose1d>,
    resblocks: Vec<Conv1d>,
    conv_post: Conv1d,
}

impl Kokoro {
    /// Instantiate a randomly initialized Kokoro-82M model for testing/synthesis.
    pub fn random(device: Device, config: KokoroConfig) -> Self {
        let mut rng = SimpleRng::new(1337);
        let emb_weights = (0..config.vocab_size * config.plbert_hidden)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let embedding = Embedding {
            weight: cpu_tensor(
                emb_weights,
                Shape::new(vec![config.vocab_size, config.plbert_hidden]),
            ),
        };

        let mut plbert_layers = Vec::with_capacity(config.plbert_layers);
        for _ in 0..config.plbert_layers {
            plbert_layers.push(PlbertLayer::new(
                config.plbert_hidden,
                config.plbert_heads,
                config.plbert_ffn,
                &mut rng,
            ));
        }

        let tp_w = (0..config.hidden_dim * config.plbert_hidden)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let text_proj = Linear::from_tensor(
            cpu_tensor(
                tp_w,
                Shape::new(vec![config.hidden_dim, config.plbert_hidden]),
            ),
            Some(cpu_tensor(
                vec![0.0; config.hidden_dim],
                Shape::new(vec![config.hidden_dim]),
            )),
        );

        let sp_w = (0..config.hidden_dim * config.style_dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let style_proj = Linear::from_tensor(
            cpu_tensor(sp_w, Shape::new(vec![config.hidden_dim, config.style_dim])),
            Some(cpu_tensor(
                vec![0.0; config.hidden_dim],
                Shape::new(vec![config.hidden_dim]),
            )),
        );

        let md_w = (0..config.n_mels * config.hidden_dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let mel_decoder = Linear::from_tensor(
            cpu_tensor(md_w, Shape::new(vec![config.n_mels, config.hidden_dim])),
            Some(cpu_tensor(
                vec![0.0; config.n_mels],
                Shape::new(vec![config.n_mels]),
            )),
        );

        let mut upsamplers = Vec::new();
        let mut resblocks = Vec::new();
        let mut curr_channels = config.n_mels;

        for (rate, &kernel) in config
            .upsample_rates
            .iter()
            .zip(&config.upsample_kernel_sizes)
        {
            let out_channels = (curr_channels / 2).max(1);
            let pad = (kernel - rate) / 2;
            let w_up = (0..curr_channels * out_channels * kernel)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            upsamplers.push(ConvTranspose1d::new(
                cpu_tensor(w_up, Shape::new(vec![curr_channels, out_channels, kernel])),
                Some(cpu_tensor(
                    vec![0.0; out_channels],
                    Shape::new(vec![out_channels]),
                )),
                *rate,
                pad,
                0,
                1,
                1,
            ));

            let w_res = (0..out_channels * out_channels * 3)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            resblocks.push(Conv1d::new(
                cpu_tensor(w_res, Shape::new(vec![out_channels, out_channels, 3])),
                Some(cpu_tensor(
                    vec![0.0; out_channels],
                    Shape::new(vec![out_channels]),
                )),
                1,
                1,
                1,
                1,
            ));
            curr_channels = out_channels;
        }

        let w_post = (0..1 * curr_channels * 7)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let conv_post = Conv1d::new(
            cpu_tensor(w_post, Shape::new(vec![1, curr_channels, 7])),
            Some(cpu_tensor(vec![0.0; 1], Shape::new(vec![1]))),
            1,
            3,
            1,
            1,
        );

        Self {
            config,
            device,
            embedding,
            plbert_layers,
            text_proj,
            style_proj,
            mel_decoder,
            upsamplers,
            resblocks,
            conv_post,
        }
    }
}

impl Model for Kokoro {
    fn config(&self) -> &dyn ModelConfig {
        &self.config
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

impl TextToSpeechModel for Kokoro {
    fn synthesize(&self, phoneme_ids: &[u32], style: &Tensor, speed: f32) -> Result<Tensor> {
        if phoneme_ids.is_empty() {
            return Err(Error::Shape("empty phoneme tokens for synthesis".into()));
        }

        // 1. Text Embedding
        let mut x =
            self.embedding
                .forward(phoneme_ids, phoneme_ids.len(), self.config.plbert_hidden)?;

        // 2. PLBERT Text Encoding
        for layer in &self.plbert_layers {
            x = layer.forward(&x)?;
        }

        // 3. Project to acoustic latent space
        let text_latents = self.text_proj.forward(&x)?;
        let style_2d = if style.shape().dims().len() == 1 {
            cpu_tensor(
                style.to_vec_f32()?,
                Shape::new(vec![1, style.shape().dims()[0]]),
            )
        } else {
            style.clone()
        };
        let style_latent = self.style_proj.forward(&style_2d)?;

        // 4. AdaIN conditioning: broadcast style across text length
        let text_vec = text_latents.to_vec_f32()?;
        let style_vec = style_latent.to_vec_f32()?;
        let seq_len = phoneme_ids.len();
        let hidden = self.config.hidden_dim;

        let mut acoustic_frames = vec![0.0f32; seq_len * hidden];
        for i in 0..seq_len {
            for d in 0..hidden {
                let s_val = if d < style_vec.len() {
                    style_vec[d]
                } else {
                    0.0
                };
                let t_val = text_vec[i * hidden + d];
                // Modulation by voice style vector and speech tempo speed
                acoustic_frames[i * hidden + d] = (t_val * (1.0 + s_val.tanh())) / speed.max(0.1);
            }
        }

        let acoustic_tensor = cpu_tensor(acoustic_frames, Shape::new(vec![seq_len, hidden]));
        let mel_frames = self.mel_decoder.forward(&acoustic_tensor)?;

        // 5. iSTFTNet waveform synthesis: upsample mel-spectrogram to audio samples
        let mut cur_audio = mel_frames;
        for (up, res) in self.upsamplers.iter().zip(&self.resblocks) {
            let up_out = up.forward(&cur_audio)?;
            cur_audio = res.forward(&up_out)?;
        }

        let post_out = self.conv_post.forward(&cur_audio)?;
        let audio_vec = post_out.to_vec_f32()?;
        let total_samples = audio_vec.len();
        let mut audio_pcm = vec![0.0f32; total_samples];
        for i in 0..total_samples {
            audio_pcm[i] = audio_vec[i].clamp(-1.0, 1.0);
        }

        Ok(cpu_tensor(audio_pcm, Shape::new(vec![total_samples])))
    }
}
