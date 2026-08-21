//! Vocos Neural Audio Vocoder Architecture.
//!
//! Uses a ConvNeXt backbone with inverse Short-Time Fourier Transform (iSTFT)
//! synthesis head for high-speed, artifact-free audio waveform generation from mel spectrograms.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AudioVocoder, ModalityHint, Model, ModelConfig};
use grim_core::rng::SimpleRng;
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

/// Configuration parameters for Vocos neural vocoder.
#[derive(Debug, Clone)]
pub struct VocosConfig {
    pub input_dim: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub n_fft: usize,
    pub hop_length: usize,
}

impl Default for VocosConfig {
    fn default() -> Self {
        Self {
            input_dim: 80,
            dim: 512,
            intermediate_dim: 1536,
            num_layers: 8,
            n_fft: 1024,
            hop_length: 256,
        }
    }
}

impl ModelConfig for VocosConfig {
    fn name(&self) -> &str {
        "vocos"
    }

    fn modality(&self) -> ModalityHint {
        ModalityHint::AudioVocoder
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// ConvNeXt Block with 7x1 depthwise conv and point-wise expansion.
struct ConvNeXtBlock {
    dw_weight: Vec<f32>,
    norm: RmsNorm,
    pw1: Linear,
    pw2: Linear,
    dim: usize,
}

impl ConvNeXtBlock {
    fn new(dim: usize, intermediate: usize, rng: &mut SimpleRng) -> Self {
        let dw_weight = (0..dim * 7)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let pw1_w = (0..intermediate * dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let pw1 = Linear::from_tensor(
            cpu_tensor(pw1_w, Shape::new(vec![intermediate, dim])),
            Some(cpu_tensor(
                vec![0.0; intermediate],
                Shape::new(vec![intermediate]),
            )),
        );
        let pw2_w = (0..dim * intermediate)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let pw2 = Linear::from_tensor(
            cpu_tensor(pw2_w, Shape::new(vec![dim, intermediate])),
            Some(cpu_tensor(vec![0.0; dim], Shape::new(vec![dim]))),
        );

        Self {
            dw_weight,
            norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
                eps: 1e-6,
            },
            pw1,
            pw2,
            dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_vec = x.to_vec_f32()?;
        let seq_len = x.shape().dims()[0];
        let dim = self.dim;

        // 1. Depthwise 1D conv (kernel size 7, padding 3)
        let mut dw_out = vec![0.0f32; seq_len * dim];
        for i in 0..seq_len {
            for d in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..7 {
                    let in_idx = i as isize + k as isize - 3;
                    if in_idx >= 0 && in_idx < seq_len as isize {
                        let w_val = self.dw_weight[d * 7 + k];
                        let in_val = x_vec[in_idx as usize * dim + d];
                        sum += w_val * in_val;
                    }
                }
                dw_out[i * dim + d] = sum;
            }
        }

        let dw_tensor = cpu_tensor(dw_out, Shape::new(vec![seq_len, dim]));
        let norm_out = self.norm.forward(&dw_tensor)?;

        // 2. Pointwise MLP with GELU activation
        let h = self.pw1.forward(&norm_out)?;
        let h_gelu = cpu_tensor(
            h.to_vec_f32()?
                .into_iter()
                .map(|v| 0.5 * v * (1.0 + (0.79788456 * (v + 0.044715 * v * v * v)).tanh()))
                .collect(),
            h.shape().clone(),
        );
        let out = self.pw2.forward(&h_gelu)?;
        let out_vec = out.to_vec_f32()?;

        let mut res = vec![0.0f32; seq_len * dim];
        for i in 0..res.len() {
            res[i] = x_vec[i] + out_vec[i];
        }
        Ok(cpu_tensor(res, Shape::new(vec![seq_len, dim])))
    }
}

/// Vocos Vocoder Model.
pub struct Vocos {
    pub config: VocosConfig,
    pub device: Device,
    in_proj: Linear,
    blocks: Vec<ConvNeXtBlock>,
    head_mag: Linear,
    head_phase: Linear,
}

impl Vocos {
    /// Instantiate a randomly initialized Vocos model for testing.
    pub fn random(device: Device, config: VocosConfig) -> Self {
        let mut rng = SimpleRng::new(9999);
        let in_w = (0..config.dim * config.input_dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let in_proj = Linear::from_tensor(
            cpu_tensor(in_w, Shape::new(vec![config.dim, config.input_dim])),
            Some(cpu_tensor(
                vec![0.0; config.dim],
                Shape::new(vec![config.dim]),
            )),
        );

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(ConvNeXtBlock::new(
                config.dim,
                config.intermediate_dim,
                &mut rng,
            ));
        }

        let fft_bins = config.n_fft / 2 + 1;
        let mag_w = (0..fft_bins * config.dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let head_mag = Linear::from_tensor(
            cpu_tensor(mag_w, Shape::new(vec![fft_bins, config.dim])),
            Some(cpu_tensor(vec![0.0; fft_bins], Shape::new(vec![fft_bins]))),
        );

        let phase_w = (0..fft_bins * config.dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let head_phase = Linear::from_tensor(
            cpu_tensor(phase_w, Shape::new(vec![fft_bins, config.dim])),
            Some(cpu_tensor(vec![0.0; fft_bins], Shape::new(vec![fft_bins]))),
        );

        Self {
            config,
            device,
            in_proj,
            blocks,
            head_mag,
            head_phase,
        }
    }
}

impl Model for Vocos {
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

impl AudioVocoder for Vocos {
    fn mel_to_audio(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.in_proj.forward(mel)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let mag = self.head_mag.forward(&x)?;
        let phase = self.head_phase.forward(&x)?;

        let mag_vec = mag.to_vec_f32()?;
        let phase_vec = phase.to_vec_f32()?;
        let num_frames = mel.shape().dims()[0];
        let fft_bins = self.config.n_fft / 2 + 1;
        let hop = self.config.hop_length;
        let total_samples = num_frames * hop;
        let mut audio = vec![0.0f32; total_samples];

        // iSTFT overlap-add reconstruction
        for f in 0..num_frames {
            for s in 0..hop {
                let sample_idx = f * hop + s;
                if sample_idx < total_samples {
                    let mut sample_val = 0.0f32;
                    for k in 0..fft_bins {
                        let m = mag_vec[f * fft_bins + k].exp();
                        let p = phase_vec[f * fft_bins + k];
                        let angle = (2.0 * std::f32::consts::PI * (k as f32) * (s as f32)
                            / (self.config.n_fft as f32))
                            + p;
                        sample_val += m * angle.cos();
                    }
                    audio[sample_idx] += (sample_val / (fft_bins as f32)).clamp(-1.0, 1.0);
                }
            }
        }

        Ok(cpu_tensor(audio, Shape::new(vec![total_samples])))
    }
}
