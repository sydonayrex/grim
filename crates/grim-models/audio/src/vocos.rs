//! Vocos Neural Audio Vocoder Architecture.
//!
//! Matches the real Vocos checkpoint layout (e.g. `MeanVC2/vocos.pt`):
//! a Conv1d stem (`backbone.embed`), LayerNorm + ConvNeXt backbone
//! (`backbone.convnext.{i}` with depthwise conv, LayerNorm, pointwise MLP
//! and layer-scale gamma), a final LayerNorm, and an iSTFT head whose linear
//! emits log-magnitude + phase halves (`head.out`) with a learned synthesis
//! window (`head.istft.window`).

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AudioVocoder, ModalityHint, Model, ModelConfig};
use grim_core::rng::SimpleRng;
use grim_nn::{Conv1d, LayerNorm, Linear, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};
use std::f32::consts::PI;

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

impl VocosConfig {
    /// Number of magnitude bins implied by `n_fft`.
    pub fn n_mag(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// ConvNeXt block: depthwise conv → LayerNorm → pointwise MLP → layer scale.
struct ConvNeXtBlock {
    dw_conv: Conv1d,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    gamma: Vec<f32>,
    dim: usize,
}

impl ConvNeXtBlock {
    fn new(dim: usize, intermediate: usize, rng: &mut SimpleRng) -> Self {
        let dw_weight = (0..dim * 1 * 7)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let dw_conv = Conv1d::new(
            cpu_tensor(dw_weight, Shape::new(vec![dim, 1, 7])),
            Some(cpu_tensor(vec![0.0; dim], Shape::new(vec![dim]))),
            1, // stride
            3, // padding
            1, // dilation
            dim, // groups (depthwise)
        );

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
            dw_conv,
            norm: LayerNorm::new(
                cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
                Some(cpu_tensor(vec![0.0; dim], Shape::new(vec![dim]))),
                1e-6,
            ),
            pw1,
            pw2,
            gamma: vec![1e-6; dim],
            dim,
        }
    }

    fn load(ws: &WeightSource<'_>, dim: usize, intermediate: usize) -> Result<Self> {
        let dw_conv = Conv1d::load(&ws.scoped("dwconv"), dim, 1, 7, 1, 3, 1, dim)?;
        let norm = LayerNorm::load(&ws.scoped("norm"), dim, 1e-6)?;
        let pw1 = Linear::load(&ws.pp("pwconv1"), dim, intermediate, true)?;
        let pw2 = Linear::load(&ws.pp("pwconv2"), intermediate, dim, true)?;
        let gamma = ws
            .scoped("gamma")
            .get([dim], "weight")
            .or_else(|_| ws.get([dim], "gamma"))
            .ok()
            .and_then(|t| t.to_vec_f32().ok())
            .unwrap_or_else(|| vec![1e-6; dim]);
        Ok(Self {
            dw_conv,
            norm,
            pw1,
            pw2,
            gamma,
            dim,
        })
    }

    /// Forward on `[seq_len, dim]` activations.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_vec = x.to_vec_f32()?;
        let seq_len = x.shape().dims()[0];
        let dim = self.dim;

        // 1. Depthwise 1D conv (kernel size 7, padding 3).
        let dw = self.dw_conv.forward(x)?;
        // 2. LayerNorm over channels.
        let normed = self.norm.forward(&dw)?;
        // 3. Pointwise MLP with GELU activation.
        let h = self.pw1.forward(&normed)?;
        let h_gelu = cpu_tensor(
            h.to_vec_f32()?
                .into_iter()
                .map(|v| 0.5 * v * (1.0 + (0.79788456 * (v + 0.044715 * v * v * v)).tanh()))
                .collect(),
            h.shape().clone(),
        );
        let out = self.pw2.forward(&h_gelu)?;

        // 4. Layer scale + residual.
        let out_vec = out.to_vec_f32()?;
        let mut res = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            for c in 0..dim {
                res[t * dim + c] = x_vec[t * dim + c] + out_vec[t * dim + c] * self.gamma[c];
            }
        }
        Ok(cpu_tensor(res, Shape::new(vec![seq_len, dim])))
    }
}

/// Vocos Vocoder Model.
pub struct Vocos {
    pub config: VocosConfig,
    pub device: Device,
    stem: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
    head: Linear,
    window: Vec<f32>,
}

fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / len as f32).cos()))
        .collect()
}

impl Vocos {
    /// Instantiate a randomly initialized Vocos model for testing.
    pub fn random(device: Device, config: VocosConfig) -> Self {
        let mut rng = SimpleRng::new(9999);
        let stem_w = (0..config.dim * config.input_dim * 7)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let stem = Conv1d::new(
            cpu_tensor(stem_w, Shape::new(vec![config.dim, config.input_dim, 7])),
            Some(cpu_tensor(vec![0.0; config.dim], Shape::new(vec![config.dim]))),
            1,
            3,
            1,
            1,
        );
        let ones = |n: usize| cpu_tensor(vec![1.0; n], Shape::new(vec![n]));
        let zeros = |n: usize| cpu_tensor(vec![0.0; n], Shape::new(vec![n]));

        let blocks = (0..config.num_layers)
            .map(|_| ConvNeXtBlock::new(config.dim, config.intermediate_dim, &mut rng))
            .collect();

        let head_w = (0..2 * config.n_mag() * config.dim)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let head = Linear::from_tensor(
            cpu_tensor(head_w, Shape::new(vec![2 * config.n_mag(), config.dim])),
            Some(cpu_tensor(
                vec![0.0; 2 * config.n_mag()],
                Shape::new(vec![2 * config.n_mag()]),
            )),
        );

        let dim = config.dim;
        let window = hann_window(config.n_fft);

        Self {
            config,
            device,
            stem,
            norm: LayerNorm::new(ones(dim), Some(zeros(dim)), 1e-6),
            blocks,
            final_norm: LayerNorm::new(ones(dim), Some(zeros(dim)), 1e-6),
            head,
            window,
        }
    }

    /// Load weights from a real Vocos checkpoint (`backbone.*`, `head.*`
    /// naming as produced by the official Vocos training stack).
    pub fn load(_device: Device, ws: &WeightSource<'_>, config: VocosConfig) -> Result<Self> {
        let backbone = ws.scoped("backbone");
        let stem = Conv1d::load(
            &backbone.scoped("embed"),
            config.dim,
            config.input_dim,
            7,
            1,
            3,
            1,
            1,
        )?;
        let norm = LayerNorm::load(&backbone.scoped("norm"), config.dim, 1e-6)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(ConvNeXtBlock::load(
                &backbone.scoped("convnext").scoped(&i.to_string()),
                config.dim,
                config.intermediate_dim,
            )?);
        }

        let final_norm =
            LayerNorm::load(&backbone.scoped("final_layer_norm"), config.dim, 1e-6)?;
        let head = Linear::load(
            &ws.scoped("head").pp("out"),
            config.dim,
            2 * config.n_mag(),
            true,
        )?;
        let window = ws
            .scoped("head")
            .scoped("istft")
            .get([config.n_fft], "window")
            .ok()
            .and_then(|t| t.to_vec_f32().ok())
            .unwrap_or_else(|| hann_window(config.n_fft));

        Ok(Self {
            config,
            device: _device,
            stem,
            norm,
            blocks,
            final_norm,
            head,
            window,
        })
    }

    /// iSTFT overlap-add from log-magnitude + phase frames.
    fn istft(&self, mag_log: &[f32], phase: &[f32], num_frames: usize) -> Vec<f32> {
        let n_mag = self.config.n_mag();
        let frame = self.window.len();
        let hop = self.config.hop_length;
        let total = (num_frames - 1) * hop + frame;
        let mut wave = vec![0.0f32; total];
        let mut win_sq = vec![0.0f32; total];

        for f in 0..num_frames {
            for s in 0..frame {
                let mut acc = 0.0f32;
                for k in 0..n_mag {
                    let m = mag_log[f * n_mag + k].exp();
                    let p = phase[f * n_mag + k];
                    let angle = 2.0 * PI * (k as f32) * (s as f32) / (frame as f32) + p;
                    acc += m * angle.cos();
                }
                let w = self.window[s];
                wave[f * hop + s] += acc * w;
                win_sq[f * hop + s] += w * w;
            }
        }
        for (v, n) in wave.iter_mut().zip(win_sq.iter()) {
            if *n > 1e-8 {
                *v /= *n;
            }
        }
        wave
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
        // Backbone operates on `[seq_len, dim]` activations.
        let mut x = self.stem.forward(mel)?;
        x = self.norm.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.final_norm.forward(&x)?;

        let head_out = self.head.forward(&x)?;
        let rows = head_out.to_vec_f32()?;
        let num_frames = mel.shape().dims()[0];
        let n_mag = self.config.n_mag();

        let mut mag_log = vec![0.0f32; num_frames * n_mag];
        let mut phase = vec![0.0f32; num_frames * n_mag];
        for f in 0..num_frames {
            for k in 0..n_mag {
                mag_log[f * n_mag + k] = rows[f * 2 * n_mag + k];
                phase[f * n_mag + k] = rows[f * 2 * n_mag + n_mag + k];
            }
        }

        let audio = self.istft(&mag_log, &phase, num_frames);
        let audio_len = audio.len();
        Ok(cpu_tensor(audio, Shape::new(vec![audio_len])))
    }
}
