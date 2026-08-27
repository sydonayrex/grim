//! WavTokenizer Vocos ConvNeXt acoustic codec decoder and iSTFT waveform synthesizer.
//!
//! # Architecture Details
//! - **Codebook Lookup**: Maps discrete audio tokens to latent feature vectors `(B, latent_dim, T)`.
//! - **Vocos Backbone**: Depthwise ConvNeXt blocks with adaptive bandwidth LayerNorm (`AdaLayerNorm`).
//! - **iSTFT Synthesis**: Linear projection to magnitude/phase spectrogram bins followed by inverse STFT overlap-add.

use std::f32::consts::PI;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for WavTokenizer decoder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WavTokenizerDecConfig {
    pub latent_dim: usize,
    pub backbone_dim: usize,
    pub backbone_num_blocks: usize,
    pub backbone_intermediate_dim: usize,
    pub backbone_kernel_size: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub head_dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub num_bandwidths: usize,
    pub sample_rate: usize,
}

impl Default for WavTokenizerDecConfig {
    fn default() -> Self {
        Self {
            latent_dim: 512,
            backbone_dim: 768,
            backbone_num_blocks: 12,
            backbone_intermediate_dim: 2304,
            backbone_kernel_size: 7,
            n_fft: 1280,
            hop_length: 320,
            head_dim: 641,
            codebook_size: 4096,
            codebook_dim: 512,
            num_bandwidths: 4,
            sample_rate: 24000,
        }
    }
}

impl ModelConfig for WavTokenizerDecConfig {
    fn name(&self) -> &str {
        "wav_tokenizer_dec"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::AudioEncoderDecoder
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNorm
// ---------------------------------------------------------------------------

/// Bandwidth-conditioned adaptive layer normalization.
pub struct AdaLayerNorm {
    pub scale: Tensor,
    pub shift: Tensor,
    pub dim: usize,
    pub eps: f32,
}

impl AdaLayerNorm {
    pub fn load(
        ws: &WeightSource<'_>,
        num_bandwidths: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let scale = ws.scoped("scale").get([num_bandwidths, dim], "weight")?;
        let shift = ws.scoped("shift").get([num_bandwidths, dim], "weight")?;
        Ok(Self {
            scale,
            shift,
            dim,
            eps,
        })
    }

    /// Normalizes `x` of shape `[channels, length]` and modulates with bandwidth conditions.
    pub fn forward(&self, x: &Tensor, bandwidth_id: usize) -> Result<Tensor> {
        let xv = x.to_vec_f32()?;
        let dims = x.shape().dims();
        let c_dim = dims[0];
        let t_dim = if dims.len() > 1 { dims[1] } else { 1 };

        let scale_v = self.scale.to_vec_f32()?;
        let shift_v = self.shift.to_vec_f32()?;
        let b_offset = (bandwidth_id.min(scale_v.len() / self.dim - 1)) * self.dim;

        let mut out = vec![0.0f32; c_dim * t_dim];

        // Normalize across channel dimension per time step
        for t in 0..t_dim {
            let mut mean = 0.0f32;
            for c in 0..c_dim {
                mean += xv[c * t_dim + t];
            }
            mean /= c_dim as f32;

            let mut var = 0.0f32;
            for c in 0..c_dim {
                let diff = xv[c * t_dim + t] - mean;
                var += diff * diff;
            }
            var /= c_dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for c in 0..c_dim {
                let gamma = scale_v[b_offset + c];
                let beta = shift_v[b_offset + c];
                let norm = (xv[c * t_dim + t] - mean) * inv_std;
                out[c * t_dim + t] = norm * (1.0 + gamma) + beta;
            }
        }

        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// ConvNeXtBlock
// ---------------------------------------------------------------------------

/// 1D ConvNeXt block with depthwise convolution and adaptive normalization.
pub struct ConvNeXtBlock {
    pub dwconv_weight: Tensor,
    pub dwconv_bias: Option<Tensor>,
    pub norm: AdaLayerNorm,
    pub pwconv1: Linear,
    pub pwconv2: Linear,
    pub gamma: Tensor,
    pub dim: usize,
    pub kernel_size: usize,
}

impl ConvNeXtBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &WavTokenizerDecConfig) -> Result<Self> {
        let dwconv_weight = ws
            .scoped("dwconv")
            .get([cfg.backbone_dim, 1, cfg.backbone_kernel_size], "weight")?;
        let dwconv_bias = ws.scoped("dwconv").get([cfg.backbone_dim], "bias").ok();

        let norm = AdaLayerNorm::load(
            &ws.scoped("norm"),
            cfg.num_bandwidths,
            cfg.backbone_dim,
            1e-6,
        )?;
        let pwconv1 = Linear::load_shape(
            &ws.scoped("pwconv1"),
            [cfg.backbone_dim, cfg.backbone_intermediate_dim],
        )?;
        let pwconv2 = Linear::load_shape(
            &ws.scoped("pwconv2"),
            [cfg.backbone_intermediate_dim, cfg.backbone_dim],
        )?;
        let gamma = ws.get([cfg.backbone_dim], "gamma").unwrap_or_else(|_| {
            cpu_tensor(
                vec![1e-6f32; cfg.backbone_dim],
                Shape::new(vec![cfg.backbone_dim]),
            )
        });

        Ok(Self {
            dwconv_weight,
            dwconv_bias,
            norm,
            pwconv1,
            pwconv2,
            gamma,
            dim: cfg.backbone_dim,
            kernel_size: cfg.backbone_kernel_size,
        })
    }

    /// Forward pass through ConvNeXt block.
    pub fn forward(&self, x: &Tensor, bandwidth_id: usize) -> Result<Tensor> {
        let c_dim = self.dim;
        let t_dim = x.shape().dims()[1];
        let xv = x.to_vec_f32()?;

        // Depthwise 1D Conv with padding
        let dw_w = self.dwconv_weight.to_vec_f32()?;
        let dw_b = self
            .dwconv_bias
            .as_ref()
            .map(|b| b.to_vec_f32())
            .transpose()?;
        let pad = self.kernel_size / 2;
        let mut conv_out = vec![0.0f32; c_dim * t_dim];

        for c in 0..c_dim {
            let kernel = &dw_w[c * self.kernel_size..(c + 1) * self.kernel_size];
            let bias_val = dw_b.as_ref().map(|b| b[c]).unwrap_or(0.0);
            for t in 0..t_dim {
                let mut sum = bias_val;
                for (k, &w) in kernel.iter().enumerate().take(self.kernel_size) {
                    let in_idx = t as isize + k as isize - pad as isize;
                    if in_idx >= 0 && (in_idx as usize) < t_dim {
                        sum += xv[c * t_dim + in_idx as usize] * w;
                    }
                }
                conv_out[c * t_dim + t] = sum;
            }
        }

        let conv_t = cpu_tensor(conv_out, x.shape().clone());
        let normed = self.norm.forward(&conv_t, bandwidth_id)?;

        // Transpose [C, T] -> [T, C] for point-wise dense layers
        let norm_v = normed.to_vec_f32()?;
        let mut tc_v = vec![0.0f32; t_dim * c_dim];
        for t in 0..t_dim {
            for c in 0..c_dim {
                tc_v[t * c_dim + c] = norm_v[c * t_dim + t];
            }
        }
        let tc_t = cpu_tensor(tc_v, Shape::new(vec![t_dim, c_dim]));

        let mid = self.pwconv1.forward(&tc_t)?;
        let mid_v = mid.to_vec_f32()?;
        let gelu_v: Vec<f32> = mid_v
            .iter()
            .map(|&v| 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v.powi(3))).tanh()))
            .collect();
        let gelu_t = cpu_tensor(gelu_v, mid.shape().clone());
        let out_dense = self.pwconv2.forward(&gelu_t)?;

        // Scale by gamma and add residual: x + gamma * out_dense
        let out_v = out_dense.to_vec_f32()?;
        let gamma_v = self.gamma.to_vec_f32()?;
        let mut final_v = vec![0.0f32; c_dim * t_dim];

        for c in 0..c_dim {
            for t in 0..t_dim {
                let res = xv[c * t_dim + t];
                let ffn = out_v[t * c_dim + c] * gamma_v[c];
                final_v[c * t_dim + t] = res + ffn;
            }
        }

        Ok(cpu_tensor(final_v, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// iSTFT Head
// ---------------------------------------------------------------------------

/// Inverse Short-Time Fourier Transform synthesis head.
pub struct ISTFTHead {
    pub out: Linear,
    pub n_fft: usize,
    pub hop_length: usize,
    pub window: Vec<f32>,
}

impl ISTFTHead {
    pub fn load(ws: &WeightSource<'_>, cfg: &WavTokenizerDecConfig) -> Result<Self> {
        let out = Linear::load_shape(&ws.scoped("out"), [cfg.backbone_dim, cfg.head_dim * 2])?;
        // Precompute Hann window
        let window: Vec<f32> = (0..cfg.n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / cfg.n_fft as f32).cos()))
            .collect();
        Ok(Self {
            out,
            n_fft: cfg.n_fft,
            hop_length: cfg.hop_length,
            window,
        })
    }

    /// Synthesizes 1D audio samples from features `[T, backbone_dim]`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let t_dim = features.shape().dims()[0];
        let spec = self.out.forward(features)?;
        let spec_v = spec.to_vec_f32()?;
        let n_freq = self.n_fft / 2 + 1;

        let total_samples = (t_dim - 1) * self.hop_length + self.n_fft;
        let mut waveform = vec![0.0f32; total_samples];
        let mut win_norm = vec![0.0f32; total_samples];

        for t in 0..t_dim {
            let row = &spec_v[t * (n_freq * 2)..(t + 1) * (n_freq * 2)];
            let mag = &row[0..n_freq];
            let phase = &row[n_freq..2 * n_freq];

            // Compute IDFT for frame t
            let mut frame = vec![0.0f32; self.n_fft];
            for (n, f) in frame.iter_mut().enumerate().take(self.n_fft) {
                let mut real_sum = 0.0f32;
                for k in 0..n_freq {
                    let m = mag[k].exp();
                    let p = phase[k];
                    let angle = 2.0 * PI * (k as f32) * (n as f32) / (self.n_fft as f32) + p;
                    real_sum += m * angle.cos();
                }
                *f = (real_sum / self.n_fft as f32) * self.window[n];
            }

            // Overlap-add
            let start = t * self.hop_length;
            for n in 0..self.n_fft {
                if start + n < total_samples {
                    waveform[start + n] += frame[n];
                    win_norm[start + n] += self.window[n] * self.window[n];
                }
            }
        }

        // Normalize overlap
        for s in 0..total_samples {
            if win_norm[s] > 1e-4 {
                waveform[s] /= win_norm[s];
            }
        }

        Ok(cpu_tensor(waveform, Shape::new(vec![1, total_samples])))
    }
}

// ---------------------------------------------------------------------------
// WavTokenizerDec Model
// ---------------------------------------------------------------------------

/// WavTokenizer complete acoustic decoder model.
pub struct WavTokenizerDec {
    pub cfg: WavTokenizerDecConfig,
    pub device: Device,
    pub codebook: Tensor,
    pub embed_conv: Linear,
    pub norm: AdaLayerNorm,
    pub blocks: Vec<ConvNeXtBlock>,
    pub head: ISTFTHead,
}

impl WavTokenizerDec {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: WavTokenizerDecConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: WavTokenizerDecConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let codebook = ws
            .scoped("feature_extractor")
            .scoped("encodec")
            .scoped("quantizer")
            .scoped("vq")
            .scoped("layers")
            .scoped("0")
            .scoped("_codebook")
            .get([cfg.codebook_size, cfg.codebook_dim], "embed")
            .or_else(|_| ws.get([cfg.codebook_size, cfg.codebook_dim], "codebook.embed"))
            .unwrap_or_else(|_| {
                cpu_tensor(
                    vec![0.0f32; cfg.codebook_size * cfg.codebook_dim],
                    Shape::new(vec![cfg.codebook_size, cfg.codebook_dim]),
                )
            });

        let backbone_ws = ws.scoped("backbone");
        let embed_conv = Linear::load_shape(
            &backbone_ws.scoped("embed"),
            [cfg.latent_dim, cfg.backbone_dim],
        )?;
        let norm = AdaLayerNorm::load(
            &backbone_ws.scoped("norm"),
            cfg.num_bandwidths,
            cfg.backbone_dim,
            1e-6,
        )?;

        let mut blocks = Vec::with_capacity(cfg.backbone_num_blocks);
        for i in 0..cfg.backbone_num_blocks {
            let block =
                ConvNeXtBlock::load(&backbone_ws.scoped("convnext").scoped(&i.to_string()), &cfg)?;
            blocks.push(block);
        }

        let head = ISTFTHead::load(&ws.scoped("head"), &cfg)?;

        Ok(Self {
            cfg,
            device,
            codebook,
            embed_conv,
            norm,
            blocks,
            head,
        })
    }

    /// Decodes discrete token IDs into time-domain audio waveform.
    pub fn decode_codes(&self, codes: &[usize], bandwidth_id: usize) -> Result<Tensor> {
        let t_dim = codes.len();
        let cb_v = self.codebook.to_vec_f32()?;
        let mut latent = vec![0.0f32; self.cfg.codebook_dim * t_dim];

        for (t, &idx) in codes.iter().enumerate() {
            let clamped = idx.min(self.cfg.codebook_size - 1);
            for c in 0..self.cfg.codebook_dim {
                latent[c * t_dim + t] = cb_v[clamped * self.cfg.codebook_dim + c];
            }
        }

        let mut feat = cpu_tensor(latent, Shape::new(vec![self.cfg.codebook_dim, t_dim]));
        feat = self.norm.forward(&feat, bandwidth_id)?;

        for block in &self.blocks {
            feat = block.forward(&feat, bandwidth_id)?;
        }

        // Transpose [C, T] -> [T, C] for iSTFT head
        let feat_v = feat.to_vec_f32()?;
        let c_dim = self.cfg.backbone_dim;
        let mut tc_v = vec![0.0f32; t_dim * c_dim];
        for t in 0..t_dim {
            for c in 0..c_dim {
                tc_v[t * c_dim + c] = feat_v[c * t_dim + t];
            }
        }

        let tc_t = cpu_tensor(tc_v, Shape::new(vec![t_dim, c_dim]));
        self.head.forward(&tc_t)
    }
}

impl Model for WavTokenizerDec {
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

impl CausalLm for WavTokenizerDec {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<usize> = input_ids
            .to_vec_f32()?
            .iter()
            .map(|&f| f as usize)
            .collect();
        let audio = self.decode_codes(&ids, 0)?;
        session.advance_pos(ids.len());
        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_tokenizer_config() {
        let cfg = WavTokenizerDecConfig::default();
        assert_eq!(cfg.backbone_dim, 768);
        assert_eq!(cfg.backbone_num_blocks, 12);
        assert_eq!(cfg.sample_rate, 24000);
    }
}
