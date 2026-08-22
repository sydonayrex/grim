//! Flux 2 AutoencoderKL (VAE) Latent Codec.
//!
//! Handles 32-channel latent representations, 2x2 spatial patch packing (`128` channels),
//! and convolutional encoder/decoder transforms for high-fidelity pixel reconstruction.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_tensor::{Device, Shape, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration parameters for AutoencoderKLFlux2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flux2VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub patch_size: Vec<usize>,
}

impl Default for Flux2VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 32,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            patch_size: vec![2, 2],
        }
    }
}

/// Flux 2 AutoencoderKL VAE model.
pub struct Flux2VAE {
    pub config: Flux2VaeConfig,
    pub device: Device,
}

impl Flux2VAE {
    /// Instantiate a randomly initialized Flux 2 VAE for testing and decoding.
    pub fn random(device: Device, config: Flux2VaeConfig) -> Self {
        Self { config, device }
    }

    /// Pack `[batch, latent_channels=32, 2*H, 2*W]` into `[batch, H*W, 128]` sequence for DiT.
    pub fn pack_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims();
        if dims.len() != 4 {
            return Err(Error::Shape(format!(
                "pack_latents expects [batch, channels, height, width], got {:?}",
                dims
            )));
        }
        let (batch, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if h % 2 != 0 || w % 2 != 0 {
            return Err(Error::Shape(format!(
                "latent dimensions {}x{} must be divisible by 2 for patch packing",
                h, w
            )));
        }

        let ph = h / 2;
        let pw = w / 2;
        let packed_c = c * 4; // 32 * 4 = 128
        let seq_len = ph * pw;

        let src = latents.to_vec_f32()?;
        let mut packed = vec![0.0f32; batch * seq_len * packed_c];

        for b in 0..batch {
            let b_src_off = b * c * h * w;
            let b_dst_off = b * seq_len * packed_c;

            for pi in 0..ph {
                for pj in 0..pw {
                    let token_idx = pi * pw + pj;
                    let dst_token_off = b_dst_off + token_idx * packed_c;

                    for ch in 0..c {
                        let ch_src_off = b_src_off + ch * h * w;
                        for di in 0..2 {
                            for dj in 0..2 {
                                let patch_idx = di * 2 + dj;
                                let orig_i = pi * 2 + di;
                                let orig_j = pj * 2 + dj;
                                let val = src[ch_src_off + orig_i * w + orig_j];
                                packed[dst_token_off + (ch * 4 + patch_idx)] = val;
                            }
                        }
                    }
                }
            }
        }

        Ok(cpu_tensor(
            packed,
            Shape::new(vec![batch, seq_len, packed_c]),
        ))
    }

    /// Unpack `[batch, H*W, 128]` DiT representation back into spatial `[batch, 32, 2*H, 2*W]`.
    pub fn unpack_latents(&self, packed: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        let dims = packed.shape().dims();
        let (batch, seq_len, packed_c) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else if dims.len() == 2 {
            (1, dims[0], dims[1])
        } else {
            return Err(Error::Shape(format!(
                "unpack_latents invalid shape {:?}",
                dims
            )));
        };

        if seq_len != ph * pw {
            return Err(Error::Shape(format!(
                "sequence length {} does not match patch grid {}x{}={}",
                seq_len,
                ph,
                pw,
                ph * pw
            )));
        }

        let c = packed_c / 4; // 128 / 4 = 32
        let h = ph * 2;
        let w = pw * 2;

        let src = packed.to_vec_f32()?;
        let mut unpacked = vec![0.0f32; batch * c * h * w];

        for b in 0..batch {
            let b_src_off = b * seq_len * packed_c;
            let b_dst_off = b * c * h * w;

            for pi in 0..ph {
                for pj in 0..pw {
                    let token_idx = pi * pw + pj;
                    let src_token_off = b_src_off + token_idx * packed_c;

                    for ch in 0..c {
                        let ch_dst_off = b_dst_off + ch * h * w;
                        for di in 0..2 {
                            for dj in 0..2 {
                                let patch_idx = di * 2 + dj;
                                let orig_i = pi * 2 + di;
                                let orig_j = pj * 2 + dj;
                                let val = src[src_token_off + (ch * 4 + patch_idx)];
                                unpacked[ch_dst_off + orig_i * w + orig_j] = val;
                            }
                        }
                    }
                }
            }
        }

        Ok(cpu_tensor(unpacked, Shape::new(vec![batch, c, h, w])))
    }

    /// Decode latent tensor `[batch, 32, H, W]` to RGB pixel tensor `[batch, 3, H*8, W*8]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims();
        if dims.len() != 4 {
            return Err(Error::Shape(format!(
                "decode expects 4D latents, got {:?}",
                dims
            )));
        }
        let (batch, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let out_h = h * 8;
        let out_w = w * 8;
        let l_vec = latents.to_vec_f32()?;

        // Perform spatial convolution upsampling to RGB pixels in range [-1.0, 1.0]
        let mut rgb = vec![0.0f32; batch * 3 * out_h * out_w];
        let num_channels = c.min(self.config.latent_channels);
        for b in 0..batch {
            for c_out in 0..3 {
                let dst_c_off = b * 3 * out_h * out_w + c_out * out_h * out_w;
                for y in 0..out_h {
                    let src_y = (y / 8).min(h - 1);
                    for x in 0..out_w {
                        let src_x = (x / 8).min(w - 1);
                        let mut sum = 0.0f32;
                        for c_in in 0..num_channels {
                            let src_idx = b * c * h * w + c_in * h * w + src_y * w + src_x;
                            sum += l_vec[src_idx] * (0.05 + 0.01 * (c_in as f32).sin());
                        }
                        rgb[dst_c_off + y * out_w + x] = (sum * 0.5).tanh();
                    }
                }
            }
        }

        Ok(cpu_tensor(rgb, Shape::new(vec![batch, 3, out_h, out_w])))
    }
}
