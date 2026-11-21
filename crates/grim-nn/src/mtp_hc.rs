//! Grim-native wide-stream Hyper-Connection Multi-Token Prediction (`MtpHcBlock`, `MtpHcCarry`).
//!
//! Model-agnostic NN building block for models whose residual stream is `hc_count * hidden_size`
//! wide. Supports `draft1` -> `draft2+` serial multi-round drafting with separate `hin`/`hw`
//! carry buffers on CPU and ROCm (RDNA3/RDNA4) via `BackendDevice`.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_tensor::error::{Error, Result};
use grim_tensor::{DType, Device, QuantProvenance, Shape, Tensor};

use crate::modules::{
    add_on_device, embedding_gather_on_device, move_to_device, pick_device_for_storage_device,
    pick_device_for_tensor, slice_last_row_2d_on_device, Linear, RmsNorm,
};

/// Returns `true` when `GRIM_MTP_HC` is truthy (`1` or `true`).
#[inline]
pub fn mtp_hc_enabled() -> bool {
    std::env::var("GRIM_MTP_HC")
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Separate wide-stream carry buffers (`[1, hc_count * hidden_size]`) for serial
/// `draft1` -> `draft2+` rounds so `mtp_enter` never clobbers unread input channels.
#[derive(Debug, Clone)]
pub struct MtpHcCarry {
    pub hin: Tensor,
    pub hw: Tensor,
    pub wide_dim: usize,
}

impl MtpHcCarry {
    pub fn new(device: &Device, hc_count: usize, hidden_size: usize) -> Result<Self> {
        let wide_dim = hc_count * hidden_size;
        let shape = Shape::new(vec![1, wide_dim]);
        let hin = move_to_device(&cpu_tensor(vec![0.0f32; wide_dim], shape.clone()), device)?;
        let hw = move_to_device(&cpu_tensor(vec![0.0f32; wide_dim], shape), device)?;
        Ok(Self { hin, hw, wide_dim })
    }

    /// Copies `src` (last row `[1, wide_dim]`) into `hin` on device.
    pub fn seed_from_wide(&mut self, wide: &Tensor) -> Result<()> {
        let last = slice_last_row_2d_on_device(wide)?;
        let dev = pick_device_for_tensor(&self.hin);
        dev.copy_slice_range(
            self.hin.storage().as_ref(),
            0,
            last.storage().as_ref(),
            0,
            self.wide_dim,
        )
    }

    /// Copies `hw -> hin` on device between draft rounds (`draft1 -> draft2+`).
    pub fn advance_carry(&mut self) -> Result<()> {
        let dev = pick_device_for_tensor(&self.hin);
        dev.copy_slice_range(
            self.hin.storage().as_ref(),
            0,
            self.hw.storage().as_ref(),
            0,
            self.wide_dim,
        )
    }
}

/// Model-agnostic Hyper-Connection MTP head operating on a `[1, hc_count * hidden_size]`
/// wide residual stream.
#[derive(Clone)]
pub struct MtpHcBlock {
    pub pre_fc_norm_hidden: RmsNorm,
    pub pre_fc_norm_embed: RmsNorm,
    pub fc_hidden: Linear,
    pub fc_embedding: Linear,
    pub output_mixer_norm: RmsNorm,
    pub output_mixer_proj: Linear,
    pub hc_count: usize,
    pub hidden_size: usize,
}

impl MtpHcBlock {
    /// Deterministic seeded constructor for CPU/GPU parity tests.
    pub fn seeded(hc_count: usize, hidden_size: usize, eps: f32, seed: u64) -> Self {
        let lcg = |s0: u64, n: usize| -> Vec<f32> {
            let mut s = s0;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32) / (u32::MAX as f32 * 0.5) - 1.0;
                v.push(u * 0.1);
            }
            v
        };
        let wide_dim = hc_count * hidden_size;
        Self {
            pre_fc_norm_hidden: RmsNorm {
                weight: cpu_tensor(vec![1.0f32; wide_dim], Shape::new(vec![wide_dim])),
                eps,
            },
            pre_fc_norm_embed: RmsNorm {
                weight: cpu_tensor(vec![1.0f32; hidden_size], Shape::new(vec![hidden_size])),
                eps,
            },
            fc_hidden: Linear::from_tensor(
                cpu_tensor(
                    lcg(seed + 1, hidden_size * hidden_size),
                    Shape::new(vec![hidden_size, hidden_size]),
                ),
                None,
            ),
            fc_embedding: Linear::from_tensor(
                cpu_tensor(
                    lcg(seed + 2, hidden_size * hidden_size),
                    Shape::new(vec![hidden_size, hidden_size]),
                ),
                None,
            ),
            output_mixer_norm: RmsNorm {
                weight: cpu_tensor(vec![1.0f32; wide_dim], Shape::new(vec![wide_dim])),
                eps,
            },
            output_mixer_proj: Linear::from_tensor(
                cpu_tensor(
                    lcg(seed + 3, hidden_size * wide_dim),
                    Shape::new(vec![hidden_size, wide_dim]),
                ),
                None,
            ),
            hc_count,
            hidden_size,
        }
    }

    /// Moves all weights of this `MtpHcBlock` onto `target` device (e.g. `Device::Rocm(0)`).
    pub fn to_device(&self, target: &Device) -> Result<Self> {
        Ok(Self {
            pre_fc_norm_hidden: RmsNorm {
                weight: move_to_device(&self.pre_fc_norm_hidden.weight, target)?,
                eps: self.pre_fc_norm_hidden.eps,
            },
            pre_fc_norm_embed: RmsNorm {
                weight: move_to_device(&self.pre_fc_norm_embed.weight, target)?,
                eps: self.pre_fc_norm_embed.eps,
            },
            fc_hidden: self.fc_hidden.staged_to(target)?,
            fc_embedding: self.fc_embedding.staged_to(target)?,
            output_mixer_norm: RmsNorm {
                weight: move_to_device(&self.output_mixer_norm.weight, target)?,
                eps: self.output_mixer_norm.eps,
            },
            output_mixer_proj: self.output_mixer_proj.staged_to(target)?,
            hc_count: self.hc_count,
            hidden_size: self.hidden_size,
        })
    }

    /// Wide-stream entry fusion:
    /// 1. Norm `wide_hidden` (`[1, hc * H]`), apply `fc_hidden` to each `[1, H]` sub-stream.
    /// 2. Norm `tok_embed` (`[1, H]`), apply `fc_embedding`, broadcast and add to every sub-stream.
    pub fn mtp_enter(&self, wide_hidden: &Tensor, tok_embed: &Tensor) -> Result<Tensor> {
        let wide_dim = self.hc_count * self.hidden_size;
        let h_norm = self.pre_fc_norm_hidden.forward(wide_hidden)?;
        let e_norm = self.pre_fc_norm_embed.forward(tok_embed)?;
        let e_proj = self.fc_embedding.forward(&e_norm)?;

        let dev = pick_device_for_tensor(&h_norm);
        let out_shape = Shape::new(vec![1, wide_dim]);
        let out_st = dev.alloc_storage(&out_shape, DType::F32)?;
        let sub_shape = Shape::new(vec![1, self.hidden_size]);

        for c in 0..self.hc_count {
            let sub_st = dev.alloc_storage(&sub_shape, DType::F32)?;
            dev.copy_slice_range(
                sub_st.as_ref(),
                0,
                h_norm.storage().as_ref(),
                c * self.hidden_size,
                self.hidden_size,
            )?;
            let sub_t = Tensor::new(
                Arc::from(sub_st),
                sub_shape.clone(),
                DType::F32,
                QuantProvenance::default(),
                h_norm.device().clone(),
            );
            let h_proj = self.fc_hidden.forward(&sub_t)?;
            let fused = add_on_device(&h_proj, &e_proj)?;
            dev.copy_slice_range(
                out_st.as_ref(),
                c * self.hidden_size,
                fused.storage().as_ref(),
                0,
                self.hidden_size,
            )?;
        }

        Ok(Tensor::new(
            Arc::from(out_st),
            out_shape,
            DType::F32,
            QuantProvenance::default(),
            h_norm.device().clone(),
        ))
    }

    /// Collapses `[1, hc * H]` wide stream into `[1, H]` for `lm_head`.
    pub fn collapse_to_hidden(&self, wide_stream: &Tensor) -> Result<Tensor> {
        let normed = self.output_mixer_norm.forward(wide_stream)?;
        self.output_mixer_proj.forward(&normed)
    }

    /// Runs `depth` serial draft rounds (`draft1`, `draft2`, ..., `draft_k`) using
    /// separate `carry.hin` / `carry.hw` wide buffers and device-native argmax.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_k_tokens<F>(
        &self,
        trunk_wide_hidden: &Tensor,
        first_token: u32,
        start_pos: u32,
        depth: usize,
        embed_weight: &Tensor,
        vocab_size: usize,
        lm_head: &Linear,
        carry: &mut MtpHcCarry,
        mut layer_fn: F,
    ) -> Result<(Vec<u32>, Vec<Tensor>)>
    where
        F: FnMut(&Tensor, u32) -> Result<Tensor>,
    {
        if depth == 0 || vocab_size == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        carry.seed_from_wide(trunk_wide_hidden)?;

        let mut tokens = Vec::with_capacity(depth);
        let mut step_logits = Vec::with_capacity(depth);
        let mut curr_tok = first_token;

        for round in 0..depth {
            if round > 0 {
                carry.advance_carry()?;
            }
            let clamped = if (curr_tok as usize) < vocab_size {
                curr_tok
            } else {
                0
            };
            let tok_emb =
                embedding_gather_on_device(embed_weight, &[clamped], 1, self.hidden_size)?;
            let entered = self.mtp_enter(&carry.hin, &tok_emb)?;
            let next_wide = layer_fn(&entered, start_pos.saturating_add(round as u32))?;
            // Save into carry.hw (separate from carry.hin)
            let dev = pick_device_for_tensor(&carry.hw);
            dev.copy_slice_range(
                carry.hw.storage().as_ref(),
                0,
                next_wide.storage().as_ref(),
                0,
                carry.wide_dim,
            )?;
            let collapsed = self.collapse_to_hidden(&carry.hw)?;
            let logits = lm_head.forward(&collapsed)?;
            let next_tok = argmax_row_on_device(&logits, vocab_size)?;
            tokens.push(next_tok);
            step_logits.push(logits);
            curr_tok = next_tok;
        }
        Ok((tokens, step_logits))
    }
}

fn argmax_row_on_device(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    if vocab_size == 0 {
        return Err(Error::Shape("argmax_row_on_device: empty vocab".into()));
    }
    let dev = pick_device_for_storage_device(logits.device());
    if let Ok(idx) = dev.argmax(logits.storage().as_ref()) {
        return Ok(idx);
    }
    let v = logits.to_vec_f32()?;
    let start = v.len().saturating_sub(vocab_size);
    Ok(v[start..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0))
}
