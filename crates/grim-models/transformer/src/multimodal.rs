//! Multimodal patch/frame projection and embedding merger module.
//!
//! Provides `VisionPatchProjection` and `AudioMelProjection` layers and
//! `merge_multimodal_embeddings` helper to interleave image/audio patch
//! representations into text token sequence embeddings.

use grim_core::error::{Error, Result};
use grim_nn::Linear;
use grim_tensor::Tensor;

/// Vision patch projection layer mapping raw pixel patches to model hidden dimension.
#[derive(Clone)]
pub struct VisionPatchProjection {
    pub proj: Linear,
    pub patch_size: usize,
    pub hidden_size: usize,
}

impl VisionPatchProjection {
    pub fn load(
        ws: &grim_nn::WeightSource<'_>,
        patch_size: usize,
        n_channels: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let input_dim = patch_size * patch_size * n_channels;
        let proj = Linear::load(&ws.pp("proj"), input_dim, hidden_size, false)?;
        Ok(Self {
            proj,
            patch_size,
            hidden_size,
        })
    }

    pub fn forward(&self, image_patches: &Tensor) -> Result<Tensor> {
        self.proj.forward(image_patches).map_err(Into::into)
    }
}

/// Audio mel-spectrogram projection layer mapping Mel frames to model hidden dimension.
#[derive(Clone)]
pub struct AudioMelProjection {
    pub proj: Linear,
    pub n_mel_bins: usize,
    pub hidden_size: usize,
}

impl AudioMelProjection {
    pub fn load(
        ws: &grim_nn::WeightSource<'_>,
        n_mel_bins: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let proj = Linear::load(&ws.pp("proj"), n_mel_bins, hidden_size, false)?;
        Ok(Self {
            proj,
            n_mel_bins,
            hidden_size,
        })
    }

    pub fn forward(&self, mel_frames: &Tensor) -> Result<Tensor> {
        self.proj.forward(mel_frames).map_err(Into::into)
    }
}

/// Substitute projected vision/audio patch vectors into text sequence embeddings at specified indices.
pub fn merge_multimodal_embeddings(
    seq_embeddings: &mut [f32],
    patch_embeddings: &[f32],
    placeholder_indices: &[usize],
    hidden_size: usize,
) -> Result<()> {
    if placeholder_indices.len() * hidden_size > patch_embeddings.len() {
        return Err(Error::Shape(
            "patch_embeddings length smaller than placeholder indices".into(),
        ));
    }

    for (i, &token_idx) in placeholder_indices.iter().enumerate() {
        let seq_start = token_idx * hidden_size;
        let patch_start = i * hidden_size;
        if seq_start + hidden_size <= seq_embeddings.len() {
            seq_embeddings[seq_start..seq_start + hidden_size]
                .copy_from_slice(&patch_embeddings[patch_start..patch_start + hidden_size]);
        }
    }

    Ok(())
}
