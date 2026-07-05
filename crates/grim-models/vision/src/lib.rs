//! Vision encoder for Grim — ViT/CLIP-style patch embedding + transformer
//! encoder. Implements the `grim_core::model::Encoder` trait per §4.4.
//!
//! Architecture follows the standard ViT recipe: project image patches
//! into tokens, prepend a learnable [CLS] token, run a stack of pre-norm
//! self-attention blocks, return the [CLS] embedding as the image feature.
//! No autoregressive head; the model is `Encoder`, not `CausalLm`.

pub mod vit;
pub mod rng;

pub use vit::{Vit, VitConfig};
