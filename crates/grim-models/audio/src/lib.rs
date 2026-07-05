//! Audio encoder-decoder for Grim — Whisper-style ASR. Implements the
//! `grim_core::model::EncoderDecoderLm` trait per §4.4.
//!
//! Pipeline:
//!   audio → encoder CNN → transformer encoder blocks →
//!   decoder autoregresses text tokens via cross-attention to encoder out.
//!
//! The decoder uses `EncoderDecoderLm::decode_step` once per token, which
//! Grim's serving layer calls in a loop until EOT.

pub mod whisper;
pub mod rng;

pub use whisper::{Whisper, WhisperConfig};
