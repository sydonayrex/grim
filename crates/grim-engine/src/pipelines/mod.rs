//! End-to-end multimodal execution pipelines (Audio TTS/Vocoder and Image Diffusion).

pub mod audio;
pub mod diffusion;

pub use audio::{AudioPipeline, AudioPipelineConfig};
pub use diffusion::{DiffusionPipeline, DiffusionPipelineConfig};
