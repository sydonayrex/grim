//! Configuration structs for encoder architecture variants.
//!
//! Includes `ModernBertConfig`, `NomicBertConfig`, `T5EncoderConfig`.

use grim_core::model::{ModalityHint, ModelConfig};

/// Configuration for ModernBERT encoder architecture.
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for ModernBertConfig {
    fn name(&self) -> &str { "modern-bert" }
    fn modality(&self) -> ModalityHint { ModalityHint::VisionEncoder }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Nomic-BERT encoder architecture.
#[derive(Debug, Clone)]
pub struct NomicBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for NomicBertConfig {
    fn name(&self) -> &str { "nomic-bert" }
    fn modality(&self) -> ModalityHint { ModalityHint::VisionEncoder }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for T5Encoder architecture.
#[derive(Debug, Clone)]
pub struct T5EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for T5EncoderConfig {
    fn name(&self) -> &str { "t5encoder" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}
