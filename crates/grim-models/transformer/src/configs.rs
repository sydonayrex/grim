//! Additional model configuration structs for unsupported and specialized transformer architectures.
//!
//! Implements `ModelConfig` for Falcon, BLOOM, Phi, Qwen, and MoE variants.

use grim_core::model::{ModalityHint, ModelConfig};

/// Configuration for Falcon model architecture family.
#[derive(Debug, Clone)]
pub struct FalconConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for FalconConfig {
    fn name(&self) -> &str { "falcon" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for BLOOM model architecture family.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for BloomConfig {
    fn name(&self) -> &str { "bloom" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Phi (Phi-2/Phi-3/Phi-Moe) model architecture family.
#[derive(Debug, Clone)]
pub struct PhiConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for PhiConfig {
    fn name(&self) -> &str { "phi" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Qwen (Qwen/Qwen2/Qwen3/Qwen3.5) model architecture family.
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for QwenConfig {
    fn name(&self) -> &str { "qwen" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Mixture of Experts (MoE) model architectures.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for MoeConfig {
    fn name(&self) -> &str { "moe" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}
