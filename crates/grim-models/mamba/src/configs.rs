//! Config definitions for RWKV and SSM architecture variants.
//!
//! Includes `Rwkv6Config`, `Rwkv7Config`, `Mamba2Config`, `JambaConfig`, `NemotronHConfig`, `GraniteHybridConfig`.

use grim_core::model::{ModalityHint, ModelConfig};

/// Configuration for RWKV6 model architecture.
#[derive(Debug, Clone)]
pub struct Rwkv6Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl ModelConfig for Rwkv6Config {
    fn name(&self) -> &str { "rwkv6" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for RWKV7 model architecture.
#[derive(Debug, Clone)]
pub struct Rwkv7Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl ModelConfig for Rwkv7Config {
    fn name(&self) -> &str { "rwkv7" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Mamba-2 SSM architecture.
#[derive(Debug, Clone)]
pub struct Mamba2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
}

impl ModelConfig for Mamba2Config {
    fn name(&self) -> &str { "mamba2" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Jamba hybrid SSM+Attention+MoE architecture.
#[derive(Debug, Clone)]
pub struct JambaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub ssm_d_state: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for JambaConfig {
    fn name(&self) -> &str { "jamba" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Nemotron-H hybrid architecture.
#[derive(Debug, Clone)]
pub struct NemotronHConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub ssm_d_state: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for NemotronHConfig {
    fn name(&self) -> &str { "nemotron-h" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Configuration for Granite-Hybrid architecture.
#[derive(Debug, Clone)]
pub struct GraniteHybridConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub ssm_d_state: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for GraniteHybridConfig {
    fn name(&self) -> &str { "granite-hybrid" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
}
