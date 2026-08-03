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
    fn name(&self) -> &str {
        "modern-bert"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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
    fn name(&self) -> &str {
        "nomic-bert"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modern_bert_config_modality() {
        let cfg = ModernBertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_heads: 12,
            num_layers: 12,
            intermediate_size: 3072,
            layer_norm_eps: 1e-5,
            max_seq_len: 512,
        };
        assert_eq!(cfg.name(), "modern-bert");
        assert_eq!(cfg.modality(), ModalityHint::TextInTextOut);
    }

    #[test]
    fn test_nomic_bert_config_modality() {
        let cfg = NomicBertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_heads: 12,
            num_layers: 12,
            intermediate_size: 3072,
            layer_norm_eps: 1e-5,
            max_seq_len: 2048,
        };
        assert_eq!(cfg.name(), "nomic-bert");
        assert_eq!(cfg.modality(), ModalityHint::TextInTextOut);
    }

    #[test]
    fn test_t5_encoder_config_modality() {
        let cfg = T5EncoderConfig {
            vocab_size: 32128,
            hidden_size: 512,
            num_heads: 8,
            num_layers: 6,
            intermediate_size: 2048,
            rms_norm_eps: 1e-6,
            max_seq_len: 512,
        };
        assert_eq!(cfg.name(), "t5encoder");
        assert_eq!(cfg.modality(), ModalityHint::TextInTextOut);
    }
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
    fn name(&self) -> &str {
        "t5encoder"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
