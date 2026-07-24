//! Centralized hyperparameter extraction for all supported model architectures.
//!
//! Provides `ArchHyperparameters` and a metadata extraction table that resolves model parameters
//! from GGUF and HuggingFace config metadata keys.

use crate::architecture::ModelArchitecture;

/// Resolved hyperparameter configuration extracted from GGUF or Safetensors metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchHyperparameters {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    // MoE specific
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
    // SSM specific
    pub ssm_d_state: Option<usize>,
    pub ssm_d_inner: Option<usize>,
    pub ssm_d_conv: Option<usize>,
}

impl Default for ArchHyperparameters {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            expert_count: None,
            expert_used_count: None,
            ssm_d_state: None,
            ssm_d_inner: None,
            ssm_d_conv: None,
        }
    }
}

/// Metadata accessor abstraction for unified GGUF / HF metadata resolution.
pub trait MetadataLookup {
    /// Retrieve string metadata by key.
    fn get_str(&self, key: &str) -> Option<String>;
    /// Retrieve u32 metadata by key with fallback.
    fn get_u32(&self, key: &str) -> Option<u32>;
    /// Retrieve f32 metadata by key with fallback.
    fn get_f32(&self, key: &str) -> Option<f32>;
}

/// Hyperparameter extraction engine that queries metadata based on architecture conventions.
pub struct HyperparameterExtractor;

impl HyperparameterExtractor {
    /// Extract `ArchHyperparameters` from a `MetadataLookup` provider for the specified architecture.
    pub fn extract<M: MetadataLookup>(arch: ModelArchitecture, metadata: &M) -> ArchHyperparameters {
        let arch_name = arch.as_str();

        let vocab_size = metadata.get_u32("tokenizer.ggml.vocab_size")
            .or_else(|| metadata.get_u32(&format!("{arch_name}.vocab_size")))
            .map(|v| v as usize)
            .unwrap_or(32000);

        let hidden_size = metadata.get_u32(&format!("{arch_name}.embedding_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.hidden_size")))
            .map(|v| v as usize)
            .unwrap_or(4096);

        let num_layers = metadata.get_u32(&format!("{arch_name}.block_count"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.num_hidden_layers")))
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_heads = metadata.get_u32(&format!("{arch_name}.attention.head_count"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.num_attention_heads")))
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_kv_heads = metadata.get_u32(&format!("{arch_name}.attention.head_count_kv"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.num_key_value_heads")))
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let head_dim = metadata.get_u32(&format!("{arch_name}.attention.key_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.head_dim")))
            .map(|v| v as usize)
            .unwrap_or_else(|| if num_heads > 0 { hidden_size / num_heads } else { 128 });

        let intermediate_size = metadata.get_u32(&format!("{arch_name}.feed_forward_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.intermediate_size")))
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);

        let rms_norm_eps = metadata.get_f32(&format!("{arch_name}.attention.layer_norm_rms_eps"))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.attention.layer_norm_epsilon")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rms_norm_eps")))
            .unwrap_or(1e-5);

        let rope_theta = metadata.get_f32(&format!("{arch_name}.rope.freq_base"))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rope_theta")))
            .unwrap_or(10000.0);

        let max_seq_len = metadata.get_u32(&format!("{arch_name}.context_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.max_position_embeddings")))
            .map(|v| v as usize)
            .unwrap_or(2048);

        let expert_count = metadata.get_u32(&format!("{arch_name}.expert_count"))
            .map(|v| v as usize);
        let expert_used_count = metadata.get_u32(&format!("{arch_name}.expert_used_count"))
            .map(|v| v as usize);

        let ssm_d_state = metadata.get_u32(&format!("{arch_name}.ssm.state_size"))
            .map(|v| v as usize);
        let ssm_d_inner = metadata.get_u32(&format!("{arch_name}.ssm.inner_size"))
            .map(|v| v as usize);
        let ssm_d_conv = metadata.get_u32(&format!("{arch_name}.ssm.conv_kernel"))
            .map(|v| v as usize);

        ArchHyperparameters {
            architecture: arch,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            max_seq_len,
            expert_count,
            expert_used_count,
            ssm_d_state,
            ssm_d_inner,
            ssm_d_conv,
        }
    }
}
