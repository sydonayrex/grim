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
    pub expert_feed_forward_length: Option<usize>,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    // SSM specific
    pub ssm_d_state: Option<usize>,
    pub ssm_d_inner: Option<usize>,
    pub ssm_d_conv: Option<usize>,
    pub ssm_dt_rank: Option<usize>,
    pub ssm_n_group: Option<usize>,
    pub full_attention_interval: Option<usize>,
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
            expert_feed_forward_length: None,
            routed_scaling_factor: 1.0,
            norm_topk_prob: false,
            ssm_d_state: None,
            ssm_d_inner: None,
            ssm_d_conv: None,
            ssm_dt_rank: None,
            ssm_n_group: None,
            full_attention_interval: None,
        }
    }
}

impl ArchHyperparameters {
    /// Computes the semantic-demand lower bound $U_{\text{sum}}$ for worst-case prefill sequence.
    ///
    /// Returns: (static_parameter_bytes, semantic_demand_bytes, kv_cache_bytes, peak_activation_bytes, demanded_experts, total_experts)
    pub fn compute_detailed_memory_bounds(
        &self,
        target_seq_len: usize,
        batch_size: usize,
        bytes_per_elem: usize,
    ) -> (u64, u64, u64, u64, usize, usize) {
        let bpe = bytes_per_elem as u64;
        let d = self.hidden_size as u64;
        let l = self.num_layers as u64;
        let v = self.vocab_size as u64;
        let intermediate = self.intermediate_size as u64;
        let kv_heads = self.num_kv_heads as u64;
        let q_heads = self.num_heads as u64;
        let head_dim = self.head_dim as u64;

        // Base attention + norm parameters per layer
        let qkv_proj = d * head_dim * (q_heads + 2 * kv_heads);
        let out_proj = d * d;
        let attn_layer_params = qkv_proj + out_proj + 2 * d; // + norms

        // FFN parameters
        let (total_static_bytes, semantic_demand_bytes, demanded_experts, total_experts) =
            if let Some(num_experts) = self.expert_count {
                let top_k = self.expert_used_count.unwrap_or(2) as u64;
                let exp_ffn = self.expert_feed_forward_length.unwrap_or(self.intermediate_size) as u64;
                let per_expert_ffn = 3 * d * exp_ffn; // SwiGLU: gate + up + down
                let shared_ffn = 3 * d * intermediate;

                let static_params = (2 * v * d) // embed + lm_head
                    + l * (attn_layer_params + shared_ffn + (num_experts as u64) * per_expert_ffn);

                // Semantic demand: for sequence length S with top-k routing,
                // worst-case distinct experts demanded across sequence = min(S * top_k, E) per layer
                let active_per_layer = ((target_seq_len as u64) * top_k).min(num_experts as u64);
                let demanded_params = (2 * v * d)
                    + l * (attn_layer_params + shared_ffn + active_per_layer * per_expert_ffn);

                (
                    static_params * bpe,
                    demanded_params * bpe,
                    (active_per_layer * l) as usize,
                    num_experts * self.num_layers,
                )
            } else {
                let dense_ffn = 3 * d * intermediate;
                let static_params = (2 * v * d) + l * (attn_layer_params + dense_ffn);
                (
                    static_params * bpe,
                    static_params * bpe,
                    0,
                    0,
                )
            };

        // KV cache reservation: 2 * L * B * S * N_kv * H_dim * bpe
        let kv_cache_bytes = 2 * l * (batch_size as u64) * (target_seq_len as u64) * kv_heads * head_dim * bpe;

        // Peak working activation buffer: 2 * B * S * D * bpe
        let peak_activation_bytes = 2 * (batch_size as u64) * (target_seq_len as u64) * d * bpe;

        (
            total_static_bytes,
            semantic_demand_bytes,
            kv_cache_bytes,
            peak_activation_bytes,
            demanded_experts,
            total_experts,
        )
    }

    /// Returns the semantic-demand lower bound in bytes for a target context and batch size.
    pub fn semantic_demand_lower_bound(&self, target_seq_len: usize, batch_size: usize, bytes_per_elem: usize) -> u64 {
        let (_, demand, kv, act, _, _) = self.compute_detailed_memory_bounds(target_seq_len, batch_size, bytes_per_elem);
        demand + kv + act
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
    pub fn extract<M: MetadataLookup>(
        arch: ModelArchitecture,
        metadata: &M,
    ) -> ArchHyperparameters {
        // SmolLM2 is exported by llama.cpp under `general.architecture = "llama"`
        // and carries `llama.*` hyperparameter keys. Use those as the lookup
        // prefix and prefer them over the often-stale `tokenizer.ggml.vocab_size`
        // key (which some SmolLM2 exports populate with a wrong value and would
        // otherwise swap vocab/hidden).
        let is_smollm2 = arch == ModelArchitecture::SmolLm2;
        let arch_name = if is_smollm2 { "llama" } else { arch.as_str() };

        let vocab_size = if is_smollm2 {
            metadata
                .get_u32("llama.vocab_size")
                .or_else(|| metadata.get_u32("tokenizer.ggml.vocab_size"))
                .or_else(|| metadata.get_u32("tokenizer.ggml.tokens"))
                .map(|v| v as usize)
                .unwrap_or(32000)
        } else {
            metadata
                .get_u32("tokenizer.ggml.vocab_size")
                .or_else(|| metadata.get_u32("tokenizer.ggml.tokens"))
                .or_else(|| metadata.get_u32(&format!("{arch_name}.vocab_size")))
                .or_else(|| metadata.get_u32("llama.vocab_size"))
                .map(|v| v as usize)
                .unwrap_or(32000)
        };

        let hidden_size = metadata
            .get_u32(&format!("{arch_name}.embedding_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.hidden_size")))
            .or_else(|| metadata.get_u32("llama.embedding_length"))
            .or_else(|| metadata.get_u32("llama.hidden_size"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        let num_layers = metadata
            .get_u32(&format!("{arch_name}.block_count"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.num_hidden_layers")))
            .or_else(|| metadata.get_u32("llama.block_count"))
            .or_else(|| metadata.get_u32("llama.num_hidden_layers"))
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_heads = metadata
            .get_u32(&format!("{arch_name}.attention.head_count"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.num_attention_heads")))
            .or_else(|| metadata.get_u32("llama.attention.head_count"))
            .or_else(|| metadata.get_u32("llama.num_attention_heads"))
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_kv_heads = metadata
            .get_u32(&format!("{arch_name}.attention.head_count_kv"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.num_key_value_heads")))
            .or_else(|| metadata.get_u32("llama.attention.head_count_kv"))
            .or_else(|| metadata.get_u32("llama.num_key_value_heads"))
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let head_dim = metadata
            .get_u32(&format!("{arch_name}.attention.key_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.head_dim")))
            .or_else(|| metadata.get_u32("llama.attention.key_length"))
            .or_else(|| metadata.get_u32("llama.head_dim"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if num_heads > 0 {
                    hidden_size.checked_div(num_heads).unwrap_or(hidden_size)
                } else {
                    128
                }
            });

        let intermediate_size = metadata
            .get_u32(&format!("{arch_name}.feed_forward_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.intermediate_size")))
            .or_else(|| metadata.get_u32("llama.feed_forward_length"))
            .or_else(|| metadata.get_u32("llama.intermediate_size"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);

        let rms_norm_eps = metadata
            .get_f32(&format!("{arch_name}.attention.layer_norm_rms_epsilon"))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.attention.layer_norm_rms_eps")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.attention.layer_norm_epsilon")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rms_norm_eps")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rms_norm_epsilon")))
            .or_else(|| metadata.get_f32("llama.attention.layer_norm_rms_epsilon"))
            .or_else(|| metadata.get_f32("llama.attention.layer_norm_rms_eps"))
            .or_else(|| metadata.get_f32("llama.attention.layer_norm_epsilon"))
            .or_else(|| metadata.get_f32("llama.rms_norm_eps"))
            .or_else(|| metadata.get_f32("llama.rms_norm_epsilon"))
            .unwrap_or(1e-5);

        let rope_theta = metadata
            .get_f32(&format!("{arch_name}.rope.freq_base"))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rope_freq_base")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rope_theta")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.rope_parameters.rope_theta")))
            .or_else(|| metadata.get_f32("llama.rope.freq_base"))
            .or_else(|| metadata.get_f32("llama.rope_freq_base"))
            .or_else(|| metadata.get_f32("llama.rope_theta"))
            .or_else(|| metadata.get_f32("rope.freq_base"))
            .or_else(|| metadata.get_f32("rope_freq_base"))
            .or_else(|| metadata.get_f32("rope_theta"))
            .unwrap_or(10000.0);

        let max_seq_len = metadata
            .get_u32(&format!("{arch_name}.context_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.max_position_embeddings")))
            .or_else(|| metadata.get_u32("llama.context_length"))
            .or_else(|| metadata.get_u32("llama.max_position_embeddings"))
            .map(|v| v as usize)
            .unwrap_or(2048);

        let expert_count = metadata
            .get_u32(&format!("{arch_name}.expert_count"))
            .map(|v| v as usize);
        let expert_used_count = metadata
            .get_u32(&format!("{arch_name}.expert_used_count"))
            .map(|v| v as usize);
        let expert_feed_forward_length = metadata
            .get_u32(&format!("{arch_name}.expert_feed_forward_length"))
            .or_else(|| metadata.get_u32(&format!("{arch_name}.expert_intermediate_size")))
            .map(|v| v as usize);
        let routed_scaling_factor = metadata
            .get_f32(&format!("{arch_name}.routed_scaling_factor"))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.moe_routed_scaling_factor")))
            .unwrap_or(1.0);

        let norm_topk_prob = metadata
            .get_u32(&format!("{arch_name}.norm_topk_prob"))
            .map(|v| v != 0)
            .unwrap_or(false);

        let ssm_d_state = metadata
            .get_u32(&format!("{arch_name}.ssm.state_size"))
            .map(|v| v as usize);
        let ssm_d_inner = metadata
            .get_u32(&format!("{arch_name}.ssm.inner_size"))
            .map(|v| v as usize);
        let ssm_d_conv = metadata
            .get_u32(&format!("{arch_name}.ssm.conv_kernel"))
            .map(|v| v as usize);
        let ssm_dt_rank = metadata
            .get_u32(&format!("{arch_name}.ssm.time_step_rank"))
            .map(|v| v as usize);
        let ssm_n_group = metadata
            .get_u32(&format!("{arch_name}.ssm.group_count"))
            .map(|v| v as usize);
        let full_attention_interval = metadata
            .get_u32(&format!("{arch_name}.full_attention_interval"))
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
            expert_feed_forward_length,
            routed_scaling_factor,
            norm_topk_prob,
            ssm_d_state,
            ssm_d_inner,
            ssm_d_conv,
            ssm_dt_rank,
            ssm_n_group,
            full_attention_interval,
        }
    }
}

#[cfg(test)]
mod extract_reference_tests {
    use super::*;
    use std::collections::HashMap;

    /// HashMap-backed `MetadataLookup` for the fallback-chain tests.
    #[derive(Default)]
    struct MockMeta(HashMap<String, String>);

    impl MockMeta {
        fn u32(mut self, pairs: &[(&str, u32)]) -> Self {
            for (k, v) in pairs {
                self.0.insert(k.to_string(), v.to_string());
            }
            self
        }
    }

    impl MetadataLookup for MockMeta {
        fn get_str(&self, key: &str) -> Option<String> {
            self.0.get(key).cloned()
        }
        fn get_u32(&self, key: &str) -> Option<u32> {
            self.0.get(key).and_then(|v| v.parse().ok())
        }
        fn get_f32(&self, key: &str) -> Option<f32> {
            self.0.get(key).and_then(|v| v.parse().ok())
        }
    }

    /// Empty metadata yields the documented defaults.
    #[test]
    fn empty_metadata_yields_defaults() {
        let meta = MockMeta::default();
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Llama, &meta);
        assert_eq!(hp.vocab_size, 32000);
        assert_eq!(hp.hidden_size, 4096);
        assert_eq!(hp.num_layers, 32);
        assert_eq!(hp.num_heads, 32);
        // num_kv_heads falls back to num_heads (MHA), not a constant.
        assert_eq!(hp.num_kv_heads, 32);
    }

    /// Architecture-specific keys win over llama.* fallbacks.
    #[test]
    fn arch_specific_keys_beat_llama_fallbacks() {
        let meta = MockMeta::default().u32(&[
            ("qwen3moe.embedding_length", 2048),
            ("qwen3moe.block_count", 48),
            ("llama.embedding_length", 1111),
            ("llama.block_count", 22),
            ("tokenizer.ggml.vocab_size", 151936),
        ]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Qwen3Moe, &meta);
        assert_eq!(hp.hidden_size, 2048);
        assert_eq!(hp.num_layers, 48);
        assert_eq!(hp.vocab_size, 151936);
    }

    /// When the arch-specific key is absent, llama.* keys are consulted
    /// before the hardcoded defaults (the llama.cpp-export path).
    #[test]
    fn llama_fallback_keys_are_consulted() {
        let meta = MockMeta::default().u32(&[
            ("llama.embedding_length", 896),
            ("llama.block_count", 4),
            ("llama.attention.head_count", 6),
            ("llama.attention.head_count_kv", 2),
            ("tokenizer.ggml.vocab_size", 49152),
        ]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Qwen3, &meta);
        assert_eq!(hp.hidden_size, 896);
        assert_eq!(hp.num_layers, 4);
        assert_eq!(hp.num_heads, 6);
        assert_eq!(hp.num_kv_heads, 2);
        assert_eq!(hp.vocab_size, 49152);
    }

    /// SmolLm2 special case: llama.* keys are PREFERRED over
    /// tokenizer.ggml.vocab_size (the documented stale-vocab workaround).
    #[test]
    fn smollm2_prefers_llama_vocab_key() {
        let meta = MockMeta::default().u32(&[
            ("llama.vocab_size", 49152),
            ("tokenizer.ggml.vocab_size", 999), // known-stale key
            ("llama.embedding_length", 576),
        ]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::SmolLm2, &meta);
        assert_eq!(hp.vocab_size, 49152, "llama.vocab_size must win over the stale tokenizer key");
        assert_eq!(hp.hidden_size, 576);
    }

    /// GQA default: when only num_heads is known, num_kv_heads inherits it.
    #[test]
    fn kv_heads_inherit_heads_when_absent() {
        let meta = MockMeta::default().u32(&[("llama.attention.head_count", 8)]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Llama, &meta);
        assert_eq!(hp.num_heads, 8);
        assert_eq!(hp.num_kv_heads, 8);
    }
}
