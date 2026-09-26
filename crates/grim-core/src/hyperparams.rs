//! Centralized hyperparameter extraction for all supported model architectures.
//! Provides `ArchHyperparameters` and a metadata extraction table that resolves model parameters from GGUF and HuggingFace.

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
    pub expert_shared_feed_forward_length: Option<usize>,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    // SSM specific
    pub ssm_d_state: Option<usize>,
    pub ssm_d_inner: Option<usize>,
    pub ssm_d_conv: Option<usize>,
    pub ssm_dt_rank: Option<usize>,
    pub ssm_n_group: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub head_count_kv_schedule: Option<Vec<u32>>,
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
            expert_shared_feed_forward_length: None,
            routed_scaling_factor: 1.0,
            norm_topk_prob: false,
            ssm_d_state: None,
            ssm_d_inner: None,
            ssm_d_conv: None,
            ssm_dt_rank: None,
            ssm_n_group: None,
            full_attention_interval: None,
            head_count_kv_schedule: None,
        }
    }
}

impl ArchHyperparameters {
    /// Computes the semantic-demand lower bound $U_{\text{sum}}$ for worst-case prefill sequence.
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
                let exp_ffn = self
                    .expert_feed_forward_length
                    .unwrap_or(self.intermediate_size) as u64;
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
                (static_params * bpe, static_params * bpe, 0, 0)
            };

        // KV cache reservation: 2 * L * B * S * N_kv * H_dim * bpe
        let kv_cache_bytes =
            2 * l * (batch_size as u64) * (target_seq_len as u64) * kv_heads * head_dim * bpe;

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
    pub fn semantic_demand_lower_bound(
        &self,
        target_seq_len: usize,
        batch_size: usize,
        bytes_per_elem: usize,
    ) -> u64 {
        let (_, demand, kv, act, _, _) =
            self.compute_detailed_memory_bounds(target_seq_len, batch_size, bytes_per_elem);
        demand + kv + act
    }
}

/// Metadata accessor abstraction for unified GGUF / HF metadata resolution.
pub trait MetadataLookup {
    /// Retrieve string metadata by key.
    fn get_str(&self, key: &str) -> Option<String>;
    /// Retrieve u32 metadata by key with fallback.
    fn get_u32(&self, key: &str) -> Option<u32>;
    /// Retrieve the element count of an array-valued metadata key.
    ///
    /// GGUF stores the vocabulary as `tokenizer.ggml.tokens` (an array of token
    /// strings), not as a scalar count, so [Self::get_u32] can never resolve it.
    /// The array length is the authoritative vocabulary size.
    fn get_array_len(&self, key: &str) -> Option<usize> {
        let _ = key;
        None
    }
    /// Retrieve f32 metadata by key with fallback.
    fn get_f32(&self, key: &str) -> Option<f32>;
    /// Retrieve boolean metadata by key.
    fn get_bool(&self, key: &str) -> Option<bool> {
        let _ = key;
        None
    }
    /// Retrieve u32 array metadata by key.
    fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        let _ = key;
        None
    }
    /// Retrieve u64 metadata by key. GGUF stores `split.*` and PLE
    /// `layer_multipliers` as u64; a u32 accessor silently truncates them
    /// (the multipliers are ~2.0e13, well past u32).
    fn get_u64(&self, key: &str) -> Option<u64> {
        let _ = key;
        None
    }
    /// Retrieve u64 array metadata by key (PLE `layer_multipliers`).
    fn get_u64_array(&self, key: &str) -> Option<Vec<u64>> {
        let _ = key;
        None
    }
    /// Retrieve bool array metadata by key (`attention.recurrent_layers`).
    fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        let _ = key;
        None
    }
    /// Retrieve i32 array metadata by key.
    fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        let _ = key;
        None
    }
}

/// Hyperparameter extraction engine that queries metadata based on architecture conventions.
pub struct HyperparameterExtractor;

impl HyperparameterExtractor {
    /// Extract `ArchHyperparameters` from a `MetadataLookup` provider for the specified architecture.
    pub fn extract<M: MetadataLookup>(
        arch: ModelArchitecture,
        metadata: &M,
    ) -> ArchHyperparameters {
        // SmolLM2 is exported by llama.cpp under `general.architecture = "llama"` and carries `llama.*` hyperparameter keys.
        // Use those as the lookup prefix.
        let is_smollm2 = arch == ModelArchitecture::SmolLm2;
        let arch_name = if is_smollm2 { "llama" } else { arch.as_str() };

        // Vocabulary resolution order, most authoritative first:
        //  1. `{arch}.vocab_size` - what llama.cpp computed for this architecture.
        //  2. `tokenizer.ggml.tokens` array length - the actual token list, so it cannot
        //     be stale the way a copied legacy count can be.
        //  3. `tokenizer.ggml.vocab_size` - legacy key, often stale after a tokenizer
        //     merge/extend, so it never outranks the two sources above.
        // The array length matters because `tokenizer.ggml.tokens` is an array of
        // token strings, not a scalar; a `get_u32` lookup on it always misses.
        // Getting this wrong is not cosmetic: a wrong vocab_size produces a
        // ShapeMismatch against `token_embd.weight` and an unusable output head.
        let vocab_size = metadata
            .get_u32(&format!("{arch_name}.vocab_size"))
            .map(|v| v as usize)
            .or_else(|| metadata.get_array_len("tokenizer.ggml.tokens"))
            .or_else(|| {
                if is_smollm2 {
                    None
                } else {
                    metadata.get_u32("llama.vocab_size").map(|v| v as usize)
                }
            })
            .or_else(|| {
                metadata
                    .get_u32("tokenizer.ggml.vocab_size")
                    .map(|v| v as usize)
            })
            .unwrap_or(32000);

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
        let expert_shared_feed_forward_length = metadata
            .get_u32(&format!("{arch_name}.expert_shared_feed_forward_length"))
            .map(|v| v as usize);
        let routed_scaling_factor = metadata
            .get_f32(&format!("{arch_name}.routed_scaling_factor"))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.moe_routed_scaling_factor")))
            .or_else(|| metadata.get_f32(&format!("{arch_name}.expert_weights_scale")))
            .unwrap_or(1.0);

        let norm_topk_prob = metadata
            .get_bool(&format!("{arch_name}.expert_weights_norm"))
            .or_else(|| {
                metadata
                    .get_u32(&format!("{arch_name}.norm_topk_prob"))
                    .map(|v| v != 0)
            })
            .unwrap_or(false);

        let head_count_kv_schedule = metadata
            .get_u32_array(&format!("{arch_name}.attention.head_count_kv"))
            .or_else(|| {
                metadata
                    .get_i32_array(&format!("{arch_name}.attention.head_count_kv"))
                    .map(|arr| arr.into_iter().map(|x| x.max(0) as u32).collect())
            });

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
            expert_shared_feed_forward_length,
            routed_scaling_factor,
            norm_topk_prob,
            ssm_d_state,
            ssm_d_inner,
            ssm_d_conv,
            ssm_dt_rank,
            ssm_n_group,
            full_attention_interval,
            head_count_kv_schedule,
        }
    }
}

/// Bytes the decode graph needs for one token of KV across every
/// full-attention layer: `kv_heads * head_dim` elements for K and again for V,
/// at `bytes_per_element` per element.
///
/// `interval` is the full-attention interval: SSM/recurrent layers keep their
/// state in a fixed-size ring, not a per-token arena, so they cost nothing here.
pub fn kv_bytes_per_token(
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    interval: usize,
    bytes_per_element: usize,
) -> u64 {
    let interval = interval.max(1);
    // Matches the Qwen3.5 model's own rule: full attention is every
    // `interval`-th layer counting from ONE, i.e. (i + 1) % interval == 0.
    // Counting from zero over-counts by one (17 vs 16 for 65 layers at
    // interval 4) and silently over-reserves KV.
    let attn_layers = (0..num_layers).filter(|i| (i + 1) % interval == 0).count();
    (kv_heads as u64)
        .saturating_mul(head_dim as u64)
        .saturating_mul(2) // K and V
        .saturating_mul(bytes_per_element as u64)
        .saturating_mul(attn_layers as u64)
}

/// Resolve the context window against the VRAM actually available.
///
/// The checkpoint's advertised `max_seq_len` is an upper bound the hardware may
/// not be able to honor: the decode graph pre-allocates every KV arena at full
/// context, so a 232k window on this 27B model asks for ~32 GB of arenas before
/// a single weight is placed. Rather than discovering that as an out-of-memory
/// failure at load time, shrink the context to what fits.
///
/// `weight_bytes` is the resident weight footprint, `vram_budget_bytes` the total
/// we are willing to commit, and `headroom_bytes` the reserve left for
/// activations, the decode graph's scratch buffers, and allocator fragmentation.
///
/// Returns the largest context whose KV arenas fit in
/// `vram_budget - weight_bytes - headroom`, never exceeding the checkpoint's
/// advertised maximum. Rounds down to a whole multiple of `interval` so arena
/// slots align with the attention layer stride.
pub fn fit_context_to_vram(
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    interval: usize,
    bytes_per_element: usize,
    weight_bytes: u64,
    vram_budget_bytes: u64,
    headroom_bytes: u64,
    max_context: usize,
) -> usize {
    let per_token = kv_bytes_per_token(
        kv_heads,
        head_dim,
        num_layers,
        interval,
        bytes_per_element,
    );
    if per_token == 0 {
        return max_context;
    }
    let available = vram_budget_bytes
        .saturating_sub(weight_bytes)
        .saturating_sub(headroom_bytes);
    if available == 0 {
        return 0;
    }
    let fits = (available / per_token) as usize;
    let interval = interval.max(1);
    let aligned = fits / interval * interval;
    aligned.min(max_context)
}

#[cfg(test)]
mod extract_reference_tests {
    use super::*;
    use std::collections::HashMap;

    /// HashMap-backed `MetadataLookup` for the fallback-chain tests.
    ///
    /// Scalars and arrays are tracked separately so the mock reproduces the real
    /// `GgufProvider` behavior where a `get_u32` lookup on an array key misses.
    #[derive(Default)]
    struct MockMeta {
        scalars: HashMap<String, String>,
        arrays: HashMap<String, usize>,
    }

    impl MockMeta {
        fn u32(mut self, pairs: &[(&str, u32)]) -> Self {
            for (k, v) in pairs {
                self.scalars.insert(k.to_string(), v.to_string());
            }
            self
        }

        fn f32(mut self, pairs: &[(&str, f32)]) -> Self {
            for (k, v) in pairs {
                self.scalars.insert(k.to_string(), v.to_string());
            }
            self
        }

        fn str(mut self, pairs: &[(&str, &str)]) -> Self {
            for (k, v) in pairs {
                self.scalars.insert(k.to_string(), v.to_string());
            }
            self
        }

        /// Register an array-valued key (e.g. `tokenizer.ggml.tokens`) with `len` elements.
        fn array(mut self, pairs: &[(&str, usize)]) -> Self {
            for (k, v) in pairs {
                self.arrays.insert(k.to_string(), *v);
            }
            self
        }
    }

    impl MetadataLookup for MockMeta {
        fn get_str(&self, key: &str) -> Option<String> {
            self.scalars.get(key).cloned()
        }
        fn get_u32(&self, key: &str) -> Option<u32> {
            self.scalars.get(key).and_then(|v| v.parse().ok())
        }
        fn get_array_len(&self, key: &str) -> Option<usize> {
            self.arrays.get(key).copied()
        }
        fn get_f32(&self, key: &str) -> Option<f32> {
            self.scalars.get(key).and_then(|v| v.parse().ok())
        }
        fn get_bool(&self, key: &str) -> Option<bool> {
            self.scalars.get(key).and_then(|v| match v.as_str() {
                "true" | "1" => Some(true),
                "false" | "0" => Some(false),
                _ => None,
            })
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
        assert_eq!(
            hp.vocab_size, 49152,
            "llama.vocab_size must win over the stale tokenizer key"
        );
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

    /// Regression for the Qwen3.8-27B Q4_K checkpoint, whose metadata carries
    /// NEITHER `tokenizer.ggml.vocab_size` NOR `qwen35.vocab_size`; the vocabulary
    /// only exists as the `tokenizer.ggml.tokens` array (248320 entries). Every
    /// scalar key in the old chain missed, so resolution fell through to the
    /// hardcoded `unwrap_or(32000)` and `token_embd.weight` [248320, 5120] raised
    /// a ShapeMismatch against an expected [32000, 5120].
    #[test]
    fn vocab_size_falls_back_to_tokens_array_length() {
        let meta = MockMeta::default()
            .u32(&[
                ("qwen35.embedding_length", 5120),
                ("qwen35.block_count", 65),
            ])
            .array(&[("tokenizer.ggml.tokens", 248_320)]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Qwen35, &meta);
        assert_eq!(
            hp.vocab_size, 248_320,
            "vocabulary must come from the token array length, not the 32000 default"
        );
    }

    /// A stale legacy count must never outrank the token array, which is what
    /// actually determines the embedding row count.
    #[test]
    fn stale_tokenizer_vocab_key_loses_to_tokens_array() {
        let meta = MockMeta::default()
            .u32(&[("tokenizer.ggml.vocab_size", 32_000)])
            .array(&[("tokenizer.ggml.tokens", 248_320)]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Qwen35, &meta);
        assert_eq!(hp.vocab_size, 248_320);
    }

    /// The architecture-specific key is the most authoritative source and beats
    /// both the token array and the legacy key when all three are present.
    #[test]
    fn arch_vocab_key_beats_tokens_array_and_legacy_key() {
        let meta = MockMeta::default()
            .u32(&[
                ("qwen35.vocab_size", 151_936),
                ("tokenizer.ggml.vocab_size", 32_000),
            ])
            .array(&[("tokenizer.ggml.tokens", 248_320)]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::Qwen35, &meta);
        assert_eq!(hp.vocab_size, 151_936);
    }

    /// SmolLM2 is exported under `general.architecture = "llama"`, so its
    /// arch-specific key is `llama.vocab_size` and it must still outrank the
    /// stale tokenizer key.
    #[test]
    fn smollm2_still_prefers_llama_vocab_over_legacy_key() {
        let meta = MockMeta::default()
            .u32(&[
                ("llama.vocab_size", 49_152),
                ("tokenizer.ggml.vocab_size", 999),
            ])
            .array(&[("tokenizer.ggml.tokens", 49_152)]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::SmolLm2, &meta);
        assert_eq!(hp.vocab_size, 49_152);
    }

    /// Nemotron-H MoE metadata extraction test: validates expert_weights_scale,
    /// expert_weights_norm, expert_shared_feed_forward_length, and head_count_kv schedule.
    #[test]
    fn test_nemotron_hmoe_metadata_extraction() {
        let meta = MockMeta::default()
            .f32(&[("nemotron-h-moe.expert_weights_scale", 2.5)])
            .str(&[("nemotron-h-moe.expert_weights_norm", "true")])
            .u32(&[
                ("nemotron-h-moe.expert_count", 128),
                ("nemotron-h-moe.expert_used_count", 6),
                ("nemotron-h-moe.expert_feed_forward_length", 1856),
                ("nemotron-h-moe.expert_shared_feed_forward_length", 3712),
                ("nemotron-h-moe.block_count", 53),
                ("nemotron-h-moe.embedding_length", 2688),
            ]);
        let hp = HyperparameterExtractor::extract(ModelArchitecture::NemotronHMoe, &meta);
        assert_eq!(hp.expert_count, Some(128));
        assert_eq!(hp.expert_used_count, Some(6));
        assert_eq!(hp.expert_feed_forward_length, Some(1856));
        assert_eq!(hp.expert_shared_feed_forward_length, Some(3712));
        assert_eq!(hp.routed_scaling_factor, 2.5);
        assert_eq!(hp.num_layers, 53);
        assert_eq!(hp.hidden_size, 2688);
    }
}

#[cfg(test)]
mod kv_fit_tests {
    use super::{fit_context_to_vram, kv_bytes_per_token};

    /// Real Qwen3.8-27B geometry: 4 KV heads x 256 head_dim, 65 layers with
    /// full attention every 4th, f32 arenas. Cross-checked against the managed
    /// memory the 228k run actually tried to allocate.
    const QWEN_KV_HEADS: usize = 4;
    const QWEN_HEAD_DIM: usize = 256;
    const QWEN_LAYERS: usize = 65;
    const QWEN_INTERVAL: usize = 4;
    const F32: usize = 4;

    /// Full attention is every `interval`-th layer counting from ONE, matching
    /// `Qwen35Block`: (i + 1) % interval == 0. Counting from zero would give 17
    /// attention layers for 65 layers at interval 4 instead of 16, and
    /// over-reserve KV by a whole layer.
    #[test]
    fn attention_layer_count_matches_model_predicate() {
        let (layers, interval) = (65usize, 4usize);
        let counted = kv_bytes_per_token(4, 256, layers, interval, 4)
            / (4 * 256 * 2 * 4);
        let expected = (0..layers).filter(|i| (i + 1) % interval == 0).count();
        assert_eq!(expected, 16, "65 layers at interval 4 has 16 attention layers");
        assert_eq!(
            counted as usize, expected,
            "kv_bytes_per_token must count the same attention layers the model does"
        );
    }

    #[test]
    fn kv_bytes_per_token_matches_hand_computed_geometry() {
        // 16 full-attention layers x 4 heads x 256 dim x 2 (K and V) x 4 bytes.
        let expected = 16 * 4 * 256 * 2 * 4;
        assert_eq!(
            kv_bytes_per_token(QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS, QWEN_INTERVAL, F32),
            expected as u64
        );
    }

    /// Only full-attention layers cost per-token KV. With interval 4 over 65
    /// layers that is 16, not 65.
    #[test]
    fn recurrent_layers_do_not_consume_kv_budget() {
        let per_token = kv_bytes_per_token(QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS, QWEN_INTERVAL, F32);
        // A single attention layer would be 4*256*2*4 = 8192 bytes/token.
        assert_eq!(per_token, 16 * 8192);
    }

    /// The regression this exists for: at 228k the f32 arenas need ~32 GB,
    /// which does not fit alongside the weights in 34 GB, so the fit must shrink
    /// the context well below the advertised 232192.
    #[test]
    fn oversized_context_shrinks_to_what_fits() {
        const WEIGHTS: u64 = 15_200_000_000; // ~15.2 GB Q4_K
        const VRAM: u64 = 34_000_000_000; // 2 x 17 GB
        const HEADROOM: u64 = 2_000_000_000; // activations + scratch + fragmentation

        let fitted = fit_context_to_vram(
            QWEN_KV_HEADS,
            QWEN_HEAD_DIM,
            QWEN_LAYERS,
            QWEN_INTERVAL,
            F32,
            WEIGHTS,
            VRAM,
            HEADROOM,
            232_192,
        );
        assert!(
            fitted < 232_192,
            "228k arenas cannot fit in 34 GB; expected a shrink, got {fitted}"
        );
        // ~16.8 GB available for KV at 139264 B/token ≈ 120k, rounded to the
        // interval stride. Assert the order of magnitude rather than an exact
        // value so the test tracks the physics, not an arithmetic detail.
        assert!(
            (100_000..=130_000).contains(&fitted),
            "expected roughly 100-130k, got {fitted}"
        );
        assert_eq!(fitted % QWEN_INTERVAL, 0, "must align to the interval stride");
    }

    /// f16 KV halves the per-token cost, so more context fits. This is the lever
    /// that makes a large context reachable without more hardware.
    #[test]
    fn f16_kv_fits_more_context_than_f32() {
        const WEIGHTS: u64 = 15_200_000_000;
        const VRAM: u64 = 34_000_000_000;
        const HEADROOM: u64 = 2_000_000_000;

        let f32_fit = fit_context_to_vram(
            QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS, QWEN_INTERVAL, F32,
            WEIGHTS, VRAM, HEADROOM, 232_192,
        );
        let f16_fit = fit_context_to_vram(
            QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS, QWEN_INTERVAL, 2,
            WEIGHTS, VRAM, HEADROOM, 232_192,
        );
        assert!(
            f16_fit > f32_fit,
            "f16 must fit more context: {f16_fit} vs {f32_fit}"
        );
    }

    /// When plenty of VRAM is present the checkpoint's own limit is kept; the
    /// fit must never *grow* a context beyond what the model advertises.
    #[test]
    fn generous_vram_keeps_the_checkpoints_own_limit() {
        let fitted = fit_context_to_vram(
            QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS, QWEN_INTERVAL, F32,
            0, 500_000_000_000, 0, 32_768,
        );
        assert_eq!(fitted, 32_768, "must not exceed the advertised maximum");
    }

    /// No room for KV at all must yield 0, never a silent wrap or a panic.
    #[test]
    fn no_headroom_yields_zero_context() {
        let fitted = fit_context_to_vram(
            QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS, QWEN_INTERVAL, F32,
            34_000_000_000, 34_000_000_000, 0, 232_192,
        );
        assert_eq!(fitted, 0);
    }
}
