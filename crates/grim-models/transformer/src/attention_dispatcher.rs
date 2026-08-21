//! Universal 3-Tier Attention Dispatcher for GRIM.
//!
//! Routes multi-head, grouped-query, DeepSeek MLA, Paged Quantized KV,
//! and SageAttention requests across hardware matrix cores, universal compute shaders,
//! and CPU fallback implementations.

use grim_tensor::dtype::QuantFormat;
use grim_tensor::tensor::Tensor;

/// Attention mechanism topology variant.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionTopology {
    /// Standard Multi-Head or Grouped-Query Attention (Llama, Mistral, Qwen).
    StandardGqa {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
    },
    /// DeepSeek Multi-Head Latent Attention (MLA) with compressed KV-cache.
    DeepSeekMla {
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        sm_scale: f32,
    },
    /// Paged block-quantized KV attention (Q8_0 or Q4_K block dequantization on-the-fly).
    PagedQuantizedKv {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        quant_format: QuantFormat,
        sm_scale: f32,
    },
    /// Block-Quantized SageAttention for ultra-long context windows.
    SageAttention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
    },
}

/// Execution tier selected by the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionTier {
    /// Tier 1: Hardware-accelerated Tensor/Matrix Cores (WMMA / MFMA / SIMDGroup / CoopMatrix).
    Tier1HardwareMatrix,
    /// Tier 2: Universal cross-backend compute shaders (HIP / MSL / Vulkan Compute).
    Tier2UniversalCompute,
    /// Tier 3: High-performance multi-threaded CPU reference fallback.
    Tier3CpuFallback,
}

/// Unified attention invocation payload.
#[derive(Debug, Clone)]
pub struct AttentionRequest {
    pub topology: AttentionTopology,
    pub causal: bool,
    pub sliding_window: Option<usize>,
}

/// Universal Attention Dispatcher.
pub struct AttentionDispatcher;

impl AttentionDispatcher {
    /// Classify the optimal execution tier based on device hardware capabilities and topology.
    pub fn select_tier(
        topology: &AttentionTopology,
        has_hardware_matrix: bool,
        is_gpu: bool,
    ) -> AttentionTier {
        if !is_gpu {
            return AttentionTier::Tier3CpuFallback;
        }

        match topology {
            AttentionTopology::StandardGqa { .. } => {
                if has_hardware_matrix {
                    AttentionTier::Tier1HardwareMatrix
                } else {
                    AttentionTier::Tier2UniversalCompute
                }
            }
            AttentionTopology::DeepSeekMla { .. }
            | AttentionTopology::PagedQuantizedKv { .. }
            | AttentionTopology::SageAttention { .. } => {
                // Specialized compute shader path
                AttentionTier::Tier2UniversalCompute
            }
        }
    }

    /// Derive the output tensor shape for an attention forward invocation.
    pub fn output_shape(q: &Tensor, req: &AttentionRequest) -> Vec<usize> {
        let q_dims = q.shape().dims();
        let (seq_len, num_heads, head_dim) = match req.topology {
            AttentionTopology::StandardGqa {
                num_heads,
                head_dim,
                ..
            }
            | AttentionTopology::SageAttention {
                num_heads,
                head_dim,
                ..
            } => {
                let s = if q_dims.len() >= 3 {
                    q_dims[q_dims.len() - 3]
                } else {
                    1
                };
                (s, num_heads, head_dim)
            }
            _ => (1, 1, 64),
        };

        vec![seq_len, num_heads, head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_selection_matrix_hardware() {
        let gqa = AttentionTopology::StandardGqa {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            sm_scale: 0.088388,
        };

        let tier_gpu_hw = AttentionDispatcher::select_tier(&gqa, true, true);
        assert_eq!(tier_gpu_hw, AttentionTier::Tier1HardwareMatrix);

        let tier_gpu_basic = AttentionDispatcher::select_tier(&gqa, false, true);
        assert_eq!(tier_gpu_basic, AttentionTier::Tier2UniversalCompute);

        let tier_cpu = AttentionDispatcher::select_tier(&gqa, false, false);
        assert_eq!(tier_cpu, AttentionTier::Tier3CpuFallback);
    }

    #[test]
    fn test_tier_selection_mla_and_sage() {
        let mla = AttentionTopology::DeepSeekMla {
            num_heads: 128,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            sm_scale: 0.072168,
        };

        let tier = AttentionDispatcher::select_tier(&mla, true, true);
        assert_eq!(tier, AttentionTier::Tier2UniversalCompute);

        let sage = AttentionTopology::SageAttention {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            sm_scale: 0.125,
        };
        let tier_sage = AttentionDispatcher::select_tier(&sage, true, true);
        assert_eq!(tier_sage, AttentionTier::Tier2UniversalCompute);
    }
}
