//! Universal 3-Tier Attention Dispatcher for GRIM.
//! Routes multi-head, grouped-query, DeepSeek MLA, Paged Quantized KV, and SageAttention requests across hardware matrix cores,.

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

    /// Execute a StandardGqa attention request end-to-end.
    /// This is the wiring point between the dispatcher's tier classification and actual execution: the shared.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gqa(
        q: &[f32],
        k_history: &[f32],
        v_history: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        steps: usize,
        window: Option<usize>,
        has_hardware_matrix: bool,
        device: &grim_tensor::Device,
    ) -> grim_core::error::Result<(Tensor, AttentionTier)> {
        let topology = AttentionTopology::StandardGqa {
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let is_gpu = !matches!(device, grim_tensor::Device::Cpu);
        let tier = Self::select_tier(&topology, has_hardware_matrix, is_gpu);
        let out = crate::shared_attention::fused_or_scalar_attention(
            q,
            k_history,
            v_history,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            window,
            device,
        )?;
        Ok((out, tier))
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

    /// Whether SWA bounded replay is enabled via `GRIM_SWA_BOUNDED_REPLAY=1`.
    pub fn swa_bounded_replay_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("GRIM_SWA_BOUNDED_REPLAY").as_deref() == Ok("1"))
    }

    /// Compute the SWA replay window `[replay_start, total_tokens)`.
    /// On a prefix match of length `matched_len`, if SWA is missing,
    /// replays at most `window` tokens of the cached prefix together with the uncached suffix.
    pub fn compute_swa_replay_range(
        matched_len: usize,
        total_tokens: usize,
        window: usize,
    ) -> std::ops::Range<usize> {
        let replay_start = matched_len.saturating_sub(window);
        replay_start..total_tokens
    }

    /// Construct a causal attention mask truncated to the replay segment:
    /// Query at absolute position `i` attends only over `[max(replay_start, i - window + 1), i]`.
    pub fn build_swa_truncated_mask(
        replay_start: usize,
        total_tokens: usize,
        window: usize,
    ) -> Vec<Vec<bool>> {
        let seq_len = total_tokens.saturating_sub(replay_start);
        let mut mask = vec![vec![false; seq_len]; seq_len];
        for (row, i) in (replay_start..total_tokens).enumerate() {
            let win_start = i.saturating_sub(window.saturating_sub(1)).max(replay_start);
            for (col, j) in (replay_start..total_tokens).enumerate() {
                if j >= win_start && j <= i {
                    mask[row][col] = true;
                }
            }
        }
        mask
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

    #[test]
    fn test_swa_bounded_replay_mechanics() {
        let n_win = 512;
        let matched = 3584; // Radix match length
        let total = 4096;   // Prefill length

        let range = AttentionDispatcher::compute_swa_replay_range(matched, total, n_win);
        assert_eq!(range.start, 3584 - 512); // exactly matched - n_win
        assert_eq!(range.end, 4096);
        assert_eq!(range.len(), 512 + (4096 - 3584)); // 512 + 512 = 1024 replay tokens

        // Build truncated mask for a small synthetic window:
        // total=8, matched=6, win=3 -> replay_start = 6 - 3 = 3 -> replay tokens 3..8 (len 5)
        let mask = AttentionDispatcher::build_swa_truncated_mask(3, 8, 3);
        assert_eq!(mask.len(), 5);
        for row in 0..5usize {
            let abs_i: usize = 3 + row;
            for col in 0..5usize {
                let abs_j: usize = 3 + col;
                let expected = abs_j <= abs_i && abs_j >= abs_i.saturating_sub(2);
                assert_eq!(
                    mask[row][col], expected,
                    "mismatch at row={row} (abs {abs_i}), col={col} (abs {abs_j})"
                );
            }
        }
    }
}
