//! GPU-Native Speculative Decoding Generation Loop.
//! Orchestrates draft token generation, target model verification, and device-side rejection sampling with minimal host-GPU synchronization.

use grim_tensor::backend::SamplingOps;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape, Tensor};

/// Configuration for the Speculative Decoding Engine.
#[derive(Debug, Clone)]
pub struct SpeculativeLoopConfig {
    pub num_draft_tokens: usize,
    pub temperature: f32,
    pub max_tokens: usize,
}

impl Default for SpeculativeLoopConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
            temperature: 0.7,
            max_tokens: 128,
        }
    }
}

/// Statistics collected during speculative decoding.
#[derive(Debug, Default, Clone)]
pub struct SpeculativeStats {
    pub total_accepted_tokens: usize,
    pub total_draft_tokens: usize,
    pub num_target_forward_passes: usize,
}

impl SpeculativeStats {
    /// Acceptance rate: ratio of accepted tokens to generated draft tokens.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_draft_tokens == 0 {
            0.0
        } else {
            self.total_accepted_tokens as f64 / self.total_draft_tokens as f64
        }
    }
}

/// Speculative decoding driver.
pub struct SpeculativeLoop {
    pub config: SpeculativeLoopConfig,
    pub stats: SpeculativeStats,
}

impl SpeculativeLoop {
    pub fn new(config: SpeculativeLoopConfig) -> Self {
        Self {
            config,
            stats: SpeculativeStats::default(),
        }
    }

    /// Run a speculative decode step with draft tokens.
    pub fn verify_draft_step(
        &mut self,
        draft_tokens: &[u32],
        target_token: u32,
        accepted_count: usize,
    ) {
        self.stats.total_draft_tokens += draft_tokens.len();
        self.stats.total_accepted_tokens += accepted_count;
        self.stats.num_target_forward_passes += 1;
        let _ = target_token;
    }

    /// Device-native rejection sampling verification across draft token sequence.
    /// Given draft token probabilities `p_draft` and target model probabilities `p_target`, evaluates exact rejection sampling criteria:.
    pub fn verify_tokens_with_rejection_sampling(
        &mut self,
        draft_tokens: &[u32],
        p_draft: &[f32],
        p_target: &[f32],
        random_uniform: &[f32],
    ) -> Vec<u32> {
        let mut accepted = Vec::with_capacity(draft_tokens.len() + 1);
        let n = draft_tokens.len().min(p_draft.len()).min(p_target.len());

        for i in 0..n {
            let p_d = p_draft[i].max(1e-8);
            let p_t = p_target[i];
            let ratio = (p_t / p_d).min(1.0);
            let r = if i < random_uniform.len() {
                random_uniform[i]
            } else {
                // Short random buffer cannot verify token; stop and reject
                break;
            };

            if r <= ratio {
                accepted.push(draft_tokens[i]);
            } else {
                // First rejection stops speculation chain
                break;
            }
        }

        self.verify_draft_step(draft_tokens, 0, accepted.len());
        accepted
    }

    /// GPU-accelerated speculative rejection sampling on ROCm/device tensors (Task 10.6).
    pub fn verify_tokens_gpu(
        &mut self,
        target_probs: &Tensor,
        draft_probs: &Tensor,
        draft_tokens: &[u32],
        uniform_rands: &[f32],
        vocab_size: usize,
    ) -> grim_core::error::Result<Vec<u32>> {
        let batch_size = 1;
        let num_draft = draft_tokens.len();

        let dev = match target_probs.device() {
            grim_tensor::Device::Rocm(ord) => grim_backend_rocm::device::roc_device::RocmDevice::try_new(*ord as usize)
                .map_err(|e| grim_core::error::Error::Backend(e.to_string()))?,
            _ => {
                // Fall back to CPU verification
                let p_t = target_probs.to_vec_f32().map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;
                let p_d = draft_probs.to_vec_f32().map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;
                return Ok(self.verify_tokens_with_rejection_sampling(draft_tokens, &p_d, &p_t, uniform_rands));
            }
        };

        let dt_i32: Vec<i32> = draft_tokens.iter().map(|&t| t as i32).collect();
        let dt_storage = dev.from_cpu(&dt_i32.iter().map(|&x| x as f32).collect::<Vec<_>>(), &Shape::new(vec![batch_size, num_draft]), DType::F32)
            .map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;
        let ur_storage = dev.from_cpu(uniform_rands, &Shape::new(vec![batch_size, num_draft + 1]), DType::F32)
            .map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;
        let at_storage = dev.alloc_storage(&Shape::new(vec![batch_size, num_draft + 1]), DType::F32)
            .map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;
        let al_storage = dev.alloc_storage(&Shape::new(vec![batch_size]), DType::F32)
            .map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;

        dev.speculative_rejection_sample(
            target_probs.storage().as_ref(),
            draft_probs.storage().as_ref(),
            dt_storage.as_ref(),
            ur_storage.as_ref(),
            at_storage.as_ref(),
            al_storage.as_ref(),
            batch_size,
            num_draft,
            vocab_size,
        ).map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;

        let al_vec = al_storage.to_cpu_vec_f32().map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;
        let accepted_len = al_vec.get(0).copied().unwrap_or(0.0) as usize;
        let at_vec = at_storage.to_cpu_vec_f32().map_err(|e| grim_core::error::Error::Backend(e.to_string()))?;

        let accepted: Vec<u32> = at_vec.into_iter().take(accepted_len).map(|t| t as u32).collect();
        self.verify_draft_step(draft_tokens, 0, accepted.len());
        Ok(accepted)
    }

    /// Multi-head Medusa / Eagle tree speculative verification.
    /// Evaluates speculative tree candidates in parallel and selects the longest valid accepted prefix path.
    pub fn verify_medusa_tree_candidates(
        &mut self,
        tree_paths: &[Vec<u32>],
        target_token_matches: &[bool],
    ) -> Vec<u32> {
        let mut longest_accepted: Vec<u32> = Vec::new();

        for (path_idx, path) in tree_paths.iter().enumerate() {
            let mut current_accepted = Vec::new();
            for (token_idx, &token) in path.iter().enumerate() {
                let match_idx = path_idx * path.len() + token_idx;
                let is_match =
                    match_idx < target_token_matches.len() && target_token_matches[match_idx];
                if is_match {
                    current_accepted.push(token);
                } else {
                    break;
                }
            }
            if current_accepted.len() > longest_accepted.len() {
                longest_accepted = current_accepted;
            }
        }

        let total_draft: usize = tree_paths.iter().map(|p| p.len()).sum();
        self.stats.total_draft_tokens += total_draft;
        self.stats.total_accepted_tokens += longest_accepted.len();
        self.stats.num_target_forward_passes += 1;

        longest_accepted
    }

    /// Host-side bookkeeping for one draft round: records the previous round's verification stats and returns how many next-round draft candidates are staged.
    /// **This is sequential accounting, not pipelining.** No second stream is created and nothing overlaps: `verify_draft_step`.
    pub fn settle_draft_round(
        &mut self,
        prev_draft_tokens: &[u32],
        prev_target_token: u32,
        prev_accepted_count: usize,
        next_draft_candidates: &[u32],
    ) -> usize {
        self.verify_draft_step(prev_draft_tokens, prev_target_token, prev_accepted_count);
        // Returns the number of new speculative tokens ready for the next iteration
        next_draft_candidates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_stats_acceptance_rate() {
        let mut loop_engine = SpeculativeLoop::new(SpeculativeLoopConfig::default());
        loop_engine.verify_draft_step(&[10, 20, 30, 40], 50, 3);

        assert_eq!(loop_engine.stats.total_draft_tokens, 4);
        assert_eq!(loop_engine.stats.total_accepted_tokens, 3);
        assert_eq!(loop_engine.stats.num_target_forward_passes, 1);
        assert!((loop_engine.stats.acceptance_rate() - 0.75).abs() < 1e-4);
    }

    #[test]
    fn test_rejection_sampling_verification() {
        let mut loop_engine = SpeculativeLoop::new(SpeculativeLoopConfig::default());
        let draft_tokens = vec![101, 102, 103, 104];
        let p_draft = vec![0.8, 0.7, 0.6, 0.5];
        let p_target = vec![0.9, 0.8, 0.2, 0.5]; // 3rd token has low target prob
        let random_uniform = vec![0.5, 0.5, 0.8, 0.1]; // 3rd token 0.8 > (0.2/0.6=0.33) -> rejected

        let accepted = loop_engine.verify_tokens_with_rejection_sampling(
            &draft_tokens,
            &p_draft,
            &p_target,
            &random_uniform,
        );

        assert_eq!(accepted, vec![101, 102]);
        assert_eq!(loop_engine.stats.total_accepted_tokens, 2);
    }

    #[test]
    fn test_medusa_tree_candidates_verification() {
        let mut loop_engine = SpeculativeLoop::new(SpeculativeLoopConfig::default());
        let candidate_paths = vec![vec![10, 20, 30], vec![10, 20, 35], vec![10, 25, 40]];
        // 1st path: 10 (ok), 20 (ok), 30 (mismatch) -> len 2
        // 2nd path: 10 (ok), 20 (ok), 35 (ok) -> len 3
        let matches = vec![true, true, false, true, true, true, true, false, false];

        let longest = loop_engine.verify_medusa_tree_candidates(&candidate_paths, &matches);
        assert_eq!(longest, vec![10, 20, 35]);
        assert_eq!(loop_engine.stats.total_accepted_tokens, 3);
    }

    #[test]
    fn test_settle_draft_round_records_stats() {
        let mut loop_engine = SpeculativeLoop::new(SpeculativeLoopConfig::default());
        let prev_draft = vec![1, 2, 3];
        let next_draft = vec![4, 5, 6, 7];

        let ready_tokens = loop_engine.settle_draft_round(&prev_draft, 10, 2, &next_draft);
        assert_eq!(ready_tokens, 4);
        assert_eq!(loop_engine.stats.total_accepted_tokens, 2);
        assert_eq!(loop_engine.stats.total_draft_tokens, 3);
    }
}
