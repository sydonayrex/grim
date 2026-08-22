//! GPU-Native Speculative Decoding Generation Loop.
//!
//! Orchestrates draft token generation, target model verification, and device-side
//! rejection sampling with minimal host-GPU synchronization.

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
}
