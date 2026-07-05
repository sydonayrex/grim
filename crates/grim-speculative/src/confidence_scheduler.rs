//! `ConfidenceScheduler` — chooses how many drafted positions to verify
//! against the target each iteration, given current load.
//!
//! §5.3.2. The second half of DSpark's contribution, and a serving-system
//! concern rather than purely a modeling one: verifying every drafted
//! position indiscriminately wastes target-model batch capacity on tail
//! tokens unlikely to be accepted anyway, and that waste gets worse
//! exactly when the engine is already under load.
//!
//! Mostly the function is deterministic. §5.8 requires "per-request-
//! seeded speculative-decoding RNG" for any sampling step inside the
//! speculative path; we keep the verifier length deterministic
//! (input-driven) but use a [`DeterministicRng`] for the optional
//! random reversal of the verification order, which is exposed via
//! `DeterminismMode::Strict`-aware sampling. The rng helper lives
//! in `grim-backend-cpu`.

use crate::draft_backbone::DraftBlock;
use grim_backend_cpu::DeterministicRng;

/// Profile of how many accepted tokens per second the verifier produces.
/// Engine-internal; measured at runtime from real per-position accept
/// statistics (today: stubbed defaults).
#[derive(Debug, Clone)]
pub struct ThroughputProfile {
    /// Approximate verification cost per drafted position (ms).
    pub verify_ms_per_token: f64,
    /// Approximate steady-state accepted tokens per second.
    pub accepted_tokens_per_sec: f64,
}

impl Default for ThroughputProfile {
    fn default() -> Self {
        Self {
            verify_ms_per_token: 0.1,
            accepted_tokens_per_sec: 50.0,
        }
    }
}

/// Configurable knobs for `ConfidenceScheduler`.
#[derive(Debug, Clone, Copy)]
pub struct SpeculationConfig {
    pub block_len: usize,
    pub min_verify_len: usize,
    pub confidence_floor: f32,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            block_len: 5,
            min_verify_len: 1,
            confidence_floor: 0.05,
        }
    }
}

/// The scheduler: per drafted-sequence, decides verify length each engine
/// tick, before the batched target-verification pass.
pub struct ConfidenceScheduler {
    pub throughput_profile: ThroughputProfile,
    pub config: SpeculationConfig,
}

impl ConfidenceScheduler {
    pub fn new(
        throughput_profile: ThroughputProfile,
        config: SpeculationConfig,
    ) -> Self {
        Self {
            throughput_profile,
            config,
        }
    }

    /// Choose how many drafted positions to actually verify against the
    /// target. The decision walks the confidence-ranked prefix; the
    /// scheduler extends verification while marginal survival probability
    /// still clears the throughput headroom implied by
    /// `throughput_profile` at current GPU utilization. Never drops
    /// below `min_verify_len`.
    ///
    /// `live_gpu_utilization`            ∈ [0, 1]
    /// `batch_pressure`                  ongoing iteration backlog in tokens
    pub fn choose_verify_len(
        &self,
        draft: &DraftBlock,
        live_gpu_utilization: f32,
        batch_pressure: usize,
    ) -> usize {
        if draft.is_empty() {
            return 0;
        }
        // 1. Determine hierarchy of available slots via descending confidence.
        // (v1 confidence is per-position; in real DSpark this is pre-sorted
        // by confidence head. We treat `draft.confidence` as already aligned.)
        let max_len = self.config.block_len.min(draft.len());

        // 2. Reduce verify length under load: target
        //     headroom = 1.0 - live_gpu_utilization
        //     squeeze = batch_pressure based factor
        let headroom = (1.0 - live_gpu_utilization).max(0.0);
        let squeeze = 1.0 / (1.0 + (batch_pressure as f64) / 16.0);
        let load_factor = (headroom as f64) * squeeze;

        // 3. Iteratively include positions whose marginal probability sum
        //    is below the calibrated throughput cost.
        let cost_per_token = self.throughput_profile.verify_ms_per_token;
        let accepted_per_sec = self.throughput_profile.accepted_tokens_per_sec;
        let total_cost_ms = cost_per_token * (max_len as f64);
        // Throughput headroom in seconds — how much verify-cost we can
        // afford at current utilization:
        let afford_ms = load_factor * 1000.0;
        if total_cost_ms > afford_ms {
            // Tight budget — only verify floor of block.
            let reduced = (self.config.min_verify_len.max(
                ((afford_ms / cost_per_token).floor() as usize)
                    .max(1)
                    .min(max_len),
            ));
            return reduced;
        }

        // 4. Walk confidence-ranked prefix, keep extending while
        //    marginal survival probability × block-cost stays under
        //    the throughput headroom. Below `confidence_floor` we drop.
        let mut len = self.config.min_verify_len.min(max_len);
        let mut cumulative_survival = 1.0f64;
        let mut cumulative_lifetime_ms = 0.0f64;
        
        for i in 0..max_len {
            let conf = draft.confidence.get(i).copied().unwrap_or(0.5) as f64;
            if conf < self.config.confidence_floor as f64 {
                break;
            }
            
            // Marginal survival: cumulative likelihood that this token is reached
            cumulative_survival *= conf;
            
            // Expected validation utility based on accepted throughput
            let marginal_cost = cost_per_token;
            cumulative_lifetime_ms += marginal_cost;
            
            // If the cumulative cost weighted by the likelihood of reaching this token
            // exceeds our load-adjusted budget (afford_ms), we truncate the verification block here.
            if cumulative_lifetime_ms * cumulative_survival > afford_ms {
                break;
            }
            
            len = i + 1;
        }
        
        len.max(self.config.min_verify_len).min(max_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft_backbone::DraftBlock;
    use grim_core::error::Result;
    use grim_tensor::{Shape, Tensor};

    fn make_draft(conf: Vec<f32>, n: usize) -> DraftBlock {
        let base_logits = Tensor::new(
            std::sync::Arc::new(grim_backend_cpu::CpuStorage::new(
                vec![0.0f32; n * 16],
                Shape::new(vec![n, 16]),
                grim_tensor::DType::F32,
            )) as std::sync::Arc<dyn grim_tensor::BackendStorage>,
            Shape::new(vec![n, 16]),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            grim_tensor::Device::Cpu,
        );
        DraftBlock {
            tokens: vec![0u32; n],
            base_logits,
            confidence: conf,
        }
    }

    #[test]
    fn verify_len_minimum_floor_holds() {
        let sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig {
                block_len: 5,
                min_verify_len: 2,
                confidence_floor: 0.0,
            },
        );
        let draft = make_draft(vec![1.0; 5], 5);
        // Even under heavy load, never drop below min_verify_len.
        let len = sched.choose_verify_len(&draft, 0.99, 1024);
        assert!(len >= 2, "expected at least min_verify_len={}, got {:?}", 2, sched.config.min_verify_len);
        assert_eq!(len, 2); // Afford_ms ≈ 0 due to high utilization
    }

    #[test]
    fn verify_len_extends_under_low_load() {
        let sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig {
                block_len: 5,
                min_verify_len: 1,
                confidence_floor: 0.0,
            },
        );
        let draft = make_draft(vec![1.0; 5], 5);
        let len = sched.choose_verify_len(&draft, 0.0, 0);
        // Low load → verify full block.
        assert_eq!(len, 5);
    }

    #[test]
    fn verify_len_truncates_at_low_confidence() {
        let sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig {
                block_len: 5,
                min_verify_len: 1,
                confidence_floor: 0.5,
            },
        );
        // First 2 highly confident, last 3 below floor.
        let conf = vec![0.9, 0.9, 0.1, 0.1, 0.1];
        let draft = make_draft(conf, 5);
        let len = sched.choose_verify_len(&draft, 0.0, 0);
        // Truncate after first 2 due to confidence floor.
        assert!(len <= 3, "expected truncation, got {len}");
        assert!(len >= 2, "expected at least 2 verified, got {len}");
    }

    #[test]
    fn deterministic_seeded_rng_is_reproducible_under_strict() {
        // Architecture §5.8: a per-request seeded RNG, used by the
        // speculative path, must produce identical noise under
        // strict mode given identical seeds.
        let mut a = DeterministicRng::from_seed(0xDEAD_BEEF);
        let mut b = DeterministicRng::from_seed(0xDEAD_BEEF);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = DeterministicRng::from_seed(0xCAFE);
        let mut d = DeterministicRng::from_seed(0xF00D);
        let mut differ = false;
        for _ in 0..64 {
            if c.next_u64() != d.next_u64() {
                differ = true;
                break;
            }
        }
        assert!(differ, "distinct seeds must produce distinct streams");
    }
}
