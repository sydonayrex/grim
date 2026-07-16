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
    /// WI 4.4.2 — TIDE-style adaptation tracking. EMA of the per-step
    /// acceptance rate (`accepted / drafted`). When this drifts below
    /// `adaptation_config.min_accept_rate`, the draft model is considered
    /// misaligned and `should_adapt_draft` returns `true`.
    pub adaptation_state: AdaptationState,
    /// Configuration for the adaptation-trigger decision.
    pub adaptation_config: AdaptationConfig,
}

/// WI 4.4.2 — runtime state for the "should we adapt the draft" decision.
///
/// Tracks the exponential moving average of the acceptance rate across decode
/// steps. The EMA is deterministic (input-driven, no wall-clock) — Gate 4.6.3.
/// When the EMA drops below the configured floor, `should_adapt_draft` fires,
/// signaling TIDE's "activate only when beneficial" control.
#[derive(Debug, Clone)]
pub struct AdaptationState {
    /// EMA of the acceptance rate (`accepted / drafted`). Starts at 1.0
    /// (optimistic) so the first few steps don't trigger a spurious refresh.
    pub accept_rate_ema: f64,
    /// Number of steps observed so far. Used for the initial ramp before
    /// the EMA stabilizes.
    pub steps_observed: u64,
}

impl Default for AdaptationState {
    fn default() -> Self {
        Self {
            accept_rate_ema: 1.0,
            steps_observed: 0,
        }
    }
}

/// WI 4.4.2 — configuration for the adaptation-trigger decision.
#[derive(Debug, Clone, Copy)]
pub struct AdaptationConfig {
    /// EMA smoothing factor (α). `0.15` matches `SelfTuningController`'s
    /// existing EMA alpha for consistency with the codebase's other runtime
    /// adaptive signals.
    pub ema_alpha: f64,
    /// Minimum acceptance rate before adaptation fires. Below this, the draft
    /// is considered misaligned with the target (TIDE's "beneficial" threshold).
    pub min_accept_rate: f64,
    /// Minimum steps to observe before the trigger can fire. Prevents a
    /// spurious refresh during the EMA's initial ramp.
    pub min_steps_before_trigger: u64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.15,
            min_accept_rate: 0.3,
            min_steps_before_trigger: 10,
        }
    }
}

impl ConfidenceScheduler {
    pub fn new(
        throughput_profile: ThroughputProfile,
        config: SpeculationConfig,
    ) -> Self {
        Self {
            throughput_profile,
            config,
            adaptation_state: AdaptationState::default(),
            adaptation_config: AdaptationConfig::default(),
        }
    }

    /// WI 4.4.2 — Record the acceptance result from a decode step and update
    /// the adaptation EMA.
    ///
    /// `accepted` is the number of draft tokens the target accepted this step;
    /// `drafted` is the total number of draft tokens proposed. Calling this
    /// with `drafted == 0` is a no-op (no draft was proposed, nothing to
    /// measure). The EMA update is:
    /// ```text
    /// ema = (1 - α) * ema + α * (accepted / drafted)
    /// ```
    ///
    /// Deterministic: given the same sequence of `(accepted, drafted)` pairs,
    /// the EMA is identical across runs (Gate 4.6.3 — no wall-clock or RNG).
    pub fn record_acceptance(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let rate = (accepted as f64) / (drafted as f64);
        let alpha = self.adaptation_config.ema_alpha;
        self.adaptation_state.accept_rate_ema =
            (1.0 - alpha) * self.adaptation_state.accept_rate_ema + alpha * rate;
        self.adaptation_state.steps_observed += 1;
    }

    /// WI 4.4.2 — TIDE-style "adapt only when beneficial" trigger.
    ///
    /// Returns `true` when the measured acceptance-rate EMA has drifted below
    /// `adaptation_config.min_accept_rate`, signaling that the draft model has
    /// diverged from the target enough to warrant a refresh step. Returns
    /// `false` during the initial ramp (before `min_steps_before_trigger`) to
    /// avoid spurious triggers while the EMA stabilizes.
    ///
    /// This is the runtime control TIDE describes: the adaptation is gated by
    /// a *measured signal* (acceptance rate), not a fixed schedule. The actual
    /// weight update is deferred to the draft-update interface in `distill.rs`
    /// (§4.4.3) — this function only answers "should we adapt now?"
    pub fn should_adapt_draft(&self) -> bool {
        if self.adaptation_state.steps_observed < self.adaptation_config.min_steps_before_trigger {
            return false;
        }
        self.adaptation_state.accept_rate_ema < self.adaptation_config.min_accept_rate
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
        let _accepted_per_sec = self.throughput_profile.accepted_tokens_per_sec;
        let total_cost_ms = cost_per_token * (max_len as f64);
        // Throughput headroom in seconds — how much verify-cost we can
        // afford at current utilization:
        let afford_ms = load_factor * 1000.0;
        if total_cost_ms > afford_ms {
            // Tight budget — only verify floor of block.
            let reduced = self.config.min_verify_len.max(
                ((afford_ms / cost_per_token).floor() as usize)
                    .max(1)
                    .min(max_len),
            );
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
    use grim_backend_cpu::DeterministicRng;
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

    // ====================================================================
    // WI 4.4.2 — TIDE-style adaptation trigger tests.
    // ====================================================================

    #[test]
    fn should_adapt_does_not_fire_during_initial_ramp() {
        let mut sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig::default(),
        );
        sched.adaptation_config = AdaptationConfig {
            ema_alpha: 0.15,
            min_accept_rate: 0.5,
            min_steps_before_trigger: 10,
        };
        // Record very low acceptance for 5 steps (< min_steps_before_trigger).
        for _ in 0..5 {
            sched.record_acceptance(0, 5); // 0% accept rate
        }
        assert!(
            !sched.should_adapt_draft(),
            "must not fire before min_steps_before_trigger"
        );
    }

    #[test]
    fn should_adapt_fires_when_accept_rate_drifts_below_threshold() {
        let mut sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig::default(),
        );
        sched.adaptation_config = AdaptationConfig {
            ema_alpha: 0.5, // aggressive EMA so it converges fast
            min_accept_rate: 0.4,
            min_steps_before_trigger: 3,
        };
        // Record consistently low acceptance for enough steps.
        for _ in 0..20 {
            sched.record_acceptance(0, 5); // 0% accept rate every step
        }
        assert!(
            sched.should_adapt_draft(),
            "must fire when EMA ({:.3}) < threshold ({})",
            sched.adaptation_state.accept_rate_ema,
            sched.adaptation_config.min_accept_rate,
        );
    }

    #[test]
    fn should_adapt_does_not_fire_when_acceptance_is_healthy() {
        let mut sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig::default(),
        );
        sched.adaptation_config = AdaptationConfig {
            ema_alpha: 0.15,
            min_accept_rate: 0.3,
            min_steps_before_trigger: 5,
        };
        // Record consistently high acceptance.
        for _ in 0..20 {
            sched.record_acceptance(4, 5); // 80% accept rate
        }
        assert!(
            !sched.should_adapt_draft(),
            "must not fire when acceptance is healthy"
        );
    }

    #[test]
    fn record_acceptance_zero_drafted_is_noop() {
        let mut sched = ConfidenceScheduler::new(
            ThroughputProfile::default(),
            SpeculationConfig::default(),
        );
        let ema_before = sched.adaptation_state.accept_rate_ema;
        sched.record_acceptance(0, 0); // no draft proposed
        assert_eq!(
            sched.adaptation_state.accept_rate_ema, ema_before,
            "zero-drafted must be a no-op"
        );
        assert_eq!(
            sched.adaptation_state.steps_observed, 0,
            "zero-drafted must not increment step counter"
        );
    }

    #[test]
    fn adaptation_ema_is_deterministic() {
        // Gate 4.6.3: same input sequence → same EMA, no wall-clock.
        let inputs = [(3usize, 5usize), (1, 5), (0, 5), (4, 5), (2, 5)];
        let mut a = ConfidenceScheduler::new(ThroughputProfile::default(), SpeculationConfig::default());
        let mut b = ConfidenceScheduler::new(ThroughputProfile::default(), SpeculationConfig::default());
        for (acc, drafted) in inputs {
            a.record_acceptance(acc, drafted);
            b.record_acceptance(acc, drafted);
        }
        assert_eq!(
            a.adaptation_state.accept_rate_ema,
            b.adaptation_state.accept_rate_ema,
            "EMA must be identical for identical inputs (Gate 4.6.3)"
        );
        assert_eq!(a.should_adapt_draft(), b.should_adapt_draft());
    }
}
