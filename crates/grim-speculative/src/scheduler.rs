//! `ConfidenceScheduler` — load-adaptive verify-length trimmer.
//!
//! §5.3.2. Verifying every drafted position indiscriminately wastes
//! target-model batch capacity on tail tokens that were unlikely to be
//! accepted anyway, and that waste gets *worse* exactly when the engine
//! is already under load. This is the second half of DSpark's
//! contribution: the model-side draft gives us confidence scores per
//! drafted position, and the serving system gets to consume those
//! scores to decide how deep to verify this iteration.
//!
//! Knob: under light load, walk the confidence-sorted prefix all the
//! way to the block tail and maximize accepted length; under heavy
//! load, stop after the high-confidence prefix and drop the
//! low-confidence tail *before* the target's verification pass ever
//! sees them.

use crate::confidence_head::ConfidenceHead;
use crate::config::SpeculationConfig;
use crate::draft_backbone::DraftBlock;

/// Profiling input — what does one batched-verification position cost on
/// this hardware, and how much verification throughput can still be
/// spent under the current system load. Built once per backend by
/// `grim-engine` from `ThroughputProfile::calibrate(...)`, then held by
/// the scheduler.
#[derive(Debug, Clone)]
pub struct ThroughputProfile {
    /// Cost in "verify-unit seconds" of one drafted position on the
    /// target model on this hardware. Used to translate the
    /// `live_gpu_utilization + batch_pressure` budget into "how many
    /// positions can we afford to verify this tick."
    pub per_position_cost_seconds: f32,
    /// Headroom fraction in `(0.0, 1.0]` measured from the current
    /// scheduler tick. `1.0` means "plenty of room — extend verify
    /// length toward the block tail"; `~0.5` means "trim hard — keep
    /// only the high-confidence prefix"; `<<0.5` should drop the
    /// speculation pass for this tick entirely (caller's decision —
    /// not the scheduler's).
    pub headroom: f32,
}

impl ThroughputProfile {
    /// Construction-time default; the real profile is built from
    /// measured timings via `calibrate`. Kept here so unit tests and
    /// early call-sites can construct one without a profiling pass.
    pub fn default_for(per_position_cost_seconds: f32, headroom: f32) -> Self {
        Self {
            per_position_cost_seconds,
            headroom: headroom.clamp(0.0, 1.0),
        }
    }
}

/// The load-adaptive verify-length decision maker.
#[derive(Debug, Clone)]
pub struct ConfidenceScheduler {
    /// Profiling input — see struct doc.
    pub throughput_profile: ThroughputProfile,
    /// Static config knobs. `min_verify_len` is the absolute floor.
    pub config: SpeculationConfig,
}

impl ConfidenceScheduler {
    pub fn new(throughput_profile: ThroughputProfile, config: SpeculationConfig) -> Self {
        Self {
            throughput_profile,
            config,
        }
    }

    /// Called once per engine iteration, per drafted sequence, *before*
    /// the batched target-verification pass.
    ///
    /// Inputs:
    /// - `draft`: the drafted block (tokens, base logits, per-position
    ///   confidence scores).
    /// - `confidence_head`: the trained acceptance-probability scorer.
    ///   Called here so the scheduler sees the same per-position scores
    ///   the verification pass would have consumed; we keep the scores
    ///   alongside the block rather than re-computing them at verify
    ///   time, since `DraftBlock.confidence` was written by the same
    ///   head during drafting.
    ///
    /// Returns the number of leading positions to verify — always
    /// `>= min_verify_len`, capped at the block length.
    pub fn choose_verify_len(
        &self,
        draft: &DraftBlock,
        live_gpu_utilization: f32,
        batch_pressure: usize,
        confidence_head: &dyn ConfidenceHead,
    ) -> usize {
        let block_len = draft.len();
        if block_len == 0 {
            return 0;
        }

        // Defensive floor: always verify at least `min_verify_len`.
        let floor = self.config.min_verify_len.max(1).min(block_len);

        // Score the block; per [`ConfidenceHead::score`] contract these
        // are already calibrated acceptance probabilities in `[0.0, 1.0]`.
        let scores = confidence_head.score(draft);
        debug_assert_eq!(
            scores.len(),
            block_len,
            "ConfidenceHead::score must return one score per drafted position",
        );

        // Step 1: per-position confidence-floor gate — drop low-score
        // positions before they consume batch capacity. This is the
        // "drop the tail" half of confidence scheduling; the throughput
        // step below extends or trims from there.
        let post_floor: usize = scores
            .iter()
            .take_while(|&&p| p >= self.config.confidence_floor)
            .count()
            .max(floor);

        // Step 2: throughput budget. Convert headroom + batch_pressure
        // into "how many positions of verify-cost we can stand" this
        // tick. With zero headroom, run only the floor; with full
        // headroom and an idle GPU, go all the way to the post-floor
        // length (i.e. verify every position not below the score
        // floor).
        //
        // Cost model: each additional position costs
        // `per_position_cost_seconds` of GPU time. We have
        // `headroom * (1 - live_gpu_utilization)` of slack to spend.
        // `batch_pressure` divides into that budget to keep the
        // scheduler fair across the active batch (more competing
        // sequences ⇒ shorter verify per-sequence).
        let cost = self.throughput_profile.per_position_cost_seconds.max(1e-9);
        let slack_seconds =
            self.throughput_profile.headroom * (1.0 - live_gpu_utilization.clamp(0.0, 1.0));
        let per_seq_budget_seconds = if batch_pressure == 0 {
            slack_seconds
        } else {
            slack_seconds / batch_pressure as f32
        };
        let affordable = (per_seq_budget_seconds / cost).floor().max(1.0) as usize;

        // Combine: throughput budget capped at post-floor,
        // floored at `min_verify_len`.
        affordable.min(post_floor).max(floor).min(block_len)
    }
}
