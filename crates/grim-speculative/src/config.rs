//! `SpeculationConfig` — production-tuning knobs for `SpeculativeCausalLm`.
//!
//! §5.3.2. Lives at the wrapper level so the same struct applies whether
//! the active strategy is DSpark (`DraftBackbone` + `MarkovHead` +
//! `ConfidenceHead`) or a model's own `NativeMtp`. MTP simply ignores
//! fields that don't apply (`min_verify_len` is honored, `block_len`
//! becomes `mtp_depth()` where the wrapper picks that depth from the
//! model).

/// Configuration for `SpeculativeCausalLm`.
#[derive(Debug, Clone)]
pub struct SpeculationConfig {
    /// Block length used by the DSpark drafter. Production default `5`
    /// ("DSpark-5"). Ignored when the active strategy is `NativeMtp` —
    /// the wrapper reads depth from the model instead.
    pub block_len: usize,
    /// Floor on how many drafted positions are batch-verified against
    /// the target on every iteration. `ConfidenceScheduler` is not
    /// allowed to drop below this regardless of how low the per-position
    /// confidence scores go. Also clamps the MTP path: MTP can't verify
    /// fewer than this even on a heavily loaded tick.
    pub min_verify_len: usize,
    /// Positions whose `ConfidenceHead::score` falls under this threshold
    /// are not offered to the target — they get dropped before the
    /// verification pass consumes batch capacity on them. This is the
    /// "drop the tail" half of confidence scheduling; combined with
    /// throughput-aware `choose_verify_len`, it trims verification cost
    /// without sacrificing too many accepted tokens.
    pub confidence_floor: f32,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        // Production default per §5.3.2.
        Self {
            block_len: 5,
            min_verify_len: 1,
            // 0 = "no floor" (let every position through to verify). The
            // documented knob is opt-in; default-on deployment should leave
            // this at zero and rely on throughput-aware trims instead.
            confidence_floor: 0.0,
        }
    }
}
