//! Cost-aware speculation width (D-cut / CAST / DAS transfers).
//!
//! Picks how many already-drafted tokens to verify in one target pass by
//! maximising expected committed tokens per millisecond. Pure Rust logic:
//! no backend dependency, no wall-clock reads, deterministic for a given
//! profile and observation sequence. Enabled with `GRIM_SPEC_WIDTH=1`.

/// Measured cost model for one verification round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WidthConfig {
    /// Fixed target-forward cost (ms) for a verify pass.
    pub verify_base_ms: f64,
    /// Marginal target-forward cost (ms) per packed token.
    pub verify_slope_ms: f64,
    /// Draft cost (ms) already paid before verification.
    pub draft_cost_ms: f64,
    /// Fallback per-position survival probability when no confidence head exists.
    pub default_confidence: f32,
    /// Hard bounds on the number of drafted tokens verified.
    pub min_k: usize,
    pub max_k: usize,
}

impl Default for WidthConfig {
    fn default() -> Self {
        Self {
            verify_base_ms: 0.5,
            verify_slope_ms: 0.1,
            draft_cost_ms: 0.3,
            default_confidence: 0.75,
            min_k: 1,
            max_k: 8,
        }
    }
}

/// Online width selector with a context-length budget class.
#[derive(Debug, Clone)]
pub struct WidthController {
    config: WidthConfig,
    accept_ema: f64,
    steps: u64,
}

impl WidthController {
    pub fn new(config: WidthConfig) -> Self {
        Self {
            config,
            accept_ema: 0.75,
            steps: 0,
        }
    }

    /// `GRIM_SPEC_WIDTH=1` enables the controller. `GRIM_SPEC_WIDTH_PROFILE`
    /// optionally supplies `verify_base_ms,verify_slope_ms,draft_cost_ms`.
    pub fn from_env() -> Option<Self> {
        if std::env::var("GRIM_SPEC_WIDTH").ok().as_deref() != Some("1") {
            return None;
        }
        let mut config = WidthConfig::default();
        if let Ok(profile) = std::env::var("GRIM_SPEC_WIDTH_PROFILE") {
            let parts: Vec<f64> = profile
                .split(',')
                .filter_map(|p| p.trim().parse::<f64>().ok())
                .collect();
            if parts.len() == 3 && parts.iter().all(|v| v.is_finite() && *v >= 0.0) {
                config.verify_base_ms = parts[0];
                config.verify_slope_ms = parts[1];
                config.draft_cost_ms = parts[2];
            }
        }
        Some(Self::new(config))
    }

    pub fn config(&self) -> &WidthConfig {
        &self.config
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn accept_rate(&self) -> f64 {
        self.accept_ema
    }

    pub fn observe(&mut self, accepted: usize, proposed: usize) {
        if proposed == 0 {
            return;
        }
        let rate = (accepted.min(proposed) as f64) / proposed as f64;
        if self.steps == 0 {
            self.accept_ema = rate;
        } else {
            self.accept_ema = 0.8 * self.accept_ema + 0.2 * rate;
        }
        self.steps += 1;
    }

    /// Choose a verify width `k` for this round.
    ///
    /// `confidence` holds per-position survival estimates; `proposed` is the
    /// number of drafted tokens available. Short contexts skip speculation
    /// (the extra draft tax rarely buys a saved target pass); long contexts get
    /// a wider budget, capped by the measured max.
    pub fn choose(&self, confidence: &[f32], proposed: usize, context_len: usize) -> usize {
        let cap = if context_len <= 64 {
            1
        } else if context_len >= 2048 {
            self.config.max_k + 1
        } else {
            self.config.max_k
        }
        .min(proposed);
        let floor = self.config.min_k.min(cap);
        if cap <= floor {
            return cap.max(1).min(proposed);
        }

        let mut best = floor;
        let mut best_utility = f64::NEG_INFINITY;
        let mut survival = 1.0f64;
        let mut expected = 1.0f64; // the guaranteed next-token (bonus) emission
        for k in 1..=cap {
            let c = confidence
                .get(k - 1)
                .copied()
                .unwrap_or(self.config.default_confidence)
                .clamp(0.01, 1.0) as f64;
            survival *= c;
            expected += survival;
            let cost = self.config.verify_base_ms
                + self.config.verify_slope_ms * k as f64
                + self.config.draft_cost_ms;
            let utility = expected / cost.max(1e-6);
            if utility > best_utility {
                best_utility = utility;
                best = k;
            }
        }
        best.max(floor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_respects_proposed_and_bounds() {
        let c = WidthController::new(WidthConfig {
            max_k: 8,
            ..Default::default()
        });
        assert_eq!(c.choose(&[1.0; 8], 3, 128), 3);
        assert_eq!(c.choose(&[1.0; 8], 3, 16), 1); // short context: no speculation
        assert!(c.choose(&[1.0; 16], 16, 4096) <= 9); // long context: max_k + 1
        assert_eq!(c.choose(&[], 0, 128), 0);
    }

    #[test]
    fn width_truncates_on_low_confidence() {
        let c = WidthController::new(WidthConfig::default());
        let conf = [0.95, 0.95, 0.05, 0.05, 0.05];
        let k = c.choose(&conf, 5, 128);
        assert!(k <= 2, "deep low-confidence positions must not be verified, got {k}");
        assert!(k >= 1);
    }

    #[test]
    fn width_is_deterministic() {
        let a = WidthController::new(WidthConfig::default());
        let b = WidthController::new(WidthConfig::default());
        assert_eq!(a.choose(&[0.9, 0.8, 0.7], 3, 512), b.choose(&[0.9, 0.8, 0.7], 3, 512));
    }

    #[test]
    fn observe_tracks_ema_without_wall_clock() {
        let mut c = WidthController::new(WidthConfig::default());
        c.observe(2, 3);
        c.observe(1, 3);
        assert_eq!(c.steps(), 2);
        assert!(c.accept_rate() > 0.0 && c.accept_rate() < 1.0);
    }
}
