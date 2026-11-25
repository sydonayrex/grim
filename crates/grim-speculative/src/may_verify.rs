//! Basalt-style adaptive `may_verify` speculation gate with mandatory probe
//! exception (`kProbeChance = 0.85`, Q9) so EMA never permanently disables
//! speculation after a low-acceptance phase.

/// Returns `true` when `GRIM_SPEC_MAY_VERIFY` is truthy (`1` or `true`).
#[inline]
pub fn spec_may_verify_enabled() -> bool {
    std::env::var("GRIM_SPEC_MAY_VERIFY")
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Adaptive speculative verification gate with per-slot acceptance EMA and
/// Basalt's probe exception (`probe_chance = 0.85` -> 15% probe probability
/// when `chance[0] < min_chance`).
#[derive(Debug, Clone)]
pub struct MayVerifyGate {
    chance: Vec<f32>,
    min_chance: f32,
    ema_alpha: f32,
    probe_chance: f32,
    rng_state: u64,
}

impl MayVerifyGate {
    pub fn new(
        depth: usize,
        min_chance: f32,
        ema_alpha: f32,
        probe_chance: f32,
        seed: u64,
    ) -> Self {
        Self {
            chance: vec![1.0; depth.max(1)],
            min_chance,
            ema_alpha: ema_alpha.clamp(0.01, 1.0),
            probe_chance: probe_chance.clamp(0.0, 1.0),
            rng_state: seed.max(1),
        }
    }

    /// Adapt the per-slot EMA to the model's actual MTP depth. The gate is
    /// constructed before a model is available, so the wrapper calls this on
    /// its first decode step. Preserves accumulated chances when the depth is
    /// unchanged.
    pub fn ensure_depth(&mut self, depth: usize) {
        let n = depth.max(1);
        if self.chance.len() != n {
            self.chance.resize(n, 1.0);
        }
    }

    /// Default Basalt parameters: `min_chance = 0.35`, `ema_alpha = 0.2`,
    /// `probe_chance = 0.85`.
    pub fn with_defaults(depth: usize) -> Self {
        Self::new(depth, 0.35, 0.2, 0.85, 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    fn next_unit_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Returns `true` when verification should run: either `chance[0] >= min_chance`
    /// OR the probe exception fires (`u >= probe_chance`).
    pub fn should_verify(&mut self) -> bool {
        if self.chance[0] >= self.min_chance {
            return true;
        }
        self.next_unit_f32() >= self.probe_chance
    }

    /// Updates per-slot acceptance EMA after a verification step.
    pub fn observe(&mut self, accepted: usize, drafted: usize) {
        let n = drafted.min(self.chance.len());
        let a = self.ema_alpha;
        for slot in 0..n {
            let hit = if slot < accepted { 1.0f32 } else { 0.0f32 };
            self.chance[slot] = (1.0 - a) * self.chance[slot] + a * hit;
        }
    }

    pub fn chance(&self, slot: usize) -> f32 {
        self.chance.get(slot).copied().unwrap_or(0.0)
    }
}

/// Request-local acceptance-triggered fallback (DeepSeek-V4 DSpark transfer).
/// When a window of `min_proposals` shows acceptance below `threshold`, the
/// next `cooldown_steps` steps run target-only; then a single probe step
/// re-enters speculation. The controller is seeded deterministically and never
/// reads the clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FallbackConfig {
    pub min_proposals: usize,
    pub accept_threshold: f32,
    pub cooldown_steps: usize,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            min_proposals: 32,
            accept_threshold: 0.60,
            cooldown_steps: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecFallback {
    config: FallbackConfig,
    accepted: usize,
    proposed: usize,
    cooldown: usize,
    speculating: bool,
    /// True once on the step that tripped the fallback; consumed for telemetry.
    tripped: bool,
}

impl SpecFallback {
    pub fn new(config: FallbackConfig) -> Self {
        Self {
            config,
            accepted: 0,
            proposed: 0,
            cooldown: 0,
            speculating: true,
            tripped: false,
        }
    }

    /// `GRIM_SPEC_FALLBACK=1` enables the controller (default OFF).
    pub fn from_env() -> Option<Self> {
        if !spec_may_verify_enabled() {
            return None;
        }
        if std::env::var("GRIM_SPEC_FALLBACK").ok().as_deref() != Some("1") {
            return None;
        }
        Some(Self::new(FallbackConfig::default()))
    }

    pub fn speculating(&self) -> bool {
        self.speculating
    }

    pub fn cooldown_remaining(&self) -> usize {
        self.cooldown
    }

    /// Consume the one-shot trip flag (for telemetry).
    pub fn take_tripped(&mut self) -> bool {
        std::mem::take(&mut self.tripped)
    }

    /// Reset on a new request id.
    pub fn reset(&mut self) {
        *self = Self::new(self.config);
    }

    pub fn observe(&mut self, accepted: usize, proposed: usize) {
        if proposed == 0 {
            return;
        }
        if !self.speculating {
            if self.cooldown > 0 {
                self.cooldown -= 1;
            }
            if self.cooldown == 0 {
                // Re-entry probe: start a fresh measurement window.
                self.speculating = true;
                self.tripped = false;
                self.accepted = 0;
                self.proposed = 0;
            }
            return;
        }
        self.accepted += accepted.min(proposed);
        self.proposed += proposed;
        if self.proposed >= self.config.min_proposals {
            let rate = self.accepted as f32 / self.proposed as f32;
            if rate < self.config.accept_threshold {
                self.speculating = false;
                self.tripped = true;
                self.cooldown = self.config.cooldown_steps;
                self.accepted = 0;
                self.proposed = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_acceptance_trips_target_only_then_probes() {
        let mut f = SpecFallback::new(FallbackConfig {
            min_proposals: 4,
            accept_threshold: 0.5,
            cooldown_steps: 3,
        });
        assert!(f.speculating());
        for _ in 0..4 {
            f.observe(0, 1);
        }
        assert!(!f.speculating());
        assert_eq!(f.cooldown_remaining(), 3);
        f.observe(1, 1);
        assert_eq!(f.cooldown_remaining(), 2);
        f.observe(1, 1);
        f.observe(1, 1);
        assert!(f.speculating(), "must probe again after the cooldown");
    }

    #[test]
    fn healthy_acceptance_never_trips() {
        let mut f = SpecFallback::new(FallbackConfig::default());
        for _ in 0..100 {
            f.observe(3, 4);
        }
        assert!(f.speculating());
    }
}
