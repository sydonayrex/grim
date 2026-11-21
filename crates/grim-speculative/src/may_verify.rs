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
