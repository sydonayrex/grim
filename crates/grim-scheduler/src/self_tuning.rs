//! Self-tuning scheduler controller — §5.7.
//!
//! Each knob self-calibrates *independently* (the architecture explicitly
//! defers coupled multi-knob tuning past per-knob calibration because knobs
//! correcting against each other's side effects is a real oscillation
//! risk). When independent per-knob calibration is already good enough,
//! the coupled controller may not need to exist at all (§11) — that's a
//! legitimate outcome.
//!
//! Per the architecture, the four knobs in scope are:
//!
//! 1. **chunked-prefill size** — how many tokens from any one request's
//!    prompt are drained per tick. Drives prefill-vs-decode TTFT balance.
//! 2. **`max_batched_tokens`** — across-the-board cap; tight under load,
//!    loose when the system has margin.
//! 3. **`speculative_block_len`** — DSpark's parallel draft width per
//!    step. Tuned to keep confidence under sustain-load while staying
//!    shallow enough that offload-to-CPU drafts don't go stale.
//! 4. **`kv_compression_bit_width`** — TurboQuant-style KV value bits
//!    on free blocks. Lower bits save memory at predictable quality cost.
//!
//! Each knob:
//! - holds an EMA of the metric it tunes against;
//! - exposes `record_*` and `tune_*`;
//! - ships with floors/ceilings so the calibration doesn't drift.
//!
//! `tune_all` drives each knob independently on the most recent
//! observations; the architecture's coupling concerns surface only in
//! future "coupled" revisions of the controller.

// (no imports needed)

/// Each knob's EMA + bounds. Owns one continuous observation per knob.
#[derive(Debug, Clone)]
pub struct KnobTuner {
    pub knob: KnobKind,
    /// Running mean of the metric we're tuning against.
    pub ema_observed: f64,
    /// Target that determines "we have margin" vs "we're over budget".
    pub target: f64,
    /// Steps up/down ratio used to translate the gap into a quantised knob delta.
    pub scale_step: f64,
    /// Floor — knob never tunes below this.
    pub floor: f64,
    /// Ceiling — knob never tunes above this.
    pub ceiling: f64,
    /// Last tuned value of the knob — EMA-aligned.
    pub current: f64,
}

/// Trait interface for a self-tuning knob (§5.7)
pub trait TunableKnob: Send + Sync {
    fn record_metric(&mut self, observed: f64, alpha: f64);
    fn tune_step(&mut self) -> f64;
    fn get_current(&self) -> f64;
}

impl TunableKnob for KnobTuner {
    fn record_metric(&mut self, observed: f64, alpha: f64) {
        self.record(observed, alpha);
    }

    fn tune_step(&mut self) -> f64 {
        self.tune()
    }

    fn get_current(&self) -> f64 {
        self.current
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnobKind {
    ChunkedPrefillSize,
    MaxBatchedTokens,
    SpeculativeBlockLen,
    KvCompressionBitWidth,
}

impl KnobTuner {
    pub fn new_fixed(
        knob: KnobKind,
        target: f64,
        floor: f64,
        ceiling: f64,
        initial: f64,
        scale_step: f64,
    ) -> Self {
        Self {
            knob,
            ema_observed: target,
            target,
            scale_step,
            floor,
            ceiling,
            current: initial,
        }
    }

    pub fn record(&mut self, observed: f64, alpha: f64) {
        let a = alpha.clamp(0.0, 1.0);
        self.ema_observed = (1.0 - a) * self.ema_observed + a * observed;
    }

    /// One-step tuning: returns the new value the knob should take,
    /// clamped to `[floor, ceiling]`. Independent per-knob — caller is
    /// responsible for any cross-knob coordination needed (none required
    /// in v1). Internally iterates to convergence so a single call
    /// drives the knob as far toward the floor/ceiling as the
    /// observed-metric supports.
    fn tune(&mut self) -> f64 {
        let sign = (self.ema_observed - self.target).signum();
        let target_bound = if sign > 0.0 { self.floor } else if sign < 0.0 { self.ceiling } else { self.current };
        let mut step = self.scale_step;
        loop {
            let next = self.current + (target_bound - self.current) * step;
            let next = next.clamp(self.floor, self.ceiling);
            // Convergence: when we hit the floor or ceiling, or the
            // absolute residual is below a single step we configured,
            // stop.
            if (next - self.current).abs() < 1e-9 {
                self.current = next;
                return self.current;
            }
            if next == self.floor || next == self.ceiling {
                self.current = next;
                return self.current;
            }
            self.current = next;
            if step >= 1.0 {
                self.current = target_bound.clamp(self.floor, self.ceiling);
                return self.current;
            }
            step = (step * 2.0).min(1.0);
            if (self.current - target_bound).abs() < 1e-9 {
                self.current = target_bound.clamp(self.floor, self.ceiling);
                return self.current;
            }
        }
    }
}

/// Container for the four independent per-knob tuners and the EMA
/// decay rate applied to each. Owns the tuners' latest calibrated values
/// and exposes a single `tune_all` for callers that want a coordinated
/// autotune pass without coupling any of the knobs.
#[derive(Debug)]
pub struct SelfTuningController {
    pub target_ttft_ms: f64,
    pub target_itl_ms: f64,
    pub target_kv_drift_quality: f64,
    pub ema_ttft_ms: f64,
    pub ema_itl_ms: f64,
    /// EMA of how much compressed-KV drift (0..1) we've seen at the
    /// current bit-width; used by the kv-bit-width knob.
    pub ema_quality: f64,
    alpha: f64,
    pub chunked_prefill_size: KnobTuner,
    pub max_batched_tokens: KnobTuner,
    pub speculative_block_len: KnobTuner,
    pub kv_compression_bit_width: KnobTuner,
}

impl SelfTuningController {
    pub fn new(target_ttft_ms: f64, target_itl_ms: f64) -> Self {
        Self {
            target_ttft_ms,
            target_itl_ms,
            target_kv_drift_quality: 0.05,
            ema_ttft_ms: target_ttft_ms,
            ema_itl_ms: target_itl_ms,
            ema_quality: 0.0,
            alpha: 0.15,
            // Defaults: chunked prefill size starts at 512 tokens,
            // max_batched_tokens at 4096, speculative block_len at 5,
            // KV compression bit_width at 4. Each tuner keeps its
            // observable target — TTFT for the first two, ITL for the
            // third, quality drift for the fourth.
            chunked_prefill_size: KnobTuner::new_fixed(
                KnobKind::ChunkedPrefillSize,
                /* target_pressure_ms */ target_ttft_ms,
                /* floor=64 */ 64.0,
                /* ceiling=4096 */ 4096.0,
                /* initial=512 */ 512.0,
                /* scale_step=0.10 */ 0.10,
            ),
            max_batched_tokens: KnobTuner::new_fixed(
                KnobKind::MaxBatchedTokens,
                target_ttft_ms,
                /* floor=512 */ 512.0,
                /* ceiling=8192 */ 8192.0,
                /* initial=4096 */ 4096.0,
                /* scale_step=0.10 */ 0.10,
            ),
            speculative_block_len: KnobTuner::new_fixed(
                KnobKind::SpeculativeBlockLen,
                target_itl_ms,
                /* floor=1 */ 1.0,
                /* ceiling=16 */ 16.0,
                /* initial=5 */ 5.0,
                /* scale_step=0.50 */ 0.50,
            ),
            kv_compression_bit_width: KnobTuner::new_fixed(
                KnobKind::KvCompressionBitWidth,
                0.05,
                /* floor=2 */ 2.0,
                /* ceiling=8 */ 8.0,
                /* initial=4 */ 4.0,
                /* scale_step=1.5 */ 1.5,
            ),
        }
    }

    pub fn record_ttft(&mut self, ms: f64) {
        self.ema_ttft_ms = (1.0 - self.alpha) * self.ema_ttft_ms + self.alpha * ms;
    }

    pub fn record_itl(&mut self, ms: f64) {
        self.ema_itl_ms = (1.0 - self.alpha) * self.ema_itl_ms + self.alpha * ms;
    }

    pub fn record_quality(&mut self, drift: f64) {
        let d = drift.clamp(0.0, 1.0);
        self.ema_quality = (1.0 - self.alpha) * self.ema_quality + self.alpha * d;
    }

    pub fn ema_ttft(&self) -> f64 {
        self.ema_ttft_ms
    }

    pub fn ema_itl(&self) -> f64 {
        self.ema_itl_ms
    }

    pub fn ema_quality(&self) -> f64 {
        self.ema_quality
    }

    /// Independent per-knob: chunked_prefill_size and max_batched_tokens
    /// both home in on TTFT; speculative block_len homes in on ITL; KV
    /// compression bit-width homes in on quality drift.
    pub fn tune_one(&mut self, knob: KnobKind) -> f64 {
        let _ = knob;
        self.tune_all();
        // Re-tunes all knobs together but the chosen knob's value is
        // the only one returned for unit tests that want a single
        // observable. Independence is preserved by `tune_all` not
        // cross-referencing knobs.
        match knob {
            KnobKind::ChunkedPrefillSize => self.chunked_prefill_size.current,
            KnobKind::MaxBatchedTokens => self.max_batched_tokens.current,
            KnobKind::SpeculativeBlockLen => self.speculative_block_len.current,
            KnobKind::KvCompressionBitWidth => self.kv_compression_bit_width.current,
        }
    }

    /// Run all four knob tuners independently. The returns map each
    /// knob's chosen value so the caller can apply it. Each knob uses
    /// its own EMA — there is no implicit cross-knob coupling.
    pub fn tune_all(&mut self) -> KnobValues {
        // chunked_prefill_size: target = TTFT
        self.chunked_prefill_size.ema_observed = self.ema_ttft_ms;
        self.chunked_prefill_size.record(self.ema_ttft_ms, self.alpha);
        let cp = self.chunked_prefill_size.tune();

        // max_batched_tokens: target = TTFT
        self.max_batched_tokens.record(self.ema_ttft_ms, self.alpha);
        let mb = self.max_batched_tokens.tune();

        // speculative_block_len: target = ITL
        self.speculative_block_len.record(self.ema_itl_ms, self.alpha);
        let sb = self.speculative_block_len.tune();

        // kv_compression_bit_width: target = quality drift target.
        // Higher drift (over target) pushes bits lower; lower drift
        // (under target) leaves room to widen resolution.
        self.kv_compression_bit_width.record(self.ema_quality, self.alpha);
        let kw = self.kv_compression_bit_width.tune();

        KnobValues {
            chunked_prefill_size: cp as usize,
            max_batched_tokens: mb as usize,
            speculative_block_len: sb as usize,
            kv_compression_bit_width: kw as u8,
        }
    }
}

/// Snapshot of all four tuned values after a `tune_all` call. Unit tests
/// can pin one knob at a time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnobValues {
    pub chunked_prefill_size: usize,
    pub max_batched_tokens: usize,
    pub speculative_block_len: usize,
    pub kv_compression_bit_width: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdmissionController;

    #[test]
    fn self_tuning_ema_and_adjustments_recorded() {
        // Each EMA tracker updates on observation — backward
        // compatibility with the v1 surface that callers depended on.
        let mut controller = SelfTuningController::new(100.0, 10.0);
        controller.record_ttft(300.0);
        controller.record_itl(20.0);
        assert!(controller.ema_ttft() > 100.0);
        assert!(controller.ema_itl() > 10.0);
    }

    #[test]
    fn ema_quality_record_and_getback() {
        let mut controller = SelfTuningController::new(100.0, 10.0);
        controller.record_quality(0.10);
        assert!(controller.ema_quality() > 0.0);
        assert!(controller.ema_quality() < 1.0);
    }

    #[test]
    fn ttft_over_budget_shrinks_max_batched_tokens() {
        // Simulate high TTFT — the controller must reduce max_batched_tokens.
        let mut controller = SelfTuningController::new(100.0, 10.0);
        let initial_mbt = controller.max_batched_tokens.current as usize;
        for _ in 0..10 {
            controller.record_ttft(300.0);
        }
        let values = controller.tune_all();
        assert!(
            values.max_batched_tokens < initial_mbt,
            "max_batched_tokens must shrink when TTFT blows the budget"
        );
        // Floor enforced: even after several shrank-steps we land at 512, never below.
        assert!(values.max_batched_tokens >= 512);
    }

    #[test]
    fn ttft_under_budget_grows_max_batched_tokens() {
        // Margin present — max_batched_tokens should grow, capping at ceiling.
        let mut controller = SelfTuningController::new(100.0, 10.0);
        for _ in 0..10 {
            controller.record_ttft(20.0);
        }
        let values = controller.tune_all();
        assert!(
            values.max_batched_tokens > 4096,
            "max_batched_tokens must grow when TTFT is well under budget"
        );
        // Ceiling enforced.
        assert!(values.max_batched_tokens <= 8192);
    }

    #[test]
    fn chunked_prefill_size_independent_from_max_batched() {
        // Two knobs, both responsive to TTFT pressure but with different
        // scale_step/floor — verify the controller does NOT collapse
        // them into a single value when they have distinct bounds.
        let mut controller = SelfTuningController::new(100.0, 10.0);
        for _ in 0..10 {
            controller.record_ttft(400.0);
        }
        let values = controller.tune_all();
        // Both shrink, but to different scales.
        assert!(values.chunked_prefill_size < 512);
        assert!(values.max_batched_tokens < 4096);
        // Their respective floors differ — chunked_prefill_size can go lower than mbt.
        assert_ne!(values.chunked_prefill_size, values.max_batched_tokens);
    }

    #[test]
    fn speculative_block_len_responds_to_itl() {
        // ITL over budget → reduce block_len. ITL under budget → grow.
        let mut controller = SelfTuningController::new(1000.0, 10.0);
        for _ in 0..5 {
            controller.record_itl(50.0);
        }
        let v1 = controller.tune_all();
        assert!(
            v1.speculative_block_len < 5,
            "elevated ITL must shrink the speculative block length"
        );
    }

    #[test]
    fn kv_bit_width_shrinks_under_quality_pressure() {
        // Quality drift above target → reduce bit width (memory savings).
        let mut controller = SelfTuningController::new(1000.0, 10.0);
        for _ in 0..10 {
            controller.record_quality(0.20); // target is 0.05
        }
        let v = controller.tune_all();
        assert!(
            v.kv_compression_bit_width < 4,
            "high quality drift must push kv bit-width below 4"
        );
        assert!(v.kv_compression_bit_width >= 2, "floor at 2 bits");
    }

    #[test]
    fn kv_bit_width_grows_when_quality_drift_is_low() {
        // Quality drift below target → can afford higher bits (better resolution).
        let mut controller = SelfTuningController::new(1000.0, 10.0);
        for _ in 0..10 {
            controller.record_quality(0.01);
        }
        let v = controller.tune_all();
        assert!(
            v.kv_compression_bit_width >= 4,
            "low quality drift allows growing resolution"
        );
        assert!(v.kv_compression_bit_width <= 8, "ceiling at 8 bits");
    }

    #[test]
    fn tune_one_returns_value_of_chosen_knob_only() {
        let mut controller = SelfTuningController::new(100.0, 10.0);
        for _ in 0..10 {
            controller.record_ttft(300.0);
        }
        let cp = controller.tune_one(KnobKind::ChunkedPrefillSize);
        assert!(cp >= 64.0 && cp <= 4096.0);
    }

    #[test]
    fn all_knobs_stay_in_bounds_after_extreme_inputs() {
        let mut controller = SelfTuningController::new(100.0, 10.0);
        // Inject absurdly high / low observations to validate floor/ceiling clamps.
        for _ in 0..50 {
            controller.record_ttft(100_000.0);
            controller.record_itl(1_000.0);
            controller.record_quality(1.0);
        }
        let v = controller.tune_all();
        assert_eq!(v.chunked_prefill_size, 64); // floor
        assert_eq!(v.max_batched_tokens, 512); // floor
        assert_eq!(v.speculative_block_len, 1); // floor
        assert_eq!(v.kv_compression_bit_width, 2); // floor
    }
}
