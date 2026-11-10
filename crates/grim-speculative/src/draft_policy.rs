//! Online draft-source policy (astrum P5): E/cost gate, output-neutral.
//!
//! Learns per (context-length bucket) EMA of committed tokens/ms for each draft
//! source and selects the better one with hysteresis. Output-neutral: changes
//! WHEN drafts are taken, never the tokens accepted. Gated by
//! `GRIM_ASTRUM_DRAFT_POLICY=1` at call sites (this module is pure logic).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DraftSource {
    Mtp,
    Suffix,
}

#[derive(Debug, Clone, Copy)]
struct Ema {
    value: f64,
    seeded: bool,
}

impl Ema {
    fn new() -> Self {
        Self { value: 0.0, seeded: false }
    }
    fn observe(&mut self, x: f64, alpha: f64) {
        if !self.seeded {
            self.value = x;
            self.seeded = true;
        } else {
            self.value = alpha * x + (1.0 - alpha) * self.value;
        }
    }
    fn get(&self) -> f64 {
        self.value
    }
    fn has_data(&self) -> bool {
        self.seeded
    }
}

const BUCKETS: usize = 3; // 0: <=512, 1: 513..=2048, 2: 2049+
const ALPHA: f64 = 0.1;

fn bucket(ctx_len: usize) -> usize {
    if ctx_len <= 512 { 0 } else if ctx_len <= 2048 { 1 } else { 2 }
}

#[derive(Debug, Clone)]
struct SourceStats {
    per_bucket: [Ema; BUCKETS],
}

impl SourceStats {
    fn new() -> Self {
        Self { per_bucket: [Ema::new(), Ema::new(), Ema::new()] }
    }
}

/// Online E/cost draft-source selector.
pub struct DraftPolicy {
    margin: f64,
    mtp: SourceStats,
    suffix: SourceStats,
    last: [DraftSource; BUCKETS],
}

impl DraftPolicy {
    pub fn new(margin: f64) -> Self {
        Self {
            margin,
            mtp: SourceStats::new(),
            suffix: SourceStats::new(),
            last: [DraftSource::Mtp; BUCKETS],
        }
    }

    /// Select source for a context length (bucketed). MTP is the default until
    /// data shows suffix wins by more than `margin` with hysteresis.
    pub fn select(&mut self, ctx_len: usize) -> DraftSource {
        let b = bucket(ctx_len);
        let mtp = &self.mtp.per_bucket[b];
        let sfx = &self.suffix.per_bucket[b];
        let chosen = if !sfx.has_data() {
            DraftSource::Mtp
        } else if !mtp.has_data() {
            DraftSource::Suffix
        } else {
            let m = mtp.get();
            let s = sfx.get();
            if s > m * (1.0 + self.margin) {
                DraftSource::Suffix
            } else if m > s * (1.0 + self.margin) {
                DraftSource::Mtp
            } else {
                self.last[b] // hysteresis: keep previous within margin
            }
        };
        self.last[b] = chosen;
        chosen
    }

    /// Observe a completed draft round: committed tokens produced at `cost_ms`.
    pub fn observe(&mut self, src: DraftSource, committed: usize, cost_ms: f64) {
        self.observe_in_bucket(src, 0, committed, cost_ms);
    }

    /// Observe in an explicit context bucket (0/1/2) for deterministic testing/wiring.
    pub fn observe_in_bucket(
        &mut self,
        src: DraftSource,
        ctx_bucket: usize,
        committed: usize,
        cost_ms: f64,
    ) {
        let b = ctx_bucket.min(BUCKETS - 1);
        let x = if cost_ms > 0.0 {
            committed as f64 / cost_ms
        } else {
            0.0
        };
        match src {
            DraftSource::Mtp => self.mtp.per_bucket[b].observe(x, ALPHA),
            DraftSource::Suffix => self.suffix.per_bucket[b].observe(x, ALPHA),
        }
    }

    /// Convenience: observe and return the current selection for a context length.
    pub fn observe_and_select(
        &mut self,
        src: DraftSource,
        ctx_len: usize,
        committed: usize,
        cost_ms: f64,
    ) -> DraftSource {
        let b = bucket(ctx_len);
        self.observe_in_bucket(src, b, committed, cost_ms);
        self.select(ctx_len)
    }
}

pub fn draft_policy_enabled() -> bool {
    std::env::var("GRIM_ASTRUM_DRAFT_POLICY").ok().as_deref() == Some("1")
}
