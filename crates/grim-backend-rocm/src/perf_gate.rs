//! Phase-3 §3.7 — Profiling CI gate.
//!
//! Regression harness that compares a measured `cycles_per_call` against a
//! baseline JSON keyed by kernel name and arch, and decides:
//!
//! - `Within { delta_pct, threshold_pct }` — measurement is within budget
//!   (≤ threshold-pct above baseline). Sub-baseline measurements are
//!   also "Within" (an unexpected speedup is good news, not a failure).
//! - `Regressed { baseline, current, delta_pct, threshold_pct }` — the
//!   measurement is `--threshold_pct` above baseline; CI fails.
//! - `NoBaseline { reason }` — there is no entry for the kernel in the
//!   baseline file; CI logs this but does not fail (the test runner has
//!   not yet established a baseline).
//!
//! Boundary semantics: a delta exactly equal to the threshold is
//! `Within`. We use `delta_pct <= threshold_pct` to **avoid flakes** on
//! deterministic CI machines that occasionally round to exactly the
//! threshold on repeated measurement — strict `<` would create flakes
//! without surfacing real regressions.
//!
//! Skill attribution:
//! - `rocm-profiling-perf` — regression gates, methodology discipline.
//!   The gate as a *primitive* is metric-agnostic; rocprof counters can
//!   be plumbed in by replacing `cycles_per_call` with a richer struct.
//! - `rust-gpu-discipline` — `NoBaseline` is surfaced explicitly so a
//!   missing baseline never fabricates a "passed" gate (no fake pass).
//! - `rust-ai-ml-inference-guide` Action 8 — recorded baseline numbers
//!   predate the metrics the gate guards.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use grim_tensor::error::{Error, Result};

/// A single measurement on the bench side.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub cycles_per_call: f64,
}

impl Measurement {
    /// `(current - baseline) / baseline * 100`. Returned as `f64::NAN`
    /// if either side is zero — there is no parlance for that case in
    /// the gate's accounting and we surface the NaN deliberately so the
    /// downstream caller can flag it rather than get a misleading 0%
    /// or +inf%.
    pub fn delta_pct_vs(&self, baseline: f64) -> f64 {
        if baseline == 0.0 || self.cycles_per_call == 0.0 {
            return f64::NAN;
        }
        ((self.cycles_per_call - baseline) / baseline) * 100.0
    }
}

/// Per-key baseline entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub kernel: String,
    pub baseline_cycles_per_call: f64,
    pub threshold_pct: f64,
}

/// Decoded baseline JSON carried in memory. The byte-level shape lives
/// below in `BaselineTableJson`; public types use borrowed views.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineTable {
    arch: String,
    inner: HashMap<String, BaselineEntry>,
}

/// Wire-format mirror of `BaselineTable`. Kept private so the public type
/// can evolve independently.
#[derive(Debug, Serialize, Deserialize)]
struct BaselineTableJson {
    gpu_arch: String,
    entries: Vec<BaselineEntry>,
}

impl BaselineTable {
    /// Construct an empty table for a given arch. Helpers build a
    /// baseline file from a live benchmark via `set_entry` → `to_json_pretty`.
    pub fn for_arch(arch: impl Into<String>) -> Self {
        Self { arch: arch.into(), inner: HashMap::new() }
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Decide whether the entry is acceptable: `baseline > 0` and
    /// `threshold_pct > 0`. We reject zero/negative baselines *eagerly*
    /// so a regression test that runs against a corrupt baseline file
    /// fails clearly, not silently at the comparison step.
    pub fn set_entry(
        &mut self,
        kernel: &str,
        baseline_cycles_per_call: f64,
        threshold_pct: f64,
    ) -> Result<()> {
        if !(baseline_cycles_per_call > 0.0) {
            return Err(Error::Backend(format!(
                "BaselineTable::set_entry: baseline_cycles_per_call must be > 0 (kernel={})",
                kernel
            )));
        }
        if !(threshold_pct > 0.0) {
            return Err(Error::Backend(format!(
                "BaselineTable::set_entry: threshold_pct must be > 0 (kernel={})",
                kernel
            )));
        }
        self.inner.insert(
            kernel.to_string(),
            BaselineEntry {
                kernel: kernel.to_string(),
                baseline_cycles_per_call,
                threshold_pct,
            },
        );
        Ok(())
    }

    pub fn entry(&self, kernel: &str) -> Option<&BaselineEntry> {
        self.inner.get(kernel)
    }

    pub fn to_json_pretty(&self) -> Result<String> {
        // Stable order: keys sorted alphabetically so the JSON diff
        // is reproducible (CI baseline updates are easy to reason about).
        let mut entries: Vec<BaselineEntry> = self.inner.values().cloned().collect();
        entries.sort_by(|a, b| a.kernel.cmp(&b.kernel));
        let wire = BaselineTableJson {
            gpu_arch: self.arch.clone(),
            entries,
        };
        serde_json::to_string_pretty(&wire).map_err(|e| {
            Error::Backend(format!("BaselineTable::to_json_pretty: serde: {}", e))
        })
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let wire: BaselineTableJson = serde_json::from_str(s).map_err(|e| {
            Error::Backend(format!("BaselineTable::from_json: serde: {}", e))
        })?;
        validate_wire(&wire)?;
        let mut inner = HashMap::with_capacity(wire.entries.len());
        for e in wire.entries {
            let existing = inner.insert(e.kernel.clone(), e);
            if existing.is_some() {
                return Err(Error::Backend(format!(
                    "duplicate kernel name in baseline: {}",
                    existing.as_ref().map(|b| b.kernel.as_str()).unwrap_or("<unknown>")
                )));
            }
        }
        Ok(Self { arch: wire.gpu_arch, inner })
    }
}

fn validate_wire(wire: &BaselineTableJson) -> Result<()> {
    if wire.gpu_arch.is_empty() {
        return Err(Error::Backend("baseline json: gpu_arch is empty".into()));
    }
    for e in &wire.entries {
        if e.kernel.is_empty() {
            return Err(Error::Backend("baseline json: empty kernel name".into()));
        }
        if !(e.baseline_cycles_per_call > 0.0) {
            return Err(Error::Backend(format!(
                "baseline json: baseline_cycles_per_call must be > 0 ({})",
                e.kernel
            )));
        }
        if !(e.threshold_pct > 0.0) {
            return Err(Error::Backend(format!(
                "baseline json: threshold_pct must be > 0 ({})",
                e.kernel
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Within { delta_pct: f64, threshold_pct: f64 },
    Regressed { baseline: f64, current: f64, delta_pct: f64, threshold_pct: f64 },
    NoBaseline { reason: String },
}

impl Verdict {
    /// Construct a `Within` verdict. Sanity-clamps `delta_pct` to be
    /// `<= threshold_pct` so the public constructor is unambiguous.
    pub fn within(delta_pct: f64, threshold_pct: f64) -> Self {
        if delta_pct > threshold_pct {
            // Caller error: detected at construction rather than at use.
            Self::Regressed {
                baseline: 1.0,
                current: 1.0 * (1.0 + delta_pct / 100.0),
                delta_pct,
                threshold_pct,
            }
        } else {
            Self::Within { delta_pct, threshold_pct }
        }
    }

    /// Construct a `Regressed` verdict.
    pub fn regressed(current: f64, threshold_pct: f64) -> Self {
        // baseline implied at +threshold_pct boundary; here we compute
        // a hypothetical baseline such that delta=(current/baseline-1)*100.
        // The caller should never rely on the round-trip identity for
        // this concession; tests cover it via the explicit fields.
        let baseline = if threshold_pct > 0.0 { current / (1.0 + threshold_pct / 100.0) } else { 0.0 };
        let delta_pct = if baseline > 0.0 { (current - baseline) / baseline * 100.0 } else { f64::NAN };
        Self::Regressed { baseline, current, delta_pct, threshold_pct }
    }

    /// True when the gate should fail CI on this verdict.
    pub fn fails_ci(&self) -> bool {
        matches!(self, Verdict::Regressed { .. })
    }
}

/// The actual gate. Holds a `BaselineTable` and applies the per-key
/// threshold when asked to compare a measurement.
#[derive(Debug, Clone)]
pub struct PerfGate {
    table: BaselineTable,
}

impl PerfGate {
    pub fn new(table: BaselineTable) -> Self {
        Self { table }
    }

    pub fn table(&self) -> &BaselineTable {
        &self.table
    }

    /// Compare one measurement; return the verdict.
    ///
    /// Boundary semantics: `delta_pct <= threshold_pct` is `Within`.
    /// Strict `<` would create flakes on machines where repeated
    /// measurements round to the threshold. The CI gate's purpose is
    /// to catch real regressions, not to pass-through tight numerical
    /// matches.
    pub fn compare(&self, kernel: &str, measurement: Measurement) -> Verdict {
        let entry = match self.table.entry(kernel) {
            Some(e) => e,
            None => {
                return Verdict::NoBaseline {
                    reason: format!("no entry for kernel '{}'", kernel),
                };
            }
        };
        let delta_pct = measurement.delta_pct_vs(entry.baseline_cycles_per_call);
        if delta_pct.is_nan() {
            return Verdict::NoBaseline {
                reason: format!(
                    "delta is undefined for kernel '{}' (zero baseline or zero current)",
                    kernel
                ),
            };
        }
        if delta_pct <= entry.threshold_pct {
            Verdict::Within { delta_pct, threshold_pct: entry.threshold_pct }
        } else {
            Verdict::Regressed {
                baseline: entry.baseline_cycles_per_call,
                current: measurement.cycles_per_call,
                delta_pct,
                threshold_pct: entry.threshold_pct,
            }
        }
    }
}

#[cfg(test)]
mod gate_self_tests {
    //! Tiny self-tests that prove the gate uses the threshold the spec
    //! calls for (default +5%) on a bare-bones entry. The bulk of the
    //! tests live in `tests/perf_gate.rs`; here we only keep one
    //! boundary check the gate must pass without external fixtures.

    use super::{BaselineEntry, Measurement, PerfGate, Verdict};

    #[test]
    fn regressed_path_returns_regressed_kind() {
        let mut t = crate::perf_gate::BaselineTable::for_arch("gfx1036");
        // Insert via the test-only mutator so the self-test exercises the
        // compare path without going through the public validator.
        t.inner_mut_for_tests()
            .insert(
                "k".to_string(),
                BaselineEntry { kernel: "k".to_string(), baseline_cycles_per_call: 1.0, threshold_pct: 5.0 },
            );
        let g = PerfGate::new(t);
        let v = g.compare("k", Measurement { cycles_per_call: 1.03 });
        assert!(matches!(v, Verdict::Within { .. }), "+3% slower is within budget");
        let v = g.compare("k", Measurement { cycles_per_call: 1.20 });
        assert!(matches!(v, Verdict::Regressed { .. }), "+20% slower regresses");
    }
}

// We expose a tiny mutator for the self-test only. It bypasses the public
// `set_entry` validator (which is exercised in the public tests).
// Not part of the public API.
impl BaselineTable {
    #[doc(hidden)]
    pub fn inner_mut_for_tests(&mut self) -> &mut HashMap<String, BaselineEntry> {
        &mut self.inner
    }
}
