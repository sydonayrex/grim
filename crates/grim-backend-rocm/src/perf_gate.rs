//! Phase-3 §3.7 - Profiling CI gate. Regression harness that compares a measured `cycles_per_call` against a baseline JSON keyed by kernel
//! name and arch, and decides: - `Within { delta_pct, threshold_pct }` - measurement is within budget (≤ threshold-pct above baseline).
//!
//! Audit §3.4 wiring (manual-only gate):
//!
//! - The committed baseline lives at `benches/baseline/rocm_perf_baseline.json`
//!   (serde alias `arch` for `gpu_arch`; see `BaselineTableJson`). It is
//!   loaded with [`BaselineTable::load_baseline`] (or
//!   [`BaselineTable::load_default_baseline`] for the canonical path).
//! - Manual local run ONLY, never CI-gated (consumer ROCm project, no GPU CI):
//!   `cargo test --test perf_gate -- --nocapture` parses the JSON on CPU;
//!   `GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --test perf_gate -- --ignored --nocapture`
//!   (or `--bench kv_dequant_perf`) runs the GPU harness. No `.github/workflows`
//!   job invokes this gate.
//! - Timing note: benches currently time with `Instant::now()` wall clock
//!   (host synchronize per iteration; see `docs/benchmarks.md` §2). GPU-event
//!   timing (`hipEvent`) is the follow-up; it is deliberately NOT rewritten in
//!   this pass.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use grim_tensor::error::{Error, Result};

/// A single measurement on the bench side.
///
/// ```
/// use grim_backend_rocm::perf_gate::Measurement;
///
/// let m = Measurement { cycles_per_call: 1100.0 };
/// assert!((m.delta_pct_vs(1000.0) - 10.0).abs() < 1e-9);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub cycles_per_call: f64,
}

impl Measurement {
    /// `(current - baseline) / baseline * 100`.
    /// Returned as `f64::NAN` if either side is zero - there is no parlance for that.
    pub fn delta_pct_vs(&self, baseline: f64) -> f64 {
        if baseline == 0.0 || self.cycles_per_call == 0.0 {
            return f64::NAN;
        }
        ((self.cycles_per_call - baseline) / baseline) * 100.0
    }
}

/// Per-key baseline entry.
///
/// ```
/// use grim_backend_rocm::perf_gate::BaselineEntry;
///
/// let entry = BaselineEntry {
///     kernel: "grim_qkv_attention".to_string(),
///     baseline_cycles_per_call: 1000.0,
///     threshold_pct: 5.0,
/// };
/// assert_eq!(entry.threshold_pct, 5.0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub kernel: String,
    pub baseline_cycles_per_call: f64,
    pub threshold_pct: f64,
}

/// Decoded baseline JSON carried in memory. The byte-level shape lives
/// below in `BaselineTableJson`; public types use borrowed views.
///
/// ```
/// use grim_backend_rocm::perf_gate::BaselineTable;
///
/// let mut table = BaselineTable::for_arch("gfx1036");
/// assert_eq!(table.arch(), "gfx1036");
/// table
///     .set_entry("grim_qkv_attention", 1000.0, 5.0)
///     .unwrap();
/// assert!(table.entry("grim_qkv_attention").is_some());
/// let json = table.to_json_pretty().unwrap();
/// let reparsed = BaselineTable::from_json(&json).unwrap();
/// assert_eq!(reparsed.arch(), "gfx1036");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineTable {
    arch: String,
    inner: HashMap<String, BaselineEntry>,
    /// `hipconfig --version` of the machine that produced this baseline.
    rocm_version: String,
    /// Decode batch size the baseline was measured at (decode regime).
    batch: Option<u32>,
    /// Regime the baseline covers (e.g. `"decode"` — decode, not prefill).
    prefill_decode: Option<String>,
    /// End-to-end tokens/sec recorded alongside the per-kernel baselines.
    tok_per_s: Option<f64>,
    /// When the baseline was measured (or `"TEMPLATE"` before the first run).
    measured_at: Option<String>,
}

/// Wire-format mirror of `BaselineTable`. Kept private so the public type
/// can evolve independently.
///
/// Schema (audit §3.4, `benches/baseline/rocm_perf_baseline.json`):
/// `{arch, rocm_version, batch, prefill_decode ("decode"), tok_per_s,
/// measured_at, entries: [{kernel, baseline_cycles_per_call, threshold_pct}]}`
/// (`gpu_arch` is accepted as an alias for `arch`.) The repo checkout is
/// nightly `main` (§2 of `docs/benchmarks.md`), so `rocm_version` records the
/// runtime the baseline was measured under (`hipconfig --version`); results
/// without arch/version are not reproducible.
#[derive(Debug, Serialize, Deserialize)]
struct BaselineTableJson {
    #[serde(rename = "arch", alias = "gpu_arch")]
    gpu_arch: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    rocm_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    batch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prefill_decode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tok_per_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    measured_at: Option<String>,
    entries: Vec<BaselineEntry>,
}

impl BaselineTable {
    /// Construct an empty table for a given arch. Helpers build a
    /// baseline file from a live benchmark via `set_entry` → `to_json_pretty`.
    pub fn for_arch(arch: impl Into<String>) -> Self {
        Self {
            arch: arch.into(),
            inner: HashMap::new(),
            rocm_version: String::new(),
            batch: None,
            prefill_decode: None,
            tok_per_s: None,
            measured_at: None,
        }
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// ROCm runtime version this baseline was measured under (`hipconfig --version`);
    /// empty when the table was built in-memory.
    pub fn rocm_version(&self) -> &str {
        &self.rocm_version
    }

    /// Decode batch size recorded with the baseline, if any.
    pub fn batch(&self) -> Option<u32> {
        self.batch
    }

    /// Regime covered by the baseline (e.g. `"decode"`).
    pub fn prefill_decode(&self) -> Option<&str> {
        self.prefill_decode.as_deref()
    }

    /// End-to-end tokens/sec recorded with the baseline, if any.
    pub fn tok_per_s(&self) -> Option<f64> {
        self.tok_per_s
    }

    /// When the baseline was measured, or `"TEMPLATE"` pre-first-run.
    pub fn measured_at(&self) -> Option<&str> {
        self.measured_at.as_deref()
    }

    /// Decide whether the entry is acceptable: `baseline > 0` and `threshold_pct > 0`.
    /// We reject zero/negative baselines *eagerly* so a regression test that runs against a corrupt baseline.
    pub fn set_entry(
        &mut self,
        kernel: &str,
        baseline_cycles_per_call: f64,
        threshold_pct: f64,
    ) -> Result<()> {
        if baseline_cycles_per_call <= 0.0 {
            return Err(Error::Backend(format!(
                "BaselineTable::set_entry: baseline_cycles_per_call must be > 0 (kernel={})",
                kernel
            )));
        }
        if threshold_pct <= 0.0 {
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
            rocm_version: self.rocm_version.clone(),
            batch: self.batch,
            prefill_decode: self.prefill_decode.clone(),
            tok_per_s: self.tok_per_s,
            measured_at: self.measured_at.clone(),
            entries,
        };
        serde_json::to_string_pretty(&wire)
            .map_err(|e| Error::Backend(format!("BaselineTable::to_json_pretty: serde: {}", e)))
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let wire: BaselineTableJson = serde_json::from_str(s)
            .map_err(|e| Error::Backend(format!("BaselineTable::from_json: serde: {}", e)))?;
        validate_wire(&wire)?;
        let mut inner = HashMap::with_capacity(wire.entries.len());
        for e in wire.entries {
            let existing = inner.insert(e.kernel.clone(), e);
            if existing.is_some() {
                return Err(Error::Backend(format!(
                    "duplicate kernel name in baseline: {}",
                    existing
                        .as_ref()
                        .map(|b| b.kernel.as_str())
                        .unwrap_or("<unknown>")
                )));
            }
        }
        let table = Self {
            arch: wire.gpu_arch,
            inner,
            rocm_version: wire.rocm_version,
            batch: wire.batch,
            prefill_decode: wire.prefill_decode,
            tok_per_s: wire.tok_per_s,
            measured_at: wire.measured_at,
        };
        Ok(table)
    }

    /// Load the committed baseline JSON from disk (audit §3.4 wiring).
    ///
    /// Missing file, unreadable file, or invalid schema all surface as
    /// `Error::Backend` — the gate never fabricates a baseline.
    pub fn load_baseline(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path).map_err(|e| {
            Error::Backend(format!(
                "BaselineTable::load_baseline: read {}: {}",
                path.display(),
                e
            ))
        })?;
        Self::from_json(&s)
    }

    /// Canonical committed-baseline path: `benches/baseline/rocm_perf_baseline.json`
    /// relative to this crate's manifest dir.
    pub fn default_baseline_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("benches/baseline/rocm_perf_baseline.json")
    }

    /// Load the committed baseline (`Desktop$CARGO_MANIFEST_DIR/benches/baseline/rocm_perf_baseline.json`).
    pub fn load_default_baseline() -> Result<Self> {
        Self::load_baseline(Self::default_baseline_path())
    }
}

fn validate_wire(wire: &BaselineTableJson) -> Result<()> {
    if wire.gpu_arch.is_empty() {
        return Err(Error::Backend("baseline json: gpu_arch is empty".into()));
    }
    if let Some(batch) = wire.batch {
        if batch == 0 {
            return Err(Error::Backend(format!(
                "baseline json: batch must be > 0, got {}",
                batch
            )));
        }
    }
    for e in &wire.entries {
        if e.kernel.is_empty() {
            return Err(Error::Backend("baseline json: empty kernel name".into()));
        }
        if e.baseline_cycles_per_call <= 0.0 {
            return Err(Error::Backend(format!(
                "baseline json: baseline_cycles_per_call must be > 0 ({})",
                e.kernel
            )));
        }
        if e.threshold_pct <= 0.0 {
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
    Within {
        delta_pct: f64,
        threshold_pct: f64,
    },
    Regressed {
        baseline: f64,
        current: f64,
        delta_pct: f64,
        threshold_pct: f64,
    },
    NoBaseline {
        reason: String,
    },
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
            Self::Within {
                delta_pct,
                threshold_pct,
            }
        }
    }

    /// Construct a `Regressed` verdict.
    pub fn regressed(current: f64, threshold_pct: f64) -> Self {
        // baseline implied at +threshold_pct boundary; here we compute a hypothetical baseline such that delta=(current/baseline-1)*100.
        // The caller should never rely on the round-trip identity for this concession; tests cover it.
        let baseline = if threshold_pct > 0.0 {
            current / (1.0 + threshold_pct / 100.0)
        } else {
            0.0
        };
        let delta_pct = if baseline > 0.0 {
            (current - baseline) / baseline * 100.0
        } else {
            f64::NAN
        };
        Self::Regressed {
            baseline,
            current,
            delta_pct,
            threshold_pct,
        }
    }

    /// True when the gate should fail CI on this verdict.
    pub fn fails_ci(&self) -> bool {
        matches!(self, Verdict::Regressed { .. })
    }
}

/// The actual gate. Holds a `BaselineTable` and applies the per-key
/// threshold when asked to compare a measurement.
///
/// ```
/// use grim_backend_rocm::perf_gate::{BaselineTable, Measurement, PerfGate, Verdict};
///
/// let mut table = BaselineTable::for_arch("gfx1036");
/// table
///     .set_entry("grim_qkv_attention", 1000.0, 5.0)
///     .unwrap();
/// let gate = PerfGate::new(table);
///
/// let within = gate.compare(
///     "grim_qkv_attention",
///     Measurement { cycles_per_call: 1030.0 },
/// );
/// assert!(matches!(within, Verdict::Within { .. }));
///
/// let regressed = gate.compare(
///     "grim_qkv_attention",
///     Measurement { cycles_per_call: 1200.0 },
/// );
/// assert!(regressed.fails_ci());
/// ```
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
    /// Boundary semantics: `delta_pct <= threshold_pct` is `Within`.
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
            Verdict::Within {
                delta_pct,
                threshold_pct: entry.threshold_pct,
            }
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

// We expose a tiny mutator for the self-test only.
// It bypasses the public `set_entry` validator (which is exercised in the public tests).
impl BaselineTable {
    #[doc(hidden)]
    pub fn inner_mut_for_tests(&mut self) -> &mut HashMap<String, BaselineEntry> {
        &mut self.inner
    }
}

#[cfg(test)]
mod gate_self_tests {
    //! Tiny self-tests that prove the gate uses the threshold the spec calls for (default +5%) on a bare-bones entry.
    //! The bulk of the tests live in `tests/perf_gate.rs`; here we only keep one boundary check.

    use super::{BaselineEntry, BaselineTable, Measurement, PerfGate, Verdict};

    #[test]
    fn regressed_path_returns_regressed_kind() {
        let mut t = crate::perf_gate::BaselineTable::for_arch("gfx1036");
        // Insert via the test-only mutator so the self-test exercises the
        // compare path without going through the public validator.
        t.inner_mut_for_tests().insert(
            "k".to_string(),
            BaselineEntry {
                kernel: "k".to_string(),
                baseline_cycles_per_call: 1.0,
                threshold_pct: 5.0,
            },
        );
        let g = PerfGate::new(t);
        let v = g.compare(
            "k",
            Measurement {
                cycles_per_call: 1.03,
            },
        );
        assert!(
            matches!(v, Verdict::Within { .. }),
            "+3% slower is within budget"
        );
        let v = g.compare(
            "k",
            Measurement {
                cycles_per_call: 1.20,
            },
        );
        assert!(
            matches!(v, Verdict::Regressed { .. }),
            "+20% slower regresses"
        );
    }

    #[test]
    fn committed_baseline_template_parses_with_metadata() {
        // CPU-only: validates the committed template
        // `benches/baseline/rocm_perf_baseline.json` parses via the same
        // `from_json` path the manual GPU gate uses, and carries the
        // reproducibility metadata (arch + ROCm version).
        let table = BaselineTable::load_default_baseline()
            .expect("committed baseline template must parse");
        assert_eq!(table.arch(), "gfx1036");
        assert!(!table.rocm_version().is_empty(), "template must record rocm_version");
        assert_eq!(table.batch(), Some(1));
        assert_eq!(table.prefill_decode(), Some("decode"));
        assert!(table.tok_per_s().is_some_and(|v| v > 0.0));
        assert_eq!(table.measured_at(), Some("TEMPLATE"));
        assert!(
            table.entry("grim_qkv_attention").is_some(),
            "template must carry a grim_qkv_attention entry"
        );
    }

    #[test]
    fn baseline_rejects_zero_batch() {
        let bad = r#"{"gpu_arch":"gfx1036","rocm_version":"7.x","batch":0,"entries":[]}"#;
        assert!(BaselineTable::from_json(bad).is_err(), "batch 0 must be rejected");
    }
}
