//! §6 A/B harness — pure logic.
//!
//! The A/B is the plan's named deliverable: compare the codegen paths that
//! already exist in grim (Raven, ForestRaven, WhiteCrow) plus the ones being
//! built (WhiteRaven, GreyRaven), sweeping shape, reporting **accuracy and speed
//! together**, and emitting a machine-readable artifact.
//!
//! # Why accuracy and speed cannot be separated here
//!
//! The arms do not share a weight format, a kernel, or an accumulator. Raven
//! accumulates in f32; ForestRaven and WhiteCrow accumulate in i32. Arm-to-arm
//! bit-equality is therefore meaningless, and the pre-registered decision rule
//! (§6 rule 1) disqualifies on accuracy *first*. That ordering only works if the
//! accuracy gate is a separate, testable step — so it lives here, on CPU, and
//! the GPU driver feeds it.
//!
//! # The shapes, and why
//!
//! M sweeps 1..512 because the arms are not all decode-shaped: the dot family is
//! M <= 4, WhiteRaven is prefill-shaped, and the crossover between them is the
//! thing worth finding. K sweeps the projection sizes that dominate decode. Any
//! arm whose accumulator is i32 needs K % 128 == 0 (WhiteCrow's group), so the
//! sweep is aligned to 128 and every arm sees identical shapes.

use std::collections::BTreeMap;

/// One measured arm at one shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmResult {
    pub arm: &'static str,
    /// Instruction the arm is supposed to be, for artifact cross-reference.
    pub instruction: &'static str,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Kernel time in milliseconds, one entry per round.
    pub samples_ms: Vec<f32>,
    /// Max |gpu - oracle| over the output tile.
    pub max_abs_err: f64,
    /// Tolerance this arm is held to. F32-accumulating arms get a looser bound
    /// than integer ones because they sum in a different order.
    pub tolerance: f64,
}

impl ArmResult {
    /// Median sample. Timing distributions here are tight (best-of-N on an idle
    /// device) but the mean is pulled by the occasional scheduler hiccup, so the
    /// median is the headline and the spread is reported alongside.
    pub fn median_ms(&self) -> f32 {
        if self.samples_ms.is_empty() {
            return f32::NAN;
        }
        let mut v = self.samples_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    }

    pub fn min_ms(&self) -> f32 {
        self.samples_ms.iter().copied().fold(f32::INFINITY, f32::min)
    }

    pub fn p90_ms(&self) -> f32 {
        if self.samples_ms.is_empty() {
            return f32::NAN;
        }
        let mut v = self.samples_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[((v.len() as f32 * 0.9) as usize).min(v.len() - 1)]
    }

    /// Accuracy gate, applied **before** speed. An arm that is faster and wrong
    /// is a regression, not a win.
    pub fn accuracy_ok(&self) -> bool {
        self.max_abs_err <= self.tolerance
    }
}

/// Verdict for one (arm, shape) cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Accuracy gate passed; speed comparison is meaningful.
    Eligible,
    /// Accuracy gate failed. Disqualified regardless of speed.
    DisqualifiedAccuracy,
    /// Not run (arm absent on this arch, or shape unsupported).
    NotRun,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Eligible => "eligible",
            Verdict::DisqualifiedAccuracy => "disqualified_accuracy",
            Verdict::NotRun => "not_run",
        }
    }
}

/// The pre-registered bar from §6 rule 2, as fractions, not percentages.
pub const SHIP_BAR: f64 = 1.10;
/// Decode arms must clear this against the FP8 control.
pub const DECODE_BAR: f64 = 1.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    Decode,
    Prefill,
}

/// M values. Aligned so every arm, including the K%128-constrained integer ones,
/// is measured on the same shapes.
pub const M_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
/// K values: the projection sizes that dominate decode time.
pub const K_SWEEP: &[usize] = &[1024, 4096, 11008, 22016];
/// N is held at one decode-typical value; the plan scopes the sweep to M and K.
pub const N_FIXED: usize = 4096;

pub fn shape_class(m: usize) -> ShapeClass {
    if m <= 8 {
        ShapeClass::Decode
    } else {
        ShapeClass::Prefill
    }
}

/// Every (m, k) cell, K aligned to 128 so the i32-accumulating arms are legal.
pub fn sweep_plan() -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for &m in M_SWEEP {
        for &k in K_SWEEP {
            debug_assert_eq!(k % 128, 0, "K sweep must stay 128-aligned for WhiteCrow");
            out.push((m, N_FIXED, k));
        }
    }
    out
}

/// Decide one cell. `baseline_ms` is the control arm's median at the same shape.
pub fn judge(arm: &ArmResult, baseline_ms: f32) -> Verdict {
    if arm.samples_ms.is_empty() {
        return Verdict::NotRun;
    }
    if !arm.accuracy_ok() {
        return Verdict::DisqualifiedAccuracy;
    }
    // Speed only consulted once accuracy has passed. The bar is shape-dependent:
    // decode arms are held to DECODE_BAR, prefill to SHIP_BAR.
    let _ = baseline_ms;
    Verdict::Eligible
}

/// Does this cell clear the ship bar? Separate from `judge` so the bar is one
/// place and the report can show "eligible but below bar" distinctly from
/// "disqualified".
///
/// # The epsilon
///
/// The comparison is `ratio >= bar`, and a ratio built from two f32 timings does
/// not round-trip: `1.0 / (1.0/1.05)` comes back as 1.0499999... , so a cell
/// that lands exactly on the bar fails. With measured timings that is a
/// guaranteed flake at the boundary, so the bar is relaxed by a relative
/// epsilon. Same reasoning as `perf_gate.rs`'s deliberate `delta_pct <=
/// threshold` — the boundary must not be where the noise lives.
const BAR_EPS: f64 = 1e-6;

pub fn clears_ship_bar(arm: &ArmResult, baseline_ms: f32) -> bool {
    if !arm.accuracy_ok() || arm.samples_ms.is_empty() || baseline_ms <= 0.0 {
        return false;
    }
    let bar = match shape_class(arm.m) {
        ShapeClass::Decode => DECODE_BAR,
        ShapeClass::Prefill => SHIP_BAR,
    };
    let ratio = (baseline_ms as f64) / (arm.median_ms() as f64);
    ratio >= bar * (1.0 - BAR_EPS)
}

/// Serialise results to the artifact JSON. Deliberately a hand-rolled string
/// builder: grim has no serde in this crate's dependency set and a dependency for
/// a test artifact would be a poor trade.
pub fn to_artifact_json(results: &[ArmResult], arch: &str) -> String {
    let mut s = String::from("{\n  \"schema\": \"grim.precision_ab.v1\",\n");
    s.push_str(&format!("  \"arch\": \"{arch}\",\n"));
    s.push_str(&format!("  \"n_fixed\": {N_FIXED},\n"));
    s.push_str("  \"results\": [\n");
    for (i, r) in results.iter().enumerate() {
        s.push_str("    {");
        s.push_str(&format!("\"arm\": \"{}\", ", r.arm));
        s.push_str(&format!("\"instruction\": \"{}\", ", r.instruction));
        s.push_str(&format!("\"m\": {}, \"n\": {}, \"k\": {}, ", r.m, r.n, r.k));
        s.push_str(&format!("\"median_ms\": {:.6}, ", r.median_ms()));
        s.push_str(&format!("\"min_ms\": {:.6}, ", r.min_ms()));
        s.push_str(&format!("\"p90_ms\": {:.6}, ", r.p90_ms()));
        s.push_str(&format!("\"max_abs_err\": {:.6e}, ", r.max_abs_err));
        s.push_str(&format!("\"tolerance\": {:.6e}, ", r.tolerance));
        s.push_str(&format!("\"verdict\": \"{}\"", r.accuracy_ok().then_some("eligible").unwrap_or("disqualified_accuracy")));
        s.push('}');
        if i + 1 < results.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

/// Group results by arm for the console report.
pub fn by_arm(results: &[ArmResult]) -> BTreeMap<&'static str, Vec<&ArmResult>> {
    let mut m: BTreeMap<&'static str, Vec<&ArmResult>> = BTreeMap::new();
    for r in results {
        m.entry(r.arm).or_default().push(r);
    }
    m
}
