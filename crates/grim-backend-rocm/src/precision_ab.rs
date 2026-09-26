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

/// Every other arm is measured relative to this one. Named once so the control
/// cannot drift between the console report and the artifact.
pub const CONTROL_ARM: &str = "Raven";

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
/// The control arm's median for `r`'s shape: Raven on the same (m, n, k).
///
/// Returns `None` when the control was not measured at this shape, which is a
/// real possibility in a partial run. A missing control must not silently
/// become "1.00x" -- that would let an unmeasured arm look like it tied the
/// baseline and clear a bar it never faced.
pub fn control_for(results: &[ArmResult], r: &ArmResult) -> Option<f32> {
    results
        .iter()
        .find(|c| c.arm == CONTROL_ARM && c.m == r.m && c.n == r.n && c.k == r.k)
        .map(|c| c.median_ms())
}

/// `baseline / arm` -- how much faster this arm is than the control. >1 is a win.
pub fn vs_control(results: &[ArmResult], r: &ArmResult) -> Option<f64> {
    let b = control_for(results, r)?;
    if r.median_ms() > 0.0 {
        Some(b as f64 / r.median_ms() as f64)
    } else {
        None
    }
}

/// What the artifact should record for one row: the accuracy verdict *and* the
/// ship decision. Kept separate because a fast arm that failed accuracy is
/// neither eligible nor a candidate, and collapsing the two is how a
/// disqualified arm ends up in a shipping table.
fn ship_label(results: &[ArmResult], r: &ArmResult) -> String {
    if !r.accuracy_ok() {
        return "disqualified_accuracy".into();
    }
    if r.arm == CONTROL_ARM {
        return "control".into();
    }
    match vs_control(results, r) {
        None => "no_control".into(),
        Some(_) if clears_ship_bar(r, control_for(results, r).unwrap_or(f32::NAN)) => {
            "ship".into()
        }
        Some(_) => "below_bar".into(),
    }
}

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
        // null, not 1.0, when this shape had no control measurement.
        match vs_control(results, r) {
            Some(v) => s.push_str(&format!("\"vs_control\": {v:.4}, ")),
            None => s.push_str("\"vs_control\": null, "),
        }
        s.push_str(&format!("\"verdict\": \"{}\", ", r.accuracy_ok().then_some("eligible").unwrap_or("disqualified_accuracy")));
        s.push_str(&format!("\"ship\": \"{}\"", ship_label(results, r)));
        s.push('}');
        if i + 1 < results.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ],\n");

    // Per-arm rollup: the shape-by-shape table answers "where", this answers
    // "does it ship" without re-deriving it from 40 rows.
    s.push_str("  \"summary\": [\n");
    let arms: Vec<&str> = by_arm(results).keys().copied().collect();
    for (i, arm) in arms.iter().enumerate() {
        let rows: Vec<&ArmResult> = results.iter().filter(|r| r.arm == *arm).collect();
        let ships = rows.iter().filter(|r| ship_label(results, r) == "ship").count();
        let disq = rows.iter().filter(|r| !r.accuracy_ok()).count();

        // Build the row as a field list joined by ", " rather than by appending
        // separators by hand. Hand-appending is what produced a stray quote
        // after a numeric field and a missing comma on the null branch -- both
        // of which produced un-parseable JSON that the tests still passed,
        // because they asserted on substrings rather than well-formedness.
        let mut f: Vec<String> = vec![
            format!("\"arm\": \"{arm}\""),
            format!("\"instruction\": \"{}\"", rows[0].instruction),
            format!("\"shapes\": {}", rows.len()),
            format!("\"disqualified\": {disq}"),
            format!("\"ships\": {ships}"),
        ];
        let ratios: Vec<f64> = rows.iter().filter_map(|r| vs_control(results, r)).collect();
        if let (Some(lo), Some(hi)) = (
            ratios.iter().cloned().fold(None, |a: Option<f64>, b| Some(a.map_or(b, |x| x.min(b)))),
            ratios.iter().cloned().fold(None, |a: Option<f64>, b| Some(a.map_or(b, |x| x.max(b)))),
        ) {
            f.push(format!("\"vs_control_min\": {lo:.4}"));
            f.push(format!("\"vs_control_max\": {hi:.4}"));
        } else {
            f.push("\"vs_control_min\": null".into());
            f.push("\"vs_control_max\": null".into());
        }
        s.push_str("    {");
        s.push_str(&f.join(", "));
        s.push('}');
        if i + 1 < arms.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

/// Structural well-formedness check for a generated artifact.
///
/// Hand-rolled JSON in a crate with no serde needs *some* guard, because a
/// malformed artifact is silently useless: it still writes, still prints a
/// success line, and fails only when something downstream tries to parse it.
/// Not a full parser -- it checks the things this emitter can actually get
/// wrong: balanced brackets, no trailing comma, no digit immediately followed
/// by a closing quote, and no field missing its comma.
pub fn json_is_well_formed(s: &str) -> bool {
    let b: Vec<char> = s.chars().collect();
    let (mut ds, mut db) = (0i32, 0i32);
    let mut in_str = false;
    let mut esc = false;
    let mut after_colon = false;

    let next_sig = |from: usize| -> Option<char> { b[from..].iter().find(|c| !c.is_whitespace()).copied() };
    let prev_sig = |i: usize| -> Option<char> { b[..i].iter().rev().find(|c| !c.is_whitespace()).copied() };

    for (i, &c) in b.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
                // A key must be followed by ':'; a value by ',', '}' or ']'.
                // This is what catches a dropped comma between fields.
                match next_sig(i + 1) {
                    Some(':') if !after_colon => {}
                    Some(x) if after_colon && matches!(x, ',' | '}' | ']') => {}
                    _ => return false,
                }
                after_colon = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            ':' => after_colon = true,
            ',' => after_colon = false,
            '[' => {
                ds += 1;
                after_colon = false;
            }
            '{' => {
                db += 1;
                after_colon = false;
            }
            ']' | '}' => {
                if c == ']' {
                    ds -= 1;
                    if ds < 0 {
                        return false;
                    }
                } else {
                    db -= 1;
                    if db < 0 {
                        return false;
                    }
                }
                // Trailing comma: nothing was pushed into the container.
                if prev_sig(i) == Some(',') {
                    return false;
                }
                // Junk between a close and the next token.
                match next_sig(i + 1) {
                    None => {}
                    Some(x) if matches!(x, ',' | '}' | ']') => {}
                    _ => return false,
                }
                after_colon = false;
            }
            _ => {}
        }
    }
    !in_str && ds == 0 && db == 0
}

/// Group results by arm for the console report.
pub fn by_arm(results: &[ArmResult]) -> BTreeMap<&'static str, Vec<&ArmResult>> {
    let mut m: BTreeMap<&'static str, Vec<&ArmResult>> = BTreeMap::new();
    for r in results {
        m.entry(r.arm).or_default().push(r);
    }
    m
}
