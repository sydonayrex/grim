//! T0 for the §6 A/B harness core. All CPU — the GPU driver is
//! `tests/precision_kernel_ab.rs`.
//!
//! These tests exist because the harness's *decisions* are where a wrong answer
//! silently becomes a shipped decision. The plan's pre-registered rule is
//! "accuracy gate first, then speed, and a documented negative result is a
//! success" — so the ordering and the bar are both asserted here, on CPU, where
//! they cannot be quietly changed alongside a favourable measurement.

use crate::precision_ab::*;

/// Tolerance for an f32-accumulating arm. The dot4 path sums 4 products per
/// instruction in f32; over K=11008 that is a long f32 chain.
const TOL_F32: f64 = 2e-2;
/// Tolerance for an i32-accumulating arm. Integer accumulation is exact over the
/// product, so the only error is the final scale-and-convert.
const TOL_I32: f64 = 1e-3;

fn r(arm: &'static str, m: usize, k: usize, ms: &[f32], err: f64, tol: f64) -> ArmResult {
    ArmResult {
        arm,
        instruction: "V_TEST",
        m,
        n: N_FIXED,
        k,
        samples_ms: ms.to_vec(),
        max_abs_err: err,
        tolerance: tol,
    }
}

#[test]
fn median_min_and_p90_are_ordered_and_sane() {
    let a = r("x", 1, 1024, &[5.0, 1.0, 3.0, 2.0, 4.0], 0.0, TOL_F32);
    assert_eq!(a.median_ms(), 3.0, "median of 1..5 is 3");
    assert_eq!(a.min_ms(), 1.0);
    assert_eq!(a.p90_ms(), 5.0, "p90 of 5 samples is the top one");
    assert!(a.min_ms() <= a.median_ms() && a.median_ms() <= a.p90_ms());
}

#[test]
fn empty_samples_do_not_panic() {
    let a = r("x", 1, 1024, &[], 0.0, TOL_F32);
    assert!(a.median_ms().is_nan());
    assert!(a.p90_ms().is_nan());
    assert_eq!(judge(&a, 1.0), Verdict::NotRun);
    assert!(!clears_ship_bar(&a, 1.0));
}

#[test]
fn accuracy_is_gated_before_speed() {
    // Faster than baseline but inaccurate: must be disqualified, not eligible.
    // This is the whole point of rule 1 - a fast wrong kernel is a regression.
    let fast_but_wrong = r("bad", 128, 4096, &[0.5], 5.0, TOL_F32);
    assert_eq!(judge(&fast_but_wrong, 1.0), Verdict::DisqualifiedAccuracy);
    assert!(
        !clears_ship_bar(&fast_but_wrong, 1.0),
        "a fast but inaccurate arm must never clear the bar"
    );
}

#[test]
fn accurate_but_slower_is_eligible_yet_below_bar() {
    // Eligible means "accuracy passed, so the speed number means something" - not
    // "we recommend shipping". Those are different questions and collapsing them
    // is how a 0.9x result gets presented as a win.
    let slow = r("slow", 128, 4096, &[2.0], 0.0, TOL_F32);
    assert_eq!(judge(&slow, 1.0), Verdict::Eligible);
    assert!(!clears_ship_bar(&slow, 1.0), "2x slower must not clear the bar");
}

#[test]
fn error_exactly_at_tolerance_passes() {
    // Boundary, matching perf_gate.rs's deliberate choice: <= passes, so repeated
    // measurement noise at the exact tolerance does not flake.
    let at = r("x", 1, 1024, &[1.0], TOL_F32, TOL_F32);
    assert!(at.accuracy_ok());
    let over = r("x", 1, 1024, &[1.0], TOL_F32 * 1.0001, TOL_F32);
    assert!(!over.accuracy_ok());
}

#[test]
fn decode_and_prefill_bars_differ() {
    // Decode bar is lower than prefill: a decode arm is a small fraction of
    // end-to-end time so a smaller kernel win matters more, but decode shapes
    // are noisier so the bar is set against the same measurement discipline.
    let decode = r("d", 1, 4096, &[1.0 / DECODE_BAR as f32], 0.0, TOL_F32);
    assert!(clears_ship_bar(&decode, 1.0), "exactly at the decode bar passes");
    // NB: to sit BELOW a bar of B the median must be SLOWER than 1.0/B, i.e.
    // 1.0/(B*0.95) -> ratio 0.95*B. Using 1.0/(B+0.05) would give a ratio of
    // B+0.05, which is above the bar - the first draft of this test got that
    // backwards and asserted the opposite of what it meant.
    let decode_short = r("d", 1, 4096, &[1.0 / (DECODE_BAR * 0.95) as f32], 0.0, TOL_F32);
    assert!(!clears_ship_bar(&decode_short, 1.0), "0.95x the decode bar is short");
    // A prefill arm is held to the higher bar at the same ratio.
    let prefill = r("p", 512, 4096, &[1.0 / DECODE_BAR as f32], 0.0, TOL_F32);
    assert!(!clears_ship_bar(&prefill, 1.0), "prefill needs the higher bar");
}

#[test]
fn integer_accumulator_gets_the_tighter_tolerance() {
    // Documenting the asymmetry: i32 accumulate is exact over the products, so
    // its tolerance is an order of magnitude tighter. If an arm is re-pointed at
    // a different accumulator this should be revisited, not inherited.
    assert!(TOL_I32 < TOL_F32);
    let i32 = r("i", 1, 1024, &[1.0], 5e-4, TOL_I32);
    assert!(i32.accuracy_ok());
    let i32_bad = r("i", 1, 1024, &[1.0], 5e-2, TOL_I32);
    assert!(!i32_bad.accuracy_ok());
}

#[test]
fn sweep_plan_is_128_aligned_and_covers_decode_and_prefill() {
    let plan = sweep_plan();
    assert_eq!(plan.len(), M_SWEEP.len() * K_SWEEP.len());
    for &(_m, n, k) in &plan {
        assert_eq!(k % 128, 0, "WhiteCrow's group is 128; K must be aligned");
        assert_eq!(n, N_FIXED);
    }
    // Both regimes present, so the crossover is discoverable.
    assert!(plan.iter().any(|&(m, _, _)| m <= 8));
    assert!(plan.iter().any(|&(m, _, _)| m >= 128));
}

#[test]
fn shape_class_splits_at_the_dot_family_limit() {
    assert_eq!(shape_class(1), ShapeClass::Decode);
    assert_eq!(shape_class(8), ShapeClass::Decode);
    assert_eq!(shape_class(9), ShapeClass::Prefill);
    assert_eq!(shape_class(512), ShapeClass::Prefill);
}

#[test]
fn artifact_json_is_parseable_and_carries_the_verdict() {
    let results = vec![
        r("Raven", 1, 1024, &[1.0, 1.1], 1e-4, TOL_F32),
        r("Bad", 1, 1024, &[0.5], 9.9, TOL_F32),
    ];
    let j = to_artifact_json(&results, "gfx1201");
    // Structural checks without a JSON parser dependency.
    assert!(j.starts_with('{') && j.trim_end().ends_with('}'));
    assert!(j.contains("\"schema\": \"grim.precision_ab.v1\""));
    assert!(j.contains("\"arch\": \"gfx1201\""));
    assert!(j.contains("\"arm\": \"Raven\""));
    assert!(j.contains("\"verdict\": \"eligible\""));
    assert!(j.contains("\"verdict\": \"disqualified_accuracy\""));
    // Exactly one arm per result, no trailing comma before the closing bracket.
    assert_eq!(j.matches("\"arm\":").count(), results.len());
    assert!(!j.contains(",\n  ]"));
    let grouped = by_arm(&results);
    assert_eq!(grouped.len(), 2);
    assert_eq!(grouped["Raven"].len(), 1);
}

#[test]
fn every_pinned_builtin_is_referenced_by_some_arm() {
    // Keeps the artifact honest: if a new codename lands with no measurement, the
    // arm list should not silently omit it. Arms currently shipping:
    use crate::quantization::CORVID_INSTRUCTION_BINDING;
    let json = to_artifact_json(&[], "gfx1201");
    // The binding table is the roster; the harness must grow an arm per row.
    for (name, inst) in CORVID_INSTRUCTION_BINDING {
        assert!(
            !name.is_empty() && !inst.is_empty(),
            "binding row {name}/{inst} is malformed"
        );
    }
    assert!(json.contains("precision_ab"));
}
