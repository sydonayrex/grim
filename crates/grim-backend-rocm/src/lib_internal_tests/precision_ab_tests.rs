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
    // Exactly one arm per result row. Scoped to the results array: the summary
    // block repeats the arm names by design, so a global count would flag the
    // rollup as a duplicate.
    let results_section = j.split("\"summary\"").next().unwrap();
    assert_eq!(results_section.matches("\"arm\":").count(), results.len());
    assert!(!j.contains(",\n  ]"));
    assert!(!j.contains(",\n  }\n  ]"), "trailing comma in summary: {j}");
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

#[test]
fn artifact_records_the_ship_decision_not_just_eligibility() {
    // Raven 1.0ms control; the challenger runs 20% faster, so it clears both
    // the 1.05x decode bar and the 1.10x prefill bar.
    let results = vec![
        r("Raven", 1, 1024, &[1.0], 1e-4, TOL_F32),
        r("ForestRaven", 1, 1024, &[0.8], 1e-4, TOL_I32),
    ];
    let j = to_artifact_json(&results, "gfx1201");
    assert!(j.contains("\"vs_control\": 1.2500"), "ratio not emitted: {j}");
    assert!(j.contains("\"ship\": \"control\""), "control mislabelled: {j}");
    assert!(
        j.contains("\"ship\": \"ship\""),
        "a 1.25x eligible arm must record ship: {j}"
    );
    assert!(j.contains("\"summary\": ["));
    assert!(j.contains("\"ships\": 1"));
}

#[test]
fn an_unmeasured_control_yields_null_not_a_fake_tie() {
    // A challenger with no Raven at its shape. Reporting 1.0x here would let it
    // clear a bar it never raced, so the ratio must be null and the label
    // explicit.
    let results = vec![r("ForestRaven", 1, 1024, &[0.5], 1e-4, TOL_I32)];
    let j = to_artifact_json(&results, "gfx1201");
    assert!(
        j.contains("\"vs_control\": null"),
        "absent control must not read as 1.0x: {j}"
    );
    assert!(j.contains("\"ship\": \"no_control\""), "{j}");
    assert!(j.contains("\"vs_control_min\": null"), "{j}");
}

#[test]
fn a_disqualified_arm_never_records_ship_however_fast_it_is() {
    // 10x faster and 1000x wrong. Speed is not a defence.
    let results = vec![
        r("Raven", 1, 1024, &[1.0], 1e-4, TOL_F32),
        r("WildRaven", 1, 1024, &[0.1], 1e3, TOL_F32),
    ];
    let j = to_artifact_json(&results, "gfx1201");
    assert!(j.contains("\"vs_control\": 10.0000"), "{j}");
    assert!(j.contains("\"verdict\": \"disqualified_accuracy\""));
    assert!(
        !j.contains("\"ship\": \"ship\""),
        "an inaccurate arm must not be marked ship: {j}"
    );
    assert!(j.contains("\"disqualified\": 1"));
    assert!(j.contains("\"ships\": 0"));
}

#[test]
fn artifact_json_is_well_formed_on_every_branch() {
    // The emitter is hand-rolled because the crate has no serde, so every branch
    // that can produce a different field set gets checked. A malformed artifact
    // is the worst failure mode here: it writes, logs success, and only breaks
    // when something downstream parses it.
    let cases: Vec<Vec<ArmResult>> = vec![
        // no arms at all
        vec![],
        // control only: no ratios to report
        vec![r("Raven", 1, 1024, &[1.0], 1e-4, TOL_F32)],
        // challenger with no control -> null branch of the ratio pair
        vec![r("ForestRaven", 1, 1024, &[0.5], 1e-4, TOL_I32)],
        // control + challenger -> numeric branch
        vec![
            r("Raven", 1, 1024, &[1.0], 1e-4, TOL_F32),
            r("ForestRaven", 1, 1024, &[0.8], 1e-4, TOL_I32),
        ],
        // several arms, mixed eligibility, multiple shapes
        vec![
            r("Raven", 1, 1024, &[1.0, 1.1], 1e-4, TOL_F32),
            r("Raven", 8, 4096, &[4.0], 1e-4, TOL_F32),
            r("ForestRaven", 1, 1024, &[0.8], 1e-4, TOL_I32),
            r("ForestRaven", 8, 4096, &[9.9], 5e3, TOL_I32),
            r("WhiteRaven", 8, 4096, &[2.0], 1e-4, TOL_F32),
        ],
    ];
    for (i, c) in cases.iter().enumerate() {
        let j = to_artifact_json(c, "gfx1201");
        assert!(
            json_is_well_formed(&j),
            "case {i} produced malformed JSON:\n{j}"
        );
        // Sanity: the guard must actually reject broken input, or it is decoration.
        assert!(!json_is_well_formed("{\"a\": 1\"}"), "guard missed stray quote");
        assert!(!json_is_well_formed("{\"a\": 1,}"), "guard missed trailing comma");
        assert!(!json_is_well_formed("{\"a\": 1"), "guard missed unclosed brace");
        assert!(!json_is_well_formed("[1, 2"), "guard missed unclosed bracket");
        assert!(!json_is_well_formed("{\"a\": 1}x"), "guard missed junk after close");
    }
}

#[test]
fn a_numeric_field_never_ends_in_a_quote() {
    // Regression for the exact defect: format!("...: {v:.4}\"") on a numeric
    // value emits `1.0"}`, which still reads fine in a substring assertion.
    let results = vec![
        r("Raven", 1, 1024, &[1.0], 1e-4, TOL_F32),
        r("ForestRaven", 1, 1024, &[0.8], 1e-4, TOL_I32),
    ];
    let j = to_artifact_json(&results, "gfx1201");
    assert!(!j.contains("0.8000\""), "numeric field closed as a string: {j}");
    assert!(j.contains("\"vs_control_max\": 1.2500"), "{j}");
}

#[test]
fn l2_eviction_width_puts_b_past_the_cache() {
    // The whole point: B must not fit in L2, or the arm that re-reads B the
    // fewest times wins on cache and the ratio measures nothing real.
    const L2: usize = 96 * 1024 * 1024;
    for &k in K_SWEEP {
        let n = n_for_l2_eviction(k);
        let b_1byte = n * k;
        assert!(
            b_1byte > L2 * 2,
            "K={k}: N={n} gives only {} MB of B, need > {} MB",
            b_1byte / 1_000_000,
            L2 * 2 / 1_000_000
        );
        // int4 is half the size, so check the denser packing too.
        assert!(
            n * k / 2 > L2,
            "K={k}: int4 B is only {} MB, still cacheable",
            n * k / 2 / 1_000_000
        );
        assert_eq!(n % 4, 0, "N={n} must tile evenly into 4-column groups");
    }
}

#[test]
fn l2_eviction_width_grows_as_k_shrinks() {
    // N is inversely proportional to K: the same B footprint at every K.
    let wide = n_for_l2_eviction(1024);
    let narrow = n_for_l2_eviction(22016);
    assert!(wide > narrow, "smaller K must use a wider N");
    // ...and lands in a comparable footprint.
    for &k in K_SWEEP {
        let n = n_for_l2_eviction(k);
        let mb = n * k / 1_000_000;
        assert!(
            (250..=600).contains(&mb),
            "K={k}: N={n} gives {mb} MB, outside the intended 250-600 MB band"
        );
    }
}
