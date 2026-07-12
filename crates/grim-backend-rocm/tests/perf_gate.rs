//! RED-GREEN-REFACTOR tests for `perf_gate::PerfGate`.
//!
//! Phase-3 §3.7 of the QKV spec, RED-first. The gate is a regression
//! harness that (a) loads a baseline JSON, (b) compares a measured
//! `cycles_per_call` against the baseline with a per-key threshold,
//! and (c) decides `Within`, `Regressed`, or `NoBaseline`.
//!
//! Skill attribution:
//! - `rocm-profiling-perf` — regression gates, metric discipline
//!   (cycles per call is a coarse metric; later we may fold in rocprof
//!   counters, but the gate as a primitive stays generic).
//! - `rust-gpu-discipline` — no fake-GPU: the gate never fabricates
//!   measurements; `NoBaseline` surfaces absence explicitly.
//! - `rust-ai-ml-inference-guide` Action 8 — TTFT/tok·s evaluation
//!   and the discipline of recording baseline numbers before claiming
//!   optimization.
use grim_backend_rocm::perf_gate::{BaselineTable, Measurement, PerfGate, Verdict};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

// =========================================================================
// RED — Verdict construction & kinds
// =========================================================================

#[test]
fn verdict_within_reports_under_threshold() -> TestResult {
    let v = Verdict::within(0.95, 5.0);
    match v {
        Verdict::Within { delta_pct, threshold_pct } => {
            assert!(delta_pct < threshold_pct, "delta_pct must be < threshold_pct");
            assert_eq!(threshold_pct, 5.0);
        }
        _ => return Err("expected Within".into()),
    }
    Ok(())
}

#[test]
fn verdict_regressed_above_threshold() -> TestResult {
    let v = Verdict::regressed(1.10, 5.0);
    match v {
        Verdict::Regressed { delta_pct, threshold_pct, baseline, current } => {
            assert!(delta_pct > threshold_pct, "regression must clear threshold");
            assert!(baseline > 0.0);
            assert!(current > baseline);
        }
        _ => return Err("expected Regressed".into()),
    }
    Ok(())
}

#[test]
fn verdict_no_baseline_key() -> TestResult {
    let v = Verdict::NoBaseline { reason: String::from("no entry for kernel") };
    match v {
        Verdict::NoBaseline { reason } => assert!(reason.contains("no entry")),
        _ => return Err("expected NoBaseline".into()),
    }
    Ok(())
}

#[test]
fn verdict_is_debug_for_diagnostics() -> TestResult {
    let kinds = [
        Verdict::within(1.0, 5.0),
        Verdict::Regressed { baseline: 100.0, current: 110.0, delta_pct: 10.0, threshold_pct: 5.0 },
        Verdict::NoBaseline { reason: String::from("x") },
    ];
    for v in &kinds {
        let _ = format!("{:?}", v);
    }
    Ok(())
}

// =========================================================================
// RED — Measurement ratio. delta_pct == (current / baseline) * 100 - 100.
// Regression threshold = +5% by default per spec §3.7.
// =========================================================================
#[test]
fn measurement_ratio_is_baseline_normalised_percentage() -> TestResult {
    let m = Measurement { cycles_per_call: 110.0 };
    assert!((m.delta_pct_vs(100.0) - 10.0).abs() < 1e-9);
    let m = Measurement { cycles_per_call: 100.0 };
    assert!(m.delta_pct_vs(100.0).abs() < 1e-9);
    let m = Measurement { cycles_per_call: 90.0 };
    assert!((m.delta_pct_vs(100.0) + 10.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn measurement_ratio_with_zero_baseline_is_undefined() -> TestResult {
    let m = Measurement { cycles_per_call: 1.0 };
    if !m.delta_pct_vs(0.0).is_nan() {
        return Err("delta_pct_vs(0.0) must be NaN, not a finite number".into());
    }
    Ok(())
}

#[test]
fn measurement_ratio_with_zero_current_is_undefined() -> TestResult {
    let m = Measurement { cycles_per_call: 0.0 };
    if !m.delta_pct_vs(1.0).is_nan() {
        return Err("delta_pct_vs(0.0) with zero current must be NaN".into());
    }
    Ok(())
}

// =========================================================================
// RED — BaselineTable: append / look up; missing-key path
// =========================================================================

#[test]
fn baseline_table_append_and_lookup() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("grim_qkv_attention", 5_000_000.0, 5.0)?;
    let entry = t
        .entry("grim_qkv_attention")
        .ok_or("missing entry for grim_qkv_attention")?;
    assert_eq!(entry.baseline_cycles_per_call, 5_000_000.0);
    assert_eq!(entry.threshold_pct, 5.0);
    Ok(())
}

#[test]
fn baseline_table_missing_key_returns_no_baseline() -> TestResult {
    let t = BaselineTable::for_arch("gfx1036");
    let v = t.entry("nope");
    assert!(v.is_none());
    Ok(())
}

#[test]
fn baseline_table_threshold_must_be_positive() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    if t.set_entry("k", 1.0, -1.0).is_ok() {
        return Err("set_entry must reject negative thresholds".into());
    }
    if t.set_entry("k", 1.0, 0.0).is_ok() {
        return Err("set_entry must reject zero thresholds".into());
    }
    Ok(())
}

#[test]
fn baseline_table_baseline_must_be_positive() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    if t.set_entry("k", 0.0, 5.0).is_ok() {
        return Err("set_entry must reject zero baselines".into());
    }
    if t.set_entry("k", -5.0, 5.0).is_ok() {
        return Err("set_entry must reject negative baselines".into());
    }
    Ok(())
}

#[test]
fn baseline_table_round_trip() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("grim_qkv_attention", 5_000_000.0, 5.0)?;
    t.set_entry("grim_matmul_f32", 12_345_678.0, 10.0)?;
    let s = t.to_json_pretty()?;
    let t2 = BaselineTable::from_json(&s)?;
    assert_eq!(t2.arch(), "gfx1036");
    let qkv = t2.entry("grim_qkv_attention").ok_or("missing qkv after roundtrip")?;
    assert_eq!(qkv.baseline_cycles_per_call, 5_000_000.0);
    let mat = t2.entry("grim_matmul_f32").ok_or("missing mat after roundtrip")?;
    assert_eq!(mat.threshold_pct, 10.0);
    Ok(())
}

#[test]
fn baseline_table_corruption_returns_err() -> TestResult {
    let broken = b"{not json";
    let res = BaselineTable::from_json(std::str::from_utf8(broken).map_err(|e| format!("utf8: {}", e))?);
    assert!(res.is_err(), "malformed JSON must surface as Err");
    Ok(())
}

#[test]
fn baseline_table_wrong_schema_returns_err() -> TestResult {
    let bad = r#"{"gpu_arch":"gfx1036","entries":{"k":{"baseline_cycles_per_call":1.0}}}"#;
    let res = BaselineTable::from_json(bad);
    assert!(res.is_err(), "missing `threshold_pct` must surface as Err");
    Ok(())
}

#[test]
fn baseline_table_default_picks_default_arch_when_unspecified() -> TestResult {
    let t = BaselineTable::for_arch("gfxunknown");
    assert_eq!(t.arch(), "gfxunknown");
    Ok(())
}

// =========================================================================
// RED — PerfGate.compare returns Within for a faster measurement, Regressed
// for a slower one, NoBaseline when no entry exists.
// =========================================================================

#[test]
fn perf_gate_within_faster_measurement() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("k", 100.0, 5.0)?;
    let gate = PerfGate::new(t);
    let v = gate.compare("k", Measurement { cycles_per_call: 95.0 });
    match v {
        Verdict::Within { .. } => Ok(()),
        _ => Err("expected Within (faster)".into()),
    }
}

#[test]
fn perf_gate_within_at_threshold_boundary_is_ok() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("k", 100.0, 5.0)?;
    let gate = PerfGate::new(t);
    // 105.0 -> +5% exactly. Boundary defined as: delta_pct <= threshold → Within.
    // (Strict `<` would create flakes for repeated-measurement noise.)
    let v = gate.compare("k", Measurement { cycles_per_call: 105.0 });
    match v {
        Verdict::Within { delta_pct, .. } => {
            assert!((delta_pct - 5.0).abs() < 1e-9);
            Ok(())
        }
        _ => Err("expected Within at +5% boundary".into()),
    }
}

#[test]
fn perf_gate_regressed_above_threshold() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("k", 100.0, 5.0)?;
    let gate = PerfGate::new(t);
    let v = gate.compare("k", Measurement { cycles_per_call: 110.0 });
    match v {
        Verdict::Regressed { delta_pct, baseline, current, .. } => {
            assert!((delta_pct - 10.0).abs() < 1e-9);
            assert_eq!(baseline, 100.0);
            assert!((current - 110.0).abs() < 1e-9);
            Ok(())
        }
        _ => Err("expected Regressed".into()),
    }
}

#[test]
fn perf_gate_unknown_key_returns_no_baseline() -> TestResult {
    let t = BaselineTable::for_arch("gfx1036");
    let gate = PerfGate::new(t);
    let v = gate.compare("not-in-baseline", Measurement { cycles_per_call: 1.0 });
    match v {
        Verdict::NoBaseline { .. } => Ok(()),
        _ => Err("expected NoBaseline".into()),
    }
}

#[test]
fn perf_gate_per_key_threshold_overrides_library_default() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("hot", 1_000.0, 5.0)?;   // hot allows 5%
    t.set_entry("cold", 1_000.0, 50.0)?;  // cold allows 50% (looser)
    let gate = PerfGate::new(t);
    // Both have the same +10% measurement, but `cold` is within budget
    // and `hot` regresses above its smaller threshold.
    assert!(matches!(gate.compare("hot", Measurement { cycles_per_call: 1_100.0 }), Verdict::Regressed { .. }));
    assert!(matches!(gate.compare("cold", Measurement { cycles_per_call: 1_100.0 }), Verdict::Within { .. }));
    Ok(())
}

// =========================================================================
// RED — Default-constructed gate accepts an empty baseline and reports
// NoBaseline for any key (we never panic on an empty benchmark file).
// =========================================================================

#[test]
fn perf_gate_empty_baseline_never_panics() -> TestResult {
    let gate = PerfGate::new(BaselineTable::for_arch("gfx1036"));
    for k in &["k1", "k2", "k3"] {
        let v = gate.compare(k, Measurement { cycles_per_call: 1.0 });
        if !matches!(v, Verdict::NoBaseline { .. }) {
            return Err(format!("empty baseline must yield NoBaseline, got {:?}", v).into());
        }
    }
    Ok(())
}

#[test]
fn perf_gate_collect_verdicts_aggregates() -> TestResult {
    let mut t = BaselineTable::for_arch("gfx1036");
    t.set_entry("ok_a", 100.0, 5.0)?;
    t.set_entry("ok_b", 100.0, 5.0)?;
    t.set_entry("bad", 100.0, 5.0)?;
    // "missing" intentionally not added — must surface as NoBaseline.
    let gate = PerfGate::new(t);
    let verdicts: Vec<_> = [
        gate.compare("ok_a", Measurement { cycles_per_call: 101.0 }),
        gate.compare("ok_b", Measurement { cycles_per_call: 102.0 }),
        gate.compare("bad", Measurement { cycles_per_call: 200.0 }),
        gate.compare("missing", Measurement { cycles_per_call: 50.0 }),
    ]
    .into_iter()
    .collect();
    assert_eq!(verdicts.len(), 4);
    let ok_count = verdicts.iter().filter(|v| matches!(v, Verdict::Within { .. })).count();
    let bad_count = verdicts.iter().filter(|v| matches!(v, Verdict::Regressed { .. })).count();
    let nobas_count = verdicts.iter().filter(|v| matches!(v, Verdict::NoBaseline { .. })).count();
    assert_eq!(ok_count, 2);
    assert_eq!(bad_count, 1);
    assert_eq!(nobas_count, 1);
    Ok(())
}
