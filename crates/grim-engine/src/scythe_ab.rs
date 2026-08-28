//! WI-SB3: TTFT/ITL A/B harness support.
//!
//! Owns the results-file protocol (§setup-4 of `scythe2_syd_beasty_plan.md`):
//! every hardware sample appends one JSON line
//! `{wi, order, metric, value, commit, ts}` to
//! `docs/benchmarks/scythe2_syd_beasty_results.jsonl`, and the WI-INF4
//! verdict rule (mean TTFT overhead ≤ 5 %, p95 ITL overhead ≤ 2 %) is
//! computed here so the example driver stays a thin loop and the math is
//! unit-testable off-box. The gate is evaluated PER ordinal order
//! (`format_ab_report` prints one line per order) — the pooled verdict is
//! reported alongside, but a per-order FAIL must not hide under pooling.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Default sink for the §setup-4 results protocol.
pub fn default_results_path() -> PathBuf {
    PathBuf::from("docs/benchmarks/scythe2_syd_beasty_results.jsonl")
}

/// One measured request from an A/B leg. Latencies are engine-reported
/// (`last_ttft_ms` / `last_itl_ms` / `tokens_per_sec_ema`); `throttle_pct`
/// rides along with every sample per the plan's thermal-drift risk note.
#[derive(Debug, Clone, PartialEq)]
pub struct ScytheAbSample {
    pub arm_on: bool,
    /// "F-first" | "S-first" | "unknown".
    pub order: String,
    pub prompt_tokens: usize,
    pub elapsed_ms: f64,
    pub ttft_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub tokens_per_sec_ema: Option<f32>,
    pub throttle_pct: f32,
}

impl ScytheAbSample {
    /// The §setup-4 JSON lines this sample contributes — one per metric.
    pub fn to_json_lines(&self, commit: &str, ts: u64) -> Vec<serde_json::Value> {
        let arm = if self.arm_on { "on" } else { "off" };
        let mut lines = Vec::new();
        for (metric, value) in [
            ("ttft_ms", self.ttft_ms),
            ("itl_ms", self.itl_ms),
            (
                "tokens_per_sec_ema",
                self.tokens_per_sec_ema.map(|v| v as f64),
            ),
            ("elapsed_ms", Some(self.elapsed_ms)),
        ] {
            let Some(value) = value else { continue };
            lines.push(serde_json::json!({
                "wi": "SB3",
                "order": self.order,
                "metric": metric,
                "value": value,
                "commit": commit,
                "ts": ts,
                "arm": arm,
                "prompt_tokens": self.prompt_tokens,
                "throttle_pct": self.throttle_pct,
            }));
        }
        lines
    }
}

/// Append one leg's samples to the results file as §setup-4 JSON lines,
/// creating parent directories on first write. Returns the number of lines.
pub fn append_samples(
    path: &Path,
    samples: &[ScytheAbSample],
    commit: &str,
    ts: u64,
) -> std::io::Result<usize> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut lines = 0;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for sample in samples {
        for line in sample.to_json_lines(commit, ts) {
            writeln!(out, "{line}")?;
            lines += 1;
        }
    }
    out.flush()?;
    Ok(lines)
}

/// A parsed `{metric, value}` observation from a stored JSONL line.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredMetric {
    pub arm_on: bool,
    pub order: String,
    pub metric: String,
    pub value: f64,
    pub prompt_tokens: usize,
    /// Unix-seconds stamp from the §setup-4 line; 0 for legacy rows without
    /// one. Verdict computations filter on this — see `parse_samples_since`
    /// and the WI-INF4 measurement-defect note (a cumulative report that
    /// mixes rows from different campaigns mixes fault-era data into the
    /// verdict; the ts-filtered computation is authoritative).
    pub ts: u64,
}

/// Parse previously-appended JSONL content, skipping malformed lines so a
/// torn final write never poisons a verdict run.
pub fn parse_samples(jsonl: &str) -> Vec<StoredMetric> {
    parse_samples_since(jsonl, 0)
}

/// Like [`parse_samples`], but keeps only rows stamped `>= since_ts` (unix
/// seconds). `since_ts == 0` keeps everything (same as [`parse_samples`]).
/// Rows without a `ts` field (legacy, stamped 0) never survive a nonzero
/// cutoff so a filtered verdict can never silently include unstaleable data.
pub fn parse_samples_since(jsonl: &str, since_ts: u64) -> Vec<StoredMetric> {
    parse_samples_impl(jsonl, Some(since_ts))
}

fn parse_samples_impl(jsonl: &str, since_ts: Option<u64>) -> Vec<StoredMetric> {
    jsonl
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|v| {
            let metric = v.get("metric")?.as_str()?.to_string();
            let value = v.get("value")?.as_f64()?;
            let ts = v.get("ts").and_then(|t| t.as_u64()).unwrap_or(0);
            if let Some(cutoff) = since_ts {
                // cutoff == 0 disables filtering entirely (the unfiltered
                // `parse_samples` view); a nonzero cutoff also drops
                // unstamped (ts == 0) rows.
                if cutoff > 0 && (ts == 0 || ts < cutoff) {
                    return None;
                }
            }
            Some(StoredMetric {
                arm_on: v.get("arm").and_then(|a| a.as_str()) == Some("on"),
                order: v
                    .get("order")
                    .and_then(|o| o.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                metric,
                value,
                prompt_tokens: v.get("prompt_tokens").and_then(|p| p.as_u64()).unwrap_or(0)
                    as usize,
                ts,
            })
        })
        .collect()
}

fn mean(values: &[f64]) -> Option<f64> {
    match values.len() {
        0 => None,
        n => Some(values.iter().sum::<f64>() / n as f64),
    }
}

/// Nearest-rank percentile of an unsorted slice (does not reorder input).
fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * sorted.len() as f64).ceil();
    let idx = (rank.max(1.0) as usize - 1).min(sorted.len() - 1);
    sorted.get(idx).copied()
}

/// The WI-INF4 verdict rule applied to pooled samples from both legs:
/// mean-TTFT overhead ≤ 5 % AND p95-ITL overhead ≤ 2 %. `None` when either
/// arm has no data — a verdict is never invented from half an experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScytheAbVerdict {
    pub ttft_overhead_pct: f64,
    pub itl_p95_overhead_pct: f64,
    pub eligible: bool,
}

pub fn scythe_ab_verdict(on: &[StoredMetric], off: &[StoredMetric]) -> Option<ScytheAbVerdict> {
    let ttfts = |samples: &[StoredMetric]| -> Vec<f64> {
        samples
            .iter()
            .filter(|s| s.metric == "ttft_ms")
            .map(|s| s.value)
            .collect()
    };
    let itls = |samples: &[StoredMetric]| -> Vec<f64> {
        samples
            .iter()
            .filter(|s| s.metric == "itl_ms")
            .map(|s| s.value)
            .collect()
    };
    let mean_on_ttft = mean(&ttfts(on))?;
    let mean_off_ttft = mean(&ttfts(off))?;
    let p95_on_itl = percentile(&itls(on), 95.0)?;
    let p95_off_itl = percentile(&itls(off), 95.0)?;

    const TTFT_BUDGET_PCT: f64 = 5.0;
    const ITL_BUDGET_PCT: f64 = 2.0;
    let ttft_overhead_pct =
        (mean_on_ttft / mean_off_ttft.max(f64::MIN_POSITIVE) - 1.0).abs() * 100.0;
    let itl_p95_overhead_pct =
        (p95_on_itl / p95_off_itl.max(f64::MIN_POSITIVE) - 1.0).abs() * 100.0;
    Some(ScytheAbVerdict {
        ttft_overhead_pct,
        itl_p95_overhead_pct,
        eligible: ttft_overhead_pct <= TTFT_BUDGET_PCT && itl_p95_overhead_pct <= ITL_BUDGET_PCT,
    })
}

/// Human-readable A/B table + verdict line for the harness printout.
pub fn format_ab_report(on: &[StoredMetric], off: &[StoredMetric]) -> String {
    use std::fmt::Write as _;
    let mut report = String::from("\n=== SCYTHE-2 A/B summary (WI-SB3) ===\n");
    let describe = |label: &str, samples: &[StoredMetric]| -> String {
        let ttfts: Vec<f64> = samples
            .iter()
            .filter(|s| s.metric == "ttft_ms")
            .map(|s| s.value)
            .collect();
        let itls: Vec<f64> = samples
            .iter()
            .filter(|s| s.metric == "itl_ms")
            .map(|s| s.value)
            .collect();
        match (mean(&ttfts), percentile(&itls, 95.0)) {
            (Some(ttft), Some(itl)) => format!(
                "{label:>4}: n={:<3} mean_ttft={ttft:8.2} ms  p95_itl={itl:8.2} ms",
                samples.iter().filter(|s| s.metric == "ttft_ms").count(),
            ),
            _ => format!("{label:>4}: n=0   (no samples)"),
        }
    };
    writeln!(&mut report, "{}", describe("off", off)).ok();
    writeln!(&mut report, "{}", describe("on", on)).ok();

    // Per-prompt-bucket mean TTFT so a regression concentrated in the 8k
    // bucket can't hide under small-prompt averages.
    for bucket in ["<1k tokens", "1k-4k tokens", ">4k tokens"] {
        let in_bucket = |s: &StoredMetric, metric: &str| {
            s.metric == metric
                && match bucket {
                    "<1k tokens" => s.prompt_tokens < 1_000,
                    "1k-4k tokens" => (1_000..4_000).contains(&s.prompt_tokens),
                    _ => s.prompt_tokens >= 4_000,
                }
        };
        let pick = |samples: &[StoredMetric]| -> Vec<f64> {
            samples
                .iter()
                .filter(|s| in_bucket(s, "ttft_ms"))
                .map(|s| s.value)
                .collect()
        };
        let (Some(o), Some(n)) = (mean(&pick(off)), mean(&pick(on))) else {
            continue;
        };
        writeln!(
            &mut report,
            "{bucket:>12}: mean ttft off={o:8.2} on={n:8.2} ({:+.1}%)",
            (n / o.max(f64::MIN_POSITIVE) - 1.0) * 100.0
        )
        .ok();
    }

    // Per-order verdict lines (the WI-INF4 gate is evaluated per ordinal
    // order — the 2026-08-23c campaign failed S-first while F-first passed;
    // a pooled-only verdict hides exactly that asymmetry).
    let mut orders: Vec<&str> = on
        .iter()
        .chain(off.iter())
        .map(|s| s.order.as_str())
        .filter(|o| *o != "unknown")
        .collect();
    orders.sort_unstable();
    orders.dedup();
    for order in orders {
        let on_order: Vec<_> = on.iter().filter(|s| s.order == order).cloned().collect();
        let off_order: Vec<_> = off.iter().filter(|s| s.order == order).cloned().collect();
        match scythe_ab_verdict(&on_order, &off_order) {
            Some(v) => writeln!(
                &mut report,
                "  [{order}] TTFT Δ={:.2}% (≤5%), ITL p95 Δ={:.2}% (≤2%) ⇒ {}",
                v.ttft_overhead_pct,
                v.itl_p95_overhead_pct,
                if v.eligible { "PASS" } else { "FAIL" }
            )
            .ok(),
            None => writeln!(
                &mut report,
                "  [{order}] INCOMPLETE — both arms need samples for this order"
            )
            .ok(),
        };
    }

    match scythe_ab_verdict(on, off) {
        Some(v) => writeln!(
            &mut report,
            "WI-INF4 verdict: TTFT Δ={:.2}% (budget 5%), ITL p95 Δ={:.2}% (budget 2%) ⇒ {}",
            v.ttft_overhead_pct,
            v.itl_p95_overhead_pct,
            if v.eligible {
                "ELIGIBLE to flip GRIM_SCYTHE_INFERENCE default"
            } else {
                "NOT eligible — stays opt-in; retune cost model with these numbers"
            }
        )
        .ok(),
        None => writeln!(
            &mut report,
            "WI-INF4 verdict: INCOMPLETE — both arms need samples in the results file"
        )
        .ok(),
    };
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(arm_on: bool, prompt: usize, ttft: f64, itl: f64) -> ScytheAbSample {
        ScytheAbSample {
            arm_on,
            order: "F-first".into(),
            prompt_tokens: prompt,
            elapsed_ms: ttft + itl,
            ttft_ms: Some(ttft),
            itl_ms: Some(itl),
            tokens_per_sec_ema: Some(42.0),
            throttle_pct: 0.0,
        }
    }

    #[test]
    fn test_sample_serializes_setup4_schema() {
        let lines = sample(true, 2048, 120.0, 9.5).to_json_lines("abc1234", 1_700_000);
        assert_eq!(lines.len(), 4, "one line per non-None metric");
        let v = &lines[0];
        assert_eq!(v["wi"], "SB3");
        assert_eq!(v["order"], "F-first");
        assert_eq!(v["metric"], "ttft_ms");
        assert_eq!(v["value"], 120.0);
        assert_eq!(v["commit"], "abc1234");
        assert_eq!(v["ts"], 1_700_000);
        assert_eq!(v["arm"], "on");
        assert_eq!(v["prompt_tokens"], 2048);
        assert!(
            v.get("throttle_pct").is_some(),
            "throttle rides every sample"
        );

        // Roundtrip through the reader.
        let parsed = parse_samples(
            &lines
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        assert_eq!(parsed.len(), 4);
        assert!(parsed.iter().all(|m| m.arm_on && m.order == "F-first"));
    }

    #[test]
    fn test_append_and_parse_roundtrip() {
        let dir = std::env::temp_dir().join(format!("grim_ab_test_{}", std::process::id()));
        let path = dir.join("nested/results.jsonl");
        let lines = append_samples(
            &path,
            &[
                sample(false, 200, 100.0, 8.0),
                sample(true, 200, 103.0, 8.1),
            ],
            "deadbee",
            42,
        )
        .unwrap();
        assert_eq!(lines, 8, "2 samples × 4 metrics");
        let parsed = parse_samples(&std::fs::read_to_string(&path).unwrap());
        assert_eq!(parsed.len(), 8);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_percentile_nearest_rank() {
        let vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&vals, 95.0), Some(95.0));
        assert_eq!(percentile(&vals, 50.0), Some(50.0));
        assert_eq!(percentile(&[30.0, 10.0, 20.0], 95.0), Some(30.0));
        assert_eq!(percentile(&[], 95.0), None);
    }

    #[test]
    fn test_verdict_rule_bounds() {
        let mk = |arm: bool, ttfts: &[f64], itls: &[f64]| -> Vec<StoredMetric> {
            let mut out = Vec::new();
            for (i, (t, l)) in ttfts.iter().zip(itls).enumerate() {
                out.push(StoredMetric {
                    arm_on: arm,
                    order: "F-first".into(),
                    metric: "ttft_ms".into(),
                    value: *t,
                    prompt_tokens: 200 + i,
                    ts: 0,
                });
                out.push(StoredMetric {
                    arm_on: arm,
                    order: "F-first".into(),
                    metric: "itl_ms".into(),
                    value: *l,
                    prompt_tokens: 200 + i,
                    ts: 0,
                });
            }
            out
        };
        let base = mk(false, &[100.0, 100.0, 100.0, 100.0], &[10.0; 4]);

        // Comfortably inside both budgets ⇒ eligible (p95 of 4 samples is
        // their max; the exact-boundary case is floating-point knife-edge
        // territory and deliberately not asserted).
        let at_budget = mk(
            true,
            &[104.0, 105.0, 104.0, 105.0],
            &[10.0, 10.05, 10.1, 10.19],
        );
        let v = scythe_ab_verdict(&at_budget, &base).unwrap();
        assert!(v.ttft_overhead_pct <= 5.0 && v.itl_p95_overhead_pct <= 2.0 && v.eligible);

        // TTFT blows the budget ⇒ not eligible even with perfect ITL.
        let slow_prefill = mk(true, &[110.0; 4], &[10.0; 4]);
        let v = scythe_ab_verdict(&slow_prefill, &base).unwrap();
        assert!(!v.eligible);
        assert!(v.ttft_overhead_pct > 5.0 && v.itl_p95_overhead_pct <= 2.0);

        // p95 ITL tail blows its budget ⇒ not eligible despite fine mean.
        let tail_latency = mk(true, &[101.0; 4], &[10.0, 10.0, 10.0, 11.0]);
        let v = scythe_ab_verdict(&tail_latency, &base).unwrap();
        assert!(!v.eligible);
        assert!(v.itl_p95_overhead_pct > 2.0);

        // Half an experiment ⇒ no verdict.
        assert!(scythe_ab_verdict(&at_budget, &[]).is_none());
    }

    /// WI-INF4 measurement-defect fix: `parse_samples` once returned every
    /// row in the file, so a cumulative verdict mixed stale fault-era rows
    /// into the computation (the plan's validation log documents the
    /// 28.5 %-vs-2.4 % ITL discrepancy this caused). `parse_samples_since`
    /// drops unstamped and pre-cutoff rows so a filtered verdict can never
    /// include unstampable data.
    #[test]
    fn test_parse_samples_since_filters_stale_campaign_rows() {
        let current = sample(true, 2048, 120.0, 9.5).to_json_lines("fresh01", 1_700_500);
        let stale = sample(false, 2048, 500.0, 28.0).to_json_lines("stale99", 1_700_000);
        let mut all = stale;
        all.extend(current);
        let jsonl = all
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(parse_samples(&jsonl).len(), 8, "unfiltered keeps both campaigns");
        let fresh = parse_samples_since(&jsonl, 1_700_400);
        assert_eq!(fresh.len(), 4, "cutoff keeps only the current campaign");
        assert!(fresh.iter().all(|m| m.arm_on), "stale off-arm rows dropped");

        // Legacy rows without a ts stamp (ts == 0) never survive a filter;
        // both stamped campaigns do at cutoff 1.
        let legacy = r#"{"wi":"SB3","order":"F-first","metric":"ttft_ms","value":1.0,"commit":"old","arm":"on","prompt_tokens":10}"#;
        let mixed = format!("{legacy}\n{jsonl}");
        assert_eq!(parse_samples_since(&mixed, 1).len(), 8);
        assert_eq!(parse_samples(&mixed).len(), 9, "unfiltered keeps legacy rows");
    }

    /// The WI-INF4 gate is per ordinal order: the 2026-08-23c campaign
    /// passed F-first and failed S-first on ITL. A pooled-only report would
    /// hide that; the report must print one verdict line per order.
    #[test]
    fn test_report_prints_per_order_verdicts() {
        let mk_arm = |arm_on: bool, order: &str, itl: f64| -> Vec<StoredMetric> {
            vec![
                StoredMetric {
                    arm_on,
                    order: order.into(),
                    metric: "ttft_ms".into(),
                    value: 100.0,
                    prompt_tokens: 2048,
                    ts: 0,
                },
                StoredMetric {
                    arm_on,
                    order: order.into(),
                    metric: "itl_ms".into(),
                    value: itl,
                    prompt_tokens: 2048,
                    ts: 0,
                },
            ]
        };
        // Baseline ITL 10.0 everywhere. F-first on-arm 10.1 (PASS ≤2%);
        // S-first on-arm 11.0 (FAIL >2%).
        let mut on = mk_arm(true, "F-first", 10.1);
        on.extend(mk_arm(true, "S-first", 11.0));
        let mut off = mk_arm(false, "F-first", 10.0);
        off.extend(mk_arm(false, "S-first", 10.0));

        let report = format_ab_report(&on, &off);
        assert!(report.contains("[F-first]") && report.contains("PASS"), "{report}");
        assert!(report.contains("[S-first]") && report.contains("FAIL"), "{report}");
        // Pooled verdict still present alongside.
        assert!(report.contains("WI-INF4 verdict"), "{report}");
    }
}
