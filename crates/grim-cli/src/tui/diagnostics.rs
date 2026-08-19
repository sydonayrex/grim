//! Diagnostics formatting helpers for the grim tui sidebar.
//!
//! Reused from the worker-side `snapshot` and the UI render path. Pure
//! functions — no engine access. Format helpers are tested directly.

use grim_speculative::Strategy;

/// Format bytes in binary units: B, KiB, MiB, GiB, TiB. One decimal place
/// except plain bytes.
pub fn format_bytes(bytes: u64) -> String {
    match bytes {
        0 => "0 B".into(),
        _ if bytes < 1_024 => format!("{} B", bytes),
        _ if bytes < 1_048_576 => {
            let kb = bytes as f64 / 1_024.0;
            format!("{:.1} KiB", kb)
        }
        _ if bytes < 1_073_741_824 => {
            let mb = bytes as f64 / 1_048_576.0;
            format!("{:.1} MiB", mb)
        }
        _ if bytes < 1_099_511_627_776 => {
            let gb = bytes as f64 / 1_073_741_824.0;
            format!("{:.1} GiB", gb)
        }
        _ => {
            let tb = bytes as f64 / 1_099_511_627_776.0;
            format!("{:.1} TiB", tb)
        }
    }
}

/// Ratio of `used / total` in percent, clamped to 100. Returns 0 when total
/// is 0.
pub fn ratio_percent(used: u64, total: u64) -> u16 {
    let pct = (used as u128 * 100 / total.max(1) as u128) as u16;
    pct.min(100)
}

/// Format an optional duration in milliseconds. `None` -> "n/a".
pub fn format_ms(opt: Option<f64>) -> String {
    match opt {
        None => "n/a".into(),
        Some(v) => format!("{:.1} ms", v),
    }
}

/// Format an optional tokens-per-second value. `None` -> "n/a".
pub fn format_tps(opt: Option<f64>) -> String {
    match opt {
        None => "n/a".into(),
        Some(v) => format!("{:.1} tok/s", v),
    }
}

/// Average accepted tokens per step. Returns `None` when there were no steps.
pub fn acceptance_rate(accepted: usize, steps: usize) -> Option<f64> {
    if steps == 0 {
        None
    } else {
        Some(accepted as f64 / steps as f64)
    }
}

/// Human label for a speculative decoding strategy.
pub fn strategy_label(s: &Strategy) -> &'static str {
    match s {
        Strategy::Plain => "plain (no speculation)",
        Strategy::DSpark => "DSpark",
        Strategy::NativeMtp => "native MTP",
    }
}

/// Bar gauge of width 18 from `used / total`. Empty on zero total.
pub fn bar(used: u64, total: u64) -> String {
    let pct = ratio_percent(used, total) as usize;
    let fill = pct * 18 / 100;
    let guard = 18 - fill;
    format!("[{}{}] {}%", "█".repeat(fill), "░".repeat(guard), pct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_speculative::Strategy;

    #[test]
    fn formats_and_gauges() {
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
        assert_eq!(format_ms(None), "n/a");
        assert_eq!(format_ms(Some(3.14)), "3.1 ms");
        assert_eq!(format_tps(Some(41.23)), "41.2 tok/s");
        assert_eq!(acceptance_rate(0, 0), None);
        assert_eq!(acceptance_rate(7, 3), Some(7.0 / 3.0));
        assert_eq!(bar(31, 100), "[█████░░░░░░░░░░░░░░░] 31%");
    }

    #[test]
    fn ratios_and_labels() {
        assert_eq!(ratio_percent(5, 10), 50);
        assert_eq!(ratio_percent(0, 0), 0);
        assert_eq!(strategy_label(&Strategy::Plain), "plain (no speculation)");
        assert_eq!(strategy_label(&Strategy::DSpark), "DSpark");
        assert_eq!(strategy_label(&Strategy::NativeMtp), "native MTP");
    }
}
