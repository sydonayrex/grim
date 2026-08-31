//! Diagnostics formatting helpers for the grim tui sidebar.
//!
//! Reused from the worker-side `snapshot` and the UI render path. Pure
//! functions — no engine access. Format helpers are tested directly.
//!
//! `sidebar_lines()` returns plain strings (preserved for unit tests).
//! `sidebar_styled_lines()` returns styled ratatui Lines for the live UI.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use grim_speculative::Strategy;

// ---------------------------------------------------------------------------
// Brand color palette (mirrors grim-garage CSS variables).
// ---------------------------------------------------------------------------

/// Primary neon purple: #a855f7 — used for section headers and live values.
const C_PURPLE: Color = Color::Rgb(168, 85, 247);
/// Soft purple: #c084fc — used for key labels.
const C_PURPLE_SOFT: Color = Color::Rgb(192, 132, 252);
/// Muted grey: #888888 — used for static/idle values.
const C_MUTED: Color = Color::Rgb(136, 136, 136);
/// Success green: #10b981.
const C_GREEN: Color = Color::Rgb(16, 185, 129);
/// Warning amber: #f59e0b.
const C_AMBER: Color = Color::Rgb(245, 158, 11);
/// Danger red: #ef4444.
const C_RED: Color = Color::Rgb(239, 68, 68);

/// Format bytes in binary units: B, KiB, MiB, GiB, TiB. One decimal place
/// except plain bytes.
pub fn format_bytes(bytes: u64) -> String {
    match bytes {
        0 if false => "0 B".into(),
        0 => format!("0 B"),
        u if u < 1_024 => format!("{} B", u),
        u if u < 1_048_576 => {
            let kb = u as f64 / 1_024.0;
            format!("{:.1} KiB", kb)
        }
        u if u < 1_073_741_824 => {
            let mb = u as f64 / 1_048_576.0;
            format!("{:.1} MiB", mb)
        }
        u if u < 1_099_511_627_776 => {
            let gb = u as f64 / 1_073_741_824.0;
            format!("{:.1} GiB", gb)
        }
        u => {
            let tb = u as f64 / 1_099_511_627_776.0;
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
    let guard = 18usize.saturating_sub(fill);
    format!("[{}{}] {}%", "█".repeat(fill), "░".repeat(guard), pct)
}

/// Per-model / per-turn snapshot rendered as the diagnostics sidebar.
///
/// Every telemetry field is optional because the engine can legitimately
/// return `None` (prefill not yet run, no tokens generated, etc.). We never
/// invent a number to fill a gap — `n/a` is correct there.
#[derive(Debug, Default, Clone)]
pub struct DiagnosticsSnapshot {
    /// Model id chosen by the user.
    pub model_name: Option<String>,
    /// Quantization label, e.g. `Q8_0`.
    pub quant: Option<String>,
    /// Backend + device, e.g. `rocm gfx1100` or `cpu`.
    pub backend: String,
    /// Speculative strategy label at load time.
    pub strategy: Option<String>,
    /// Encode time in ms (tokenizer.encode).
    pub encode_ms: Option<f64>,
    /// Prompt token count (after chat template + BOS).
    pub prompt_tokens: usize,
    /// Most recent TTFT in ms.
    pub prefill_ms: Option<f64>,
    /// Engine tokens-per-sec EMA.
    pub decode_tps: Option<f64>,
    /// Per-turn measured tok/s.
    pub turn_tps: Option<f64>,
    /// Tokens generated in the current turn.
    pub tokens_generated: usize,
    /// KV cache used bytes.
    pub kv_used_bytes: u64,
    /// KV cache total bytes.
    pub kv_total_bytes: u64,
    /// KV cache blocks used.
    pub kv_blocks_used: u64,
    /// KV cache total blocks.
    pub kv_blocks_total: u64,
    /// Tokens currently in context.
    pub ctx_used: u64,
    /// Context length limit from the model (or user override).
    pub ctx_limit: u64,
    /// Average accepted speculative tokens per step this turn.
    pub accepted_per_step: Option<f64>,
    /// Total VRAM used.
    pub vram_used_bytes: u64,
    /// Total VRAM.
    pub vram_total_bytes: u64,
    /// Total system RAM used.
    pub ram_used_bytes: u64,
    /// Total system RAM.
    pub ram_total_bytes: u64,
    /// True during a load or a load-retry.
    pub loading: bool,
    /// True during a streaming generation.
    pub generating: bool,
}

/// Render `snap` as a list of sidebar lines.
pub fn sidebar_lines(snap: &DiagnosticsSnapshot) -> Vec<String> {
    let mut out = Vec::new();

    if snap.loading {
        out.push("model: loading ...".into());
    } else if let Some(name) = &snap.model_name {
        let quant = snap
            .quant
            .as_deref()
            .map(|q| format!(" ({q})"))
            .unwrap_or_default();
        out.push(format!("model: {name}{quant}"));
    } else {
        out.push("model: none loaded (/model <name>)".into());
    }

    if !snap.loading {
        out.push(format!("backend: {}", snap.backend));
    }

    // spec line: strategy label + acceptance when available.
    let spec = if let Some(s) = &snap.strategy {
        let acc = snap
            .accepted_per_step
            .map(|a| format!(" ({a:.1} tok/step)"))
            .unwrap_or_default();
        format!("spec: {s}{acc}")
    } else {
        "spec: n/a".into()
    };
    out.push(spec);

    if snap.prompt_tokens > 0 && !snap.loading {
        out.push(format!(
            "encode: {} ({} tok)",
            format_ms(snap.encode_ms),
            snap.prompt_tokens
        ));
    }
    if let Some(v) = snap.prefill_ms {
        out.push(format!("prefill: {v:.1} ms"));
    }
    out.push(format!("decode: {}", format_tps(snap.decode_tps)));
    if let Some(t) = snap.turn_tps {
        out.push(format!(
            "turn: {} ({} tok)",
            format_tps(Some(t)),
            snap.tokens_generated
        ));
    }

    // KV cache.
    if snap.kv_total_bytes > 0 {
        out.push(format!(
            "kv {}",
            bar(snap.kv_used_bytes, snap.kv_total_bytes)
        ));
        out.push(format!(
            "{} / {} ({} / {} blk)",
            format_bytes(snap.kv_used_bytes),
            format_bytes(snap.kv_total_bytes),
            snap.kv_blocks_used,
            snap.kv_blocks_total
        ));
    } else {
        out.push("kv: n/a".into());
    }

    // context.
    if snap.ctx_limit > 0 {
        out.push(format!("ctx {}", bar(snap.ctx_used, snap.ctx_limit)));
        out.push(format!("{} / {} tok", snap.ctx_used, snap.ctx_limit));
        if snap.ctx_used * 100 / snap.ctx_limit >= 85 {
            out.push("! ctx >= 85% (try /clear)".into());
        }
    } else {
        out.push(format!("ctx {} tok", snap.ctx_used));
        out.push("ctx limit: ?".into());
    }

    // vram.
    if snap.vram_total_bytes > 0 {
        out.push(format!(
            "vram {}",
            bar(snap.vram_used_bytes, snap.vram_total_bytes)
        ));
        out.push(format!(
            "{} / {}",
            format_bytes(snap.vram_used_bytes),
            format_bytes(snap.vram_total_bytes)
        ));
    } else {
        out.push("vram: n/a".into());
    }

    // ram.
    if snap.ram_total_bytes > 0 {
        out.push(format!(
            "ram {}",
            bar(snap.ram_used_bytes, snap.ram_total_bytes)
        ));
        out.push(format!(
            "{} / {}",
            format_bytes(snap.ram_used_bytes),
            format_bytes(snap.ram_total_bytes)
        ));
    } else {
        out.push("ram: n/a".into());
    }

    out
}

// ---------------------------------------------------------------------------
// Styled variant (used by the live TUI; sidebar_lines kept for unit tests).
// ---------------------------------------------------------------------------

/// Build a gradient-colored bar gauge Line.
///
/// Color selection: green below 60%, amber 60-84%, red at 85%+.
/// Width is fixed at 16 fill cells so it fits a narrow sidebar.
fn bar_line(used: u64, total: u64) -> Line<'static> {
    let pct = ratio_percent(used, total) as usize;
    let fill = pct * 16 / 100;
    let empty = 16usize.saturating_sub(fill);
    let bar_color = if pct >= 85 {
        C_RED
    } else if pct >= 60 {
        C_AMBER
    } else {
        C_GREEN
    };
    Line::from(vec![
        Span::styled("▓".repeat(fill), Style::default().fg(bar_color)),
        Span::styled("░".repeat(empty), Style::default().fg(C_MUTED)),
        Span::styled(format!(" {}%", pct), Style::default().fg(C_MUTED)),
    ])
}

/// Render a key/value pair as a single styled Line.
///
/// Key uses soft-purple, value uses white (for readable contrast).
fn kv(key: &'static str, value: String, value_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<10}", key), Style::default().fg(C_PURPLE_SOFT)),
        Span::styled(value, Style::default().fg(value_color)),
    ])
}

/// Render `snap` as styled ratatui Lines for the sidebar panel.
///
/// Contrast contract: all text is white or light-colored on the dark terminal
/// background. Purple is used only for key labels and section accents, never
/// as body text color on its own.
pub fn sidebar_styled_lines(snap: &DiagnosticsSnapshot) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();

    // Model row.
    if snap.loading {
        out.push(Line::from(vec![
            Span::styled("model     ", Style::default().fg(C_PURPLE_SOFT)),
            Span::styled("loading...", Style::default().fg(C_AMBER).add_modifier(Modifier::BOLD)),
        ]));
    } else if let Some(name) = &snap.model_name {
        let quant = snap.quant.as_deref().unwrap_or("");
        out.push(Line::from(vec![
            Span::styled("model     ", Style::default().fg(C_PURPLE_SOFT)),
            Span::styled(name.clone(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(
                if quant.is_empty() { String::new() } else { format!(" ({})", quant) },
                Style::default().fg(C_MUTED),
            ),
        ]));
    } else {
        out.push(kv("model", "none loaded".into(), C_MUTED));
        out.push(Line::from(Span::styled(
            "  /model <name>",
            Style::default().fg(C_MUTED),
        )));
    }

    // Backend.
    if !snap.loading {
        out.push(kv("backend", snap.backend.clone(), Color::White));
    }

    // Speculative decoding strategy.
    let spec_val = if let Some(s) = &snap.strategy {
        let acc = snap
            .accepted_per_step
            .map(|a| format!(" ({:.1}/step)", a))
            .unwrap_or_default();
        format!("{}{}", s, acc)
    } else {
        "n/a".into()
    };
    out.push(kv("spec", spec_val, Color::White));

    // Encode / prefill timing.
    if snap.prompt_tokens > 0 && !snap.loading {
        out.push(kv(
            "encode",
            format!("{} ({} tok)", format_ms(snap.encode_ms), snap.prompt_tokens),
            Color::White,
        ));
    }
    if let Some(v) = snap.prefill_ms {
        out.push(kv("prefill", format!("{:.1} ms", v), Color::White));
    }

    // Decode speed: highlight live values in purple.
    let tps_color = if snap.decode_tps.is_some() { C_PURPLE } else { C_MUTED };
    out.push(kv("decode", format_tps(snap.decode_tps), tps_color));
    if let Some(t) = snap.turn_tps {
        out.push(kv(
            "turn",
            format!("{} ({} tok)", format_tps(Some(t)), snap.tokens_generated),
            Color::White,
        ));
    }

    // Section divider.
    out.push(Line::from(Span::styled(
        "──────────────────",
        Style::default().fg(C_MUTED),
    )));

    // KV cache.
    if snap.kv_total_bytes > 0 {
        out.push(Line::from(Span::styled("kv cache", Style::default().fg(C_PURPLE_SOFT))));
        out.push(bar_line(snap.kv_used_bytes, snap.kv_total_bytes));
        out.push(Line::from(Span::styled(
            format!(
                "{} / {} ({}/{} blk)",
                format_bytes(snap.kv_used_bytes),
                format_bytes(snap.kv_total_bytes),
                snap.kv_blocks_used,
                snap.kv_blocks_total
            ),
            Style::default().fg(C_MUTED),
        )));
    } else {
        out.push(kv("kv", "n/a".into(), C_MUTED));
    }

    // Context usage.
    if snap.ctx_limit > 0 {
        out.push(Line::from(Span::styled("context", Style::default().fg(C_PURPLE_SOFT))));
        out.push(bar_line(snap.ctx_used, snap.ctx_limit));
        out.push(Line::from(Span::styled(
            format!("{} / {} tok", snap.ctx_used, snap.ctx_limit),
            Style::default().fg(C_MUTED),
        )));
        if snap.ctx_used * 100 / snap.ctx_limit >= 85 {
            out.push(Line::from(Span::styled(
                "  ctx >= 85% — /clear",
                Style::default().fg(C_RED).add_modifier(Modifier::BOLD),
            )));
        }
    } else {
        out.push(kv("ctx", format!("{} tok", snap.ctx_used), C_MUTED));
        out.push(kv("ctx lim", "?".into(), C_MUTED));
    }

    // VRAM.
    if snap.vram_total_bytes > 0 {
        out.push(Line::from(Span::styled("vram", Style::default().fg(C_PURPLE_SOFT))));
        out.push(bar_line(snap.vram_used_bytes, snap.vram_total_bytes));
        out.push(Line::from(Span::styled(
            format!(
                "{} / {}",
                format_bytes(snap.vram_used_bytes),
                format_bytes(snap.vram_total_bytes)
            ),
            Style::default().fg(C_MUTED),
        )));
    } else {
        out.push(kv("vram", "n/a".into(), C_MUTED));
    }

    // System RAM.
    if snap.ram_total_bytes > 0 {
        out.push(Line::from(Span::styled("ram", Style::default().fg(C_PURPLE_SOFT))));
        out.push(bar_line(snap.ram_used_bytes, snap.ram_total_bytes));
        out.push(Line::from(Span::styled(
            format!(
                "{} / {}",
                format_bytes(snap.ram_used_bytes),
                format_bytes(snap.ram_total_bytes)
            ),
            Style::default().fg(C_MUTED),
        )));
    } else {
        out.push(kv("ram", "n/a".into(), C_MUTED));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_speculative::Strategy;

    #[test]
    fn formats_and_gauges() {
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_ms(None), "n/a");
        assert_eq!(format_ms(Some(3.5)), "3.5 ms");
        assert_eq!(format_tps(Some(41.23)), "41.2 tok/s");
        assert_eq!(acceptance_rate(0, 0), None);
        assert_eq!(acceptance_rate(7, 3), Some(7.0 / 3.0));
        assert_eq!(
            bar(31, 100),
            format!("[{}] 31%", "█".repeat(5) + &"░".repeat(13))
        );
        assert_eq!(bar(0, 0), "[░░░░░░░░░░░░░░░░░░] 0%");
    }

    #[test]
    fn ratios_and_labels() {
        assert_eq!(ratio_percent(5, 10), 50);
        assert_eq!(ratio_percent(11, 10), 100);
        assert_eq!(ratio_percent(0, 0), 0);
        assert_eq!(strategy_label(&Strategy::Plain), "plain (no speculation)");
        assert_eq!(strategy_label(&Strategy::DSpark), "DSpark");
        assert_eq!(strategy_label(&Strategy::NativeMtp), "native MTP");
    }

    #[test]
    fn sidebar_lines_render_full_snapshot() {
        let snap = DiagnosticsSnapshot {
            model_name: Some("LFM2.5-230M".into()),
            quant: Some("Q8_0".into()),
            backend: "rocm gfx1100".into(),
            strategy: Some("DSpark".into()),
            encode_ms: Some(3.1),
            prompt_tokens: 128,
            prefill_ms: Some(142.0),
            decode_tps: Some(41.2),
            turn_tps: Some(38.9),
            tokens_generated: 57,
            kv_used_bytes: 1_288_490_187,
            kv_total_bytes: 4_294_967_296,
            kv_blocks_used: 312,
            kv_blocks_total: 1024,
            ctx_used: 2412,
            ctx_limit: 8192,
            accepted_per_step: Some(2.3),
            vram_used_bytes: 3_221_225_472,
            vram_total_bytes: 12_884_901_888,
            ram_used_bytes: 16_106_127_360,
            ram_total_bytes: 32_212_254_720,
            loading: false,
            generating: false,
        };
        assert_eq!(
            sidebar_lines(&snap),
            vec![
                "model: LFM2.5-230M (Q8_0)".to_string(),
                "backend: rocm gfx1100".to_string(),
                "spec: DSpark (2.3 tok/step)".to_string(),
                "encode: 3.1 ms (128 tok)".to_string(),
                "prefill: 142.0 ms".to_string(),
                "decode: 41.2 tok/s".to_string(),
                "turn: 38.9 tok/s (57 tok)".to_string(),
                format!("kv {}", bar(1_288_490_187, 4_294_967_296)),
                "1.2 GiB / 4.0 GiB (312 / 1024 blk)".to_string(),
                format!("ctx {}", bar(2412, 8192)),
                "2412 / 8192 tok".to_string(),
                format!("vram {}", bar(3_221_225_472, 12_884_901_888)),
                "3.0 GiB / 12.0 GiB".to_string(),
                format!("ram {}", bar(16_106_127_360, 32_212_254_720)),
                "15.0 GiB / 30.0 GiB".to_string(),
            ]
        );
    }

    #[test]
    fn sidebar_lines_empty_state() {
        let snap = DiagnosticsSnapshot::default();
        let lines = sidebar_lines(&snap);
        assert_eq!(lines[0], "model: none loaded (/model <name>)");
        assert!(
            lines.iter().any(|l| l == "vram: n/a"),
            "missing vram n/a line"
        );
        assert!(
            lines.iter().any(|l| l == "ram: n/a"),
            "missing ram n/a line"
        );
    }

    #[test]
    fn sidebar_lines_context_warning() {
        let snap = DiagnosticsSnapshot {
            ctx_used: 7200,
            ctx_limit: 8000,
            ..Default::default()
        };
        let lines = sidebar_lines(&snap);
        assert!(lines.iter().any(|l| l == "! ctx >= 85% (try /clear)"));
    }
}
