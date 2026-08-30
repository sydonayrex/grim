//! Toast notification system for the TUI.
//!
//! Transient info/success/warning/error notifications that auto-dismiss after
//! a timeout. Borrowed from the opencode-dev pattern: a single toast slot in
//! `App` with a deadline; the `ui()` loop renders it as an overlay in the
//! top-right corner and clears it when the deadline passes.

use std::time::{Duration, Instant};

use ratatui::style::{Color, Style};
use ratatui::text::Line;

/// Visual variant determining color and default duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastVariant {
    Info,
    Success,
    Warning,
    Error,
}

/// A single toast notification.
#[derive(Debug, Clone)]
pub struct Toast {
    pub title: Option<String>,
    pub message: String,
    pub variant: ToastVariant,
    pub deadline: Instant,
}

/// Toast styling: color per variant.
impl ToastVariant {
    /// Color used for the border and title.
    pub fn color(self) -> Color {
        match self {
            ToastVariant::Info => Color::Cyan,
            ToastVariant::Success => Color::Green,
            ToastVariant::Warning => Color::Yellow,
            ToastVariant::Error => Color::Red,
        }
    }

    /// Default display duration for this variant.
    pub fn default_duration(self) -> Duration {
        match self {
            ToastVariant::Info => Duration::from_secs(3),
            ToastVariant::Success => Duration::from_secs(3),
            ToastVariant::Warning => Duration::from_secs(5),
            ToastVariant::Error => Duration::from_secs(8),
        }
    }
}

impl Toast {
    /// Create a new toast with the variant's default duration.
    pub fn new(variant: ToastVariant, message: impl Into<String>) -> Self {
        Self {
            title: None,
            message: message.into(),
            variant,
            deadline: Instant::now() + variant.default_duration(),
        }
    }

    /// Create a toast with a title.
    pub fn with_title(variant: ToastVariant, title: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            title: Some(title.into()),
            message: message.into(),
            variant,
            deadline: Instant::now() + variant.default_duration(),
        }
    }

    /// Create a toast with a custom duration.
    pub fn with_duration(
        variant: ToastVariant,
        message: impl Into<String>,
        duration: Duration,
    ) -> Self {
        Self {
            title: None,
            message: message.into(),
            variant,
            deadline: Instant::now() + duration,
        }
    }

    /// True if the toast has expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.deadline }
}

/// Convenience constructors.
impl Toast {
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(ToastVariant::Info, message)
    }

    pub fn success(message: impl Into<String>) -> Self {
        Self::new(ToastVariant::Success, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(ToastVariant::Warning, message)
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(ToastVariant::Error, message)
    }
}

/// Render a toast as styled Lines for a given width.
pub fn render_toast(toast: &Toast, width: u16) -> Vec<Line<'_>> {
    let inner_width = width.saturating_sub(2) as usize;
    let style = Style::default().fg(toast.variant.color());
    let mut lines = Vec::new();

    if let Some(title) = &toast.title {
        lines.push(Line::from(ratatui::text::Span::styled(
            truncate(title, inner_width),
            style.add_modifier(ratatui::style::Modifier::BOLD),
        )));
    }
    lines.push(Line::from(ratatui::text::Span::styled(
        truncate(&toast.message, inner_width),
        Style::default().fg(Color::White),
    )));

    lines
}

/// Truncate a string to fit within `max_cols` (ASCII-safe).
fn truncate(s: &str, max_cols: usize) -> String {
    if s.len() <= max_cols {
        s.to_string()
    } else {
        let mut result = s.chars().take(max_cols.saturating_sub(1)).collect::<String>();
        result.push('…');
        result
    }
}
