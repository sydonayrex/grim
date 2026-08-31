//! Structured chat transcript with role-based styling and reasoning folding.
//!
//! Separates message nodes by role, extracts `<think>...</think>` CoT traces,
//! and renders styled Ratatui Lines with folding capability.

use std::cell::RefCell;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Message author role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    ToolCall,
    ToolResult,
    System,
    Error,
    Hint,
}

/// One structured node in the chat history.
#[derive(Debug, Clone)]
pub struct MessageNode {
    /// Author role determining visual styling and prefix.
    pub role: Role,
    /// Message textual content.
    pub content: String,
    /// Extracted `<think>` chain-of-thought trace if present.
    pub thinking: Option<String>,
    /// Whether the thinking block is currently folded in the UI.
    pub thought_folded: bool,
    /// Optional turn statistics line rendered under assistant responses.
    pub turn_stats: Option<String>,
    /// Tool name for ToolCall/ToolResult roles.
    pub tool_name: Option<String>,
    /// Pretty-printed tool arguments (JSON) for ToolCall.
    pub tool_arguments: Option<String>,
}

impl Default for MessageNode {
    fn default() -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            thinking: None,
            thought_folded: false,
            turn_stats: None,
            tool_name: None,
            tool_arguments: None,
        }
    }
}

/// Structured transcript container managing message history and active streaming state.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// List of completed message nodes.
    pub nodes: Vec<MessageNode>,
    /// Ongoing streaming buffer for the active turn.
    pub streaming_raw: String,
    /// Cached rendered lines invalidated only when `nodes.len()` or `streaming_raw` changes.
    cached_lines: RefCell<Option<Vec<Line<'static>>>>,
    cached_nodes_len: RefCell<usize>,
    cached_streaming_len: RefCell<usize>,
    cached_max_width: RefCell<usize>,
}

impl Default for Transcript {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            streaming_raw: String::new(),
            cached_lines: RefCell::new(None),
            cached_nodes_len: RefCell::new(usize::MAX),
            cached_streaming_len: RefCell::new(usize::MAX),
            cached_max_width: RefCell::new(usize::MAX),
        }
    }
}

impl Transcript {
    /// Create a new empty transcript.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all transcript messages and reset streaming buffer.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.streaming_raw.clear();
        self.invalidate_cache();
    }

    fn invalidate_cache(&self) {
        *self.cached_lines.borrow_mut() = None;
    }

    fn is_cache_valid(&self, max_width: usize) -> bool {
        self.cached_lines.borrow().is_some()
            && *self.cached_nodes_len.borrow() == self.nodes.len()
            && *self.cached_streaming_len.borrow() == self.streaming_raw.len()
            && *self.cached_max_width.borrow() == max_width
    }

    fn store_cache(&self, lines: Vec<Line<'static>>, max_width: usize) {
        *self.cached_nodes_len.borrow_mut() = self.nodes.len();
        *self.cached_streaming_len.borrow_mut() = self.streaming_raw.len();
        *self.cached_max_width.borrow_mut() = max_width;
        *self.cached_lines.borrow_mut() = Some(lines);
    }

    /// Returns true when `content` contains an unclosed fenced code block.
    fn has_incomplete_fence(content: &str) -> bool {
        content.matches("```").count() % 2 == 1
    }

    /// Render plain lines for streaming content with an open fence (no markdown re-highlight).
    fn render_plain_streaming(content: &str, c_purple_dim: Color, _c_muted: Color) -> Vec<Line<'static>> {
        content
            .lines()
            .map(|l| {
                Line::from(vec![
                    ratatui::text::Span::styled("│ ", ratatui::style::Style::default().fg(c_purple_dim)),
                    ratatui::text::Span::raw(l.to_string()),
                ])
            })
            .collect()
    }

    /// Add a user message node.
    pub fn push_user(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::User,
            content: text,
            ..Default::default()
        });
        self.invalidate_cache();
    }

    /// Add a system notification message.
    pub fn push_system(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::System,
            content: text,
            ..Default::default()
        });
        self.invalidate_cache();
    }

    /// Add an error notification message.
    pub fn push_error(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::Error,
            content: text,
            ..Default::default()
        });
        self.invalidate_cache();
    }

    /// Add a tool call invocation message with structured name + arguments.
    pub fn push_tool_call(&mut self, name: &str, arguments: &str) {
        let pretty = serde_json::from_str::<serde_json::Value>(arguments)
            .and_then(|v| serde_json::to_string_pretty(&v))
            .unwrap_or_else(|_| arguments.to_string());
        self.nodes.push(MessageNode {
            role: Role::ToolCall,
            content: pretty,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
            tool_name: Some(name.to_string()),
            tool_arguments: Some(arguments.to_string()),
        });
        self.invalidate_cache();
    }

    /// Add a tool result output message.
    pub fn push_tool_result(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::ToolResult,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
            tool_name: None,
            tool_arguments: None,
        });
        self.invalidate_cache();
    }

    /// Append a token chunk to the active turn streaming buffer.
    pub fn append_token(&mut self, token: &str) {
        self.streaming_raw.push_str(token);
        // Streaming changes invalidate the cached lines (streaming_raw length changed).
        // We keep the cache but is_cache_valid will fail; no need to explicitly clear here
        // but clearing avoids stale cache holding large vec during rapid streaming.
        // We do not clear aggressively to allow the virtualized path to reuse partial.
        self.invalidate_cache();
    }

    /// Finalize assistant turn, parsing optional `<think>...</think>` tags and attaching metrics.
    pub fn finish_turn(&mut self, summary_line: String) {
        let (thinking, content) = parse_thinking_tags(&self.streaming_raw);
        self.nodes.push(MessageNode {
            role: Role::Assistant,
            content,
            thinking,
            thought_folded: true,
            turn_stats: Some(summary_line),
            ..Default::default()
        });
        self.streaming_raw.clear();
        self.invalidate_cache();
    }

    /// Toggle collapsed state of the latest assistant reasoning block.
    pub fn toggle_fold_last_thought(&mut self) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.thinking.is_some()) {
            node.thought_folded = !node.thought_folded;
        }
        self.invalidate_cache();
    }

/// Build styled ratatui Lines for rendering in the main chat viewport.
    ///
    /// Visual contract:
    /// - Role chips use box-drawing (top-left corner + label) in role color.
    /// - Body text is always white so purple borders never clash with content.
    /// - Thinking blocks have a dim italic style with a left-gutter `╎ ` marker.
    /// - Tool calls render as a diff-like block with a magenta header.
    /// - Streaming output appends a block cursor `▋`.
    ///
    /// Lines longer than `max_width` chars are hard-wrapped so that model
    /// output containing raw template syntax or other long content cannot
    /// corrupt the TUI layout.
    pub fn render_lines(&self) -> Vec<Line<'static>> {
        self.render_lines_wrapped(200)
    }

    /// Same as `render_lines` but with a configurable max line width.
    /// Uses an internal cache invalidated only when `nodes.len()` or `streaming_raw` changes.
    pub fn render_lines_wrapped(&self, max_width: usize) -> Vec<Line<'static>> {
        if self.is_cache_valid(max_width) {
            if let Some(cached) = self.cached_lines.borrow().clone() {
                return cached;
            }
        }
        // Brand colors (mirrored from grim-garage palette).
        let c_purple     = Color::Rgb(168, 85, 247);   // #a855f7 — user chip
        let c_purple_dim = Color::Rgb(112, 50, 180);   // dim purple — borders
        let c_cyan       = Color::Rgb(34, 211, 238);    // assistant chip
        let c_magenta    = Color::Rgb(232, 121, 249);   // tool chip
        let c_green      = Color::Rgb(16, 185, 129);    // tool result / success
        let c_red        = Color::Rgb(239, 68, 68);     // error
        let c_amber      = Color::Rgb(245, 158, 11);    // system / warning
        let c_muted      = Color::Rgb(136, 136, 136);   // stats / dim text
        let c_thinking   = Color::Rgb(180, 140, 255);   // thinking gutter

        if self.nodes.is_empty() {
            // Centered welcome with pills when transcript is empty
            return vec![
                Line::from(Span::styled(" Welcome to GRIM ", Style::default().fg(c_purple).add_modifier(Modifier::BOLD))).centered(),
                Line::from(Span::styled(" Type a message or use a command to get started ", Style::default().fg(c_muted))).centered(),
                Line::raw(""),
                Line::from(vec![
                    Span::styled(" /model ", Style::default().fg(Color::White).bg(c_purple_dim).add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(" /help ", Style::default().fg(Color::White).bg(c_purple_dim)),
                    Span::raw(" "),
                    Span::styled(" @ file ", Style::default().fg(Color::White).bg(c_purple_dim)),
                    Span::raw(" "),
                    Span::styled(" F4 ", Style::default().fg(Color::White).bg(c_purple_dim)),
                ]).centered(),
                Line::raw(""),
                Line::from(Span::styled(" no model — /model or F4 ", Style::default().fg(c_amber))).centered(),
            ];
        }

        let mut lines: Vec<Line<'static>> = Vec::new();

        for node in &self.nodes {
            match node.role {
                Role::User => {
                    // ╭─ you ──────────
                    lines.push(Line::from(vec![
                        Span::styled("╭─ ", Style::default().fg(c_purple_dim)),
                        Span::styled("you", Style::default().fg(c_purple).add_modifier(Modifier::BOLD)),
                        Span::styled(" ─────────────────────────────────", Style::default().fg(c_purple_dim)),
                    ]));
                    // Indented content lines.
                    for content_line in node.content.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("│ ", Style::default().fg(c_purple_dim)),
                            Span::raw(content_line.to_string()),
                        ]));
                    }
                    lines.push(Line::from(vec![
                        Span::styled("╰───────────────────────────────────", Style::default().fg(c_purple_dim)),
                    ]));
                    lines.push(Line::raw(""));
                }
                Role::Assistant => {
                    // Thinking block (folded or expanded accordion).
                    if let Some(think) = &node.thinking {
                        if node.thought_folded {
                            lines.push(Line::from(vec![
                                Span::styled("  ▶ ", Style::default().fg(c_muted)),
                                Span::styled(
                                    "[reasoning — Tab to expand]",
                                    Style::default().fg(c_muted).add_modifier(Modifier::ITALIC),
                                ),
                            ]));
                        } else {
                            lines.push(Line::from(vec![
                                Span::styled("  ▼ ", Style::default().fg(c_thinking)),
                                Span::styled(
                                    "reasoning",
                                    Style::default().fg(c_thinking).add_modifier(Modifier::ITALIC),
                                ),
                                Span::styled(" — Tab to collapse", Style::default().fg(c_muted)),
                            ]));
                            for tline in think.lines() {
                                lines.push(Line::from(vec![
                                    Span::styled("  ╎ ", Style::default().fg(c_thinking)),
                                    Span::styled(
                                        tline.to_string(),
                                        Style::default()
                                            .fg(c_muted)
                                            .add_modifier(Modifier::ITALIC),
                                    ),
                                ]));
                            }
                            lines.push(Line::from(vec![
                                Span::styled("  ╎", Style::default().fg(c_thinking)),
                            ]));
                        }
                    }

                    // ╭─ grim ──────────
                    lines.push(Line::from(vec![
                        Span::styled("╭─ ", Style::default().fg(c_purple_dim)),
                        Span::styled("grim", Style::default().fg(c_cyan).add_modifier(Modifier::BOLD)),
                        Span::styled(" ─────────────────────────────────", Style::default().fg(c_purple_dim)),
                    ]));

                    // Markdown-rendered content with gutter.
                    let md_lines = crate::tui::markdown::render_markdown(&node.content);
                    if md_lines.is_empty() || md_lines.iter().all(|l| l.spans.is_empty()) {
                        // Blank assistant turn (e.g. thinking-only).
                    } else {
                        for md_line in md_lines {
                            let mut spans = vec![Span::styled("│ ", Style::default().fg(c_purple_dim))];
                            spans.extend(md_line.spans);
                            lines.push(Line::from(spans));
                        }
                    }

                    // Turn stats footer.
                    if let Some(stats) = &node.turn_stats {
                        lines.push(Line::from(vec![
                            Span::styled("│ ", Style::default().fg(c_purple_dim)),
                            Span::styled(stats.clone(), Style::default().fg(c_muted)),
                        ]));
                    }
                    lines.push(Line::from(vec![
                        Span::styled("╰───────────────────────────────────", Style::default().fg(c_purple_dim)),
                    ]));
                    lines.push(Line::raw(""));
                }
                Role::ToolCall => {
                    // ╭─ tool: <name> ──
                    let name = node.tool_name.as_deref().unwrap_or("unknown");
                    lines.push(Line::from(vec![
                        Span::styled("╭─ ", Style::default().fg(c_purple_dim)),
                        Span::styled("tool", Style::default().fg(c_magenta).add_modifier(Modifier::BOLD)),
                        Span::styled(format!(": {} ", name), Style::default().fg(Color::White)),
                        Span::styled("────────────────────────────", Style::default().fg(c_purple_dim)),
                    ]));
                    // Arguments as diff-like green lines.
                    for arg_line in node.content.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("│ + ", Style::default().fg(c_green)),
                            Span::styled(arg_line.to_string(), Style::default().fg(Color::White)),
                        ]));
                    }
                    lines.push(Line::from(vec![
                        Span::styled("╰───────────────────────────────────", Style::default().fg(c_purple_dim)),
                    ]));
                }
                Role::ToolResult => {
                    // Compact result block indented under the tool call.
                    lines.push(Line::from(vec![
                        Span::styled("  ✓ ", Style::default().fg(c_green)),
                        Span::styled("result: ", Style::default().fg(c_muted)),
                        Span::styled(
                            // Truncate long results to first line.
                            node.content.lines().next().unwrap_or("(empty)").to_string(),
                            Style::default().fg(Color::White),
                        ),
                    ]));
                    // If multi-line, show remaining lines indented.
                    for extra in node.content.lines().skip(1).take(3) {
                        lines.push(Line::from(vec![
                            Span::styled("       ", Style::default()),
                            Span::styled(extra.to_string(), Style::default().fg(Color::White)),
                        ]));
                    }
                    lines.push(Line::raw(""));
                }
                Role::System => {
                    lines.push(Line::from(vec![
                        Span::styled("  ℹ ", Style::default().fg(c_amber)),
                        Span::styled(node.content.clone(), Style::default().fg(Color::White)),
                    ]));
                }
                Role::Error => {
                    lines.push(Line::from(vec![
                        Span::styled("  ✖ ", Style::default().fg(c_red).add_modifier(Modifier::BOLD)),
                        Span::styled(
                            node.content.clone(),
                            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                Role::Hint => {
                    lines.push(Line::from(vec![
                        Span::styled("  ─ ", Style::default().fg(c_muted)),
                        Span::styled(node.content.clone(), Style::default().fg(c_muted)),
                    ]));
                }
            }
        }

        // Active streaming output — show thinking gutter + content + blinking cursor.
        if !self.streaming_raw.is_empty() {
            let (thinking, content) = parse_thinking_tags(&self.streaming_raw);

            if let Some(ref think) = thinking {
                lines.push(Line::from(vec![
                    Span::styled("  ▼ ", Style::default().fg(c_thinking)),
                    Span::styled("reasoning...", Style::default().fg(c_thinking).add_modifier(Modifier::ITALIC)),
                ]));
                for tline in think.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("  ╎ ", Style::default().fg(c_thinking)),
                        Span::styled(tline.to_string(), Style::default().fg(c_muted).add_modifier(Modifier::ITALIC)),
                    ]));
                }
            }

            if !content.is_empty() {
                // Role chip for in-progress assistant message.
                lines.push(Line::from(vec![
                    Span::styled("╭─ ", Style::default().fg(c_purple_dim)),
                    Span::styled("grim", Style::default().fg(c_cyan).add_modifier(Modifier::BOLD)),
                    Span::styled(" ─────────────────────────────────", Style::default().fg(c_purple_dim)),
                ]));
                // Code-aware streaming: buffer until fence closes before re-highlighting.
                // If content contains an unclosed "```", render as plain to avoid flicker.
                let (md_lines, is_plain) = if Self::has_incomplete_fence(&content) {
                    let plain: Vec<Line<'static>> = Self::render_plain_streaming(&content, c_purple_dim, c_muted)
                        .into_iter()
                        .map(|l| {
                            // plain lines already have gutter, keep as-is
                            l
                        })
                        .collect();
                    (plain, true)
                } else {
                    (crate::tui::markdown::render_markdown(&content), false)
                };
                if is_plain {
                    let last_idx = md_lines.len().saturating_sub(1);
                    for (i, md_line) in md_lines.into_iter().enumerate() {
                        let mut spans = md_line.spans;
                        if i == last_idx {
                            spans.push(Span::styled("▋", Style::default().fg(c_cyan)));
                        }
                        lines.push(Line::from(spans));
                    }
                } else {
                    let last_idx = md_lines.len().saturating_sub(1);
                    for (i, md_line) in md_lines.into_iter().enumerate() {
                        let mut spans = vec![Span::styled("│ ", Style::default().fg(c_purple_dim))];
                        spans.extend(md_line.spans);
                        // Append streaming cursor on the final line.
                        if i == last_idx {
                            spans.push(Span::styled("▋", Style::default().fg(c_cyan)));
                        }
                        lines.push(Line::from(spans));
                    }
                }
            } else if thinking.is_some() {
                // Pure thinking state (no content yet): show a cursor line under the gutter.
                lines.push(Line::from(vec![
                    Span::styled("  ╎ ", Style::default().fg(c_thinking)),
                    Span::styled("▋", Style::default().fg(c_thinking)),
                ]));
            }
        }

        // Hard-wrap any lines that exceed max_width chars so that model
        // output containing raw template syntax or other long content
        // cannot corrupt the TUI layout.
        let wrapped = wrap_lines(lines, max_width);
        self.store_cache(wrapped.clone(), max_width);
        wrapped
    }

    /// Virtualized transcript: return only the visible window of `content_height`
    /// lines around `scroll_offset` from the cached full render. Uses the same
    /// cache key as `render_lines_wrapped` (invalidated only when nodes.len() or
    /// streaming_raw changes).
    pub fn render_lines_virtualized(
        &self,
        max_width: usize,
        content_height: usize,
        scroll_offset: usize,
    ) -> Vec<Line<'static>> {
        let full = self.render_lines_wrapped(max_width);
        if full.len() <= content_height {
            return full;
        }
        // scroll_offset == 0 means pinned to bottom (newest lines visible).
        // Otherwise offset counts from bottom: 0 => tail, N => N lines scrolled up.
        let total = full.len();
        let end = total.saturating_sub(scroll_offset);
        let start = end.saturating_sub(content_height);
        full[start..end].to_vec()
    }
}

/// Hard-wrap a Vec<Line> at `max_width` chars. Lines already shorter than
/// the limit are passed through unchanged.
fn wrap_lines(lines: Vec<Line<'static>>, max_width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::with_capacity(lines.len());
    for line in lines {
        let width = line.width();
        if width <= max_width {
            out.push(line);
            continue;
        }
        // Split the line's spans into chunks that fit within max_width.
        let mut current_spans: Vec<Span<'static>> = Vec::new();
        let mut current_width = 0;
        for span in line.spans {
            let span_width = span.width();
            if current_width + span_width <= max_width || current_spans.is_empty() {
                current_spans.push(span);
                current_width += span_width;
            } else {
                out.push(Line::from(current_spans));
                current_spans = vec![span];
                current_width = span_width;
            }
        }
        if !current_spans.is_empty() {
            out.push(Line::from(current_spans));
        }
    }
    out
}

/// Format content with fenced code block syntax framing.
pub fn format_content_lines(prefix: Span<'static>, content: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut in_code_block = false;
    let mut is_first_line = true;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code_block {
                out.push(Line::from(vec![
                    Span::styled("  └────", Style::default().fg(Color::DarkGray)),
                ]));
                in_code_block = false;
            } else {
                let code_lang = trimmed.trim_start_matches('`').trim();
                let label = if code_lang.is_empty() { "code" } else { code_lang };
                out.push(Line::from(vec![
                    Span::styled(format!("  ┌── [{}]", label), Style::default().fg(Color::DarkGray)),
                ]));
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            out.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
                Span::styled(line.to_string(), Style::default().fg(Color::Green)),
            ]));
        } else if is_first_line {
            out.push(Line::from(vec![
                prefix.clone(),
                Span::raw(line.to_string()),
            ]));
            is_first_line = false;
        } else {
            out.push(Line::from(vec![
                Span::raw(line.to_string()),
            ]));
        }
    }

    if out.is_empty() {
        out.push(Line::from(vec![prefix, Span::raw("")]));
    }
    out
}

/// Helper parsing `<think>...</think>` wrapper tags.
fn parse_thinking_tags(raw: &str) -> (Option<String>, String) {
    if let Some(start) = raw.find("<think>") {
        if let Some(end) = raw.find("</think>") {
            let think_content = raw[start + 7..end].trim().to_string();
            let rest = format!("{}{}", &raw[..start], &raw[end + 8..])
                .trim()
                .to_string();
            return (Some(think_content), rest);
        } else {
            let think_content = raw[start + 7..].trim().to_string();
            return (Some(think_content), String::new());
        }
    }
    (None, raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_thinking_and_content() {
        let mut transcript = Transcript::new();
        transcript.push_user("Why is sky blue?".into());
        transcript.append_token("<think>Rayleigh scattering</think>It scatters blue light.");
        transcript.finish_turn("· ttft 45ms | 38.2 tok/s".into());

        assert_eq!(transcript.nodes.len(), 2);
        let assistant = &transcript.nodes[1];
        assert_eq!(assistant.thinking.as_deref(), Some("Rayleigh scattering"));
        assert_eq!(assistant.content, "It scatters blue light.");
    }

    #[test]
    fn test_toggle_fold() {
        let mut transcript = Transcript::new();
        transcript.push_user("hi".into());
        transcript.append_token("<think>thinking...</think>hello");
        transcript.finish_turn("done".into());

        assert!(transcript.nodes[1].thought_folded);
        transcript.toggle_fold_last_thought();
        assert!(!transcript.nodes[1].thought_folded);
    }

    #[test]
    fn test_tool_call_and_result_rendering() {
        let mut transcript = Transcript::new();
        transcript.push_tool_call("read_file", r#"{"path": "model.rs"}"#);
        transcript.push_tool_result("pub struct Model...".into());

        assert_eq!(transcript.nodes.len(), 2);
        assert_eq!(transcript.nodes[0].role, Role::ToolCall);
        assert_eq!(transcript.nodes[1].role, Role::ToolResult);

        // New rendering: ToolCall = chip header + arg line(s) + footer border,
        // ToolResult = result line + blank separator.
        // {"path": "model.rs"} is 1 line, so: 3 + 2 = 5 lines minimum.
        let lines = transcript.render_lines();
        assert!(lines.len() >= 5, "expected at least 5 lines, got {}", lines.len());
    }

    #[test]
    fn test_format_content_lines_code_blocks() {
        let prefix = Span::raw("assistant: ");
        let content = "Here is code:\n```rust\nfn main() {}\n```\nDone.";
        let lines = format_content_lines(prefix, content);
        assert_eq!(lines.len(), 5);
    }
}
