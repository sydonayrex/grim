//! Structured chat transcript with role-based styling and reasoning folding.
//!
//! Separates message nodes by role, extracts `<think>...</think>` CoT traces,
//! and renders styled Ratatui Lines with folding capability.

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
}

/// Structured transcript container managing message history and active streaming state.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    /// List of completed message nodes.
    pub nodes: Vec<MessageNode>,
    /// Ongoing streaming buffer for the active turn.
    pub streaming_raw: String,
}

impl Transcript {
    /// Create a new empty transcript.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            streaming_raw: String::new(),
        }
    }

    /// Clear all transcript messages and reset streaming buffer.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.streaming_raw.clear();
    }

    /// Add a user message node.
    pub fn push_user(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::User,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Add a system notification message.
    pub fn push_system(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::System,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Add an error notification message.
    pub fn push_error(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::Error,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Add a tool call invocation message.
    pub fn push_tool_call(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::ToolCall,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Add a tool result output message.
    pub fn push_tool_result(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::ToolResult,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Append a token chunk to the active turn streaming buffer.
    pub fn append_token(&mut self, token: &str) {
        self.streaming_raw.push_str(token);
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
        });
        self.streaming_raw.clear();
    }

    /// Toggle collapsed state of the latest assistant reasoning block.
    pub fn toggle_fold_last_thought(&mut self) {
        if let Some(node) = self.nodes.iter_mut().rev().find(|n| n.thinking.is_some()) {
            node.thought_folded = !node.thought_folded;
        }
    }

    /// Build styled Ratatui Lines for rendering in the main chat viewport.
    pub fn render_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for node in &self.nodes {
            match node.role {
                Role::User => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "you: ",
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(node.content.clone()),
                    ]));
                    lines.push(Line::raw(""));
                }
                Role::Assistant => {
                    if let Some(think) = &node.thinking {
                        if node.thought_folded {
                            lines.push(Line::from(vec![Span::styled(
                                "▶ [thought collapsed - press Tab/Space to expand]",
                                Style::default().fg(Color::DarkGray),
                            )]));
                        } else {
                            lines.push(Line::from(vec![Span::styled(
                                "▼ [thought]:",
                                Style::default().fg(Color::DarkGray),
                            )]));
                            for tline in think.lines() {
                                lines.push(Line::from(vec![Span::styled(
                                    format!("  {}", tline),
                                    Style::default().fg(Color::DarkGray),
                                )]));
                            }
                        }
                    }
                    let prefix = Span::styled(
                        "assistant: ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    );
                    lines.extend(format_content_lines(prefix, &node.content));
                    if let Some(stats) = &node.turn_stats {
                        lines.push(Line::from(vec![Span::styled(
                            stats.clone(),
                            Style::default().fg(Color::Yellow),
                        )]));
                    }
                    lines.push(Line::raw(""));
                }
                Role::ToolCall => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "⚙ [tool call]: ",
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(node.content.clone(), Style::default().fg(Color::LightMagenta)),
                    ]));
                }
                Role::ToolResult => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "✓ [tool result]: ",
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(node.content.clone(), Style::default().fg(Color::DarkGray)),
                    ]));
                }
                Role::System => {
                    lines.push(Line::from(vec![Span::styled(
                        format!("[system] {}", node.content),
                        Style::default().fg(Color::Blue),
                    )]));
                }
                Role::Error => {
                    lines.push(Line::from(vec![Span::styled(
                        format!("[error] {}", node.content),
                        Style::default()
                            .fg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    )]));
                }
                Role::Hint => {
                    lines.push(Line::from(vec![Span::styled(
                        format!("[hint] {}", node.content),
                        Style::default().fg(Color::DarkGray),
                    )]));
                }
            }
        }

        // Active streaming output
        if !self.streaming_raw.is_empty() {
            let (thinking, content) = parse_thinking_tags(&self.streaming_raw);
            if let Some(think) = thinking {
                lines.push(Line::from(vec![Span::styled(
                    "▼ [thinking...]:",
                    Style::default().fg(Color::DarkGray),
                )]));
                for tline in think.lines() {
                    lines.push(Line::from(vec![Span::styled(
                        format!("  {}", tline),
                        Style::default().fg(Color::DarkGray),
                    )]));
                }
            }
            if !content.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled(
                        "assistant: ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(content),
                ]));
            }
        }

        lines
    }
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
        transcript.push_tool_call("read_file(path: \"model.rs\")".into());
        transcript.push_tool_result("pub struct Model...".into());

        assert_eq!(transcript.nodes.len(), 2);
        assert_eq!(transcript.nodes[0].role, Role::ToolCall);
        assert_eq!(transcript.nodes[1].role, Role::ToolResult);

        let lines = transcript.render_lines();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_format_content_lines_code_blocks() {
        let prefix = Span::raw("assistant: ");
        let content = "Here is code:\n```rust\nfn main() {}\n```\nDone.";
        let lines = format_content_lines(prefix, content);
        assert_eq!(lines.len(), 5);
    }
}
