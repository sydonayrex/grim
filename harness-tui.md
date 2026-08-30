# Harness-Inspired TUI Modernization for GRIM CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the `grim-cli` interactive chat TUI with patterns borrowed from `deepseek-harness`: an extensible slash command registry with autocomplete popup, an input composer with cursor and history navigation, structured message blocks with collapsible `<think>` reasoning traces, and real-time generation sparklines.

**Architecture:** Maintain the two-thread design: UI thread owns the terminal and Ratatui widgets, Worker thread owns Engine and GPU inference communicating across `mpsc` channels. Split UI state into modular components: `Composer` for editing and history, `CommandRegistry` for command descriptors and tab completion, `Transcript` for role-tagged blocks and fold state, and `SpeedHistory` for decode token rate sparklines.

**Tech Stack:** Rust 2021, Ratatui 0.29, Crossterm 0.28, grim-core, grim-engine, grim-format.

---

## Global Constraints

* Language: Rust 2021 edition.
* Dependencies: Use existing workspace crates plus `ratatui` (0.29) and `crossterm` (0.28). Do not introduce heavy markdown engines, webviews, or non-terminal graphics libraries.
* Thread Safety: Never call Engine, GPU, or I/O blocking functions from the UI render thread. All state synchronization must pass across the existing `mpsc` channel.
* Performance Floor: Frame render time must stay under 16ms (60 FPS). Cache rendered text layouts rather than recomputing on every frame.
* Code Style: Every public function, struct, and non-trivial block must have documentation comments explaining contracts and invariants.
* Writing and Punctuation: No em dashes (`—`) or en dashes (`–`) anywhere in comments, docs, or UI labels. Use colons, commas, or parentheses instead.

---

## Project Boundaries

### What Already Exists (Do Not Break)
* `crates/grim-cli/src/tui/diagnostics.rs`: Snapshot formatting helpers (`format_bytes`, `ratio_percent`, `format_ms`, `format_tps`, `acceptance_rate`, `bar`, `DiagnosticsSnapshot`).
* `crates/grim-cli/src/tui/worker.rs`: Worker thread loop, `WorkerCommand`, `WorkerEvent`, `TurnStats`, model loading and streaming generation via `grim_engine`.
* `crates/grim-cli/src/tui/mod.rs`: `cmd_tui` entry point, `TerminalGuard` raw mode cleanup, basic event polling.

### What Needs to Be Added / Updated
* [NEW] `crates/grim-cli/src/tui/composer.rs`: Text input editor with cursor index, horizontal navigation (`Left`, `Right`, `Home`, `End`), word deletion (`Ctrl+W`), and command history ring buffer (`Up`, `Down`).
* [NEW] `crates/grim-cli/src/tui/commands.rs`: Slash command registry with command descriptors, argument validation, and autocomplete candidate search.
* [NEW] `crates/grim-cli/src/tui/transcript.rs`: Structured message nodes (`User`, `Assistant`, `System`, `Error`, `Thinking`, `TurnMetrics`), `<think>` tag extraction and folding toggle (`Tab`/`Space`).
* [NEW] `crates/grim-cli/src/tui/sparkline.rs`: Telemetry history buffer tracking decode tokens-per-second samples for the sidebar sparkline widget.
* [MODIFY] `crates/grim-cli/src/tui/worker.rs`: Support dynamic parameter adjustments (`SetSamplingParams { temperature, top_p }`).
* [MODIFY] `crates/grim-cli/src/tui/mod.rs`: Integrate composer, command autocomplete popup, structured transcript rendering, and sparklines.

### What Should NOT Be Changed (Strict Left/Right Limits)
* Do NOT modify core ROCm/CUDA/CPU backend kernels or tensor math in `crates/grim-backend-*` or `crates/grim-tensor`.
* Do NOT rewrite `grim_engine::Engine` scheduling or KV cache internals.
* Do NOT change existing non-TUI CLI commands (`grim run`, `grim bench`, `grim server`, `grim quant`).
* Do NOT discard the `TerminalGuard` panic hook that restores the terminal.

---

## Task Breakdown

### Task 1: Input Composer with Cursor Navigation and History

**Files:**
* Create: `crates/grim-cli/src/tui/composer.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs:1-30`
* Test: `crates/grim-cli/src/tui/composer.rs` (inline test module)

**Interfaces:**
* Produces:
  * `pub struct Composer`
  * `pub fn Composer::new() -> Self`
  * `pub fn Composer::insert_char(&mut self, c: char)`
  * `pub fn Composer::delete_prev_char(&mut self)`
  * `pub fn Composer::delete_word_back(&mut self)`
  * `pub fn Composer::move_cursor_left(&mut self)`
  * `pub fn Composer::move_cursor_right(&mut self)`
  * `pub fn Composer::move_cursor_home(&mut self)`
  * `pub fn Composer::move_cursor_end(&mut self)`
  * `pub fn Composer::history_prev(&mut self)`
  * `pub fn Composer::history_next(&mut self)`
  * `pub fn Composer::submit(&mut self) -> String`
  * `pub fn Composer::text(&self) -> String`
  * `pub fn Composer::cursor_offset(&self) -> usize`

- [ ] **Step 1: Write the failing tests for Composer**

Create `crates/grim-cli/src/tui/composer.rs` with test cases covering character insertion, left/right navigation, backspace in the middle of a string, `Ctrl+W` word deletion, and history ring navigation.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_movement_and_insert() {
        let mut composer = Composer::new();
        composer.insert_char('h');
        composer.insert_char('l');
        composer.insert_char('o');
        composer.move_cursor_left();
        composer.move_cursor_left();
        composer.insert_char('e');
        composer.insert_char('l');
        assert_eq!(composer.text(), "hello");
        assert_eq!(composer.cursor_offset(), 3);
    }

    #[test]
    fn test_delete_word_back() {
        let mut composer = Composer::new();
        for c in "/model llama3-8b".chars() {
            composer.insert_char(c);
        }
        composer.delete_word_back();
        assert_eq!(composer.text(), "/model ");
    }

    #[test]
    fn test_history_navigation() {
        let mut composer = Composer::new();
        for c in "first prompt".chars() { composer.insert_char(c); }
        let s1 = composer.submit();
        assert_eq!(s1, "first prompt");

        for c in "second prompt".chars() { composer.insert_char(c); }
        let s2 = composer.submit();
        assert_eq!(s2, "second prompt");

        composer.history_prev();
        assert_eq!(composer.text(), "second prompt");
        composer.history_prev();
        assert_eq!(composer.text(), "first prompt");
        composer.history_next();
        assert_eq!(composer.text(), "second prompt");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::composer`
Expected: FAIL with module or struct not found.

- [ ] **Step 3: Implement Composer**

Write the implementation in `crates/grim-cli/src/tui/composer.rs`:

```rust
//! Input composer managing text editing, cursor navigation, and input history.

/// Text composer for terminal chat input.
#[derive(Debug, Clone)]
pub struct Composer {
    /// Internal buffer as Unicode characters for correct slicing.
    chars: Vec<char>,
    /// Character index of the cursor (0 <= cursor <= chars.len()).
    cursor: usize,
    /// Ring buffer of submitted lines.
    history: Vec<String>,
    /// History navigation index. None when typing a new line.
    history_idx: Option<usize>,
    /// Saved draft when user navigates into history.
    draft: String,
    /// Maximum number of lines kept in history.
    max_history: usize,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new()
    }
}

impl Composer {
    /// Create a new empty composer with a default history capacity of 100 lines.
    pub fn new() -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            draft: String::new(),
            max_history: 100,
        }
    }

    /// Current input as a String.
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    /// Current cursor position in characters.
    pub fn cursor_offset(&self) -> usize {
        self.cursor
    }

    /// Check if composer text is empty.
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Clear current text and reset cursor.
    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.history_idx = None;
    }

    /// Set composer text explicitly.
    pub fn set_text(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    /// Insert a single character at current cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
    }

    /// Delete character immediately before the cursor (Backspace).
    pub fn delete_prev_char(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    /// Delete character at the cursor (Delete key).
    pub fn delete_current_char(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    /// Delete word backwards from cursor (Ctrl+W).
    pub fn delete_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut idx = self.cursor;
        while idx > 0 && self.chars[idx - 1].is_whitespace() {
            idx -= 1;
        }
        while idx > 0 && !self.chars[idx - 1].is_whitespace() {
            idx -= 1;
        }
        self.chars.drain(idx..self.cursor);
        self.cursor = idx;
    }

    /// Move cursor left by one character.
    pub fn move_cursor_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Move cursor right by one character.
    pub fn move_cursor_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
    }

    /// Move cursor to the start of the line (Home / Ctrl+A).
    pub fn move_cursor_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to the end of the line (End / Ctrl+E).
    pub fn move_cursor_end(&mut self) {
        self.cursor = self.chars.len();
    }

    /// Navigate to older entry in command history (Up arrow).
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_idx.is_none() {
            self.draft = self.text();
            let last_idx = self.history.len() - 1;
            self.history_idx = Some(last_idx);
            self.set_text(&self.history[last_idx]);
        } else if let Some(idx) = self.history_idx {
            if idx > 0 {
                let next_idx = idx - 1;
                self.history_idx = Some(next_idx);
                self.set_text(&self.history[next_idx]);
            }
        }
    }

    /// Navigate to newer entry in command history (Down arrow).
    pub fn history_next(&mut self) {
        let Some(idx) = self.history_idx else {
            return;
        };
        if idx + 1 < self.history.len() {
            let next_idx = idx + 1;
            self.history_idx = Some(next_idx);
            self.set_text(&self.history[next_idx]);
        } else {
            self.history_idx = None;
            let draft = std::mem::take(&mut self.draft);
            self.set_text(&draft);
        }
    }

    /// Submit current line, adding non-empty strings to history and clearing text.
    pub fn submit(&mut self) -> String {
        let text = self.text();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if self.history.last().map(|s| s.as_str()) != Some(trimmed) {
                self.history.push(trimmed.to_string());
                if self.history.len() > self.max_history {
                    self.history.remove(0);
                }
            }
        }
        self.clear();
        self.draft.clear();
        text
    }
}
```

- [ ] **Step 4: Register module and run tests**

Add `pub mod composer;` to `crates/grim-cli/src/tui/mod.rs`.
Run: `cargo test --package grim-cli --lib tui::composer`
Expected: PASS (all tests succeed).

---

### Task 2: Extensible Slash Command Registry and Autocomplete

**Files:**
* Create: `crates/grim-cli/src/tui/commands.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs`
* Test: `crates/grim-cli/src/tui/commands.rs` (inline test module)

**Interfaces:**
* Produces:
  * `pub struct CommandSpec` with name, hint, description.
  * `pub struct CommandRegistry`
  * `pub fn CommandRegistry::default_commands() -> Self`
  * `pub fn CommandRegistry::register(&mut self, spec: CommandSpec)`
  * `pub fn CommandRegistry::find_completions(&self, prefix: &str) -> Vec<&CommandSpec>`
  * `pub fn CommandRegistry::parse(&self, line: &str) -> Option<ParsedCommand>`

- [ ] **Step 1: Write the failing tests for CommandRegistry**

Create `crates/grim-cli/src/tui/commands.rs` with tests verifying prefix matching, argument splitting, and unknown command handling.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocomplete_candidates() {
        let registry = CommandRegistry::default_commands();
        let matches = registry.find_completions("/m");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "model");

        let all = registry.find_completions("/");
        assert!(all.len() >= 5);
    }

    #[test]
    fn test_parse_arguments() {
        let registry = CommandRegistry::default_commands();
        let cmd = registry.parse("/temp 0.85").unwrap();
        assert_eq!(cmd.name, "temp");
        assert_eq!(cmd.args, "0.85");

        let empty = registry.parse("hello world");
        assert!(empty.is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::commands`
Expected: FAIL with module not found.

- [ ] **Step 3: Implement CommandRegistry**

Write the implementation in `crates/grim-cli/src/tui/commands.rs`:

```rust
//! Slash command descriptors, registry, and autocomplete matching.

/// Metadata descriptor for a slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    /// Command name without leading slash (e.g. "model").
    pub name: &'static str,
    /// Argument hint (e.g. "<name>").
    pub hint: &'static str,
    /// Human-readable description for autocomplete popup.
    pub description: &'static str,
}

/// Parsed command invocation from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCommand {
    /// Command name (lowercase without slash).
    pub name: String,
    /// Arguments string after command name (trimmed).
    pub args: String,
}

/// Registry holding all known slash commands.
#[derive(Debug, Clone)]
pub struct CommandRegistry {
    commands: Vec<CommandSpec>,
}

impl CommandRegistry {
    /// Create registry loaded with standard GRIM commands.
    pub fn default_commands() -> Self {
        let mut reg = Self {
            commands: Vec::new(),
        };
        reg.register(CommandSpec {
            name: "model",
            hint: "[name]",
            description: "List local models or load/hot-swap a model by name",
        });
        reg.register(CommandSpec {
            name: "temp",
            hint: "<value>",
            description: "Set temperature parameter (e.g. /temp 0.7)",
        });
        reg.register(CommandSpec {
            name: "topp",
            hint: "<value>",
            description: "Set top-p nucleus sampling parameter (e.g. /topp 0.9)",
        });
        reg.register(CommandSpec {
            name: "ctx",
            hint: "<limit|auto>",
            description: "Set context token limit override (e.g. /ctx 8192)",
        });
        reg.register(CommandSpec {
            name: "clear",
            hint: "",
            description: "Reset session history and clear transcript",
        });
        reg.register(CommandSpec {
            name: "save",
            hint: "<path>",
            description: "Export current chat transcript to JSONL or text file",
        });
        reg.register(CommandSpec {
            name: "help",
            hint: "",
            description: "Show available commands and keybindings",
        });
        reg.register(CommandSpec {
            name: "exit",
            hint: "",
            description: "Quit GRIM TUI",
        });
        reg
    }

    /// Add a command descriptor to the registry.
    pub fn register(&mut self, spec: CommandSpec) {
        self.commands.push(spec);
    }

    /// List all registered commands.
    pub fn all_commands(&self) -> &[CommandSpec] {
        &self.commands
    }

    /// Find command candidates matching a typed prefix (e.g. "/m" -> ["model"]).
    pub fn find_completions(&self, prefix: &str) -> Vec<&CommandSpec> {
        let query = prefix.strip_prefix('/').unwrap_or(prefix).trim_start();
        self.commands
            .iter()
            .filter(|cmd| cmd.name.starts_with(query))
            .collect()
    }

    /// Parse an input line into a command name and argument string if it starts with '/'.
    pub fn parse(&self, line: &str) -> Option<ParsedCommand> {
        let trimmed = line.trim();
        if !trimmed.starts_with('/') {
            return None;
        }
        let content = trimmed[1..].trim();
        if content.is_empty() {
            return Some(ParsedCommand {
                name: String::new(),
                args: String::new(),
            });
        }
        let (name, args) = match content.split_once(char::is_whitespace) {
            Some((n, a)) => (n.to_lowercase(), a.trim().to_string()),
            None => (content.to_lowercase(), String::new()),
        };
        Some(ParsedCommand { name, args })
    }
}
```

- [ ] **Step 4: Register module and verify tests**

Add `pub mod commands;` to `crates/grim-cli/src/tui/mod.rs`.
Run: `cargo test --package grim-cli --lib tui::commands`
Expected: PASS.

---

### Task 3: Structured Transcript and CoT Reasoning Folding

**Files:**
* Create: `crates/grim-cli/src/tui/transcript.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs`
* Test: `crates/grim-cli/src/tui/transcript.rs` (inline test module)

**Interfaces:**
* Produces:
  * `pub enum Role { User, Assistant, System, Error, Hint }`
  * `pub struct MessageNode` with role, content, thinking trace, folded flag.
  * `pub struct Transcript`
  * `pub fn Transcript::push_user(&mut self, text: String)`
  * `pub fn Transcript::append_token(&mut self, token: &str)`
  * `pub fn Transcript::finish_turn(&mut self, summary_line: String)`
  * `pub fn Transcript::toggle_fold_last_thought(&mut self)`
  * `pub fn Transcript::render_lines(&self) -> Vec<ratatui::text::Line<'static>>`

- [ ] **Step 1: Write failing tests for reasoning extraction and fold toggling**

Create `crates/grim-cli/src/tui/transcript.rs` test module:

```rust
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
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::transcript`
Expected: FAIL with module not found.

- [ ] **Step 3: Implement Transcript with Reasoning Blocks**

Write implementation in `crates/grim-cli/src/tui/transcript.rs`:

```rust
//! Structured chat transcript with role-based styling and reasoning folding.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Message author role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Error,
    Hint,
}

/// One structured node in the chat history.
#[derive(Debug, Clone)]
pub struct MessageNode {
    pub role: Role,
    pub content: String,
    pub thinking: Option<String>,
    pub thought_folded: bool,
    pub turn_stats: Option<String>,
}

/// Structured transcript container.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub nodes: Vec<MessageNode>,
    /// Ongoing streaming buffer for the active turn.
    pub streaming_raw: String,
}

impl Transcript {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            streaming_raw: String::new(),
        }
    }

    /// Clear all transcript messages.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.streaming_raw.clear();
    }

    /// Add a user turn.
    pub fn push_user(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::User,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Add a system message.
    pub fn push_system(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::System,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Add an error notification.
    pub fn push_error(&mut self, text: String) {
        self.nodes.push(MessageNode {
            role: Role::Error,
            content: text,
            thinking: None,
            thought_folded: false,
            turn_stats: None,
        });
    }

    /// Append a token chunk to streaming buffer.
    pub fn append_token(&mut self, token: &str) {
        self.streaming_raw.push_str(token);
    }

    /// Finalize assistant turn, parsing optional `<think>...</think>` tags.
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

    /// Build styled Ratatui Lines for rendering.
    pub fn render_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for node in &self.nodes {
            match node.role {
                Role::User => {
                    lines.push(Line::from(vec![
                        Span::styled("you: ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        Span::raw(node.content.clone()),
                    ]));
                    lines.push(Line::raw(""));
                }
                Role::Assistant => {
                    if let Some(think) = &node.thinking {
                        if node.thought_folded {
                            lines.push(Line::from(vec![
                                Span::styled("▶ [thought collapsed - press Tab/Space to expand]", Style::default().fg(Color::DarkGray)),
                            ]));
                        } else {
                            lines.push(Line::from(vec![
                                Span::styled("▼ [thought]:", Style::default().fg(Color::DarkGray)),
                            ]));
                            for tline in think.lines() {
                                lines.push(Line::from(vec![
                                    Span::styled(format!("  {}", tline), Style::default().fg(Color::DarkGray)),
                                ]));
                            }
                        }
                    }
                    lines.push(Line::from(vec![
                        Span::styled("assistant: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        Span::raw(node.content.clone()),
                    ]));
                    if let Some(stats) = &node.turn_stats {
                        lines.push(Line::from(vec![
                            Span::styled(stats.clone(), Style::default().fg(Color::Yellow)),
                        ]));
                    }
                    lines.push(Line::raw(""));
                }
                Role::System => {
                    lines.push(Line::from(vec![
                        Span::styled(format!("[system] {}", node.content), Style::default().fg(Color::Blue)),
                    ]));
                }
                Role::Error => {
                    lines.push(Line::from(vec![
                        Span::styled(format!("[error] {}", node.content), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                    ]));
                }
                Role::Hint => {
                    lines.push(Line::from(vec![
                        Span::styled(format!("[hint] {}", node.content), Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }
        }

        // Active streaming output
        if !self.streaming_raw.is_empty() {
            let (thinking, content) = parse_thinking_tags(&self.streaming_raw);
            if let Some(think) = thinking {
                lines.push(Line::from(vec![
                    Span::styled("▼ [thinking...]:", Style::default().fg(Color::DarkGray)),
                ]));
                for tline in think.lines() {
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {}", tline), Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }
            if !content.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("assistant: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(content),
                ]));
            }
        }

        lines
    }
}

/// Helper parsing `<think>...</think>` wrapper tags.
fn parse_thinking_tags(raw: &str) -> (Option<String>, String) {
    if let Some(start) = raw.find("<think>") {
        if let Some(end) = raw.find("</think>") {
            let think_content = raw[start + 7..end].trim().to_string();
            let rest = format!("{}{}", &raw[..start], &raw[end + 8..]).trim().to_string();
            return (Some(think_content), rest);
        } else {
            let think_content = raw[start + 7..].trim().to_string();
            return (Some(think_content), String::new());
        }
    }
    (None, raw.to_string())
}
```

- [ ] **Step 4: Register module and verify tests**

Add `pub mod transcript;` to `crates/grim-cli/src/tui/mod.rs`.
Run: `cargo test --package grim-cli --lib tui::transcript`
Expected: PASS.

---

### Task 4: Real-Time Token Generation Sparklines

**Files:**
* Create: `crates/grim-cli/src/tui/sparkline.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs`
* Test: `crates/grim-cli/src/tui/sparkline.rs` (inline test module)

**Interfaces:**
* Produces:
  * `pub struct SpeedHistory`
  * `pub fn SpeedHistory::new(capacity: usize) -> Self`
  * `pub fn SpeedHistory::record(&mut self, tps: u64)`
  * `pub fn SpeedHistory::as_slice(&self) -> &[u64]`

- [ ] **Step 1: Write test for SpeedHistory buffer**

Create `crates/grim-cli/src/tui/sparkline.rs` with tests verifying fixed-capacity push behavior.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_history_capacity() {
        let mut hist = SpeedHistory::new(3);
        hist.record(10);
        hist.record(20);
        hist.record(30);
        hist.record(40);
        assert_eq!(hist.as_slice(), &[20, 30, 40]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::sparkline`
Expected: FAIL with module not found.

- [ ] **Step 3: Implement SpeedHistory**

Write implementation in `crates/grim-cli/src/tui/sparkline.rs`:

```rust
//! Fixed-capacity circular ring tracking generation speeds for sparkline widget.

/// Fixed-capacity buffer storing recent tokens-per-second integer metrics.
#[derive(Debug, Clone)]
pub struct SpeedHistory {
    buffer: Vec<u64>,
    capacity: usize,
}

impl SpeedHistory {
    /// Create new history buffer with a maximum number of data points.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a speed sample in tok/s.
    pub fn record(&mut self, tps: u64) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(tps);
    }

    /// View history as a continuous slice for `ratatui::widgets::Sparkline`.
    pub fn as_slice(&self) -> &[u64] {
        &self.buffer
    }
}
```

- [ ] **Step 4: Register module and verify tests**

Add `pub mod sparkline;` to `crates/grim-cli/src/tui/mod.rs`.
Run: `cargo test --package grim-cli --lib tui::sparkline`
Expected: PASS.

---

### Task 5: Dynamic Sampling Parameters in Worker Channel

**Files:**
* Modify: `crates/grim-cli/src/tui/worker.rs:20-60`
* Modify: `crates/grim-cli/src/tui/worker.rs:145-180`
* Test: `crates/grim-cli/src/tui/worker.rs`

**Interfaces:**
* Modifies:
  * `WorkerCommand::SetSamplingParams { temperature: Option<f32>, top_p: Option<f32> }`
  * Updates sampler instance dynamically during live chat session.

- [ ] **Step 1: Write test for dynamic sampling parameters**

Add unit test in `crates/grim-cli/src/tui/worker.rs` verifying `WorkerCommand::SetSamplingParams` changes sampler parameters.

- [ ] **Step 2: Update WorkerCommand and Worker handler**

In `crates/grim-cli/src/tui/worker.rs`, add the enum variant and handle dynamic update:

```rust
// In WorkerCommand enum:
SetSamplingParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
},

// In Worker::handle:
WorkerCommand::SetSamplingParams { temperature, top_p } => {
    let mut params = SamplingParams::default();
    if let Some(t) = temperature {
        params.temperature = t;
    }
    if let Some(p) = top_p {
        params.top_p = p;
    }
    self.sampler = params.into_sampler(42);
    WorkerOutcome::Ignored
}
```

- [ ] **Step 3: Run worker tests**

Run: `cargo test --package grim-cli --lib tui::worker`
Expected: PASS.

---

### Task 6: Integrate Components into UI Loop and Ratatui Rendering

**Files:**
* Modify: `crates/grim-cli/src/tui/mod.rs`
* Test: `crates/grim-cli/src/tui/mod.rs` (inline test suite)

**Interfaces:**
* Connects:
  * `App::composer: Composer`
  * `App::registry: CommandRegistry`
  * `App::transcript: Transcript`
  * `App::speed_history: SpeedHistory`
  * Render autocomplete popup floating menu when typing `/`.
  * Render sparkline in diagnostics sidebar.
  * Connect keybindings (`Left`, `Right`, `Home`, `End`, `Ctrl+W`, `Tab`, `Space`, `Up`, `Down`).

- [ ] **Step 1: Update App struct and event loop in `mod.rs`**

Refactor `App` in `crates/grim-cli/src/tui/mod.rs` to replace raw `String` input with `Composer`, replace raw `Vec<String>` transcript with `Transcript`, and add `CommandRegistry` and `SpeedHistory`.

- [ ] **Step 2: Update `ui()` render function**

Implement layout:
1. Upper area: Split into Chat transcript and Diagnostics sidebar (with Sparkline widget when data exists).
2. Autocomplete overlay: If composer text starts with `/` and is not submitted, render floating candidate list.
3. Lower area: Composer input block showing cursor position via `f.set_cursor_position`.

- [ ] **Step 3: Run full CLI test suite**

Run: `cargo test --package grim-cli`
Expected: PASS (all unit tests and integration tests pass).

---

## Verification Plan

### Automated Tests
* Run unit tests for all components:
  ```bash
  cargo test --package grim-cli --lib tui
  ```
* Run full workspace test suite:
  ```bash
  cargo test --package grim-cli
  ```

### Manual Verification
1. Launch interactive TUI:
   ```bash
   cargo run --bin grim-cli -- tui
   ```
2. Test cursor movements: Type `hello`, press `Left` twice, type `X`, verify line becomes `helXlo`. Press `Ctrl+W`, verify word deleted.
3. Test slash autocomplete: Type `/m`, verify `/model` candidate popup appears. Press `Tab` to complete.
4. Test reasoning trace: Chat with a model emitting `<think>`, verify `[thought collapsed]` line appears. Press `Tab` or `Space` to expand/collapse.
5. Test history: Submit two prompts, press `Up` twice to recall first prompt.
6. Verify clean exit: Type `/exit` or press `Ctrl+C` and ensure shell raw mode is completely restored without terminal artifacts.
