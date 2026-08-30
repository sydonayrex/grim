//! grim tui: Ratatui chat interface over the in-process engine.
//!
//! Two threads: the UI thread owns the terminal, input composer, and ratatui loop;
//! the worker thread owns Engine, tokenizer, and sampler. The UI thread sends
//! `WorkerCommand`s and drains `WorkerEvent`s over `std::sync::mpsc` channels.
//! GPU and model code runs only on the worker.

use std::io::IsTerminal;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use grim_core::error::{Error, Result};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Sparkline, Wrap};

/// Slash command descriptors and registry.
pub mod commands;

/// Input composer with cursor navigation and history ring buffer.
pub mod composer;

/// Diagnostics formatting helpers for the TUI.
pub mod diagnostics;

/// External editor ($EDITOR / $VISUAL) integration.
pub mod editor;

/// Stateless fuzzy matching for autocomplete.
pub mod fuzzy;

/// File path completion for @ triggers.
pub mod file_complete;

/// Emacs kill-ring buffer.
pub mod kill_ring;

/// Frecency-based file ranking for autocomplete.
pub mod frecency;

/// Speed history ring buffer for sparklines.
pub mod sparkline;

/// Structured transcript with reasoning trace folding.
pub mod transcript;

/// Keyboard-navigable selection menu.
pub mod select_list;

/// Constrained VStack/HStack/ScrollView layout engine.
pub mod layout;

/// Toast notification system.
pub mod toast;

/// 16ms render-throttle scheduler.
pub mod throttle;

/// Generic bounded undo stack.
pub mod undo_stack;

/// Worker thread and channel protocol.
pub mod worker;

pub use commands::{CommandRegistry, CommandSpec, ParsedCommand};
pub use composer::Composer;
pub use file_complete::{FileSuggestion, apply_file_completion, extract_at_prefix, get_file_suggestions};
pub use fuzzy::{FuzzyMatch, fuzzy_filter, fuzzy_match};
pub use frecency::Frecency;
pub use kill_ring::{KillPushOpts, KillRing};
pub use layout::{Basis, HStack, LayoutNode, ScrollView, ScrollViewOptions, StackEntry, StackOptions, VStack};
pub use select_list::{SelectAction, SelectItem, SelectList, SelectListTheme};
pub use sparkline::SpeedHistory;
pub use toast::{Toast, ToastVariant, render_toast};
pub use throttle::{RenderScheduler, MIN_FRAME_INTERVAL};
pub use transcript::{MessageNode, Role, Transcript};
pub use undo_stack::UndoStack;
pub use worker::{DiagnosticsSnapshot, TurnStats, WorkerCommand, WorkerEvent, WorkerParams};

/// Slash commands parsed from the input line.
#[derive(Debug, PartialEq)]
pub enum SlashCommand {
    Model(Option<String>),
    Temp(Option<f32>),
    TopP(Option<f32>),
    Ctx(Option<u64>),
    System(Option<String>),
    Load(String),
    Save(String),
    Edit,
    ShowEditor,
    Exit,
    Clear,
    Help,
    NotACommand,
    Unknown(String),
}

/// Parse a single input line into a slash command or plain text.
pub fn parse_slash_command(input: &str) -> SlashCommand {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return SlashCommand::NotACommand;
    }
    let rest = trimmed[1..].trim();
    if rest.is_empty() {
        return SlashCommand::Unknown(String::new());
    }
    let (first_word, after) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    match first_word.to_lowercase().as_str() {
        "exit" | "quit" => SlashCommand::Exit,
        "help" => SlashCommand::Help,
        "clear" => SlashCommand::Clear,
        "model" if after.is_empty() => SlashCommand::Model(None),
        "model" => SlashCommand::Model(Some(after.trim().to_string())),
        "temp" => SlashCommand::Temp(after.trim().parse::<f32>().ok()),
        "topp" => SlashCommand::TopP(after.trim().parse::<f32>().ok()),
        "ctx" if after.trim() == "auto" || after.trim().is_empty() => SlashCommand::Ctx(None),
        "ctx" => match after.trim().parse::<u64>() {
            Ok(n) => SlashCommand::Ctx(Some(n)),
            Err(_) => SlashCommand::Unknown(first_word.to_string()),
        },
        "system" if after.is_empty() => SlashCommand::System(None),
        "system" => SlashCommand::System(Some(after.trim().to_string())),
        "load" => SlashCommand::Load(after.trim().to_string()),
        "save" => SlashCommand::Save(after.trim().to_string()),
        "edit" => SlashCommand::Edit,
        "editor" => SlashCommand::ShowEditor,
        _ => SlashCommand::Unknown(first_word.to_string()),
    }
}

/// Parse the content of the F3 context-limit override input.
#[derive(Debug, PartialEq, Eq)]
pub enum CtxOverride {
    Apply(u64),
    Auto,
    Invalid,
}

/// Parse a context-limit override: empty / whitespace-only -> Auto; a
/// non-negative integer -> Apply; anything else -> Invalid.
pub fn parse_ctx_override(input: &str) -> CtxOverride {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return CtxOverride::Auto;
    }
    match trimmed.parse::<u64>() {
        Ok(n) => CtxOverride::Apply(n),
        Err(_) => CtxOverride::Invalid,
    }
}

/// Character jump direction for the composer overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpMode {
    None,
    Forward,
    Backward,
}

/// Shape of the input bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Chat,
    CtxOverride,
    ModelPicker { selected: usize },
    /// Fuzzy-searchable command palette overlay (borrowed from opencode-dev).
    CommandPalette { selected: usize },
    /// Interactive session browser overlay.
    SessionBrowser { selected: usize },
}

/// State driving the terminal render loop.
pub struct App {
    pub composer: Composer,
    pub transcript: Transcript,
    pub registry: CommandRegistry,
    pub speed_history: SpeedHistory,
    pub snap: DiagnosticsSnapshot,
    pub cmd_tx: Sender<WorkerCommand>,
    pub messages: Vec<grim_format::ChatMessage>,
    pub system_prompt: Option<String>,
    pub should_quit: bool,
    pub generating: bool,
    pub scroll_offset: usize,
    pub show_sidebar: bool,
    pub input_mode: InputMode,
    pub selected_completion: usize,
    pub jump_mode: JumpMode,
    /// Current toast notification, or None when idle.
    pub toast: Option<Toast>,
    /// Frecency tracker for file autocomplete ranking.
    pub frecency: Frecency,
    /// Desktop notification sent on generation complete (dedup flag).
    pub generation_complete_notified: bool,
}

impl App {
    pub fn new(cmd_tx: Sender<WorkerCommand>) -> Self {
        Self {
            composer: Composer::new(),
            transcript: Transcript::new(),
            registry: CommandRegistry::default_commands(),
            speed_history: SpeedHistory::new(24),
            snap: DiagnosticsSnapshot::default(),
            cmd_tx,
            messages: Vec::new(),
            system_prompt: None,
            should_quit: false,
            generating: false,
            scroll_offset: 0,
            show_sidebar: true,
            input_mode: InputMode::Chat,
            selected_completion: 0,
            jump_mode: JumpMode::None,
            toast: None,
            frecency: Frecency::new(),
            generation_complete_notified: false,
        }
    }

    /// Show a toast notification, replacing any existing one.
    pub fn show_toast(&mut self, toast: Toast) {
        self.toast = Some(toast);
    }

    /// Clear the current toast if it has expired. Returns true if a toast was
    /// cleared (so the caller can request a render).
    pub fn expire_toast(&mut self) -> bool {
        if let Some(t) = &self.toast {
            if t.is_expired() {
                self.toast = None;
                return true;
            }
        }
        false
    }

    /// Apply a worker event to app state.
    pub fn handle_event(&mut self, evt: WorkerEvent) {
        match evt {
            WorkerEvent::Token { text } => {
                self.transcript.append_token(&text);
            }
            WorkerEvent::TurnComplete { stats } => {
                if let Some(tps) = stats.decode_tps {
                    self.speed_history.record(tps as u64);
                }
                let prefill = diagnostics::format_ms(stats.prefill_ms);
                let decode = diagnostics::format_tps(stats.decode_tps);
                let summary = format!(
                    "· enc {:.1} ms | ttft {} | {} | {} tok{}",
                    stats.encode_ms,
                    prefill,
                    decode,
                    stats.tokens_generated,
                    if stats.cancelled { " (cancelled)" } else { "" }
                );
                self.transcript.finish_turn(summary);
                self.generating = false;
                // Desktop notification on completion (once per turn).
                if !self.generation_complete_notified {
                    self.generation_complete_notified = true;
                    send_desktop_notification("GRIM", "Generation complete");
                }
                self.show_toast(Toast::success(format!(
                    "Generation complete — {} tok {}",
                    stats.tokens_generated, decode
                )));
            }
            WorkerEvent::Diagnostics { snap } => {
                self.snap = snap;
            }
            WorkerEvent::Error { message } => {
                self.transcript.push_error(message);
                self.generating = false;
            }
            WorkerEvent::ModelLoadStarted { name } => {
                self.snap.loading = true;
                self.snap.model_name = Some(name.clone());
                self.transcript.push_system(format!("loading {name}"));
            }
            WorkerEvent::ModelLoadOk {
                name,
                quant,
                context_length: _,
                strategy,
            } => {
                self.snap.loading = false;
                self.snap.model_name = Some(name.clone());
                self.snap.quant = quant;
                self.snap.strategy = Some(strategy.clone());
                self.transcript
                    .push_system(format!("model loaded: {name} ({strategy})"));
                self.messages.clear();
            }
            WorkerEvent::ModelLoadFailed { name, error } => {
                self.snap.loading = false;
                self.transcript
                    .push_system(format!("model '{name}' failed to load: {error}"));
            }
        }
    }

    /// Handle a single key press.
    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.input_mode {
            InputMode::CtxOverride => self.handle_ctx_key(key),
            InputMode::Chat => self.handle_chat_key(key),
            InputMode::ModelPicker { selected } => self.handle_model_picker_key(key, selected),
            InputMode::CommandPalette { .. } => self.handle_palette_key(key),
            InputMode::SessionBrowser { .. } => self.handle_session_browser_key(key),
        }
    }

    fn handle_model_picker_key(&mut self, key: KeyEvent, selected: usize) {
        let models = grim_core::catalog::list_local_models();
        match key.code {
            KeyCode::Up => {
                self.input_mode = InputMode::ModelPicker {
                    selected: selected.saturating_sub(1),
                };
            }
            KeyCode::Down => {
                let next = if models.is_empty() {
                    0
                } else {
                    (selected + 1).min(models.len() - 1)
                };
                self.input_mode = InputMode::ModelPicker { selected: next };
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Chat;
                if let Some(entry) = models.get(selected) {
                    let _ = self
                        .cmd_tx
                        .send(WorkerCommand::LoadModel { name: entry.name.clone() });
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Chat;
            }
            _ => {}
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        let is_ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let is_alt = key.modifiers.contains(KeyModifiers::ALT);
        let is_shift = key.modifiers.contains(KeyModifiers::SHIFT);

        // Jump mode: awaiting a target character after Alt+F or Alt+B.
        if self.jump_mode != JumpMode::None {
            // Cancel if the jump hotkey is pressed again.
            if (is_alt && matches!(key.code, KeyCode::Char('f') | KeyCode::Char('b')))
                || (is_ctrl && matches!(key.code, KeyCode::Char('f') | KeyCode::Char('b')))
            {
                self.jump_mode = JumpMode::None;
                return;
            }
            if let KeyCode::Char(c) = key.code {
                let mode = self.jump_mode;
                self.jump_mode = JumpMode::None;
                if mode == JumpMode::Forward {
                    self.composer.jump_forward(c);
                } else {
                    self.composer.jump_backward(c);
                }
                return;
            }
            // Control character while in jump mode: cancel and fall through.
            self.jump_mode = JumpMode::None;
        }

        match key.code {
            KeyCode::Enter if is_alt || is_shift => {
                self.composer.insert_char('\n');
            }
            KeyCode::Char('j') if is_ctrl => {
                self.composer.insert_char('\n');
            }
            KeyCode::Enter => {
                let text = self.composer.submit();
                self.submit_chat(&text);
            }
            KeyCode::Backspace => {
                self.composer.delete_prev_char();
            }
            KeyCode::Delete => {
                self.composer.delete_current_char();
            }
            KeyCode::Left => {
                self.composer.move_cursor_left();
            }
            KeyCode::Right => {
                self.composer.move_cursor_right();
            }
            KeyCode::Home => {
                self.composer.move_cursor_home();
            }
            KeyCode::End => {
                self.composer.move_cursor_end();
            }
            KeyCode::Up => {
                self.composer.move_cursor_up();
            }
            KeyCode::Down => {
                self.composer.move_cursor_down();
            }
            KeyCode::Char('a') if is_ctrl => {
                self.composer.move_cursor_home();
            }
            KeyCode::Char('e') if is_ctrl => {
                self.composer.move_cursor_end();
            }
            KeyCode::Char('w') if is_ctrl => {
                self.composer.delete_word_back();
            }
            KeyCode::Char('u') if is_ctrl => {
                self.composer.clear();
            }
            KeyCode::Char('k') if is_ctrl => {
                self.composer.kill_to_end();
            }
            KeyCode::Char('y') if is_ctrl => {
                self.composer.yank();
            }
            KeyCode::Char('y') if is_alt => {
                self.composer.yank_pop();
            }
            KeyCode::Char('z') if is_ctrl => {
                self.composer.undo();
            }
            KeyCode::Char('f') if is_alt => {
                self.jump_mode = JumpMode::Forward;
            }
            KeyCode::Char('b') if is_alt => {
                self.jump_mode = JumpMode::Backward;
            }
            KeyCode::Char('p') if is_ctrl => {
                // Command palette: fuzzy-searchable command list.
                self.input_mode = InputMode::CommandPalette { selected: 0 };
            }
            KeyCode::Char('o') if is_ctrl => {
                // Session browser: interactive session list.
                self.input_mode = InputMode::SessionBrowser { selected: 0 };
            }
            KeyCode::Char(c) => {
                self.composer.insert_char(c);
            }
            KeyCode::Tab => {
                let text = self.composer.text();
                let cursor = self.composer.cursor_offset();
                // Prefer @file completion when an @ trigger is active near the cursor.
                if let Some((start, prefix)) =
                    crate::tui::file_complete::extract_at_prefix(&text, cursor)
                {
                    let base = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                    let after_at = prefix.trim_start_matches('@');
                    let suggestions =
                        crate::tui::file_complete::get_file_suggestions_ranked(
                            after_at,
                            &base,
                            50,
                            &self.frecency,
                        );
                    if !suggestions.is_empty() {
                        // Record frecency for the selected file.
                        let selected = &suggestions[0];
                        self.frecency.record_open(&selected.value);
                        crate::tui::file_complete::apply_file_completion(
                            &mut self.composer,
                            start,
                            selected,
                        );
                        return;
                    }
                }
                if text.starts_with('/') {
                    // Use fuzzy matching so typos like "/mdoel" still resolve to "/model".
                    let query = text.trim_start_matches('/').trim_start();
                    let all_items: Vec<crate::tui::select_list::SelectItem> = self
                        .registry
                        .all_commands()
                        .iter()
                        .map(|spec| crate::tui::select_list::SelectItem {
                            value: spec.name.to_string(),
                            label: spec.name.to_string(),
                            description: Some(spec.description.to_string()),
                        })
                        .collect();
                    let mut menu = crate::tui::select_list::SelectList::new(
                        all_items,
                        16,
                        crate::tui::select_list::SelectListTheme::default(),
                    );
                    if query.is_empty() {
                        menu.set_filter("");
                    } else {
                        menu.set_filter(query);
                    }
                    if menu.filtered_len() > 0 {
                        if let Some(item) = menu.selected() {
                            self.composer.set_text(&format!("/{} ", item.value));
                        }
                    }
                } else {
                    self.transcript.toggle_fold_last_thought();
                }
            }
            KeyCode::Esc => {
                let _ = self.cmd_tx.send(WorkerCommand::Cancel);
            }
            KeyCode::PageUp => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            KeyCode::PageDown => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
            }
            KeyCode::F(2) => {
                self.show_sidebar = !self.show_sidebar;
            }
            KeyCode::F(3) => {
                self.composer.clear();
                self.input_mode = InputMode::CtxOverride;
            }
            KeyCode::F(4) => {
                self.input_mode = InputMode::ModelPicker { selected: 0 };
            }
            _ => {}
        }
    }

    fn handle_palette_key(&mut self, key: KeyEvent) {
        let (filtered_count, selected) = match &self.input_mode {
            InputMode::CommandPalette { selected } => {
                let filtered = self.palette_filtered_commands();
                (filtered.len(), *selected)
            }
            _ => return,
        };
        match key.code {
            KeyCode::Up => {
                let new_sel = selected.saturating_sub(1);
                self.input_mode = InputMode::CommandPalette { selected: new_sel };
            }
            KeyCode::Down => {
                let new_sel = if filtered_count == 0 {
                    0
                } else {
                    (selected + 1).min(filtered_count - 1)
                };
                self.input_mode = InputMode::CommandPalette { selected: new_sel };
            }
            KeyCode::Enter => {
                let filtered = self.palette_filtered_commands();
                if let Some(cmd) = filtered.get(selected) {
                    self.input_mode = InputMode::Chat;
                    self.composer.set_text(&format!("/{} ", cmd.name));
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Chat;
            }
            KeyCode::Backspace => {
                let mut filter = self.composer.text().trim_start_matches('/').to_string();
                filter.pop();
                if filter.is_empty() {
                    self.composer.clear();
                    self.input_mode = InputMode::Chat;
                } else {
                    self.composer.set_text(&format!("/{filter}"));
                    self.input_mode = InputMode::CommandPalette { selected: 0 };
                }
            }
            KeyCode::Char(c) => {
                let current = self.composer.text();
                let filter = current.trim_start_matches('/');
                self.composer.set_text(&format!("/{filter}{c}"));
                self.input_mode = InputMode::CommandPalette { selected: 0 };
            }
            _ => {}
        }
    }

    fn handle_session_browser_key(&mut self, key: KeyEvent) {
        let sessions = self.discover_session_files();
        let selected = match &self.input_mode {
            InputMode::SessionBrowser { selected } => *selected,
            _ => return,
        };
        match key.code {
            KeyCode::Up => {
                self.input_mode = InputMode::SessionBrowser {
                    selected: selected.saturating_sub(1),
                };
            }
            KeyCode::Down => {
                let new_sel = if sessions.is_empty() {
                    0
                } else {
                    (selected + 1).min(sessions.len() - 1)
                };
                self.input_mode = InputMode::SessionBrowser { selected: new_sel };
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Chat;
                if let Some(path) = sessions.get(selected) {
                    self.submit_chat(&format!("/load {path}"));
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Chat;
            }
            _ => {}
        }
    }

    /// Return commands filtered by the current palette query.
    fn palette_filtered_commands(&self) -> Vec<crate::tui::commands::CommandSpec> {
        let query = self
            .composer
            .text()
            .trim_start_matches('/')
            .trim()
            .to_lowercase();
        let all = self.registry.all_commands();
        if query.is_empty() {
            return all.to_vec();
        }
        let mut scored: Vec<(i32, crate::tui::commands::CommandSpec)> = all
            .iter()
            .filter_map(|spec| {
                let name_lower = spec.name.to_lowercase();
                crate::tui::fuzzy::fuzzy_match(&query, &name_lower)
                    .map(|fm| (fm.score, spec.clone()))
            })
            .collect();
        scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        scored.into_iter().map(|(_, spec)| spec).collect()
    }

    /// Discover session files in the current directory.
    fn discover_session_files(&self) -> Vec<String> {
        let base =
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let mut sessions = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    if let Some(s) = path.to_str() {
                        sessions.push(s.to_string());
                    }
                }
            }
        }
        sessions.sort();
        sessions
    }

    fn submit_chat(&mut self, text: &str) {
        match parse_slash_command(text) {
            SlashCommand::Exit => {
                self.should_quit = true;
            }
            SlashCommand::Help => {
                let mut help_msg = String::from("Available commands and shortcuts:\n");
                for spec in self.registry.all_commands() {
                    let hint_str = if spec.hint.is_empty() {
                        String::new()
                    } else {
                        format!(" {}", spec.hint)
                    };
                    help_msg.push_str(&format!(
                        "  /{:<18} {}\n",
                        format!("{}{}", spec.name, hint_str),
                        spec.description
                    ));
                }
                help_msg.push_str(
                    "\nShortcuts:\n  Tab: Autocomplete command or @file / Toggle reasoning\n  F2: Toggle sidebar | F3: Context override | Esc: Cancel turn\n  Ctrl+A / Ctrl+E: Line start/end | Ctrl+W: Delete word | Ctrl+K: Kill | Ctrl+Y: Yank | Alt+Y: Yank-pop | Ctrl+Z: Undo | Alt+F/B: Jump",
                );
                self.transcript.push_system(help_msg);
            }
            SlashCommand::Clear => {
                self.transcript.clear();
                self.messages.clear();
                self.speed_history.clear();
                self.scroll_offset = 0;
            }
            SlashCommand::Model(None) => {
                let list = grim_core::catalog::list_local_models();
                if list.is_empty() {
                    self.transcript
                        .push_system("no local models found".into());
                } else {
                    self.input_mode = InputMode::ModelPicker { selected: 0 };
                }
            }
            SlashCommand::Model(Some(name)) => {
                let _ = self.cmd_tx.send(WorkerCommand::LoadModel { name });
            }
            SlashCommand::Temp(Some(val)) => {
                let _ = self.cmd_tx.send(WorkerCommand::SetSamplingParams {
                    temperature: Some(val),
                    top_p: None,
                });
                self.transcript
                    .push_system(format!("temperature set to {val:.2}"));
            }
            SlashCommand::Temp(None) => {
                self.transcript
                    .push_error("invalid temperature: use /temp <float> (e.g. /temp 0.7)".into());
            }
            SlashCommand::TopP(Some(val)) => {
                let _ = self.cmd_tx.send(WorkerCommand::SetSamplingParams {
                    temperature: None,
                    top_p: Some(val),
                });
                self.transcript
                    .push_system(format!("top-p set to {val:.2}"));
            }
            SlashCommand::TopP(None) => {
                self.transcript
                    .push_error("invalid top-p: use /topp <float> (e.g. /topp 0.9)".into());
            }
            SlashCommand::Ctx(opt) => {
                let _ = self
                    .cmd_tx
                    .send(WorkerCommand::SetContextLimit { limit: opt });
                match opt {
                    Some(n) => self.transcript.push_system(format!("ctx limit: {n}")),
                    None => self.transcript.push_system("ctx limit: auto".into()),
                }
            }
            SlashCommand::System(Some(prompt)) => {
                self.system_prompt = Some(prompt.clone());
                // Prepend or replace system message at head of conversation
                if let Some(first) = self.messages.first_mut() {
                    if first.role == "system" {
                        first.content = prompt.clone();
                    } else {
                        self.messages.insert(0, grim_format::ChatMessage {
                            role: "system".to_string(),
                            content: prompt.clone(),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                } else {
                    self.messages.push(grim_format::ChatMessage {
                        role: "system".to_string(),
                        content: prompt.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
                self.transcript
                    .push_system(format!("system prompt updated: \"{prompt}\""));
            }
            SlashCommand::System(None) => {
                if let Some(sys) = &self.system_prompt {
                    self.transcript
                        .push_system(format!("current system prompt: \"{sys}\""));
                } else {
                    self.transcript
                        .push_system("no system prompt set (use /system <prompt>)".into());
                }
            }
            SlashCommand::Load(path) => {
                if path.is_empty() {
                    self.transcript
                        .push_error("specify file path: /load <filename>".into());
                } else {
                    match import_transcript(&path) {
                        Ok((nodes, chat_msgs)) => {
                            let count = nodes.len();
                            self.transcript.nodes = nodes;
                            self.messages = chat_msgs;
                            self.transcript.push_system(format!(
                                "loaded {count} message nodes from {path}"
                            ));
                        }
                        Err(e) => self
                            .transcript
                            .push_error(format!("failed to load transcript: {e}")),
                    }
                }
            }
            SlashCommand::Save(path) => {
                if path.is_empty() {
                    self.transcript
                        .push_error("specify export path: /save <filename>".into());
                } else {
                    match export_transcript(&self.transcript, &path) {
                        Ok(count) => self.transcript.push_system(format!(
                            "saved {count} message nodes to {path}"
                        )),
                        Err(e) => self
                            .transcript
                            .push_error(format!("failed to save transcript: {e}")),
                    }
                }
            }
            SlashCommand::Edit => {
                let draft = self.composer.text();
                match crate::tui::editor::open_editor(&draft) {
                    Ok(Some(edited)) => {
                        self.composer.set_text(&edited);
                        self.show_toast(Toast::success("Editor content loaded"));
                    }
                    Ok(None) => {
                        self.show_toast(Toast::warning(
                            "No $VISUAL/$EDITOR set; export EDITOR to use /edit",
                        ));
                    }
                    Err(e) => {
                        self.show_toast(Toast::error(format!("Editor failed: {e}")));
                    }
                }
            }
            SlashCommand::ShowEditor => {
                let editor = std::env::var("VISUAL")
                    .or_else(|_| std::env::var("EDITOR"))
                    .unwrap_or_else(|_| "(not set)".to_string());
                self.show_toast(Toast::info(format!("Editor: {editor}")));
            }
            SlashCommand::NotACommand => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return;
                }
                if self.generating {
                    self.transcript
                        .push_system("generation in progress; Esc to cancel first".into());
                    return;
                }
                self.messages.push(grim_format::ChatMessage {
                    role: "user".to_string(),
                    content: trimmed.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
                self.transcript.push_user(trimmed.to_string());
                self.generating = true;
                self.generation_complete_notified = false;
                let _ = self.cmd_tx.send(WorkerCommand::Generate {
                    messages: self.messages.clone(),
                });
            }
            SlashCommand::Unknown(word) => {
                self.transcript
                    .push_error(format!("unknown command: /{word} (try /help)"));
            }
        }
    }

    fn handle_ctx_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                let text = self.composer.submit();
                self.input_mode = InputMode::Chat;
                match parse_ctx_override(&text) {
                    CtxOverride::Auto => {
                        let _ = self
                            .cmd_tx
                            .send(WorkerCommand::SetContextLimit { limit: None });
                        self.transcript.push_system("ctx limit: auto".into());
                    }
                    CtxOverride::Apply(n) => {
                        let _ = self
                            .cmd_tx
                            .send(WorkerCommand::SetContextLimit { limit: Some(n) });
                        self.transcript.push_system(format!("ctx limit: {n}"));
                    }
                    CtxOverride::Invalid => {
                        self.transcript
                            .push_system("enter a number or empty for auto".into());
                    }
                }
            }
            KeyCode::Backspace => {
                self.composer.delete_prev_char();
            }
            KeyCode::Char(c) => {
                self.composer.insert_char(c);
            }
            KeyCode::Esc => {
                self.composer.clear();
                self.input_mode = InputMode::Chat;
            }
            _ => {}
        }
    }
}

/// Helper exporting transcript nodes to a text file or JSONL.
fn export_transcript(transcript: &Transcript, path: &str) -> std::io::Result<usize> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let mut count = 0;
    if path.ends_with(".jsonl") {
        for node in &transcript.nodes {
            let role_str = match node.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::ToolCall => "tool_call",
                Role::ToolResult => "tool_result",
                Role::System => "system",
                Role::Error => "error",
                Role::Hint => "hint",
            };
            let obj = serde_json::json!({
                "role": role_str,
                "content": node.content,
                "thinking": node.thinking,
                "stats": node.turn_stats,
            });
            writeln!(file, "{}", obj)?;
            count += 1;
        }
    } else {
        for node in &transcript.nodes {
            let role_str = match node.role {
                Role::User => "USER",
                Role::Assistant => "ASSISTANT",
                Role::ToolCall => "TOOL_CALL",
                Role::ToolResult => "TOOL_RESULT",
                Role::System => "SYSTEM",
                Role::Error => "ERROR",
                Role::Hint => "HINT",
            };
            writeln!(file, "=== {} ===", role_str)?;
            if let Some(think) = &node.thinking {
                writeln!(file, "[THINKING]\n{}\n[/THINKING]", think)?;
            }
            writeln!(file, "{}\n", node.content)?;
            if let Some(stats) = &node.turn_stats {
                writeln!(file, "{}\n", stats)?;
            }
            count += 1;
        }
    }
    Ok(count)
}

/// Helper importing transcript nodes and chat messages from JSONL session file.
fn import_transcript(
    path: &str,
) -> std::io::Result<(Vec<MessageNode>, Vec<grim_format::ChatMessage>)> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut nodes = Vec::new();
    let mut messages = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            let role_str = val.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = val
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let thinking = val
                .get("thinking")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let stats = val
                .get("stats")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let role = match role_str {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool_call" => Role::ToolCall,
                "tool_result" => Role::ToolResult,
                "system" => Role::System,
                "error" => Role::Error,
                _ => Role::Hint,
            };

            if matches!(role, Role::User | Role::Assistant | Role::System) {
                messages.push(grim_format::ChatMessage {
                    role: role_str.to_string(),
                    content: content.clone(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }

            nodes.push(MessageNode {
                role,
                content,
                thinking,
                thought_folded: true,
                turn_stats: stats,
            });
        }
    }
    Ok((nodes, messages))
}

/// Send a desktop notification using the system's native mechanism.
///
/// Best-effort: failures are silently ignored (notifications are a nice-to-have,
/// not a correctness requirement). Uses platform-specific commands.
fn send_desktop_notification(title: &str, body: &str) {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("osascript")
            .arg("-e")
            .arg(format!("display notification \"{body}\" with title \"{title}\""))
            .spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("notify-send")
            .arg(title)
            .arg(body)
            .spawn();
    }
    #[cfg(target_os = "windows")]
    {
        // Windows: use PowerShell to show a balloon tip.
        let _ = std::process::Command::new("powershell")
            .arg("-Command")
            .arg(format!(
                "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
                 $balloon = New-Object System.Windows.Forms.NotifyIcon; \
                 $balloon.Icon = [System.Drawing.SystemIcons]::Information; \
                 $balloon.BalloonTipTitle = '{title}'; \
                 $balloon.BalloonTipText = '{body}'; \
                 $balloon.ShowBalloonTip(3000)"
            ))
            .spawn();
    }
}

/// Terminal lifecycle guard: restores the alternate screen on drop and on
/// panic, so the user's shell is never left in raw mode.
pub struct TerminalGuard;

impl TerminalGuard {
    pub fn new() -> Self {
        let prior = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = ratatui::restore();
            prior(info);
        }));
        TerminalGuard
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = ratatui::restore();
    }
}

/// Render one frame.
fn ui(f: &mut Frame, app: &App) {
    let input_height = (app.composer.line_count() as u16 + 2).clamp(3, 8);
    let outer = Layout::vertical([Constraint::Min(3), Constraint::Length(input_height)]).split(f.area());

    let (chat_area, side_area) = if app.show_sidebar {
        let main = Layout::horizontal([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(outer[0]);
        (main[0], Some(main[1]))
    } else {
        (outer[0], None)
    };

    let chat_items = app.transcript.render_lines();
    let chat = Paragraph::new(chat_items)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_offset as u16, 0))
        .block(Block::bordered().title("Chat (Tab: reasoning fold | F2: sidebar | Esc: cancel)"));
    f.render_widget(chat, chat_area);

    if let Some(area) = side_area {
        let side_chunks = Layout::vertical([Constraint::Min(12), Constraint::Length(4)]).split(area);
        let lines: Vec<Line> = diagnostics::sidebar_lines(&app.snap)
            .into_iter()
            .map(Line::from)
            .collect();
        let side = Paragraph::new(lines).block(Block::bordered().title("Diagnostics"));
        f.render_widget(side, side_chunks[0]);

        let spark_data = app.speed_history.as_slice();
        let spark = Sparkline::default()
            .block(Block::bordered().title("Decode Speed (tok/s)"))
            .data(spark_data)
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(spark, side_chunks[1]);
    }

    let input_text = app.composer.text();
    let title = match app.input_mode {
        InputMode::Chat => "Input (/help · /model · /temp · /topp · /save · /edit · /exit · Ctrl+P palette · Ctrl+O sessions · F4 model picker)",
        InputMode::CtxOverride => {
            "Context limit override (Enter applies, empty = auto, Esc cancels)"
        }
        InputMode::ModelPicker { .. } => "Select model using arrow keys, press Enter to load",
        InputMode::CommandPalette { .. } => "Filter commands (type to search, Enter to select, Esc to cancel)",
        InputMode::SessionBrowser { .. } => "Select session to load (Enter loads, Esc cancels)",
    };
    f.render_widget(
        Paragraph::new(input_text.as_str()).block(Block::bordered().title(title)),
        outer[1],
    );

    // Render autocomplete popups. Slash commands and @file are mutually exclusive.
    // Uses fuzzy matching via SelectList for typo tolerance.
    if app.input_mode == InputMode::Chat && input_text.starts_with('/') && !input_text.contains(' ') {
        let query = input_text.trim_start_matches('/').trim_start();
        let all_items: Vec<crate::tui::select_list::SelectItem> = app
            .registry
            .all_commands()
            .iter()
            .map(|spec| crate::tui::select_list::SelectItem {
                value: spec.name.to_string(),
                label: spec.name.to_string(),
                description: Some(spec.description.to_string()),
            })
            .collect();
        let mut menu = crate::tui::select_list::SelectList::new(
            all_items,
            6,
            crate::tui::select_list::SelectListTheme::default(),
        );
        if query.is_empty() {
            // Show all commands when only "/" has been typed.
            menu.set_filter("");
        } else {
            menu.set_filter(query);
            // Move selection to match the previously selected index for continuity.
            for _ in 0..(app.selected_completion % 6) {
                menu.move_down();
            }
        }
        let filtered_count = menu.filtered_len();
        if filtered_count > 0 {
            let height = (filtered_count as u16 + 2).min(8);
            let popup_area = Rect {
                x: outer[1].x + 1,
                y: outer[1].y.saturating_sub(height),
                width: 48.min(outer[1].width.saturating_sub(2)),
                height,
            };
            let completion_lines = menu.render(popup_area.width.saturating_sub(2));
            let popup = Paragraph::new(completion_lines)
                .block(Block::bordered().title("Autocomplete (Tab selects)"));
            f.render_widget(popup, popup_area);
        }
    } else if app.input_mode == InputMode::Chat {
        // Check for @file trigger at the cursor position.
        let cursor = app.composer.cursor_offset();
        if let Some((_start, prefix)) =
            crate::tui::file_complete::extract_at_prefix(&input_text, cursor)
        {
            let after_at = prefix.trim_start_matches('@');
            let base = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            let suggestions =
                crate::tui::file_complete::get_file_suggestions_ranked(
                    after_at,
                    &base,
                    50,
                    &app.frecency,
                );
            if !suggestions.is_empty() {
                let items: Vec<crate::tui::select_list::SelectItem> = suggestions
                    .iter()
                    .map(|s| crate::tui::select_list::SelectItem {
                        value: s.value.clone(),
                        label: s.label.clone(),
                        description: None,
                    })
                    .collect();
                let menu = crate::tui::select_list::SelectList::new(
                    items,
                    6,
                    crate::tui::select_list::SelectListTheme::default(),
                );
                let count = menu.filtered_len();
                if count > 0 {
                    let height = (count as u16 + 2).min(8);
                    let popup_area = Rect {
                        x: outer[1].x + 1,
                        y: outer[1].y.saturating_sub(height),
                        width: 48.min(outer[1].width.saturating_sub(2)),
                        height,
                    };
                    let completion_lines = menu.render(popup_area.width.saturating_sub(2));
                    let popup = Paragraph::new(completion_lines)
                        .block(Block::bordered().title("Files (Tab selects)"));
                    f.render_widget(popup, popup_area);
                }
            }
        }
    }

    // Render model selection modal when in ModelPicker mode
    if let InputMode::ModelPicker { selected } = app.input_mode {
        let models = grim_core::catalog::list_local_models();
        let height = (models.len() as u16 + 4).clamp(5, 16);
        let width = 64.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        if models.is_empty() {
            lines.push(Line::from("  No local models discovered in catalog."));
        } else {
            for (idx, m) in models.iter().enumerate() {
                let is_sel = idx == selected;
                let prefix = if is_sel { "▶ " } else { "  " };
                let style = if is_sel {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                let ctx_str = if m.context_length > 0 {
                    format!("ctx {}", m.context_length)
                } else {
                    "ctx ?".into()
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("{}{:<24}", prefix, m.name), style),
                    Span::styled(format!(" {:<8} {}", m.quant, ctx_str), Style::default().fg(Color::DarkGray)),
                ]));
            }
        }
        let modal = Paragraph::new(lines)
            .block(Block::bordered().title("Select Model to Hot-Swap (Enter loads, Esc cancels)"));
        f.render_widget(modal, modal_area);
    }

    // Render command palette overlay.
    if let InputMode::CommandPalette { selected } = app.input_mode {
        let filtered = app.palette_filtered_commands();
        let height = (filtered.len() as u16 + 4).clamp(5, 20);
        let width = 72.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled(
            "Type to filter commands, Enter to select, Esc to cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
        for (idx, cmd) in filtered.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let style = if is_sel {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            let hint_str = if cmd.hint.is_empty() {
                String::new()
            } else {
                format!(" {}", cmd.hint)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{}{}{}", prefix, cmd.name, hint_str), style),
                Span::styled(
                    format!("  {}", cmd.description),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }
        if filtered.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No matching commands",
                Style::default().fg(Color::DarkGray),
            )));
        }
        let modal = Paragraph::new(lines).block(Block::bordered().title("Command Palette (Ctrl+P)"));
        f.render_widget(modal, modal_area);
    }

    // Render session browser overlay.
    if let InputMode::SessionBrowser { selected } = app.input_mode {
        let sessions = app.discover_session_files();
        let height = (sessions.len() as u16 + 4).clamp(5, 20);
        let width = 64.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled(
            "Select a session to load, Enter to load, Esc to cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
        for (idx, path) in sessions.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let style = if is_sel {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(Span::styled(
                format!("{}{}", prefix, path),
                style,
            )));
        }
        if sessions.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No .jsonl session files found in current directory",
                Style::default().fg(Color::DarkGray),
            )));
        }
        let modal = Paragraph::new(lines).block(Block::bordered().title("Session Browser (Ctrl+O)"));
        f.render_widget(modal, modal_area);
    }

    let (c_row, c_col) = app.composer.cursor_row_col();
    f.set_cursor_position(Position::new(
        outer[1].x + 1 + c_col as u16,
        outer[1].y + 1 + c_row as u16,
    ));

    // Render toast overlay in the top-right corner.
    if let Some(toast) = &app.toast {
        let toast_width = 40.min(f.area().width.saturating_sub(4));
        let toast_lines = render_toast(toast, toast_width);
        let toast_height = toast_lines.len() as u16 + 2;
        let toast_area = Rect {
            x: f.area().width.saturating_sub(toast_width + 2),
            y: 1,
            width: toast_width,
            height: toast_height,
        };
        let toast_widget = Paragraph::new(toast_lines).block(
            Block::bordered()
                .title("Notice")
                .border_style(Style::default().fg(toast.variant.color())),
        );
        f.render_widget(toast_widget, toast_area);
    }
}

/// Entry point for the `grim tui` command.
///
/// Runs the terminal loop until the user quits. Requires an interactive
/// terminal; otherwise returns a config error instead of garbling a pipe.
pub async fn cmd_tui(
    model: Option<String>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    max_tokens: usize,
    seed: u64,
    repeat_penalty: f32,
) -> Result<()> {
    if !std::io::stdout().is_terminal() {
        return Err(Error::Config(
            "grim tui needs an interactive terminal".into(),
        ));
    }

    let mut term = ratatui::init();
    let _guard = TerminalGuard::new();

    let (cmd_tx, cmd_rx): (Sender<WorkerCommand>, Receiver<WorkerCommand>) =
        std::sync::mpsc::channel();
    let (evt_tx, evt_rx): (Sender<WorkerEvent>, Receiver<WorkerEvent>) = std::sync::mpsc::channel();

    let params = WorkerParams {
        temperature,
        top_p,
        top_k,
        max_tokens,
        seed,
        repeat_penalty,
    };
    let worker = worker::spawn_worker(params, cmd_rx, evt_tx);
    if let Some(m) = &model {
        let _ = cmd_tx.send(WorkerCommand::LoadModel { name: m.clone() });
    }

    let mut app = App::new(cmd_tx.clone());
    let mut scheduler = RenderScheduler::new();
    // Initial frame should render immediately.
    scheduler.request_render();
    loop {
        while let Ok(evt) = evt_rx.try_recv() {
            app.handle_event(evt);
            scheduler.request_render();
        }
        // Expire toasts so they auto-dismiss.
        if app.expire_toast() {
            scheduler.request_render();
        }
        // Handle terminal resize as a full redraw trigger.
        if crossterm::event::poll(Duration::from_millis(50))
            .map_err(|e| Error::Config(format!("terminal poll failed: {e}")))?
        {
            let event = crossterm::event::read()
                .map_err(|e| Error::Config(format!("terminal read failed: {e}")))?;
            match event {
                crossterm::event::Event::Resize(_, _) => {
                    scheduler.reset();
                    scheduler.request_render();
                }
                crossterm::event::Event::Key(KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                }) => {
                    app.should_quit = true;
                }
                crossterm::event::Event::Key(k) => {
                    app.handle_key(k);
                    scheduler.request_immediate();
                }
                _ => {}
            }
        }
        if scheduler.should_render() {
            term.draw(|f| ui(f, &app))
                .map_err(|e| Error::Config(format!("render failed: {e}")))?;
        }
        if app.should_quit {
            break;
        }
    }

    let _ = cmd_tx.send(WorkerCommand::Quit);
    let _ = worker.join();
    let _ = ratatui::restore();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_loads() {
        assert!(!diagnostics::format_bytes(0).is_empty());
    }

    #[test]
    fn parses_known_commands() {
        assert!(matches!(parse_slash_command("/exit"), SlashCommand::Exit));
        assert!(matches!(parse_slash_command("/help"), SlashCommand::Help));
        assert!(matches!(
            parse_slash_command("/model"),
            SlashCommand::Model(None)
        ));
        assert!(matches!(
            parse_slash_command("/model llama3"),
            SlashCommand::Model(Some(m)) if m == "llama3"
        ));
        assert!(matches!(
            parse_slash_command("/temp 0.8"),
            SlashCommand::Temp(Some(t)) if (t - 0.8).abs() < 1e-4
        ));
        assert!(matches!(
            parse_slash_command("/topp 0.95"),
            SlashCommand::TopP(Some(p)) if (p - 0.95).abs() < 1e-4
        ));
        assert!(matches!(
            parse_slash_command("/ctx 4096"),
            SlashCommand::Ctx(Some(4096))
        ));
        assert!(matches!(
            parse_slash_command("/ctx auto"),
            SlashCommand::Ctx(None)
        ));
        assert!(matches!(
            parse_slash_command("/system you are helpful"),
            SlashCommand::System(Some(s)) if s == "you are helpful"
        ));
        assert!(matches!(
            parse_slash_command("/system"),
            SlashCommand::System(None)
        ));
        assert!(matches!(
            parse_slash_command("/load session.jsonl"),
            SlashCommand::Load(p) if p == "session.jsonl"
        ));
        assert!(matches!(
            parse_slash_command("/save chat.txt"),
            SlashCommand::Save(p) if p == "chat.txt"
        ));
        assert!(matches!(
            parse_slash_command("hello"),
            SlashCommand::NotACommand
        ));
        assert!(matches!(
            parse_slash_command("/nope"),
            SlashCommand::Unknown(s) if s == "nope"
        ));
        assert!(matches!(
            parse_slash_command("  /model x "),
            SlashCommand::Model(Some(m)) if m == "x"
        ));
    }

    #[test]
    fn parses_ctx_override() {
        assert!(matches!(parse_ctx_override(""), CtxOverride::Auto));
        assert!(matches!(parse_ctx_override("  "), CtxOverride::Auto));
        assert!(matches!(
            parse_ctx_override("8192"),
            CtxOverride::Apply(8192)
        ));
        assert!(matches!(parse_ctx_override("abc"), CtxOverride::Invalid));
        assert!(matches!(parse_ctx_override("-1"), CtxOverride::Invalid));
    }

    #[test]
    fn test_export_import_transcript_jsonl_roundtrip() {
        let mut transcript = Transcript::new();
        transcript.push_user("What is GRIM?".into());
        transcript.append_token("<think>Analyzing GRIM architecture</think>GRIM is a high performance inference engine.");
        transcript.finish_turn("· ttft 30ms | 45.0 tok/s".into());

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_session.jsonl");
        let path_str = path.to_str().unwrap();

        let count = export_transcript(&transcript, path_str).unwrap();
        assert_eq!(count, 2);

        let (loaded_nodes, chat_msgs) = import_transcript(path_str).unwrap();
        assert_eq!(loaded_nodes.len(), 2);
        assert_eq!(chat_msgs.len(), 2);
        assert_eq!(loaded_nodes[0].role, Role::User);
        assert_eq!(loaded_nodes[0].content, "What is GRIM?");
        assert_eq!(loaded_nodes[1].role, Role::Assistant);
        assert_eq!(loaded_nodes[1].thinking.as_deref(), Some("Analyzing GRIM architecture"));
        assert_eq!(loaded_nodes[1].content, "GRIM is a high performance inference engine.");
    }

    #[test]
    fn test_model_picker_key_handling() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        assert_eq!(app.input_mode, InputMode::Chat);

        app.handle_key(KeyEvent::from(KeyCode::F(4)));
        assert_eq!(app.input_mode, InputMode::ModelPicker { selected: 0 });

        app.handle_key(KeyEvent::from(KeyCode::Esc));
        assert_eq!(app.input_mode, InputMode::Chat);
    }

    #[test]
    fn test_toast_show_and_expire() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        assert!(app.toast.is_none());

        app.show_toast(Toast::info("test message"));
        assert!(app.toast.is_some());
        assert_eq!(app.toast.as_ref().unwrap().message, "test message");

        // Toast should not be expired immediately.
        assert!(!app.expire_toast());

        // Manually create an already-expired toast.
        app.show_toast(Toast::with_duration(
            ToastVariant::Info,
            "expired",
            std::time::Duration::from_secs(0),
        ));
        // Deadline is now + 0, so it should be expired (or very close).
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(app.expire_toast());
        assert!(app.toast.is_none());
    }

    #[test]
    fn test_toast_variants_and_colors() {
        use toast::ToastVariant;
        let info = Toast::info("info");
        let success = Toast::success("ok");
        let warning = Toast::warning("warn");
        let error = Toast::error("err");
        assert_eq!(info.variant, ToastVariant::Info);
        assert_eq!(success.variant, ToastVariant::Success);
        assert_eq!(warning.variant, ToastVariant::Warning);
        assert_eq!(error.variant, ToastVariant::Error);
    }

    #[test]
    fn test_command_palette_filtering() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Open palette and type a query.
        app.input_mode = InputMode::CommandPalette { selected: 0 };
        app.composer.set_text("/mod");

        let filtered = app.palette_filtered_commands();
        // "model" should match "/mod" via fuzzy matching.
        assert!(filtered.iter().any(|c| c.name == "model"));
        // "exit" should not match.
        assert!(!filtered.iter().any(|c| c.name == "exit"));
    }

    #[test]
    fn test_command_palette_key_handling() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Open palette with Ctrl+P.
        app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));
        assert!(matches!(
            app.input_mode,
            InputMode::CommandPalette { selected: 0 }
        ));

        // Escape returns to chat.
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()));
        assert_eq!(app.input_mode, InputMode::Chat);
    }

    #[test]
    fn test_session_browser_key_handling() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Open session browser with Ctrl+O.
        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(matches!(
            app.input_mode,
            InputMode::SessionBrowser { selected: 0 }
        ));

        // Escape returns to chat.
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()));
        assert_eq!(app.input_mode, InputMode::Chat);
    }

    #[test]
    fn test_slash_command_edit() {
        assert!(matches!(
            parse_slash_command("/edit"),
            SlashCommand::Edit
        ));
        assert!(matches!(
            parse_slash_command("/editor"),
            SlashCommand::ShowEditor
        ));
    }

    #[test]
    fn test_session_browser_discovers_files() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let app = App::new(tx);
        // Just verify the function doesn't panic; actual files depend on CWD.
        let _sessions = app.discover_session_files();
    }

    #[test]
    fn test_generation_complete_resets_flag() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.generation_complete_notified = true;

        // Starting a new generation resets the flag.
        app.submit_chat("hello");
        assert!(!app.generation_complete_notified);
    }
}
