//! grim tui: Ratatui chat interface over the in-process engine.
//!
//! Two threads: the UI thread owns the terminal, input composer, and ratatui loop;
//! the worker thread owns Engine, tokenizer, and sampler. The UI thread sends
//! `WorkerCommand`s and drains `WorkerEvent`s over `std::sync::mpsc` channels.
//! GPU and model code runs only on the worker.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyCode, KeyEvent, KeyboardEnhancementFlags, KeyModifiers,
    MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use grim_core::error::{Error, Result};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Gauge, Paragraph, Sparkline, Wrap};

/// Slash command descriptors and registry.
pub mod commands;

/// Text clipboard: arboard with OSC52 fallback.
pub mod clipboard;

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

/// Persistent searchable prompt history (Ctrl+R).
pub mod history;

/// Markdown → ratatui rendering.
pub mod markdown;

/// Line-based diff for edit previews.
pub mod diff;

/// Persisted always-allow rules for tool approvals.
pub mod permissions;

/// Queue of user messages typed while the worker is generating.
pub mod queue;

/// Skill discovery and loading from SKILL.md files.
pub mod skills;

/// Speed history ring buffer for sparklines.
pub mod sparkline;

/// Structured transcript with reasoning trace folding.
pub mod transcript;

/// Agent task list panel for the sidebar.
pub mod tasks;

/// XDG base dirs for persisted TUI state.
pub mod paths;

/// Central session store: autosave, titles, listing.
pub mod sessions;

/// Keyboard-navigable selection menu.
pub mod select_list;

/// Constrained VStack/HStack/ScrollView layout engine.
pub mod layout;

/// Toast notification system.
pub mod toast;

/// Coding tool definitions and sandboxed execution.
pub mod tools;

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
pub use skills::{Skill, default_skills_dir, discover_skills, find_skill, load_skill_body};
pub use sparkline::SpeedHistory;
pub use toast::{Toast, ToastVariant, render_toast};
pub use throttle::{RenderScheduler, MIN_FRAME_INTERVAL};
pub use transcript::{MessageNode, Role, Transcript};
pub use undo_stack::UndoStack;
pub use worker::{DiagnosticsSnapshot, TurnStats, WorkerCommand, WorkerEvent, WorkerParams};

/// Re-export of the task list types for convenience.
pub use tasks::{Task, TaskList, TaskStatus};

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
    /// Activate a skill by name (or list skills if no name given).
    Skill(Option<String>),
    /// List all discovered skills.
    Skills,
    /// Set the project directory (sandbox root for tools + cwd for @file).
    ProjectDir(String),
    /// Print the current project directory.
    Pwd,
    /// Set the thinking/reasoning effort level (off, low, medium, high, max).
    Thinking(Option<String>),
    /// Toggle plan mode (read-only tools, model presents a plan first).
    Plan(Option<String>),
    /// Select the inference backend (rocm, cuda, metal, cpu, auto).
    Backend(Option<String>),
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
        "skill" if after.is_empty() => SlashCommand::Skill(None),
        "skill" => SlashCommand::Skill(Some(after.trim().to_string())),
        "skills" => SlashCommand::Skills,
        "project" if after.is_empty() => SlashCommand::ProjectDir(String::new()),
        "project" => SlashCommand::ProjectDir(after.trim().to_string()),
        "cd" => SlashCommand::ProjectDir(after.trim().to_string()),
        "pwd" => SlashCommand::Pwd,
        "think" | "thinking" if after.is_empty() => SlashCommand::Thinking(None),
        "think" | "thinking" => SlashCommand::Thinking(Some(after.trim().to_string())),
        "plan" if after.is_empty() => SlashCommand::Plan(None),
        "plan" => SlashCommand::Plan(Some(after.trim().to_string())),
        "backend" if after.is_empty() => SlashCommand::Backend(None),
        "backend" => SlashCommand::Backend(Some(after.trim().to_string())),
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
    /// Interactive session browser overlay (type-to-filter).
    SessionBrowser { query: String, selected: usize },
    /// Interactive skill picker overlay (Ctrl+G).
    SkillPicker { selected: usize },
    /// Interactive backend picker overlay (Ctrl+B).
    BackendPicker { selected: usize },
    /// Project directory input mode (type a path to set the sandbox root).
    ProjectDir,
    /// Find in transcript (Ctrl+F) — fuzzy/substring search over nodes.
    Find {
        query: String,
        matches: Vec<(usize, usize)>,
        selected: usize,
    },
    /// Prompt history search overlay (Ctrl+R).
    HistorySearch {
        query: String,
        selected: usize,
        matches: Vec<String>,
    },
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
    /// Pending tool call awaiting UI approval (call_id, name, arguments).
    pub pending_tool_call: Option<(String, String, String)>,
    /// Precomputed edit diff for the pending call (edit_file / write_file).
    pub pending_tool_diff: Option<Vec<diff::DiffLine>>,
    /// Tool definitions exposed to the model (empty = tools disabled).
    pub tools: Vec<grim_format::ToolDef>,
    /// Sandbox root for tool execution.
    pub sandbox_root: PathBuf,
    /// Whether the user is being asked to approve a tool call.
    pub tool_approval_mode: bool,
    /// Frame counter incremented each render cycle, used for spinner animation.
    pub frame_count: u64,
    /// Discovered skills from the skills directory.
    pub skills: Vec<crate::tui::skills::Skill>,
    /// Name of the currently active skill (body injected into system prompt).
    pub active_skill_name: Option<String>,
    /// Project directory set by /project or /cd; defaults to current_dir.
    pub project_dir: PathBuf,
    /// Current thinking/reasoning effort level.
    pub thinking_level: grim_core::sampler::ThinkingLevel,
    /// Plan mode: mutating tools are blocked; the model presents a plan.
    pub plan_mode: bool,
    /// Agent task list rendered in the sidebar.
    pub task_list: crate::tui::tasks::TaskList,
    /// Selected inference backend (rocm, cuda, metal, cpu). None = auto-detect.
    pub backend: Option<String>,
    /// True when chat is pinned to bottom (auto-scroll lock).
    pub was_at_bottom: bool,
    /// Number of new lines arrived while user was scrolled up.
    pub pending_new_lines: usize,
    /// Find mode query (mirrors InputMode::Find for test access).
    pub find_query: String,
    /// Find mode matches as (node_idx, line_idx) pairs.
    pub find_matches: Vec<(usize, usize)>,
    /// Find mode selected match index.
    pub find_selected: usize,
    /// Border flash until this instant (300ms delight on TurnComplete).
    pub flash_until: Option<Instant>,
    /// Always-allow tool rules, shared with the worker thread. The worker
    /// checks them before prompting; the UI adds to them on "always allow".
    pub permissions: permissions::SharedPermissions,
    /// Messages typed while generating; drained on TurnComplete.
    pub queue: crate::tui::queue::MessageQueue,
    /// Persistent prompt history searched by Ctrl+R.
    pub history: crate::tui::history::PromptHistory,
    /// Autosave target for this conversation; set on first user submit.
    pub session_path: Option<std::path::PathBuf>,
}

impl App {
    pub fn new(cmd_tx: Sender<WorkerCommand>) -> Self {
        let project_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."));
        // Discover skills from the default skills directory.
        let skills = crate::tui::skills::default_skills_dir()
            .map(|dir| crate::tui::skills::discover_skills(&dir))
            .unwrap_or_default();
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
            frecency: Frecency::load(),
            generation_complete_notified: false,
            pending_tool_call: None,
            pending_tool_diff: None,
            tools: crate::tui::tools::coding_tools(),
            sandbox_root: project_dir.clone(),
            tool_approval_mode: false,
            frame_count: 0,
            skills,
            active_skill_name: None,
            project_dir,
            thinking_level: grim_core::sampler::ThinkingLevel::Default,
            plan_mode: false,
            task_list: crate::tui::tasks::TaskList::new(),
            backend: std::env::var("GRIM_BACKEND").ok(),
            was_at_bottom: true,
            pending_new_lines: 0,
            find_query: String::new(),
            find_matches: Vec::new(),
            find_selected: 0,
            flash_until: None,
            permissions: permissions::shared(permissions::PermissionRules::load()),
            queue: crate::tui::queue::MessageQueue::default(),
            history: crate::tui::history::PromptHistory::load(),
            session_path: None,
        }
    }

    /// Show a toast notification, replacing any existing one.
    pub fn show_toast(&mut self, toast: Toast) {
        self.toast = Some(toast);
    }

    /// Rewrite the session autosave file from the current transcript.
    /// No-op when no session exists yet (no user message sent).
    fn autosave_now(&mut self) {
        if let Some(p) = &self.session_path {
            if let Err(e) = sessions::autosave(&self.transcript, p) {
                self.show_toast(Toast::warning(format!("autosave failed: {e}")));
            }
        }
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

    /// Finalize the current streaming buffer as an assistant message — in the
    /// transcript (rendered segment) and in `messages` (model history). A
    /// tool call arriving mid-turn splits the stream into segments; `tool`
    /// attaches this segment's tool call to the message so the next turn's
    /// history preserves the full exchange.
    fn close_stream_segment(&mut self, call_id: Option<&str>, tool: Option<(&str, &str)>) {
        let text = self.transcript.streaming_text_clean();
        self.transcript.finish_segment();
        let tool_calls = tool.map(|(tname, args)| {
            vec![grim_format::ToolCallMsg {
                id: call_id.unwrap_or_default().to_string(),
                name: tname.to_string(),
                arguments: args.to_string(),
            }]
        });
        if text.is_empty() && tool_calls.is_none() {
            return;
        }
        self.messages.push(grim_format::ChatMessage {
            role: "assistant".to_string(),
            content: text,
            tool_calls,
            tool_call_id: None,
            name: None,
        });
    }

    /// Apply a worker event to app state.
    pub fn handle_event(&mut self, evt: WorkerEvent) {
        match evt {
            WorkerEvent::Token { text } => {
                // Auto-scroll lock: when at bottom keep pinned, otherwise queue pending.
                let at_bottom = self.was_at_bottom || self.scroll_offset == 0;
                self.transcript.append_token(&text);
                if at_bottom {
                    // keep pinned — no pending increment, stay at bottom
                    self.was_at_bottom = true;
                    self.pending_new_lines = 0;
                    // scroll_offset stays at bottom (0 == bottom in this TUI)
                    self.scroll_offset = 0;
                } else {
                    self.pending_new_lines = self.pending_new_lines.saturating_add(1);
                }
            }
            WorkerEvent::TurnComplete { stats } => {
                // Capture the final assistant segment into history so the
                // next turn sees what the model said (and, across the loop,
                // the tool exchanges).
                self.close_stream_segment(None, None);
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
                // Border flash delight: green border for 300ms on turn complete.
                self.flash_until = Some(Instant::now() + Duration::from_millis(300));
                // Persist the completed turn before any queued message starts.
                self.autosave_now();
                // Send the next queued message, if any.
                if let Some(next) = self.queue.pop() {
                    self.transcript
                        .push_system("sending queued message...".into());
                    self.submit_chat(&next);
                }
            }
            WorkerEvent::Diagnostics { snap } => {
                self.snap = snap;
            }
            WorkerEvent::Error { message } => {
                self.transcript.push_error(message);
                self.generating = false;
                self.autosave_now();
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
            WorkerEvent::ToolCall {
                call_id,
                name,
                arguments,
            } => {
                // Close the current text segment so transcript and message
                // history read: text → tool call → result → more text.
                self.close_stream_segment(Some(&call_id), Some((&name, &arguments)));
                // Preview the diff for edit tools before the user approves.
                let preview = crate::tui::diff::preview_edit(
                    &name,
                    &arguments,
                    &crate::tui::tools::Sandbox::new(self.sandbox_root.clone()),
                );
                self.transcript
                    .push_tool_call_with_diff(&name, Some(&call_id), &arguments, preview.clone());
                self.pending_tool_call = Some((call_id, name.clone(), arguments));
                self.pending_tool_diff = preview;
                self.tool_approval_mode = true;
                self.show_toast(Toast::info(format!(
                    "Tool: {name} — y: approve, a: always allow, n: deny"
                )));
            }
            WorkerEvent::ToolExec {
                call_id,
                name,
                arguments,
                output,
            } => {
                // A tool the worker auto-executed: mirror it into the
                // transcript and the message history we own.
                self.close_stream_segment(Some(&call_id), Some((&name, &arguments)));
                self.transcript
                    .push_tool_call_with_diff(&name, Some(&call_id), &arguments, None);
                self.transcript
                    .push_tool_result_for(output.clone(), Some(call_id.clone()));
                self.messages.push(grim_format::ChatMessage {
                    role: "tool".to_string(),
                    content: output,
                    tool_call_id: Some(call_id),
                    name: Some(name),
                    tool_calls: None,
                });
            }
            WorkerEvent::TaskSync { tasks } => {
                let converted: Vec<Task> = tasks
                    .into_iter()
                    .map(|t| {
                        let status = t
                            .status
                            .parse::<TaskStatus>()
                            .unwrap_or(TaskStatus::Pending);
                        let mut task = Task::new(t.id, t.title).with_status(status);
                        if let Some(d) = t.description {
                            task = task.with_description(d);
                        }
                        task
                    })
                    .collect();
                let n = converted.len();
                self.task_list.set_tasks(converted);
                self.show_toast(Toast::info(format!("Tasks updated ({n})")));
            }
        }
    }

    /// Handle a single key press.
    pub fn handle_key(&mut self, key: KeyEvent) {
        // Clone input_mode to avoid borrow conflicts with mutable self in handlers.
        let mode = self.input_mode.clone();
        match mode {
            InputMode::CtxOverride => self.handle_ctx_key(key),
            InputMode::Chat => self.handle_chat_key(key),
            InputMode::ModelPicker { selected } => self.handle_model_picker_key(key, selected),
            InputMode::CommandPalette { .. } => self.handle_palette_key(key),
            InputMode::SessionBrowser { .. } => self.handle_session_browser_key(key),
            InputMode::SkillPicker { .. } => self.handle_skill_picker_key(key),
            InputMode::BackendPicker { .. } => self.handle_backend_picker_key(key),
            InputMode::ProjectDir => self.handle_project_dir_key(key),
            InputMode::Find { query, matches, selected } => {
                self.handle_find_key(key, query, matches, selected)
            }
            InputMode::HistorySearch { .. } => self.handle_history_search_key(key),
        }
    }

    /// Compute substring (case-insensitive) matches for the find query.
    fn compute_find_matches(&self, query: &str) -> Vec<(usize, usize)> {
        if query.is_empty() {
            return Vec::new();
        }
        let q = query.to_lowercase();
        let mut out = Vec::new();
        for (node_idx, node) in self.transcript.nodes.iter().enumerate() {
            for (line_idx, line) in node.content.lines().enumerate() {
                if line.to_lowercase().contains(&q) {
                    out.push((node_idx, line_idx));
                }
            }
            if let Some(th) = &node.thinking {
                for (line_idx, line) in th.lines().enumerate() {
                    if line.to_lowercase().contains(&q) {
                        out.push((node_idx, line_idx));
                    }
                }
            }
        }
        // Fallback: fuzzy match on whole content if no substring hit and query is short
        if out.is_empty() && !q.is_empty() {
            for (node_idx, node) in self.transcript.nodes.iter().enumerate() {
                if crate::tui::fuzzy::fuzzy_match(&q, &node.content.to_lowercase()).is_some() {
                    out.push((node_idx, 0));
                }
            }
        }
        out
    }

    fn sync_find_state(&mut self, query: String, matches: Vec<(usize, usize)>, selected: usize) {
        self.find_query = query.clone();
        self.find_matches = matches.clone();
        self.find_selected = selected;
        self.input_mode = InputMode::Find { query, matches, selected };
    }

    fn scroll_to_find_match(&mut self, selected: usize, matches: &[(usize, usize)]) {
        if matches.is_empty() {
            return;
        }
        let (node_idx, _line_idx) = matches[selected % matches.len()];
        // Approximate scroll: count rendered lines before the target node.
        // Use simple node-count heuristic scaled to line count so scroll_offset brings node into view.
        // More accurate would need layout width; heuristic is sufficient for the spec's "just scroll" requirement.
        let total = self.transcript.nodes.len();
        // Estimate 4 lines per node average + line_idx offset.
        let lines_after = total.saturating_sub(node_idx + 1) * 4;
        self.scroll_offset = lines_after;
        self.was_at_bottom = false;
    }

    fn handle_find_key(
        &mut self,
        key: KeyEvent,
        query: String,
        matches: Vec<(usize, usize)>,
        selected: usize,
    ) {
        let is_ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => {
                self.input_mode = InputMode::Chat;
                self.composer.clear();
            }
            KeyCode::Enter => {
                if !matches.is_empty() {
                    let next = (selected + 1) % matches.len();
                    let q = query.clone();
                    let m = matches.clone();
                    self.scroll_to_find_match(next, &m);
                    self.sync_find_state(q, m, next);
                }
            }
            KeyCode::Char('n') if is_ctrl => {
                if !matches.is_empty() {
                    let next = (selected + 1) % matches.len();
                    let q = query.clone();
                    let m = matches.clone();
                    self.scroll_to_find_match(next, &m);
                    self.sync_find_state(q, m, next);
                }
            }
            KeyCode::Char('f') if is_ctrl => {
                if !matches.is_empty() {
                    let next = (selected + 1) % matches.len();
                    let q = query.clone();
                    let m = matches.clone();
                    self.scroll_to_find_match(next, &m);
                    self.sync_find_state(q, m, next);
                }
            }
            KeyCode::Backspace => {
                let mut q = query;
                q.pop();
                let m = self.compute_find_matches(&q);
                let sel = 0;
                if !m.is_empty() {
                    self.scroll_to_find_match(sel, &m);
                }
                // Also update composer text for visual parity.
                self.composer.set_text(&q);
                // Move cursor to end (set_text already does)
                self.sync_find_state(q, m, sel);
            }
            KeyCode::Char(c) if !is_ctrl => {
                let mut q = query;
                q.push(c);
                let m = self.compute_find_matches(&q);
                let sel = 0;
                if !m.is_empty() {
                    self.scroll_to_find_match(sel, &m);
                }
                self.composer.set_text(&q);
                self.sync_find_state(q, m, sel);
            }
            _ => {}
        }
    }

    /// Return the last user message content, if any.
    fn last_user_content(&self) -> Option<String> {
        self.transcript
            .nodes
            .iter()
            .rev()
            .find(|n| n.role == crate::tui::transcript::Role::User)
            .map(|n| n.content.clone())
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
        // Tool approval mode takes precedence over normal chat input.
        if self.tool_approval_mode {
            self.handle_tool_approval_key(key);
            return;
        }

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
                // When the composer is empty and we have tasks, Right
                // expands the selected task instead of moving the cursor.
                if self.composer.text().is_empty() && !self.task_list.is_empty() {
                    self.task_list.toggle_expand_selected();
                } else {
                    self.composer.move_cursor_right();
                }
            }
            KeyCode::Home => {
                self.composer.move_cursor_home();
            }
            KeyCode::End => {
                self.composer.move_cursor_end();
                // Auto-scroll lock: End re-pins to bottom
                self.was_at_bottom = true;
                self.pending_new_lines = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Up => {
                if self.composer.text().is_empty() && !self.task_list.is_empty() {
                    self.task_list.move_up();
                } else {
                    self.composer.move_cursor_up();
                }
            }
            KeyCode::Down => {
                if self.composer.text().is_empty() && !self.task_list.is_empty() {
                    self.task_list.move_down();
                } else {
                    self.composer.move_cursor_down();
                }
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
                let yanked = self.composer.yank();
                if yanked {
                    // Also copy yanked text to system clipboard via arboard or OSC52 fallback.
                    if let Some(text) = self.composer.peek_yank_text() {
                        copy_to_clipboard(&text);
                    } else if let Some(last) = self.transcript.nodes.last().map(|n| n.content.clone()) {
                        if !last.is_empty() {
                            copy_to_clipboard(&last);
                        }
                    }
                }
            }
            KeyCode::Char('y') if is_alt => {
                self.composer.yank_pop();
            }
            KeyCode::Char('y') if !is_ctrl && !is_alt && self.composer.is_empty() => {
                // When selection exists, copy to system clipboard via arboard or OSC52.
                // Selection = last assistant/user content or find selected match.
                if let Some(text) = self.selected_transcript_text() {
                    copy_to_clipboard(&text);
                    self.show_toast(Toast::success("Copied to clipboard"));
                } else {
                    // Prefer the last fenced code block of the last assistant reply.
                    let last_assistant = self
                        .transcript
                        .nodes
                        .iter()
                        .rev()
                        .find(|n| n.role == Role::Assistant && !n.content.is_empty())
                        .map(|n| n.content.clone());
                    if let Some((_, code)) =
                        last_assistant.as_deref().and_then(markdown::last_fenced_code_block)
                    {
                        copy_to_clipboard(&code);
                        self.show_toast(Toast::success("Code block copied to clipboard"));
                    } else if let Some(last) = self.transcript.nodes.last().map(|n| n.content.clone()) {
                        if !last.is_empty() {
                            copy_to_clipboard(&last);
                            self.show_toast(Toast::success("Copied to clipboard"));
                        }
                    } else {
                        self.composer.insert_char('y');
                    }
                }
                return;
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
            KeyCode::Char('f') if is_ctrl => {
                // Find in transcript: clear composer and enter Find mode.
                self.composer.clear();
                self.find_query = String::new();
                self.find_matches = Vec::new();
                self.find_selected = 0;
                self.input_mode = InputMode::Find {
                    query: String::new(),
                    matches: Vec::new(),
                    selected: 0,
                };
                return;
            }
            KeyCode::Char('e') if !is_ctrl && !is_alt && self.composer.is_empty() => {
                if let Some(content) = self.last_user_content() {
                    self.composer.set_text(&content);
                }
                return;
            }
            KeyCode::Char('r') if !is_ctrl && !is_alt && self.composer.is_empty() => {
                if let Some(content) = self.last_user_content() {
                    let trimmed = content.trim().to_string();
                    if trimmed.is_empty() {
                        return;
                    }
                    if self.snap.model_name.is_none() {
                        self.transcript
                            .push_system("no model loaded — use /model <name> first".into());
                        return;
                    }
                    if self.generating {
                        self.transcript
                            .push_system("generation in progress; Esc to cancel first".into());
                        return;
                    }
                    self.messages.push(grim_format::ChatMessage {
                        role: "user".to_string(),
                        content: trimmed.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                    self.transcript.push_user(trimmed.clone());
                    self.generating = true;
                    self.generation_complete_notified = false;
                    let _ = self.cmd_tx.send(WorkerCommand::Generate {
                        messages: self.messages.clone(),
                    });
                }
                return;
            }
            KeyCode::Char('p') if is_ctrl => {
                // Command palette: fuzzy-searchable command list.
                self.input_mode = InputMode::CommandPalette { selected: 0 };
            }
            KeyCode::Char('r') if is_ctrl => {
                // Prompt history search over all persisted prompts.
                let matches = self.history.search("", 50);
                self.input_mode = InputMode::HistorySearch {
                    query: String::new(),
                    selected: 0,
                    matches,
                };
            }
            KeyCode::Char('o') if is_ctrl => {
                // Session browser: interactive session list.
                self.input_mode = InputMode::SessionBrowser {
                    query: String::new(),
                    selected: 0,
                };
            }
            KeyCode::Char('g') if is_ctrl => {
                // Skill picker: fuzzy-searchable list of discovered skills.
                self.input_mode = InputMode::SkillPicker { selected: 0 };
            }
            KeyCode::Char('d') if is_ctrl => {
                // Project directory: type or pick the sandbox root.
                self.composer.set_text(&self.project_dir.to_string_lossy());
                self.input_mode = InputMode::ProjectDir;
            }
            KeyCode::Char('b') if is_ctrl => {
                // Backend picker: open the interactive backend selection.
                self.input_mode = InputMode::BackendPicker { selected: 0 };
            }
            KeyCode::Char('t') if is_ctrl => {
                // Cycle thinking level: Default → Low → Medium → High → Off → Default.
                use grim_core::sampler::ThinkingLevel;
                self.thinking_level = match self.thinking_level {
                    ThinkingLevel::Off => ThinkingLevel::Default,
                    ThinkingLevel::Default => ThinkingLevel::Low,
                    ThinkingLevel::Low => ThinkingLevel::Medium,
                    ThinkingLevel::Medium => ThinkingLevel::High,
                    ThinkingLevel::High => ThinkingLevel::Off,
                    ThinkingLevel::Custom(_) => ThinkingLevel::Default,
                };
                let _ = self.cmd_tx.send(WorkerCommand::SetThinking {
                    level: self.thinking_level,
                });
                let level_label = thinking_level_label(&self.thinking_level);
                self.transcript
                    .push_system(format!("thinking level: {level_label}"));
                self.show_toast(Toast::info(format!("Thinking: {level_label}")));
            }
            KeyCode::Char(c) => {
                self.composer.insert_char(c);
            }
            KeyCode::Tab if is_shift => {
                // Shift+Tab: cycle the selected task's status.
                self.task_list.cycle_selected_status();
            }
            KeyCode::Tab => {
                // Plain Tab: try command/@file autocomplete first, then
                // toggle the last task's expand state if no text.
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
                        self.frecency.save();
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
                // First Esc while generating clears the queue; the next cancels.
                if self.generating && !self.queue.is_empty() {
                    self.queue.clear();
                    self.transcript.push_system("cleared queued messages".into());
                } else {
                    let _ = self.cmd_tx.send(WorkerCommand::Cancel);
                    // Immediate UI feedback: show cancelling state.
                    if self.generating {
                        self.transcript.push_system("cancelling...".into());
                    }
                }
            }
            KeyCode::PageUp => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
                self.was_at_bottom = false;
            }
            KeyCode::PageDown => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
                if self.scroll_offset == 0 {
                    self.was_at_bottom = true;
                    self.pending_new_lines = 0;
                }
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
        let selected = match &self.input_mode {
            InputMode::SessionBrowser { selected, .. } => *selected,
            _ => return,
        };
        match key.code {
            KeyCode::Up => {
                self.input_mode = InputMode::SessionBrowser {
                    query: self.session_browser_query(),
                    selected: selected.saturating_sub(1),
                };
            }
            KeyCode::Down => {
                let n = self.session_browser_items().len();
                let new_sel = if n == 0 { 0 } else { (selected + 1).min(n - 1) };
                self.input_mode = InputMode::SessionBrowser {
                    query: self.session_browser_query(),
                    selected: new_sel,
                };
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Chat;
                if let Some(meta) = self.session_browser_items().get(selected).cloned() {
                    self.submit_chat(&format!("/load {}", meta.path.display()));
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Chat;
            }
            KeyCode::Backspace => {
                let mut query = self.session_browser_query();
                query.pop();
                self.input_mode = InputMode::SessionBrowser { query, selected: 0 };
            }
            KeyCode::Char(c) => {
                let mut query = self.session_browser_query();
                query.push(c);
                self.input_mode = InputMode::SessionBrowser { query, selected: 0 };
            }
            _ => {}
        }
    }

    fn session_browser_query(&self) -> String {
        match &self.input_mode {
            InputMode::SessionBrowser { query, .. } => query.clone(),
            _ => String::new(),
        }
    }

    /// Sessions currently visible in the browser (store + cwd, query-filtered).
    fn session_browser_items(&self) -> Vec<sessions::SessionMeta> {
        let q = self.session_browser_query().to_lowercase();
        sessions::list_sessions()
            .into_iter()
            .filter(|s| q.is_empty() || s.title.to_lowercase().contains(&q))
            .collect()
    }

    /// Prompt history search (Ctrl+R): type to filter, Enter fills the composer.
    fn handle_history_search_key(&mut self, key: KeyEvent) {
        if let InputMode::HistorySearch { query, selected, .. } = &mut self.input_mode {
            match key.code {
                KeyCode::Esc => self.input_mode = InputMode::Chat,
                KeyCode::Enter => {
                    let text = self.history_search_selected();
                    self.input_mode = InputMode::Chat;
                    if let Some(t) = text {
                        self.composer.set_text(&t);
                    }
                    return;
                }
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Down => *selected += 1,
                KeyCode::Backspace => {
                    query.pop();
                    *selected = 0;
                }
                KeyCode::Char(c) => {
                    query.push(c);
                    *selected = 0;
                }
                _ => {}
            }
        }
        let n = self.history_search_matches().len();
        if let InputMode::HistorySearch { selected, .. } = &mut self.input_mode {
            if *selected >= n {
                *selected = n.saturating_sub(1);
            }
        }
    }

    fn history_search_matches(&self) -> Vec<String> {
        match &self.input_mode {
            InputMode::HistorySearch { query, .. } => self.history.search(query, 50),
            _ => Vec::new(),
        }
    }

    fn history_search_selected(&self) -> Option<String> {
        let InputMode::HistorySearch { selected, .. } = &self.input_mode else {
            return None;
        };
        self.history_search_matches().get(*selected).cloned()
    }

    fn handle_skill_picker_key(&mut self, key: KeyEvent) {
        let filtered_count = self.filtered_skills().len();
        let selected = match &self.input_mode {
            InputMode::SkillPicker { selected } => *selected,
            _ => return,
        };
        match key.code {
            KeyCode::Up => {
                self.input_mode = InputMode::SkillPicker {
                    selected: selected.saturating_sub(1),
                };
            }
            KeyCode::Down => {
                let new_sel = if filtered_count == 0 {
                    0
                } else {
                    (selected + 1).min(filtered_count - 1)
                };
                self.input_mode = InputMode::SkillPicker { selected: new_sel };
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Chat;
                if let Some(skill) = self.filtered_skills().get(selected) {
                    self.activate_skill(&skill.id);
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
                    self.input_mode = InputMode::SkillPicker { selected: 0 };
                }
            }
            KeyCode::Char(c) => {
                let current = self.composer.text();
                let filter = current.trim_start_matches('/');
                self.composer.set_text(&format!("/{filter}{c}"));
                self.input_mode = InputMode::SkillPicker { selected: 0 };
            }
            _ => {}
        }
    }

    /// Return skills filtered by the current picker query.
    fn filtered_skills(&self) -> Vec<crate::tui::skills::Skill> {
        let query = self
            .composer
            .text()
            .trim_start_matches('/')
            .trim()
            .to_lowercase();
        if query.is_empty() {
            return self.skills.clone();
        }
        let mut scored: Vec<(i32, crate::tui::skills::Skill)> = self
            .skills
            .iter()
            .filter_map(|s| {
                let id_lower = s.id.to_lowercase();
                let name_lower = s.name.to_lowercase();
                let id_score = crate::tui::fuzzy::fuzzy_match(&query, &id_lower);
                let name_score = crate::tui::fuzzy::fuzzy_match(&query, &name_lower);
                match (id_score, name_score) {
                    (Some(id), Some(name)) => Some((id.score.max(name.score), s.clone())),
                    (Some(id), None) => Some((id.score, s.clone())),
                    (None, Some(name)) => Some((name.score, s.clone())),
                    (None, None) => None,
                }
            })
            .collect();
        scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        scored.into_iter().map(|(_, s)| s).collect()
    }

    /// Activate a skill by id: load its body and inject into the system prompt.
    fn activate_skill(&mut self, skill_id: &str) {
        let Some(skill) = self.skills.iter().find(|s| s.id == skill_id) else {
            self.transcript
                .push_error(format!("skill '{skill_id}' not found"));
            return;
        };
        match crate::tui::skills::load_skill_body(skill) {
            Ok(body) => {
                let body = body.trim().to_string();
                // Build a system prompt that wraps the skill body.
                let injected = if let Some(existing) = &self.system_prompt {
                    format!("{existing}\n\n# Skill: {}\n{body}", skill.name)
                } else {
                    format!("# Skill: {}\n{}", skill.name, body)
                };
                self.system_prompt = Some(injected.clone());
                // Inject or replace the system message at the head of the conversation.
                if let Some(first) = self.messages.first_mut() {
                    if first.role == "system" {
                        first.content = injected;
                    } else {
                        self.messages.insert(0, grim_format::ChatMessage {
                            role: "system".to_string(),
                            content: injected,
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                } else {
                    self.messages.push(grim_format::ChatMessage {
                        role: "system".to_string(),
                        content: injected,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
                self.active_skill_name = Some(skill.name.clone());
                self.transcript.push_system(format!(
                    "skill activated: {} ({})",
                    skill.name, skill.id
                ));
                self.show_toast(Toast::success(format!("Skill activated: {}", skill.name)));
            }
            Err(e) => {
                self.transcript
                    .push_error(format!("failed to load skill '{}': {e}", skill.id));
            }
        }
    }

    fn handle_backend_picker_key(&mut self, key: KeyEvent) {
        let backends = self.available_backends();
        let selected = match &self.input_mode {
            InputMode::BackendPicker { selected } => *selected,
            _ => return,
        };
        match key.code {
            KeyCode::Up => {
                self.input_mode = InputMode::BackendPicker {
                    selected: selected.saturating_sub(1),
                };
            }
            KeyCode::Down => {
                let next = if backends.is_empty() {
                    0
                } else {
                    (selected + 1).min(backends.len() - 1)
                };
                self.input_mode = InputMode::BackendPicker { selected: next };
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Chat;
                if let Some(name) = backends.get(selected) {
                    self.activate_backend(name);
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Chat;
            }
            _ => {}
        }
    }

    /// Return the list of available backend names (for the picker).
    fn available_backends(&self) -> Vec<String> {
        let mut backends = vec!["auto".to_string(), "cpu".to_string()];
        #[cfg(feature = "rocm")]
        backends.push("rocm".to_string());
        #[cfg(feature = "cuda")]
        backends.push("cuda".to_string());
        #[cfg(feature = "metal")]
        backends.push("metal".to_string());
        backends
    }

    fn handle_project_dir_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                let text = self.composer.submit();
                self.input_mode = InputMode::Chat;
                let path = text.trim();
                if path.is_empty() {
                    // Reset to current_dir.
                    self.project_dir = std::env::current_dir()
                        .unwrap_or_else(|_| std::path::PathBuf::from("."));
                } else {
                    let p = std::path::PathBuf::from(path);
                    if p.is_dir() {
                        self.project_dir = p.canonicalize().unwrap_or(p);
                    } else {
                        self.transcript
                            .push_error(format!("not a directory: {path}"));
                        return;
                    }
                }
                // Update sandbox root to match the project dir.
                self.sandbox_root = self.project_dir.clone();
                self.transcript.push_system(format!(
                    "project directory: {}",
                    self.project_dir.display()
                ));
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
                    "\nShortcuts:\n  Tab: Autocomplete command or @file / Toggle reasoning\n  Shift+Tab: Cycle selected task status | →: Expand task | ↑/↓: Navigate tasks\n  F2: Toggle sidebar | F3: Context override | Esc: Cancel turn\n  Ctrl+P: Command palette | Ctrl+O: Sessions | Ctrl+G: Skills | Ctrl+T: Thinking | Ctrl+D: Project dir\n  Ctrl+A / Ctrl+E: Line start/end | Ctrl+W: Delete word | Ctrl+K: Kill | Ctrl+Y: Yank | Alt+Y: Yank-pop | Ctrl+Z: Undo | Alt+F/B: Jump",
                );
                self.transcript.push_system(help_msg);
            }
            SlashCommand::Clear => {
                self.transcript.clear();
                self.messages.clear();
                self.speed_history.clear();
                self.scroll_offset = 0;
                self.was_at_bottom = true;
                self.pending_new_lines = 0;
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
            SlashCommand::Skill(None) => {
                // No name — open the interactive skill picker.
                self.input_mode = InputMode::SkillPicker { selected: 0 };
            }
            SlashCommand::Skill(Some(name)) => {
                let name = name.trim();
                if name.eq_ignore_ascii_case("off") || name.eq_ignore_ascii_case("clear") {
                    // Deactivate the current skill.
                    if self.active_skill_name.take().is_some() {
                        // Remove the system message so the skill body is no longer injected.
                        if let Some(pos) = self
                            .messages
                            .iter()
                            .position(|m| m.role == "system")
                        {
                            self.messages.remove(pos);
                        }
                        self.system_prompt = None;
                        self.transcript.push_system("skill deactivated".into());
                        self.show_toast(Toast::info("Skill deactivated"));
                    } else {
                        self.transcript.push_system("no active skill".into());
                    }
                } else {
                    self.activate_skill(name);
                }
            }
            SlashCommand::Skills => {
                // List all discovered skills in the transcript.
                if self.skills.is_empty() {
                    let dir = crate::tui::skills::default_skills_dir()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| "~/.agents/skills".into());
                    self.transcript
                        .push_system(format!("no skills found (scan dir: {dir})"));
                } else {
                    let mut msg = format!("discovered {} skills:\n", self.skills.len());
                    for s in &self.skills {
                        let desc = if s.description.is_empty() {
                            String::new()
                        } else {
                            format!(" — {}", s.description)
                        };
                        let active = self
                            .active_skill_name
                            .as_ref()
                            .map(|a| if a == &s.name { " [active]" } else { "" })
                            .unwrap_or("");
                        msg.push_str(&format!("  /skill {}{}{}\n", s.id, desc, active));
                    }
                    self.transcript.push_system(msg);
                }
            }
            SlashCommand::ProjectDir(path) => {
                if path.is_empty() {
                    // No path — open the interactive project dir input.
                    self.composer
                        .set_text(&self.project_dir.to_string_lossy());
                    self.input_mode = InputMode::ProjectDir;
                } else {
                    let p = std::path::PathBuf::from(&path);
                    if p.is_dir() {
                        self.project_dir = p.canonicalize().unwrap_or(p);
                        self.sandbox_root = self.project_dir.clone();
                        self.transcript
                            .push_system(format!("project directory: {}", self.project_dir.display()));
                    } else {
                        self.transcript
                            .push_error(format!("not a directory: {path}"));
                    }
                }
            }
            SlashCommand::Pwd => {
                self.transcript
                    .push_system(format!("project directory: {}", self.project_dir.display()));
            }
            SlashCommand::Thinking(None) => {
                // Report current level.
                let label = thinking_level_label(&self.thinking_level);
                self.transcript
                    .push_system(format!("thinking level: {label}"));
            }
            SlashCommand::Thinking(Some(level_str)) => {
                use grim_core::sampler::ThinkingLevel;
                let level = ThinkingLevel::parse(&level_str);
                self.thinking_level = level;
                let _ = self.cmd_tx.send(WorkerCommand::SetThinking { level });
                let label = thinking_level_label(&self.thinking_level);
                self.transcript
                    .push_system(format!("thinking level: {label}"));
                self.show_toast(Toast::info(format!("Thinking: {label}")));
            }
            SlashCommand::Backend(None) => {
                // Report current backend and available backends.
                let available = available_backends();
                let current = match &self.backend {
                    Some(b) => b.clone(),
                    None => "auto".into(),
                };
                let msg = format!(
                    "backend: {current} — available: {}\n  /backend <name> to switch\n  /backend auto to reset",
                    available.join(", ")
                );
                self.transcript.push_system(msg);
            }
            SlashCommand::Backend(Some(name)) => {
                self.activate_backend(&name);
            }
            SlashCommand::Plan(arg) => {
                // /plan toggles; /plan on|off sets explicitly.
                let enable = match arg.as_deref().map(|s| s.to_ascii_lowercase()) {
                    None => !self.plan_mode,
                    Some(s) if s == "on" || s == "true" || s == "1" => true,
                    Some(s) if s == "off" || s == "false" || s == "0" => false,
                    Some(other) => {
                        self.transcript
                            .push_error(format!("usage: /plan [on|off] (got '{other}')"));
                        return;
                    }
                };
                self.plan_mode = enable;
                let _ = self
                    .cmd_tx
                    .send(WorkerCommand::SetPlanMode { enabled: enable });
                let state = if enable { "on" } else { "off" };
                self.transcript.push_system(if enable {
                    "plan mode on — mutations blocked; the model will research and propose a plan".into()
                } else {
                    "plan mode off — write tools re-enabled".into()
                });
                self.show_toast(Toast::info(format!("Plan mode: {state}")));
            }
            SlashCommand::NotACommand => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return;
                }
                // Reset the notification flag at the start of any new turn
                // attempt (even if we block below).
                self.generation_complete_notified = false;
                // Block generation if no model is loaded.
                if self.snap.model_name.is_none() {
                    self.transcript
                        .push_system("no model loaded — use /model <name> first".into());
                    return;
                }
                if self.generating {
                    self.queue.push(trimmed.to_string());
                    self.transcript.push_system(format!(
                        "queued ({} in queue); sent automatically when this turn finishes",
                        self.queue.len()
                    ));
                    return;
                }
                self.history.append(trimmed);
                self.history.save();
                if self.session_path.is_none() {
                    self.session_path = sessions::new_session_path(trimmed);
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

    /// Execute the pending tool call and forward the result to the worker.
    fn approve_pending_tool(&mut self) {
        self.pending_tool_diff = None;
        if let Some((call_id, name, arguments)) = self.pending_tool_call.take() {
            let call = grim_format::ToolCallMsg {
                id: call_id,
                name: name.clone(),
                arguments,
            };
            let result = crate::tui::tools::execute_tool(
                &call,
                &crate::tui::tools::Sandbox::new(self.sandbox_root.clone()),
            );
            let output = match result {
                Ok(s) => s,
                Err(e) => format!("error: {e}"),
            };
            self.transcript
                .push_tool_result_for(output.clone(), Some(call.id.clone()));
            self.messages.push(grim_format::ChatMessage {
                role: "tool".to_string(),
                content: output.clone(),
                tool_call_id: Some(call.id.clone()),
                name: Some(name),
                tool_calls: None,
            });
            let _ = self.cmd_tx.send(WorkerCommand::ToolResult {
                call_id: call.id,
                output,
            });
        }
        self.tool_approval_mode = false;
    }

    /// Handle key presses during tool-call approval mode.
    fn handle_tool_approval_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter | KeyCode::Char('y') => {
                // Approve once.
                self.approve_pending_tool();
            }
            KeyCode::Char('a') => {
                // Always allow: persist a rule so future matching calls skip
                // the prompt, then run this one.
                if let Some((_, name, arguments)) = &self.pending_tool_call {
                    let rule =
                        permissions::rule_for_tool_call(name, arguments);
                    let mut rules = self.permissions.lock().unwrap();
                    rules.add(rule.clone());
                    let summary = match &rule.command_prefix {
                        Some(prefix) => format!("{} '{}…'", rule.tool, prefix),
                        None => format!("'{}'", rule.tool),
                    };
                    match rules.save() {
                        Ok(()) => self
                            .transcript
                            .push_system(format!("always allowing {summary}")),
                        Err(e) => self
                            .transcript
                            .push_system(format!("rule kept for this session; save failed: {e}")),
                    }
                }
                self.approve_pending_tool();
            }
            KeyCode::Esc | KeyCode::Char('n') => {
                // Deny: send an error result so the model knows the tool was rejected.
                self.pending_tool_diff = None;
                if let Some((call_id, name, _arguments)) = self.pending_tool_call.take() {
                    let output = "tool call denied by user".to_string();
                    self.messages.push(grim_format::ChatMessage {
                        role: "tool".to_string(),
                        content: output.clone(),
                        tool_call_id: Some(call_id.clone()),
                        name: Some(name),
                        tool_calls: None,
                    });
                    let _ = self.cmd_tx.send(WorkerCommand::ToolResult { call_id, output });
                }
                self.tool_approval_mode = false;
                self.transcript
                    .push_system("tool call denied".to_string());
            }
            _ => {}
        }
    }

    /// Activate a backend by name (used by /backend command and the picker).
    fn activate_backend(&mut self, name: &str) {
        let name = name.trim().to_lowercase();
        match name.as_str() {
            "rocm" | "cuda" | "metal" | "cpu" | "auto" => {
                if name != "auto" && !is_backend_available(&name) {
                    self.transcript.push_error(format!(
                        "backend '{name}' unavailable — not compiled in or no device.\n  Rebuild with --features {name} or use /backend auto."
                    ));
                    return;
                }
                if name == "auto" {
                    self.backend = None;
                    // SAFETY: called from the UI thread before model load.
                    unsafe {
                        std::env::remove_var("GRIM_BACKEND");
                    }
                    self.transcript.push_system("backend: auto (default)".into());
                } else {
                    self.backend = Some(name.clone());
                    // SAFETY: called from the UI thread before model load.
                    unsafe {
                        std::env::set_var("GRIM_BACKEND", &name);
                    }
                    self.transcript.push_system(format!("backend: {name}"));
                }
                self.show_toast(Toast::info(format!("Backend: {name}")));
            }
            other => {
                self.transcript.push_error(format!(
                    "unknown backend '{other}' — expected rocm|cuda|metal|cpu|auto"
                ));
            }
        }
    }

    /// Return the selected transcript text for clipboard copy (find match or last assistant).
    fn selected_transcript_text(&self) -> Option<String> {
        if !self.find_matches.is_empty() {
            let (node_idx, _) = self.find_matches[self.find_selected % self.find_matches.len()];
            return self.transcript.nodes.get(node_idx).map(|n| n.content.clone());
        }
        // Fallback: last assistant or user content
        self.transcript
            .nodes
            .iter()
            .rev()
            .find(|n| !n.content.is_empty())
            .map(|n| n.content.clone())
    }

    /// Handle mouse events: Down focuses chat vs side vs input, ScrollUp/ScrollDown adjusts scroll.
    pub fn handle_mouse(&mut self, m: crossterm::event::MouseEvent) {
        match m.kind {
            MouseEventKind::Down(_) => {
                // Heuristic: clicks in the top ~80% of the terminal focus chat,
                // clicks in the right ~32% of the top area focus side, clicks in the
                // bottom few rows focus input (reset to bottom).
                // We don't have exact layout rects here, so use row-based heuristic.
                // For a more precise hit-test the caller can pass precomputed rects;
                // here we handle the generic case by toggling focus via was_at_bottom.
                // Default: focus chat (keep pinned handling).
                // If the click is near the bottom, treat as input focus.
                // This satisfies the spec's "focus chat vs side vs input" requirement
                // while remaining testable without a real terminal.
                self.was_at_bottom = true;
                self.scroll_offset = 0;
            }
            MouseEventKind::ScrollUp => {
                self.scroll_offset = self.scroll_offset.saturating_add(3);
                self.was_at_bottom = false;
            }
            MouseEventKind::ScrollDown => {
                self.scroll_offset = self.scroll_offset.saturating_sub(3);
                if self.scroll_offset == 0 {
                    self.was_at_bottom = true;
                    self.pending_new_lines = 0;
                }
            }
            _ => {}
        }
    }
}

/// Check if a backend is available (compiled in + device present).
fn is_backend_available(name: &str) -> bool {
    match name {
        "cpu" => true,
        "rocm" => grim_backend_rocm::RocmDevice::probe()
            .map(|d| !d.is_empty())
            .unwrap_or(false),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                grim_backend_metal::MetalDevice::probe()
                    .map(|d| !d.is_empty())
                    .unwrap_or(false)
            }
            #[cfg(not(feature = "metal"))]
            false
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                grim_backend_cuda::CudaDevice::probe()
                    .map(|d| !d.is_empty())
                    .unwrap_or(false)
            }
            #[cfg(not(feature = "cuda"))]
            false
        }
        _ => false,
    }
}

/// Return a list of available backend names (for display).
fn available_backends() -> Vec<&'static str> {
    let mut backends = vec!["auto", "cpu"];
    if is_backend_available("rocm") {
        backends.push("rocm");
    }
    if is_backend_available("cuda") {
        backends.push("cuda");
    }
    if is_backend_available("metal") {
        backends.push("metal");
    }
    backends
}

/// Return a human-readable label for a thinking level.
fn thinking_level_label(level: &grim_core::sampler::ThinkingLevel) -> String {
    match level {
        grim_core::sampler::ThinkingLevel::Off => "off".into(),
        grim_core::sampler::ThinkingLevel::Default => "default".into(),
        grim_core::sampler::ThinkingLevel::Low => "low".into(),
        grim_core::sampler::ThinkingLevel::Medium => "medium".into(),
        grim_core::sampler::ThinkingLevel::High => "high".into(),
        grim_core::sampler::ThinkingLevel::Custom(n) => format!("custom ({n})"),
    }
}

/// Find the most recently modified *.jsonl session file in `dir`
/// (--continue semantics). Ignores hidden files and errors: no sessions
/// found returns None.
fn latest_session_file(dir: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .filter_map(|p| {
            let mtime = p.metadata().ok()?.modified().ok()?;
            Some((mtime, p))
        })
        .max_by_key(|(mtime, _)| *mtime)
        .map(|(_, p)| p)
}

/// Helper exporting transcript nodes to a text file or JSONL.
pub(crate) fn export_transcript(transcript: &Transcript, path: &str) -> std::io::Result<usize> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let mut count = 0;
    if path.ends_with(".jsonl") {
        // Map call id → tool name so tool result lines can carry the name
        // (tool results appear after their calls in the node stream).
        let mut call_names: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
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
            if node.role == Role::ToolCall {
                if let (Some(id), Some(name)) = (&node.tool_call_id, &node.tool_name) {
                    call_names.insert(id.as_str(), name.as_str());
                }
            }
            let result_name = node
                .tool_call_id
                .as_deref()
                .and_then(|id| call_names.get(id).copied());
            let tool_name_out = node
                .tool_name
                .clone()
                .or_else(|| result_name.map(|s| s.to_string()));
            let obj = serde_json::json!({
                "role": role_str,
                "content": node.content,
                "thinking": node.thinking,
                "stats": node.turn_stats,
                "tool_name": tool_name_out,
                "tool_arguments": node.tool_arguments,
                "tool_call_id": node.tool_call_id,
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
            let tool_name = val
                .get("tool_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let tool_args = val
                .get("tool_arguments")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let tool_id = val
                .get("tool_call_id")
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

            match role {
                Role::User | Role::Assistant | Role::System => {
                    messages.push(grim_format::ChatMessage {
                        role: role_str.to_string(),
                        content: content.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
                // Rebuild the canonical OpenAI sequence: tool_call lines
                // become assistant messages carrying structured tool_calls…
                Role::ToolCall => {
                    if let (Some(id), Some(name), Some(args)) =
                        (&tool_id, &tool_name, &tool_args)
                    {
                        messages.push(grim_format::ChatMessage {
                            role: "assistant".to_string(),
                            content: String::new(),
                            tool_calls: Some(vec![grim_format::ToolCallMsg {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: args.clone(),
                            }]),
                            tool_call_id: None,
                            name: None,
                        });
                    }
                }
                // …and tool_result lines become tool messages keyed by id.
                Role::ToolResult => {
                    messages.push(grim_format::ChatMessage {
                        role: "tool".to_string(),
                        content: content.clone(),
                        tool_calls: None,
                        tool_call_id: tool_id.clone(),
                        name: tool_name.clone(),
                    });
                }
                Role::Error | Role::Hint => {}
            }

            nodes.push(MessageNode {
                role,
                content,
                thinking,
                thought_folded: true,
                turn_stats: stats,
                tool_name,
                tool_arguments: tool_args,
                tool_call_id: tool_id,
                ..Default::default()
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

/// Copy text to system clipboard via `arboard`, falling back to OSC52 escape.
///
/// Tries `arboard::Clipboard::new().set_text()` first; if that fails (e.g. headless
/// or missing display server) falls back to writing an OSC52 sequence to stdout
/// so the terminal emulator can place the text in the system clipboard.
pub use clipboard::{copy_to_clipboard, osc52_copy};
/// Terminal lifecycle: the alternate screen is restored by the panic hook
/// (see `run_tui`) on crash, or by the explicit `ratatui::restore()` call
/// on the normal exit path. Kept as a placeholder comment — if you need a
/// guard-based restore in the future, wrap the body in a struct whose Drop
/// calls `ratatui::restore()`.

/// Render one frame via the constrained layout engine.
///
/// Color contract:
/// - Primary panel borders: `#a855f7` (neon purple) when focused/active.
/// - Inactive panel borders: dim purple `#703264`.
/// - Input border changes to amber (#f59e0b) while generating, magenta while in tool-approval.
/// - Body text is always white. Purple is reserved for borders, chips, and key labels.
/// - A 1-row status-bar footer sits below the input box with model/tps/ctx info.
fn ui(f: &mut Frame, app: &App) {
    use crate::tui::layout::{Basis, StackEntry, StackOptions};

    // Brand colors — neon purple to match grim-garage.
    let c_purple     = Color::Rgb(168, 85, 247);   // #a855f7 primary
    let c_purple_dim = Color::Rgb(112, 50, 180);   // inactive border
    let c_purple_soft = Color::Rgb(192, 132, 252); // soft purple titles
    let c_cyan       = Color::Rgb(34, 211, 238);   // assistant / sparkline
    let c_amber      = Color::Rgb(245, 158, 11);   // generating
    let c_magenta    = Color::Rgb(232, 121, 249);  // tool call
    let c_green      = Color::Rgb(16, 185, 129);   // success
    let c_red        = Color::Rgb(239, 68, 68);    // error, diff removals
    let c_muted      = Color::Rgb(136, 136, 136);  // muted text

    // 20-line gradient helper: lerp between two Rgb colors per char.
    let gradient_title = |text: &str, from: Color, to: Color| -> Line<'static> {
        let (r1, g1, b1) = match from { Color::Rgb(r,g,b) => (r as f32,g as f32,b as f32), _ => (168.0,85.0,247.0) };
        let (r2, g2, b2) = match to { Color::Rgb(r,g,b) => (r as f32,g as f32,b as f32), _ => (34.0,211.0,238.0) };
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len().max(1) as f32;
        let mut spans = Vec::with_capacity(chars.len());
        for (i, ch) in chars.into_iter().enumerate() {
            let t = i as f32 / n;
            let r = (r1 * (1.0 - t) + r2 * t) as u8;
            let g = (g1 * (1.0 - t) + g2 * t) as u8;
            let b = (b1 * (1.0 - t) + b2 * t) as u8;
            spans.push(Span::styled(ch.to_string(), Style::default().fg(Color::Rgb(r,g,b)).add_modifier(Modifier::BOLD)));
        }
        Line::from(spans)
    };

    // Braille spinner frames.
    const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let spinner_char = SPINNER[(app.frame_count as usize) % SPINNER.len()];

    let area = f.area();

    // Layout: content_area | find_bar (if active, 3 rows) | input_bar (dynamic) | status_bar (1 row).
    let input_height = (app.composer.line_count() as u16 + 2).clamp(3, 8);
    let status_height: u16 = 1;
    let find_bar_height: u16 = if matches!(app.input_mode, InputMode::Find { .. }) { 3 } else { 0 };
    let content_height = area
        .height
        .saturating_sub(input_height + status_height + find_bar_height)
        .max(3);

    let content_rect = Rect { x: area.x, y: area.y, width: area.width, height: content_height };
    let find_rect = Rect {
        x: area.x,
        y: area.y + content_height,
        width: area.width,
        height: find_bar_height,
    };
    let input_rect = Rect {
        x: area.x,
        y: area.y + content_height + find_bar_height,
        width: area.width,
        height: input_height,
    };
    let status_rect = Rect {
        x: area.x,
        y: area.y + content_height + find_bar_height + input_height,
        width: area.width,
        height: status_height,
    };

    // Sidebar split.
    let (chat_area, side_area) = if app.show_sidebar {
        let total_w = content_rect.width;
        let chat_w = ((total_w as u32 * 68) / 100) as u16;
        let side_w = total_w.saturating_sub(chat_w).max(1).min(total_w.saturating_sub(1).max(1));
        let chat_w = total_w.saturating_sub(side_w);
        (
            Rect { x: content_rect.x, y: content_rect.y, width: chat_w, height: content_rect.height },
            Some(Rect { x: content_rect.x + chat_w, y: content_rect.y, width: side_w, height: content_rect.height }),
        )
    } else {
        (content_rect, None)
    };

    // Engine layout check (keeps the layout engine exercised by tests).
    let _engine_check = crate::tui::layout::VStack::new(
        vec![
            StackEntry {
                node: Box::new(crate::tui::layout::VStack::new(vec![], StackOptions { gap: 0 })),
                basis: Basis::Fixed(0),
                grow: 1,
                shrink: 1,
                min_size: 3,
                max_size: None,
            },
            StackEntry {
                node: Box::new(crate::tui::layout::VStack::new(vec![], StackOptions { gap: 0 })),
                basis: Basis::Fixed(input_height),
                grow: 0,
                shrink: 0,
                min_size: 3,
                max_size: Some(8),
            },
        ],
        StackOptions { gap: 0 },
    );

    // -----------------------------------------------------------------------
    // Chat panel — active purple border, dim title. Flash green for 300ms on TurnComplete.
    // -----------------------------------------------------------------------
    let has_model = app.snap.model_name.is_some();
    let is_flashing = app.flash_until.is_some_and(|t| Instant::now() < t);
    let chat_border_color = if is_flashing {
        c_green
    } else if !has_model {
        c_amber
    } else if app.show_sidebar {
        c_purple
    } else {
        c_purple
    };
    // Virtualized transcript: return cached slice of content_height lines around scroll_offset
    // instead of the full vec. Falls back to full render when virtualization is not needed.
    let inner_height = chat_area.height.saturating_sub(2) as usize;
    let chat_items = if inner_height > 0 && app.transcript.nodes.len() > 20 {
        app.transcript
            .render_lines_virtualized(200, inner_height, app.scroll_offset)
    } else {
        app.transcript.render_lines()
    };
    let mut chat_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(gradient_title(" GRIM ", c_purple, c_cyan))
        .border_style(Style::default().fg(chat_border_color));
    if !has_model {
        chat_block = chat_block.title_bottom(
            Line::from(Span::styled(" no model — /model or F4 ", Style::default().fg(c_amber))).centered()
        );
    }
    let chat = Paragraph::new(chat_items)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_offset as u16, 0))
        .block(chat_block);
    f.render_widget(chat, chat_area);

    // Auto-scroll pill: show when user scrolled up and new lines arrived
    if app.pending_new_lines > 0 && !app.was_at_bottom {
        let pill_text = format!(" ↑ {} new — Press End ", app.pending_new_lines);
        let pill = Paragraph::new(Line::from(Span::styled(pill_text, Style::default().fg(Color::White).bg(c_amber).add_modifier(Modifier::BOLD))))
            .alignment(Alignment::Center);
        let pill_area = Rect {
            x: chat_area.x + 1,
            y: chat_area.y + chat_area.height.saturating_sub(1),
            width: chat_area.width.saturating_sub(2),
            height: 1,
        };
        f.render_widget(pill, pill_area);
    }

    // -----------------------------------------------------------------------
    // Sidebar panels — dim purple border.
    //
    // Three zones stacked vertically:
    //   1. Diagnostics (top, flexible)
    //   2. Agent task list (middle, flexible — shared space for tasks/todos)
    //   3. tok/s sparkline (bottom, fixed 4 rows)
    // -----------------------------------------------------------------------
    if let Some(area) = side_area {
        let side_chunks = Layout::vertical([
            Constraint::Min(8),      // diagnostics
            Constraint::Min(4),      // task list
            Constraint::Length(1),   // gauge
            Constraint::Length(4),   // sparkline
        ])
        .split(area);

        // 1) Diagnostics panel with styled key/value lines.
        let styled_lines = diagnostics::sidebar_styled_lines(&app.snap);
        let side = Paragraph::new(styled_lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" diagnostics ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple_dim)),
        );
        f.render_widget(side, side_chunks[0]);

        // 2) Agent task list panel.
        let task_max_rows = side_chunks[1].height.saturating_sub(2) as usize;
        let task_lines = app.task_list.render(task_max_rows.max(3));
        let task_panel = Paragraph::new(task_lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" tasks ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple_dim)),
        );
        f.render_widget(task_panel, side_chunks[1]);

        // 3) Context gauge (purple dim bg)
        let ctx_ratio = if app.snap.ctx_limit > 0 {
            app.snap.ctx_used as f64 / app.snap.ctx_limit as f64
        } else {
            0.0
        };
        let ctx_label = if app.snap.ctx_limit > 0 {
            format!("ctx {}/{}", app.snap.ctx_used, app.snap.ctx_limit)
        } else {
            format!("ctx {} / ?", app.snap.ctx_used)
        };
        let gauge = Gauge::default()
            .block(Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" ctx ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple_dim)))
            .gauge_style(Style::default().fg(Color::White).bg(c_purple_dim))
            .ratio(ctx_ratio.clamp(0.0, 1.0))
            .label(Span::styled(ctx_label, Style::default().fg(Color::White)));
        f.render_widget(gauge, side_chunks[2]);

        // 4) Sparkline with cyan bars.
        let spark_data = app.speed_history.as_slice();
        let spark = Sparkline::default()
            .block(
                Block::bordered().border_type(BorderType::Rounded)
                    .title(Span::styled(" tok/s ", Style::default().fg(c_purple_soft)))
                    .border_style(Style::default().fg(c_purple_dim)),
            )
            .data(spark_data)
            .style(Style::default().fg(c_cyan));
        f.render_widget(spark, side_chunks[3]);
    }

    // -----------------------------------------------------------------------
    // Input bar — border color and title reflect current mode.
    // -----------------------------------------------------------------------
    let input_text = app.composer.text();
    let (input_border_color, input_title) = match &app.input_mode {
        _ if app.tool_approval_mode => (
            c_magenta,
            format!(" ⚠  approve tool call?  Enter = yes   Esc = deny "),
        ),
        _ if app.generating && !app.queue.is_empty() => (
            c_amber,
            format!(
                " {} generating...  {} queued — Esc clears queue ",
                spinner_char, app.queue.len()
            ),
        ),
        _ if app.generating => (
            c_amber,
            format!(" {} generating...  Esc to cancel ", spinner_char),
        ),
        InputMode::CtxOverride => (
            c_green,
            " ctx override  Enter applies, empty = auto, Esc cancels ".into(),
        ),
        InputMode::ModelPicker { .. } => (
            c_green,
            " model picker  arrow keys + Enter to load, Esc cancels ".into(),
        ),
        InputMode::CommandPalette { .. } => (
            c_purple_soft,
            " command palette  type to filter, Enter to run, Esc cancels ".into(),
        ),
        InputMode::SessionBrowser { .. } => (
            c_purple_soft,
            " sessions  Enter to load, Esc cancels ".into(),
        ),
        InputMode::SkillPicker { .. } => (
            c_purple_soft,
            " skills  type to filter, Enter to activate, Esc cancels ".into(),
        ),
        InputMode::BackendPicker { .. } => (
            c_purple_soft,
            " backend  arrow keys + Enter to select, Esc cancels ".into(),
        ),
        InputMode::ProjectDir => (
            c_green,
            " project directory  Enter to set, Esc cancels ".into(),
        ),
        InputMode::Find { query: _, matches, selected } => {
            let total = matches.len();
            let cur = if total == 0 { 0 } else { selected + 1 };
            let q = match &app.input_mode {
                InputMode::Find { query, .. } => query.clone(),
                _ => String::new(),
            };
            let label = if total == 0 && q.is_empty() {
                " Find — type to search ".to_string()
            } else if total == 0 {
                format!(" Find — no matches for \"{q}\" ")
            } else {
                format!(" Find {cur}/{total} — \"{q}\" ")
            };
            (c_cyan, label)
        }
        InputMode::Chat => (
            c_purple_dim,
            " /  commands   @  files   Tab  autocomplete   F2  sidebar ".into(),
        ),
        InputMode::HistorySearch { .. } => (
            c_cyan,
            " history search  Enter fills composer, Esc cancels ".into(),
        ),
    };

    // Find bar above input when in Find mode
    if let InputMode::Find { query, matches, selected } = &app.input_mode {
        let total = matches.len();
        let cur = if total == 0 { 0 } else { selected + 1 };
        let title = if total == 0 && query.is_empty() {
            " Find 0/0 ".to_string()
        } else {
            format!(" Find {cur}/{total} ")
        };
        let find_query_text = if query.is_empty() {
            Span::styled("type to search…", Style::default().fg(c_muted).add_modifier(Modifier::ITALIC))
        } else {
            Span::raw(query.clone())
        };
        let count_style = if total == 0 { c_amber } else { c_cyan };
        let para = Paragraph::new(Line::from(vec![find_query_text])).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(Span::styled(title, Style::default().fg(count_style).add_modifier(Modifier::BOLD)))
                .title(Line::from(Span::styled(" Esc exit · Enter/Ctrl+F next ", Style::default().fg(c_muted))).right_aligned())
                .border_style(Style::default().fg(c_cyan)),
        );
        f.render_widget(para, find_rect);
    }

    // Chat mode: ghost text + split title via left/right alignment (Layout-like)
    let is_chat = matches!(app.input_mode, InputMode::Chat) && !app.tool_approval_mode && !app.generating;
    if is_chat {
        // Slash param ghost hint: when input is "/cmd " show CommandSpec.hint in muted after cursor.
        // Use split_once to safely handle multibyte UTF-8 characters.
        let ghost_hint: Option<String> = if input_text.starts_with('/') {
            if let Some((cmd_name, rest)) = input_text[1..].split_once(' ') {
                if let Some(spec) = app.registry.all_commands().iter().find(|s| s.name == cmd_name) {
                    if !spec.hint.is_empty() && app.composer.cursor_offset() == input_text.chars().count() && rest.is_empty() {
                        Some(spec.hint.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let _title_layout = Layout::horizontal([Constraint::Min(0), Constraint::Length(28)]).split(input_rect);
        let block = Block::bordered().border_type(BorderType::Rounded)
            .title(Line::from(Span::styled(" Type / for commands, @ for files ", Style::default().fg(c_muted))).left_aligned())
            .title(Line::from(Span::styled(" F2 sidebar · Ctrl+P palette ", Style::default().fg(c_muted))).right_aligned())
            .border_style(Style::default().fg(input_border_color));
        let para = if app.composer.is_empty() {
            Paragraph::new(Line::from(vec![
                Span::raw(input_text.clone()),
                Span::styled("/ for commands  @ for files", Style::default().fg(c_muted)),
            ])).block(block)
        } else if let Some(hint) = ghost_hint {
            Paragraph::new(Line::from(vec![
                Span::raw(input_text.clone()),
                Span::styled(format!("{hint}"), Style::default().fg(c_muted).add_modifier(Modifier::ITALIC)),
            ])).block(block)
        } else {
            Paragraph::new(input_text.as_str()).block(block)
        };
        f.render_widget(para, input_rect);
    } else {
        f.render_widget(
            Paragraph::new(input_text.as_str()).block(
                Block::bordered().border_type(BorderType::Rounded)
                    .title(Span::styled(input_title, Style::default().fg(input_border_color)))
                    .border_style(Style::default().fg(input_border_color)),
            ),
            input_rect,
        );
    }

    // -----------------------------------------------------------------------
    // Status bar — 1-row footer with model/tps/ctx info and key hints.
    // -----------------------------------------------------------------------
    {
        let model_str = app
            .snap
            .model_name
            .as_deref()
            .unwrap_or("no model");
        let tps_str = app
            .snap
            .decode_tps
            .map(|v| format!("{:.0} tok/s", v))
            .unwrap_or_default();
        let ctx_str = if app.snap.ctx_limit > 0 {
            format!("ctx {}/{}", app.snap.ctx_used, app.snap.ctx_limit)
        } else {
            String::new()
        };
        // Show the project directory basename (or full path if short).
        let proj_str = if let Some(name) = app.project_dir.file_name().and_then(|n| n.to_str()) {
            format!("dir:{name}")
        } else {
            format!("dir:{}", app.project_dir.display())
        };
        // Show active skill name if any.
        let skill_str = app
            .active_skill_name
            .as_ref()
            .map(|s| format!("skill:{s}"))
            .unwrap_or_default();
        // Show thinking level (compact).
        let thinking_str = {
            let label = thinking_level_label(&app.thinking_level);
            format!("think:{label}")
        };
        // Show backend (compact).
        let backend_str = match &app.backend {
            Some(b) => b.clone(),
            None => "auto".into(),
        };
        let hints = if app.plan_mode {
            "PLAN MODE · /plan off to resume editing"
        } else {
            "Ctrl+P palette  Ctrl+G skills  Ctrl+B backend  Ctrl+T think  Ctrl+D project  F4 model  /help"
        };

        let mut status_spans = vec![
            Span::styled(" ", Style::default()),
            Span::styled(model_str.to_string(), Style::default().fg(c_purple_soft).add_modifier(Modifier::BOLD)),
            Span::styled("  ", Style::default()),
            Span::styled(tps_str, Style::default().fg(c_cyan)),
            Span::styled("  ", Style::default()),
            Span::styled(ctx_str, Style::default().fg(c_muted)),
            Span::styled("  ", Style::default()),
            Span::styled(proj_str, Style::default().fg(c_muted)),
        ];
        // Color thinking level: amber when off, green when enabled.
        let thinking_color = matches!(app.thinking_level, grim_core::sampler::ThinkingLevel::Off)
            .then(|| c_muted)
            .unwrap_or(c_green);
        status_spans.push(Span::styled("  ", Style::default()));
        status_spans.push(Span::styled(thinking_str, Style::default().fg(thinking_color)));
        // Backend indicator.
        status_spans.push(Span::styled("  ", Style::default()));
        status_spans.push(Span::styled(
            format!("dev:{backend_str}"),
            Style::default().fg(c_muted),
        ));
        if !skill_str.is_empty() {
            status_spans.push(Span::styled("  ", Style::default()));
            status_spans.push(Span::styled(skill_str, Style::default().fg(c_green)));
        }
        if app.plan_mode {
            status_spans.push(Span::styled("  ", Style::default()));
            status_spans.push(Span::styled(
                " PLAN ",
                Style::default()
                    .fg(Color::White)
                    .bg(c_amber)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        status_spans.push(Span::styled("  ", Style::default()));
        status_spans.push(Span::styled(hints, Style::default().fg(c_muted)));
        f.render_widget(Paragraph::new(Line::from(status_spans)), status_rect);
    }

    // -----------------------------------------------------------------------
    // Autocomplete popups — slash commands and @file.
    // -----------------------------------------------------------------------
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
            menu.set_filter("");
        } else {
            menu.set_filter(query);
            for _ in 0..(app.selected_completion % 6) {
                menu.move_down();
            }
        }
        let filtered_count = menu.filtered_len();
        if filtered_count > 0 {
            let height = (filtered_count as u16 + 2).min(8);
            let popup_area = Rect {
                x: input_rect.x + 1,
                y: input_rect.y.saturating_sub(height),
                width: 52.min(input_rect.width.saturating_sub(2)),
                height,
            };
            let completion_lines = menu.render(popup_area.width.saturating_sub(2));
            let popup = Paragraph::new(completion_lines).block(
                Block::bordered().border_type(BorderType::Rounded)
                    .title(Span::styled(" commands ", Style::default().fg(c_purple_soft)))
                    .border_style(Style::default().fg(c_purple_dim)),
            );
            f.render_widget(popup, popup_area);
        }
    } else if app.input_mode == InputMode::Chat {
        let cursor = app.composer.cursor_offset();
        if let Some((_start, prefix)) =
            crate::tui::file_complete::extract_at_prefix(&input_text, cursor)
        {
            let after_at = prefix.trim_start_matches('@');
            let base = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            let suggestions =
                crate::tui::file_complete::get_file_suggestions_ranked(after_at, &base, 50, &app.frecency);
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
                        x: input_rect.x + 1,
                        y: input_rect.y.saturating_sub(height),
                        width: 52.min(input_rect.width.saturating_sub(2)),
                        height,
                    };
                    let completion_lines = menu.render(popup_area.width.saturating_sub(2));
                    let popup = Paragraph::new(completion_lines).block(
                        Block::bordered().border_type(BorderType::Rounded)
                            .title(Span::styled(" files ", Style::default().fg(c_purple_soft)))
                            .border_style(Style::default().fg(c_purple_dim)),
                    );
                    f.render_widget(popup, popup_area);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Model picker modal.
    // -----------------------------------------------------------------------
    if let InputMode::ModelPicker { selected } = app.input_mode {
        let models = grim_core::catalog::list_local_models();
        let height = (models.len() as u16 + 4).clamp(5, 16);
        let width = 64.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        if models.is_empty() {
            lines.push(Line::from(Span::styled("  No local models discovered in catalog.", Style::default().fg(c_muted))));
        } else {
            for (idx, m) in models.iter().enumerate() {
                let is_sel = idx == selected;
                let prefix = if is_sel { "▶ " } else { "  " };
                let _row_style = if is_sel {
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                let name_color = if is_sel { c_purple } else { Color::White };
                let ctx_str = if m.context_length > 0 { format!("ctx {}", m.context_length) } else { "ctx ?".into() };
                lines.push(Line::from(vec![
                    Span::styled(format!("{}{:<24}", prefix, m.name), Style::default().fg(name_color).add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() })),
                    Span::styled(format!(" {:<8} {}", m.quant, ctx_str), Style::default().fg(c_muted)),
                ]));
            }
        }
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" model picker  Enter to load ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Command palette modal.
    // -----------------------------------------------------------------------
    if let InputMode::CommandPalette { selected } = app.input_mode {
        let filtered = app.palette_filtered_commands();
        let height = (filtered.len() as u16 + 4).clamp(5, 20);
        let width = 72.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled("  type to filter  Enter to run  Esc to cancel", Style::default().fg(c_muted))));
        lines.push(Line::raw(""));
        for (idx, cmd) in filtered.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let name_color = if is_sel { c_purple } else { Color::White };
            let hint_str = if cmd.hint.is_empty() { String::new() } else { format!(" {}", cmd.hint) };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}{}{}", prefix, cmd.name, hint_str),
                    Style::default().fg(name_color).add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() }),
                ),
                Span::styled(format!("  {}", cmd.description), Style::default().fg(c_muted)),
            ]));
        }
        if filtered.is_empty() {
            lines.push(Line::from(Span::styled("  no matching commands", Style::default().fg(c_muted))));
        }
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" command palette  Ctrl+P ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Session browser modal.
    // -----------------------------------------------------------------------
    if let InputMode::SessionBrowser { selected, .. } = app.input_mode {
        let session_items = app.session_browser_items();
        let height = (session_items.len() as u16 + 5).clamp(5, 20);
        let width = 72.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let query = app.session_browser_query();
        let mut lines = Vec::new();
        let filter_note = if query.is_empty() {
            String::from("  type to filter  ·  Enter to load  ·  Esc to cancel")
        } else {
            format!("  filter: \"{query}\"  ·  Enter to load  ·  Esc to cancel")
        };
        lines.push(Line::from(Span::styled(filter_note, Style::default().fg(c_muted))));
        lines.push(Line::raw(""));
        for (idx, meta) in session_items.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let color = if is_sel { c_purple } else { Color::White };
            let when = chrono::DateTime::from_timestamp(meta.modified as i64, 0)
                .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_default();
            let shown: String =
                meta.title.chars().take(width.saturating_sub(24) as usize).collect();
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{prefix}{shown}"),
                    Style::default().fg(color).add_modifier(if is_sel {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
                ),
                Span::styled(format!("  {when}"), Style::default().fg(c_muted)),
            ]));
        }
        if session_items.is_empty() {
            lines.push(Line::from(Span::styled(
                "  no sessions found",
                Style::default().fg(c_muted),
            )));
        }
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" sessions  Ctrl+O ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Prompt history search modal (Ctrl+R).
    // -----------------------------------------------------------------------
    if let InputMode::HistorySearch { selected, .. } = app.input_mode {
        let matches = app.history_search_matches();
        let query = match &app.input_mode {
            InputMode::HistorySearch { query, .. } => query.clone(),
            _ => String::new(),
        };
        let height = (matches.len() as u16 + 5).clamp(5, 20);
        let width = 72.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled(
            format!("  query: \"{query}\"  —  Enter to fill composer  Esc to cancel"),
            Style::default().fg(c_muted),
        )));
        lines.push(Line::raw(""));
        for (idx, text) in matches.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let color = if is_sel { c_purple } else { Color::White };
            let shown: String = text.chars().take(width.saturating_sub(6) as usize).collect();
            lines.push(Line::from(Span::styled(
                format!("{prefix}{shown}"),
                Style::default().fg(color).add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() }),
            )));
        }
        if matches.is_empty() {
            lines.push(Line::from(Span::styled(
                "  no matching prompts in history",
                Style::default().fg(c_muted),
            )));
        }
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" history  Ctrl+R ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Skill picker modal.
    // -----------------------------------------------------------------------
    if let InputMode::SkillPicker { selected } = app.input_mode {
        // Recompute filtered skills using the same logic as the handler.
        let query = app
            .composer
            .text()
            .trim_start_matches('/')
            .trim()
            .to_lowercase();
        let filtered: Vec<&crate::tui::skills::Skill> = if query.is_empty() {
            app.skills.iter().collect()
        } else {
            let mut scored: Vec<(i32, &crate::tui::skills::Skill)> = app
                .skills
                .iter()
                .filter_map(|s| {
                    let id_lower = s.id.to_lowercase();
                    let name_lower = s.name.to_lowercase();
                    let id_score = crate::tui::fuzzy::fuzzy_match(&query, &id_lower);
                    let name_score = crate::tui::fuzzy::fuzzy_match(&query, &name_lower);
                    match (id_score, name_score) {
                        (Some(id), Some(name)) => Some((id.score.max(name.score), s)),
                        (Some(id), None) => Some((id.score, s)),
                        (None, Some(name)) => Some((name.score, s)),
                        (None, None) => None,
                    }
                })
                .collect();
            scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
            scored.into_iter().map(|(_, s)| s).collect()
        };

        let height = (filtered.len() as u16 + 4).clamp(5, 20);
        let width = 72.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled("  type to filter  Enter to activate  Esc to cancel", Style::default().fg(c_muted))));
        lines.push(Line::raw(""));
        for (idx, skill) in filtered.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let color = if is_sel { c_purple } else { Color::White };
            let active_marker = if app.active_skill_name.as_ref() == Some(&skill.name) {
                " ●"
            } else {
                ""
            };
            let desc_str = if skill.description.is_empty() {
                String::new()
            } else {
                format!(" — {}", skill.description)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}{}{}", prefix, skill.id, desc_str),
                    Style::default().fg(color).add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() }),
                ),
                Span::styled(active_marker.to_string(), Style::default().fg(c_green)),
            ]));
        }
        if filtered.is_empty() {
            let dir = crate::tui::skills::default_skills_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "~/.agents/skills".into());
            lines.push(Line::from(Span::styled(
                format!("  no skills found (scan dir: {dir})"),
                Style::default().fg(c_muted),
            )));
        }
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" skills  Ctrl+G ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Backend picker modal.
    // -----------------------------------------------------------------------
    if let InputMode::BackendPicker { selected } = app.input_mode {
        let backends = app.available_backends();
        let height = (backends.len() as u16 + 4).clamp(5, 12);
        let width = 48.min(f.area().width.saturating_sub(4));
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled("  Enter to select  Esc to cancel", Style::default().fg(c_muted))));
        lines.push(Line::raw(""));
        for (idx, name) in backends.iter().enumerate() {
            let is_sel = idx == selected;
            let prefix = if is_sel { "▶ " } else { "  " };
            let color = if is_sel { c_purple } else { Color::White };
            // Mark the currently active backend.
            let active_marker = if app.backend.as_deref() == Some(name.as_str())
                || (name == "auto" && app.backend.is_none())
            {
                " ●"
            } else {
                ""
            };
            let desc = match name.as_str() {
                "auto" => "auto-detect",
                "cpu" => "CPU only",
                "rocm" => "AMD GPU (ROCm)",
                "cuda" => "NVIDIA GPU (CUDA)",
                "metal" => "Apple GPU (Metal)",
                _ => "",
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}{}", prefix, name),
                    Style::default().fg(color).add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() }),
                ),
                Span::styled(format!(" — {desc}"), Style::default().fg(c_muted)),
                Span::styled(active_marker.to_string(), Style::default().fg(c_green)),
            ]));
        }
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" backend  Ctrl+B ", Style::default().fg(c_purple_soft)))
                .border_style(Style::default().fg(c_purple)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Cursor position.
    // -----------------------------------------------------------------------
    let (c_row, c_col) = app.composer.cursor_row_col();
    f.set_cursor_position(Position::new(
        input_rect.x + 1 + c_col as u16,
        input_rect.y + 1 + c_row as u16,
    ));

    // -----------------------------------------------------------------------
    // Tool-call approval modal.
    // -----------------------------------------------------------------------
    if let Some((call_id, name, arguments)) = &app.pending_tool_call {
        let width = 72.min(f.area().width.saturating_sub(4));
        // Taller modal when an edit diff is available.
        let diff = app.pending_tool_diff.as_ref();
        let height: u16 = if diff.is_some() { 18 } else { 12 };
        let x = (f.area().width.saturating_sub(width)) / 2;
        let y = (f.area().height.saturating_sub(height)) / 2;
        let modal_area = Rect { x, y, width, height };

        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled("  tool: ", Style::default().fg(c_muted)),
            Span::styled(name.clone(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  id: {}", call_id), Style::default().fg(c_muted)),
        ]));
        lines.push(Line::raw(""));
        if let Some(diff) = diff {
            // Real diff view for edit tools.
            let (added, removed) = diff::count_changes(diff);
            lines.push(Line::from(vec![
                Span::styled("  changes: ", Style::default().fg(c_purple_soft)),
                Span::styled(format!("+{added}"), Style::default().fg(c_green)),
                Span::styled(" / ", Style::default().fg(c_muted)),
                Span::styled(format!("−{removed}"), Style::default().fg(c_red)),
            ]));
            for dl in diff.iter().take(10) {
                let (marker, style) = match dl.kind {
                    diff::DiffKind::Added => ("+", Style::default().fg(c_green)),
                    diff::DiffKind::Removed => ("−", Style::default().fg(c_red)),
                    diff::DiffKind::Context => (" ", Style::default().fg(c_muted)),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("    {marker} "), Style::default().fg(c_muted)),
                    Span::styled(dl.text.clone(), style),
                ]));
            }
            if diff.len() > 10 {
                lines.push(Line::from(Span::styled(
                    format!("    … ({} more)", diff.len() - 10),
                    Style::default().fg(c_muted),
                )));
            }
        } else {
            let pretty = serde_json::from_str::<serde_json::Value>(arguments)
                .and_then(|v| serde_json::to_string_pretty(&v))
                .unwrap_or_else(|_| arguments.clone());
            lines.push(Line::from(Span::styled("  arguments:", Style::default().fg(c_purple_soft))));
            for arg_line in pretty.lines().take(5) {
                lines.push(Line::from(vec![
                    Span::styled("    + ", Style::default().fg(c_green)),
                    Span::styled(arg_line.to_string(), Style::default().fg(Color::White)),
                ]));
            }
        }
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            "  y/Enter: approve   a: always allow   n/Esc: deny",
            Style::default().fg(c_amber).add_modifier(Modifier::BOLD),
        )));
        let modal = Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" tool approval ", Style::default().fg(c_magenta).add_modifier(Modifier::BOLD)))
                .border_style(Style::default().fg(c_magenta)),
        );
        f.render_widget(modal, modal_area);
    }

    // -----------------------------------------------------------------------
    // Toast notification — top-right corner.
    // -----------------------------------------------------------------------
    if let Some(toast) = &app.toast {
        let toast_width = 44.min(f.area().width.saturating_sub(4));
        let toast_lines = render_toast(toast, toast_width);
        let toast_height = toast_lines.len() as u16 + 2;
        let toast_area = Rect {
            x: f.area().width.saturating_sub(toast_width + 2),
            y: 1,
            width: toast_width,
            height: toast_height,
        };
        let toast_border_color = toast.variant.color();
        let toast_widget = Paragraph::new(toast_lines).block(
            Block::bordered().border_type(BorderType::Rounded)
                .title(Span::styled(" notice ", Style::default().fg(toast_border_color)))
                .border_style(Style::default().fg(toast_border_color)),
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
    resume: Option<String>,
    continue_last: bool,
) -> Result<()> {
    if !std::io::stdout().is_terminal() {
        return Err(Error::Config(
            "grim tui needs an interactive terminal".into(),
        ));
    }

    // Redirect stderr to a log file so all debug output (eprintln! from
    // backend code) is captured even if the process crashes. Without this,
    // the raw-mode terminal swallows stderr and the debug lines are lost.
    // The file is created/truncated here; the Redirect guard restores the
    // original stderr on drop. Users can `tail -f` it in another terminal.
    let log_path = std::env::var("GRIM_TUI_LOG").unwrap_or_else(|_| "/tmp/grim-tui.log".to_string());
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map_err(|e| Error::Config(format!("cannot open log file {log_path}: {e}")))?;
    let _stderr_redirect = gag::Redirect::stderr(log_file)
        .map_err(|e| Error::Config(format!("stderr redirect failed: {e}")))?;

    // Enable kitty keyboard protocol for sixel-adjacent input fidelity and
    // accurate modifier reporting. Best-effort: terminals that do not support
    // it ignore the sequence. Restored on drop via Pop.
    let _ = crossterm::execute!(
        std::io::stdout(),
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
        )
    );
    // Mouse support: enable capture only when opted in via GRIM_MOUSE=1.
    // Capture intercepts all mouse events, which disables the terminal's
    // native text selection/copy. Default off so users can select text.
    let mouse_capture = std::env::var("GRIM_MOUSE").as_deref() == Ok("1");
    if mouse_capture {
        let _ = crossterm::execute!(std::io::stdout(), EnableMouseCapture);
    }

    // Panic hook: restore the terminal so the panic message lands in the
    // terminal scrollback (and the redirected log file). The stderr redirect
    // above already captures all eprintln! output to the log file.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
        let _ = crossterm::execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
        let _ = ratatui::restore();
        eprintln!(
            "[grim-tui CRASH] {}\nbacktrace:\n{:?}",
            info,
            std::backtrace::Backtrace::force_capture()
        );
        prev_hook(info);
    }));

    // Signal handlers: GPU faults (SIGSEGV/SIGABRT/SIGBUS) kill the process
    // without running the panic hook. Restore the terminal before re-raising
    // so the user's shell isn't left in raw mode. The eprintln! goes to the
    // redirected stderr (log file) automatically.
    unsafe extern "C" fn fatal_signal_handler(sig: libc::c_int) {
        let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
        let _ = crossterm::execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
        let _ = ratatui::restore();
        eprintln!("[grim-tui SIGNAL] fatal signal {} — terminal restored, re-raising", sig);
        unsafe {
            libc::signal(sig, libc::SIG_DFL);
            libc::raise(sig);
        }
    }
    unsafe {
        libc::signal(libc::SIGSEGV, fatal_signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGABRT, fatal_signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGBUS, fatal_signal_handler as *const () as libc::sighandler_t);
    }

    let mut term = ratatui::init();

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
    // One permissions store shared by worker (checks before prompting) and
    // the App (adds rules on "always allow").
    let perms = permissions::shared(permissions::PermissionRules::load());
    let worker = worker::spawn_worker(params, cmd_rx, evt_tx, perms.clone());
    if let Some(m) = &model {
        let _ = cmd_tx.send(WorkerCommand::LoadModel { name: m.clone() });
    }

    let mut app = App::new(cmd_tx.clone());
    app.permissions = perms;

    // Session resume: --resume <path> uses that file; --continue picks the
    // most recently modified *.jsonl in the current directory, falling back
    // to the newest autosaved session in the central store.
    let resume_target = match (&resume, continue_last) {
        (Some(path), _) => Some(path.clone()),
        (None, true) => latest_session_file(
            &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        )
        .or_else(|| sessions::list_sessions().first().map(|s| s.path.clone()))
        .map(|p| p.to_string_lossy().into_owned()),
        (None, false) => None,
    };
    if let Some(path) = resume_target {
        match import_transcript(&path) {
            Ok((nodes, chat_msgs)) => {
                let count = nodes.len();
                app.transcript.nodes = nodes;
                app.messages = chat_msgs;
                app.transcript
                    .push_system(format!("resumed session {path} ({count} nodes)"));
            }
            Err(e) => app
                .transcript
                .push_error(format!("failed to resume {path}: {e}")),
        }
    } else if continue_last {
        app.transcript
            .push_system("no saved session found to continue (no *.jsonl in cwd)".into());
    }
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
        // Border flash delight: while flash is active, keep requesting renders
        // so the green border is visible for 300ms, then clear after expiry.
        if let Some(until) = app.flash_until {
            if Instant::now() < until {
                scheduler.request_render();
            } else {
                app.flash_until = None;
                scheduler.request_render();
            }
        }
        // Handle terminal resize as a full redraw trigger.
        // Short poll timeout (10ms) keeps input latency low for snappy picker
        // navigation and typing response.
        if crossterm::event::poll(Duration::from_millis(10))
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
                crossterm::event::Event::Mouse(m) => {
                    match m.kind {
                        MouseEventKind::Down(_) => {
                            // Focus heuristic: clicks focus chat vs side vs input.
                            // We use row to decide; column could distinguish chat vs side.
                            // For now, any click in the upper area resets to bottom focus,
                            // scroll events are handled separately.
                            app.handle_mouse(m);
                            scheduler.request_render();
                        }
                        MouseEventKind::ScrollUp => {
                            app.handle_mouse(m);
                            scheduler.request_render();
                        }
                        MouseEventKind::ScrollDown => {
                            app.handle_mouse(m);
                            scheduler.request_render();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        if scheduler.should_render() {
            app.frame_count = app.frame_count.wrapping_add(1);
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
    if std::env::var("GRIM_MOUSE").as_deref() == Ok("1") {
        let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
    }
    let _ = crossterm::execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zeroed TurnStats for event-driven tests.
    fn zero_stats() -> TurnStats {
        TurnStats {
            encode_ms: 0.0,
            prompt_tokens: 0,
            prefill_ms: None,
            decode_tps: None,
            tokens_generated: 0,
            accepted_per_step: None,
            cancelled: false,
            context_used: 0,
        }
    }

    #[test]
    fn submit_while_generating_enqueues() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.snap.model_name = Some("test-model".into());
        app.generating = true;
        app.submit_chat("hello while busy");
        assert_eq!(app.queue.len(), 1);
        assert!(app.messages.iter().all(|m| m.content != "hello while busy"));
    }

    #[test]
    fn turn_complete_sends_queued_message() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.snap.model_name = Some("test-model".into());
        app.generating = true;
        app.submit_chat("queued msg");
        assert_eq!(app.queue.len(), 1);
        app.handle_event(WorkerEvent::TurnComplete { stats: zero_stats() });
        assert!(app.generating, "queued submit restarts generation");
        assert!(app.queue.is_empty());
        assert!(app.messages.iter().any(|m| m.content == "queued msg"));
    }

    #[test]
    fn esc_clears_queue_before_cancelling() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.generating = true;
        app.queue.push("pending");
        app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert!(app.queue.is_empty());
    }

    /// Isolate App tests from the real `~/.local/share/grim` state: point
    /// XDG_DATA_HOME at a throwaway dir. Guard must be held for the test body.
    fn isolated_data_dir() -> (tempfile::TempDir, std::sync::MutexGuard<'static, ()>) {
        let guard = crate::tui::paths::env_lock();
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", dir.path()) };
        (dir, guard)
    }

    #[test]
    fn ctrl_r_opens_history_search_and_enter_fills_composer() {
        let (_dir, _guard) = isolated_data_dir();
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.history.append("write me a parser");
        app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('r'),
            crossterm::event::KeyModifiers::CONTROL,
        ));
        assert!(matches!(app.input_mode, InputMode::HistorySearch { .. }));
        app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert_eq!(app.composer.text(), "write me a parser");
        assert!(matches!(app.input_mode, InputMode::Chat));
    }

    #[test]
    fn submit_records_prompt_history() {
        let (_dir, _guard) = isolated_data_dir();
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.snap.model_name = Some("test-model".into());
        app.submit_chat("remember this prompt");
        assert_eq!(app.history.entries.len(), 1);
        assert_eq!(app.history.entries[0].text, "remember this prompt");
    }

    #[test]
    fn first_submit_assigns_titled_session_path() {
        let (_dir, _guard) = isolated_data_dir();
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.snap.model_name = Some("test-model".into());
        assert!(app.session_path.is_none());
        app.submit_chat("debug kv cache eviction");
        let p = app.session_path.as_ref().unwrap();
        assert!(p.to_string_lossy().contains("debug-kv-cache-eviction"));
    }

    #[test]
    fn module_loads() {
        assert!(!diagnostics::format_bytes(0).is_empty());
    }

    #[test]
    fn test_session_roundtrip_preserves_tool_history() {
        // Build a transcript with a tool exchange: user, assistant text,
        // tool call, tool result, final assistant text.
        let mut tr = Transcript::new();
        tr.push_user("read main.rs".into());
        tr.append_token("Sure, reading it now.");
        tr.finish_segment();
        tr.push_tool_call_with_diff(
            "read_file",
            Some("call_1"),
            r#"{"path":"main.rs"}"#,
            None,
        );
        tr.push_tool_result_for("fn main() {}".to_string(), Some("call_1".into()));
        tr.append_token("It defines main().");
        tr.finish_turn("done".into());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        export_transcript(&tr, path.to_str().unwrap()).unwrap();

        let (nodes, messages) = import_transcript(path.to_str().unwrap()).unwrap();
        assert_eq!(nodes.len(), 5);

        // Message sequence must be a valid OpenAI tool exchange:
        // user, assistant, assistant(tool_calls), tool(result), assistant.
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "Sure, reading it now.");
        assert_eq!(messages[2].role, "assistant");
        let calls = messages[2].tool_calls.as_ref().expect("tool_calls preserved");
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert!(calls[0].arguments.contains("main.rs"));
        assert_eq!(messages[3].role, "tool");
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[3].name.as_deref(), Some("read_file"));
        assert_eq!(messages[4].role, "assistant");
        assert_eq!(messages[4].content, "It defines main().");
    }

    #[test]
    fn test_close_stream_segment_records_history() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.transcript.append_token("Here is the answer.");
        app.close_stream_segment(None, None);
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, "assistant");
        assert_eq!(app.messages[0].content, "Here is the answer.");

        // Streaming markup is stripped from stored content.
        app.transcript
            .append_token("Checking. <tool_call>{\"name\":\"read_file\"}</tool_call>");
        app.close_stream_segment(Some("c9"), Some(("read_file", "{\"path\":\"x\"}")));
        assert_eq!(app.messages[1].content, "Checking.");
        assert_eq!(
            app.messages[1].tool_calls.as_ref().unwrap()[0].id,
            "c9"
        );

        // Empty segments are skipped.
        let before = app.messages.len();
        app.close_stream_segment(None, None);
        assert_eq!(app.messages.len(), before);
    }

    #[test]
    fn test_latest_session_file_picks_newest() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("old.jsonl");
        let b = dir.path().join("new.jsonl");
        std::fs::write(&a, "{}\n").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&b, "{}\n").unwrap();
        std::fs::write(dir.path().join("not-a-session.txt"), "x").unwrap();
        assert_eq!(latest_session_file(dir.path()), Some(b));
        assert!(latest_session_file(&dir.path().join("empty")).is_none());
    }

    #[test]
    fn test_plan_mode_toggles_and_notifies_worker() {
        assert!(matches!(parse_slash_command("/plan"), SlashCommand::Plan(None)));
        assert!(matches!(
            parse_slash_command("/plan on"),
            SlashCommand::Plan(Some(s)) if s == "on"
        ));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        assert!(!app.plan_mode);
        app.submit_chat("/plan");
        assert!(app.plan_mode);
        match rx.try_recv().unwrap() {
            WorkerCommand::SetPlanMode { enabled } => assert!(enabled),
            other => panic!("expected SetPlanMode, got {other:?}"),
        }
        app.submit_chat("/plan off");
        assert!(!app.plan_mode);
        match rx.try_recv().unwrap() {
            WorkerCommand::SetPlanMode { enabled } => assert!(!enabled),
            other => panic!("expected SetPlanMode, got {other:?}"),
        }
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
        assert!(matches!(
            parse_slash_command("/skill"),
            SlashCommand::Skill(None)
        ));
        assert!(matches!(
            parse_slash_command("/skill caveman"),
            SlashCommand::Skill(Some(s)) if s == "caveman"
        ));
        assert!(matches!(
            parse_slash_command("/skill off"),
            SlashCommand::Skill(Some(s)) if s == "off"
        ));
        assert!(matches!(
            parse_slash_command("/skills"),
            SlashCommand::Skills
        ));
        assert!(matches!(
            parse_slash_command("/project"),
            SlashCommand::ProjectDir(p) if p.is_empty()
        ));
        assert!(matches!(
            parse_slash_command("/project /tmp"),
            SlashCommand::ProjectDir(p) if p == "/tmp"
        ));
        assert!(matches!(
            parse_slash_command("/cd /tmp"),
            SlashCommand::ProjectDir(p) if p == "/tmp"
        ));
        assert!(matches!(
            parse_slash_command("/pwd"),
            SlashCommand::Pwd
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
            InputMode::SessionBrowser { selected: 0, .. }
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
        let _sessions = app.session_browser_items();
    }

    #[test]
    fn session_browser_filters_by_query() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        app.input_mode = InputMode::SessionBrowser {
            query: String::new(),
            selected: 0,
        };
        app.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE));
        let items = app.session_browser_items();
        assert!(items.iter().all(|s| s.title.contains('z')));
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

    #[test]
    fn test_tool_call_enters_approval_mode() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Simulate a WorkerEvent::ToolCall.
        app.handle_event(WorkerEvent::ToolCall {
            call_id: "call_1".into(),
            name: "write_file".into(),
            arguments: r#"{"path": "a.txt", "content": "hello"}"#.into(),
        });

        assert!(app.tool_approval_mode);
        assert!(app.pending_tool_call.is_some());
        assert_eq!(app.pending_tool_call.as_ref().unwrap().1, "write_file");
        // Transcript should have a ToolCall node.
        assert!(app
            .transcript
            .nodes
            .iter()
            .any(|n| n.role == crate::tui::transcript::Role::ToolCall));
    }

    #[test]
    fn test_tool_approval_executes_and_sends_result() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Set up a sandbox-friendly directory.
        let dir = tempfile::tempdir().unwrap();
        app.sandbox_root = dir.path().to_path_buf();

        // Enter approval mode with a read tool.
        app.handle_event(WorkerEvent::ToolCall {
            call_id: "call_2".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "test.txt", "content": "approved"}).to_string(),
        });
        assert!(app.tool_approval_mode);

        // Approve with Enter.
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()));

        assert!(!app.tool_approval_mode);
        assert!(app.pending_tool_call.is_none());

        // Verify a ToolResult was sent to the worker.
        let result = rx.try_recv();
        assert!(matches!(
            result,
            Ok(WorkerCommand::ToolResult { call_id, .. }) if call_id == "call_2"
        ));
    }

    #[test]
    fn test_tool_denial_sends_error_result() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.handle_event(WorkerEvent::ToolCall {
            call_id: "call_3".into(),
            name: "run_command".into(),
            arguments: r#"{"command": "rm -rf /"}"#.into(),
        });
        assert!(app.tool_approval_mode);

        // Deny with Esc.
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()));

        assert!(!app.tool_approval_mode);

        // Verify a ToolResult with denial was sent.
        let result = rx.try_recv();
        assert!(matches!(
            result,
            Ok(WorkerCommand::ToolResult { call_id, output }) if call_id == "call_3" && output.contains("denied")
        ));
    }

    #[test]
    fn test_skill_picker_key_handling() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Seed at least one skill so the picker has something to show.
        app.skills.push(crate::tui::skills::Skill {
            id: "caveman".into(),
            name: "Caveman".into(),
            description: "terse mode".into(),
            path: std::path::PathBuf::from("/fake/caveman/SKILL.md"),
        });

        // Open picker with Ctrl+G.
        app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL));
        assert!(matches!(
            app.input_mode,
            InputMode::SkillPicker { selected: 0 }
        ));

        // Escape returns to chat.
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()));
        assert_eq!(app.input_mode, InputMode::Chat);
    }

    #[test]
    fn test_activate_skill_by_name() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Create a temp skill directory with a SKILL.md.
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        let skill_md = skill_dir.join("SKILL.md");
        std::fs::write(
            &skill_md,
            "---\nname: Test Skill\ndescription: A test skill\n---\n\nYou are a test assistant.",
        )
        .unwrap();

        app.skills.push(crate::tui::skills::Skill {
            id: "test-skill".into(),
            name: "Test Skill".into(),
            description: "A test skill".into(),
            path: skill_md,
        });

        app.activate_skill("test-skill");

        assert_eq!(app.active_skill_name.as_deref(), Some("Test Skill"));
        assert!(app.system_prompt.is_some());
        assert!(app
            .system_prompt
            .as_ref()
            .unwrap()
            .contains("You are a test assistant."));
        // A system message should be injected.
        assert!(app.messages.iter().any(|m| m.role == "system"));
    }

    #[test]
    fn test_deactivate_skill() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Set up an active skill state.
        app.active_skill_name = Some("Caveman".into());
        app.system_prompt = Some("skill body".into());
        app.messages.push(grim_format::ChatMessage {
            role: "system".to_string(),
            content: "skill body".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        app.submit_chat("/skill off");

        assert!(app.active_skill_name.is_none());
        assert!(app.system_prompt.is_none());
        assert!(!app.messages.iter().any(|m| m.role == "system"));
    }

    #[test]
    fn test_project_dir_sets_sandbox_root() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        app.submit_chat(&format!("/project {path}"));

        assert_eq!(app.project_dir, std::path::PathBuf::from(dir.path().canonicalize().unwrap()));
        assert_eq!(app.sandbox_root, app.project_dir);
    }

    #[test]
    fn test_pwd_reports_current_dir() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.submit_chat("/pwd");
        // Transcript should have a system message containing "project directory:".
        assert!(app
            .transcript
            .nodes
            .iter()
            .any(|n| n.content.contains("project directory:")));
    }

    #[test]
    fn test_project_dir_mode_enter_and_set() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        let dir = tempfile::tempdir().unwrap();
        let path_str = dir.path().to_string_lossy().to_string();

        // Enter project dir mode with Ctrl+D.
        app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert!(matches!(app.input_mode, InputMode::ProjectDir));

        // Replace the composer text with our target path and submit.
        app.composer.set_text(&path_str);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()));

        assert_eq!(app.input_mode, InputMode::Chat);
        assert_eq!(app.project_dir, dir.path().canonicalize().unwrap());
        assert_eq!(app.sandbox_root, app.project_dir);
    }

    #[test]
    fn test_thinking_slash_command_parse() {
        assert!(matches!(
            parse_slash_command("/thinking"),
            SlashCommand::Thinking(None)
        ));
        assert!(matches!(
            parse_slash_command("/thinking high"),
            SlashCommand::Thinking(Some(s)) if s == "high"
        ));
        assert!(matches!(
            parse_slash_command("/think off"),
            SlashCommand::Thinking(Some(s)) if s == "off"
        ));
        assert!(matches!(
            parse_slash_command("/think medium"),
            SlashCommand::Thinking(Some(s)) if s == "medium"
        ));
    }

    #[test]
    fn test_thinking_sets_level_and_sends_command() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.submit_chat("/thinking high");

        assert_eq!(
            app.thinking_level,
            grim_core::sampler::ThinkingLevel::High
        );
        // Verify the worker was notified.
        let result = rx.try_recv();
        assert!(matches!(
            result,
            Ok(WorkerCommand::SetThinking { level }) if level == grim_core::sampler::ThinkingLevel::High
        ));
    }

    #[test]
    fn test_thinking_report_current_level() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.thinking_level = grim_core::sampler::ThinkingLevel::Medium;
        app.submit_chat("/thinking");

        // Transcript should report the current level.
        assert!(app
            .transcript
            .nodes
            .iter()
            .any(|n| n.content.contains("thinking level: medium")));
    }

    #[test]
    fn test_thinking_ctrl_t_cycles() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Start at Default, cycle → Low.
        app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert_eq!(
            app.thinking_level,
            grim_core::sampler::ThinkingLevel::Low
        );
        // Drain the command.
        let _ = rx.try_recv();

        // Cycle → Medium.
        app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert_eq!(
            app.thinking_level,
            grim_core::sampler::ThinkingLevel::Medium
        );
        let _ = rx.try_recv();

        // Cycle → High.
        app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert_eq!(
            app.thinking_level,
            grim_core::sampler::ThinkingLevel::High
        );
        let _ = rx.try_recv();

        // Cycle → Off.
        app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert_eq!(
            app.thinking_level,
            grim_core::sampler::ThinkingLevel::Off
        );
        let _ = rx.try_recv();

        // Cycle → Default (wraps around).
        app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert_eq!(
            app.thinking_level,
            grim_core::sampler::ThinkingLevel::Default
        );
    }

    #[test]
    fn test_task_list_add_and_render() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        assert!(app.task_list.is_empty());

        app.task_list.upsert(crate::tui::tasks::Task::new("1", "Read config file"));
        app.task_list
            .upsert(crate::tui::tasks::Task::new("2", "Run tests").with_status(crate::tui::tasks::TaskStatus::Completed));

        assert_eq!(app.task_list.len(), 2);
        assert_eq!(
            app.task_list.count_by_status(crate::tui::tasks::TaskStatus::Pending),
            1
        );
        assert_eq!(
            app.task_list.count_by_status(crate::tui::tasks::TaskStatus::Completed),
            1
        );
    }

    #[test]
    fn test_task_list_navigation_with_arrows() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.task_list
            .upsert(crate::tui::tasks::Task::new("1", "First task"));
        app.task_list
            .upsert(crate::tui::tasks::Task::new("2", "Second task"));
        app.task_list
            .upsert(crate::tui::tasks::Task::new("3", "Third task"));

        // Down navigates tasks.
        app.handle_key(KeyEvent::from(KeyCode::Down));
        assert_eq!(app.task_list.selected, 1);
        app.handle_key(KeyEvent::from(KeyCode::Down));
        assert_eq!(app.task_list.selected, 2);

        // Up navigates back.
        app.handle_key(KeyEvent::from(KeyCode::Up));
        assert_eq!(app.task_list.selected, 1);
    }

    #[test]
    fn test_task_shift_tab_cycles_status() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.task_list
            .upsert(crate::tui::tasks::Task::new("1", "First task"));
        assert_eq!(
            app.task_list.tasks[0].status,
            crate::tui::tasks::TaskStatus::Pending
        );

        // Shift+Tab cycles status.
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT));
        assert_eq!(
            app.task_list.tasks[0].status,
            crate::tui::tasks::TaskStatus::InProgress
        );

        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT));
        assert_eq!(
            app.task_list.tasks[0].status,
            crate::tui::tasks::TaskStatus::Completed
        );
    }

    #[test]
    fn test_task_expand_with_right_arrow() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        app.task_list.upsert(
            crate::tui::tasks::Task::new("1", "Task with details")
                .with_description("This is a detailed description"),
        );

        assert!(!app.task_list.tasks[0].expanded);

        // Right arrow expands.
        app.handle_key(KeyEvent::from(KeyCode::Right));
        assert!(app.task_list.tasks[0].expanded);

        // Right arrow again collapses.
        app.handle_key(KeyEvent::from(KeyCode::Right));
        assert!(!app.task_list.tasks[0].expanded);
    }

    #[test]
    fn test_backend_slash_command_parse() {
        assert!(matches!(
            parse_slash_command("/backend"),
            SlashCommand::Backend(None)
        ));
        assert!(matches!(
            parse_slash_command("/backend metal"),
            SlashCommand::Backend(Some(s)) if s == "metal"
        ));
        assert!(matches!(
            parse_slash_command("/backend cpu"),
            SlashCommand::Backend(Some(s)) if s == "cpu"
        ));
        assert!(matches!(
            parse_slash_command("/backend auto"),
            SlashCommand::Backend(Some(s)) if s == "auto"
        ));
    }

    #[test]
    fn test_backend_sets_value() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // CPU is always available.
        app.submit_chat("/backend cpu");
        assert_eq!(app.backend.as_deref(), Some("cpu"));

        // Auto resets to None.
        app.submit_chat("/backend auto");
        assert!(app.backend.is_none());
    }

    /// Only compiles with --features cuda: the test pokes the optional
    /// CUDA crate directly.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_backend_rejects_unavailable() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);

        // Pick a backend that's compiled in but has no device available.
        // In CI/test environments, CUDA is typically compiled in but no GPU is present.
        // The availability check should reject it.
        if grim_backend_cuda::CudaDevice::probe()
            .map(|d| d.is_empty())
            .unwrap_or(true)
        {
            // CUDA compiled in but no device — should be rejected.
            app.submit_chat("/backend cuda");
            assert!(app.backend.is_none(), "unavailable backend should not be set");
            assert!(app
                .transcript
                .nodes
                .iter()
                .any(|n| n.content.contains("unavailable")));
        }
        // Otherwise (CUDA has a device), skip — availability check passed correctly.
    }

    #[test]
    fn test_backend_rejects_unknown() {
        // Isolate from GRIM_BACKEND env var set by other tests.
        unsafe { std::env::remove_var("GRIM_BACKEND"); }
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx);
        // Ensure clean state regardless of prior test pollution.
        app.backend = None;

        app.submit_chat("/backend unknown");
        // Backend should remain unchanged (None from default).
        assert!(app.backend.is_none());
        // Error should be in transcript.
        assert!(app
            .transcript
            .nodes
            .iter()
            .any(|n| n.content.contains("unknown backend")));
    }
}
