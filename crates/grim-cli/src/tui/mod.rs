//! grim tui — ratatui chat interface over the in-process engine.
//!
//! Two threads: the UI thread owns the terminal and `App`; the worker thread
//! owns `Engine`, the tokenizer, and the sampler. The UI thread sends
//! `WorkerCommand`s and drains `WorkerEvent`s over `std::sync::mpsc`
//! channels. GPU and model code runs only on the worker.

use std::io::IsTerminal;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent};
use grim_core::error::{Error, Result};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Wrap};

/// Diagnostics formatting helpers for the TUI.
pub mod diagnostics;

/// Worker thread and channel protocol.
pub mod worker;

pub use worker::{DiagnosticsSnapshot, TurnStats, WorkerCommand, WorkerEvent, WorkerParams};

/// Slash commands typed in the input line.
#[derive(Debug, PartialEq, Eq)]
pub enum SlashCommand {
    Model(Option<String>),
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
    match first_word {
        "exit" => SlashCommand::Exit,
        "help" => SlashCommand::Help,
        "clear" => SlashCommand::Clear,
        "model" if after.is_empty() => SlashCommand::Model(None),
        "model" => SlashCommand::Model(Some(after.trim().to_string())),
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

/// Shape of the input bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Chat,
    CtxOverride,
}

/// State driving the terminal render loop.
pub struct App {
    pub input: String,
    pub transcript: Vec<String>,
    pub streaming: String,
    pub snap: DiagnosticsSnapshot,
    pub cmd_tx: Sender<WorkerCommand>,
    pub messages: Vec<grim_format::ChatMessage>,
    pub should_quit: bool,
    pub generating: bool,
    pub scroll_offset: usize,
    pub show_sidebar: bool,
    pub input_mode: InputMode,
}

impl App {
    pub fn new(cmd_tx: Sender<WorkerCommand>) -> Self {
        Self {
            input: String::new(),
            transcript: Vec::new(),
            streaming: String::new(),
            snap: DiagnosticsSnapshot::default(),
            cmd_tx,
            messages: Vec::new(),
            should_quit: false,
            generating: false,
            scroll_offset: 0,
            show_sidebar: true,
            input_mode: InputMode::Chat,
        }
    }

    /// Apply a worker event to app state.
    pub fn handle_event(&mut self, evt: WorkerEvent) {
        match evt {
            WorkerEvent::Token { text } => {
                self.streaming.push_str(&text);
            }
            WorkerEvent::TurnComplete { stats } => {
                if !self.streaming.is_empty() {
                    self.transcript
                        .push(format!("assistant: {}", self.streaming));
                    self.streaming.clear();
                }
                let prefill = diagnostics::format_ms(stats.prefill_ms);
                let decode = diagnostics::format_tps(stats.decode_tps);
                self.transcript.push(format!(
                    "· enc {:.1} ms | ttft {} | {} | {} tok{}",
                    stats.encode_ms,
                    prefill,
                    decode,
                    stats.tokens_generated,
                    if stats.cancelled { " (cancelled)" } else { "" }
                ));
                self.generating = false;
            }
            WorkerEvent::Diagnostics { snap } => {
                self.snap = snap;
            }
            WorkerEvent::Error { message } => {
                self.transcript.push(format!("[error] {message}"));
            }
            WorkerEvent::ModelLoadStarted { name } => {
                self.snap.loading = true;
                self.snap.model_name = Some(name.clone());
                self.transcript.push(format!("[system] loading {name}"));
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
                    .push(format!("[system] model loaded: {name} ({strategy})"));
                self.messages.clear();
                self.streaming.clear();
            }
            WorkerEvent::ModelLoadFailed { name, error } => {
                self.snap.loading = false;
                self.transcript
                    .push(format!("[system] model '{name}' failed to load: {error}"));
            }
        }
    }

    /// Handle a single key press.
    pub fn handle_key(&mut self, key: KeyEvent) {
        match self.input_mode {
            InputMode::CtxOverride => self.handle_ctx_key(key),
            InputMode::Chat => self.handle_chat_key(key),
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                let text = std::mem::take(&mut self.input);
                self.submit_chat(&text);
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) => {
                self.input.push(c);
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
                self.input.clear();
                self.input_mode = InputMode::CtxOverride;
            }
            _ => {}
        }
    }

    fn submit_chat(&mut self, text: &str) {
        match parse_slash_command(text) {
            SlashCommand::Exit => {
                self.should_quit = true;
            }
            SlashCommand::Help => {
                self.transcript.push(
                    "/model               list local models\n\
                     /model <name>        load or hot-swap a model\n\
                     /exit                quit\n\
                     /clear               reset session\n\
                     F2                   toggle diagnostics sidebar\n\
                     F3                   set context limit override\n\
                     Esc                  cancel generation\n\
                     Ctrl+C               quit"
                        .to_string(),
                );
            }
            SlashCommand::Clear => {
                self.transcript.clear();
                self.messages.clear();
                self.streaming.clear();
                self.scroll_offset = 0;
            }
            SlashCommand::Model(None) => {
                let list = grim_core::catalog::list_local_models();
                if list.is_empty() {
                    self.transcript
                        .push("[system] no local models found".into());
                } else {
                    for entry in &list {
                        let ctx = if entry.context_length > 0 {
                            entry.context_length.to_string()
                        } else {
                            "?".into()
                        };
                        self.transcript
                            .push(format!("  {}  {}  ctx {}", entry.name, entry.quant, ctx));
                    }
                    self.transcript
                        .push("[hint] /model <name> to load one".into());
                }
            }
            SlashCommand::Model(Some(name)) => {
                let _ = self.cmd_tx.send(WorkerCommand::LoadModel { name });
            }
            SlashCommand::NotACommand => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return;
                }
                if self.generating {
                    self.transcript
                        .push("[hint] generation in progress; Esc to cancel first".into());
                    return;
                }
                self.messages.push(grim_format::ChatMessage {
                    role: "user".to_string(),
                    content: trimmed.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
                self.transcript.push(format!("you: {}", trimmed));
                self.generating = true;
                let _ = self.cmd_tx.send(WorkerCommand::Generate {
                    messages: self.messages.clone(),
                });
            }
            SlashCommand::Unknown(word) => {
                self.transcript
                    .push(format!("[error] unknown command: /{word}"));
            }
        }
    }

    fn handle_ctx_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                let text = std::mem::take(&mut self.input);
                self.input_mode = InputMode::Chat;
                match parse_ctx_override(&text) {
                    CtxOverride::Auto => {
                        let _ = self
                            .cmd_tx
                            .send(WorkerCommand::SetContextLimit { limit: None });
                        self.transcript.push("[system] ctx limit: auto".into());
                    }
                    CtxOverride::Apply(n) => {
                        let _ = self
                            .cmd_tx
                            .send(WorkerCommand::SetContextLimit { limit: Some(n) });
                        self.transcript.push(format!("[system] ctx limit: {n}"));
                    }
                    CtxOverride::Invalid => {
                        self.transcript
                            .push("[hint] enter a number or empty".into());
                    }
                }
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) => {
                self.input.push(c);
            }
            KeyCode::Esc => {
                self.input.clear();
                self.input_mode = InputMode::Chat;
            }
            _ => {}
        }
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
    let outer = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).split(f.area());

    let (chat_area, side_area) = if app.show_sidebar {
        let main = Layout::horizontal([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(outer[0]);
        (main[0], Some(main[1]))
    } else {
        (outer[0], None)
    };

    let mut chat_items: Vec<Line> = app.transcript.iter().cloned().map(Line::from).collect();
    if app.generating && !app.streaming.is_empty() {
        chat_items.push(Line::from(format!("assistant: {}", app.streaming)));
    }
    let chat = Paragraph::new(chat_items)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_offset as u16, 0))
        .block(Block::bordered().title("Chat"));
    f.render_widget(chat, chat_area);

    if let Some(area) = side_area {
        let lines: Vec<Line> = diagnostics::sidebar_lines(&app.snap)
            .into_iter()
            .map(Line::from)
            .collect();
        let side = Paragraph::new(lines).block(Block::bordered().title("Diagnostics"));
        f.render_widget(side, area);
    }

    let title = match app.input_mode {
        InputMode::Chat => "Input (/help · /model · /exit · F2 sidebar · F3 ctx · Esc cancels)",
        InputMode::CtxOverride => {
            "Context limit override (Enter applies, empty = auto, Esc cancels)"
        }
    };
    f.render_widget(
        Paragraph::new(app.input.as_str()).block(Block::bordered().title(title)),
        outer[1],
    );
    f.set_cursor_position(Position::new(
        outer[1].x + 1 + app.input.chars().count() as u16,
        outer[1].y + 1,
    ));
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
    loop {
        while let Ok(evt) = evt_rx.try_recv() {
            app.handle_event(evt);
        }
        if crossterm::event::poll(Duration::from_millis(50))
            .map_err(|e| Error::Config(format!("terminal poll failed: {e}")))?
        {
            let key = crossterm::event::read()
                .map_err(|e| Error::Config(format!("terminal read failed: {e}")))?;
            if matches!(
                key,
                crossterm::event::Event::Key(KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: crossterm::event::KeyModifiers::CONTROL,
                    ..
                })
            ) {
                app.should_quit = true;
            } else if let crossterm::event::Event::Key(k) = key {
                app.handle_key(k);
            }
        }
        term.draw(|f| ui(f, &app))
            .map_err(|e| Error::Config(format!("render failed: {e}")))?;
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
}
