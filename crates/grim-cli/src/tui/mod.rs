//! grim tui — ratatui chat interface over the in-process engine.
//!
//! Two threads: the UI thread owns the terminal and `App`; the worker thread
//! owns `Engine`, the tokenizer, and the sampler. They talk over two
//! `std::sync::mpsc` channels. GPU and model code runs only on the worker.

use grim_core::error::Error;

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
    pub cmd_tx: tokio::sync::mpsc::UnboundedSender<WorkerCommand>,
    pub messages: Vec<grim_format::ChatMessage>,
    pub should_quit: bool,
    pub generating: bool,
    pub scroll_offset: usize,
    pub show_sidebar: bool,
    pub input_mode: InputMode,
}

impl App {
    pub fn new(cmd_tx: tokio::sync::mpsc::UnboundedSender<WorkerCommand>) -> Self {
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

    pub fn handle_event(&mut self, evt: WorkerEvent) {
        match evt {
            WorkerEvent::Token { text } => {
                self.streaming.push_str(&text);
            }
            WorkerEvent::TurnComplete { stats: _stats } => {
                if !self.streaming.is_empty() {
                    self.transcript.push(self.streaming.clone());
                    self.streaming.clear();
                }
                self.generating = false;
            }
            WorkerEvent::Diagnostics { snap } => {
                self.snap = snap;
            }
            WorkerEvent::Error { message } => {
                self.transcript.push(format!("[error] {message}"));
            }
            WorkerEvent::ModelLoadStarted { name } => {
                self.snap.model_name = Some(name);
                self.snap.loading = true;
                self.transcript
                    .push(format!("[system] loading {}", name));
            }
            WorkerEvent::ModelLoadOk {
                name: _name,
                quant,
                strategy,
                ..
            } => {
                // strategy and quant are written into the snapshot by the
                // render path after the worker emits them; keep them as
                // context for the render to pick up.
                self.snap.quant = quant;
                self.snap.strategy = Some(strategy);
                self.snap.loading = false;
                self.transcript.push(format!("[system] model loaded"));
            }
            WorkerEvent::ModelLoadFailed { name: _name, error } => {
                self.snap.loading = false;
                self.transcript.push(format!("[system] load failed: {error}"));
            }
        }
    }

    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        match self.input_mode {
            InputMode::CtxOverride => self.handle_ctx_key(key),
            InputMode::Chat => self.handle_chat_key(key),
        }
    }

    fn handle_chat_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            crossterm::event::KeyCode::Enter => {
                let cmd = parse_slash_command(&self.input);
                self.dispatch_chat_command(cmd);
            }
            crossterm::event::KeyCode::Backspace => {
                self.input.pop();
            }
            crossterm::event::KeyCode::Char(c) => {
                self.input.push(c);
            }
            crossterm::event::KeyCode::Escape => {
                self.dispatch_chat_command(SlashCommand::Exit);
            }
            crossterm::event::KeyCode::PageUp => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            crossterm::event::KeyCode::PageDown => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
            }
            crossterm::event::KeyCode::F(2) => {
                self.show_sidebar = !self.show_sidebar;
            }
            crossterm::event::KeyCode::F(3) => {
                self.input.clear();
                self.input_mode = InputMode::CtxOverride;
            }
            _ => {}
        }
    }

    fn dispatch_chat_command(&mut self, cmd: SlashCommand) {
        self.input.clear();
        match cmd {
            SlashCommand::Exit => {
                self.should_quit = true;
            }
            SlashCommand::Help => {
                self.transcript.push(
                    "[help]".into()
                        + " /model            list local models\n"
                        + " /model <name>     load model\n"
                        + " /exit             quit\n"
                        + " /clear            reset session\n"
                        + " /model           set context limit\n"
                        + " F2               toggle sidebar\n"
                        + " F3               set context limit\n"
                        + " Esc              cancel generation\n"
                        + " Ctrl+C           quit",
                );
            }
            SlashCommand::Clear => {
                self.transcript.clear();
                self.messages.clear();
                self.scroll_offset = 0;
                self.streaming.clear();
            }
            SlashCommand::Model(None) => {
                let list = grim_core::catalog::list_local_models();
                if list.is_empty() {
                    self.transcript.push("[system] no local models found".into());
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
                    self.transcript.push("[hint] /model <name> to load one".into());
                }
            }
            SlashCommand::Model(Some(name)) => {
                if let Err(e) = self.cmd_tx
                    .send(WorkerCommand::LoadModel { name })
                    .await
                {
                    self.transcript.push(format!("[system] send failed: {e}"));
                }
            }
            SlashCommand::NotACommand => {
                // Plain user input: append as a user message.
                if !self.input.is_empty() {
                    self.messages.push(grim_format::ChatMessage {
                        role: "user".into(),
                        content: self.input.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                    self.generating = true;
                    if let Err(e) = self.cmd_tx
                        .send(WorkerCommand::Generate {
                            messages: self.messages.clone(),
                        })
                        .await
                    {
                        self.transcript.push(format!("[system] send failed: {e}"));
                        self.generating = false;
                    }
                }
            }
            SlashCommand::Unknown(word) => {
                self.transcript
                    .push(format!("[error] unknown command: /{word}"));
            }
        }
    }

    fn handle_ctx_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            crossterm::event::KeyCode::Enter => {
                let ov = parse_ctx_override(&self.input);
                self.input.clear();
                self.input_mode = InputMode::Chat;
                match ov {
                    CtxOverride::Auto => {
                        let _ = self.cmd_tx
                            .send(WorkerCommand::SetContextLimit { limit: None })
                            .await;
                        self.transcript.push("[system] ctx limit: auto".into());
                    }
                    CtxOverride::Apply(n) => {
                        let _ = self.cmd_tx
                            .send(WorkerCommand::SetContextLimit { limit: Some(n) })
                            .await;
                        self.transcript.push(format!("[system] ctx limit: {n}"));
                    }
                    CtxOverride::Invalid => {
                        self.transcript
                            .push("[hint] enter a number or empty".into());
                    }
                }
            }
            crossterm::event::KeyCode::Backspace => {
                self.input.pop();
            }
            crossterm::event::KeyCode::Char(c) => {
                self.input.push(c);
            }
            crossterm::event::KeyCode::Escape => {
                self.input.clear();
                self.input_mode = InputMode::Chat;
            }
            _ => {}
        }
    }
}

/// Terminal lifecycle guard: restores the terminal on drop.
pub struct TerminalGuard {
    _priority: std::cell::Cell<Option<Box<dyn Fn() + Send>>>,
}

impl TerminalGuard {
    pub fn new() -> Self {
        let prior = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = ratatui::restore();
            if let Some(prior) = prior AsMutAny {
                prior(info);
            } else {
                eprintln!("[panic] {}", info);
            }
        }));
        TerminalGuard {
            _priority: std::cell::Cell::new(None),
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = ratatui::restore();
    }
}

impl TerminalGuard {
    fn as_mut_any(&self) -> Option<&dyn Fn(&std::panic::PanicInfo<'_>) + Send> {
        None
    }
}

/// Entry point for the TUI command.
///
/// Implemented in Tasks 8-10. The stub here exists so Task 7's clap wiring
/// compiles; the real terminal loop replaces this body before manual
/// verification.
pub async fn cmd_tui(
    _model: Option<String>,
    _temperature: f32,
    _top_p: f32,
    _top_k: u32,
    _max_tokens: usize,
    _seed: u64,
    _repeat_penalty: f32,
) -> Result<()> {
    if !std::io::stdout().is_terminal() {
        return Err(Error::Config("grim tui needs an interactive terminal".into()));
    }
    Err(Error::Config("tui not implemented yet".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn module_loads() {
        assert!(diagnostics::format_bytes(0).is_empty() == false);
    }

    #[test]
    fn parses_known_commands() {
        assert!(matches!(parse_slash_command("/exit"), SlashCommand::Exit));
        assert!(matches!(parse_slash_command("/help"), SlashCommand::Help));
        assert!(matches!(parse_slash_command("/model"), SlashCommand::Model(None)));
        assert!(matches!(
            parse_slash_command("/model llama3"),
            SlashCommand::Model(Some(m)) if m == "llama3"
        ));
        assert!(matches!(parse_slash_command("hello"), SlashCommand::NotACommand));
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
        assert!(matches!(parse_ctx_override("8192"), CtxOverride::Apply(8192)));
        assert!(matches!(parse_ctx_override("abc"), CtxOverride::Invalid));
        assert!(matches!(parse_ctx_override("-1"), CtxOverride::Invalid));
    }

    #[test]
    fn slash_and_ctx_override_consistency() {
        // NotACommand leaves '/' unparsed.
        assert!(matches!(parse_slash_command("not a cmd"), SlashCommand::NotACommand));

        // parse_slash_command splits on first whitespace; parse_ctx_override
        // reads the whole content as a single override value.
        let cmd = parse_slash_command("/model lmf2.5-230m");
        assert!(matches!(cmd, SlashCommand::Model(Some(ref n)) if n == "lmf2.5-230m"));

        let ov = parse_ctx_override("  8192  ");
        assert!(matches!(ov, CtxOverride::Apply(8192)));
    }
}