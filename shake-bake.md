# grim tui implementation plan

On approval, save this document to `docs/superpowers/plans/2026-08-18-grim-tui.md`, then execute tasks in order.

**Goal:** Add `grim tui`, a ratatui chat interface over the in-process engine with live diagnostics (encode, prefill TTFT, decode tok/s, KV cache, context, VRAM, system RAM, speculative strategy plain/DSpark/native MTP), `/model` hot-swap, `/exit`, F2 sidebar toggle, F3 context-limit override.

**Architecture:** Two threads: UI thread owns terminal + state; worker thread owns `grim_engine::Engine`, tokenizer, sampler, joined by two `std::sync::mpsc` channels. GPU/model code runs only on the worker inside `catch_unwind`, so a backend panic becomes an on-screen error. The worker handles one command at a time, so a `Generate` can never interleave with an in-progress hot-swap.

**Stack:** ratatui 0.29 + crossterm 0.28 (new deps, grim-cli only) + existing grim-engine/core/format/speculative/server crates.

## Global constraints

- New code only in `crates/grim-cli`. One exception: in grim-server make `probe_sys_ram` (lib.rs:5294) and `probe_vram_and_gpus` (lib.rs:5318) `pub fn`. No other grim-server changes; none to grim-engine/core/speculative.
- No `unwrap()`/`expect()` outside `#[cfg(test)]`; worker errors become `WorkerEvent::Error` lines.
- Never invent telemetry: `Option` telemetry renders `n/a`; measured fields are `Option` too; zero VRAM/RAM totals render `n/a`.
- Slice logits to last `vocab` entries before sampling (engine pads to 65536; run.rs:500-502).
- Respect `GRIM_BACKEND`; CPU smoke tests use `GRIM_BACKEND=cpu`.
- `cargo fmt` + `cargo clippy -p grim-cli` clean before each commit; commits use `feat(cli): ...`.

## Reference map (verified)

- grim-engine lib.rs: `Engine::new(EngineConfig::default())`; `register_model(id, Box<dyn CausalLm>)` (:411, auto-wraps speculative); `strategy_for` (:534); `tick`; `last_outcome`; `record_generated_token`; `finish_request`; `unload_model`; `kv_cache_telemetry()` (:400); `tokens_per_sec() -> Option<f32>`; `last_ttft_ms() -> Option<f64>`. `StepOutcome { logits: Option<Arc<Tensor>> (None when the request was not driven that tick), accepted_tokens, speculative }` (:121-131).
- `model_loader::load_from_path` (:3050), `resolve_discrete_rocm_devices` (:118). catalog.rs: `resolve_model_preferring_grim` (:311), `list_local_models` (:449), `ModelEntry` at :30-45.
- Tokenizer per run.rs:615-636; 13 pub fields + `Default` (tokenizer.rs:5-30). Prompt/EOS rules: run.rs:863-886, 931-939.
- Tests: inline `#[cfg(test)] mod tests`.

---

### Task 1: Dependencies and module skeleton

**Files:** Create `crates/grim-cli/src/tui/mod.rs`, `tui/diagnostics.rs`, `tui/worker.rs`; modify `crates/grim-cli/Cargo.toml`, `src/main.rs`.

- [ ] Cargo.toml: add `ratatui = "0.29"`, `crossterm = "0.28"`.
- [ ] Three files with one `//!` doc comment each; `mod.rs` holds `pub mod diagnostics; pub mod worker;`.
- [ ] main.rs after `pub mod train;` (line 26): `pub mod tui;`.
- [ ] `cargo build -p grim-cli` succeeds. Commit `feat(cli): scaffold tui module with ratatui deps`.

### Task 2: Diagnostics formatting helpers (TDD)

**Files:** `tui/diagnostics.rs`.
**Interfaces:** Produces `format_bytes(u64)->String`, `ratio_percent(u64,u64)->u16`, `format_ms(Option<f64>)->String`, `format_tps(Option<f64>)->String`, `acceptance_rate(usize,usize)->Option<f64>`, `strategy_label(&Strategy)->&'static str`, `bar(u64,u64)->String`.

- [ ] Failing test at bottom of file:

```rust
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
```

- [ ] Verify RED: `cargo test -p grim-cli tui::diagnostics` fails, unresolved names.
- [ ] Implement: `format_bytes` = largest of B/KiB/MiB/GiB/TiB, one decimal except plain bytes; ms/tps = `n/a` on None else `{:.1}`; `ratio_percent` = `((used*100)/total.max(1)) as u16` clamped 100; `bar` = `let pct = ratio_percent(used,total) as usize; let fill = pct*18/100; format!("[{}{}] {}%", "█".repeat(fill), "░".repeat(18-fill), pct)`; `strategy_label` = 3-arm match.
- [ ] Verify GREEN, fmt+clippy. Commit `feat(cli): tui diagnostics formatting helpers`.

### Task 3: Snapshot and sidebar lines (TDD)

**Files:** `tui/diagnostics.rs`.
**Interfaces:** Produces `DiagnosticsSnapshot` (fields below) with `Default + Clone`, `sidebar_lines(&DiagnosticsSnapshot) -> Vec<String>`.

- [ ] Failing tests in same `mod tests`:

```rust
#[test]
fn sidebar_lines_render_full_snapshot() {
    let snap = DiagnosticsSnapshot {
        model_name: Some("LFM2.5-230M".into()), quant: Some("Q8_0".into()),
        backend: "rocm gfx1100".into(), strategy: Some("DSpark".into()),
        encode_ms: Some(3.1), prompt_tokens: 128, prefill_ms: Some(142.0),
        decode_tps: Some(41.2), turn_tps: Some(38.9), tokens_generated: 57,
        kv_used_bytes: 1_288_490_187, kv_total_bytes: 4_294_967_296,
        kv_blocks_used: 312, kv_blocks_total: 1024,
        ctx_used: 2412, ctx_limit: 8192, accepted_per_step: Some(2.3),
        vram_used_bytes: 3_221_225_472, vram_total_bytes: 12_884_901_888,
        ram_used_bytes: 16_106_127_360, ram_total_bytes: 32_212_254_720,
        loading: false, generating: false,
    };
    assert_eq!(sidebar_lines(&snap), vec![
        "model: LFM2.5-230M (Q8_0)", "backend: rocm gfx1100",
        "spec: DSpark (2.3 tok/step)", "encode: 3.1 ms (128 tok)",
        "prefill: 142.0 ms", "decode: 41.2 tok/s (EMA)",
        "turn: 38.9 tok/s (57 tok)",
        "kv [█████░░░░░░░░░░░░░░░] 30%", "1.2 / 4.0 GiB (312/1024 blk)",
        "ctx [█████░░░░░░░░░░░░░░░] 29%", "2412 / 8192 tok",
        "vram [████░░░░░░░░░░░░░░░░] 25%", "3.0 / 12.0 GiB",
        "ram [█████████░░░░░░░░░] 50%", "15.0 / 30.0 GiB",
    ].into_iter().map(String::from).collect::<Vec<_>>());
}

#[test]
fn sidebar_lines_empty_state() {
    let lines = sidebar_lines(&DiagnosticsSnapshot::default());
    assert_eq!(lines[0], "model: none loaded (/model <name>)");
    assert!(lines.iter().any(|l| l == "vram: n/a" || l == "ram: n/a"));
}
```

- [ ] Verify RED. Implement struct with exactly those fields (Options: `encode_ms/prefill_ms/decode_tps/turn_tps/accepted_per_step/strategy/quant/model_name`; usize: `prompt_tokens/tokens_generated`; u64: kv/ctx/vram/ram counters; bool: `loading/generating`; String `backend`). `sidebar_lines`: quant in parens; loading -> `model: loading ...`; spec = strategy plus `(x.x tok/step)` when acceptance Some, `n/a` when strategy None; ctx `used/limit` or `used/?` when limit 0; vram/ram total 0 -> single `n/a` line, else `bar` line plus `X / Y GiB` line.
- [ ] Verify GREEN, fmt+clippy. Commit `feat(cli): diagnostics snapshot and sidebar rendering`.

### Task 4: Slash-command and ctx-override parsers (TDD)

**Files:** `tui/mod.rs`.
**Interfaces:** Produces `enum SlashCommand { Model(Option<String>), Exit, Clear, Help, NotACommand, Unknown(String) }`, `parse_slash_command(&str) -> SlashCommand`, `enum CtxOverride { Apply(u64), Auto, Invalid }`, `parse_ctx_override(&str) -> CtxOverride` (trim; empty -> Auto; parses u64 -> Apply; else Invalid).

- [ ] Failing test at bottom of `mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_commands() {
        assert!(matches!(parse_slash_command("/exit"), SlashCommand::Exit));
        assert!(matches!(parse_slash_command("/model"), SlashCommand::Model(None)));
        assert!(matches!(parse_slash_command("/model llama3"), SlashCommand::Model(Some(m)) if m == "llama3"));
        assert!(matches!(parse_slash_command("hello"), SlashCommand::NotACommand));
        assert!(matches!(parse_slash_command("/nope"), SlashCommand::Unknown(s) if s == "nope"));
    }

    #[test]
    fn parses_ctx_override() {
        assert!(matches!(parse_ctx_override(""), CtxOverride::Auto));
        assert!(matches!(parse_ctx_override("8192"), CtxOverride::Apply(8192)));
        assert!(matches!(parse_ctx_override("abc"), CtxOverride::Invalid));
        assert!(matches!(parse_ctx_override("-1"), CtxOverride::Invalid));
    }
}
```

- [ ] Verify RED. Implement: trim; non-`/` prefix = `NotACommand`; `/model` empty rest = `Model(None)` else `Model(Some(rest))`; unknown word = `Unknown(word)`.
- [ ] Verify GREEN, fmt+clippy. Commit `feat(cli): tui slash and ctx-override parsers`.

### Task 5: Worker pure helpers (TDD)

**Files:** `tui/worker.rs`.
**Interfaces:** Produces `is_eos_token(&GgufTokenizer, u32) -> bool`, `bos_prefix(&GgufTokenizer) -> Vec<u32>`, `panic_message(Box<dyn Any + Send>) -> String`.

- [ ] Failing test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use grim_format::GgufTokenizer;

    fn tok() -> GgufTokenizer {
        let pairs = [("<s>", 0u32), ("<|im_end|>", 3), ("<|endoftext|>", 4), ("</s>", 5)];
        GgufTokenizer {
            token_to_id: pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            eos_token_id: Some(2), bos_token_id: Some(0),
            ..Default::default()
        }
    }

    #[test]
    fn helpers() {
        let t = tok();
        assert!(is_eos_token(&t, 2) && is_eos_token(&t, 3));
        assert!(is_eos_token(&t, 4) && is_eos_token(&t, 5));
        assert!(!is_eos_token(&t, 9));
        assert_eq!(bos_prefix(&tok()), vec![0]);
        let p = std::panic::catch_unwind(|| panic!("boom")).unwrap_err();
        assert!(panic_message(p).contains("boom"));
    }
}
```

- [ ] Verify RED (fixture uses the struct's `Default`, tokenizer.rs:30, so future field additions cannot break it).
- [ ] Implement: `is_eos_token` mirrors run.rs:931-939; `bos_prefix` mirrors run.rs:873-879 (candidates `<|startoftext|>`, `<s>`, `<|im_start|>`; first hit, else empty); `panic_message` downcasts `&str` then `String`, else `"unknown panic"`.
- [ ] Verify GREEN, fmt+clippy. Commit `feat(cli): tui worker token helpers`.

### Task 6: Channel protocol and worker skeleton (TDD)

**Files:** `tui/worker.rs`.
**Interfaces:** Produces `enum WorkerCommand { LoadModel{name:String}, Generate{messages:Vec<grim_format::ChatMessage>}, SetContextLimit{limit:Option<u64>}, Cancel, Quit }`; `enum WorkerEvent { ModelLoadStarted{name}, ModelLoadOk{name, quant:Option<String>, context_length:u64, strategy:String}, ModelLoadFailed{name, error}, Token{text:String}, TurnComplete{stats:TurnStats}, Diagnostics{snap:DiagnosticsSnapshot}, Error{message} }`; `struct TurnStats { encode_ms: f64, prompt_tokens: usize, prefill_ms: Option<f64>, decode_tps: Option<f64>, tokens_generated: usize, accepted_per_step: Option<f64>, cancelled: bool, context_used: u64 }` (`decode_tps` Option: None when the turn produced no tokens); `struct WorkerParams { temperature:f32, top_p:f32, top_k:u32, max_tokens:usize, seed:u64, repeat_penalty:f32 }`; `spawn_worker(WorkerParams, Receiver<WorkerCommand>, Sender<WorkerEvent>) -> JoinHandle<()>`.

- [ ] Failing test:

```rust
#[test]
fn worker_starts_and_quits_cleanly() {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
    let (evt_tx, evt_rx) = std::sync::mpsc::channel();
    let h = spawn_worker(WorkerParams { temperature: 0.7, top_p: 0.9, top_k: 40, max_tokens: 256, seed: 42, repeat_penalty: 1.1 }, cmd_rx, evt_tx);
    cmd_tx.send(WorkerCommand::Quit).unwrap();
    h.join().unwrap();
    assert!(evt_rx.try_recv().is_err());
}
```

- [ ] Verify RED. Implement: `Engine::new(EngineConfig::default())`, sampler from `SamplingParams` (run.rs:638-645), `vocab = 512` placeholder, field `ctx_override: Option<u64>`; loop on `rx.recv()`; `Quit` breaks; `SetContextLimit` stores the Option; `Cancel` outside a turn is ignored; other commands run inside `catch_unwind(AssertUnwindSafe(...))`, panics send `Error { message: panic_message(p) }` and continue. `LoadModel`/`Generate` stubs send `Error { "not implemented" }`.
- [ ] Verify GREEN, fmt+clippy. Commit `feat(cli): tui worker thread and channel protocol`.

### Task 7: Clap subcommand and entry point

**Files:** `main.rs`, `tui/mod.rs`.
**Interfaces:** Produces `pub async fn cmd_tui(model: Option<String>, temperature: f32, top_p: f32, top_k: u32, max_tokens: usize, seed: u64, repeat_penalty: f32) -> Result<()>`.

- [ ] Add `Commands::Tui` near `Run` (main.rs:56-415): positional `model: Option<String>`; flags `--temperature` 0.7, `--top-p` 0.9, `--top-k` 40, `--max-tokens` 512, `--seed` 42, `--repeat-penalty` 1.1, matching Run's clap style.
- [ ] Dispatch arm (main.rs:674-1590): `tui::cmd_tui(...).await?` with the 7 args in order.
- [ ] Stub `cmd_tui`: if `!std::io::stdout().is_terminal()` return `Err(Error::Config("grim tui needs an interactive terminal".into()))`, else same error "tui not implemented yet".
- [ ] `cargo run -p grim-cli -- tui --help` lists MODEL + flags; `echo | cargo run -p grim-cli -- tui` errors cleanly. fmt+clippy. Commit `feat(cli): wire tui subcommand`.

### Task 8: Terminal shell, app state, render loop

**Files:** `tui/mod.rs`. Consumes Tasks 3/4/6. Rendering and raw-key input need a real TTY, so they are manual-verified below; this is the one intentional TDD exception. All pure logic was TDD'd in Tasks 2-4.

- [ ] `enum InputMode { Chat, CtxOverride }`. `struct App { input: String, transcript: Vec<String>, streaming: String, snap: DiagnosticsSnapshot, cmd_tx: Sender<WorkerCommand>, messages: Vec<ChatMessage>, should_quit: bool, generating: bool, scroll_offset: usize, show_sidebar: bool, input_mode: InputMode }` (sidebar true, Chat initially). `struct TerminalGuard;`: `new()` installs panic hook calling `ratatui::restore()` then prior hook; `Drop` also restores.
- [ ] `cmd_tui` (replace the stub):

```rust
let mut term = ratatui::init();
let _guard = TerminalGuard::new();
let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
let (evt_tx, evt_rx) = std::sync::mpsc::channel();
let worker = worker::spawn_worker(params_from_args, cmd_rx, evt_tx);
if let Some(m) = &model { cmd_tx.send(WorkerCommand::LoadModel { name: m.clone() }).ok(); }
let mut app = App::new(cmd_tx.clone());
loop {
    while let Ok(evt) = evt_rx.try_recv() { app.handle_event(evt); }
    if crossterm::event::poll(Duration::from_millis(50))? { app.handle_key(crossterm::event::read()?); }
    term.draw(|f| ui(f, &app))?;
    if app.should_quit { break; }
}
cmd_tx.send(WorkerCommand::Quit).ok();
worker.join().ok();
ratatui::restore();
Ok(())
```

- [ ] `ui(f, &app)`: vertical `[Min(3), Length(3)]`. With sidebar: top split `[Percentage(68), Percentage(32)]`; left = chat `Paragraph` (transcript + `streaming` when generating, `Wrap { trim: false }`, block "Chat", scrolled by offset); right = `sidebar_lines(&snap)` block "Diagnostics". Hidden: chat full width. Bottom: input `Paragraph`, title by mode: Chat = "Input (/help · /model · /exit · F2 sidebar · F3 ctx · Esc cancels)", CtxOverride = "Context limit override (Enter applies, empty = auto, Esc cancels)"; `f.set_cursor_position` at input + char count.
- [ ] `handle_key` Chat mode: Char inserts, Backspace pops, Enter submits (`parse_slash_command`: Exit quits; Help prints command list incl. F2/F3/Esc/Ctrl+C; Clear resets transcript+`messages`+scroll; Model(None) prints `list_local_models()` as `name  quant  ctx` lines; Model(Some) sends `LoadModel`; Unknown prints error; plain text appends user message to transcript + `messages`), Ctrl+C quits, PageUp/Down scroll, F2 toggles `show_sidebar`, F3 clears input + enters CtxOverride, Esc sends `Cancel`. CtxOverride mode: Esc returns to Chat unchanged; Enter runs `parse_ctx_override(input)`: Auto -> send `SetContextLimit { None }` + hint "ctx limit: auto"; Apply(n) -> `SetContextLimit { Some(n) }` + hint "ctx limit: n"; Invalid -> hint "enter a number or empty" and stay; typing edits input.
- [ ] `handle_event`: `Token` appends `streaming`; `TurnComplete` pushes `streaming` + stats line (`enc {:.1} ms | ttft {} | {} | {} tok`, decode via `format_tps(stats.decode_tps)`) to transcript, clears streaming, clears `generating`; `Diagnostics` replaces `snap`; `Error` prints as system line.
- [ ] Manual: UI opens on "model: none loaded", typing echoes, `/help`, `/model` list, F2 toggles, F3 full flow, Ctrl+C quits leaving shell clean. fmt+clippy. Commit `feat(cli): tui terminal shell and event loop`.

### Task 9: Model loading, hot-swap, and telemetry

**Files:** `tui/worker.rs` + `mod.rs` hook. Replace the `LoadModel` stub.

- [ ] Send `ModelLoadStarted`; resolve `resolve_model_preferring_grim(&name)` else literal existing path, else `ModelLoadFailed` (old model, if any, stays loaded and usable).
- [ ] Hot-swap order (never a silent no-model state): (1) `load_from_path(new)`; (2) Ok -> `register_model(new_id)` THEN `unload_model(old)` if any; (3) Err with an old model resident -> `unload_model(old)`, retry the load once (frees VRAM overlap); (4) retry fails -> `ModelLoadFailed`, snapshot honestly shows no model. Two models coexist briefly in (2); the serial worker prevents any `Generate` interleaving, and Task 10 passes an explicit `model_id` regardless. After success: tokenizer by extension (run.rs:615-636), `vocab` via config downcasts (run.rs:647-661) else tokenizer len else 512, catalog entry match by path from `list_local_models()` gives `quant` + `context_length`, `strategy_label(engine.strategy_for(id))`, rebuild sampler, send `ModelLoadOk`.
- [ ] `fn snapshot(&self, ...) -> DiagnosticsSnapshot`: kv from `kv_cache_telemetry()`, prefill `last_ttft_ms()` (already `Option<f64>`), decode `tokens_per_sec().map(|v| v as f64)` (f32 EMA -> f64, stays `Option`), tracked turn fields, `ctx_used` = prompt + generated tokens, `ctx_limit` = `ctx_override.unwrap_or(catalog_limit)`; VRAM/RAM:

```rust
let n = grim_engine::model_loader::resolve_discrete_rocm_devices().len();
let (vram_used, vram_total, _) = grim_server::probe_vram_and_gpus(n);
let (ram_used, ram_total) = grim_server::probe_sys_ram();
```

Send `Diagnostics` after every terminal event.
- [ ] `mod.rs` on `ModelLoadOk`: clear `messages`/`streaming`, print `model loaded: <name> (strategy)`. On `ModelLoadFailed`: error line; if the old model survived, add "still loaded: <old>".
- [ ] Manual: `GRIM_BACKEND=cpu cargo run -p grim-cli -- tui models/LFM2.5-230M-Q8_0.gguf` fills sidebar; `/model LFM2.5-350M-Q8_0` swaps live; `/model bogus` fails with 230M still loaded. fmt+clippy. Commit `feat(cli): tui model loading and hot-swap`.

### Task 10: Generation loop with streaming, cancel, absent-logits handling

**Files:** `tui/worker.rs` + `mod.rs`. Replace the `Generate` stub, mirroring grim-server/src/lib.rs:299-384.

- [ ] Time prompt building for `encode_ms`: chat-template render + `bos_prefix` + `tok.encode` (copy run.rs:863-886).
- [ ] Enqueue `grim_scheduler::Request { id: next_id, prompt_tokens: len, priority: 0, consumed_tokens: 0, model_id: Some(current_id.clone()) (always explicit; never rely on the None default, whose "first registered" ordering is wrong during the Task 9 hot-swap window), adapter_ids: vec![], input_ids: Some(ids) }`.
- [ ] Loop while `generated < max_tokens`: (a) `try_recv` a command: `Cancel` sets `cancelled = true`, break; other commands stay queued for the outer loop; (b) `engine.tick()` (Err -> `Error` event, break); (c) `last_outcome(id)`: if `speculative`, accumulate `accepted_tokens`; `outcome.logits` is `Option<Arc<Tensor>>`: on `None` the request was not driven this tick, so skip sampling entirely (emit nothing, not EOS) and check a wall-clock stall guard: `last_logits_at: Instant` (init at loop start, reset on every `Some`) against `const NO_LOGITS_TIMEOUT: Duration = Duration::from_secs(10)`; wall-clock, not iteration count, so tick cost is irrelevant; timeout -> `Error { "no logits for 10s" }` + break. On `Some`: `to_vec_f32()`, slice `len - vocab..`, sample with history (prompt + generated so far), clamp to `vocab-1`, `record_generated_token`, break on `is_eos_token`, else push, send `Token { text: tok.decode(&[t]) }`; (d) every 100 ms send `Diagnostics`.
- [ ] After the loop (all exits: EOS, max_tokens, cancel, error, stall): `engine.finish_request(id)`; send `TurnComplete` (prefill = `last_ttft_ms()`, `decode_tps` = tokens/elapsed when `tokens_generated > 0` else `None`, `accepted_per_step` via `acceptance_rate`, `cancelled`, `context_used`) + final snapshot.
- [ ] `mod.rs`: Enter sends `Generate { messages }` only when a model is loaded and not generating (else hint); Esc sends `Cancel`; `generating` set true on submit, false on `TurnComplete`.
- [ ] Manual on cpu + 230M: tokens stream, decode/kv/ram update live, Esc mid-stream stops but prints stats (decode `n/a` if cancelled before first token), second turn includes history. fmt+clippy. Commit `feat(cli): tui streaming generation with live diagnostics`.

### Task 11: Polish and full verification

- [ ] `/help` covers commands + F2/F3/Esc/Ctrl+C.
- [ ] Enter during generation rejected with visible hint; input stays editable.
- [ ] `cargo fmt && cargo clippy -p grim-cli --all-targets && cargo test -p grim-cli && cargo test -p grim-server` clean.
- [ ] GPU smoke (no GRIM_BACKEND): load 230M, one turn, `/model` hot-swap, F2, F3, cancel, `/exit`; vram shows real GiB; terminal clean after. Commit `feat(cli): complete grim tui diagnostics chat interface`.
