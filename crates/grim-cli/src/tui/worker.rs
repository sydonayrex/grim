//! Worker thread and channel protocol for the in-process engine.
//!
//! The worker owns the `Engine`, tokenizer, and sampler. The UI thread owns
//! the terminal. GPU and model code runs only here, wrapped in
//! `catch_unwind` so a backend panic becomes an `Error` event instead of
//! killing the UI thread.

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use super::diagnostics::{self};
use grim_core::sampler::{Sampler, SamplingParams, ThinkingLevel};
use grim_engine::{Engine, EngineConfig, Request};
use grim_format::{GgufProvider, GgufTokenizer, render_messages_or_last};
use grim_speculative::Strategy;
use grim_tensor::Device;

/// Command sent from the UI thread to the worker.
#[derive(Debug)]
pub enum WorkerCommand {
    LoadModel {
        name: String,
    },
    Generate {
        messages: Vec<grim_format::ChatMessage>,
    },
    SetContextLimit {
        limit: Option<u64>,
    },
    SetSamplingParams {
        temperature: Option<f32>,
        top_p: Option<f32>,
    },
    Cancel,
    Quit,
}

/// Event produced by the worker and consumed by the UI thread.
#[derive(Debug)]
pub enum WorkerEvent {
    ModelLoadStarted {
        name: String,
    },
    ModelLoadOk {
        name: String,
        quant: Option<String>,
        context_length: u64,
        strategy: String,
    },
    ModelLoadFailed {
        name: String,
        error: String,
    },
    Token {
        text: String,
    },
    TurnComplete {
        stats: TurnStats,
    },
    Diagnostics {
        snap: DiagnosticsSnapshot,
    },
    Error {
        message: String,
    },
}

pub use super::diagnostics::DiagnosticsSnapshot;

/// Turn-level statistics emitted with `TurnComplete`.
///
/// `decode_tps` is `None` when the turn produced no tokens (e.g. immediate
/// cancel). We never invent a number to fill the gap.
#[derive(Debug)]
pub struct TurnStats {
    pub encode_ms: f64,
    pub prompt_tokens: usize,
    pub prefill_ms: Option<f64>,
    pub decode_tps: Option<f64>,
    pub tokens_generated: usize,
    pub accepted_per_step: Option<f64>,
    pub cancelled: bool,
    pub context_used: u64,
}

/// Sampling parameters forwarded to the worker at construction time.
#[derive(Debug, Clone)]
pub struct WorkerParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
}

/// True if the token id is an end-of-stream token for this tokenizer.
/// Mirrors the EOS check in run.rs:931-939.
pub fn is_eos_token(tok: &GgufTokenizer, id: u32) -> bool {
    tok.eos_token_id.map_or(false, |eos| id == eos)
        || tok.token_to_id.get("<|im_end|>").copied() == Some(id)
        || tok.token_to_id.get("<|endoftext|>").copied() == Some(id)
        || tok.token_to_id.get("</s>").copied() == Some(id)
}

/// BOS prefix candidates, mirroring run.rs:873-879.
pub fn bos_prefix(tok: &GgufTokenizer) -> Vec<u32> {
    for candidate in ["<|startoftext|>", "<s>", "<|im_start|>"] {
        if let Some(&id) = tok.token_to_id.get(candidate) {
            return vec![id];
        }
    }
    Vec::new()
}

/// Extracts a panic message string from a panic payload, if possible.
pub fn panic_message(p: Box<dyn Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        return s.to_string();
    }
    if let Some(s) = p.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic".into()
}

/// Outcome of a single worker command.
enum WorkerOutcome {
    Quit,
    Ignored,
    ModelLoadOk {
        name: String,
        quant: Option<String>,
        context_length: u64,
        strategy: String,
    },
    ModelLoadFailed {
        name: String,
        error: String,
    },
}

/// Owns all engine state for the lifetime of the worker thread.
///
/// `rx` is intentionally *not* stored here: it lives in `spawn_worker` so a
/// panic in any handler leaves it intact and the loop keeps draining commands.
struct Worker {
    engine: Engine,
    sampler: Box<dyn Sampler>,
    sampling_params: SamplingParams,
    seed: u64,
    tokenizer: Option<GgufTokenizer>,
    vocab: usize,
    current_id: Option<String>,
    ctx_override: Option<u64>,
    next_request_id: u64,
    max_tokens: usize,
    tx: Sender<WorkerEvent>,
}

impl Worker {
    fn handle(&mut self, cmd: WorkerCommand, rx: &Receiver<WorkerCommand>) -> WorkerOutcome {
        match cmd {
            WorkerCommand::Quit => WorkerOutcome::Quit,
            WorkerCommand::Cancel => WorkerOutcome::Ignored,
            WorkerCommand::SetContextLimit { limit } => {
                self.ctx_override = limit;
                WorkerOutcome::Ignored
            }
            WorkerCommand::SetSamplingParams {
                temperature,
                top_p,
            } => {
                if let Some(t) = temperature {
                    self.sampling_params.temperature = t;
                }
                if let Some(p) = top_p {
                    self.sampling_params.top_p = p;
                }
                self.sampler = self.sampling_params.clone().into_sampler(self.seed);
                WorkerOutcome::Ignored
            }
            WorkerCommand::LoadModel { name } => self.load_model(name),
            WorkerCommand::Generate { messages } => {
                self.generate(messages, rx);
                WorkerOutcome::Ignored
            }
        }
    }

    /// Load a model, hot-swapping the previous one. Never leaves the worker
    /// in a silent no-model state: on failure the old model stays loaded and
    /// the UI is told the honest state.
    fn load_model(&mut self, name: String) -> WorkerOutcome {
        let _ = self
            .tx
            .send(WorkerEvent::ModelLoadStarted { name: name.clone() });

        let resolved = grim_core::catalog::resolve_model_preferring_grim(&name).or_else(|| {
            let p = std::path::Path::new(&name);
            if p.exists() {
                Some(p.to_path_buf())
            } else {
                None
            }
        });

        let Some(resolved) = resolved else {
            return WorkerOutcome::ModelLoadFailed {
                name,
                error: "not found".into(),
            };
        };
        let path_str = resolved.to_string_lossy().to_string();

        // 1. new load (old model stays resident so a failure keeps it usable).
        let model = match grim_engine::model_loader::load_from_path(&path_str) {
            Ok(m) => m,
            Err(e) => {
                // 3. old model resident: retry once after dropping it to free VRAM.
                if let Some(old) = self.current_id.clone() {
                    self.engine.unload_model(&old);
                    match grim_engine::model_loader::load_from_path(&path_str) {
                        Ok(m) => m,
                        Err(e2) => {
                            return WorkerOutcome::ModelLoadFailed {
                                name,
                                error: format!("load failed after retry: {e2}"),
                            };
                        }
                    }
                } else {
                    return WorkerOutcome::ModelLoadFailed {
                        name,
                        error: format!("load failed: {e}"),
                    };
                }
            }
        };

        let id = resolved
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&name)
            .to_string();

        // 2. register new, then unload old (brief coexistence; the serial
        // worker prevents any Generate from interleaving).
        self.engine.register_model(&id, model);
        if let Some(old) = self.current_id.clone() {
            if old != id {
                self.engine.unload_model(&old);
            }
        }

        // tokenizer by extension (run.rs:615-636 pattern).
        self.tokenizer = tokenizer_for_path(&path_str);
        self.vocab = self
            .tokenizer
            .as_ref()
            .map(|t| t.tokens.len())
            .unwrap_or(512);

        // sampler is rebuilt with the configured sampling params and seed on every load.
        self.sampler = self.sampling_params.clone().into_sampler(self.seed);

        let catalog_entry = grim_core::catalog::list_local_models()
            .into_iter()
            .find(|e| e.path == path_str);
        let quant = catalog_entry.as_ref().map(|e| e.quant.clone());
        let context_length = catalog_entry
            .as_ref()
            .map(|e| e.context_length)
            .unwrap_or(0);
        let strategy = self
            .engine
            .strategy_for(&id)
            .map(|s| match s {
                Strategy::Plain => "plain (no speculation)",
                Strategy::DSpark => "DSpark",
                Strategy::NativeMtp => "native MTP",
            })
            .unwrap_or("plain")
            .to_string();

        self.current_id = Some(id.clone());
        let snapshot = self.make_snapshot(context_length, 0);
        let _ = self.tx.send(WorkerEvent::Diagnostics { snap: snapshot });
        WorkerOutcome::ModelLoadOk {
            name,
            quant,
            context_length,
            strategy,
        }
    }

    /// Build a snapshot from engine telemetry + turn fields.
    fn make_snapshot(&self, context_length: u64, context_used: u64) -> DiagnosticsSnapshot {
        let (kv_used, kv_total, kv_used_blk, kv_total_blk) = self.engine.kv_cache_telemetry();
        let n = grim_engine::model_loader::resolve_discrete_rocm_devices(&Device::Cpu).len();
        let (vram_used, vram_total, _) = grim_server::probe_vram_and_gpus(n);
        let (ram_used, ram_total) = grim_server::probe_sys_ram();

        DiagnosticsSnapshot {
            model_name: self.current_id.clone(),
            quant: None,
            backend: "rocm".into(),
            strategy: None,
            encode_ms: None,
            prompt_tokens: 0,
            prefill_ms: self.engine.last_ttft_ms(),
            decode_tps: self.engine.tokens_per_sec().map(|v| v as f64),
            turn_tps: None,
            tokens_generated: 0,
            kv_used_bytes: kv_used,
            kv_total_bytes: kv_total,
            kv_blocks_used: kv_used_blk,
            kv_blocks_total: kv_total_blk,
            ctx_used: context_used,
            ctx_limit: self.ctx_override.unwrap_or(context_length),
            accepted_per_step: None,
            vram_used_bytes: vram_used,
            vram_total_bytes: vram_total,
            ram_used_bytes: ram_used,
            ram_total_bytes: ram_total,
            loading: false,
            generating: false,
        }
    }

    /// Run one turn of generation, streaming tokens and diagnostics.
    fn generate(&mut self, messages: Vec<grim_format::ChatMessage>, rx: &Receiver<WorkerCommand>) {
        let Some(model_id) = self.current_id.clone() else {
            let _ = self.tx.send(WorkerEvent::Error {
                message: "no model loaded; use /model first".into(),
            });
            return;
        };
        let Some(tok) = self.tokenizer.clone() else {
            let _ = self.tx.send(WorkerEvent::Error {
                message: "tokenizer unavailable".into(),
            });
            return;
        };

        let t0 = Instant::now();
        let prompt_ids = build_prompt_ids(&tok, &messages);
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let ctx_limit = self.ctx_override.unwrap_or_else(|| {
            grim_core::catalog::list_local_models()
                .iter()
                .find(|e| e.path.ends_with(&model_id))
                .map(|e| e.context_length)
                .unwrap_or(0)
        });

        let req = Request {
            id: request_id,
            prompt_tokens: prompt_ids.len(),
            max_new_tokens: 0,
            priority: 0,
            consumed_tokens: 0,
            // always explicit: the None default routes to "first registered",
            // which is wrong during the hot-swap window.
            model_id: Some(model_id),
            adapter_ids: vec![],
            input_ids: Some(prompt_ids.clone()),
        };

        let mut history: Vec<u32> = prompt_ids.clone();
        let mut generated_count: usize = 0;
        let mut accepted_total: usize = 0;
        let mut accepted_steps: usize = 0;
        let mut cancelled = false;
        let mut last_logits_at = Instant::now();
        let no_logits_timeout = Duration::from_secs(10);
        let mut last_diag = Instant::now();
        let turn_start = Instant::now();

        if let Err(e) = self.engine.enqueue_request(req) {
            let _ = self.tx.send(WorkerEvent::Error {
                message: format!("enqueue failed: {e}"),
            });
            return;
        }

        loop {
            match rx.try_recv() {
                Ok(WorkerCommand::Cancel) => {
                    cancelled = true;
                    break;
                }
                Ok(_) => {}
                Err(_) => {}
            }
            if generated_count >= self.max_tokens {
                break;
            }
            if let Err(e) = self.engine.tick() {
                let _ = self.tx.send(WorkerEvent::Error {
                    message: format!("engine tick failed: {e}"),
                });
                break;
            }
            let outcome = self.engine.last_outcome(request_id);
            let Some(outcome) = outcome else {
                // Request not driven this tick (e.g. paused). Skip sampling;
                // stall-guard bounds wall-clock time, not tick count.
                if last_logits_at.elapsed() > no_logits_timeout {
                    let _ = self.tx.send(WorkerEvent::Error {
                        message: "engine produced no logits for 10s".into(),
                    });
                    break;
                }
                continue;
            };
            let Some(logits) = outcome.logits.as_deref() else {
                continue;
            };
            last_logits_at = Instant::now();

            if outcome.speculative {
                accepted_total += outcome.accepted_tokens.max(1);
                accepted_steps += 1;
            }

            let full = match logits.to_vec_f32() {
                Ok(v) => v,
                Err(e) => {
                    let _ = self.tx.send(WorkerEvent::Error {
                        message: format!("logits readback failed: {e}"),
                    });
                    break;
                }
            };
            let vocab = self.vocab;
            let start = full.len().saturating_sub(vocab);
            let slice = &full[start..];
            let local = grim_backend_cpu::cpu_tensor(
                slice.to_vec(),
                grim_tensor::Shape::new(vec![slice.len()]),
            );
            let token = self
                .sampler
                .sample(&local, &history)
                .unwrap_or(0)
                .min((vocab as u32).saturating_sub(1));
            history.push(token);
            generated_count += 1;
            self.engine.record_generated_token(request_id, token);

            if is_eos_token(&tok, token) {
                break;
            }
            let text = tok.decode(&[token]);
            let _ = self.tx.send(WorkerEvent::Token { text });

            if last_diag.elapsed() >= Duration::from_millis(100) {
                let snap =
                    self.make_snapshot(ctx_limit, (prompt_ids.len() + generated_count) as u64);
                let _ = self.tx.send(WorkerEvent::Diagnostics { snap });
                last_diag = Instant::now();
            }
        }

        self.engine.finish_request(request_id);

        let elapsed = turn_start.elapsed().as_secs_f64();
        let stats = TurnStats {
            encode_ms,
            prompt_tokens: prompt_ids.len(),
            prefill_ms: self.engine.last_ttft_ms(),
            decode_tps: if generated_count > 0 {
                Some(generated_count as f64 / elapsed.max(1e-9))
            } else {
                None
            },
            tokens_generated: generated_count,
            accepted_per_step: diagnostics::acceptance_rate(accepted_total, accepted_steps),
            cancelled,
            context_used: (prompt_ids.len() + generated_count) as u64,
        };
        let snap = self.make_snapshot(ctx_limit, stats.context_used);
        let _ = self.tx.send(WorkerEvent::Diagnostics { snap });
        let _ = self.tx.send(WorkerEvent::TurnComplete { stats });
    }
}

/// Build prompt token ids: BOS (when configured or a candidate exists) plus
/// the chat-template rendered messages, mirroring run.rs:863-886.
fn build_prompt_ids(tok: &GgufTokenizer, messages: &[grim_format::ChatMessage]) -> Vec<u32> {
    let mut ids = Vec::new();
    if tok.add_bos_token {
        if let Some(b) = tok.bos_token_id {
            ids.push(b);
        }
    } else if let Some(&b) = bos_prefix(tok).first() {
        ids.push(b);
    }
    let text = if tok.chat_template.is_some() {
        render_messages_or_last(tok, messages)
    } else {
        messages
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default()
    };
    ids.extend(tok.encode(&text));
    ids
}

/// Load a tokenizer by file extension (run.rs:615-636 pattern). Best-effort:
/// a missing tokenizer results in raw-id decoding.
fn tokenizer_for_path(path_str: &str) -> Option<GgufTokenizer> {
    let lower = path_str.to_lowercase();
    if lower.ends_with(".gguf") {
        if let Ok(provider) = GgufProvider::open(path_str) {
            return provider.tokenizer().ok();
        }
    } else if lower.ends_with(".grim") {
        let sibling = format!("{}.gguf", path_str.trim_end_matches(".grim"));
        if std::path::Path::new(&sibling).exists() {
            if let Ok(provider) = GgufProvider::open(&sibling) {
                return provider.tokenizer().ok();
            }
        }
    } else if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        let dir = std::path::Path::new(path_str)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let tj = dir.join("tokenizer.json");
        if tj.exists() {
            return GgufTokenizer::from_hf_json(tj.to_str().unwrap()).ok();
        }
    }
    None
}

/// Spawn the worker thread. The worker exits on `Quit`; a backend panic in
/// any command handler becomes an `Error` event rather than killing the UI
/// thread. `rx` lives in this scope (outside `Worker`) so a panic leaves it
/// intact and the loop keeps draining commands afterwards.
pub fn spawn_worker(
    params: WorkerParams,
    rx: Receiver<WorkerCommand>,
    tx: Sender<WorkerEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let sampling = SamplingParams {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
            thinking_level: ThinkingLevel::Default,
        };
        let mut worker = Worker {
            engine: Engine::new(EngineConfig::default()),
            sampler: sampling.clone().into_sampler(params.seed),
            sampling_params: sampling,
            seed: params.seed,
            tokenizer: None,
            vocab: 512,
            current_id: None,
            ctx_override: None,
            next_request_id: 1,
            max_tokens: params.max_tokens,
            tx,
        };

        loop {
            let cmd = match rx.recv() {
                Ok(c) => c,
                Err(_) => break,
            };
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| worker.handle(cmd, &rx)));
            match result {
                Ok(WorkerOutcome::Quit) => break,
                Ok(WorkerOutcome::Ignored) => {}
                Ok(WorkerOutcome::ModelLoadOk {
                    name,
                    quant,
                    context_length,
                    strategy,
                }) => {
                    let _ = worker.tx.send(WorkerEvent::ModelLoadOk {
                        name,
                        quant,
                        context_length,
                        strategy,
                    });
                }
                Ok(WorkerOutcome::ModelLoadFailed { name, error }) => {
                    let _ = worker.tx.send(WorkerEvent::ModelLoadFailed { name, error });
                }
                Err(payload) => {
                    let _ = worker.tx.send(WorkerEvent::Error {
                        message: panic_message(payload),
                    });
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tok() -> GgufTokenizer {
        let pairs = [
            ("<s>", 0u32),
            ("<|im_end|>", 3),
            ("<|endoftext|>", 4),
            ("</s>", 5),
        ];
        let mut token_to_id = HashMap::new();
        for (k, v) in pairs {
            token_to_id.insert(k.to_string(), v);
        }
        GgufTokenizer {
            tokens: vec!["<s>".into()],
            token_to_id,
            scores: None,
            model_type: String::new(),
            bpe_merges: None,
            byte_decoder: None,
            eos_token_id: Some(2),
            bos_token_id: Some(0),
            add_bos_token: false,
            unk_token_id: None,
            chat_template: None,
        }
    }

    #[test]
    fn eos_detection_covers_all_stop_tokens() {
        let t = tok();
        assert!(is_eos_token(&t, 2));
        assert!(is_eos_token(&t, 3));
        assert!(is_eos_token(&t, 4));
        assert!(is_eos_token(&t, 5));
        assert!(!is_eos_token(&t, 9));
    }

    #[test]
    fn bos_prefix_finds_first_candidate() {
        assert_eq!(bos_prefix(&tok()), vec![0]);
    }

    #[test]
    fn panic_message_extracts_string() {
        let p = std::panic::catch_unwind(|| panic!("boom")).unwrap_err();
        assert!(panic_message(p).contains("boom"));
        assert_eq!(panic_message(Box::new("plain string")), "plain string");
        assert_eq!(panic_message(Box::new(42_i32)), "unknown panic");
    }

    #[test]
    fn build_prompt_ids_prepends_bos() {
        let t = tok();
        let ids = build_prompt_ids(&t, &[]);
        assert_eq!(ids.first(), Some(&0));
    }
}
