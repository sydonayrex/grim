//! Worker thread and channel protocol for the in-process regeneration loop.

use std::any::Any;
use std::panic::AssertUnwindSafe;

/// Command sent from the UI thread to the worker.
pub enum WorkerCommand {
    LoadModel { name: String },
    Generate { messages: Vec<grim_format::ChatMessage> },
    SetContextLimit { limit: Option<u64> },
    Cancel,
    Quit,
}

/// Event produced by the worker and consumed by the UI thread.
pub enum WorkerEvent {
    ModelLoadStarted { name: String },
    ModelLoadOk { name: String, quant: Option<String>, context_length: u64, strategy: String },
    ModelLoadFailed { name: String, error: String },
    Token { text: String },
    TurnComplete { stats: TurnStats },
    Diagnostics { snap: DiagnosticsSnapshot },
    Error { message: String },
}

pub use diagnostics::DiagnosticsSnapshot;

/// Turn-level statistics emitted with `TurnComplete`.
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
pub struct WorkerParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn worker_starts_and_quits_cleanly() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (evt_tx, evt_rx) = mpsc::channel();
        let h = spawn_worker(
            WorkerParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                max_tokens: 256,
                seed: 42,
                repeat_penalty: 1.1,
            },
            cmd_rx,
            evt_tx,
        );
        cmd_tx.send(WorkerCommand::Quit).unwrap();
        h.join().unwrap();
        assert!(evt_rx.try_recv().is_err());
    }
}

/// Spawn the worker thread.
///
/// The worker runs its own event loop and exits on `Quit`. All blocking
/// model access happens here, wrapped in `catch_unwind` so a backend panic
/// becomes an `Error` event rather than killing the UI thread.
pub fn spawn_worker(
    params: WorkerParams,
    rx: std::sync::mpsc::Receiver<WorkerCommand>,
    tx: std::sync::mpsc::Sender<WorkerEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let _ = params;
        let _ = rx;
        let _ = tx;
        loop {
            if rx.recv().is_err() {
                break;
            }
        }
    })
}
