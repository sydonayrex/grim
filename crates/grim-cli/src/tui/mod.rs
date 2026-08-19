//! grim tui — ratatui chat interface over the in-process engine.
//!
//! Two threads: the UI thread owns the terminal and `App`; the worker thread
//! owns `Engine`, the tokenizer, and the sampler. They talk over two
//! `std::sync::mpsc` channels. GPU and model code runs only on the worker.

/// Diagnostics formatting helpers for the TUI.
pub mod diagnostics;

/// Worker thread and channel protocol.
pub mod worker;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_loads() {
        assert!(diagnostics::format_bytes::<u32>(0).is_empty() == false);
    }
}
