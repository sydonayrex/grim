//! `grim-speculative` — default-on speculative decoding for Grim.
//!
//! Per §5.3, the engine picks one of three paths automatically:
//!
//! - **Native MTP** — when the target model implements `NativeMtp` itself
//!   (deepseek-V3 / Gemma-4-assistant style). Zero-config; no extra model,
//!   no distillation, no weight file. Speculation is folded into the model's
//!   own compute (shared trunk, shared KV cache).
//!
//! - **DSpark** — confidence-scheduled semi-autoregressive drafter. Default
//!   when an attached `DraftBackbone` + `MarkovHead` + `ConfidenceHead`
//!   bundle is available. Bundle produced by `grim spec train` against the
//!   exact deployment checkpoint (QAT-aware).
//!
//! - **Plain** — pure autoregressive. Honest fallback. Always available.

pub mod confidence_head;
pub mod confidence_scheduler;
pub mod draft_backbone;
pub mod markov_head;
pub mod native_mtp;
pub mod speculative_wrapper;

pub use confidence_head::ConfidenceHead;
pub use confidence_scheduler::{
    ConfidenceScheduler, SpeculationConfig, ThroughputProfile,
};
pub use draft_backbone::{DraftBackbone, DraftBlock};
pub use markov_head::MarkovHead;
pub use native_mtp::NativeMtp;
pub use speculative_wrapper::{SpeculativeCausalLm, Strategy};
