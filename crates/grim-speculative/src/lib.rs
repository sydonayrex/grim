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
pub mod entropy_confidence_head;
pub mod llama_mtp_adapter;
pub mod markov_head;
pub mod native_mtp;
pub mod speculative_wrapper;
pub mod distill;
pub mod mamba_speculative;
pub mod test_rng;
pub mod tiny_draft_backbone;
pub mod uniform_markov_head;

pub use confidence_head::ConfidenceHead;
pub use confidence_scheduler::{
    ConfidenceScheduler, SpeculationConfig, ThroughputProfile,
};
pub use draft_backbone::{DraftBackbone, DraftBlock};
pub use entropy_confidence_head::EntropyConfidenceHead;
pub use llama_mtp_adapter::LlamaMtpAdapter;
pub use markov_head::MarkovHead;
pub use native_mtp::NativeMtp;
pub use speculative_wrapper::{SpeculativeCausalLm, Strategy};
pub use distill::train_speculative_draft;
pub use mamba_speculative::{MambaSpeculativeEngine, MambaStepState};
pub use tiny_draft_backbone::TinyDraftBackbone;
pub use uniform_markov_head::UniformMarkovHead;
