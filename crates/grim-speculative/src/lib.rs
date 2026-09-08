//! `grim-speculative` - default-on speculative decoding for Grim.
//! Per §5.3, the engine picks one of three paths automatically: - **Native MTP** - when.

pub mod confidence_head;
pub mod confidence_scheduler;
pub mod depth_tuner;
pub mod distill;
pub mod draft_backbone;
pub mod eagle3_drafter;
pub mod entropy_confidence_head;
pub mod llama_mtp_adapter;
pub mod mamba_speculative;
pub mod markov_head;
pub mod native_mtp;
pub mod speculative_wrapper;

pub mod tiny_draft_backbone;
pub mod uniform_markov_head;

pub use confidence_head::ConfidenceHead;
pub use confidence_scheduler::{
    ConfidenceScheduler, SpeculationConfig, SpeculationDepthConfig, ThroughputProfile,
};
pub use depth_tuner::{SpeculativeDepthPidConfig, SpeculativeDepthPidController};
pub use distill::{compress_distill_report, train_speculative_draft};
pub use draft_backbone::{DraftBackbone, DraftBlock};
pub use eagle3_drafter::Eagle3Drafter;
pub use entropy_confidence_head::EntropyConfidenceHead;
pub use llama_mtp_adapter::LlamaMtpAdapter;
pub use mamba_speculative::{MambaSpeculativeEngine, MambaStepState};
pub use markov_head::MarkovHead;
pub use native_mtp::NativeMtp;
pub use speculative_wrapper::{SpeculativeCausalLm, SpeculativeTelemetry, Strategy};
pub use tiny_draft_backbone::TinyDraftBackbone;
pub use uniform_markov_head::UniformMarkovHead;
