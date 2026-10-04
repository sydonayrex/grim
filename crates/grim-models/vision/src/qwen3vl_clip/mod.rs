//! Qwen3-VL clip (`mmproj`) support: geometry, weights, and forward pass.
//!
//! The vision tower lives in its own GGUF file beside the text model, declared
//! `general.architecture=clip` / `general.type=mmproj` with
//! `clip.projector_type=qwen3vl_merger`. Its forward-pass spec is
//! `old/repo/llama.cpp-master/tools/mtmd/models/qwen3vl.cpp`; every module here
//! cites the line it implements.
//!
//! Structure:
//! * [`config`] - geometry read from the checkpoint's metadata, never from
//!   literals.
//! * [`weights`] - tensor loading, the two-kernel patch-embed sum, and the
//!   degeneracy gate.

mod config;
mod weights;

pub use config::Qwen3VlClipConfig;
pub use weights::{Qwen3VlBlock, Qwen3VlClip, assert_non_degenerate};
