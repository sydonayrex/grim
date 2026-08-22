//! `grim-constrain` — structured/grammar-constrained decoding for grim.
//!
//! WI-3a (JSON-mode) + WI-3b (JSON-Schema): samplers that wrap any
//! `grim_core::sampler::Sampler` and mask logits so generated tokens keep
//! the output on a valid FSM path.
//!
//! Design constraints (per the plan):
//! - Native Rust, no Python bridge, no external grammar library.
//! - Backend-agnostic: operates purely on `Tensor` logits + token strings,
//!   same as the existing `Sampler` trait. No CUDA/ROCm/Metal dependency.
//! - The `Sampler` trait itself is **unmodified** — this is wrapping, not
//!   altering, so no existing sampler needs changes.

pub mod json_fsm;
pub mod sampler;
pub mod schema;

pub use json_fsm::{FsmCheck, JsonState, TokenMaskCache, apply_mask};
pub use sampler::{ConstrainedSampler, Constraint, constrained_json_object};
pub use schema::{BoundedRegex, JsonSchemaCompilerError, compile_json_schema, validate_pattern};
