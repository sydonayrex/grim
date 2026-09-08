//! `grim-constrain` - structured/grammar-constrained decoding for grim.
//! WI-3a (JSON-mode) + WI-3b (JSON-Schema): samplers that wrap any `grim_core::sampler::Sampler` and mask logits so generated.

pub mod json_fsm;
pub mod sampler;
pub mod schema;

pub use json_fsm::{FsmCheck, JsonState, TokenMaskCache, apply_mask};
pub use sampler::{ConstrainedSampler, Constraint, constrained_json_object};
pub use schema::{BoundedRegex, JsonSchemaCompilerError, compile_json_schema, validate_pattern};
