pub mod norm;
pub mod linear;

pub use norm::{NormSpec, emit_norm, NormKind, NormEmitter};
pub use linear::{QuantFlags, emit_linear_dedup, LinearEmitter};
