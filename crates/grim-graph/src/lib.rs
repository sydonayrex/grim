pub mod norm;
pub mod linear;
pub mod rope;

pub use norm::{NormSpec, emit_norm, NormKind};
pub use linear::{QuantFlags, emit_linear_dedup};
pub use rope::{RopeArgs, emit_rope};
