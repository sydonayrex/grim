//! `grim-nn` — neural-network building blocks & `WeightSource`
//! (VarBuilder-equivalent). No transport / scheduling code here; this
//! crate is the natural seam between raw `grim-tensor` data movement and
//! model-shaped code in `grim-models`.

pub mod modules;
pub mod varbuilder;

pub use modules::{Embedding, Linear, RmsNorm, Rope};
pub use varbuilder::WeightSource;
