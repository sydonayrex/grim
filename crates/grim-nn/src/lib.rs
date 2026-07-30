//! `grim-nn` — neural-network building blocks & `WeightSource`
//! (VarBuilder-equivalent). No transport / scheduling code here; this
//! crate is the natural seam between raw `grim-tensor` data movement and
//! model-shaped code in `grim-models`.

pub mod modules;
pub mod varbuilder;
/// SCYTHE-2 WI-3: capacity-calibrated sharded linears.
pub mod scythe2;

pub use modules::{
    add_tensors, pick_device_for_storage_device, pick_device_for_tensor, ColumnParallelLinear,
    Embedding, Linear, RmsNorm, Rope, RowParallelLinear, TensorParallelConfig,
};
pub use varbuilder::WeightSource;
pub use scythe2::{Scythe2Linear, slice_input_dim, slice_output_dim};
