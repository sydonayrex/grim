//! Neural-network modules (`Linear`, `Embedding`, `RmsNorm`, `RoPE`, `SwiGLU`) and `WeightSource` loading.

pub mod modules;
/// SCYTHE-2 WI-3: capacity-calibrated sharded linears.
pub mod scythe2;
pub mod varbuilder;

pub use modules::{
    ColumnParallelLinear, Embedding, Linear, RmsNorm, Rope, RowParallelLinear,
    TensorParallelConfig, add_tensors, pick_device_for_storage_device, pick_device_for_tensor,
};
pub use scythe2::{Scythe2Linear, slice_input_dim, slice_output_dim};
pub use varbuilder::WeightSource;
