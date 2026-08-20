//! Neural-network modules (`Linear`, `Embedding`, `RmsNorm`, `RoPE`, `SwiGLU`) and `WeightSource` loading.

pub mod modules;
/// Mixture-of-Experts primitives (router, expert bank, routed FFN).
pub mod moe;
pub mod sparse_attention;
/// SCYTHE-2 WI-3: capacity-calibrated sharded linears.
pub mod scythe2;
pub mod varbuilder;

pub use modules::{
    ColumnParallelLinear, Embedding, KdaAttention, KdaLayerCache, LayerCache, LayerNorm, Linear,
    LinearAttentionBlock, LinearAttentionLayerCache, MlaAttention, MlaKvCache, RmsNorm, Rope,
    RowParallelLinear, TensorParallelConfig, add_tensors, pick_device_for_storage_device,
    pick_device_for_tensor, require_single_device, short_conv1d,
};


pub use scythe2::{Scythe2Linear, slice_input_dim, slice_output_dim};
pub use varbuilder::WeightSource;
