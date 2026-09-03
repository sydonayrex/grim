//! Neural-network modules (`Linear`, `Embedding`, `RmsNorm`, `RoPE`, `SwiGLU`) and `WeightSource` loading.

pub mod modules;
/// Mixture-of-Experts primitives (router, expert bank, routed FFN).
pub mod moe;
/// Deterministic token mapping and scoreboard synchronization (UniEP).
pub mod moe_deterministic;
/// Bandwidth-adaptive CPU-GPU hybrid execution (FreeToken).
pub mod moe_hybrid;
/// SCYTHE-2 WI-3: capacity-calibrated sharded linears.
pub mod scythe2;
pub mod sparse_attention;
pub mod varbuilder;
/// Tiered embedding lookup with optional NVMe spill path (Issue 3 of scythe_fixes_and_ngram_spill_plan.md).
/// `SpillableEmbedding` is a drop-in wrapper around `Embedding` that adds a config-gated
/// NvMe spill path for large embedding tables. Inert (zero overhead) when the spill threshold
/// is not exceeded or no spill path is configured.
pub mod embedding_spill;

pub use modules::{
    ColumnParallelLinear, Conv1d, ConvTranspose1d, Embedding, ExpertParallelConfig, KdaAttention,
    KdaLayerCache, LayerCache, LayerNorm, Linear, LinearAttentionBlock, LinearAttentionLayerCache,
    MlaAttention, MlaKvCache, RmsNorm, Rope, RowParallelLinear, TensorParallelConfig, add_tensors,
    broadcast_bias, embedding_gather_on_device, is_kernel_unimplemented,
    pick_device_for_storage_device, pick_device_for_tensor, require_single_device, short_conv1d,
};

pub use moe_deterministic::{DeterministicTokenMap, ScoreboardSync};
pub use moe_hybrid::{HybridExecutor, PcieBench};
pub use scythe2::{Scythe2Linear, slice_input_dim, slice_output_dim};
pub use varbuilder::WeightSource;
pub use embedding_spill::SpillableEmbedding;
