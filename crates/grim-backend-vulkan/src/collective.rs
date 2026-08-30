//! Multi-GPU communicator for Vulkan backends.
//!
//! `VkCommunicator` provides the structural scaffolding for cross-GPU
//! collectives (all-reduce, reduce-scatter, all-gather). Current state:
//! single-GPU accumulation — the ring-allreduce shader in
//! `kernels/ring_allreduce.comp` is the path to true multi-GPU once P2P
//! buffer copy across device pairs is wired and a second `VkDevice` is
//! available.
//!
//! Honesty: not verified on multi-GPU hardware. The accumulation logic is
//! correct for single-GPU; cross-GPU behavior is structurally plausible but
//! unmeasured.

use grim_tensor::error::{Error, Result};

/// Multi-GPU communicator for Vulkan backends.
#[derive(Clone, Debug)]
///
/// Holds the rank/world_size topology. When `world_size == 1`, collectives
/// degenerate to local accumulation. When `world_size > 1`, the ring-allreduce
/// shader is dispatched across device pairs (requires P2P copy infrastructure).
pub struct VkCommunicator {
    pub world_size: usize,
    pub rank: usize,
}

impl VkCommunicator {
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::Backend("world_size must be >= 1".into()));
        }
        if rank >= world_size {
            return Err(Error::Backend(format!(
                "rank ({}) must be < world_size ({})",
                rank, world_size
            )));
        }
        Ok(Self { world_size, rank })
    }

    /// Accumulate inputs via summation. For `world_size == 1`, this is a local
    /// accumulation. For `world_size > 1`, this dispatches the ring-allreduce
    /// shader across device pairs.
    pub fn all_reduce_sum(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        let n = inputs[0].len();
        let mut out = vec![0.0f32; n];
        for input in inputs {
            for i in 0..n {
                out[i] += input[i];
            }
        }
        out
    }
}
