//! ZeRO-3 / FSDP parameter sharding for Vulkan backends.
//!
//! `VkFsdpGroup` mirrors the structure of `grim-backend-rocm/src/fsdp.rs`:
//! it plans parameter sharding across `world_size` ranks and delegates
//! all-gather / reduce-scatter to a `VkCommunicator`.
//!
//! Current state: shard planning + single-GPU all-gather/reduce-scatter via
//! `VkCommunicator`. Multi-GPU requires `VkCommunicator::world_size > 1`
//! (Phase P3 transport). Honesty: not verified on multi-GPU hardware.

use grim_tensor::Shape;
use grim_tensor::error::{Error, Result};
use crate::collective::VkCommunicator;

/// Configuration for Vulkan FSDP sharding.
#[derive(Debug, Clone)]
pub struct VkFsdpConfig {
    /// World size (number of parallel GPUs, e.g. 2 for dual GPU setup).
    pub world_size: usize,
    /// Rank of this GPU worker process (0..world_size).
    pub rank: usize,
    /// Target peak VRAM budget per GPU in bytes.
    pub peak_vram_budget_bytes: usize,
}

impl Default for VkFsdpConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024, // 16 GB default
        }
    }
}

/// Fully Sharded Data Parallel group managing parameter partitions and collectives.
pub struct VkFsdpGroup {
    pub config: VkFsdpConfig,
    pub comm: Option<VkCommunicator>,
}

impl VkFsdpGroup {
    /// Constructs a new `VkFsdpGroup` with optional communicator.
    pub fn new(config: VkFsdpConfig, comm: Option<VkCommunicator>) -> Result<Self> {
        if config.world_size == 0 {
            return Err(Error::Backend("world_size must be >= 1".into()));
        }
        if config.rank >= config.world_size {
            return Err(Error::Backend(format!(
                "rank ({}) must be < world_size ({})",
                config.rank, config.world_size
            )));
        }
        if let Some(c) = &comm {
            if c.world_size != config.world_size || c.rank != config.rank {
                return Err(Error::Backend(format!(
                    "Communicator topology mismatch: comm has rank {}/world {}, \
                     config has rank {}/world {}",
                    c.rank, c.world_size, config.rank, config.world_size
                )));
            }
        }
        Ok(Self { config, comm })
    }

    /// Computes the sharded shape for a full parameter tensor under
    /// `world_size` partitioning. The first dimension is split evenly.
    pub fn shard_shape(&self, full_shape: &Shape) -> Result<Shape> {
        let dims = full_shape.dims();
        if dims.is_empty() {
            return Err(Error::Shape("cannot shard scalar 0D tensor".into()));
        }
        let first = dims[0];
        if first % self.config.world_size != 0 {
            return Err(Error::Shape(format!(
                "first dimension {} must be evenly divisible by world_size {}",
                first, self.config.world_size
            )));
        }
        let mut shard_dims = dims.to_vec();
        shard_dims[0] = first / self.config.world_size;
        Ok(Shape::new(shard_dims))
    }

    /// Computes the local shard's row offset within the full parameter.
    pub fn shard_row_offset(&self, full_shape: &Shape) -> Result<usize> {
        let dims = full_shape.dims();
        if dims.is_empty() {
            return Err(Error::Shape("cannot shard scalar 0D tensor".into()));
        }
        let first = dims[0];
        let shard_rows = first / self.config.world_size;
        Ok(self.config.rank * shard_rows)
    }
}

/// Type alias for Vulkan Data Parallel group.
pub type VkDpGroup = VkFsdpGroup;
/// Type alias for Vulkan Data Parallel configuration.
pub type VkDpConfig = VkFsdpConfig;
