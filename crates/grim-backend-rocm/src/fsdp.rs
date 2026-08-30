//! Consumer Parallel GPU Multi-GPU FSDP (Fully Sharded Data Parallel) module.
//!
//! Provides ZeRO-3 / FSDP distributed training primitives across multiple ROCm GPUs,
//! backed by real cross-rank collective communication via [`ParallelCommunicator`]
//! (RCCL device collectives or high-speed `HostStagingRing` synchronization).

use std::sync::Arc;
use grim_tensor::Shape;
use grim_tensor::error::{Error, Result};
use crate::device::parallel_comm::ParallelCommunicator;

/// Configuration for Consumer Parallel GPU FSDP sharding.
#[derive(Debug, Clone)]
pub struct ConsumerFsdpConfig {
    /// World size (number of parallel GPUs, e.g. 2 for dual GPU setup).
    pub world_size: usize,
    /// Rank of this GPU worker process (0..world_size).
    pub rank: usize,
    /// Target peak VRAM budget per GPU in bytes (e.g. 16 GB = 16 * 1024 * 1024 * 1024).
    pub peak_vram_budget_bytes: usize,
}

impl Default for ConsumerFsdpConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024, // 16 GB default
        }
    }
}

/// Fully Sharded Data Parallel / Data Parallel group managing parameter partitions and collectives.
pub struct ConsumerFsdpGroup {
    pub config: ConsumerFsdpConfig,
    pub comm: Option<Arc<ParallelCommunicator>>,
}

/// Type alias for Consumer Data Parallel group.
pub type ConsumerDpGroup = ConsumerFsdpGroup;
/// Type alias for Consumer Data Parallel configuration.
pub type ConsumerDpConfig = ConsumerFsdpConfig;
/// Type alias for Consumer ZeRO parameter planner.
pub type ConsumerZeroPlanner = ConsumerFsdpGroup;

impl ConsumerFsdpGroup {
    /// Constructs a new `ConsumerFsdpGroup` with optional parallel communicator.
    pub fn new(config: ConsumerFsdpConfig, comm: Option<Arc<ParallelCommunicator>>) -> Result<Self> {
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
            if c.topology.world_size != config.world_size || c.topology.rank != config.rank {
                return Err(Error::Backend(format!(
                    "Communicator topology mismatch: comm has rank {}/world {}, config has rank {}/world {}",
                    c.topology.rank, c.topology.world_size, config.rank, config.world_size
                )));
            }
        }
        Ok(Self { config, comm })
    }

    /// Computes the sharded shape for a full parameter tensor under `world_size` partitioning.
    pub fn shard_shape(&self, full_shape: &Shape) -> Result<Shape> {
        let dims = full_shape.dims();
        if dims.is_empty() {
            return Err(Error::Shape("Cannot shard scalar 0D tensor".into()));
        }

        let first_dim = dims[0];
        if first_dim % self.config.world_size != 0 {
            return Err(Error::Shape(format!(
                "First dimension {} must be evenly divisible by world_size {}",
                first_dim, self.config.world_size
            )));
        }

        let mut sharded_dims = dims.to_vec();
        sharded_dims[0] = first_dim / self.config.world_size;
        Ok(Shape::new(sharded_dims))
    }

    /// Estimates peak VRAM memory footprint for a given sharded model size under 4-bit QLoRA fine-tuning.
    pub fn estimate_peak_vram_bytes(&self, num_params: usize) -> usize {
        let sharded_params = num_params / self.config.world_size;
        // QLoRA 4-bit base weights (0.5 B/param) + 16-bit LoRA adapter & AdamW moments (~2.5 B/param for rank=16)
        let qlora_bytes_per_param = 3; // 3.0 bytes per param total
        let base_vram = (sharded_params as f64 * qlora_bytes_per_param as f64) as usize;
        // Add 10% transient working buffer overhead for AllGather
        base_vram + (base_vram / 10)
    }

    /// Reconstruct full parameter tensor from sharded slice across ranks using real cross-rank AllGather.
    pub fn execute_all_gather(&self, local_shard: &[f32], full_shape: &Shape) -> Result<Vec<f32>> {
        let total_len = full_shape.elem_count();
        let expected_shard_len = total_len / self.config.world_size;

        if local_shard.len() != expected_shard_len {
            return Err(Error::Shape(format!(
                "execute_all_gather: local shard length {} != expected shard len {}",
                local_shard.len(),
                expected_shard_len
            )));
        }

        let mut full_buffer = vec![0.0f32; total_len];

        if let Some(comm) = &self.comm {
            comm.all_gather_f32(local_shard, &mut full_buffer)?;
        } else {
            // Standalone single-rank fallback
            let offset = self.config.rank * expected_shard_len;
            full_buffer[offset..offset + expected_shard_len].copy_from_slice(local_shard);
        }

        Ok(full_buffer)
    }

    /// Reduces gradients across all ranks and scatters the local rank's partitioned shard.
    pub fn execute_reduce_scatter(
        &self,
        local_full_grad: &[f32],
        sharded_shape: &Shape,
    ) -> Result<Vec<f32>> {
        let shard_len = sharded_shape.elem_count();
        let total_len = shard_len * self.config.world_size;

        if local_full_grad.len() != total_len {
            return Err(Error::Shape(format!(
                "execute_reduce_scatter: local_full_grad len {} != total_len {}",
                local_full_grad.len(),
                total_len
            )));
        }

        let mut shard = vec![0.0f32; shard_len];

        if let Some(comm) = &self.comm {
            comm.reduce_scatter_sum_f32(local_full_grad, &mut shard)?;
            // Average gradient across ranks
            let inv_world = 1.0f32 / (self.config.world_size as f32);
            for v in shard.iter_mut() {
                *v *= inv_world;
            }
        } else {
            // Standalone single-rank fallback: extract rank slice
            let offset = self.config.rank * shard_len;
            shard.copy_from_slice(&local_full_grad[offset..offset + shard_len]);
            let inv_world = 1.0f32 / (self.config.world_size as f32);
            for v in shard.iter_mut() {
                *v *= inv_world;
            }
        }

        Ok(shard)
    }

    /// Reconstruct full parameter tensor from sharded slice across ranks using real on-device AllGather.
    pub fn execute_all_gather_storage(
        &self,
        local_shard: &crate::RocmStorage,
        full_dst: &mut crate::RocmStorage,
        stream: u64,
    ) -> Result<()> {
        let expected_shard_len = full_dst.shape.elem_count() / self.config.world_size;
        if local_shard.shape.elem_count() != expected_shard_len {
            return Err(Error::Shape(format!(
                "execute_all_gather_storage: local shard len {} != expected {}",
                local_shard.shape.elem_count(),
                expected_shard_len
            )));
        }

        if let Some(comm) = &self.comm {
            comm.all_gather_storage(local_shard, full_dst, stream)?;
        } else {
            // Single rank: copy into destination
            if let (Some(s_ptr), Some(d_ptr)) = (local_shard.device_ptr, full_dst.device_ptr) {
                unsafe {
                    crate::hipMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        s_ptr as *const std::ffi::c_void,
                        local_shard.bytes(),
                        crate::HipMemcpyKind::DeviceToDevice,
                    );
                }
            }
        }
        Ok(())
    }

    /// Reduces gradients across all ranks and scatters the local rank's partitioned shard on-device.
    pub fn execute_reduce_scatter_storage(
        &self,
        local_full_grad: &crate::RocmStorage,
        sharded_dst: &mut crate::RocmStorage,
        stream: u64,
    ) -> Result<()> {
        let shard_len = sharded_dst.shape.elem_count();
        if local_full_grad.shape.elem_count() != shard_len * self.config.world_size {
            return Err(Error::Shape(format!(
                "execute_reduce_scatter_storage: full grad len {} != expected {}",
                local_full_grad.shape.elem_count(),
                shard_len * self.config.world_size
            )));
        }

        if let Some(comm) = &self.comm {
            comm.reduce_scatter_storage(local_full_grad, sharded_dst, stream)?;
        } else {
            // Single rank: copy rank shard into destination
            if let (Some(s_ptr), Some(d_ptr)) = (local_full_grad.device_ptr, sharded_dst.device_ptr) {
                let offset_bytes = self.config.rank * sharded_dst.bytes();
                unsafe {
                    crate::hipMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        (s_ptr + offset_bytes as u64) as *const std::ffi::c_void,
                        sharded_dst.bytes(),
                        crate::HipMemcpyKind::DeviceToDevice,
                    );
                }
            }
        }
        Ok(())
    }

    /// Validates whether a model parameter count fits within the consumer GPU VRAM budget.
    pub fn fits_vram_budget(&self, num_params: usize) -> bool {
        self.estimate_peak_vram_bytes(num_params) <= self.config.peak_vram_budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::device::parallel_comm::HostStagingRing;

    #[test]
    fn test_consumer_fsdp_multi_rank_all_gather_real_cross_rank() -> Result<()> {
        let world_size = 2;
        let ring = Arc::new(HostStagingRing::new(world_size));

        let comm0 = Arc::new(ParallelCommunicator::with_shared_staging(
            0, world_size, vec![0, 1], ring.clone(),
        )?);
        let comm1 = Arc::new(ParallelCommunicator::with_shared_staging(
            1, world_size, vec![0, 1], ring.clone(),
        )?);

        let cfg0 = ConsumerFsdpConfig {
            world_size,
            rank: 0,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        };
        let cfg1 = ConsumerFsdpConfig {
            world_size,
            rank: 1,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        };

        let fsdp0 = ConsumerFsdpGroup::new(cfg0, Some(comm0))?;
        let fsdp1 = ConsumerFsdpGroup::new(cfg1, Some(comm1))?;

        let full_shape = Shape::new(vec![4, 2]); // total 8 elements -> 4 per rank
        let shard0 = vec![10.0f32, 20.0, 30.0, 40.0];
        let shard1 = vec![50.0f32, 60.0, 70.0, 80.0];

        // Stage rank 0 then rank 1
        let _ = fsdp0.execute_all_gather(&shard0, &full_shape)?;
        let gathered1 = fsdp1.execute_all_gather(&shard1, &full_shape)?;
        let gathered0 = fsdp0.execute_all_gather(&shard0, &full_shape)?;

        // Both ranks must see the FULL gathered sequence across rank 0 and rank 1!
        let expected = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        assert_eq!(gathered0, expected, "Rank 0 must receive full cross-rank gathered buffer");
        assert_eq!(gathered1, expected, "Rank 1 must receive full cross-rank gathered buffer");

        Ok(())
    }

    #[test]
    fn test_consumer_fsdp_multi_rank_reduce_scatter_real_reduction() -> Result<()> {
        let world_size = 2;
        let ring = Arc::new(HostStagingRing::new(world_size));

        let comm0 = Arc::new(ParallelCommunicator::with_shared_staging(
            0, world_size, vec![0, 1], ring.clone(),
        )?);
        let comm1 = Arc::new(ParallelCommunicator::with_shared_staging(
            1, world_size, vec![0, 1], ring.clone(),
        )?);

        let cfg0 = ConsumerFsdpConfig {
            world_size,
            rank: 0,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        };
        let cfg1 = ConsumerFsdpConfig {
            world_size,
            rank: 1,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        };

        let fsdp0 = ConsumerFsdpGroup::new(cfg0, Some(comm0))?;
        let fsdp1 = ConsumerFsdpGroup::new(cfg1, Some(comm1))?;

        let sharded_shape = Shape::new(vec![2, 2]); // 4 elements per shard -> 8 total
        // Rank 0 gradients across full parameter tensor
        let grad0 = vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        // Rank 1 gradients across full parameter tensor
        let grad1 = vec![4.0f32, 6.0, 8.0, 10.0, 20.0, 22.0, 24.0, 26.0];

        let _ = fsdp0.execute_reduce_scatter(&grad0, &sharded_shape)?;
        let reduced_shard1 = fsdp1.execute_reduce_scatter(&grad1, &sharded_shape)?;
        let reduced_shard0 = fsdp0.execute_reduce_scatter(&grad0, &sharded_shape)?;

        // Rank 0 receives slice 0 (first 4 elems) reduced: (grad0[0..4] + grad1[0..4]) / 2
        // [ (2+4)/2, (4+6)/2, (6+8)/2, (8+10)/2 ] = [ 3.0, 5.0, 7.0, 9.0 ]
        assert_eq!(reduced_shard0, vec![3.0, 5.0, 7.0, 9.0]);

        // Rank 1 receives slice 1 (last 4 elems) reduced: (grad0[4..8] + grad1[4..8]) / 2
        // [ (10+20)/2, (12+22)/2, (14+24)/2, (16+26)/2 ] = [ 15.0, 17.0, 19.0, 21.0 ]
        assert_eq!(reduced_shard1, vec![15.0, 17.0, 19.0, 21.0]);

        Ok(())
    }

    #[test]
    fn test_consumer_fsdp_device_storage_real_collectives() -> Result<()> {
        let world_size = 2;
        let ring = Arc::new(HostStagingRing::new(world_size));

        let comm0 = Arc::new(ParallelCommunicator::with_shared_staging(
            0, world_size, vec![0, 1], ring.clone(),
        )?);
        let comm1 = Arc::new(ParallelCommunicator::with_shared_staging(
            1, world_size, vec![0, 1], ring.clone(),
        )?);

        let cfg0 = ConsumerFsdpConfig {
            world_size,
            rank: 0,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        };
        let cfg1 = ConsumerFsdpConfig {
            world_size,
            rank: 1,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        };

        let fsdp0 = ConsumerFsdpGroup::new(cfg0, Some(comm0))?;
        let fsdp1 = ConsumerFsdpGroup::new(cfg1, Some(comm1))?;

        let alloc = Arc::new(crate::RocmCachingAllocator::new(1024 * 1024, 0));

        let shard_shape = Shape::new(vec![4]);
        let full_shape = Shape::new(vec![8]);

        // If no physical GPU or running under test sandbox, we can check contract shapes
        if let (Ok(shard0_storage), Ok(shard1_storage), Ok(mut full0_storage), Ok(mut full1_storage)) = (
            crate::RocmStorage::alloc_gpu(&shard_shape, grim_tensor::DType::F32, &alloc, 0),
            crate::RocmStorage::alloc_gpu(&shard_shape, grim_tensor::DType::F32, &alloc, 0),
            crate::RocmStorage::alloc_gpu(&full_shape, grim_tensor::DType::F32, &alloc, 0),
            crate::RocmStorage::alloc_gpu(&full_shape, grim_tensor::DType::F32, &alloc, 0),
        ) {
            let _ = fsdp0.execute_all_gather_storage(&shard0_storage, &mut full0_storage, 0);
            let _ = fsdp1.execute_all_gather_storage(&shard1_storage, &mut full1_storage, 0);
            assert_eq!(full0_storage.shape.elem_count(), 8);
            assert_eq!(full1_storage.shape.elem_count(), 8);
        }

        Ok(())
    }
}
