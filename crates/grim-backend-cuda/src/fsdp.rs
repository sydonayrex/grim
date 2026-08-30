//! Multi-GPU FSDP (Fully Sharded Data Parallel) module for CUDA.
//!
//! Provides ZeRO-3 / FSDP distributed training primitives across multiple CUDA GPUs,
//! backed by real cross-rank collective communication via [`ParallelCommunicator`]
//! (NCCL device collectives or high-speed `HostStagingRing` synchronization).

use std::sync::Arc;
use grim_tensor::Shape;
use grim_tensor::error::{Error, Result};
use crate::device::parallel_comm::ParallelCommunicator;
use crate::memory::storage::CudaStorage;

/// Configuration for CUDA Parallel GPU FSDP sharding.
#[derive(Debug, Clone)]
pub struct ConsumerFsdpConfig {
    /// World size (number of parallel GPUs).
    pub world_size: usize,
    /// Rank of this GPU worker process (0..world_size).
    pub rank: usize,
    /// Target peak VRAM budget per GPU in bytes.
    pub peak_vram_budget_bytes: usize,
}

impl Default for ConsumerFsdpConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            peak_vram_budget_bytes: 16 * 1024 * 1024 * 1024,
        }
    }
}

/// Fully Sharded Data Parallel / Data Parallel group managing parameter partitions and collectives.
pub struct ConsumerFsdpGroup {
    pub config: ConsumerFsdpConfig,
    pub comm: Option<Arc<ParallelCommunicator>>,
}

pub type ConsumerDpGroup = ConsumerFsdpGroup;
pub type ConsumerDpConfig = ConsumerFsdpConfig;
pub type ConsumerZeroPlanner = ConsumerFsdpGroup;

impl ConsumerFsdpGroup {
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
        let qlora_bytes_per_param = 3;
        let base_vram = (sharded_params as f64 * qlora_bytes_per_param as f64) as usize;
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
            let inv_world = 1.0f32 / (self.config.world_size as f32);
            for v in shard.iter_mut() {
                *v *= inv_world;
            }
        } else {
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
        local_shard: &CudaStorage,
        full_dst: &mut CudaStorage,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        let expected_shard_len = full_dst.shape_metadata().elem_count() / self.config.world_size;
        if local_shard.shape_metadata().elem_count() != expected_shard_len {
            return Err(Error::Shape(format!(
                "execute_all_gather_storage: local shard len {} != expected {}",
                local_shard.shape_metadata().elem_count(),
                expected_shard_len
            )));
        }

        if let Some(comm) = &self.comm {
            comm.all_gather_storage(local_shard, full_dst, stream)?;
        } else {
            if let (Some(s_ptr), Some(d_ptr)) = (local_shard.device_ptr(), full_dst.device_ptr()) {
                unsafe {
                    crate::device::handles::cudaMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        s_ptr as *const std::ffi::c_void,
                        local_shard.bytes(),
                        crate::device::handles::cudaMemcpyDeviceToDevice,
                    );
                }
            }
        }
        Ok(())
    }

    /// Reduces gradients across all ranks and scatters the local rank's partitioned shard on-device.
    pub fn execute_reduce_scatter_storage(
        &self,
        local_full_grad: &CudaStorage,
        sharded_dst: &mut CudaStorage,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        let shard_len = sharded_dst.shape_metadata().elem_count();
        if local_full_grad.shape_metadata().elem_count() != shard_len * self.config.world_size {
            return Err(Error::Shape(format!(
                "execute_reduce_scatter_storage: full grad len {} != expected {}",
                local_full_grad.shape_metadata().elem_count(),
                shard_len * self.config.world_size
            )));
        }

        if let Some(comm) = &self.comm {
            comm.reduce_scatter_storage(local_full_grad, sharded_dst, stream)?;
        } else {
            if let (Some(s_ptr), Some(d_ptr)) = (local_full_grad.device_ptr(), sharded_dst.device_ptr()) {
                let offset_bytes = self.config.rank * sharded_dst.bytes();
                unsafe {
                    crate::device::handles::cudaMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        (s_ptr + offset_bytes as u64) as *const std::ffi::c_void,
                        sharded_dst.bytes(),
                        crate::device::handles::cudaMemcpyDeviceToDevice,
                    );
                }
            }
        }
        Ok(())
    }

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
    fn test_cuda_fsdp_multi_rank_all_gather() -> Result<()> {
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

        let full_shape = Shape::new(vec![4, 2]);
        let shard0 = vec![10.0f32, 20.0, 30.0, 40.0];
        let shard1 = vec![50.0f32, 60.0, 70.0, 80.0];

        let _ = fsdp0.execute_all_gather(&shard0, &full_shape)?;
        let gathered1 = fsdp1.execute_all_gather(&shard1, &full_shape)?;
        let gathered0 = fsdp0.execute_all_gather(&shard0, &full_shape)?;

        let expected = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        assert_eq!(gathered0, expected);
        assert_eq!(gathered1, expected);

        Ok(())
    }

    #[test]
    fn test_cuda_fsdp_multi_rank_reduce_scatter() -> Result<()> {
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

        let sharded_shape = Shape::new(vec![2, 2]);
        let grad0 = vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let grad1 = vec![4.0f32, 6.0, 8.0, 10.0, 20.0, 22.0, 24.0, 26.0];

        let _ = fsdp0.execute_reduce_scatter(&grad0, &sharded_shape)?;
        let reduced_shard1 = fsdp1.execute_reduce_scatter(&grad1, &sharded_shape)?;
        let reduced_shard0 = fsdp0.execute_reduce_scatter(&grad0, &sharded_shape)?;

        assert_eq!(reduced_shard0, vec![3.0, 5.0, 7.0, 9.0]);
        assert_eq!(reduced_shard1, vec![15.0, 17.0, 19.0, 21.0]);

        Ok(())
    }
}
