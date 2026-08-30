//! Parallel communication primitives for CUDA intra-model Tensor Parallelism and FSDP.

use std::sync::{Arc, Mutex};
use grim_tensor::backend::BackendStorage;
use grim_tensor::error::{Error, Result};
use crate::memory::storage::CudaStorage;
use crate::nccl::CudaComm;

/// Communication topology descriptor for a parallel execution group.
#[derive(Debug, Clone)]
pub struct ParallelTopology {
    /// Rank of this worker within the parallel group (0..world_size-1).
    pub rank: usize,
    /// Total number of participating ranks.
    pub world_size: usize,
    /// Physical CUDA device ordinals assigned to each rank.
    pub device_ordinals: Vec<usize>,
}

impl ParallelTopology {
    pub fn new(rank: usize, world_size: usize, device_ordinals: Vec<usize>) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::Backend("ParallelTopology: world_size must be >= 1".into()));
        }
        if rank >= world_size {
            return Err(Error::Backend(format!(
                "ParallelTopology: rank {} >= world_size {}",
                rank, world_size
            )));
        }
        if !device_ordinals.is_empty() && device_ordinals.len() != world_size {
            return Err(Error::Backend(format!(
                "ParallelTopology: device_ordinals length {} != world_size {}",
                device_ordinals.len(),
                world_size
            )));
        }
        Ok(Self {
            rank,
            world_size,
            device_ordinals,
        })
    }

    pub fn local_device_ordinal(&self) -> usize {
        self.device_ordinals.get(self.rank).copied().unwrap_or(self.rank)
    }
}

/// Underlying transport backend for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackendType {
    /// Peer-to-peer PCIe/NVLink direct memory copy.
    P2pDirect,
    /// Inter-process or thread-shared staging buffer.
    SharedHostStaging,
    /// NCCL (NVIDIA Collective Communications Library).
    Nccl,
}

/// Shared rendezvous ring buffer for multi-rank host staging.
#[derive(Debug)]
pub struct HostStagingRing {
    buffers: Vec<Mutex<Vec<f32>>>,
}

impl HostStagingRing {
    pub fn new(world_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            buffers.push(Mutex::new(Vec::new()));
        }
        Self { buffers }
    }

    pub fn buffer(&self, rank: usize) -> &Mutex<Vec<f32>> {
        &self.buffers[rank]
    }
}

/// Parallel communicator managing collective operations across CUDA GPUs.
pub struct ParallelCommunicator {
    pub topology: ParallelTopology,
    pub backend: CommBackendType,
    staging_ring: Option<Arc<HostStagingRing>>,
    nccl_comm: Option<Arc<CudaComm>>,
}

impl ParallelCommunicator {
    pub fn with_shared_staging(
        rank: usize,
        world_size: usize,
        device_ordinals: Vec<usize>,
        staging_ring: Arc<HostStagingRing>,
    ) -> Result<Self> {
        let topology = ParallelTopology::new(rank, world_size, device_ordinals)?;
        Ok(Self {
            topology,
            backend: CommBackendType::SharedHostStaging,
            staging_ring: Some(staging_ring),
            nccl_comm: None,
        })
    }

    pub fn with_nccl(
        rank: usize,
        world_size: usize,
        device_ordinals: Vec<usize>,
        nccl_comm: Arc<CudaComm>,
    ) -> Result<Self> {
        let topology = ParallelTopology::new(rank, world_size, device_ordinals)?;
        Ok(Self {
            topology,
            backend: CommBackendType::Nccl,
            staging_ring: None,
            nccl_comm: Some(nccl_comm),
        })
    }

    pub fn all_reduce_sum_f32(&self, buf: &mut [f32]) -> Result<()> {
        if self.topology.world_size <= 1 {
            return Ok(());
        }

        if let Some(ring) = &self.staging_ring {
            let my_rank = self.topology.rank;
            let world_size = self.topology.world_size;

            {
                let mut slot = ring.buffers[my_rank].lock().unwrap_or_else(|e| e.into_inner());
                slot.clear();
                slot.extend_from_slice(buf);
            }

            for r in 0..world_size {
                if r != my_rank {
                    let other_slot = ring.buffers[r].lock().unwrap_or_else(|e| e.into_inner());
                    if !other_slot.is_empty() {
                        let len = buf.len().min(other_slot.len());
                        for i in 0..len {
                            buf[i] += other_slot[i];
                        }
                    }
                }
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn all_gather_f32(&self, src: &[f32], dst: &mut [f32]) -> Result<()> {
        let world_size = self.topology.world_size;
        if world_size <= 1 {
            if dst.len() != src.len() {
                return Err(Error::Backend(format!(
                    "all_gather_f32 dst len {} != src len {}",
                    dst.len(),
                    src.len()
                )));
            }
            dst.copy_from_slice(src);
            return Ok(());
        }

        let chunk_size = src.len();
        if dst.len() != chunk_size * world_size {
            return Err(Error::Backend(format!(
                "all_gather_f32: dst len {} != chunk_size {} * world_size {}",
                dst.len(),
                chunk_size,
                world_size
            )));
        }

        let my_rank = self.topology.rank;
        let my_offset = my_rank * chunk_size;
        dst[my_offset..my_offset + chunk_size].copy_from_slice(src);

        if let Some(ring) = &self.staging_ring {
            {
                let mut slot = ring.buffers[my_rank].lock().unwrap_or_else(|e| e.into_inner());
                slot.clear();
                slot.extend_from_slice(src);
            }

            for r in 0..world_size {
                if r != my_rank {
                    let other_slot = ring.buffers[r].lock().unwrap_or_else(|e| e.into_inner());
                    if !other_slot.is_empty() {
                        let offset = r * chunk_size;
                        let copy_len = chunk_size.min(other_slot.len());
                        dst[offset..offset + copy_len].copy_from_slice(&other_slot[..copy_len]);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn reduce_scatter_sum_f32(&self, src: &[f32], dst: &mut [f32]) -> Result<()> {
        let world_size = self.topology.world_size;
        let chunk_size = dst.len();

        if world_size <= 1 {
            if src.len() != chunk_size {
                return Err(Error::Backend(format!(
                    "reduce_scatter_sum_f32: src len {} != dst len {}",
                    src.len(),
                    chunk_size
                )));
            }
            dst.copy_from_slice(src);
            return Ok(());
        }

        if src.len() != chunk_size * world_size {
            return Err(Error::Backend(format!(
                "reduce_scatter_sum_f32: src len {} != chunk_size {} * world_size {}",
                src.len(),
                chunk_size,
                world_size
            )));
        }

        let my_rank = self.topology.rank;

        if let Some(ring) = &self.staging_ring {
            {
                let mut slot = ring.buffers[my_rank].lock().unwrap_or_else(|e| e.into_inner());
                slot.clear();
                slot.extend_from_slice(src);
            }

            let my_offset = my_rank * chunk_size;
            dst.copy_from_slice(&src[my_offset..my_offset + chunk_size]);

            for r in 0..world_size {
                if r != my_rank {
                    let other_slot = ring.buffers[r].lock().unwrap_or_else(|e| e.into_inner());
                    if other_slot.len() >= my_offset + chunk_size {
                        for i in 0..chunk_size {
                            dst[i] += other_slot[my_offset + i];
                        }
                    }
                }
            }
            Ok(())
        } else {
            let my_offset = my_rank * chunk_size;
            dst.copy_from_slice(&src[my_offset..my_offset + chunk_size]);
            Ok(())
        }
    }

    pub fn all_gather_storage(
        &self,
        local_shard: &CudaStorage,
        full_dst: &mut CudaStorage,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        if let Some(nccl) = &self.nccl_comm {
            if let (Some(s_ptr), Some(d_ptr)) = (local_shard.device_ptr(), full_dst.device_ptr()) {
                return nccl.all_gather(
                    s_ptr as *const std::ffi::c_void,
                    d_ptr as *mut std::ffi::c_void,
                    local_shard.shape_metadata().elem_count(),
                    &local_shard.dtype(),
                    stream,
                );
            }
        }
        let shard_cpu = local_shard.to_cpu_vec_f32()?;
        let mut full_cpu = vec![0.0f32; full_dst.shape_metadata().elem_count()];
        self.all_gather_f32(&shard_cpu, &mut full_cpu)?;
        let uploaded = CudaStorage::copy_from_host(
            &full_cpu,
            full_dst.shape_metadata(),
            full_dst.dtype(),
            self.topology.local_device_ordinal(),
        )?;
        *full_dst = uploaded;
        Ok(())
    }

    pub fn reduce_scatter_storage(
        &self,
        full_grad: &CudaStorage,
        sharded_dst: &mut CudaStorage,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        if let Some(nccl) = &self.nccl_comm {
            if let (Some(s_ptr), Some(d_ptr)) = (full_grad.device_ptr(), sharded_dst.device_ptr()) {
                return nccl.reduce_scatter(
                    s_ptr as *const std::ffi::c_void,
                    d_ptr as *mut std::ffi::c_void,
                    sharded_dst.shape_metadata().elem_count(),
                    &full_grad.dtype(),
                    stream,
                );
            }
        }
        let grad_cpu = full_grad.to_cpu_vec_f32()?;
        let mut shard_cpu = vec![0.0f32; sharded_dst.shape_metadata().elem_count()];
        self.reduce_scatter_sum_f32(&grad_cpu, &mut shard_cpu)?;
        let inv_w = 1.0f32 / (self.topology.world_size as f32);
        for v in shard_cpu.iter_mut() {
            *v *= inv_w;
        }
        let uploaded = CudaStorage::copy_from_host(
            &shard_cpu,
            sharded_dst.shape_metadata(),
            sharded_dst.dtype(),
            self.topology.local_device_ordinal(),
        )?;
        *sharded_dst = uploaded;
        Ok(())
    }
}
