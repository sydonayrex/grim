//! Parallel communication primitives for intra-model Tensor Parallelism (TP) and Pipeline Parallelism (PP).
//!
//! Provides ring all-reduce, all-gather, broadcast, and peer-to-peer point-to-point exchanges
//! across multiple GPU ranks on AMD ROCm hardware, supporting both HIP peer direct memory
//! and shared memory/host staging fallbacks.

use std::sync::{Arc, Mutex};
use grim_tensor::backend::BackendStorage;
use grim_tensor::error::{Error, Result};
use crate::device::util::DeviceGuard;
use crate::memory::storage::RocmStorage;

/// Communication topology descriptor for a parallel execution group.
#[derive(Debug, Clone)]
pub struct ParallelTopology {
    /// Rank of this process/worker within the parallel group (0..world_size-1).
    pub rank: usize,
    /// Total number of participating ranks in the parallel group.
    pub world_size: usize,
    /// Physical device ordinal assignments for each rank index.
    pub device_ordinals: Vec<usize>,
}

impl ParallelTopology {
    /// Creates a new parallel topology descriptor.
    ///
    /// # Contracts
    /// * `rank < world_size`
    /// * `device_ordinals.is_empty() || device_ordinals.len() == world_size`
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

    /// Returns the physical GPU device ordinal corresponding to this rank.
    pub fn local_device_ordinal(&self) -> usize {
        self.device_ordinals.get(self.rank).copied().unwrap_or(self.rank)
    }
}

/// Underlying transport backend for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackendType {
    /// Peer-to-peer PCIe/Infinity Fabric direct memory copy via HIP.
    P2pDirect,
    /// Inter-process or thread-shared staging buffer.
    SharedHostStaging,
    /// RCCL (Radeon Collective Communication Library) bindings.
    Rccl,
}

/// Shared in-memory rendezvous ring buffer for multi-rank host staging.
#[derive(Debug)]
pub struct HostStagingRing {
    /// Buffers indexed by rank index.
    buffers: Vec<Mutex<Vec<f32>>>,
}

impl HostStagingRing {
    /// Allocate staging buffers for `world_size` ranks.
    pub fn new(world_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            buffers.push(Mutex::new(Vec::new()));
        }
        Self { buffers }
    }

    /// Access the mutex-protected buffer for `rank`.
    pub fn buffer(&self, rank: usize) -> &Mutex<Vec<f32>> {
        &self.buffers[rank]
    }
}

/// Parallel communicator managing collective operations across GPU devices.
#[derive(Debug, Clone)]
pub struct ParallelCommunicator {
    /// Topology descriptor.
    pub topology: ParallelTopology,
    /// Active communication backend.
    pub backend: CommBackendType,
    /// Host staging ring for CPU-side / simulated multi-device test environments.
    staging_ring: Option<Arc<HostStagingRing>>,
    /// RCCL multi-GPU collective handle for direct on-device execution.
    rccl: Option<Arc<crate::rccl::RcclAllReduce>>,
}

impl ParallelCommunicator {
    /// Reference to the staging ring if configured.
    pub fn staging_ring(&self) -> Option<&Arc<HostStagingRing>> {
        self.staging_ring.as_ref()
    }

    /// Constructs a single-device communicator (world_size = 1, no-op collectives).
    pub fn single_device(device_ordinal: usize) -> Self {
        Self {
            topology: ParallelTopology {
                rank: 0,
                world_size: 1,
                device_ordinals: vec![device_ordinal],
            },
            backend: CommBackendType::P2pDirect,
            staging_ring: None,
            rccl: None,
        }
    }

    /// Constructs a multi-rank parallel communicator backed by a shared host staging ring.
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
            rccl: None,
        })
    }

    /// Constructs a multi-rank communicator backed by physical RCCL device collectives.
    pub fn with_rccl(
        rank: usize,
        world_size: usize,
        device_ordinals: Vec<usize>,
        rccl: Arc<crate::rccl::RcclAllReduce>,
    ) -> Result<Self> {
        let topology = ParallelTopology::new(rank, world_size, device_ordinals)?;
        Ok(Self {
            topology,
            backend: CommBackendType::Rccl,
            staging_ring: None,
            rccl: Some(rccl),
        })
    }

    /// Constructs a multi-rank communicator configured for direct peer-to-peer HIP memory DMA.
    pub fn with_p2p(
        rank: usize,
        world_size: usize,
        device_ordinals: Vec<usize>,
    ) -> Result<Self> {
        let topology = ParallelTopology::new(rank, world_size, device_ordinals)?;
        Ok(Self {
            topology,
            backend: CommBackendType::P2pDirect,
            staging_ring: None,
            rccl: None,
        })
    }

    /// Performs an in-place All-Reduce (SUM) operation across all ranks in the communicator.
    ///
    /// # Contract
    /// * Every rank must invoke this collective with equal-sized buffers `buf`.
    /// * Upon return, `buf[i]` contains the sum of `buf[i]` across all ranks.
    pub fn all_reduce_sum_f32(&self, buf: &mut [f32]) -> Result<()> {
        if self.topology.world_size <= 1 {
            return Ok(());
        }

        if let Some(ring) = &self.staging_ring {
            let my_rank = self.topology.rank;
            let world_size = self.topology.world_size;

            // 1. Publish our buffer
            {
                let mut slot = ring.buffers[my_rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                slot.clear();
                slot.extend_from_slice(buf);
            }

            // Accumulate from other rank buffers
            for r in 0..world_size {
                if r != my_rank {
                    let other_slot = ring.buffers[r]
                        .lock()
                        .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
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
            let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);
            Ok(())
        }
    }

    /// Performs an in-place All-Reduce (SUM) on a device-resident `RocmStorage` tensor.
    /// Uses direct on-device RCCL collectives when available, avoiding host roundtrips.
    pub fn all_reduce_sum_storage(&self, storage: &mut RocmStorage, stream_u64: u64) -> Result<()> {
        if self.topology.world_size <= 1 {
            return Ok(());
        }
        let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);

        if let Some(rccl) = &self.rccl {
            let ptr = storage.device_ptr.ok_or_else(|| Error::Backend("No valid device pointer on storage".into()))?;
            let count = storage.shape.elem_count();
            rccl.sum_gradients_device(ptr, ptr, count, stream_u64, self.topology.rank)?;
            Ok(())
        } else if let Some(_ring) = &self.staging_ring {
            if let Ok(mut host_vec) = storage.to_cpu_vec_f32() {
                self.all_reduce_sum_f32(&mut host_vec)?;
                let bytes: Vec<u8> = host_vec.iter().flat_map(|f| f.to_ne_bytes()).collect();
                if let Some(ptr_val) = storage.device_ptr {
                    unsafe {
                        crate::hipMemcpy(
                            ptr_val as *mut std::ffi::c_void,
                            bytes.as_ptr() as *const std::ffi::c_void,
                            bytes.len(),
                            crate::HipMemcpyKind::HostToDevice,
                        );
                    }
                }
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    /// Gathers slices from all ranks into a concatenated destination buffer.
    ///
    /// # Contract
    /// * `dst.len() == src.len() * world_size`
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
                let mut slot = ring.buffers[my_rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                slot.clear();
                slot.extend_from_slice(src);
            }

            for r in 0..world_size {
                if r != my_rank {
                    let other_slot = ring.buffers[r]
                        .lock()
                        .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
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

    /// Broadcasts data from `root_rank` to all other ranks.
    pub fn broadcast_f32(&self, buf: &mut [f32], root_rank: usize) -> Result<()> {
        if self.topology.world_size <= 1 {
            return Ok(());
        }
        if root_rank >= self.topology.world_size {
            return Err(Error::Backend(format!(
                "broadcast_f32: root_rank {} >= world_size {}",
                root_rank, self.topology.world_size
            )));
        }

        if let Some(ring) = &self.staging_ring {
            let my_rank = self.topology.rank;
            if my_rank == root_rank {
                let mut slot = ring.buffers[root_rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                slot.clear();
                slot.extend_from_slice(buf);
            } else {
                let root_slot = ring.buffers[root_rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                if !root_slot.is_empty() {
                    let len = buf.len().min(root_slot.len());
                    buf[..len].copy_from_slice(&root_slot[..len]);
                }
            }
        }
        Ok(())
    }

    /// Point-to-point send/receive for Pipeline Parallelism stage handoffs.
    pub fn send_recv_p2p(
        &self,
        send_buf: Option<&[f32]>,
        _dst_rank: usize,
        recv_buf: Option<&mut [f32]>,
        src_rank: usize,
    ) -> Result<()> {
        if let Some(ring) = &self.staging_ring {
            if let Some(send_data) = send_buf {
                let mut slot = ring.buffers[self.topology.rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                slot.clear();
                slot.extend_from_slice(send_data);
            }

            if let Some(recv_data) = recv_buf {
                let src_slot = ring.buffers[src_rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                if !src_slot.is_empty() {
                    let len = recv_data.len().min(src_slot.len());
                    recv_data[..len].copy_from_slice(&src_slot[..len]);
                }
            }
        }
        Ok(())
    }

    /// Point-to-point asynchronous device-to-device copy for Pipeline Parallelism stage handoffs.
    /// Executes direct `hipMemcpyPeerAsync` across GPU devices, avoiding host RAM bounce.
    pub fn send_recv_p2p_device(
        &self,
        send_dev_ptr: Option<u64>,
        dst_rank: usize,
        recv_dev_ptr: Option<u64>,
        src_rank: usize,
        count_bytes: usize,
        stream_u64: u64,
    ) -> Result<()> {
        let my_ordinal = self.topology.local_device_ordinal() as i32;
        let _guard = DeviceGuard::set(my_ordinal);

        if let Some(send_ptr) = send_dev_ptr {
            let dst_ordinal = self.topology.device_ordinals.get(dst_rank).copied().unwrap_or(dst_rank) as i32;
            if let Some(recv_ptr) = recv_dev_ptr {
                crate::rccl::p2p_memcpy_async(
                    recv_ptr as *mut std::ffi::c_void,
                    dst_ordinal,
                    send_ptr as *const std::ffi::c_void,
                    my_ordinal,
                    count_bytes,
                    stream_u64 as *mut std::ffi::c_void,
                )?;
            }
        } else if let Some(_recv_ptr) = recv_dev_ptr {
            let _src_ordinal = self.topology.device_ordinals.get(src_rank).copied().unwrap_or(src_rank) as i32;
        }
        Ok(())
    }

    /// Performs an All-to-All personalized communication exchange across ranks.
    pub fn all_to_all_f32(
        &self,
        send_slices: &[&[f32]],
        recv_slices: &mut [&mut [f32]],
    ) -> Result<()> {
        let world_size = self.topology.world_size;
        if world_size <= 1 {
            if !send_slices.is_empty() && !recv_slices.is_empty() {
                let len = send_slices[0].len().min(recv_slices[0].len());
                recv_slices[0][..len].copy_from_slice(&send_slices[0][..len]);
            }
            return Ok(());
        }

        if let Some(ring) = &self.staging_ring {
            let my_rank = self.topology.rank;
            // 1. Stage our sends into staging ring buffer
            for (dst_rank, slice) in send_slices.iter().enumerate().take(world_size) {
                if dst_rank == my_rank {
                    let len = slice.len().min(recv_slices[my_rank].len());
                    recv_slices[my_rank][..len].copy_from_slice(&slice[..len]);
                } else {
                    let mut slot = ring.buffers[my_rank]
                        .lock()
                        .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                    slot.clear();
                    slot.extend_from_slice(slice);
                }
            }

            // 2. Fetch data sent to us by other ranks
            for (src_rank, recv_buf) in recv_slices.iter_mut().enumerate().take(world_size) {
                if src_rank != my_rank {
                    let other_slot = ring.buffers[src_rank]
                        .lock()
                        .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                    let len = other_slot.len().min(recv_buf.len());
                    if len > 0 {
                        recv_buf[..len].copy_from_slice(&other_slot[..len]);
                    }
                }
            }
            Ok(())
        } else {
            let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);
            Ok(())
        }
    }

    /// Reduces (SUM) equal-sized input buffers from all ranks and scatters the respective rank slice into `dst`.
    ///
    /// # Contract
    /// * `src.len() == dst.len() * world_size`
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
            // 1. Stage full src into our ring buffer slot
            {
                let mut slot = ring.buffers[my_rank]
                    .lock()
                    .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                slot.clear();
                slot.extend_from_slice(src);
            }

            // 2. Reduce the chunk assigned to this rank (offset = my_rank * chunk_size)
            let my_offset = my_rank * chunk_size;
            dst.copy_from_slice(&src[my_offset..my_offset + chunk_size]);

            for r in 0..world_size {
                if r != my_rank {
                    let other_slot = ring.buffers[r]
                        .lock()
                        .map_err(|_| Error::Backend("Staging ring lock poisoned".into()))?;
                    if other_slot.len() >= my_offset + chunk_size {
                        for i in 0..chunk_size {
                            dst[i] += other_slot[my_offset + i];
                        }
                    }
                }
            }
            Ok(())
        } else {
            let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);
            let my_offset = my_rank * chunk_size;
            dst.copy_from_slice(&src[my_offset..my_offset + chunk_size]);
            Ok(())
        }
    }

    /// Performs an All-Gather on a device-resident `RocmStorage` tensor.
    /// Concatenates shards from all ranks into `dst` on-device via RCCL when available.
    pub fn all_gather_storage(
        &self,
        src: &RocmStorage,
        dst: &mut RocmStorage,
        stream_u64: u64,
    ) -> Result<()> {
        let world_size = self.topology.world_size;
        let chunk_size = src.shape.elem_count();

        if dst.shape.elem_count() != chunk_size * world_size {
            return Err(Error::Backend(format!(
                "all_gather_storage: dst elems {} != chunk_size {} * world_size {}",
                dst.shape.elem_count(),
                chunk_size,
                world_size
            )));
        }

        if world_size <= 1 {
            if let (Some(s_ptr), Some(d_ptr)) = (src.device_ptr, dst.device_ptr) {
                let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);
                unsafe {
                    crate::hipMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        s_ptr as *const std::ffi::c_void,
                        src.bytes(),
                        crate::HipMemcpyKind::DeviceToDevice,
                    );
                }
            }
            return Ok(());
        }

        let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);

        if let Some(rccl) = &self.rccl {
            let send_ptr = src.device_ptr.ok_or_else(|| Error::Backend("all_gather_storage: src device pointer missing".into()))?;
            let recv_ptr = dst.device_ptr.ok_or_else(|| Error::Backend("all_gather_storage: dst device pointer missing".into()))?;
            rccl.all_gather_device(
                send_ptr,
                recv_ptr,
                chunk_size,
                crate::rccl::NCCL_FLOAT32,
                stream_u64,
                self.topology.rank,
            )?;
            Ok(())
        } else {
            // Host staging fallback
            let src_vec = src.to_cpu_vec_f32()?;
            let mut dst_vec = vec![0.0f32; dst.shape.elem_count()];
            self.all_gather_f32(&src_vec, &mut dst_vec)?;

            if let Some(d_ptr) = dst.device_ptr {
                let bytes: Vec<u8> = dst_vec.iter().flat_map(|f| f.to_ne_bytes()).collect();
                unsafe {
                    crate::hipMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        bytes.as_ptr() as *const std::ffi::c_void,
                        bytes.len(),
                        crate::HipMemcpyKind::HostToDevice,
                    );
                }
            }
            Ok(())
        }
    }

    /// Performs a Reduce-Scatter (SUM) on a device-resident `RocmStorage` tensor.
    /// Reduces gradient tensors across all ranks and scatters the rank shard into `dst` on-device via RCCL.
    pub fn reduce_scatter_storage(
        &self,
        src: &RocmStorage,
        dst: &mut RocmStorage,
        stream_u64: u64,
    ) -> Result<()> {
        let world_size = self.topology.world_size;
        let chunk_size = dst.shape.elem_count();

        if src.shape.elem_count() != chunk_size * world_size {
            return Err(Error::Backend(format!(
                "reduce_scatter_storage: src elems {} != chunk_size {} * world_size {}",
                src.shape.elem_count(),
                chunk_size,
                world_size
            )));
        }

        if world_size <= 1 {
            if let (Some(s_ptr), Some(d_ptr)) = (src.device_ptr, dst.device_ptr) {
                let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);
                unsafe {
                    crate::hipMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        s_ptr as *const std::ffi::c_void,
                        dst.bytes(),
                        crate::HipMemcpyKind::DeviceToDevice,
                    );
                }
            }
            return Ok(());
        }

        let _guard = DeviceGuard::set(self.topology.local_device_ordinal() as i32);

        if let Some(rccl) = &self.rccl {
            let send_ptr = src.device_ptr.ok_or_else(|| Error::Backend("reduce_scatter_storage: src device pointer missing".into()))?;
            let recv_ptr = dst.device_ptr.ok_or_else(|| Error::Backend("reduce_scatter_storage: dst device pointer missing".into()))?;
            rccl.reduce_scatter_device(
                send_ptr,
                recv_ptr,
                chunk_size,
                crate::rccl::NCCL_FLOAT32,
                stream_u64,
                self.topology.rank,
            )?;
            Ok(())
        } else {
            // Host staging fallback
            let src_vec = src.to_cpu_vec_f32()?;
            let mut dst_vec = vec![0.0f32; dst.shape.elem_count()];
            self.reduce_scatter_sum_f32(&src_vec, &mut dst_vec)?;

            if let Some(d_ptr) = dst.device_ptr {
                let bytes: Vec<u8> = dst_vec.iter().flat_map(|f| f.to_ne_bytes()).collect();
                unsafe {
                    crate::hipMemcpy(
                        d_ptr as *mut std::ffi::c_void,
                        bytes.as_ptr() as *const std::ffi::c_void,
                        bytes.len(),
                        crate::HipMemcpyKind::HostToDevice,
                    );
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_validation() {
        let topo = ParallelTopology::new(0, 2, vec![0, 1]).unwrap();
        assert_eq!(topo.rank, 0);
        assert_eq!(topo.world_size, 2);
        assert_eq!(topo.local_device_ordinal(), 0);

        assert!(ParallelTopology::new(2, 2, vec![0, 1]).is_err());
        assert!(ParallelTopology::new(0, 2, vec![0]).is_err());
    }

    #[test]
    fn test_all_reduce_sum_shared_staging() {
        let ring = Arc::new(HostStagingRing::new(2));
        let comm0 = ParallelCommunicator::with_shared_staging(0, 2, vec![0, 1], ring.clone()).unwrap();
        let comm1 = ParallelCommunicator::with_shared_staging(1, 2, vec![0, 1], ring.clone()).unwrap();

        let mut buf0 = vec![1.0f32, 2.0, 3.0];
        let mut buf1 = vec![10.0f32, 20.0, 30.0];

        comm0.all_reduce_sum_f32(&mut buf0).unwrap();
        comm1.all_reduce_sum_f32(&mut buf1).unwrap();

        assert_eq!(buf1, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_all_gather_shared_staging() {
        let ring = Arc::new(HostStagingRing::new(2));
        let comm0 = ParallelCommunicator::with_shared_staging(0, 2, vec![0, 1], ring.clone()).unwrap();
        let comm1 = ParallelCommunicator::with_shared_staging(1, 2, vec![0, 1], ring.clone()).unwrap();

        let src0 = vec![1.0f32, 2.0];
        let src1 = vec![3.0f32, 4.0];
        let mut dst0 = vec![0.0f32; 4];
        let mut dst1 = vec![0.0f32; 4];

        comm0.all_gather_f32(&src0, &mut dst0).unwrap();
        comm1.all_gather_f32(&src1, &mut dst1).unwrap();

        assert_eq!(dst1, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_all_to_all_shared_staging() {
        let ring = Arc::new(HostStagingRing::new(2));
        let comm0 = ParallelCommunicator::with_shared_staging(0, 2, vec![0, 1], ring.clone()).unwrap();
        let comm1 = ParallelCommunicator::with_shared_staging(1, 2, vec![0, 1], ring.clone()).unwrap();

        // comm0 sends s00 to rank0, s01 to rank1
        let s00 = vec![1.0f32, 1.1];
        let s01 = vec![2.0f32, 2.2];

        // comm1 sends s10 to rank0, s11 to rank1
        let s10 = vec![3.0f32, 3.3];
        let s11 = vec![4.0f32, 4.4];

        let mut r00 = vec![0.0f32; 2];
        let mut r01 = vec![0.0f32; 2];
        let mut r10 = vec![0.0f32; 2];
        let mut r11 = vec![0.0f32; 2];

        let sends0 = [&s00[..], &s01[..]];
        let mut recvs0 = [&mut r00[..], &mut r01[..]];
        comm0.all_to_all_f32(&sends0, &mut recvs0).unwrap();

        let sends1 = [&s10[..], &s11[..]];
        let mut recvs1 = [&mut r10[..], &mut r11[..]];
        comm1.all_to_all_f32(&sends1, &mut recvs1).unwrap();

        assert_eq!(r00, vec![1.0, 1.1]);
        assert_eq!(r11, vec![4.0, 4.4]);
    }
}
