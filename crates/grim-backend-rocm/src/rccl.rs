//! Wrappers and FFI bindings for system RCCL (NCCL) collectives (WI-R1, WI-R3).

use grim_tensor::DType;
use grim_tensor::error::{Error, Result};
use std::ffi::{c_char, c_void};

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct NcclComm(pub *mut c_void);
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    pub internal: [c_char; 128],
}

pub type NcclResult = i32;
pub const NCCL_SUCCESS: NcclResult = 0;

pub type NcclDataType = i32;
pub const NCCL_FLOAT16: NcclDataType = 6;
pub const NCCL_FLOAT32: NcclDataType = 7;

pub type NcclRedOp = i32;
pub const NCCL_SUM: NcclRedOp = 0;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CollectiveConfig {
    pub enabled: bool,
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CommComputeOverlapConfig {
    pub enabled: bool,
}

impl Default for CommComputeOverlapConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[cfg(feature = "rccl")]
#[link(name = "rccl", kind = "dylib")]
unsafe extern "C" {
    pub fn ncclGetUniqueId(id: *mut NcclUniqueId) -> NcclResult;
    pub fn ncclCommInitRank(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
    ) -> NcclResult;
    pub fn ncclCommInitAll(comms: *mut NcclComm, ndev: i32, devlist: *const i32) -> NcclResult;
    pub fn ncclCommDestroy(comm: NcclComm) -> NcclResult;
    pub fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult;
    pub fn ncclReduceScatter(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult;
    pub fn ncclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult;
    pub fn ncclGroupStart() -> NcclResult;
    pub fn ncclGroupEnd() -> NcclResult;
    pub fn ncclSend(
        sendbuff: *const c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult;
    pub fn ncclRecv(
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult;

    // FFI for P2P copy
    pub fn hipMemcpyPeerAsync(
        dst: *mut c_void,
        dstDevice: i32,
        src: *const c_void,
        srcDevice: i32,
        count: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(not(feature = "rccl"))]
unsafe extern "C" {}

/// Unique identifier for establishing communication groups.
pub struct UniqueId(pub NcclUniqueId);

impl UniqueId {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "rccl")]
        unsafe {
            let mut id = NcclUniqueId { internal: [0; 128] };
            let res = ncclGetUniqueId(&mut id);
            if res == NCCL_SUCCESS {
                Ok(UniqueId(id))
            } else {
                Err(Error::Backend(format!(
                    "ncclGetUniqueId failed with status {}",
                    res
                )))
            }
        }
        #[cfg(not(feature = "rccl"))]
        {
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }
}

/// A wrapper around `NcclComm` managing the lifetime of a communicator.
pub struct RocmComm {
    comm: NcclComm,
}

impl RocmComm {
    pub fn new(nranks: i32, id: UniqueId, rank: i32) -> Result<Self> {
        #[cfg(feature = "rccl")]
        unsafe {
            let mut comm = NcclComm(std::ptr::null_mut());
            let res = ncclCommInitRank(&mut comm, nranks, id.0, rank);
            if res == NCCL_SUCCESS {
                Ok(RocmComm { comm })
            } else {
                Err(Error::Backend(format!(
                    "ncclCommInitRank failed with status {}",
                    res
                )))
            }
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (nranks, id, rank);
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }

    pub fn raw_comm(&self) -> NcclComm {
        self.comm
    }

    pub fn all_reduce(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        count: usize,
        dtype: &DType,
        stream: *mut c_void,
    ) -> Result<()> {
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => {
                    return Err(Error::Backend(format!(
                        "Unsupported RCCL dtype {:?}",
                        dtype
                    )));
                }
            };
            let res = ncclAllReduce(send, recv, count, nccl_dtype, NCCL_SUM, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclAllReduce failed with status {}",
                    res
                )))
            }
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send, recv, count, dtype, stream);
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }

    pub fn reduce_scatter(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        recv_count: usize,
        dtype: &DType,
        stream: *mut c_void,
    ) -> Result<()> {
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => {
                    return Err(Error::Backend(format!(
                        "Unsupported RCCL dtype {:?}",
                        dtype
                    )));
                }
            };
            let res = ncclReduceScatter(
                send, recv, recv_count, nccl_dtype, NCCL_SUM, self.comm, stream,
            );
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclReduceScatter failed with status {}",
                    res
                )))
            }
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send, recv, recv_count, dtype, stream);
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }

    pub fn all_gather(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        send_count: usize,
        dtype: &DType,
        stream: *mut c_void,
    ) -> Result<()> {
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => {
                    return Err(Error::Backend(format!(
                        "Unsupported RCCL dtype {:?}",
                        dtype
                    )));
                }
            };
            let res = ncclAllGather(send, recv, send_count, nccl_dtype, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclAllGather failed with status {}",
                    res
                )))
            }
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send, recv, send_count, dtype, stream);
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }

    pub fn fuse_reduce_scatter(
        &self,
        send_buffs: &[(*const c_void, i32)],
        recv_buff: *mut c_void,
        recv_count: usize,
        dtype: &DType,
        stream: *mut c_void,
    ) -> Result<()> {
        if send_buffs.is_empty() {
            return Err(Error::Backend(
                "fuse_reduce_scatter: send_buffs list cannot be empty".into(),
            ));
        }
        if recv_buff.is_null() {
            return Err(Error::Backend(
                "fuse_reduce_scatter: recv_buff cannot be null".into(),
            ));
        }
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => {
                    return Err(Error::Backend(format!(
                        "Unsupported RCCL dtype {:?}",
                        dtype
                    )));
                }
            };
            if send_buffs.len() == 1 {
                let local_send = send_buffs[0].0;
                if local_send.is_null() {
                    return Err(Error::Backend(
                        "fuse_reduce_scatter: send buffer is null".into(),
                    ));
                }
                let res = ncclReduceScatter(
                    local_send, recv_buff, recv_count, nccl_dtype, NCCL_SUM, self.comm, stream,
                );
                if res != NCCL_SUCCESS {
                    return Err(Error::Backend(format!(
                        "ncclReduceScatter failed with status {}",
                        res
                    )));
                }
            } else {
                let _ = ncclGroupStart();
                for &(send_ptr, _rank) in send_buffs {
                    if !send_ptr.is_null() {
                        let _ = ncclReduceScatter(
                            send_ptr, recv_buff, recv_count, nccl_dtype, NCCL_SUM, self.comm,
                            stream,
                        );
                    }
                }
                let res = ncclGroupEnd();
                if res != NCCL_SUCCESS {
                    return Err(Error::Backend(format!(
                        "ncclReduceScatter group failed with status {}",
                        res
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send_buffs, recv_buff, recv_count, dtype, stream);
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }

    pub fn fuse_all_gather(
        &self,
        send_buff: *const c_void,
        recv_buffs: &[(*mut c_void, i32)],
        send_count: usize,
        dtype: &DType,
        stream: *mut c_void,
    ) -> Result<()> {
        if recv_buffs.is_empty() {
            return Err(Error::Backend(
                "fuse_all_gather: recv_buffs list cannot be empty".into(),
            ));
        }
        if send_buff.is_null() {
            return Err(Error::Backend(
                "fuse_all_gather: send_buff cannot be null".into(),
            ));
        }
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => {
                    return Err(Error::Backend(format!(
                        "Unsupported RCCL dtype {:?}",
                        dtype
                    )));
                }
            };
            if recv_buffs.len() == 1 {
                let local_recv = recv_buffs[0].0;
                if local_recv.is_null() {
                    return Err(Error::Backend(
                        "fuse_all_gather: recv buffer is null".into(),
                    ));
                }
                let res = ncclAllGather(
                    send_buff, local_recv, send_count, nccl_dtype, self.comm, stream,
                );
                if res != NCCL_SUCCESS {
                    return Err(Error::Backend(format!(
                        "ncclAllGather failed with status {}",
                        res
                    )));
                }
            } else {
                let _ = ncclGroupStart();
                for &(recv_ptr, _rank) in recv_buffs {
                    if !recv_ptr.is_null() {
                        let _ = ncclAllGather(
                            send_buff, recv_ptr, send_count, nccl_dtype, self.comm, stream,
                        );
                    }
                }
                let res = ncclGroupEnd();
                if res != NCCL_SUCCESS {
                    return Err(Error::Backend(format!(
                        "ncclAllGather group failed with status {}",
                        res
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send_buff, recv_buffs, send_count, dtype, stream);
            Err(Error::Backend("RCCL feature not enabled".into()))
        }
    }
}

impl Drop for RocmComm {
    fn drop(&mut self) {
        #[cfg(feature = "rccl")]
        unsafe {
            if !self.comm.0.is_null() {
                let _ = ncclCommDestroy(self.comm);
                self.comm.0 = std::ptr::null_mut();
            }
        }
    }
}

/// Asynchronous Peer-to-Peer copy wrapping hipMemcpyPeerAsync.
pub fn p2p_memcpy_async(
    dst: *mut c_void,
    dst_device: i32,
    src: *const c_void,
    src_device: i32,
    count: usize,
    stream: *mut c_void,
) -> Result<()> {
    #[cfg(feature = "rccl")]
    unsafe {
        // rust-ffi-grim §1.3: guard null pointers before the FFI call so
        // a bad caller gets a clean error instead of a HIP runtime abort.
        if dst.is_null() || src.is_null() {
            return Err(Error::Backend("hipMemcpyPeerAsync: null buffer".into()));
        }
        let res = hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
        if res == 0 {
            Ok(())
        } else {
            Err(Error::Backend(format!(
                "hipMemcpyPeerAsync failed with status {}",
                res
            )))
        }
    }
    #[cfg(not(feature = "rccl"))]
    {
        let _ = (dst, dst_device, src, src_device, count, stream);
        Err(Error::Backend("RCCL feature not enabled".into()))
    }
}

/// Tensor-parallel all-reduce hook for the serving path (P2-WI-2 / WI-R3).
///
/// This is the **single, canonical call site** for TP all-reduce so that:
/// 1. The serving path has one place to enable/disable/profile the collective.
/// 2. A future `CommComputeOverlapConfig` can intercept here for stream-overlap.
///
/// Delegates directly to `comm.all_reduce`; the thin wrapper exists to keep
/// call sites unaware of the `RocmComm` API details, and to serve as the
/// correct hook point for comm-compute overlap (P2-WI-2 Phase 2).
///
/// Returns `Err(Unsupported)` when the `rccl` feature is disabled so
/// single-GPU builds compile cleanly without `#[cfg]` at every call site.
#[allow(unused_variables)]
pub fn tp_all_reduce(
    comm: &RocmComm,
    buf: *mut std::ffi::c_void,
    count: usize,
    dtype: &DType,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    #[cfg(feature = "rccl")]
    {
        // Safety: buf must be a valid GPU device buffer for `count` elements of
        // the given dtype; stream must be a valid HIP stream. These invariants
        // are upheld by the caller (the serving scheduler that owns the buffer).
        comm.all_reduce(buf as *const std::ffi::c_void, buf, count, dtype, stream)
    }
    #[cfg(not(feature = "rccl"))]
    {
        Err(Error::Backend(
            "tp_all_reduce: RCCL feature not enabled; \
             build with --features rccl for multi-GPU TP"
                .into(),
        ))
    }
}

/// Multi-GPU data-parallel all-reduce for training gradients.
///
/// Wraps the RCCL `allReduce` collective across `num_gpus` devices.
/// When `num_gpus <= 1`, this is a no-op (gradients are not modified).
///
/// The struct owns an `NcclComm` handle initialised via `ncclCommInitAll`
/// so that `sum_gradients` can perform a real all-reduce before applying
/// the `1/num_gpus` averaging scale.
pub struct RcclAllReduce {
    /// Number of GPUs participating in the data-parallel group.
    pub num_gpus: u32,
    /// Underlying NCCL communicator. `None` when `num_gpus <= 1` or the
    /// `rccl` feature is disabled.
    #[allow(dead_code)] // read only inside #[cfg(feature = "rccl")]
    comm: Option<NcclComm>,
}

impl RcclAllReduce {
    /// Create a new RCCL all-reduce handle for `num_gpus` devices.
    ///
    /// When `num_gpus > 1` and the `rccl` feature is enabled this calls
    /// `ncclCommInitAll` to obtain a communicator over all devices.
    pub fn new(num_gpus: u32) -> Self {
        let comm = Self::init_comm(num_gpus);
        Self { num_gpus, comm }
    }

    #[cfg(feature = "rccl")]
    fn init_comm(num_gpus: u32) -> Option<NcclComm> {
        if num_gpus <= 1 {
            return None;
        }
        let ndev = num_gpus as i32;
        let devlist: Vec<i32> = (0..ndev).collect();
        let mut comm = NcclComm(std::ptr::null_mut());
        // SAFETY: devlist contains `ndev` valid device ordinals; comm is a
        // local with stable address for the call.
        let status = unsafe { ncclCommInitAll(&mut comm, ndev, devlist.as_ptr()) };
        if status != NCCL_SUCCESS {
            log::warn!(
                "RcclAllReduce::new: ncclCommInitAll failed (status {}); \
                 falling back to local-only gradient scaling",
                status,
            );
            return None;
        }
        Some(comm)
    }

    #[cfg(not(feature = "rccl"))]
    fn init_comm(_num_gpus: u32) -> Option<NcclComm> {
        None
    }

    /// Sum gradients across all GPUs using RCCL all-reduce on device memory.
    ///
    /// When `num_gpus <= 1`, this is a no-op. When multi-GPU with a valid
    /// communicator, `ncclAllReduce` performs an in-place sum across all
    /// devices directly in GPU memory, then each gradient element is divided
    /// by `num_gpus` to produce the averaged gradient.
    ///
    /// `send_dev_ptr` / `recv_dev_ptr` are raw HIP device pointers (`u64`)
    /// to `count` contiguous `f32` elements. Both may alias (in-place reduce).
    /// `stream` is a HIP stream handle (`u64`); pass `0` for the default stream.
    pub fn sum_gradients_device(
        &self,
        send_dev_ptr: u64,
        recv_dev_ptr: u64,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.num_gpus <= 1 || count == 0 {
            return Ok(());
        }
        #[cfg(feature = "rccl")]
        {
            let Some(comm) = self.comm else {
                log::warn!(
                    "RcclAllReduce::sum_gradients_device: no NCCL comm; \
                     skipping cross-GPU reduce"
                );
                return Ok(());
            };
            // SAFETY: send/recv must be valid device pointers for `count`
            // f32 elements; comm must be a valid NCCL communicator; stream
            // must be a valid HIP stream (0 = default). These invariants
            // are upheld by the caller (the training gradient sync path).
            let status = unsafe {
                ncclAllReduce(
                    send_dev_ptr as *const c_void,
                    recv_dev_ptr as *mut c_void,
                    count,
                    NCCL_FLOAT32,
                    NCCL_SUM,
                    comm,
                    stream as *mut c_void,
                )
            };
            if status != NCCL_SUCCESS {
                return Err(Error::Backend(format!(
                    "RcclAllReduce::sum_gradients_device: ncclAllReduce failed (status {})",
                    status,
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send_dev_ptr, recv_dev_ptr, count, stream);
            Err(Error::Backend(
                "RcclAllReduce::sum_gradients_device: multi-GPU RCCL \
                 all-reduce requires the `rccl` feature flag"
                    .into(),
            ))
        }
    }

    /// Average already-reduced gradients in-place by `1/num_gpus`.
    ///
    /// This operates on host memory and is useful when the caller has
    /// already performed the cross-GPU reduction via another path (e.g.
    /// `sum_gradients_device`) and just needs to scale the result.
    pub fn scale_gradients(&self, grads: &mut [f32]) -> Result<()> {
        if self.num_gpus <= 1 || grads.is_empty() {
            return Ok(());
        }
        let scale = 1.0 / self.num_gpus as f32;
        for g in grads.iter_mut() {
            *g *= scale;
        }
        Ok(())
    }
}

impl Drop for RcclAllReduce {
    fn drop(&mut self) {
        #[cfg(feature = "rccl")]
        if let Some(comm) = self.comm.take() {
            if !comm.0.is_null() {
                // Best-effort destroy; ignore status.
                unsafe {
                    let _ = ncclCommDestroy(comm);
                }
            }
        }
    }
}
