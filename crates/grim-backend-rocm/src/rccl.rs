//! Wrappers and FFI bindings for system RCCL (NCCL) collectives (WI-R1, WI-R3).

use std::ffi::{c_char, c_void};
use grim_tensor::error::{Error, Result};
use grim_tensor::DType;

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
    pub fn ncclCommInitRank(comm: *mut NcclComm, nranks: i32, id: NcclUniqueId, rank: i32) -> NcclResult;
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
                Err(Error::Backend(format!("ncclGetUniqueId failed with status {}", res)))
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
                Err(Error::Backend(format!("ncclCommInitRank failed with status {}", res)))
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
                _ => return Err(Error::Backend(format!("Unsupported RCCL dtype {:?}", dtype))),
            };
            let res = ncclAllReduce(send, recv, count, nccl_dtype, NCCL_SUM, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!("ncclAllReduce failed with status {}", res)))
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
                _ => return Err(Error::Backend(format!("Unsupported RCCL dtype {:?}", dtype))),
            };
            let res = ncclReduceScatter(send, recv, recv_count, nccl_dtype, NCCL_SUM, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!("ncclReduceScatter failed with status {}", res)))
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
                _ => return Err(Error::Backend(format!("Unsupported RCCL dtype {:?}", dtype))),
            };
            let res = ncclAllGather(send, recv, send_count, nccl_dtype, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!("ncclAllGather failed with status {}", res)))
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
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => return Err(Error::Backend(format!("Unsupported RCCL dtype {:?}", dtype))),
            };
            let local_send = send_buffs[0].0;
            let res = ncclReduceScatter(local_send, recv_buff, recv_count, nccl_dtype, NCCL_SUM, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!("ncclReduceScatter failed with status {}", res)))
            }
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
        #[cfg(feature = "rccl")]
        unsafe {
            let nccl_dtype = match dtype.arith {
                grim_tensor::ArithType::F16 | grim_tensor::ArithType::BF16 => NCCL_FLOAT16,
                grim_tensor::ArithType::F32 => NCCL_FLOAT32,
                _ => return Err(Error::Backend(format!("Unsupported RCCL dtype {:?}", dtype))),
            };
            let local_recv = recv_buffs[0].0;
            let res = ncclAllGather(send_buff, local_recv, send_count, nccl_dtype, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!("ncclAllGather failed with status {}", res)))
            }
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
            Err(Error::Backend(format!("hipMemcpyPeerAsync failed with status {}", res)))
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
             build with --features rccl for multi-GPU TP".into(),
        ))
    }
}
