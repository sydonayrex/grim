//! Wrappers and FFI bindings for system RCCL (NCCL) collectives (WI-R1, WI-R3).

use grim_tensor::DType;
use grim_tensor::error::{Error, Result};
use std::ffi::{c_char, c_void};

/// Opaque NCCL communicator handle.
///
/// # Safety
///
/// `NcclComm` wraps a raw NCCL communicator pointer that is valid process-wide
/// once initialized via `ncclCommInitRank` or `ncclCommInitAll`. Moving the
/// handle between threads (Send) is safe because NCCL communicators are
/// process-global resources. The handle is also Sync because NCCL collectives
/// (`ncclAllReduce`, etc.) are designed to be called from any thread that holds
/// a valid communicator — the library internally synchronizes access.
///
/// Current enforcement: all live call paths into this type pass through
/// `AppState.engine: Mutex<Engine>` in grim-server, so no concurrent access is
/// possible through the server's actual API today. Do NOT remove that lock or add
/// a second concurrent access path without verifying that the underlying NCCL
/// usage is thread-safe for the specific collective pattern being used.
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
/// `ncclBfloat16` — supported by RCCL shipped with ROCm 5.x+. Mapping BF16
/// buffers to `NCCL_FLOAT16` instead would reinterpret the bits and corrupt
/// every collective, so BF16 gets its real type (older RCCL builds return an
/// error status we surface instead of silently producing garbage).
pub const NCCL_BFLOAT16: NcclDataType = 9;

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
                grim_tensor::ArithType::F16 => NCCL_FLOAT16,
                grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
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
                grim_tensor::ArithType::F16 => NCCL_FLOAT16,
                grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
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
                grim_tensor::ArithType::F16 => NCCL_FLOAT16,
                grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
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
                grim_tensor::ArithType::F16 => NCCL_FLOAT16,
                grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
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
                grim_tensor::ArithType::F16 => NCCL_FLOAT16,
                grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
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
#[derive(Debug)]
pub struct RcclAllReduce {
    /// Number of GPUs participating in the data-parallel group.
    pub num_gpus: u32,
    /// Underlying NCCL communicators (one per GPU rank). Empty when `num_gpus <= 1`
    /// or the `rccl` feature is disabled.
    #[allow(dead_code)] // read only inside #[cfg(feature = "rccl")]
    comms: Vec<NcclComm>,
}

impl RcclAllReduce {
    /// Create a communicator over the explicitly selected device ordinals.
    ///
    /// The old constructor inferred ordinals as `0..num_gpus`, which is not a
    /// valid contract once rank selection can be non-contiguous. Multi-GPU
    /// callers must now provide the same ordinals used to construct replicas;
    /// initialization failure is returned instead of becoming a silent
    /// local-only training run.
    pub fn try_new(device_ordinals: &[usize]) -> Result<Self> {
        let num_gpus = device_ordinals.len() as u32;
        if num_gpus <= 1 {
            return Ok(Self {
                num_gpus,
                comms: Vec::new(),
            });
        }
        let comms = Self::init_comm(device_ordinals)?;
        Ok(Self { num_gpus, comms })
    }

    #[cfg(feature = "rccl")]
    fn init_comm(device_ordinals: &[usize]) -> Result<Vec<NcclComm>> {
        let ndev = device_ordinals.len() as i32;
        let devlist: Vec<i32> = device_ordinals
            .iter()
            .map(|&ordinal| ordinal as i32)
            .collect();
        let mut comms = vec![NcclComm(std::ptr::null_mut()); device_ordinals.len()];
        // SAFETY: devlist contains `ndev` valid device ordinals; comms has
        // space allocated for `ndev` communicators.
        let status = unsafe { ncclCommInitAll(comms.as_mut_ptr(), ndev, devlist.as_ptr()) };
        if status != NCCL_SUCCESS {
            return Err(Error::Backend(format!(
                "RCCL communicator initialization failed for ordinals {:?} (status {})",
                device_ordinals, status
            )));
        }
        Ok(comms)
    }

    #[cfg(not(feature = "rccl"))]
    fn init_comm(_device_ordinals: &[usize]) -> Result<Vec<NcclComm>> {
        Err(Error::Backend(
            "multi-GPU RCCL training requires the `rccl` feature flag".into(),
        ))
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
        rank: usize,
    ) -> Result<()> {
        if self.num_gpus <= 1 || count == 0 {
            return Ok(());
        }
        #[cfg(feature = "rccl")]
        {
            let comm = self
                .comms
                .get(rank)
                .copied()
                .unwrap_or(NcclComm(std::ptr::null_mut()));
            if comm.0.is_null() {
                return Err(Error::Backend(
                    "RCCL communicator is unavailable for a multi-GPU reduction".into(),
                ));
            }
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
            let _ = (send_dev_ptr, recv_dev_ptr, count, stream, rank);
            Err(Error::Backend(
                "RcclAllReduce::sum_gradients_device: multi-GPU RCCL \
                 all-reduce requires the `rccl` feature flag"
                    .into(),
            ))
        }
    }

    /// All-reduce device buffers of an explicit NCCL dtype on `rank`'s
    /// communicator (used by the TP activation path for F16/BF16 tensors).
    /// See `sum_gradients_device` for the F32 gradient special case.
    pub fn all_reduce_device(
        &self,
        send_dev_ptr: u64,
        recv_dev_ptr: u64,
        count: usize,
        nccl_dtype: NcclDataType,
        stream: u64,
        rank: usize,
    ) -> Result<()> {
        if self.num_gpus <= 1 || count == 0 {
            return Ok(());
        }
        #[cfg(feature = "rccl")]
        {
            let comm = self
                .comms
                .get(rank)
                .copied()
                .unwrap_or(NcclComm(std::ptr::null_mut()));
            if comm.0.is_null() {
                return Err(Error::Backend(
                    "RCCL communicator is unavailable for a multi-GPU reduction".into(),
                ));
            }
            // SAFETY: send/recv must be valid device pointers for `count`
            // elements of `nccl_dtype`; comm and stream are owned by the
            // caller's device/rank pairing.
            let status = unsafe {
                ncclAllReduce(
                    send_dev_ptr as *const c_void,
                    recv_dev_ptr as *mut c_void,
                    count,
                    nccl_dtype,
                    NCCL_SUM,
                    comm,
                    stream as *mut c_void,
                )
            };
            if status != NCCL_SUCCESS {
                return Err(Error::Backend(format!(
                    "RcclAllReduce::all_reduce_device: ncclAllReduce failed (status {})",
                    status,
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (send_dev_ptr, recv_dev_ptr, count, nccl_dtype, stream, rank);
            Err(Error::Backend(
                "RcclAllReduce::all_reduce_device: multi-GPU RCCL \
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
        for comm in self.comms.drain(..) {
            if !comm.0.is_null() {
                // Best-effort destroy; ignore status.
                unsafe {
                    let _ = ncclCommDestroy(comm);
                }
            }
        }
    }
}

/// Multi-node RCCL communicator group using TCP rendezvous.
#[derive(Debug)]
pub struct RocmMultiNodeGroup {
    pub world_size: usize,
    pub rank: usize,
    pub comm: Option<NcclComm>,
}

impl RocmMultiNodeGroup {
    /// Create and rendezvous a multi-node group across nodes using TCP socket exchange.
    pub fn new(
        world_size: usize,
        rank: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Self> {
        if world_size <= 1 {
            return Ok(Self {
                world_size: 1,
                rank: 0,
                comm: None,
            });
        }

        #[cfg(feature = "rccl")]
        {
            let mut unique_id = NcclUniqueId { internal: [0; 128] };
            if rank == 0 {
                unsafe {
                    let res = ncclGetUniqueId(&mut unique_id);
                    if res != NCCL_SUCCESS {
                        return Err(Error::Backend(format!(
                            "ncclGetUniqueId failed with code {res}"
                        )));
                    }
                }
                let listener = std::net::TcpListener::bind(format!("{master_addr}:{master_port}"))
                    .map_err(|e| {
                        Error::Backend(format!("Failed to bind master rendezvous socket: {e}"))
                    })?;
                for _ in 1..world_size {
                    if let Ok((mut stream, _)) = listener.accept() {
                        use std::io::Write;
                        let slice = unsafe {
                            std::slice::from_raw_parts(
                                &unique_id as *const _ as *const u8,
                                std::mem::size_of::<NcclUniqueId>(),
                            )
                        };
                        let _ = stream.write_all(slice);
                    }
                }
            } else {
                let mut stream = std::net::TcpStream::connect(format!(
                    "{master_addr}:{master_port}"
                ))
                .map_err(|e| {
                    Error::Backend(format!(
                        "Worker failed to connect to master rendezvous: {e}"
                    ))
                })?;
                use std::io::Read;
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut unique_id as *mut _ as *mut u8,
                        std::mem::size_of::<NcclUniqueId>(),
                    )
                };
                stream.read_exact(slice).map_err(|e| {
                    Error::Backend(format!("Worker failed to read unique_id from master: {e}"))
                })?;
            }

            let mut comm = NcclComm(std::ptr::null_mut());
            unsafe {
                let res = ncclCommInitRank(&mut comm, world_size as i32, unique_id, rank as i32);
                if res != NCCL_SUCCESS {
                    return Err(Error::Backend(format!(
                        "ncclCommInitRank failed with code {res}"
                    )));
                }
            }
            Ok(Self {
                world_size,
                rank,
                comm: Some(comm),
            })
        }
        #[cfg(not(feature = "rccl"))]
        {
            let _ = (master_addr, master_port);
            Ok(Self {
                world_size,
                rank,
                comm: None,
            })
        }
    }
}
