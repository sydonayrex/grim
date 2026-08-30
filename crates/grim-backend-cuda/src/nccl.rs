//! Dynamic NCCL collective communication bindings and wrappers for CUDA.

use std::ffi::{c_char, c_void};
use std::sync::{Arc, Mutex};
use grim_tensor::DType;
use grim_tensor::error::{Error, Result};

/// Opaque NCCL communicator handle.
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
pub const NCCL_BFLOAT16: NcclDataType = 9;

pub type NcclRedOp = i32;
pub const NCCL_SUM: NcclRedOp = 0;
pub const NCCL_PROD: NcclRedOp = 1;
pub const NCCL_MAX: NcclRedOp = 2;
pub const NCCL_MIN: NcclRedOp = 3;
pub const NCCL_AVG: NcclRedOp = 4;

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CollectiveConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CommComputeOverlapConfig {
    pub enabled: bool,
}

/// Unique identifier for establishing communication groups.
pub struct UniqueId(pub NcclUniqueId);

impl UniqueId {
    pub fn new() -> Result<Self> {
        let bindings = NcclBindings::get()?;
        unsafe {
            let mut id = NcclUniqueId { internal: [0; 128] };
            let res = (bindings.ncclGetUniqueId)(&mut id);
            if res == NCCL_SUCCESS {
                Ok(UniqueId(id))
            } else {
                Err(Error::Backend(format!(
                    "ncclGetUniqueId failed with status {}",
                    res
                )))
            }
        }
    }
}

/// RAII wrapper for an NCCL communicator.
pub struct CudaComm {
    comm: NcclComm,
}

impl CudaComm {
    pub fn new(nranks: i32, id: UniqueId, rank: i32) -> Result<Self> {
        let bindings = NcclBindings::get()?;
        unsafe {
            let mut comm = NcclComm(std::ptr::null_mut());
            let res = (bindings.ncclCommInitRank)(&mut comm, nranks, id.0, rank);
            if res == NCCL_SUCCESS {
                Ok(CudaComm { comm })
            } else {
                Err(Error::Backend(format!(
                    "ncclCommInitRank failed with status {}",
                    res
                )))
            }
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
        let bindings = NcclBindings::get()?;
        let nccl_dtype = match dtype.arith {
            grim_tensor::ArithType::F16 => NCCL_FLOAT16,
            grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
            grim_tensor::ArithType::F32 => NCCL_FLOAT32,
            _ => {
                return Err(Error::Backend(format!(
                    "Unsupported NCCL dtype {:?}",
                    dtype
                )));
            }
        };
        unsafe {
            let res = (bindings.ncclAllReduce)(send, recv, count, nccl_dtype, NCCL_SUM, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclAllReduce failed with status {}",
                    res
                )))
            }
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
        let bindings = NcclBindings::get()?;
        let nccl_dtype = match dtype.arith {
            grim_tensor::ArithType::F16 => NCCL_FLOAT16,
            grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
            grim_tensor::ArithType::F32 => NCCL_FLOAT32,
            _ => {
                return Err(Error::Backend(format!(
                    "Unsupported NCCL dtype {:?}",
                    dtype
                )));
            }
        };
        unsafe {
            let res = (bindings.ncclReduceScatter)(
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
    }

    pub fn all_gather(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        send_count: usize,
        dtype: &DType,
        stream: *mut c_void,
    ) -> Result<()> {
        let bindings = NcclBindings::get()?;
        let nccl_dtype = match dtype.arith {
            grim_tensor::ArithType::F16 => NCCL_FLOAT16,
            grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
            grim_tensor::ArithType::F32 => NCCL_FLOAT32,
            _ => {
                return Err(Error::Backend(format!(
                    "Unsupported NCCL dtype {:?}",
                    dtype
                )));
            }
        };
        unsafe {
            let res = (bindings.ncclAllGather)(
                send, recv, send_count, nccl_dtype, self.comm, stream,
            );
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclAllGather failed with status {}",
                    res
                )))
            }
        }
    }

    pub fn send(
        &self,
        send: *const c_void,
        count: usize,
        dtype: &DType,
        peer: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        let bindings = NcclBindings::get()?;
        let nccl_dtype = match dtype.arith {
            grim_tensor::ArithType::F16 => NCCL_FLOAT16,
            grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
            grim_tensor::ArithType::F32 => NCCL_FLOAT32,
            _ => {
                return Err(Error::Backend(format!(
                    "Unsupported NCCL dtype {:?}",
                    dtype
                )));
            }
        };
        unsafe {
            let res = (bindings.ncclSend)(send, count, nccl_dtype, peer, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclSend failed with status {}",
                    res
                )))
            }
        }
    }

    pub fn recv(
        &self,
        recv: *mut c_void,
        count: usize,
        dtype: &DType,
        peer: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        let bindings = NcclBindings::get()?;
        let nccl_dtype = match dtype.arith {
            grim_tensor::ArithType::F16 => NCCL_FLOAT16,
            grim_tensor::ArithType::BF16 => NCCL_BFLOAT16,
            grim_tensor::ArithType::F32 => NCCL_FLOAT32,
            _ => {
                return Err(Error::Backend(format!(
                    "Unsupported NCCL dtype {:?}",
                    dtype
                )));
            }
        };
        unsafe {
            let res = (bindings.ncclRecv)(recv, count, nccl_dtype, peer, self.comm, stream);
            if res == NCCL_SUCCESS {
                Ok(())
            } else {
                Err(Error::Backend(format!(
                    "ncclRecv failed with status {}",
                    res
                )))
            }
        }
    }
}

impl Drop for CudaComm {
    fn drop(&mut self) {
        if !self.comm.0.is_null() {
            if let Ok(bindings) = NcclBindings::get() {
                unsafe {
                    let _ = (bindings.ncclCommDestroy)(self.comm);
                }
            }
        }
    }
}

#[allow(non_snake_case, dead_code)]
struct NcclBindings {
    #[cfg(feature = "nccl")]
    _lib: libloading::Library,
    ncclGetUniqueId: unsafe extern "C" fn(id: *mut NcclUniqueId) -> NcclResult,
    ncclCommInitRank: unsafe extern "C" fn(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
    ) -> NcclResult,
    ncclCommDestroy: unsafe extern "C" fn(comm: NcclComm) -> NcclResult,
    ncclAllReduce: unsafe extern "C" fn(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult,
    ncclReduceScatter: unsafe extern "C" fn(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult,
    ncclAllGather: unsafe extern "C" fn(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult,
    ncclGroupStart: unsafe extern "C" fn() -> NcclResult,
    ncclGroupEnd: unsafe extern "C" fn() -> NcclResult,
    ncclSend: unsafe extern "C" fn(
        sendbuff: *const c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult,
    ncclRecv: unsafe extern "C" fn(
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> NcclResult,
}

static NCCL_BINDINGS: Mutex<Option<Arc<NcclBindings>>> = Mutex::new(None);

impl NcclBindings {
    #[allow(unused_mut)]
    fn get() -> Result<Arc<Self>> {
        let mut guard = NCCL_BINDINGS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(bindings) = guard.as_ref() {
            return Ok(bindings.clone());
        }

        #[cfg(feature = "nccl")]
        {
            let candidate_names = [
                "libnccl.so.2",
                "libnccl.so",
                "/usr/local/cuda/lib64/libnccl.so.2",
                "/usr/local/cuda/lib64/libnccl.so",
                "/opt/cuda/lib64/libnccl.so.2",
                "/opt/cuda/lib64/libnccl.so",
            ];

            let mut loaded_lib = None;
            for name in candidate_names {
                if let Ok(lib) = unsafe { libloading::Library::new(name) } {
                    loaded_lib = Some(lib);
                    break;
                }
            }

            let lib = loaded_lib.ok_or_else(|| {
                Error::Backend("NCCL library (libnccl.so.2 / libnccl.so) not found in library search paths".into())
            })?;

            unsafe {
                let ncclGetUniqueId = *lib.get(b"ncclGetUniqueId\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclGetUniqueId: {e}")))?;
                let ncclCommInitRank = *lib.get(b"ncclCommInitRank\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclCommInitRank: {e}")))?;
                let ncclCommDestroy = *lib.get(b"ncclCommDestroy\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclCommDestroy: {e}")))?;
                let ncclAllReduce = *lib.get(b"ncclAllReduce\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclAllReduce: {e}")))?;
                let ncclReduceScatter = *lib.get(b"ncclReduceScatter\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclReduceScatter: {e}")))?;
                let ncclAllGather = *lib.get(b"ncclAllGather\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclAllGather: {e}")))?;
                let ncclGroupStart = *lib.get(b"ncclGroupStart\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclGroupStart: {e}")))?;
                let ncclGroupEnd = *lib.get(b"ncclGroupEnd\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclGroupEnd: {e}")))?;
                let ncclSend = *lib.get(b"ncclSend\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclSend: {e}")))?;
                let ncclRecv = *lib.get(b"ncclRecv\0")
                    .map_err(|e| Error::Backend(format!("Failed to load ncclRecv: {e}")))?;

                let bindings = Arc::new(NcclBindings {
                    _lib: lib,
                    ncclGetUniqueId,
                    ncclCommInitRank,
                    ncclCommDestroy,
                    ncclAllReduce,
                    ncclReduceScatter,
                    ncclAllGather,
                    ncclGroupStart,
                    ncclGroupEnd,
                    ncclSend,
                    ncclRecv,
                });
                *guard = Some(bindings.clone());
                Ok(bindings)
            }
        }

        #[cfg(not(feature = "nccl"))]
        {
            Err(Error::Backend("NCCL feature not enabled in Cargo.toml (enable feature \"nccl\")".into()))
        }
    }
}
