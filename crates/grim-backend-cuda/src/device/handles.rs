//! CUDA FFI bindings, error codes, stream definitions, and execution handles.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};

// ---------- CUDA FFI Root Error Codes & Constants ----------

#[allow(non_upper_case_globals)]
pub const cudaSuccess: i32 = 0;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToDevice: i32 = 1;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToHost: i32 = 2;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToDevice: i32 = 3;

pub const CUBLAS_STATUS_SUCCESS: i32 = 0;
pub const CUBLAS_OP_N: i32 = 0;
pub const CUBLAS_OP_T: i32 = 1;

#[allow(non_camel_case_types)]
pub type CUdevice = i32;
#[allow(non_camel_case_types)]
pub type CUcontext = *mut c_void;
#[allow(non_camel_case_types)]
pub type CUmodule = *mut c_void;
#[allow(non_camel_case_types)]
pub type CUfunction = *mut c_void;
#[allow(non_camel_case_types)]
pub type CUstream = *mut c_void;

pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: i32 = 23;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

unsafe extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(devPtr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemcpyPeer(
        dst: *mut c_void,
        dstDevice: i32,
        src: *const c_void,
        srcDevice: i32,
        count: usize,
    ) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
    pub fn cudaSetDevice(device: i32) -> i32;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    pub fn cudaMemset(devPtr: *mut c_void, value: i32, size: usize) -> i32;
    pub fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;

    pub fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    pub fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    pub fn cublasSgemm_v2(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: *const f32,
        C: *mut f32,
        ldc: i32,
    ) -> i32;

    pub fn cuInit(flags: u32) -> i32;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> i32;
    pub fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> i32;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;

    pub fn cudaStreamCreate(stream: *mut CUstream) -> i32;
    pub fn cudaStreamDestroy(stream: CUstream) -> i32;
    pub fn cudaStreamSynchronize(stream: CUstream) -> i32;
    pub fn cudaStreamBeginCapture(stream: CUstream, mode: i32) -> i32;
    pub fn cudaStreamEndCapture(stream: CUstream, graph: *mut *mut c_void) -> i32;
    pub fn cudaGraphCreate(graph: *mut *mut c_void, flags: u32) -> i32;
    pub fn cudaGraphInstantiate(
        graphExec: *mut *mut c_void,
        graph: *mut c_void,
        pErrorNode: *mut *mut c_void,
        pLogBuffer: *mut i8,
        bufferSize: usize,
    ) -> i32;
    pub fn cudaGraphLaunch(graphExec: *mut c_void, stream: CUstream) -> i32;
    pub fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    pub fn cudaGraphExecDestroy(graphExec: *mut c_void) -> i32;
}

/// Handle to a queued CUDA operation.
#[derive(Debug)]
pub struct CudaHandle {
    pub completed: Arc<Mutex<bool>>,
}

impl CudaHandle {
    pub fn ready(_ordinal: usize) -> Self {
        Self {
            completed: Arc::new(Mutex::new(true)),
        }
    }
}

impl ComputeHandle for CudaHandle {
    /// Blocks the host thread until all tracked ops complete.
    fn synchronize(&self) -> Result<()> {
        let mut completed = self.completed.lock().unwrap_or_else(|e| e.into_inner());
        if !*completed {
            // SAFETY: `cudaDeviceSynchronize` blocks until all previous CUDA work
            // on the current device completes. It is always valid to call on the
            // current device.
            let res = unsafe { cudaDeviceSynchronize() };
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaDeviceSynchronize failed with error code {}",
                    res
                )));
            }
            *completed = true;
        }
        Ok(())
    }

    /// Returns whether tracked ops have finished.
    fn is_ready(&self) -> bool {
        *self.completed.lock().unwrap_or_else(|e| e.into_inner())
    }
}
