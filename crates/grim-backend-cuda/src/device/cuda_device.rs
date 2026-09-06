
//! Primary `CudaDevice` struct and mathematical tensor trait implementations.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, LazyLock, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
pub use grim_tensor::{
    AttentionOps, AutogradOps, BackendDevice, BackendStorage, CollectiveOps,
    CoreTensorOps, ElementwiseOps, FusionOps, GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps,
    RecurrentOps, SamplingOps, Shape,
};

use crate::autotune::{CudaAutotuner, CudaTileConfig, GemmOp};
use crate::caps::CudaCaps;
use crate::device::cublas::CublasHandle;
use crate::device::handles::{
    cublasCreate_v2, cuLaunchKernel, cuModuleGetFunction,
    cudaDeviceGetAttribute, cudaGetDeviceCount,
    cudaMemGetInfo, cudaSetDevice, cudaSuccess, CUBLAS_STATUS_SUCCESS,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, CUfunction, CudaHandle,
};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::CudaStorage;

/// Lazily-initialized pool of one `CudaDevice` per ordinal, so every caller —
/// in particular `to_cpu_vec_f32` on quantized weights — reuses a single cuBLAS
/// handle per GPU instead of creating (and leaking) one per tensor.
static DEVICE_POOL: LazyLock<Mutex<HashMap<usize, CudaDevice>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub(crate) ordinal: usize,
    pub caps: CudaCaps,
    pub autotuner: CudaAutotuner,
    pub(crate) cublas_handle: Arc<Mutex<Option<CublasHandle>>>,
}

// SAFETY: `CudaDevice` contains `usize`, `CudaCaps`, `CudaAutotuner` (Mutex-guarded HashMap), and `Arc<Mutex<Option<CublasHandle>>>`.
// All fields are `Send + Sync` by construction; the CUDA driver serializes concurrent use.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Returns a `CudaDevice` for the given ordinal, reusing a single pooled
    /// device (and its cuBLAS handle) per ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let mut pool = DEVICE_POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dev) = pool.get(&ordinal) {
            return Ok(dev.clone());
        }
        unsafe {
            cudaSetDevice(ordinal as i32);
        }
        let mut handle_ptr: *mut c_void = std::ptr::null_mut();
        let cublas_handle = unsafe {
            if cublasCreate_v2(&mut handle_ptr) == CUBLAS_STATUS_SUCCESS {
                Some(CublasHandle(handle_ptr))
            } else {
                None
            }
        };
        let caps = unsafe {
            let mut major = 0i32;
            let mut minor = 0i32;
            let mut sm_count = 0i32;
            let mut shared_mem = 0i32;
            let mut max_threads = 0i32;
            let mut pitch = 0i32;

            if cudaDeviceGetAttribute(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                ordinal as i32,
            ) == cudaSuccess
                && cudaDeviceGetAttribute(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    ordinal as i32,
                ) == cudaSuccess
            {
                cudaDeviceGetAttribute(
                    &mut sm_count,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    ordinal as i32,
                );
                cudaDeviceGetAttribute(
                    &mut shared_mem,
                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    ordinal as i32,
                );
                cudaDeviceGetAttribute(
                    &mut max_threads,
                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    ordinal as i32,
                );
                cudaDeviceGetAttribute(
                    &mut pitch,
                    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
                    ordinal as i32,
                );

                let mut total_mem = 0usize;
                let mut free_mem = 0usize;
                cudaMemGetInfo(&mut free_mem, &mut total_mem);

                CudaCaps {
                    device_name: format!("CUDA Device {ordinal}"),
                    ordinal,
                    compute_major: major as u32,
                    compute_minor: minor as u32,
                    multi_processor_count: sm_count.max(1) as u32,
                    total_global_mem: total_mem as u64,
                    shared_mem_per_block: shared_mem.max(49152) as u32,
                    max_threads_per_block: max_threads.max(1024) as u32,
                    max_grid_dims: [2147483647, 65535, 65535],
                    mem_pitch: pitch.max(512) as u64,
                    epoch: CudaCaps::current_epoch(),
                }
            } else {
                CudaCaps::probe_default(ordinal, format!("CUDA Device {ordinal}"), 8, 9)
            }
        };
        let autotuner = CudaAutotuner::new();
        autotuner.load_cache(&caps);
        let dev = Self {
            ordinal,
            caps,
            autotuner,
            cublas_handle: Arc::new(Mutex::new(cublas_handle)),
        };
        pool.insert(ordinal, dev.clone());
        Ok(dev)
    }

    pub fn caps(&self) -> &CudaCaps {
        &self.caps
    }

    pub fn hw_fingerprint(&self) -> u64 {
        self.caps.cache_key_hash()
    }

    /// Return the tile config for a (m,n,k) GEMM shape tagged by op-identity.
    /// cuBLAS (the current CUDA GEMM path) ignores this — the autotuner is the
    /// ROCm-parity dispatch glue; the tile config is logged for diagnostics and
    /// will drive a custom-kernel path once one exists.
    pub fn gemm_tile_config(&self, m: usize, n: usize, k: usize, op: GemmOp) -> CudaTileConfig {
        self.autotuner
            .search_tile_config(&self.caps, m, n, k, Some(op))
    }

    /// Persist the on-disk autotune cache for this device's hardware fingerprint.
    pub fn save_autotune_cache(&self) {
        self.autotuner.save_cache(&self.caps);
    }

    /// Probes for available CUDA GPUs and returns a device per instance.
    pub fn probe() -> Result<Vec<CudaDevice>> {
        if let Ok(s) = std::env::var("GRIM_CUDA_ORDINAL_OVERRIDE") {
            if let Ok(n) = s.parse::<usize>() {
                let dev = CudaDevice::new(n)?;
                return Ok(vec![dev]);
            }
        }

        let mut count: i32 = 0;
        // SAFETY: `cudaGetDeviceCount` reads the number of available CUDA devices
        // into `count`. The pointer is valid and initialized; this is a read-only query.
        let res = unsafe { cudaGetDeviceCount(&mut count) };
        if res != cudaSuccess {
            // Log the error so operators can diagnose CUDA init failures
            // (e.g. driver/runtime version mismatch, no GPU, exclusive mode).
            // Common codes: 35=cudaErrorInsufficientDriver, 100=cudaErrorNoDevice.
            eprintln!(
                "[grim-backend-cuda] cudaGetDeviceCount failed (error code: {res}). \
                 Common causes: driver/runtime mismatch (code 35), no GPU (code 100), \
                 or GPU in exclusive mode."
            );
            return Ok(vec![]);
        }
        if count == 0 {
            eprintln!("[grim-backend-cuda] cudaGetDeviceCount returned 0 devices");
            return Ok(vec![]);
        }
        let mut devices = Vec::with_capacity(count as usize);
        for i in 0..count {
            match CudaDevice::new(i as usize) {
                Ok(dev) => devices.push(dev),
                Err(e) => eprintln!(
                    "[grim-backend-cuda] CudaDevice::new({i}) failed: {e}"
                ),
            }
        }
        if devices.is_empty() {
            eprintln!(
                "[grim-backend-cuda] cudaGetDeviceCount={count} but CudaDevice::new() failed for all devices"
            );
        }
        Ok(devices)
    }

    /// Returns the raw cuBLAS handle pointer for this device, lazily initializing if needed.
    /// The caller must not free the handle — it is owned by the pooled `CudaDevice`.
    pub fn get_cublas_handle(&self) -> Result<*mut c_void> {
        let mut handle = self.cublas_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = handle.as_ref() {
            return Ok(h.0);
        }
        let mut handle_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cudaSetDevice(self.ordinal as i32);
        }
        let res = unsafe { cublasCreate_v2(&mut handle_ptr) };
        if res == CUBLAS_STATUS_SUCCESS {
            *handle = Some(CublasHandle(handle_ptr));
            Ok(handle_ptr)
        } else {
            Err(Error::Backend(format!(
                "cublasCreate failed with status {}",
                res
            )))
        }
    }

    /// Returns the device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Rejects non-F32 input early; all kernels are float* and would silently miscompute on F16/BF16.
    pub(crate) fn ensure_f32_input(name: &str, storage: &CudaStorage) -> Result<()> {
        if storage.dtype != DType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "{name}: CUDA kernel only supports F32 input (got {:?})",
                storage.dtype
            )));
        }
        Ok(())
    }

    /// Resolves a device pointer or returns Error; never panics across the FFI boundary.
    pub(crate) fn dev_ptr_or_err(name: &str, storage: &CudaStorage) -> Result<*mut c_void> {
        storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{name}: storage has no device pointer")))
            .map(|p| p as *mut c_void)
    }

    /// Launches a 1-D grid kernel from KERNELS_SOURCE with signature (ptr*, int n).
    /// Args are *mut c_void slots in declaration order; grid = ceil(n/256), block = (256,1,1).
    /// Runs on the default stream; returns an async handle.
    pub(crate) fn launch_rank1_kernel(
        &self,
        kernel_name: &str,
        args: &mut [*mut c_void],
        n: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        // SAFETY: `cuModuleGetFunction` resolves a PTX kernel name to a callable
        // function handle within the loaded module. `func` is initialized to null
        // and checked on error; the module was loaded for this device.
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name {kernel_name:?}: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }

            let block_size: usize = 256;
            let grid_size = (n + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                block_size as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel({kernel_name}) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }


}

///
/// Ties together all granular sub-traits to allow `Arc<dyn BackendDevice>` dispatch across the engine.
impl grim_tensor::BackendDevice for CudaDevice {}



/// Returns (free_bytes, total_bytes) VRAM via cudaMemGetInfo.
pub fn vram_info(ordinal: usize) -> Option<(u64, u64)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let _ = cudaSetDevice(ordinal as i32);
        let status = cudaMemGetInfo(&mut free, &mut total);
        if status != 0 {
            return None;
        }
    }
    Some((free as u64, total as u64))
}

/// WI-1: live compute utilization for `ordinal`.
///
/// Scope note (per WI-1): `grim-backend-cuda` does not link NVML, and adding
/// NVML is out of scope for this WI. Returns `None` rather than fabricating a
/// value from indirect signals — `null` on the wire is the honest answer.
pub fn compute_utilization(_ordinal: usize) -> Option<u32> {
    None
}


