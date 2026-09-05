
//! Primary `CudaDevice` struct and mathematical tensor trait implementations.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, LazyLock, Mutex};

use grim_tensor::backend::{block_table_block_id, ComputeHandle};
use grim_tensor::dtype::{
    ArithType, DType, FloatPackScheme, KQuantScheme, Storage as DTypeStorage,
};
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
    cublasCreate_v2, cublasSgemm_v2, cuLaunchKernel, cuModuleGetFunction,
    cudaDeviceGetAttribute, cudaDeviceSynchronize, cudaFree, cudaGetDeviceCount,
    cudaMalloc, cudaMemGetInfo, cudaMemcpy, cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice, cudaMemcpyPeer, cudaSetDevice, cudaSuccess, CUBLAS_OP_N,
    CUBLAS_STATUS_SUCCESS, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, CUfunction, CudaHandle,
};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::{
    cuda_dequant_quantized_storage, read_length_prefixed, stage_packed_bytes, CudaStorage,
};

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
    cublas_handle: Arc<Mutex<Option<CublasHandle>>>,
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
    fn launch_rank1_kernel(
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

    /// Launches the fused Q8_0 quantized GEMM on a 2-D grid.
    /// out[M,N] = a[M,K] · b_q8[K,N]; b is int8 raw bytes with per-32-element
    /// block scales; requires K % 32 == 0. Runs on the default stream.
    fn launch_quantized_matmul_q8_0(
        &self,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        b_scales_ptr: *const c_void,
        out_ptr: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        b_data_offset: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_quantized_matmul_q8_0")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_quantized_matmul_q8_0) failed: {res}"
                )));
            }

            let mut a_arg = a_ptr;
            let mut b_arg = b_ptr;
            let mut bs_arg = b_scales_ptr;
            let mut out_arg = out_ptr;
            let mut m_arg = m as i32;
            let mut n_arg = n as i32;
            let mut k_arg = k as i32;
            let mut bdo_arg = b_data_offset as i32;
            let mut args: [*mut c_void; 8] = [
                &mut a_arg as *mut *const c_void as *mut c_void,
                &mut b_arg as *mut *const c_void as *mut c_void,
                &mut bs_arg as *mut *const c_void as *mut c_void,
                &mut out_arg as *mut *mut c_void as *mut c_void,
                &mut m_arg as *mut i32 as *mut c_void,
                &mut n_arg as *mut i32 as *mut c_void,
                &mut k_arg as *mut i32 as *mut c_void,
                &mut bdo_arg as *mut i32 as *mut c_void,
            ];

            const BLOCK_X: u32 = 32;
            const BLOCK_Y: u32 = 8;
            let grid_x = (n as u32).div_ceil(BLOCK_X);
            let grid_y = (m as u32).div_ceil(BLOCK_Y);

            let launch_res = cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                BLOCK_X,
                BLOCK_Y,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_quantized_matmul_q8_0) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches a standalone dequant kernel of signature
    /// `(const u8* packed, float* out, int n_blocks)` — one thread per
    /// 256-weight super-block. Used by the Q5_K/Q6_K/IQ4/IQ3/IQ2 family.
    /// `n` is the number of super-blocks; grid = ceil(n/256), block = 256.
    fn launch_dequant_generic(
        &self,
        kernel_name: &str,
        packed_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_blocks: usize,
        weights_per_block: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name {kernel_name:?}: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }

            let mut packed = packed_ptr;
            let mut out = out_ptr;
            let mut n_blk = n_blocks as i32;
            let mut args: [*mut c_void; 3] = [
                &mut packed as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n_blk as *mut i32 as *mut c_void,
            ];

            const BLOCK_SIZE: u32 = 256;
            // The dequantization kernels (grim_dequant_q8_0, etc.) expect one
            // thread per output weight, checking `id >= n_blocks * 32` (or
            // weights_per_block). So the grid must cover n_blocks *
            // weights_per_block threads, not just n_blocks.
            let total_weights = n_blocks.checked_mul(weights_per_block).ok_or_else(|| {
                Error::Backend("launch_dequant_generic: total weight count overflow".into())
            })?;
            let grid_size =
                ((total_weights as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
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

    /// Launches `grim_dequant_fp8(packed, out, n_weights)` — one thread per
    /// weight. The first 4 bytes of `packed` are the LE f32 global scale.
    fn launch_dequant_fp8(
        &self,
        packed_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_weights: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        if !self
            .caps
            .supports_quant_format(grim_tensor::dtype::QuantFormat::Fp8)
        {
            return Err(Error::Backend(format!(
                "FP8 dequantization is not supported on CUDA Compute Capability {}.{} (requires >= 8.9 / Ada)",
                self.caps.compute_major, self.caps.compute_minor
            )));
        }
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_dequant_fp8")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_dequant_fp8) failed: {res}"
                )));
            }
            let mut packed = packed_ptr;
            let mut out = out_ptr;
            let mut n = n_weights as i32;
            let mut args: [*mut c_void; 3] = [
                &mut packed as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            const BLOCK_SIZE: u32 = 256;
            let grid_size = ((n_weights as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_dequant_fp8) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_dequant_mxfp4(codes, exps, out, n_values)` — one thread
    /// per value. `codes` is packed E2M1 nibbles (2/byte); `exps` holds one
    /// E8M0 shared exponent per 32-element group.
    fn launch_dequant_mxfp4(
        &self,
        codes_ptr: *const c_void,
        exps_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_values: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        self.launch_dequant_mxfp_kernel(
            "grim_dequant_mxfp4",
            codes_ptr,
            exps_ptr,
            out_ptr,
            n_values,
        )
    }

    /// Launches `grim_dequant_mxfp8(codes, exps, out, n_values)` — one thread
    /// per value. `codes` is packed E4M3 bytes; `exps` holds one E8M0 shared exponent per 32-element group.
    fn launch_dequant_mxfp8(
        &self,
        codes_ptr: *const c_void,
        exps_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_values: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        self.launch_dequant_mxfp_kernel(
            "grim_dequant_mxfp8",
            codes_ptr,
            exps_ptr,
            out_ptr,
            n_values,
        )
    }

    fn launch_dequant_mxfp_kernel(
        &self,
        kernel_name: &str,
        codes_ptr: *const c_void,
        exps_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_values: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }
            let mut codes = codes_ptr;
            let mut exps = exps_ptr;
            let mut out = out_ptr;
            let mut n = n_values as i32;
            let mut args: [*mut c_void; 4] = [
                &mut codes as *mut *const c_void as *mut c_void,
                &mut exps as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            const BLOCK_SIZE: u32 = 256;
            let grid_size = ((n_values as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
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

    /// Dequantize a CUDA-resident packed `CudaStorage` to a new F32 `CudaStorage`
    /// entirely on device, returning the F32 storage. Falls back to
    /// `Err(Backend)` for block types without a device kernel so the caller
    /// (`to_cpu_vec_f32`) can use the bit-accurate host path. This is the
    /// primary mechanism by which the quantized GEMM path stays on-GPU.
    ///
    /// Block byte-size table (matches `grim_quant::dequant_*`):
    ///   Q5_K=176, Q6_K=210, IQ4_NL=170, IQ4_XS=136, IQ3_XXS=96,
    ///   IQ3_S=110, IQ2_XXS=66, IQ2_XS=74, IQ2_S=82. All are 256 weights/block.
    pub fn dequantize_on_device(&self, packed: &CudaStorage) -> Result<CudaStorage> {
        let elem_count = packed.shape.elem_count();
        let packed_ptr = Self::dev_ptr_or_err("dequantize_on_device", packed)? as *const c_void;

        let (kernel, block_bytes, weights_per_block): (&str, usize, usize) =
            match &packed.dtype.storage {
                DTypeStorage::KQuant(scheme) => match scheme {
                    KQuantScheme::Q4K => ("grim_dequant_q4k", 144, 256),
                    KQuantScheme::Q80 => ("grim_dequant_q8_0", 34, 32),
                    KQuantScheme::Q5K => ("grim_dequant_q5k", 176, 256),
                    KQuantScheme::Q6K => ("grim_dequant_q6k", 210, 256),
                    KQuantScheme::IQ4NL => ("grim_dequant_iq4nl", 144, 256),
                    KQuantScheme::IQ4XS => ("grim_dequant_iq4xs", 136, 256),
                    KQuantScheme::IQ3XXS => ("grim_dequant_iq3xxs", 96, 256),
                    KQuantScheme::IQ3S => ("grim_dequant_iq3s", 110, 256),
                    KQuantScheme::IQ2XXS => ("grim_dequant_iq2xxs", 66, 256),
                    KQuantScheme::IQ2XS => ("grim_dequant_iq2xs", 74, 256),
                    KQuantScheme::IQ2S => ("grim_dequant_iq2s", 82, 256),
                    _ => {
                        return Err(Error::Backend(format!(
                            "dequantize_on_device: no GPU kernel for KQuant {:?}",
                            scheme
                        )));
                    }
                },
                DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                    // FP8: 4-byte f32 scale header + 1 byte/weight. n_weights = elem_count.
                    let out = CudaStorage::alloc_gpu(&packed.shape, DType::F32, self.ordinal)?;
                    let out_ptr = Self::dev_ptr_or_err("dequantize_on_device(fp8)", &out)?;
                    let handle = self.launch_dequant_fp8(packed_ptr, out_ptr, elem_count)?;
                    handle.synchronize()?;
                    return Ok(out);
                }
                DTypeStorage::FloatPack(FloatPackScheme::MxFp4)
                | DTypeStorage::FloatPack(FloatPackScheme::MxFp8) => {
                    let is_mxfp4 = matches!(
                        packed.dtype.storage,
                        DTypeStorage::FloatPack(FloatPackScheme::MxFp4)
                    );
                    let raw = stage_packed_bytes(packed)?;
                    let mut cursor = 0usize;
                    let codes = read_length_prefixed(&raw, &mut cursor)?;
                    let exps = read_length_prefixed(&raw, &mut cursor)?;
                    let num_groups = elem_count.div_ceil(32);
                    if exps.len() < num_groups {
                        return Err(Error::Backend(format!(
                            "dequantize_on_device(mxfp): expected {num_groups} exp bytes, got {}",
                            exps.len()
                        )));
                    }
                    let min_codes_len = if is_mxfp4 {
                        elem_count.div_ceil(2)
                    } else {
                        elem_count
                    };
                    if codes.len() < min_codes_len {
                        return Err(Error::Backend(format!(
                            "dequantize_on_device(mxfp): expected {} code bytes, got {}",
                            min_codes_len,
                            codes.len()
                        )));
                    }
                    let codes_shape_external = Shape::new(vec![codes.len()]);
                    let codes_storage = CudaStorage::copy_from_host_raw_bytes(
                        &codes,
                        &codes_shape_external,
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        self.ordinal,
                    )?;
                    let exps_shape = Shape::new(vec![exps.len()]);
                    let exps_storage = CudaStorage::copy_from_host_raw_bytes(
                        &exps,
                        &exps_shape,
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        self.ordinal,
                    )?;
                    let out = CudaStorage::alloc_gpu(&packed.shape, DType::F32, self.ordinal)?;
                    let codes_ptr =
                        Self::dev_ptr_or_err("dequantize_on_device(mxfp codes)", &codes_storage)?;
                    let exps_ptr =
                        Self::dev_ptr_or_err("dequantize_on_device(mxfp exps)", &exps_storage)?;
                    let out_ptr = Self::dev_ptr_or_err("dequantize_on_device(mxfp out)", &out)?;
                    let handle = if is_mxfp4 {
                        self.launch_dequant_mxfp4(codes_ptr, exps_ptr, out_ptr, elem_count)?
                    } else {
                        self.launch_dequant_mxfp8(codes_ptr, exps_ptr, out_ptr, elem_count)?
                    };
                    handle.synchronize()?;
                    return Ok(out);
                }
                // FP4/NF4/MXFP8 and Block(Fp4/Nf4/Fp8Block16) keep the host path.
                _ => {
                    return Err(Error::Backend(format!(
                        "dequantize_on_device: no GPU kernel for dtype {:?}",
                        packed.dtype
                    )));
                }
            };

        // Super-block path (Q5_K/Q6_K/IQ*).
        let n_blocks = elem_count.div_ceil(weights_per_block);
        if packed.bytes < n_blocks * block_bytes {
            return Err(Error::Backend(format!(
                "dequantize_on_device({kernel}): packed buffer too short ({} < {}*{})",
                packed.bytes, n_blocks, block_bytes
            )));
        }
        let out = CudaStorage::alloc_gpu(&packed.shape, DType::F32, self.ordinal)?;
        let out_ptr = Self::dev_ptr_or_err("dequantize_on_device(out)", &out)?;
        let handle = self.launch_dequant_generic(kernel, packed_ptr, out_ptr, n_blocks, weights_per_block)?;
        handle.synchronize()?;
        Ok(out)
    }

    /// Quantize a CUDA-resident F32 `CudaStorage` into packed quantized bytes,
    /// entirely on-device — the device-side mirror of `grim_quant::quant_*`.
    ///
    /// Returns a new `CudaStorage` holding the packed bytes with the
    /// appropriate `Storage` dtype. Currently supports Q8_0 and FP8 (E4M3).
    pub fn quantize_on_device(
        &self,
        x: &CudaStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<CudaStorage> {
        Self::ensure_f32_input("quantize_on_device", x)?;
        let n_weights = x.shape.elem_count();
        let x_ptr = Self::dev_ptr_or_err("quantize_on_device x", x)? as *const c_void;

        match format {
            grim_tensor::QuantFormat::Q8_0 => {
                if n_weights % 32 != 0 {
                    return Err(Error::Backend(format!(
                        "quantize_on_device(Q8_0): n_weights ({n_weights}) must be a multiple of 32"
                    )));
                }
                let n_blocks = n_weights / 32;
                let out_bytes = n_blocks * 34;
                let out_shape = Shape::new(vec![out_bytes]);
                let out = CudaStorage::alloc_gpu_bytes(
                    &out_shape,
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                    out_bytes,
                    self.ordinal,
                )?;
                let out_ptr = Self::dev_ptr_or_err("quantize_on_device(Q8_0 out)", &out)?;
                let handle = self.launch_quant_q8_0(x_ptr, out_ptr, n_blocks)?;
                handle.synchronize()?;
                Ok(out)
            }
            grim_tensor::QuantFormat::Fp8 => {
                // T1 caps gate: a device without native FP8 (compute < 8.9) must not
                // dispatch the fp8 quantize kernel.
                if !self
                    .caps
                    .supports_quant_format(grim_tensor::QuantFormat::Fp8)
                {
                    return Err(Error::Backend(
                        "quantize_on_device: FP8 not supported on this device".into(),
                    ));
                }
                let out_bytes = 4 + n_weights;
                let out_shape = Shape::new(vec![out_bytes]);
                let out = CudaStorage::alloc_gpu_bytes(
                    &out_shape,
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                    },
                    out_bytes,
                    self.ordinal,
                )?;
                let out_ptr = Self::dev_ptr_or_err("quantize_on_device(FP8 out)", &out)?;
                let handle = self.launch_quant_fp8(x_ptr, out_ptr, n_weights)?;
                handle.synchronize()?;
                Ok(out)
            }
            other => Err(Error::Backend(format!(
                "quantize_on_device: no GPU kernel for format {other:?}"
            ))),
        }
    }

    /// Launches `grim_quant_q8_0(x, out, n_blocks)` — one 32-thread block per
    /// Q8_0 block. Warp shuffle reduction for amax.
    fn launch_quant_q8_0(
        &self,
        x_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_blocks: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_quant_q8_0")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_quant_q8_0) failed: {res}"
                )));
            }

            let mut x = x_ptr;
            let mut out = out_ptr;
            let mut n_blk = n_blocks as i32;
            let mut args: [*mut c_void; 3] = [
                &mut x as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n_blk as *mut i32 as *mut c_void,
            ];

            // One block of 32 threads per Q8_0 block.
            const BLOCK_SIZE: u32 = 32;
            let launch_res = cuLaunchKernel(
                func,
                n_blocks as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_quant_q8_0) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_quant_fp8(x, out, n_weights)` — one thread per weight.
    /// The first 4 bytes of `out` are the LE f32 scale (1.0f).
    fn launch_quant_fp8(
        &self,
        x_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_weights: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_quant_fp8")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_quant_fp8) failed: {res}"
                )));
            }

            let mut x = x_ptr;
            let mut out = out_ptr;
            let mut n = n_weights as i32;
            let mut args: [*mut c_void; 3] = [
                &mut x as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];

            const BLOCK_SIZE: u32 = 256;
            let grid_size = ((n_weights as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_quant_fp8) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_fused_quant_gemm_{format}(A, B, C, M, N, K)` — one thread
    /// per output element. Grid/block match `launch_quantized_matmul_q8_0`.
    fn launch_fused_quant_gemm(
        &self,
        kernel_name: &str,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        out_ptr: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }

            let mut a = a_ptr;
            let mut b = b_ptr;
            let mut out = out_ptr;
            let mut m_arg = m as i32;
            let mut n_arg = n as i32;
            let mut k_arg = k as i32;
            let mut args: [*mut c_void; 6] = [
                &mut a as *mut *const c_void as *mut c_void,
                &mut b as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut m_arg as *mut i32 as *mut c_void,
                &mut n_arg as *mut i32 as *mut c_void,
                &mut k_arg as *mut i32 as *mut c_void,
            ];

            const BLOCK_X: u32 = 32;
            const BLOCK_Y: u32 = 8;
            let grid_x = (n as u32).div_ceil(BLOCK_X);
            let grid_y = (m as u32).div_ceil(BLOCK_Y);

            let launch_res = cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                BLOCK_X,
                BLOCK_Y,
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

impl CudaDevice {
    /// Fused QKV attention (Phase-1, mirrors `RocmDevice::qkv_attention`).
    ///
    /// Parameters (q: [S, H, D], k/v: [kv_S, kv_H, D], f32).
    /// Uses grim_qkv_attention kernel (online softmax, per-wave partials merged by wave-0).
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Compute the host-side window_lo lower bound, exactly mirroring the
        // ROCm path. For decode (seq_len == 1) this is the per-query bound
        // max(0, cache_offset - window + 1); for prefill it is the block-wide
        // minimum (conservative lower bound) so the kernel stays branch-free
        // in the common full-causal case (window_lo == 0). See the ROCm
        // `qkv_attention` commentary for the Laguna-S-2.1 rationale.
        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => {
                let abs_first = cache_offset as usize;
                abs_first.saturating_sub(w.saturating_sub(1)) as i32
            }
        };

        let out_dims = out.dims();
        let (seq_len, num_heads, head_dim) = if out_dims.len() == 3 {
            (out_dims[0], out_dims[1], out_dims[2])
        } else if out_dims.len() == 2 {
            let seq_len = out_dims[0];
            let hidden_dim = out_dims[1];
            let q_dims = q.shape().dims();
            let head_dim = if q_dims.len() == 3 {
                q_dims[2]
            } else if q_dims.len() == 2 && num_kv_heads > 0 {
                q_dims[1] / num_kv_heads
            } else {
                hidden_dim / num_kv_heads.max(1)
            };
            if head_dim == 0 {
                return Err(Error::Shape(
                    "qkv_attention head_dim resolved to zero; malformed model dimension".into(),
                ));
            }
            let num_heads = hidden_dim / head_dim;
            (seq_len, num_heads, head_dim)
        } else {
            return Err(Error::Shape(
                "qkv_attention expects 2-D [seq_len, hidden_dim] or 3-D [seq_len, num_heads, head_dim] output shape".into(),
            ));
        };
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(Error::Shape(
                "qkv_attention: zero-sized num_heads / num_kv_heads / head_dim".into(),
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "qkv_attention: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "qkv_attention: head_dim <= 256 supported (got {head_dim})"
            )));
        }

        let q_s = q
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention q is not CudaStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention k is not CudaStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention v is not CudaStorage".into()))?;
        Self::ensure_f32_input("qkv_attention q", q_s)?;
        Self::ensure_f32_input("qkv_attention k", k_s)?;
        Self::ensure_f32_input("qkv_attention v", v_s)?;

        let max_s = match out_max {
            Some(m) => Some(m.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention out_max is not CudaStorage".into())
            })?),
            None => None,
        };
        let sum_s = match out_sum {
            Some(s) => Some(s.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention out_sum is not CudaStorage".into())
            })?),
            None => None,
        };

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();

        let mut q_ptr = Self::dev_ptr_or_err("qkv_attention q", q_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("qkv_attention k", k_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("qkv_attention v", v_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("qkv_attention out", &out_storage)?;
        let mut max_ptr: u64 = match max_s {
            Some(m) => m.device_ptr.unwrap_or(0),
            None => 0,
        };
        let mut sum_ptr: u64 = match sum_s {
            Some(s) => s.device_ptr.unwrap_or(0),
            None => 0,
        };
        let mut num_heads_i = num_heads as i32;
        let mut num_kv_heads_i = num_kv_heads as i32;
        let mut head_dim_i = head_dim as i32;
        let mut seq_len_i = seq_len as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut cache_offset_i = cache_offset as i32;
        let mut inv_sqrt_d_val = inv_sqrt_d;
        let mut window_lo_val = window_lo_i;

        // 14 kernel args: q, k, v, out, out_max, out_sum, num_heads, num_kv_heads,
        // head_dim, seq_len, kv_seq_len, cache_offset, inv_sqrt_d, window_lo.
        let mut args: [*mut c_void; 14] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut max_ptr as *mut u64 as *mut c_void,
            &mut sum_ptr as *mut u64 as *mut c_void,
            &mut num_heads_i as *mut i32 as *mut c_void,
            &mut num_kv_heads_i as *mut i32 as *mut c_void,
            &mut head_dim_i as *mut i32 as *mut c_void,
            &mut seq_len_i as *mut i32 as *mut c_void,
            &mut kv_seq_len_i as *mut i32 as *mut c_void,
            &mut cache_offset_i as *mut i32 as *mut c_void,
            &mut inv_sqrt_d_val as *mut f32 as *mut c_void,
            &mut window_lo_val as *mut i32 as *mut c_void,
        ];

        // 2-D grid (seq_len, num_heads) vs launch_rank1_kernel's 1-D.
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_qkv_attention")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_qkv_attention) failed: {res}"
                )));
            }
            let launch_res = cuLaunchKernel(
                func,
                seq_len as u32,
                num_heads as u32,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_qkv_attention) failed: {launch_res}"
                )));
            }
        }
        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Fused grouped MoE dispatch (WI-M5). Mirrors `grim_moe_fused_dispatch` on ROCm
    /// and `moe_fused_dispatch` on Vulkan: one CUDA thread block per routed
    /// (token, expert) pair, computing the full SwiGLU expert contribution and
    /// atomicAdding the `routed_scaling_factor * weight`-scaled result into the
    /// shared token output. `grid_x = num_pairs`.
    pub fn moe_fused_dispatch(
        &self,
        x: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        router_tokens: &dyn BackendStorage,
        router_experts: &dyn BackendStorage,
        router_weights: &dyn BackendStorage,
        out_shape: &Shape,
        hidden: u32,
        inter: u32,
        num_experts: u32,
        batch: u32,
        routed_scaling_factor: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("moe_fused_dispatch: x is not CudaStorage".into()))?;
        let gw_s = gate_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: gate_w is not CudaStorage".into())
            })?;
        let uw_s = up_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("moe_fused_dispatch: up_w is not CudaStorage".into()))?;
        let dw_s = down_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: down_w is not CudaStorage".into())
            })?;
        let tok_s = router_tokens
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: router_tokens is not CudaStorage".into())
            })?;
        let exp_s = router_experts
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: router_experts is not CudaStorage".into())
            })?;
        let wt_s = router_weights
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: router_weights is not CudaStorage".into())
            })?;
        Self::ensure_f32_input("moe_fused_dispatch x", x_s)?;
        Self::ensure_f32_input("moe_fused_dispatch gate_w", gw_s)?;
        Self::ensure_f32_input("moe_fused_dispatch up_w", uw_s)?;
        Self::ensure_f32_input("moe_fused_dispatch down_w", dw_s)?;

        // Output is zero-initialized; the kernel atomicAdds contributions.
        // cudaMalloc does not zero memory, so clear it explicitly before launch.
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        out_storage.fill_zeroes()?;
        let mut out_ptr = out_storage.device_ptr.unwrap();

        let mut x_ptr = x_s.device_ptr.unwrap();
        let mut gw_ptr = gw_s.device_ptr.unwrap();
        let mut uw_ptr = uw_s.device_ptr.unwrap();
        let mut dw_ptr = dw_s.device_ptr.unwrap();
        let mut tok_ptr = tok_s.device_ptr.unwrap();
        let mut exp_ptr = exp_s.device_ptr.unwrap();
        let mut wt_ptr = wt_s.device_ptr.unwrap();
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_experts_i = num_experts as i32;
        let mut batch_i = batch as i32;
        let mut rsf = routed_scaling_factor;

        let mut args: [*mut c_void; 13] = [
            &mut x_ptr as *mut u64 as *mut c_void,
            &mut gw_ptr as *mut u64 as *mut c_void,
            &mut uw_ptr as *mut u64 as *mut c_void,
            &mut dw_ptr as *mut u64 as *mut c_void,
            &mut tok_ptr as *mut u64 as *mut c_void,
            &mut exp_ptr as *mut u64 as *mut c_void,
            &mut wt_ptr as *mut u64 as *mut c_void,
            &mut out_ptr as *mut u64 as *mut c_void,
            &mut hidden_i as *mut i32 as *mut c_void,
            &mut inter_i as *mut i32 as *mut c_void,
            &mut num_experts_i as *mut i32 as *mut c_void,
            &mut batch_i as *mut i32 as *mut c_void,
            &mut rsf as *mut f32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_moe_fused_dispatch")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_moe_fused_dispatch) failed: {res}"
                )));
            }
            // grid_x = number of routed pairs (one block per pair).
            let num_pairs = tok_s.shape.elem_count() as u32;
            let launch_res = cuLaunchKernel(
                func,
                num_pairs,
                1,
                1,
                1,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_moe_fused_dispatch) failed: {launch_res}"
                )));
            }
        }
        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Fused MoE dispatch against resident weights.
    ///
    /// Provides parity with resident MoE dispatch entry points on other backend devices.
    pub fn moe_fused_dispatch_resident(
        &self,
        x: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        router_tokens: &dyn BackendStorage,
        router_experts: &dyn BackendStorage,
        router_weights: &dyn BackendStorage,
        out_shape: &Shape,
        hidden: u32,
        inter: u32,
        num_experts: u32,
        batch: u32,
        routed_scaling_factor: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.moe_fused_dispatch(
            x,
            gate_w,
            up_w,
            down_w,
            router_tokens,
            router_experts,
            router_weights,
            out_shape,
            hidden,
            inter,
            num_experts,
            batch,
            routed_scaling_factor,
        )
    }

    pub fn matmul_op(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        op: Option<GemmOp>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("matmul a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("matmul b is not CudaStorage".into()))?;

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape("matmul expects 2-D inputs".into()));
        }
        let (m, k) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }
        if out_shape.dims() != &[m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {out_shape:?}"
            )));
        }

        let dtype_out = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        };
        if a_storage.dtype != DType::F32 || b_storage.dtype != DType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "matmul: CUDA backend only supports F32 inputs (a={:?}, b={:?})",
                a_storage.dtype, b_storage.dtype
            )));
        }
        let out_storage = CudaStorage::alloc_gpu(out_shape, dtype_out, self.ordinal)?;

        unsafe {
            let _ = cudaSetDevice(self.ordinal as i32);
        }

        let handle_guard = self.cublas_handle.lock().unwrap_or_else(|e| e.into_inner());
        let handle = handle_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("cuBLAS handle not initialized".into()))?
            .0;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("matmul: A storage has no valid device pointer".into()))?
            as *const c_void;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("matmul: B storage has no valid device pointer".into()))?
            as *const c_void;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("matmul: out storage has no valid device pointer".into())
        })? as *mut c_void;

        unsafe {
            let status = cublasSgemm_v2(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                b_ptr as *const f32,
                n as i32,
                a_ptr as *const f32,
                k as i32,
                &beta,
                out_ptr as *mut f32,
                n as i32,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(Error::Backend(format!(
                    "cublasSgemm_v2 failed with status {}",
                    status
                )));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(true)),
        });

        let effective_op = op.unwrap_or(GemmOp::Other);
        let tile_cfg = self.gemm_tile_config(m, n, k, effective_op);
        if !tile_cfg.is_valid(&self.caps) {
            tracing::warn!(
                target: "grim_cuda",
                m = m,
                n = n,
                k = k,
                tile = ?tile_cfg,
                op = ?op,
                "CUDA matmul: tile config exceeds device resource limits"
            );
        }
        tracing::debug!(
            target: "grim_cuda",
            m = m,
            n = n,
            k = k,
            tile = ?tile_cfg,
            op = ?op,
            "CUDA matmul: tile config logged"
        );

        Ok((Box::new(out_storage), compute_handle))
    }

    /// Launch GPU Speculative Rejection Sampling kernel on CUDA.
    pub fn launch_speculative_rejection_sample(
        &self,
        target_probs_storage: &CudaStorage,
        draft_probs_storage: &CudaStorage,
        draft_tokens_storage: &CudaStorage,
        uniform_rands_storage: &CudaStorage,
        accepted_tokens_storage: &CudaStorage,
        accepted_lens_storage: &CudaStorage,
        batch_size: usize,
        num_draft_tokens: usize,
        vocab_size: usize,
    ) -> Result<()> {
        let tp_ptr = target_probs_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: target_probs has no device ptr".into()))?;
        let dp_ptr = draft_probs_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: draft_probs has no device ptr".into()))?;
        let dt_ptr = draft_tokens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: draft_tokens has no device ptr".into()))?;
        let ur_ptr = uniform_rands_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: uniform_rands has no device ptr".into()))?;
        let at_ptr = accepted_tokens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: accepted_tokens has no device ptr".into()))?;
        let al_ptr = accepted_lens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: accepted_lens has no device ptr".into()))?;

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut kernel: *mut c_void = std::ptr::null_mut();
        let name = std::ffi::CString::new("grim_speculative_rejection_sample").unwrap();
        unsafe {
            let res = cuModuleGetFunction(&mut kernel, module, name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction failed for grim_speculative_rejection_sample: {res}"
                )));
            }

            let mut tp = tp_ptr as *mut c_void;
            let mut dp = dp_ptr as *mut c_void;
            let mut dt = dt_ptr as *mut c_void;
            let mut ur = ur_ptr as *mut c_void;
            let mut at = at_ptr as *mut c_void;
            let mut al = al_ptr as *mut c_void;
            let mut bs = batch_size as i32;
            let mut ndt = num_draft_tokens as i32;
            let mut vs = vocab_size as i32;

            let mut args: [*mut c_void; 9] = [
                &mut tp as *mut _ as *mut c_void,
                &mut dp as *mut _ as *mut c_void,
                &mut dt as *mut _ as *mut c_void,
                &mut ur as *mut _ as *mut c_void,
                &mut at as *mut _ as *mut c_void,
                &mut al as *mut _ as *mut c_void,
                &mut bs as *mut _ as *mut c_void,
                &mut ndt as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
            ];

            let res = cuLaunchKernel(
                kernel,
                batch_size as u32, 1, 1,
                256, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel failed for grim_speculative_rejection_sample: {res}"
                )));
            }
        }
        Ok(())
    }

    /// Cross-device direct copy between CUDA devices using cudaMemcpyPeer or staging fallback.
    pub fn copy_via_route(
        &self,
        src_ordinal: i32,
        dst_ordinal: i32,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
        bytes: usize,
    ) -> Result<()> {
        if src_ordinal == dst_ordinal {
            unsafe {
                let res = cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice);
                if res != cudaSuccess {
                    return Err(Error::Backend(format!(
                        "copy_via_route D2D failed on device {dst_ordinal}: {res}"
                    )));
                }
            }
            return Ok(());
        }

        // Try direct peer memcpy
        unsafe {
            let res = cudaMemcpyPeer(
                dst_ptr,
                dst_ordinal,
                src_ptr,
                src_ordinal,
                bytes,
            );
            if res == cudaSuccess {
                return Ok(());
            }

            // Fallback via host staging buffer
            let mut staging = vec![0u8; bytes];
            let res_d2h = cudaMemcpy(
                staging.as_mut_ptr() as *mut c_void,
                src_ptr,
                bytes,
                cudaMemcpyDeviceToHost,
            );
            if res_d2h != cudaSuccess {
                return Err(Error::Backend(format!(
                    "copy_via_route D2H fallback failed: {res_d2h}"
                )));
            }
            let _ = cudaSetDevice(dst_ordinal);
            let res_h2d = cudaMemcpy(
                dst_ptr,
                staging.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyHostToDevice,
            );
            if res_h2d != cudaSuccess {
                return Err(Error::Backend(format!(
                    "copy_via_route H2D fallback failed: {res_h2d}"
                )));
            }
        }
        Ok(())
    }

    /// Explicit engine-floor tagger for the lm_head / logit-projection GEMM. Forces
    /// `ShapeClass::TLOLog` (op-identity) regardless of M, so the wide-N tile is selected
    /// instead of relying on the n>=16384 dimension heuristic in the trait `matmul`.
    pub fn matmul_lm_head(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, Some(GemmOp::LmHead))
    }

    /// Fused Add + RMSNorm: `y_out = x + residual`, `norm_out = rms_norm(y_out, w, eps)`.
    /// Returns `(y_out, res_out, compute_handle)`. Mirrors the ROCm `grim_add_rms_norm` HIP
    /// kernel and the Metal MSL shader 1:1 — one PTX thread per output element, no shared mem.
    pub fn fused_add_rms_norm(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_add_rms_norm x is not CudaStorage".into()))?;
        let res_storage = residual
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("fused_add_rms_norm residual is not CudaStorage".into())
            })?;
        let w_storage = weight
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_add_rms_norm weight is not CudaStorage".into()))?;
        Self::ensure_f32_input("fused_add_rms_norm x", x_storage)?;
        Self::ensure_f32_input("fused_add_rms_norm residual", res_storage)?;
        Self::ensure_f32_input("fused_add_rms_norm weight", w_storage)?;

        let dtype_out = DType::F32;
        let y_storage = CudaStorage::alloc_gpu(out_shape, dtype_out.clone(), self.ordinal)?;
        let norm_storage = CudaStorage::alloc_gpu(out_shape, dtype_out, self.ordinal)?;

        let total = out_shape.elem_count();
        let row_len = out_shape.dims()[out_shape.dims().len() - 1];

        let mut x_ptr = Self::dev_ptr_or_err("fused_add_rms_norm x", x_storage)?;
        let mut res_ptr = Self::dev_ptr_or_err("fused_add_rms_norm residual", res_storage)?;
        let mut w_ptr = Self::dev_ptr_or_err("fused_add_rms_norm weight", w_storage)?;
        let mut y_ptr = Self::dev_ptr_or_err("fused_add_rms_norm y_out", &y_storage)?;
        let mut norm_ptr = Self::dev_ptr_or_err("fused_add_rms_norm norm_out", &norm_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_val = eps;
        let mut total_i = total as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut res_ptr as *mut *mut c_void as *mut c_void,
            &mut w_ptr as *mut *mut c_void as *mut c_void,
            &mut y_ptr as *mut *mut c_void as *mut c_void,
            &mut norm_ptr as *mut *mut c_void as *mut c_void,
            &mut row_len_i as *mut i32 as *mut c_void,
            &mut eps_val as *mut f32 as *mut c_void,
            &mut total_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_add_rms_norm", &mut args, total)?;
        Ok((Box::new(y_storage), Box::new(norm_storage), handle))
    }
}

impl CoreTensorOps for CudaDevice {

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        if dtype != DType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "zeros: CUDA backend only supports F32 (got {dtype:?})"
            )));
        }
        let storage = CudaStorage::alloc_gpu(shape, dtype, self.ordinal)?;
        let dev_ptr = storage
            .device_ptr
            .ok_or_else(|| Error::Backend("zeros: device_ptr was null after alloc_gpu".into()))?
            as *mut c_void;

        let zeros_host = vec![0.0f32; shape.elem_count()];
        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                zeros_host.as_ptr() as *const c_void,
                storage.bytes,
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpy failed to initialize zeros with error {}",
                res
            )));
        }

        Ok(Box::new(storage))
    }


    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, None)
    }


    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("add a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("add b is not CudaStorage".into()))?;
        Self::ensure_f32_input("add a", a_storage)?;
        Self::ensure_f32_input("add b", b_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let mut a_ptr = Self::dev_ptr_or_err("add a", a_storage)?;
        let mut b_ptr = Self::dev_ptr_or_err("add b", b_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("add out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut a_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_add", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }


    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mul a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mul b is not CudaStorage".into()))?;
        Self::ensure_f32_input("mul a", a_storage)?;
        Self::ensure_f32_input("mul b", b_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let mut a_ptr = Self::dev_ptr_or_err("mul a", a_storage)?;
        let mut b_ptr = Self::dev_ptr_or_err("mul b", b_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("mul out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut a_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_mul", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }


    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_storage = gate
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul gate is not CudaStorage".into()))?;
        let up_storage = up
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul up is not CudaStorage".into()))?;
        Self::ensure_f32_input("silu_mul gate", gate_storage)?;
        Self::ensure_f32_input("silu_mul up", up_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let mut gate_ptr = Self::dev_ptr_or_err("silu_mul gate", gate_storage)?;
        let mut up_ptr = Self::dev_ptr_or_err("silu_mul up", up_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("silu_mul out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut gate_ptr as *mut *mut c_void as *mut c_void,
            &mut up_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_silu_mul", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }


    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rms_norm x is not CudaStorage".into()))?;
        let w_storage = w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rms_norm w is not CudaStorage".into()))?;
        Self::ensure_f32_input("rms_norm x", x_storage)?;
        Self::ensure_f32_input("rms_norm w", w_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let total = out.elem_count();
        let row_len = out.dims()[out.dims().len() - 1];

        let mut x_ptr = Self::dev_ptr_or_err("rms_norm x", x_storage)?;
        let mut w_ptr = Self::dev_ptr_or_err("rms_norm w", w_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("rms_norm out", &out_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_val = eps;
        let mut total_i = total as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut w_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut row_len_i as *mut i32 as *mut c_void,
            &mut eps_val as *mut f32 as *mut c_void,
            &mut total_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_rms_norm", &mut args, total)?;
        Ok((Box::new(out_storage), handle))
    }


    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("softmax x is not CudaStorage".into()))?;
        Self::ensure_f32_input("softmax x", x_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let total = out.elem_count();
        let last_dim = out.dims()[out.dims().len() - 1];

        let mut x_ptr = Self::dev_ptr_or_err("softmax x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("softmax out", &out_storage)?;
        let mut last_dim_i = last_dim as i32;
        let mut total_i = total as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut last_dim_i as *mut i32 as *mut c_void,
            &mut total_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_softmax", &mut args, total)?;
        Ok((Box::new(out_storage), handle))
    }


    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let weight_storage = weight
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("embedding weight is not CudaStorage".into()))?;
        Self::ensure_f32_input("embedding weight", weight_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let num_indices = indices.len();
        let embedding_dim = out.dims()[out.dims().len() - 1];

        // Allocate, upload, run, sync, free indices. Error paths free before returning.
        let mut dev_indices_ptr: *mut c_void = std::ptr::null_mut();
        let size_indices = num_indices * 4;
        unsafe {
            let res = cudaMalloc(&mut dev_indices_ptr, size_indices);
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMalloc for indices failed: {res}"
                )));
            }
            let res = cudaMemcpy(
                dev_indices_ptr,
                indices.as_ptr() as *const c_void,
                size_indices,
                cudaMemcpyHostToDevice,
            );
            if res != cudaSuccess {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!(
                    "cudaMemcpy for indices failed: {res}"
                )));
            }
        }

        // Embedding takes dev_indices_ptr and uses num_indices * embedding_dim threads,
        // so it can't use launch_rank1_kernel directly.
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_embedding")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {res}")));
            }

            let mut w_ptr = Self::dev_ptr_or_err("embedding weight", weight_storage)?;
            let mut indices_ptr = dev_indices_ptr;
            let mut out_ptr = Self::dev_ptr_or_err("embedding out", &out_storage)?;
            let mut emb_dim_i = embedding_dim as i32;
            let mut num_idx_i = num_indices as i32;

            let mut args = [
                &mut w_ptr as *mut *mut c_void as *mut c_void,
                &mut indices_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut emb_dim_i as *mut i32 as *mut c_void,
                &mut num_idx_i as *mut i32 as *mut c_void,
            ];

            let block_size: usize = 256;
            let total_threads = num_indices * embedding_dim;
            let grid_size = (total_threads + block_size - 1) / block_size;

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
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!(
                    "cuLaunchKernel failed: {launch_res}"
                )));
            }
        }

        // Sync so staging buffer is safe to free.
        unsafe {
            let _ = cudaDeviceSynchronize();
            let _ = cudaFree(dev_indices_ptr);
        }
        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(true)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }


    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = CudaStorage::copy_from_host(data, shape, dtype, self.ordinal)?;
        Ok(Box::new(storage))
    }


    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        Ok(())
    }
}

impl ElementwiseOps for CudaDevice {


    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mul_scalar x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut x_ptr = Self::dev_ptr_or_err("mul_scalar x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("mul_scalar out", &out_storage)?;
        let mut s_val = scalar;
        let mut n_i = n as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut s_val as *mut f32 as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_mul_scalar", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }


    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sqrt x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut x_ptr = Self::dev_ptr_or_err("sqrt x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("sqrt out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_sqrt", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }


    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("recip x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut x_ptr = Self::dev_ptr_or_err("recip x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("recip out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_recip", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn sub(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sub a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sub b is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut a_ptr = Self::dev_ptr_or_err("sub a", a_storage)?;
        let mut b_ptr = Self::dev_ptr_or_err("sub b", b_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("sub out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut a_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_sub", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn reduce_sum(&self, x: &dyn BackendStorage) -> Result<f32> {
        let v = x.to_cpu_vec_f32()?;
        if v.is_empty() {
            return Err(Error::Backend("reduce_sum: empty tensor".into()));
        }
        Ok(v.iter().sum())
    }

    fn reduce_max(&self, x: &dyn BackendStorage) -> Result<f32> {
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error::Backend("reduce_max: empty tensor".into()))
    }

    fn argmax(&self, x: &dyn BackendStorage) -> Result<u32> {
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| Error::Backend("argmax: empty tensor".into()))
    }
}

impl SamplingOps for CudaDevice {
}

impl AttentionOps for CudaDevice {


    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        CudaDevice::qkv_attention(
            self,
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
            out_max,
            out_sum,
        )
    }

    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // CUDA paged attention runs the SDPA on the host (CPU readback) and
        // uploads the result, mirroring the pre-SWA implementation. The
        // sliding-window mask is applied here as `window_start` so a passed
        // `Some(w)` no longer hard-fails and matches the CPU reference SWA.
        let window_w = window;

        let qd = q.to_cpu_vec_f32()?;
        let btd = block_tables.to_cpu_vec_f32()?;
        let kd = k_pages.to_cpu_vec_f32()?;
        let vd = v_pages.to_cpu_vec_f32()?;

        // The KV pool is laid out as [pool_blocks * page_size, kv_stride].
        // Derive the pool's real block count from the pages buffer dimensions
        // so every block-table entry is validated against the actual pool
        // capacity rather than blindly trusting the f32->usize cast below.
        let k_dims = k_pages.shape().dims();
        if k_dims.len() < 2 || page_size == 0 || k_dims[0] % page_size != 0 {
            return Err(Error::Shape(format!(
                "qkv_attention_paged: k_pages shape {k_dims:?} is not a [pool_blocks*page_size, kv_stride] KV pool for page_size {page_size}"
            )));
        }
        let v_dims = v_pages.shape().dims();
        if v_dims.len() < 2 || v_dims[0] != k_dims[0] {
            return Err(Error::Shape(format!(
                "qkv_attention_paged: k_pages and v_pages pool shapes differ: {k_dims:?} vs {v_dims:?}"
            )));
        }
        let pool_blocks = k_dims[0] / page_size;

        // Resolve a sequence-block position to a physical pool block id. The
        // table carries packed `BlockTableEntry { block_id, page_size }` words
        // (see `paged_self_attention` in grim-models-transformer); the entry
        // word is decoded with `to_bits` and validated against the real pool
        // capacity before any buffer index is computed; an out-of-range id is a
        // hard error instead of an out-of-bounds index.
        let resolve_block = |block_idx_in_seq: usize| -> Result<usize> {
            let id = if block_idx_in_seq < max_blocks {
                if block_idx_in_seq * 2 >= btd.len() {
                    return Err(Error::Backend(format!(
                        "qkv_attention_paged: block table entry {block_idx_in_seq} is out of range (table holds {} words, 2 per entry)",
                        btd.len()
                    )));
                }
                block_table_block_id(&btd, block_idx_in_seq, max_blocks)
            } else {
                block_idx_in_seq
            };
            if id >= pool_blocks {
                return Err(Error::Backend(format!(
                    "qkv_attention_paged: block-table entry {block_idx_in_seq} maps to physical block {id}, which exceeds the KV pool capacity of {pool_blocks} blocks"
                )));
            }
            Ok(id)
        };

        let q_dims = q.shape().dims();
        if q_dims.len() != 3 {
            return Err(Error::Shape("qkv_attention_paged: q must be 3-D".into()));
        }
        let seq_len = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];

        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "qkv_attention_paged: num_heads must be multiple of num_kv_heads".into(),
            ));
        }

        let kv_stride = num_kv_heads * head_dim;
        let num_head_dims = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_len * num_head_dims];

        for h in 0..num_heads {
            let kvh = (h * num_kv_heads) / num_heads;
            for t in 0..seq_len {
                let q_abs = cache_offset as usize + t;
                // Sliding-window lower bound: a key at t2 is visible iff
                // t2 <= q_abs (causal) and, when window is set, t2 >= window_start.
                let window_start = match window_w {
                    Some(w) => q_abs.saturating_sub(w.saturating_sub(1)),
                    None => 0,
                };
                let mut scores = vec![0.0f32; kv_seq_len];
                for t2 in 0..kv_seq_len {
                    if t2 > q_abs || t2 < window_start {
                        scores[t2] = f32::NEG_INFINITY;
                    } else {
                        let block_idx_in_seq = t2 / page_size;
                        let offset_in_block = t2 % page_size;
                        let block_id = resolve_block(block_idx_in_seq)?;

                        let k_offset =
                            (block_id * page_size + offset_in_block) * kv_stride + kvh * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += qd[t * num_head_dims + h * head_dim + d] * kd[k_offset + d];
                        }
                        scores[t2] = dot * scale;
                    }
                }

                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..kv_seq_len {
                        if scores[t2] > 0.0 {
                            let block_idx_in_seq = t2 / page_size;
                            let offset_in_block = t2 % page_size;
                            let block_id = resolve_block(block_idx_in_seq)?;
                            let v_offset = (block_id * page_size + offset_in_block) * kv_stride
                                + kvh * head_dim;
                            acc += scores[t2] * vd[v_offset + d];
                        }
                    }
                    out[t * num_head_dims + h * head_dim + d] = acc;
                }
            }
        }

        let gpu_out = self.from_cpu(&out, out_shape, DType::F32)?;
        Ok((gpu_out, Box::new(CudaHandle::ready(self.ordinal))))
    }


    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dim = cfg.dim;
        let base = cfg.base;
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rope x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;

        if !cfg.is_plain() {
            // Partial-rotary / YaRN: dispatch `grim_rope_yarn` with a host-built
            // YaRN-ramp inv_freq[] and mscale (attention_factor). Mirrors the ROCm
            // `rope_launch_yarn` path so the two devices agree bit-for-bit.
            //
            // out_shape is [B, S, D]; positions has length S.
            let dims = out_shape.dims();
            if dims.len() != 3 || dims[2] != cfg.dim {
                return Err(Error::Shape(format!(
                    "rope (yarn): expected [B,S,D={}] out_shape, got {:?}",
                    cfg.dim, dims
                )));
            }
            let (b, s, d) = (dims[0], dims[1], dims[2]);
            let rotary_dim = cfg.rotary_dim.min(d);
            let rotary_half = rotary_dim / 2;
            let yarn = cfg.yarn;
            if positions.len() != s {
                return Err(Error::Shape(
                    "rope (yarn): positions length must match seq_len".into(),
                ));
            }

            // Build the YaRN-ramp-corrected inv_freq[] on the host — O(rotary_half)
            // work, negligible vs kernel launch overhead. This avoids per-layer
            // buffers. Identical math to `rope_launch_yarn` on ROCm.
            let inv_freq: Vec<f32> = (0..rotary_half)
                .map(|i| {
                    let freq = 1.0_f32 / base.powf((2 * i) as f32 / d as f32);
                    match yarn {
                        None => freq,
                        Some(y) => {
                            let wavelength = 2.0 * std::f32::consts::PI / freq;
                            let low = y.original_max_pos as f32 / y.beta_slow;
                            let high = y.original_max_pos as f32 / y.beta_fast;
                            if wavelength < high {
                                freq
                            } else if wavelength > low {
                                freq / y.factor
                            } else {
                                let ramp = (y.original_max_pos as f32 / wavelength - y.beta_slow)
                                    / (y.beta_fast - y.beta_slow);
                                (1.0 - ramp) * (freq / y.factor) + ramp * freq
                            }
                        }
                    }
                })
                .collect();
            let mscale = yarn.map(|y| y.attention_factor).unwrap_or(1.0_f32);

            let pos_bytes = positions.len() * 4;
            let freq_bytes = inv_freq.len() * 4;
            let mut pos_dev_ptr: *mut c_void = std::ptr::null_mut();
            let mut freq_dev_ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                let res = cudaMalloc(&mut pos_dev_ptr, pos_bytes);
                if res != cudaSuccess {
                    return Err(Error::Backend(format!(
                        "cudaMalloc for rope yarn pos failed: {}",
                        res
                    )));
                }
                let res = cudaMemcpy(
                    pos_dev_ptr,
                    positions.as_ptr() as *const c_void,
                    pos_bytes,
                    cudaMemcpyHostToDevice,
                );
                if res != cudaSuccess {
                    cudaFree(pos_dev_ptr);
                    return Err(Error::Backend(format!(
                        "cudaMemcpy for rope yarn pos failed: {}",
                        res
                    )));
                }
                let res = cudaMalloc(&mut freq_dev_ptr, freq_bytes);
                if res != cudaSuccess {
                    cudaFree(pos_dev_ptr);
                    return Err(Error::Backend(format!(
                        "cudaMalloc for rope yarn inv_freq failed: {}",
                        res
                    )));
                }
                let res = cudaMemcpy(
                    freq_dev_ptr,
                    inv_freq.as_ptr() as *const c_void,
                    freq_bytes,
                    cudaMemcpyHostToDevice,
                );
                if res != cudaSuccess {
                    cudaFree(pos_dev_ptr);
                    cudaFree(freq_dev_ptr);
                    return Err(Error::Backend(format!(
                        "cudaMemcpy for rope yarn inv_freq failed: {}",
                        res
                    )));
                }
            }

            let mut x_ptr = Self::dev_ptr_or_err("rope yarn x", x_storage)?;
            let mut out_ptr = Self::dev_ptr_or_err("rope yarn out", &out_storage)?;
            let mut pos_ptr = pos_dev_ptr;
            let mut freq_ptr = freq_dev_ptr;
            let mut b_i = b as i32;
            let mut s_i = s as i32;
            let mut d_i = d as i32;
            let mut rh_i = rotary_half as i32;
            let mut ms_f = mscale;

            let mut args = [
                &mut x_ptr as *mut *mut c_void as *mut c_void,
                &mut pos_ptr as *mut *mut c_void as *mut c_void,
                &mut freq_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut b_i as *mut i32 as *mut c_void,
                &mut s_i as *mut i32 as *mut c_void,
                &mut d_i as *mut i32 as *mut c_void,
                &mut rh_i as *mut i32 as *mut c_void,
                &mut ms_f as *mut f32 as *mut c_void,
            ];

            // Grid covers max(b*s*rotary_half, b*s*copy_len) threads so the
            // rotate pass and the verbatim-copy pass both fit in one launch.
            let copy_len = d - 2 * rotary_half;
            let total = b
                * s
                * rotary_half
                    .max(if copy_len > 0 { copy_len } else { 0 })
                    .max(1);
            let handle = self.launch_rank1_kernel("grim_rope_yarn", &mut args, total)?;

            unsafe {
                let _ = cudaDeviceSynchronize();
                cudaFree(pos_dev_ptr);
                cudaFree(freq_dev_ptr);
            }

            return Ok((Box::new(out_storage), handle));
        }

        // Plain full-rotary RoPE (cfg.is_plain()).
        let num_tokens = positions.len();
        let num_heads = out_shape.elem_count() / (num_tokens * dim);
        let head_dim = dim;

        let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let pos_bytes = pos_i32.len() * 4;
        let mut pos_dev_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut pos_dev_ptr, pos_bytes);
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMalloc for rope pos failed: {}",
                    res
                )));
            }
            let res = cudaMemcpy(
                pos_dev_ptr,
                pos_i32.as_ptr() as *const c_void,
                pos_bytes,
                cudaMemcpyHostToDevice,
            );
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMemcpy for rope pos failed: {}",
                    res
                )));
            }
        }

        let mut x_ptr = Self::dev_ptr_or_err("rope x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("rope out", &out_storage)?;
        let mut num_t_i = num_tokens as i32;
        let mut num_h_i = num_heads as i32;
        let mut h_dim_i = head_dim as i32;
        let mut base_f = base;

        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut pos_dev_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut num_t_i as *mut i32 as *mut c_void,
            &mut num_h_i as *mut i32 as *mut c_void,
            &mut h_dim_i as *mut i32 as *mut c_void,
            &mut base_f as *mut f32 as *mut c_void,
        ];

        let total_pairs = num_tokens * num_heads * (head_dim / 2);
        let handle = self.launch_rank1_kernel("grim_rope", &mut args, total_pairs)?;

        unsafe {
            let _ = cudaDeviceSynchronize();
            cudaFree(pos_dev_ptr);
        }

        Ok((Box::new(out_storage), handle))
    }


    fn flash_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        _causal: bool,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let (out_storage, _h) = self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )?;
        let _ = num_heads;
        let _ = head_dim;
        tracing::debug!(
            target: "grim_cuda",
            seq_len = seq_len,
            num_heads = num_heads,
            "CUDA flash_attention → qkv_attention (op=Attention)"
        );
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }


    fn cross_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let (out_storage, _h) = self.qkv_attention(
            q, k, v, num_heads, kv_seq_len, 0, None, out_shape, None, None,
        )?;
        tracing::debug!(
            target: "grim_cuda",
            seq_len = seq_len,
            kv_seq_len = kv_seq_len,
            num_heads = num_heads,
            "CUDA cross_attention → qkv_attention (op=Attention)"
        );
        let _ = head_dim;
        let _ = seq_len;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }

    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )
    }
}

impl FusionOps for CudaDevice {


    /// Override the trait default with the real fused `grim_add_rms_norm` PTX kernel.
    fn fused_add_rms_norm(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        CudaDevice::fused_add_rms_norm(self, x, residual, weight, eps, out_shape)
    }
}

impl AutogradOps for CudaDevice {


    /// Fused (non-fused first cut) dequantized matmul backward on CUDA.
    ///
    /// Computes `dX[M, K] = dY[M, N] @ B_dequant^T` where `B` is a quantized,
    /// CUDA-resident weight of shape `[K, N]`. The packed codes are copied to
    /// the host, dequantized via `grim-quant` (mirroring
    /// `grim-format::convert::dequant_tensor_data`), re-uploaded as F32, and
    /// multiplied with `dY` through cuBLAS. This is the CUDA counterpart of the
    /// ROCm fused path in `RocmDevice::quantized_matmul_backward_dx`; it fires
    /// from `grim-autograd::matmul_backward` once quantized storage is kept
    /// resident on CUDA (see `varbuilder::materialize`).
    fn silu_mul_backward(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let gate_s = gate
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul_backward: gate is not CudaStorage".into()))?;
        let up_s = up
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul_backward: up is not CudaStorage".into()))?;
        let dw_s = dw
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul_backward: dw is not CudaStorage".into()))?;

        Self::ensure_f32_input("silu_mul_backward gate", gate_s)?;
        Self::ensure_f32_input("silu_mul_backward up", up_s)?;
        Self::ensure_f32_input("silu_mul_backward dw", dw_s)?;

        if out_shape.dims() != gate_s.shape().dims() {
            return Err(Error::Shape(format!(
                "silu_mul_backward: out_shape must match gate shape, got {:?} vs {:?}",
                out_shape.dims(),
                gate_s.shape().dims()
            )));
        }

        let n = out_shape.elem_count();
        let mut gate_ptr = Self::dev_ptr_or_err("silu_mul_backward gate", gate_s)?;
        let mut up_ptr = Self::dev_ptr_or_err("silu_mul_backward up", up_s)?;
        let mut dw_ptr = Self::dev_ptr_or_err("silu_mul_backward dw", dw_s)?;

        let df_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let de_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let mut df_ptr = Self::dev_ptr_or_err("silu_mul_backward df", &df_storage)?;
        let mut de_ptr = Self::dev_ptr_or_err("silu_mul_backward de", &de_storage)?;

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut f: CUfunction = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("grim_silu_mul_backward")
            .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
        let res = unsafe {
            cuModuleGetFunction(
                &mut f as *mut *mut c_void as *mut CUfunction,
                module,
                func_name.as_ptr(),
            )
        };
        if res != 0 || f.is_null() {
            return Err(Error::Backend(format!(
                "cuModuleGetFunction(grim_silu_mul_backward) failed: {res}"
            )));
        }

        let mut n_i = n as i32;
        let mut args = [
            &mut gate_ptr as *mut *mut c_void as *mut c_void,
            &mut up_ptr as *mut *mut c_void as *mut c_void,
            &mut dw_ptr as *mut *mut c_void as *mut c_void,
            &mut df_ptr as *mut *mut c_void as *mut c_void,
            &mut de_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];

        let handle = self.launch_rank1_kernel("silu_mul_backward", &mut args, n)?;
        Ok((Box::new(df_storage), Box::new(de_storage), handle))
    }
}

impl OptimizerOps for CudaDevice {
}

impl QuantOps for CudaDevice {


    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // ── GPU fast path: fused Q8_0 kernel (K-aligned, GPU-resident only) ──
        if format == grim_tensor::QuantFormat::Q8_0 {
            let a_storage = a.as_any().downcast_ref::<CudaStorage>();
            let b_storage = b_packed.as_any().downcast_ref::<CudaStorage>();
            if let (Some(a_storage), Some(b_storage)) = (a_storage, b_storage) {
                if k >= 32 && k % 32 == 0 && b_storage.bytes() >= k * n {
                    Self::ensure_f32_input("quantized_matmul a", a_storage)?;

                    let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
                    let a_ptr = Self::dev_ptr_or_err("quantized_matmul a", a_storage)?;
                    let b_ptr = Self::dev_ptr_or_err("quantized_matmul b", b_storage)?;
                    let out_ptr = out_storage.device_ptr.ok_or_else(|| {
                        Error::Backend("quantized_matmul: failed to allocate output buffer".into())
                    })? as *mut c_void;

                    // Q8_0: check if B data uses the real 34-byte packed layout
                    // (f16_scale + 32 i8 codes per block) or the simplified raw-u8 layout.
                    // For real packed data, extract per-block f16 scales from the headers.
                    // For simplified data (k*n raw u8 bytes), use the externally-provided
                    // b_scales directly.
                    let blocks_per_col = k / 32;
                    let scale_len = n * blocks_per_col;

                    // Real Q8_0 packed: n * blocks_per_col * 34 bytes
                    // Simplified: k * n bytes (used in tests/legacy path)
                    let b_bytes = b_storage.bytes();
                    let real_packed_size = n * blocks_per_col * 34;

                    let (_scales_storage, scales_device_ptr, b_data_offset) = if b_bytes
                        == real_packed_size
                    {
                        // Real packed: extract f16 scales from headers.
                        let mut host_packed = vec![0u8; b_bytes];
                        if let Some(dev_ptr) = b_storage.device_ptr {
                            unsafe {
                                let res = cudaMemcpy(
                                    host_packed.as_mut_ptr() as *mut c_void,
                                    dev_ptr as *const c_void,
                                    b_bytes,
                                    cudaMemcpyDeviceToHost,
                                );
                                if res != 0 {
                                    return Err(Error::Backend(format!(
                                        "quantized_matmul(Q8_0): cudaMemcpy scales D2H failed: {res}"
                                    )));
                                }
                            }
                        }
                        let mut scales_host = vec![1.0f32; scale_len];
                        for col in 0..n {
                            for block in 0..blocks_per_col {
                                let block_offset = (col * blocks_per_col + block) * 34;
                                let scale = half::f16::from_le_bytes([
                                    host_packed[block_offset],
                                    host_packed[block_offset + 1],
                                ])
                                .to_f32();
                                scales_host[col * blocks_per_col + block] = scale;
                            }
                        }
                        let scales_storage = CudaStorage::copy_from_host(
                            &scales_host,
                            &Shape::new(vec![scale_len]),
                            DType::F32,
                            self.ordinal,
                        )?;
                        let sptr = scales_storage.device_ptr.ok_or_else(|| {
                            Error::Backend(
                                "quantized_matmul: failed to upload scales buffer".into(),
                            )
                        })? as *const c_void;
                        (Some(scales_storage), sptr, 2usize) // kernel skips 2-byte f16 header per block
                    } else {
                        // Simplified: use externally-provided b_scales directly.
                        // The kernel still applies the B data at offset 0 (no f16 header skip).
                        let default_scales: Vec<f32> = vec![1.0f32; scale_len];
                        let scales_storage = if b_scales.is_empty() {
                            CudaStorage::copy_from_host(
                                &default_scales,
                                &Shape::new(vec![scale_len]),
                                DType::F32,
                                self.ordinal,
                            )?
                        } else if b_scales.len() == scale_len {
                            CudaStorage::copy_from_host(
                                &b_scales,
                                &Shape::new(vec![scale_len]),
                                DType::F32,
                                self.ordinal,
                            )?
                        } else {
                            return Err(Error::Shape(format!(
                                "quantized_matmul(Q8_0): b_scales length {} != expected {}",
                                b_scales.len(),
                                scale_len
                            )));
                        };
                        let sptr = scales_storage.device_ptr.ok_or_else(|| {
                            Error::Backend(
                                "quantized_matmul: failed to upload scales buffer".into(),
                            )
                        })? as *const c_void;
                        (Some(scales_storage), sptr, 0usize) // no f16 header to skip
                    };

                    let handle = self.launch_quantized_matmul_q8_0(
                        a_ptr,
                        b_ptr,
                        scales_device_ptr,
                        out_ptr,
                        m,
                        n,
                        k,
                        b_data_offset,
                    )?;
                    return Ok((Box::new(out_storage), handle));
                }
            }
        }

        // ── CPU fallback: format-accurate dequantization via grim_quant ──────
        // Dispatches on `format` so every supported variant uses its canonical
        // bit-unpacking algorithm. Non-supported formats return Err immediately
        // rather than producing silently wrong output.
        tracing::warn!(
            "CUDA quantized_matmul: falling back to CPU for format {format:?} \
             (m={m}, k={k}, n={n})"
        );

        let a_vec = a.to_cpu_vec_f32()?;

        // Download packed B bytes from GPU if resident.
        let b_bytes: Vec<u8> = if let Some(cs) = b_packed.as_any().downcast_ref::<CudaStorage>() {
            let mut host_bytes = vec![0u8; cs.bytes()];
            if let Some(dev_ptr) = cs.device_ptr {
                unsafe {
                    let res = cudaMemcpy(
                        host_bytes.as_mut_ptr() as *mut c_void,
                        dev_ptr as *const c_void,
                        cs.bytes(),
                        cudaMemcpyDeviceToHost,
                    );
                    if res != 0 {
                        return Err(Error::Backend(format!(
                            "quantized_matmul: cudaMemcpy(B) D2H failed: {res}"
                        )));
                    }
                }
            }
            host_bytes
        } else {
            vec![0u8; k * n]
        };

        // Dequantize B using the canonical grim_quant function for `format`.
        // CONTRACT: every arm must produce exactly `k * n` f32 values in
        // row-major order matching the B[K, N] layout expected by the GEMM below.
        let b_dequant: Vec<f32> = match format {
            grim_tensor::QuantFormat::Q8_0 => {
                // Layout detection: if b_scales is non-empty with the correct length,
                // use the simplified layout (raw u8 at 32-byte stride with external scales).
                // Otherwise, check if the byte layout matches real packed Q8_0 (34-byte blocks
                // with embedded f16 scales).
                let blocks_per_col = k / 32;
                let real_packed_size = n * blocks_per_col * 34;
                let use_simplified = !b_scales.is_empty() && b_scales.len() == n * blocks_per_col;
                let mut out = vec![0.0f32; k * n];
                if use_simplified {
                    // Simplified layout: raw u8 bytes with 32-byte block stride,
                    // scales provided externally via b_scales
                    for col in 0..n {
                        for block in 0..blocks_per_col {
                            let block_offset = (col * blocks_per_col + block) * 32;
                            let scale = b_scales
                                .get(col * blocks_per_col + block)
                                .copied()
                                .unwrap_or(1.0f32);
                            for i in 0..32 {
                                let byte_offset = block_offset + i;
                                let q_val = b_bytes
                                    .get(byte_offset)
                                    .map(|&b| (b as i8) as f32)
                                    .unwrap_or(0.0f32);
                                let r = block * 32 + i;
                                if r < k {
                                    out[r * n + col] = q_val * scale;
                                }
                            }
                        }
                    }
                } else if b_bytes.len() == real_packed_size {
                    // Real Q8_0 packed layout: extract scale from f16 header
                    for col in 0..n {
                        for block in 0..blocks_per_col {
                            let block_offset = (col * blocks_per_col + block) * 34;
                            let scale_bytes = &b_bytes[block_offset..block_offset + 2];
                            let scale =
                                half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
                            for i in 0..32 {
                                let byte_offset = block_offset + 2 + i;
                                let q_val = b_bytes
                                    .get(byte_offset)
                                    .map(|&b| (b as i8) as f32)
                                    .unwrap_or(0.0f32);
                                let r = block * 32 + i;
                                if r < k {
                                    out[r * n + col] = q_val * scale;
                                }
                            }
                        }
                    }
                } else {
                    // Unknown layout: treat as simplified with default scale 1.0
                    for col in 0..n {
                        for block in 0..blocks_per_col {
                            let block_offset = (col * blocks_per_col + block) * 32;
                            for i in 0..32 {
                                let byte_offset = block_offset + i;
                                let q_val = b_bytes
                                    .get(byte_offset)
                                    .map(|&b| (b as i8) as f32)
                                    .unwrap_or(0.0f32);
                                let r = block * 32 + i;
                                if r < k {
                                    out[r * n + col] = q_val;
                                }
                            }
                        }
                    }
                }
                out
            }
            grim_tensor::QuantFormat::Q4K => grim_quant::dequant_q4k(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul Q4K dequant: {e}")))?,
            grim_tensor::QuantFormat::Q5K => grim_quant::dequant_q5k(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul Q5K dequant: {e}")))?,
            grim_tensor::QuantFormat::Q6K => grim_quant::dequant_q6k(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul Q6K dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq4Nl => grim_quant::dequant_iq4nl(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ4NL dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq4Xs => grim_quant::dequant_iq4xs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ4XS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq3Xxs => grim_quant::dequant_iq3xxs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ3XXS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq3S => grim_quant::dequant_iq3s(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ3S dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq2Xxs => grim_quant::dequant_iq2xxs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ2XXS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq2Xs => grim_quant::dequant_iq2xs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ2XS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq2S => grim_quant::dequant_iq2s(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ2S dequant: {e}")))?,
            grim_tensor::QuantFormat::Fp4 => grim_quant::dequant_fp4(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul FP4 dequant: {e}")))?,
            grim_tensor::QuantFormat::Fp4Block16 => {
                grim_quant::dequant_fp4_block16(&b_bytes, k * n).map_err(|e| {
                    Error::Backend(format!("quantized_matmul FP4Block16 dequant: {e}"))
                })?
            }
            unsupported => {
                return Err(Error::Backend(format!(
                    "CUDA quantized_matmul: no GPU kernel or CPU dequant path \
                     for format {unsupported:?}"
                )));
            }
        };

        // GEMM: C[M, N] = A[M, K] · B_dequant[K, N]
        let mut c_vec = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_vec[row * k + p] * b_dequant[p * n + col];
                }
                c_vec[row * n + col] = sum;
            }
        }

        let out_storage = self.from_cpu(&c_vec, out_shape, a.dtype())?;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }


    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        _residuals: Option<&grim_tensor::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dy_storage = dy.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("quantized_matmul_backward_dx: dy not CudaStorage".into())
        })?;

        let b_storage = b_packed
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("quantized_matmul_backward_dx: b_packed not CudaStorage".into())
            })?;

        // Validate the output shape contract (dX is [M, K]).
        if out_shape.dims() != [m, k] {
            return Err(Error::Shape(format!(
                "quantized_matmul_backward_dx: out_shape must be [{m},{k}], got {:?}",
                out_shape.dims()
            )));
        }

        // dy must be F32 to feed cuBLAS.
        if dy_storage.dtype.arith != ArithType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "quantized_matmul_backward_dx: dy must be F32, got {:?}",
                dy_storage.dtype
            )));
        }

        // B is stored as [K, N] row-major in the packed payload.
        let b_elem_count = k * n;
        let b_bytes = b_storage.copy_to_host_raw_bytes()?;
        let b_scales = b_storage.quant_scales();

        // Host dequant: packed bytes -> F32 [K, N] row-major, via grim-quant.
        let b_dequant =
            cuda_dequant_quantized_storage(&b_bytes, b_scales, b_elem_count, &b_storage.dtype)?;

        // Re-upload B as F32, then compute dX = dY @ B^T directly via cuBLAS.
        //
        // The generic `matmul` cannot be reused here: its column-major
        // transpose trick assumes the forward orientation (inner dim K is the
        // leading dimension of B), which only holds when m == n == k and breaks
        // for the backward shape dY[M,N] @ B^T[N,K] -> dX[M,K].
        //
        // Correct row-major GEMM for dX[M,K] = dY[M,N] @ B^T[N,K]:
        //   C_col(K,M) = B_col(K,N) * A_col(N,M)
        // where B^T row-major [N,K] read column-major is B [K,N] (lda = K), and
        // dY row-major [M,N] read column-major is dY^T [N,M] (ldb = N).
        let b_shape = b_storage.shape().clone();
        let (b_rows, b_cols) = (b_shape.dims()[0], b_shape.dims()[1]);
        let mut b_t = vec![0.0f32; b_elem_count];
        for r in 0..b_rows {
            for c in 0..b_cols {
                b_t[c * b_rows + r] = b_dequant[r * b_cols + c];
            }
        }
        let b_t_shape = Shape::new(vec![b_cols, b_rows]);
        let b_t_storage = CoreTensorOps::from_cpu(
            self,
            &b_t,
            &b_t_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
        )?;

        let dx_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;

        let handle = self.get_cublas_handle()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let b_t_ptr = b_t_storage
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("quantized_matmul_backward_dx: b_t not CudaStorage".into())
            })?
            .device_ptr
            .ok_or_else(|| {
                Error::Backend("quantized_matmul_backward_dx: b_t has no device pointer".into())
            })? as *const c_void;
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("quantized_matmul_backward_dx: dY has no device pointer".into())
        })? as *const c_void;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("quantized_matmul_backward_dx: dX has no device pointer".into())
        })? as *mut c_void;

        // SAFETY: all pointers are freshly allocated/uploaded on this device;
        // leading dims are the row counts of the column-major views, all >= 1.
        unsafe {
            let status = cublasSgemm_v2(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                k as i32, // m_cublas = rows of B_col = K
                m as i32, // n_cublas = cols of A_col = M
                n as i32, // k_cublas = inner N
                &alpha,
                b_t_ptr as *const f32, // B^T [N,K] row-major, read col-major as B [K,N]
                k as i32,              // lda = K
                dy_ptr as *const f32,  // dY [M,N] row-major, read col-major as dY^T [N,M]
                n as i32,              // ldb = N
                &beta,
                dx_ptr as *mut f32, // dX [M,K] row-major, read col-major as dX^T [K,M]
                k as i32,           // ldc = K
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(Error::Backend(format!(
                    "cublasSgemm_v2 (backward dx) failed with status {}",
                    status
                )));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(true)),
        });

        Ok((Box::new(dx_storage), compute_handle))
    }


    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("quantize: x is not CudaStorage".into()))?;
        let out = self.quantize_on_device(x_storage, format)?;
        Ok(Box::new(out))
    }


    fn fused_quant_gemm(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let kernel_name = match format {
            grim_tensor::QuantFormat::Q8_0 => "grim_fused_quant_gemm_q8_0_packed",
            grim_tensor::QuantFormat::Q4K => "grim_fused_quant_gemm_q4_k",
            grim_tensor::QuantFormat::Q5K => "grim_fused_quant_gemm_q5_k",
            grim_tensor::QuantFormat::Q6K => "grim_fused_quant_gemm_q6_k",
            grim_tensor::QuantFormat::Iq4Nl => "grim_fused_quant_gemm_iq4nl",
            grim_tensor::QuantFormat::Iq4Xs => "grim_fused_quant_gemm_iq4xs",
            grim_tensor::QuantFormat::Fp4 => "grim_fused_quant_gemm_mxfp4",
            grim_tensor::QuantFormat::Fp4Block16 => "grim_fused_quant_gemm_nvfp4",
            grim_tensor::QuantFormat::Fp8 => {
                // T1 caps gate: without native FP8 (compute < 8.9), don't select the fp8 shader.
                if !self
                    .caps
                    .supports_quant_format(grim_tensor::QuantFormat::Fp8)
                {
                    return Err(Error::Backend(
                        "fused_quant_gemm: FP8 not supported on this device".into(),
                    ));
                }
                "grim_fused_quant_gemm_fp8"
            }
            other => {
                return Err(Error::Unimplemented(format!(
                    "fused_quant_gemm: no GPU kernel for format {other:?}"
                )));
            }
        };

        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_quant_gemm: a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_quant_gemm: b is not CudaStorage".into()))?;

        Self::ensure_f32_input("fused_quant_gemm a", a_storage)?;

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let out_dims = out_shape.dims();
        if a_dims.len() != 2 || b_dims.len() != 2 || out_dims.len() != 2 {
            return Err(Error::Shape(
                "fused_quant_gemm expects 2-D a, b, out".into(),
            ));
        }
        let (m, k) = (a_dims[0], a_dims[1]);
        // GGUF packed weight b is [out_dim, in_dim] = [N, K].
        // In fused GEMM (A @ B^T), K matches b_dims[1] if b is untransposed [N, K],
        // or b_dims[0] if b_dims was transposed to [K, N].
        let (n, k2) = if b_dims[0] == k {
            (b_dims[1], b_dims[0])
        } else {
            (b_dims[0], b_dims[1])
        };
        if k != k2 {
            return Err(Error::Shape(format!(
                "fused_quant_gemm: a is ({m},{k}) but b is ({n},{k2})"
            )));
        }
        if format == grim_tensor::QuantFormat::Q8_0 && k % 32 != 0 {
            return Err(Error::Shape(format!(
                "fused_quant_gemm(Q8_0): K ({k}) must be a multiple of 32"
            )));
        }

        let a_ptr = Self::dev_ptr_or_err("fused_quant_gemm a", a_storage)?;
        let out = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let out_ptr = Self::dev_ptr_or_err("fused_quant_gemm out", &out)?;

        // For Q8_0, pass the packed unsigned char* directly — the new kernel
        // (grim_fused_quant_gemm_q8_0_packed) reads f16 scales and i8 codes
        // from the 34-byte blocks on-device.
        let b_ptr = if format == grim_tensor::QuantFormat::Q8_0 {
            let b_storage = b
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| Error::Backend("fused_quant_gemm: b is not CudaStorage".into()))?;
            let b_bytes = b_storage.bytes();
            let blocks_per_col = k / 32;
            let expected_packed = n * blocks_per_col * 34;
            if b_bytes != expected_packed {
                return Err(Error::Shape(format!(
                    "fused_quant_gemm(Q8_0): B packed size {} != expected {}",
                    b_bytes, expected_packed
                )));
            }
            Self::dev_ptr_or_err("fused_quant_gemm b (packed)", b_storage)?
        } else {
            Self::dev_ptr_or_err("fused_quant_gemm b", b_storage)?
        };

        let handle = self.launch_fused_quant_gemm(kernel_name, a_ptr, b_ptr, out_ptr, m, n, k)?;
        Ok((Box::new(out), handle))
    }
}

impl RecurrentOps for CudaDevice {


    fn selective_scan(
        &self,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        state: &dyn BackendStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        tracing::warn!("CUDA selective_scan: falling back to CPU execution");
        let x_v = x.to_cpu_vec_f32()?;
        let a_v = a.to_cpu_vec_f32()?;
        let b_v = b.to_cpu_vec_f32()?;
        let c_v = c.to_cpu_vec_f32()?;
        let d_v = d.to_cpu_vec_f32()?;
        let state_v = state.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim_dinner];
        for b_idx in 0..batch {
            for d_idx in 0..dim_dinner {
                // Initialize state from the provided state buffer.
                let mut h = vec![0.0f32; dim_dstate];
                for s in 0..dim_dstate {
                    let state_idx = (b_idx * dim_dinner + d_idx) * dim_dstate + s;
                    h[s] = if state_v.len() > state_idx {
                        state_v[state_idx]
                    } else {
                        0.0
                    };
                }
                let d_val = if d_v.len() > d_idx { d_v[d_idx] } else { 0.0 };

                for t in 0..seq_len {
                    let x_idx = (b_idx * seq_len + t) * dim_dinner + d_idx;
                    let x_t = x_v[x_idx];
                    let mut y_t = d_val * x_t;

                    for s in 0..dim_dstate {
                        let a_idx = d_idx * dim_dstate + s;
                        let b_idx_off = (b_idx * seq_len + t) * dim_dstate + s;
                        let c_idx_off = (b_idx * seq_len + t) * dim_dstate + s;

                        let a_val = if a_v.len() > a_idx { a_v[a_idx] } else { 1.0 };
                        let b_val = if b_v.len() > b_idx_off {
                            b_v[b_idx_off]
                        } else {
                            1.0
                        };
                        let c_val = if c_v.len() > c_idx_off {
                            c_v[c_idx_off]
                        } else {
                            1.0
                        };

                        h[s] = a_val * h[s] + x_t * b_val;
                        y_t += c_val * h[s];
                    }
                    out[x_idx] = y_t;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }


    fn rwkv_time_mix(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        g: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        tracing::warn!("CUDA rwkv_time_mix: falling back to CPU execution");
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let w_vec = w.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim];
        for b in 0..batch {
            for d in 0..dim {
                let mut state = 0.0f32;
                let w_val = if w_vec.len() > d { w_vec[d] } else { 0.9f32 };

                for t in 0..seq_len {
                    let idx = (b * seq_len + t) * dim + d;
                    let k_t = if k_vec.len() > idx {
                        k_vec[idx]
                    } else {
                        x_vec[idx]
                    };
                    let v_t = if v_vec.len() > idx {
                        v_vec[idx]
                    } else {
                        x_vec[idx]
                    };
                    let g_t = if g_vec.len() > idx {
                        g_vec[idx]
                    } else {
                        1.0f32
                    };

                    state = w_val * state + k_t * v_t;
                    let sig = 1.0f32 / (1.0f32 + (-g_t).exp());
                    out[idx] = state * sig;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }


    fn rwkv_channel_mix(
        &self,
        x: &dyn BackendStorage,
        k: &dyn BackendStorage,
        r: &dyn BackendStorage,
        v: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        tracing::warn!("CUDA rwkv_channel_mix: falling back to CPU execution");
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let r_vec = r.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;

        let elem_count = out_shape.elem_count();
        let mut out = vec![0.0f32; elem_count];
        for i in 0..elem_count {
            let x_val = x_vec[i];
            let k_val = if k_vec.len() > i { k_vec[i] } else { x_val };
            let r_val = if r_vec.len() > i { r_vec[i] } else { 1.0f32 };
            let v_val = if v_vec.len() > i { v_vec[i] } else { x_val };

            let sig_r = 1.0f32 / (1.0f32 + (-r_val).exp());
            let relu_k = k_val.max(0.0f32);
            out[i] = sig_r * (relu_k * relu_k) * v_val;
        }

        let _ = batch;
        let _ = dim;

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }

    /// Depthwise 1D causal convolution step on CUDA GPU.
    ///
    /// Executes the causal convolution step kernel against resident GPU state buffers.
    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        conv_state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("short_conv1d: x is not CudaStorage".into()))?;
        let w_s = weight
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("short_conv1d: weight is not CudaStorage".into()))?;
        let st_s = conv_state
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("short_conv1d: conv_state is not CudaStorage".into()))?;

        Self::ensure_f32_input("short_conv1d x", x_s)?;
        Self::ensure_f32_input("short_conv1d weight", w_s)?;
        Self::ensure_f32_input("short_conv1d conv_state", st_s)?;

        let mut x_ptr = Self::dev_ptr_or_err("short_conv1d x", x_s)?;
        let mut w_ptr = Self::dev_ptr_or_err("short_conv1d weight", w_s)?;
        let mut b_ptr = match bias {
            Some(b) => {
                let b_s = b
                    .as_any()
                    .downcast_ref::<CudaStorage>()
                    .ok_or_else(|| Error::Backend("short_conv1d: bias is not CudaStorage".into()))?;
                Self::ensure_f32_input("short_conv1d bias", b_s)?;
                Self::dev_ptr_or_err("short_conv1d bias", b_s)?
            }
            None => std::ptr::null_mut(),
        };
        let mut st_ptr = Self::dev_ptr_or_err("short_conv1d conv_state", st_s)?;

        let out = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let mut out_ptr = Self::dev_ptr_or_err("short_conv1d out", &out)?;

        let dims = out_shape.dims();
        let mut batch = dims[0] as i32;
        let mut channels = *dims.last().unwrap_or(&1) as i32;
        let mut k_size = (w_s.bytes() / (channels as usize * 4)) as i32;
        let total = (batch * channels) as usize;

        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut w_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut st_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut batch as *mut i32 as *mut c_void,
            &mut channels as *mut i32 as *mut c_void,
            &mut k_size as *mut i32 as *mut c_void,
        ];

        let handle = self.launch_rank1_kernel("grim_short_conv1d_causal_step", &mut args, total)?;
        Ok((Box::new(out), handle))
    }
}

impl CollectiveOps for CudaDevice {


    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        _placement: &grim_tensor::backend::ScythePlacement,
    ) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = match dtype.arith {
            ArithType::F16 | ArithType::BF16 => 150.0,
            ArithType::F32 => 75.0,
            _ => 40.0,
        };
        (flops / (tflops * 1e12) * 1000.0).max(0.01)
    }
}

impl MemoryOps for CudaDevice {


    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        // Packed quantized storage (KQuant, FloatPack, Block, GroupInt) is
        // smaller than `elem_count * arith.byte_size`; allocate the exact byte
        // length so `CudaStorage::bytes()` reflects the real packed payload.
        // For Native storage `data.len()` already equals `elem_count * byte_size`,
        // so this remains correct for both cases.
        let storage = CudaStorage::copy_from_host_raw_bytes(data, shape, dtype, self.ordinal)?;
        let dev_ptr = storage.device_ptr.ok_or_else(|| {
            Error::Backend("from_cpu_bytes: device_ptr is null after raw byte alloc".into())
        })? as *mut c_void;

        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                data.as_ptr() as *const c_void,
                data.len(),
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpy from_cpu_bytes failed: {}",
                res
            )));
        }

        Ok(Box::new(storage))
    }
}

impl GraphCaptureOps for CudaDevice {
}

/// Implement umbrella `BackendDevice` trait for `CudaDevice`.
///
/// Ties together all granular sub-traits to allow `Arc<dyn BackendDevice>` dispatch across the engine.
impl grim_tensor::BackendDevice for CudaDevice {}


impl CudaDevice {
    /// Dequantize Q8_0 packed bytes to an f32 host Vec via host / GPU.
    pub fn dequantize_q8_0_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::KQuant(KQuantScheme::Q80),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        let mut out = Vec::with_capacity(elem_count);
        for blk in bytes.chunks_exact(34) {
            let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            for &q in &blk[2..34] {
                out.push(d * (q as i8 as f32));
            }
        }
        out.truncate(elem_count);
        Ok(out)
    }

    /// Dequantize Q4_K packed bytes to an f32 host Vec via host / GPU.
    pub fn dequantize_q4k_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::KQuant(KQuantScheme::Q4K),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_q4k(bytes, elem_count)
    }

    pub(crate) fn dequantize_iq_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        scheme: KQuantScheme,
    ) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::KQuant(scheme),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        match scheme {
            KQuantScheme::IQ2XXS => grim_quant::dequant_iq2xxs(bytes, elem_count),
            KQuantScheme::IQ2XS => grim_quant::dequant_iq2xs(bytes, elem_count),
            KQuantScheme::IQ2S => grim_quant::dequant_iq2s(bytes, elem_count),
            KQuantScheme::IQ3XXS => grim_quant::dequant_iq3xxs(bytes, elem_count),
            KQuantScheme::IQ3S => grim_quant::dequant_iq3s(bytes, elem_count),
            KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(bytes, elem_count),
            KQuantScheme::IQ4XS => grim_quant::dequant_iq4xs(bytes, elem_count),
            _ => Err(Error::Backend(format!("Unknown iq scheme {:?}", scheme))),
        }
    }

    pub fn dequantize_iq2xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ2XXS)
    }
    pub fn dequantize_iq2xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ2XS)
    }
    pub fn dequantize_iq2s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ2S)
    }
    pub fn dequantize_iq3xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ3XXS)
    }
    pub fn dequantize_iq3s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ3S)
    }
    pub fn dequantize_iq4nl_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ4NL)
    }
    pub fn dequantize_iq4xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ4XS)
    }

    /// Dequantize FP8 packed bytes.
    pub fn dequantize_fp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_fp8(bytes, elem_count)
    }

    /// Dequantize MXFP4 packed bytes.
    pub fn dequantize_mxfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::FloatPack(FloatPackScheme::MxFp4),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_mxfp4(bytes, elem_count)
    }

    /// Dequantize MXFP8 packed bytes.
    pub fn dequantize_mxfp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::FloatPack(FloatPackScheme::MxFp8),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_mxfp8(bytes, elem_count)
    }
}

impl grim_format::convert::GpuDequant for CudaDevice {
    fn dequantize(
        &self,
        storage: &grim_tensor::dtype::Storage,
        bytes: &[u8],
        elem_count: usize,
    ) -> grim_tensor::error::Result<Option<Vec<f32>>> {
        match storage {
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80) => {
                Ok(Some(self.dequantize_q8_0_host(bytes, elem_count)?))
            }
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K) => {
                Ok(Some(self.dequantize_q4k_host(bytes, elem_count)?))
            }
            _ => Ok(None),
        }
    }
}

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

