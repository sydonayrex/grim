pub mod autotune;
pub mod caps;
pub mod kernels;

pub use autotune::{CudaAutotuner, CudaTileConfig, GemmOp, ShapeClass};
pub use caps::CudaCaps;

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::process::Command;
use std::sync::LazyLock;
use std::sync::{Arc, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{
    ArithType, BlockDtype, DType, FloatPackScheme, KQuantScheme, QuantProvenance,
    Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

// ---------- CUDA FFI ----------

#[allow(non_upper_case_globals)]
pub const cudaSuccess: i32 = 0;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToDevice: i32 = 1;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToHost: i32 = 2;

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
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, size: usize) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;

    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    fn cublasSgemm_v2(
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

    fn cuInit(flags: u32) -> i32;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> i32;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> i32;
    fn cuLaunchKernel(
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
}

#[derive(Debug, Clone, Copy)]
pub struct SendCmodule(pub CUmodule);

// SAFETY: `CUmodule` is owned and managed by the CUDA driver.  Concurrent
// `cuModuleLoadData` / `cuLaunchKernel` calls on the same module are
// serialized by the driver; the JIT cache (`JIT_CACHE`) is additionally
// protected by a `Mutex`.  `Send` is safe because the driver tracks the
// module independently of the creating thread.  `Sync` is safe because
// the driver itself serializes concurrent launches on the same module.
unsafe impl Send for SendCmodule {}
unsafe impl Sync for SendCmodule {}

static JIT_CACHE: LazyLock<Mutex<HashMap<u64, SendCmodule>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn compile_and_load_kernel(src: &str, device_ordinal: usize) -> Result<CUmodule> {
    let hash = seahash::hash(src.as_bytes());
    let mut cache = JIT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&module) = cache.get(&hash) {
        return Ok(module.0);
    }

    // SAFETY: `cuInit(0)` initializes the CUDA driver API. It is a no-op if
    // already initialized, and must be called before any other driver API call.
    unsafe {
        let res = cuInit(0);
        if res != 0 {
            return Err(Error::Backend(format!("cuInit failed with status {}", res)));
        }
    }

    let cache_dir = std::env::var("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default().join("target"))
        .join("grim_cuda_cache");
    fs::create_dir_all(&cache_dir).ok();

    let cu_path = cache_dir.join(format!("{}.cu", hash));
    let ptx_path = cache_dir.join(format!("{}.ptx", hash));

    fs::write(&cu_path, src)
        .map_err(|e| Error::Backend(format!("Failed to write CUDA source: {e}")))?;

    /// Resolves the path to the `nvcc` executable.
    /// Checks `NVCC`, `CUDA_PATH`, system `PATH`, Arch Linux `/opt/cuda/bin/nvcc`,
    /// and standard Linux `/usr/local/cuda/bin/nvcc`.
    fn resolve_nvcc_path() -> std::path::PathBuf {
        if let Ok(env_nvcc) = std::env::var("NVCC") {
            let p = std::path::PathBuf::from(env_nvcc);
            if p.exists() {
                return p;
            }
        }
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let p = std::path::PathBuf::from(cuda_path).join("bin").join("nvcc");
            if p.exists() {
                return p;
            }
        }

        // Common installation paths across Linux distros (including Arch Linux /opt/cuda).
        let candidate_paths = [
            "/opt/cuda/bin/nvcc",
            "/usr/local/cuda/bin/nvcc",
            "/usr/bin/nvcc",
        ];

        for path_str in candidate_paths {
            let p = std::path::PathBuf::from(path_str);
            if p.exists() {
                return p;
            }
        }

        // Fall back to relying on PATH resolution by executable name.
        std::path::PathBuf::from("nvcc")
    }

    let nvcc = resolve_nvcc_path();

    let status = Command::new(&nvcc)
        .arg("-ptx")
        .arg("-O3")
        .arg("--gpu-architecture=sm_80")
        .arg(&cu_path)
        .arg("-o")
        .arg(&ptx_path)
        .status();

    let success = match status {
        Ok(s) => s.success(),
        Err(_) => false,
    };

    if !success {
        let status2 = Command::new(&nvcc)
            .arg("-ptx")
            .arg("-O3")
            .arg(&cu_path)
            .arg("-o")
            .arg(&ptx_path)
            .status();
        let success2 = match status2 {
            Ok(s) => s.success(),
            Err(_) => false,
        };
        if !success2 {
            return Err(Error::Backend(format!(
                "nvcc compilation failed (using compiler at {:?})",
                nvcc
            )));
        }
    }

    let ptx_content = fs::read_to_string(&ptx_path)
        .map_err(|e| Error::Backend(format!("Failed to read compiled PTX: {e}")))?;

    let mut module: CUmodule = std::ptr::null_mut();
    // SAFETY: `cudaSetDevice` selects the CUDA device for the current thread.
    // The `device_ordinal` was validated at construction time; `cuModuleLoadData`
    // loads PTX into that device's context.
    unsafe {
        let select_res = cudaSetDevice(device_ordinal as i32);
        if select_res != 0 {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed: {}",
                select_res
            )));
        }

        let mut ptx_bytes = ptx_content.into_bytes();
        ptx_bytes.push(0); // Null-terminate the PTX string!
        let load_res = cuModuleLoadData(&mut module, ptx_bytes.as_ptr() as *const c_void);
        if load_res != 0 {
            return Err(Error::Backend(format!(
                "cuModuleLoadData failed with error {}",
                load_res
            )));
        }
    }

    cache.insert(hash, SendCmodule(module));
    Ok(module)
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

/// CUDA tensor storage.
#[derive(Debug)]
pub struct CudaStorage {
    device_ptr: Option<u64>,
    bytes: usize,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    ordinal: usize,
}

impl CudaStorage {
    /// Allocates GPU memory sized to exactly `byte_len` bytes (for packed
    /// quantized representations whose packed length is smaller than
    /// `shape.elem_count() * dtype.arith.byte_size()`).
    pub fn alloc_gpu_bytes(
        shape: &Shape,
        dtype: DType,
        byte_len: usize,
        device_ordinal: usize,
    ) -> Result<Self> {
        let select_res = unsafe { cudaSetDevice(device_ordinal as i32) };
        if select_res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                device_ordinal
            )));
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let mut res = unsafe { cudaMalloc(&mut dev_ptr, byte_len) };
        if res != cudaSuccess {
            unsafe {
                let _ = cudaDeviceSynchronize();
            }
            res = unsafe { cudaMalloc(&mut dev_ptr, byte_len) };
        }
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMalloc failed to allocate {} bytes with error {}",
                byte_len, res
            )));
        }

        Ok(Self {
            device_ptr: Some(dev_ptr as u64),
            bytes: byte_len,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal: device_ordinal,
        })
    }

    /// Allocates GPU memory on a CUDA device.
    pub fn alloc_gpu(shape: &Shape, dtype: DType, device_ordinal: usize) -> Result<Self> {
        let bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype))
            .ok_or_else(|| {
                Error::Backend(format!(
                    "alloc_gpu: byte count overflow for shape {:?} dtype {:?}",
                    shape, dtype
                ))
            })?;

        // SAFETY: `cudaSetDevice` sets the active device for the current thread.
        // `device_ordinal` is validated at construction; this is a pure device switch.
        let select_res = unsafe { cudaSetDevice(device_ordinal as i32) };
        if select_res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                device_ordinal
            )));
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let mut res = unsafe { cudaMalloc(&mut dev_ptr, bytes) };
        if res != cudaSuccess {
            unsafe {
                let _ = cudaDeviceSynchronize();
            }
            res = unsafe { cudaMalloc(&mut dev_ptr, bytes) };
        }
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMalloc failed to allocate {} bytes with error {}",
                bytes, res
            )));
        }

        Ok(Self {
            device_ptr: Some(dev_ptr as u64),
            bytes,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal: device_ordinal,
        })
    }

    /// Zeroes the backing GPU buffer. `cudaMalloc` does not initialize memory,
    /// so callers that atomicAdd into otherwise-uninitialized output (e.g. the
    /// MoE fused-dispatch kernel) must clear it first.
    pub fn fill_zeroes(&self) -> Result<()> {
        let dev_ptr = match self.device_ptr {
            Some(p) => p as *mut c_void,
            None => return Ok(()),
        };
        if unsafe { cudaSetDevice(self.ordinal as i32) } != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                self.ordinal
            )));
        }
        // SAFETY: `cudaMemset` fills `self.bytes` bytes with 0. All callers
        // allocate f32 buffers whose length is a whole number of bytes.
        let res = unsafe { cudaMemset(dev_ptr, 0, self.bytes) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemset failed to zero {} bytes (err {})",
                self.bytes, res
            )));
        }
        Ok(())
    }

    /// Copies host data to GPU via cudaMemcpy.
    pub fn copy_from_host(
        host_data: &[f32],
        shape: &Shape,
        dtype: DType,
        device_ordinal: usize,
    ) -> Result<Self> {
        let storage = Self::alloc_gpu(shape, dtype, device_ordinal)?;
        let dev_ptr = storage.device_ptr.unwrap() as *mut c_void;

        // SAFETY: `cudaMemcpy` copies `storage.bytes` from host to device.
        // `dev_ptr` was allocated by `cudaMalloc` in `alloc_gpu`; `host_data`
        // is a valid host vector; the direction flag matches the copy direction.
        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                host_data.as_ptr() as *const c_void,
                storage.bytes,
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            // SAFETY: free the allocated buffer on upload failure to avoid a leak.
            unsafe {
                let _ = cudaFree(dev_ptr);
            }
            return Err(Error::Backend(format!(
                "cudaMemcpyHostToDevice failed with error code {}",
                res
            )));
        }

        Ok(storage)
    }

    /// Copies raw packed bytes (e.g. Q4_K, Q8_0, GPTQ) from host memory to GPU,
    /// sizing the device buffer to `host_bytes.len()` exactly rather than
    /// `shape.elem_count() * dtype.arith.byte_size()`. Mirrors the ROCm
    /// `copy_from_host_raw_bytes` contract.
    pub fn copy_from_host_raw_bytes(
        host_bytes: &[u8],
        shape: &Shape,
        dtype: DType,
        device_ordinal: usize,
    ) -> Result<Self> {
        let storage = Self::alloc_gpu_bytes(shape, dtype, host_bytes.len(), device_ordinal)?;
        let dev_ptr = storage.device_ptr.ok_or_else(|| {
            Error::Backend("copy_from_host_raw_bytes: device_ptr is null after alloc".into())
        })? as *mut c_void;

        // SAFETY: `cudaMemcpy` copies `host_bytes.len()` bytes from host to the
        // freshly allocated device buffer; direction matches the copy.
        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                host_bytes.as_ptr() as *const c_void,
                host_bytes.len(),
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            // SAFETY: free the allocated buffer on upload failure to avoid a leak.
            unsafe {
                let _ = cudaFree(dev_ptr);
            }
            return Err(Error::Backend(format!(
                "cudaMemcpyHostToDevice (raw bytes) failed with error code {}",
                res
            )));
        }

        Ok(storage)
    }

    /// Returns the tensor shape.
    pub fn shape_metadata(&self) -> &Shape {
        &self.shape
    }

    /// Returns the device ordinal.
    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns the device pointer if allocated.
    pub fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }

    /// Returns the storage size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Download the raw packed bytes (regardless of arith/storage encoding)
    /// into a host `Vec<u8>` of length `self.bytes`. Used by the host-dequant
    /// backward path to copy quantized codes from the GPU to the host.
    pub fn copy_to_host_raw_bytes(&self) -> Result<Vec<u8>> {
        let dev_ptr = self
            .device_ptr
            .ok_or_else(|| Error::Backend("CudaStorage has no valid device pointer".into()))?
            as *const c_void;
        let mut host = vec![0u8; self.bytes];
        // SAFETY: `cudaMemcpy` copies `self.bytes` from device to host.
        let res = unsafe {
            cudaMemcpy(
                host.as_mut_ptr() as *mut c_void,
                dev_ptr,
                self.bytes,
                cudaMemcpyDeviceToHost,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpyDeviceToHost failed with error code {}",
                res
            )));
        }
        Ok(host)
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        if let Some(ptr_val) = self.device_ptr {
            if ptr_val != 0 {
                // SAFETY: sync before free ensures no in-flight kernel uses the buffer.
                // A stream-ordered free (cudaFreeAsync) would be more efficient
                // once per-buffer stream handles are tracked. Drop cannot
                // propagate errors; absorb sync and free silently.
                unsafe {
                    let _ = cudaDeviceSynchronize();
                    let _ = cudaFree(ptr_val as *mut c_void);
                }
            }
        }
    }
}

impl BackendStorage for CudaStorage {
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copies GPU buffer to host as F32 vector.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        let dev_ptr = self
            .device_ptr
            .ok_or_else(|| Error::Backend("CudaStorage has no valid device pointer".into()))?
            as *mut c_void;

        let elem_count = self.shape.elem_count();

        // Quantized resident storage (KQuant/FloatPack/Block): the device buffer
        // holds packed codes smaller than `elem_count * 4` bytes. Try the GPU
        // dequant path first (bit-accurate kernels in `kernels.rs`); if there is
        // no device kernel for this dtype, fall back to staging the raw bytes to
        // host and dequantizing via grim-quant — the same contract as ROCm's
        // `to_cpu_vec_f32` → `dequant_cpu`, so `transpose_last_two`/`Linear::load`
        // (which call `to_vec_f32` on raw quantized weights) keep working.
        if self.dtype.is_quantized() {
            // Try the on-device dequant path first (bit-accurate kernels in
            // `kernels.rs`): dequant to a new F32 GPU buffer, then copy back to
            // host. Requires a real CudaDevice handle and a device ptr.
            if self.device_ptr.is_some() {
                if let Ok(dev) = CudaDevice::new(self.ordinal) {
                    if let Ok(f32_storage) = dev.dequantize_on_device(self) {
                        let mut host_data = vec![0.0f32; elem_count];
                        // SAFETY: copy the freshly dequantized F32 buffer to host.
                        let res = unsafe {
                            cudaMemcpy(
                                host_data.as_mut_ptr() as *mut c_void,
                                CudaDevice::dev_ptr_or_err("to_cpu_vec_f32(gpu)", &f32_storage)?
                                    as *mut c_void,
                                f32_storage.bytes,
                                cudaMemcpyDeviceToHost,
                            )
                        };
                        if res == cudaSuccess {
                            drop(f32_storage);
                            return Ok(host_data);
                        }
                    }
                }
            }

            // Fallback: stage the raw packed bytes to host and dequant via
            // grim-quant — the same contract as ROCm's `to_cpu_vec_f32`.
            let mut raw = vec![0u8; self.bytes];
            // SAFETY: `cudaMemcpy` copies `self.bytes` from device to host.
            let res = unsafe {
                cudaMemcpy(
                    raw.as_mut_ptr() as *mut c_void,
                    dev_ptr,
                    self.bytes,
                    cudaMemcpyDeviceToHost,
                )
            };
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMemcpyDeviceToHost (quantized) failed with error code {}",
                    res
                )));
            }
            let b_scales = <CudaStorage as BackendStorage>::quant_scales(self);
            return cuda_dequant_quantized_storage(&raw, b_scales, elem_count, &self.dtype);
        }

        // Native F32 storage: copy `self.bytes` worth of f32 elements.
        let mut host_data = vec![0.0f32; elem_count];
        // SAFETY: `cudaMemcpy` copies `self.bytes` from device to host.
        // `dev_ptr` is a valid device pointer; `host_data` is a valid host vector.
        let res = unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                dev_ptr,
                self.bytes,
                cudaMemcpyDeviceToHost,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpyDeviceToHost failed with error code {}",
                res
            )));
        }

        Ok(host_data)
    }

    /// Returns `self` as `Any` for internal downcasting.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device_ordinal(&self) -> u32 {
        self.ordinal as u32
    }

    fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }
}

/// Wrapper making cuBLAS FFI types Send + Sync.
///
/// # Safety
/// `CublasHandle` wraps a `*mut c_void` cuBLAS handle obtained from
/// `cublasCreate_v2`.  The handle is protected by an `Arc<Mutex<>>` in
/// `CudaDevice`, so concurrent access from multiple threads is
/// serialized.  cuBLAS handles are thread-local in the driver; moving
/// the handle between threads is safe because the underlying driver
/// state is not thread-local — only concurrent *use* requires
/// synchronization, which the `Mutex` provides.
#[derive(Debug)]
pub struct CublasHandle(pub *mut c_void);
// SAFETY: `CublasHandle` wraps a raw CUDA driver handle. `Send` is safe because
// the handle is bound to a specific CUDA context on one device, and the driver
// tracks it independently of the creating thread. `Sync` is safe because the
// cuBLAS API serializes concurrent calls through its internal stream/context lock.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid cuBLAS handle created via `cublasCreate_v2`.
        // It is destroyed exactly once, when the last `CudaDevice` clone sharing
        // this `Arc<Mutex<Option<CublasHandle>>>` is dropped. Previously the handle
        // was leaked (one cuBLAS handle per `CudaDevice::new`, which also allocates
        // GPU-side workspace), exhausting VRAM while loading quantized GGUF weights.
        if !self.0.is_null() {
            unsafe {
                let _ = cublasDestroy_v2(self.0);
            }
        }
    }
}

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
        if res == cudaSuccess && count > 0 {
            let mut devices = Vec::with_capacity(count as usize);
            for i in 0..count {
                if let Ok(dev) = CudaDevice::new(i as usize) {
                    devices.push(dev);
                }
            }
            return Ok(devices);
        }

        Ok(vec![])
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
    fn ensure_f32_input(name: &str, storage: &CudaStorage) -> Result<()> {
        if storage.dtype != DType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "{name}: CUDA kernel only supports F32 input (got {:?})",
                storage.dtype
            )));
        }
        Ok(())
    }

    /// Resolves a device pointer or returns Error; never panics across the FFI boundary.
    fn dev_ptr_or_err(name: &str, storage: &CudaStorage) -> Result<*mut c_void> {
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
            let mut args: [*mut c_void; 7] = [
                &mut a_arg as *mut *const c_void as *mut c_void,
                &mut b_arg as *mut *const c_void as *mut c_void,
                &mut bs_arg as *mut *const c_void as *mut c_void,
                &mut out_arg as *mut *mut c_void as *mut c_void,
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
            let grid_size = ((n_blocks as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
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
        let handle = self.launch_dequant_generic(kernel, packed_ptr, out_ptr, n_blocks)?;
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
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];
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

impl BackendDevice for CudaDevice {
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
                        let block_id = if block_idx_in_seq < max_blocks {
                            btd[block_idx_in_seq] as usize
                        } else {
                            block_idx_in_seq
                        };

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
                            let block_id = if block_idx_in_seq < max_blocks {
                                btd[block_idx_in_seq] as usize
                            } else {
                                block_idx_in_seq
                            };
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
            cudaFree(pos_dev_ptr);
        }

        Ok((Box::new(out_storage), handle))
    }

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
                    h[s] = if state_v.len() > state_idx { state_v[state_idx] } else { 0.0 };
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

                    // Build n*(k/32) scales buffer; missing entries default to 1.0.
                    let blocks_per_col = k / 32;
                    let scale_len = n * blocks_per_col;
                    let mut scales_host = vec![1.0f32; scale_len];
                    let copy_len = b_scales.len().min(scale_len);
                    scales_host[..copy_len].copy_from_slice(&b_scales[..copy_len]);
                    let scales_storage = CudaStorage::copy_from_host(
                        &scales_host,
                        &Shape::new(vec![scale_len]),
                        DType::F32,
                        self.ordinal,
                    )?;
                    let scales_ptr = scales_storage.device_ptr.ok_or_else(|| {
                        Error::Backend("quantized_matmul: failed to upload scales buffer".into())
                    })? as *const c_void;

                    let handle = self
                        .launch_quantized_matmul_q8_0(a_ptr, b_ptr, scales_ptr, out_ptr, m, n, k)?;
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
                // Q8_0 packed layout: 34-byte blocks, each with f16 scale (2 bytes)
                // followed by 32 i8 codes. Real Q8_0 embeds the scale in every block —
                // read it from the byte stream rather than consulting b_scales (which
                // is empty for the Linear path).
                let blocks_per_col = k / 32;
                let mut out = vec![0.0f32; k * n];
                for col in 0..n {
                    for block in 0..blocks_per_col {
                        let block_offset = (col * blocks_per_col + block) * 34;
                        let scale_bytes = &b_bytes[block_offset..block_offset + 2];
                        let scale = half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
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
        let b_t_storage = BackendDevice::from_cpu(
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
            grim_tensor::QuantFormat::Q8_0 => "grim_fused_quant_gemm_q8_0",
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

        // Q8_0 weights are packed (34-byte blocks: f16 scale + 32 i8 codes), but
        // the fused kernel declares B as `const float*`. Dequantize B to f32 on
        // device before launch so the kernel reads correct weight values.
        let b_ptr = if format == grim_tensor::QuantFormat::Q8_0 {
            let b_storage = b
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| Error::Backend("fused_quant_gemm: b is not CudaStorage".into()))?;
            let b_bytes = b_storage.bytes();
            let blocks_per_col = k / 32;
            let n_cols = n;
            // Q8_0: ceil(k/32)*34 bytes per column, but storage is [n, k] packed.
            // Total packed size = n * ceil(k/32) * 34.
            let expected_packed = n_cols * blocks_per_col * 34;
            if b_bytes != expected_packed {
                return Err(Error::Shape(format!(
                    "fused_quant_gemm(Q8_0): B packed size {} != expected {}",
                    b_bytes, expected_packed
                )));
            }
            // Read packed bytes to host, dequantize, upload as f32.
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
                            "fused_quant_gemm(Q8_0): cudaMemcpy(B) D2H failed: {res}"
                        )));
                    }
                }
            }
            // Dequantize: for each of n*blocks_per_col blocks, read f16 scale + 32 i8.
            let total_elements = k * n;
            let mut host_f32 = vec![0.0f32; total_elements];
            for col in 0..n_cols {
                for block in 0..blocks_per_col {
                    let block_offset = (col * blocks_per_col + block) * 34;
                    let scale = half::f16::from_le_bytes([
                        host_packed[block_offset], host_packed[block_offset + 1],
                    ]).to_f32();
                    for i in 0..32 {
                        let idx = block * 32 + i;
                        if idx < k {
                            let q = host_packed[block_offset + 2 + i] as i8 as f32;
                            host_f32[idx * n_cols + col] = q * scale;
                        }
                    }
                }
            }
            let b_f32_storage = CudaStorage::copy_from_host(
                &host_f32,
                &Shape::new(vec![k, n]),
                DType::F32,
                self.ordinal,
            )?;
            Self::dev_ptr_or_err("fused_quant_gemm b (dequantized)", &b_f32_storage)?
        } else {
            Self::dev_ptr_or_err("fused_quant_gemm b", b_storage)?
        };

        let handle = self.launch_fused_quant_gemm(kernel_name, a_ptr, b_ptr, out_ptr, m, n, k)?;
        Ok((Box::new(out), handle))
    }
}

/// Returns the byte size of a DType.
fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 | ArithType::BF16 => 2,
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}

/// Stages a CUDA-resident packed quantized buffer to a host `Vec<u8>` via
/// `cudaMemcpy` (D→H). Used by `dequantize_on_device` for formats that need
/// host-side framing (e.g. MXFP4 length-prefixed segments) before a device launch.
fn stage_packed_bytes(packed: &CudaStorage) -> Result<Vec<u8>> {
    let dev_ptr = CudaDevice::dev_ptr_or_err("stage_packed_bytes", packed)? as *const c_void;
    let mut raw = vec![0u8; packed.bytes];
    // SAFETY: `cudaMemcpy` copies `packed.bytes` from device to host.
    let res = unsafe {
        cudaMemcpy(
            raw.as_mut_ptr() as *mut c_void,
            dev_ptr as *mut c_void,
            packed.bytes,
            cudaMemcpyDeviceToHost,
        )
    };
    if res != cudaSuccess {
        return Err(Error::Backend(format!(
            "stage_packed_bytes: cudaMemcpy D→H failed with error code {}",
            res
        )));
    }
    Ok(raw)
}

/// Reads a length-prefixed segment from `bytes` starting at `*cursor`:
/// 8-byte LE `u64` length, then `len` payload bytes. Advances `cursor`
/// past the segment. Mirrors `grim_quant::dequant_mxfp4`'s `read_segment`.
fn read_length_prefixed(bytes: &[u8], cursor: &mut usize) -> Result<Vec<u8>> {
    if bytes.len() < *cursor + 8 {
        return Err(Error::Backend(
            "read_length_prefixed: truncated segment length prefix".into(),
        ));
    }
    let len = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap()) as usize;
    *cursor += 8;
    if bytes.len() < *cursor + len {
        return Err(Error::Backend(format!(
            "read_length_prefixed: truncated segment (expected {len} bytes)"
        )));
    }
    let segment = bytes[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Ok(segment)
}

/// Host-side dequantization of CUDA-resident packed storage. Dispatches on the
/// `Storage` variant to the matching `grim_quant::dequant_*` entry point
/// (the same family used by `grim-format::convert::dequant_tensor_data`).
/// Returns a row-major F32 vector of length `elem_count` matching `B`'s logical
/// `[K, N]` layout.
fn cuda_dequant_quantized_storage(
    b_bytes: &[u8],
    _b_scales: Option<&[f32]>,
    elem_count: usize,
    dtype: &DType,
) -> Result<Vec<f32>> {
    match &dtype.storage {
        DTypeStorage::KQuant(scheme) => match scheme {
            KQuantScheme::Q2K => grim_quant::dequant_q2k(b_bytes, elem_count),
            KQuantScheme::Q3K => grim_quant::dequant_q3k(b_bytes, elem_count),
            KQuantScheme::Q4K => grim_quant::dequant_q4k(b_bytes, elem_count),
            KQuantScheme::Q5K => grim_quant::dequant_q5k(b_bytes, elem_count),
            KQuantScheme::Q6K => grim_quant::dequant_q6k(b_bytes, elem_count),
            KQuantScheme::Q80 => grim_quant::dequant_q80(b_bytes, elem_count),
            KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(b_bytes, elem_count),
            KQuantScheme::IQ4XS => grim_quant::dequant_iq4xs(b_bytes, elem_count),
            KQuantScheme::IQ3XXS => grim_quant::dequant_iq3xxs(b_bytes, elem_count),
            KQuantScheme::IQ3S => grim_quant::dequant_iq3s(b_bytes, elem_count),
            KQuantScheme::IQ2XXS => grim_quant::dequant_iq2xxs(b_bytes, elem_count),
            KQuantScheme::IQ2XS => grim_quant::dequant_iq2xs(b_bytes, elem_count),
            KQuantScheme::IQ2S => grim_quant::dequant_iq2s(b_bytes, elem_count),
        },
        DTypeStorage::FloatPack(scheme) => match scheme {
            FloatPackScheme::Fp4 => grim_quant::dequant_fp4(b_bytes, elem_count),
            FloatPackScheme::Nf4 => grim_quant::dequant_nf4(b_bytes, elem_count),
            FloatPackScheme::Fp8 => grim_quant::dequant_fp8(b_bytes, elem_count),
            FloatPackScheme::MxFp4 => grim_quant::dequant_mxfp4(b_bytes, elem_count),
            FloatPackScheme::MxFp8 => grim_quant::dequant_mxfp8(b_bytes, elem_count),
        },
        DTypeStorage::Block(bd) => match bd {
            BlockDtype::Fp4 | BlockDtype::Fp4Block16 => {
                grim_quant::dequant_fp4_block16(b_bytes, elem_count)
            }
            BlockDtype::Nf4 => grim_quant::dequant_nf4(b_bytes, elem_count),
            BlockDtype::Fp8 | BlockDtype::Fp8Block16 => {
                grim_quant::dequant_fp8_block16(b_bytes, elem_count)
            }
        },
        // ResidualPacked has no host dequant implementation and is intentionally
        // not kept resident on CUDA (see varbuilder::materialize).
        DTypeStorage::ResidualPacked(cfg) => Err(Error::Unimplemented(format!(
            "quantized_matmul_backward_dx: ResidualPacked (bpw {}) host dequant not implemented; \
             this layout requires a fused ROCm device kernel",
            cfg.bpw
        ))),
        DTypeStorage::GroupInt(_) => Err(Error::Unimplemented(
            "quantized_matmul_backward_dx: GroupInt storage is dequantized to F32 at load time \
             on CUDA and does not reach the fused path"
                .into(),
        )),
        DTypeStorage::Native => Err(Error::Backend(format!(
            "quantized_matmul_backward_dx: expected quantized b, got Native ({:?})",
            dtype
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::{DType, Shape};

    /// Source-presence guard for the partial-rotary / YaRN RoPE kernel and the
    /// sliding-window attention extension. No GPU required — this just asserts
    /// the CUDA kernel source declares the symbols and parameters the host
    /// dispatchers expect, mirroring the ROCm `test_rope_yarn_kernel_presence`.
    #[test]
    fn test_rope_yarn_and_window_lo_kernel_presence() {
        let src = crate::kernels::KERNELS_SOURCE;
        assert!(
            src.contains("grim_rope_yarn"),
            "KERNELS_SOURCE must declare grim_rope_yarn"
        );
        assert!(
            src.contains("inv_freq"),
            "grim_rope_yarn must take a pre-computed inv_freq buffer"
        );
        assert!(
            src.contains("mscale"),
            "grim_rope_yarn must take an mscale (attention_factor) parameter"
        );
        assert!(
            src.contains("rotary_half"),
            "grim_rope_yarn must take a rotary_half (partial-rotary) parameter"
        );
        assert!(
            src.contains("window_lo"),
            "grim_qkv_attention must take a window_lo parameter for SWA"
        );
    }

    #[test]
    fn test_cuda_device_probe() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].ordinal, 0);
    }

    #[test]
    fn test_cuda_zeros() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![0.0; 8]);
    }

    #[test]
    fn test_cuda_from_cpu() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![3, 2]);
        let host_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, host_data);
    }

    #[test]
    fn test_cuda_math_ops() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![4]);
        let host_data = vec![4.0f32, 9.0, 16.0, 25.0];
        let x = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();

        let (out_sqrt, _) = dev.sqrt(x.as_ref(), &shape).unwrap();
        assert_eq!(out_sqrt.to_cpu_vec_f32().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

        let (out_recip, _) = dev.recip(out_sqrt.as_ref(), &shape).unwrap();
        assert_eq!(
            out_recip.to_cpu_vec_f32().unwrap(),
            vec![0.5, 1.0 / 3.0, 0.25, 0.2]
        );

        let (out_mul, _) = dev.mul_scalar(x.as_ref(), 0.5, &shape).unwrap();
        assert_eq!(out_mul.to_cpu_vec_f32().unwrap(), vec![2.0, 4.5, 8.0, 12.5]);
    }

    /// GPU-gated parity test for the fused grouped MoE dispatch kernel.
    /// Compares the GPU output against a hand-computed CPU reference for a tiny
    /// 2-expert / 2-token / top-1 routing. Numerical tolerance is loose because
    /// FP32 atomic adds can reorder; the contract is correctness of the fused
    /// gate+up SiLU combine + down + routed_scaling_factor accumulate.
    #[test]
    fn test_cuda_moe_fused_dispatch_parity() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        if devices.is_empty() {
            return;
        }
        let dev = &devices[0];

        let hidden: usize = 4;
        let inter: usize = 3;
        let num_experts: usize = 2;
        let batch: usize = 2;
        let rsf: f32 = 0.5;

        // activations [batch, hidden]
        let x_data: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.1).collect();
        let x = dev
            .from_cpu(&x_data, &Shape::new(vec![batch, hidden]), DType::F32)
            .unwrap();

        // per-expert gate/up [inter, hidden], down [hidden, inter]
        let mk = |e: usize, sign: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; inter * hidden];
            for i in 0..inter {
                for h in 0..hidden {
                    v[i * hidden + h] =
                        sign * (1.0 + (i as f32) * 0.1 + (h as f32) * 0.01 + e as f32);
                }
            }
            v
        };
        let gate_flat: Vec<f32> = (0..num_experts).flat_map(|e| mk(e, 1.0)).collect();
        let up_flat: Vec<f32> = (0..num_experts).flat_map(|e| mk(e, 1.0)).collect();
        let down_flat: Vec<f32> = (0..num_experts)
            .flat_map(|e| {
                let mut v = vec![0.0f32; hidden * inter];
                for h in 0..hidden {
                    for i in 0..inter {
                        v[h * inter + i] = 1.0 + (h as f32) * 0.05 + (i as f32) * 0.02 + e as f32;
                    }
                }
                v
            })
            .collect();

        // top-1 routing: token0 -> expert0, token1 -> expert1
        let rtok = vec![0u32, 1u32];
        let rexp = vec![0u32, 1u32];
        let rw = vec![1.0f32, 1.0f32];
        let num_pairs = rtok.len();

        let gate_buf = dev
            .from_cpu(
                &gate_flat,
                &Shape::new(vec![num_experts * inter * hidden]),
                DType::F32,
            )
            .unwrap();
        let up_buf = dev
            .from_cpu(
                &up_flat,
                &Shape::new(vec![num_experts * inter * hidden]),
                DType::F32,
            )
            .unwrap();
        let down_buf = dev
            .from_cpu(
                &down_flat,
                &Shape::new(vec![num_experts * hidden * inter]),
                DType::F32,
            )
            .unwrap();
        let rtok_bytes: Vec<u8> = rtok.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rexp_bytes: Vec<u8> = rexp.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rw_bytes: Vec<u8> = rw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tok_buf = Box::new(
            CudaStorage::copy_from_host_raw_bytes(
                &rtok_bytes,
                &Shape::new(vec![num_pairs]),
                DType::F32,
                0,
            )
            .unwrap(),
        );
        let exp_buf = Box::new(
            CudaStorage::copy_from_host_raw_bytes(
                &rexp_bytes,
                &Shape::new(vec![num_pairs]),
                DType::F32,
                0,
            )
            .unwrap(),
        );
        let w_buf = Box::new(
            CudaStorage::copy_from_host_raw_bytes(
                &rw_bytes,
                &Shape::new(vec![num_pairs]),
                DType::F32,
                0,
            )
            .unwrap(),
        );

        let out_shape = Shape::new(vec![batch, hidden]);
        let (out, _h) = dev
            .moe_fused_dispatch(
                &*x,
                &*gate_buf,
                &*up_buf,
                &*down_buf,
                &*tok_buf,
                &*exp_buf,
                &*w_buf,
                &out_shape,
                hidden as u32,
                inter as u32,
                num_experts as u32,
                batch as u32,
                rsf,
            )
            .unwrap();
        let res = out.to_cpu_vec_f32().unwrap();

        // CPU reference
        let silu = |a: f32| a / (1.0 + (-a).exp());
        let dot = |w: &[f32], xx: &[f32]| -> f32 { (0..w.len()).map(|i| w[i] * xx[i]).sum() };
        for t in 0..batch {
            let e = rexp[t] as usize;
            let xt = &x_data[t * hidden..(t + 1) * hidden];
            let gw = &gate_flat[e * inter * hidden..(e + 1) * inter * hidden];
            let uw = &up_flat[e * inter * hidden..(e + 1) * inter * hidden];
            let dw = &down_flat[e * hidden * inter..(e + 1) * hidden * inter];
            let mut routed = vec![0.0f32; hidden];
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for i in 0..inter {
                    let g = dot(&gw[i * hidden..i * hidden + hidden], xt);
                    let u = dot(&uw[i * hidden..i * hidden + hidden], xt);
                    acc += dw[h * inter + i] * (silu(g) * u);
                }
                routed[h] = rsf * acc;
            }
            for h in 0..hidden {
                let got = res[t * hidden + h];
                let tol = routed[h].abs().max(1.0) * 1e-3 + 1e-3;
                assert!(
                    (got - routed[h]).abs() < tol,
                    "moe tok{} dim{}: gpu {} vs ref {} (tol {})",
                    t,
                    h,
                    got,
                    routed[h],
                    tol
                );
            }
        }
    }

    #[test]
    fn test_cuda_matmul() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let a_shape = Shape::new(vec![2, 2]);
        let b_shape = Shape::new(vec![2, 2]);
        let out_shape = Shape::new(vec![2, 2]);

        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_storage = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

        let (out_storage, handle) = dev
            .matmul(a_storage.as_ref(), b_storage.as_ref(), &out_shape)
            .unwrap();
        handle.synchronize().unwrap();

        let res = out_storage.to_cpu_vec_f32().unwrap();
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cuda_ops() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let shape = Shape::new(vec![4]);

        let a = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        // 1. Add
        let (out_add, h) = dev.add(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        assert_eq!(
            out_add.to_cpu_vec_f32().unwrap(),
            vec![6.0, 8.0, 10.0, 12.0]
        );

        // 2. Mul
        let (out_mul, h) = dev.mul(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        assert_eq!(
            out_mul.to_cpu_vec_f32().unwrap(),
            vec![5.0, 12.0, 21.0, 32.0]
        );

        // 3. SiLU Mul
        let (out_silu, h) = dev.silu_mul(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        let res_silu = out_silu.to_cpu_vec_f32().unwrap();
        let expected_silu0 = (1.0f32 / (1.0f32 + (-1.0f32).exp())) * 5.0f32;
        assert!((res_silu[0] - expected_silu0).abs() < 1e-4);

        // 4. RMS Norm
        let weight_data = vec![1.0, 1.0, 1.0, 1.0];
        let weight = dev.from_cpu(&weight_data, &shape, DType::F32).unwrap();
        let (out_rms, h) = dev
            .rms_norm(a.as_ref(), weight.as_ref(), 1e-5, &shape)
            .unwrap();
        h.synchronize().unwrap();
        let res_rms = out_rms.to_cpu_vec_f32().unwrap();
        // RMS([1,2,3,4]) = sqrt((1+4+9+16)/4) ≈ 2.7386
        let rms_val = 7.5f32.sqrt();
        assert!((res_rms[0] - 1.0 / rms_val).abs() < 1e-4);

        // 5. Softmax
        let (out_sm, h) = dev.softmax(a.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        let res_sm = out_sm.to_cpu_vec_f32().unwrap();
        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp() + 4.0f32.exp();
        assert!((res_sm[0] - 1.0f32.exp() / sum_exp).abs() < 1e-4);

        // 6. Embedding
        let weight_emb_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let weight_emb = dev
            .from_cpu(&weight_emb_data, &Shape::new(vec![3, 2]), DType::F32)
            .unwrap();
        let indices = vec![2u32, 0u32];
        let out_emb_shape = Shape::new(vec![2, 2]);
        let (out_emb, h) = dev
            .embedding(weight_emb.as_ref(), &indices, &out_emb_shape)
            .unwrap();
        h.synchronize().unwrap();
        let res_emb = out_emb.to_cpu_vec_f32().unwrap();
        assert_eq!(res_emb, vec![50.0, 60.0, 10.0, 20.0]);
    }

    #[test]
    fn test_cuda_matmul_shape_mismatch_returns_error() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a_shape = Shape::new(vec![2, 3]);
        let b_shape = Shape::new(vec![4, 2]); // K mismatch: 3 != 4
        let out_shape = Shape::new(vec![2, 2]);

        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_storage = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

        let res = dev.matmul(a_storage.as_ref(), b_storage.as_ref(), &out_shape);
        assert!(
            res.is_err(),
            "matmul with mismatched inner dimension K must return Err"
        );
    }

    #[test]
    fn test_cuda_rms_norm_exact() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let x_data = vec![3.0f32, 4.0]; // mean(x^2) = (9 + 16)/2 = 12.5
        let weight_data = vec![1.0f32, 2.0];
        let shape = Shape::new(vec![2]);

        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let weight = dev.from_cpu(&weight_data, &shape, DType::F32).unwrap();

        let (out_rms, h) = dev
            .rms_norm(x.as_ref(), weight.as_ref(), 1e-6, &shape)
            .unwrap();
        h.synchronize().unwrap();

        let res = out_rms.to_cpu_vec_f32().unwrap();
        let rms_val = (12.5f32 + 1e-6).sqrt();
        let expected_0 = (3.0 / rms_val) * 1.0;
        let expected_1 = (4.0 / rms_val) * 2.0;

        assert!(
            (res[0] - expected_0).abs() < 1e-4,
            "res[0] = {}, want {}",
            res[0],
            expected_0
        );
        assert!(
            (res[1] - expected_1).abs() < 1e-4,
            "res[1] = {}, want {}",
            res[1],
            expected_1
        );
    }

    #[test]
    fn test_cuda_softmax_exact() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let x_data = vec![1.0f32, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();

        let (out_sm, h) = dev.softmax(x.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();

        let res = out_sm.to_cpu_vec_f32().unwrap();
        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        let expected_0 = 1.0f32.exp() / sum_exp;
        let expected_1 = 2.0f32.exp() / sum_exp;
        let expected_2 = 3.0f32.exp() / sum_exp;

        assert!((res[0] - expected_0).abs() < 1e-4);
        assert!((res[1] - expected_1).abs() < 1e-4);
        assert!((res[2] - expected_2).abs() < 1e-4);
    }

    /// CPU reference for quantized_matmul using Q8_0 convention:
    /// B is packed [col][block][32] raw int8, scales holds n*(k/32) per-block values.
    fn q8_matmul_ref(
        a: &[f32],
        b: &[u8],
        scales: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let blocks = k / 32;
        let mut out = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let blk = p / 32;
                    let i = p % 32;
                    if blk >= blocks {
                        continue;
                    }
                    let b_idx = (ni * blocks + blk) * 32 + i;
                    let q = (b[b_idx] as i8) as f32;
                    let scale = if ni * blocks + blk < scales.len() {
                        scales[ni * blocks + blk]
                    } else {
                        1.0
                    };
                    sum += a[mi * k + p] * (q * scale);
                }
                out[mi * n + ni] = sum;
            }
        }
        out
    }

    fn assert_q8_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let max_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "Q8_0 quantized matmul max error {max_err} exceeds 1e-3"
        );
    }

    #[test]
    fn test_cuda_quantized_matmul_q8_0_gpu_fast_path() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let (m, k, n) = (2usize, 256usize, 8usize);
        let blocks = k / 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b_bytes: Vec<u8> = (0..k * n).map(|i| ((i * 7) % 251) as u8).collect();
        let b_scales: Vec<f32> = (0..n * blocks)
            .map(|i| 0.5 + (i as f32 * 0.1).fract())
            .collect();
        let expected = q8_matmul_ref(&a_data, &b_bytes, &b_scales, m, k, n);

        let a_shape = Shape::new(vec![m, k]);
        let b_shape = Shape::new(vec![n, k]);
        let out_shape = Shape::new(vec![m, n]);
        let a_dev = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_dev = dev
            .from_cpu_bytes(
                &b_bytes,
                &b_shape,
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )
            .unwrap();

        let (out, handle) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &b_scales,
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        assert_q8_close(&out.to_cpu_vec_f32().unwrap(), &expected);
    }

    #[test]
    fn test_cuda_quantized_matmul_q8_0_empty_scales_defaults() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let (m, k, n) = (3usize, 64usize, 4usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).cos()).collect();
        let b_bytes: Vec<u8> = (0..k * n).map(|i| ((i * 11) % 256) as u8).collect();
        let expected = q8_matmul_ref(&a_data, &b_bytes, &[], m, k, n);

        let a_shape = Shape::new(vec![m, k]);
        let b_shape = Shape::new(vec![n, k]);
        let out_shape = Shape::new(vec![m, n]);
        let a_dev = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_dev = dev
            .from_cpu_bytes(
                &b_bytes,
                &b_shape,
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )
            .unwrap();

        let (out, handle) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &[],
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        assert_q8_close(&out.to_cpu_vec_f32().unwrap(), &expected);
    }

    #[test]
    fn test_cuda_quantized_matmul_q8_0_cpu_fallback() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        // K not a multiple of 32, forcing CPU fallback.
        let (m, k, n) = (3usize, 34usize, 5usize);
        let blocks = k / 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).cos()).collect();
        let b_bytes: Vec<u8> = (0..k * n).map(|i| ((i * 13) % 200) as u8).collect();
        let b_scales: Vec<f32> = (0..n * blocks)
            .map(|i| 1.0 + (i as f32 * 0.25).fract())
            .collect();
        let expected = q8_matmul_ref(&a_data, &b_bytes, &b_scales, m, k, n);

        let a_shape = Shape::new(vec![m, k]);
        let b_shape = Shape::new(vec![n, k]);
        let out_shape = Shape::new(vec![m, n]);
        let a_dev = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_dev = dev
            .from_cpu_bytes(
                &b_bytes,
                &b_shape,
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )
            .unwrap();

        let (out, handle) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &b_scales,
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        assert_q8_close(&out.to_cpu_vec_f32().unwrap(), &expected);
    }

    #[test]
    fn test_cuda_quantized_matmul_backward_dx_q8_0() {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        let devices = CudaDevice::probe().unwrap();
        let dev = &devices[0];

        let (m, k, n) = (4usize, 64usize, 8usize);
        let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
        let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

        // Host reference: dequantize B to F32, then dX = dY @ B^T.
        let b_packed = grim_quant::quant_q80(&b_orig).unwrap();
        let b_dequant = grim_quant::dequant_q80(&b_packed, k * n).unwrap();
        let mut dx_ref = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                for l in 0..n {
                    sum += dy_host[i * n + l] * b_dequant[j * n + l];
                }
                dx_ref[i * k + j] = sum;
            }
        }

        // Upload dY as F32 [M, N].
        let dy_shape = Shape::new(vec![m, n]);
        let dy_dev = dev.from_cpu(&dy_host, &dy_shape, DType::F32).unwrap();

        // Upload packed B as KQuant(Q80) [K, N] (stays quantized resident).
        let b_shape = Shape::new(vec![k, n]);
        let b_dev = dev
            .from_cpu_bytes(
                &b_packed,
                &b_shape,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                },
            )
            .unwrap();

        let out_shape = Shape::new(vec![m, k]);
        let (dx_dev, handle) = dev
            .quantized_matmul_backward_dx(
                dy_dev.as_ref(),
                b_dev.as_ref(),
                &[],
                8, // bpw for Q8_0
                m,
                n,
                k,
                &out_shape,
                None,
            )
            .expect("CUDA quantized_matmul_backward_dx must succeed on a real CUDA device");
        handle.synchronize().unwrap();

        let dx_actual = dx_dev
            .to_cpu_vec_f32()
            .expect("CUDA backward result must be readable");
        assert_eq!(dx_actual.len(), m * k);
        for (a, e) in dx_actual.iter().zip(dx_ref.iter()) {
            let err = (a - e).abs();
            assert!(
                err < 1e-3,
                "CUDA Q8_0 backward dX error {err} at actual={a} expected={e}"
            );
        }
    }

    // ===================================================================
    //  GPU dequant kernel golden tests — bit-accurate parity vs the
    //  `grim_quant::dequant_*` CPU oracle. Each test:
    //    1. Builds the packed bytes for one or more super-blocks via
    //       `grim_quant::quant_<type>` (or hand-fabricated for MXFP4).
    //    2. Uploads the packed bytes to a `CudaStorage` with the matching
    //       quantized `DType.storage` (so `dequantize_on_device` dispatches).
    //    3. Calls `dev.dequantize_on_device(as_cuda_storage(storage.as_ref()))` (GPU kernel).
    //    4. Compares `out.to_cpu_vec_f32()` to the CPU oracle within a tight
    //       tolerance that admits only floating-point rounding (1e-4).
    // Skipped (not failed) when no CUDA device is present.
    // ===================================================================

    fn dequant_test_device() -> Option<CudaDevice> {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        CudaDevice::probe()
            .ok()
            .filter(|d| !d.is_empty())
            .map(|d| d[0].clone())
    }

    fn assert_dequant_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        let max_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "{label}: GPU dequant max error {max_err} exceeds 1e-4 \
             (first 4 actual={:?} expected={:?})",
            &actual[..actual.len().min(4)],
            &expected[..expected.len().min(4)],
        );
    }

    /// Upload raw packed quantized bytes with the given quantized `DType.storage`
    /// to a device-resident `CudaStorage`, returned as `Box<dyn BackendStorage>`.
    fn upload_packed(
        dev: &CudaDevice,
        bytes: &[u8],
        shape: &Shape,
        storage_kind: DTypeStorage,
    ) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: ArithType::U8,
            storage: storage_kind,
        };
        dev.from_cpu_bytes(bytes, shape, dtype)
            .expect("from_cpu_bytes for packed quantized storage")
    }

    /// Downcast a `BackendStorage` to `&CudaStorage` (the only concrete type
    /// `from_cpu_bytes` produces on this backend).
    fn as_cuda_storage(s: &dyn BackendStorage) -> &CudaStorage {
        s.as_any()
            .downcast_ref::<CudaStorage>()
            .expect("expected CudaStorage from from_cpu_bytes")
    }

    fn build_mxfp4_single_buffer(codes: &[u8], exps: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        buf.extend_from_slice(codes);
        buf.extend_from_slice(&(exps.len() as u64).to_le_bytes());
        buf.extend_from_slice(exps);
        buf
    }

    #[test]
    fn test_cuda_dequant_q5k_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // 2 super-blocks × 256 weights = 512 weights.
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let packed = grim_quant::quant_q5k(&src).expect("quant_q5k");
        let expected = grim_quant::dequant_q5k(&packed, n).expect("cpu oracle q5k");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::Q5K),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant q5k");
        let actual = out.to_cpu_vec_f32().expect("readback q5k");
        assert_dequant_close("q5k", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_q4k_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let packed = grim_quant::quant_q4k(&src).expect("quant_q4k");
        let expected = grim_quant::dequant_q4k(&packed, n).expect("cpu oracle q4k");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::Q4K),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant q4k");
        let actual = out.to_cpu_vec_f32().expect("readback q4k");
        assert_dequant_close("q4k", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_q6k_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos() * 0.7).collect();
        let packed = grim_quant::quant_q6k(&src).expect("quant_q6k");
        let expected = grim_quant::dequant_q6k(&packed, n).expect("cpu oracle q6k");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::Q6K),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant q6k");
        let actual = out.to_cpu_vec_f32().expect("readback q6k");
        assert_dequant_close("q6k", &actual, &expected);
    }

    #[test]
    fn test_cuda_fused_quant_gemm_q4k_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // K must be a multiple of 256 for the Q quantized GEMM block layout.
        let m = 4u32;
        let k = 512u32;
        let n = 256u32;

        let a_src: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.03).sin()).collect();
        let b_src: Vec<f32> = (0..(k as usize * n as usize))
            .map(|i| (i as f32 * 0.017).cos())
            .collect();

        let a_shape = Shape::new(vec![m as usize, k as usize]);
        let b_shape = Shape::new(vec![n as usize, k as usize]); // [N, K] packed layout
        let a_storage = dev
            .from_cpu(&a_src, &a_shape, DType::F32)
            .expect("a from_cpu");
        let a_cuda = as_cuda_storage(a_storage.as_ref());

        let packed = grim_quant::quant_q4k(&b_src).expect("quant_q4k");
        let b_storage = upload_packed(
            &dev,
            &packed,
            &b_shape,
            DTypeStorage::KQuant(KQuantScheme::Q4K),
        );
        let b_cuda = as_cuda_storage(b_storage.as_ref());

        let out_shape = Shape::new(vec![m as usize, n as usize]);
        let (fused_out, h) = dev
            .fused_quant_gemm(a_cuda, b_cuda, grim_tensor::QuantFormat::Q4K, &out_shape)
            .expect("fused_quant_gemm q4k");
        h.synchronize().expect("sync");
        let actual = fused_out.to_cpu_vec_f32().expect("readback");

        // Reference: A @ B^T where B is dequantized [N, K] then transposed.
        let b_deq = grim_quant::dequant_q4k(&packed, (k * n) as usize).expect("cpu dequant q4k");
        let mut expected = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut s = 0.0f32;
                for t in 0..k as usize {
                    s += a_src[i * k as usize + t] * b_deq[j * k as usize + t];
                }
                expected[i * n as usize + j] = s;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..expected.len() {
            let e = (actual[i] - expected[i]).abs();
            let denom = expected[i].abs().max(1.0);
            max_err = max_err.max(e / denom);
        }
        assert!(
            max_err < 0.05,
            "fused q4k GEMM mismatch: max_rel_err={max_err}"
        );
    }

    #[test]
    fn test_cuda_fused_quant_gemm_q6k_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // K must be a multiple of 256 for the Q quantized GEMM block layout.
        let m = 4u32;
        let k = 512u32;
        let n = 256u32;

        let a_src: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.03).sin()).collect();
        let b_src: Vec<f32> = (0..(k as usize * n as usize))
            .map(|i| (i as f32 * 0.017).cos())
            .collect();

        let a_shape = Shape::new(vec![m as usize, k as usize]);
        let b_shape = Shape::new(vec![n as usize, k as usize]); // [N, K] packed layout
        let a_storage = dev
            .from_cpu(&a_src, &a_shape, DType::F32)
            .expect("a from_cpu");
        let a_cuda = as_cuda_storage(a_storage.as_ref());

        let packed = grim_quant::quant_q6k(&b_src).expect("quant_q6k");
        let b_storage = upload_packed(
            &dev,
            &packed,
            &b_shape,
            DTypeStorage::KQuant(KQuantScheme::Q6K),
        );
        let b_cuda = as_cuda_storage(b_storage.as_ref());

        let out_shape = Shape::new(vec![m as usize, n as usize]);
        let (fused_out, h) = dev
            .fused_quant_gemm(a_cuda, b_cuda, grim_tensor::QuantFormat::Q6K, &out_shape)
            .expect("fused_quant_gemm q6k");
        h.synchronize().expect("sync");
        let actual = fused_out.to_cpu_vec_f32().expect("readback");

        let b_deq = grim_quant::dequant_q6k(&packed, (k * n) as usize).expect("cpu dequant q6k");
        let mut expected = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut s = 0.0f32;
                for t in 0..k as usize {
                    s += a_src[i * k as usize + t] * b_deq[j * k as usize + t];
                }
                expected[i * n as usize + j] = s;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..expected.len() {
            let e = (actual[i] - expected[i]).abs();
            let denom = expected[i].abs().max(1.0);
            max_err = max_err.max(e / denom);
        }
        assert!(
            max_err < 0.05,
            "fused q6k GEMM mismatch: max_rel_err={max_err}"
        );
    }

    #[test]
    fn test_cuda_fused_quant_gemm_real_model_q4k_q6k() {
        // Definitive check: run the actual CUDA fused GEMM kernels against
        // REAL Q4_K / Q6_K tensors extracted from the on-disk Q4_K_M model,
        // comparing to grim_quant::dequant_*k + a CPU A@B^T reference.
        let Some(dev) = dequant_test_device() else {
            return;
        };
        use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find(|p| p.join("models").is_dir())
            .expect("repo root with models/")
            .to_path_buf();
        let path = repo_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
        let Ok(f) = std::fs::File::open(&path) else {
            eprintln!("skip: model not present");
            return;
        };
        let mut reader = std::io::BufReader::new(f);
        let file = read_gguf(&mut reader).expect("read_gguf");

        let mut run_one = |dtype_want: GgufDType, name: &str| {
            let target = file
                .tensors
                .iter()
                .find(|t| t.dtype == dtype_want && t.name == name)
                .expect("target tensor");
            let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
            let dims = &target.dims;
            let (n, k) = (dims[0] as u32, dims[1] as u32);
            let m = 8u32;
            let a_src: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.013).sin()).collect();
            let a_shape = Shape::new(vec![m as usize, k as usize]);
            let a_storage = dev
                .from_cpu(&a_src, &a_shape, DType::F32)
                .expect("a from_cpu");
            let a_cuda = as_cuda_storage(a_storage.as_ref());
            let b_shape = Shape::new(vec![n as usize, k as usize]);
            let b_storage = upload_packed(
                &dev,
                &bytes,
                &b_shape,
                DTypeStorage::KQuant(match dtype_want {
                    GgufDType::Q4K => KQuantScheme::Q4K,
                    GgufDType::Q6K => KQuantScheme::Q6K,
                    _ => panic!("unexpected dtype"),
                }),
            );
            let b_cuda = as_cuda_storage(b_storage.as_ref());
            let out_shape = Shape::new(vec![m as usize, n as usize]);
            let fmt = match dtype_want {
                GgufDType::Q4K => grim_tensor::QuantFormat::Q4K,
                GgufDType::Q6K => grim_tensor::QuantFormat::Q6K,
                _ => panic!("unexpected dtype"),
            };
            let (fused_out, h) = dev
                .fused_quant_gemm(a_cuda, b_cuda, fmt, &out_shape)
                .expect("fused_quant_gemm");
            h.synchronize().expect("sync");
            let actual = fused_out.to_cpu_vec_f32().expect("readback");
            let elem = (n as usize) * (k as usize);
            let b_deq = match dtype_want {
                GgufDType::Q4K => grim_quant::dequant_q4k(&bytes, elem).expect("deq"),
                GgufDType::Q6K => grim_quant::dequant_q6k(&bytes, elem).expect("deq"),
                _ => panic!("unexpected dtype"),
            };
            let mut expected = vec![0.0f32; (m * n) as usize];
            for i in 0..m as usize {
                for j in 0..n as usize {
                    let mut s = 0.0f32;
                    for t in 0..k as usize {
                        s += a_src[i * k as usize + t] * b_deq[j * k as usize + t];
                    }
                    expected[i * n as usize + j] = s;
                }
            }
            let mut max_err = 0.0f32;
            for i in 0..expected.len() {
                let e = (actual[i] - expected[i]).abs();
                let denom = expected[i].abs().max(1.0);
                max_err = max_err.max(e / denom);
            }
            (name.to_string(), max_err)
        };

        let (n1, e1) = run_one(GgufDType::Q4K, "token_embd.weight");
        eprintln!("[real-q4k] {n1}: max_rel_err={e1}");
        assert!(e1 < 0.05, "real-model Q4K fused GEMM mismatch: {e1}");
        let (n1b, e1b) = run_one(GgufDType::Q4K, "blk.0.attn_q.weight");
        eprintln!("[real-q4k] {n1b}: max_rel_err={e1b}");
        assert!(
            e1b < 0.05,
            "real-model Q4K attn_q fused GEMM mismatch: {e1b}"
        );
        let (n2, e2) = run_one(GgufDType::Q6K, "output.weight");
        eprintln!("[real-q6k] {n2}: max_rel_err={e2}");
        assert!(e2 < 0.05, "real-model Q6K fused GEMM mismatch: {e2}");
        let (n2b, e2b) = run_one(GgufDType::Q6K, "blk.0.attn_v.weight");
        eprintln!("[real-q6k] {n2b}: max_rel_err={e2b}");
        assert!(
            e2b < 0.05,
            "real-model Q6K attn_v fused GEMM mismatch: {e2b}"
        );

        // CLI orientation: get transposes attn_v to [256, 1536]; verify the
        // kernel on that transposed layout with several A patterns.
        {
            let target = file
                .tensors
                .iter()
                .find(|t| t.dtype == GgufDType::Q6K && t.name == "blk.0.attn_v.weight")
                .expect("target tensor");
            let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
            let (n, k) = (256u32, 1536u32); // transposed [out, in]
            let m = 16u32;
            let patterns: Vec<(&str, Vec<f32>)> = vec![
                (
                    "sin",
                    (0..(m * k)).map(|i| (i as f32 * 0.013).sin()).collect(),
                ),
                ("ones", vec![1.0f32; (m * k) as usize]),
                (
                    "realmag",
                    (0..(m * k))
                        .map(|i| {
                            // mimic real x_norm range ~[-3.4, 1.7]
                            if i % 7 == 0 { -3.37f32 } else { 1.71f32 }
                        })
                        .collect(),
                ),
            ];
            for (name, a_src) in patterns {
                let a_shape = Shape::new(vec![m as usize, k as usize]);
                let a_storage = dev
                    .from_cpu(&a_src, &a_shape, DType::F32)
                    .expect("a from_cpu");
                let a_cuda = as_cuda_storage(a_storage.as_ref());
                let b_shape = Shape::new(vec![n as usize, k as usize]);
                let b_storage = upload_packed(
                    &dev,
                    &bytes,
                    &b_shape,
                    DTypeStorage::KQuant(KQuantScheme::Q6K),
                );
                let b_cuda = as_cuda_storage(b_storage.as_ref());
                let out_shape = Shape::new(vec![m as usize, n as usize]);
                let (fused_out, h) = dev
                    .fused_quant_gemm(a_cuda, b_cuda, grim_tensor::QuantFormat::Q6K, &out_shape)
                    .expect("fused_quant_gemm");
                h.synchronize().expect("sync");
                let actual = fused_out.to_cpu_vec_f32().expect("readback");
                let nan = actual.iter().filter(|x| x.is_nan()).count();
                let max_abs = actual.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("[cli-q6k][{name}] attn_v [256,1536]: nan={nan} max_abs={max_abs:.4}");
            }
        }
    }

    #[test]
    fn test_cuda_dequant_iq4nl_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 0.3).collect();
        let packed = grim_quant::quant_iq4nl(&src).expect("quant_iq4nl");
        let expected = grim_quant::dequant_iq4nl(&packed, n).expect("cpu oracle iq4nl");
        let fallback = dev
            .dequantize_iq_host(&packed, n, KQuantScheme::IQ4NL)
            .expect("host fallback");
        assert_eq!(fallback.len(), expected.len());
    }

    #[test]
    fn test_cuda_dequant_iq4xs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.09).cos() * 0.4).collect();
        let packed = grim_quant::quant_iq4xs(&src).expect("quant_iq4xs");
        let expected = grim_quant::dequant_iq4xs(&packed, n).expect("cpu oracle iq4xs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ4XS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq4xs");
        let actual = out.to_cpu_vec_f32().expect("readback iq4xs");
        assert_dequant_close("iq4xs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq3xxs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 0.25).collect();
        let packed = grim_quant::quant_iq3xxs(&src).expect("quant_iq3xxs");
        let expected = grim_quant::dequant_iq3xxs(&packed, n).expect("cpu oracle iq3xxs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq3xxs");
        let actual = out.to_cpu_vec_f32().expect("readback iq3xxs");
        assert_dequant_close("iq3xxs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq3s_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.08).cos() * 0.6).collect();
        let packed = grim_quant::quant_iq3s(&src).expect("quant_iq3s");
        let expected = grim_quant::dequant_iq3s(&packed, n).expect("cpu oracle iq3s");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ3S),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq3s");
        let actual = out.to_cpu_vec_f32().expect("readback iq3s");
        assert_dequant_close("iq3s", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq2xxs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 512;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
        let packed = grim_quant::quant_iq2xxs(&src).expect("quant_iq2xxs");
        let expected = grim_quant::dequant_iq2xxs(&packed, n).expect("cpu oracle iq2xxs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq2xxs");
        let actual = out.to_cpu_vec_f32().expect("readback iq2xxs");
        assert_dequant_close("iq2xxs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq2xs_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos() * 0.35).collect();
        let packed = grim_quant::quant_iq2xs(&src).expect("quant_iq2xs");
        let expected = grim_quant::dequant_iq2xs(&packed, n).expect("cpu oracle iq2xs");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ2XS),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq2xs");
        let actual = out.to_cpu_vec_f32().expect("readback iq2xs");
        assert_dequant_close("iq2xs", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_iq2s_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.06).sin() * 0.45).collect();
        let packed = match grim_quant::quant_iq2s(&src) {
            Ok(p) => p,
            Err(_) => return,
        };
        let Ok(expected) = grim_quant::dequant_iq2s(&packed, n) else {
            return;
        };
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::KQuant(KQuantScheme::IQ2S),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant iq2s");
        let actual = out.to_cpu_vec_f32().expect("readback iq2s");
        assert_dequant_close("iq2s", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_fp8_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let n = 64;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
        let packed = grim_quant::quant_fp8(&src).expect("quant_fp8");
        let expected = grim_quant::dequant_fp8(&packed, n).expect("cpu oracle fp8");
        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::FloatPack(FloatPackScheme::Fp8),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant fp8");
        let actual = out.to_cpu_vec_f32().expect("readback fp8");
        assert_dequant_close("fp8", &actual, &expected);
    }

    #[test]
    fn test_cuda_dequant_mxfp4_gpu_matches_cpu() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        // 64 values = 2 groups of 32. Hand-build codes + shared exponents so the
        // test is independent of a MXFP4 encoder (grim-quant has none public).
        // code i = (i % 16); shared exp = 127 + (group // ) so group 0 = 127,
        // group 1 = 128 (scale 2^1 = 2.0). Packed nibble: low = even element.
        let n = 64;
        let mut codes_pairs = Vec::with_capacity(n / 2);
        for i in 0..(n / 2) {
            let lo = (i * 2) % 16;
            let hi = (i * 2 + 1) % 16;
            codes_pairs.push((lo as u8) | ((hi as u8) << 4));
        }
        let exps = vec![127u8, 128u8];
        let packed = build_mxfp4_single_buffer(&codes_pairs, &exps);
        let expected = grim_quant::dequant_mxfp4(&packed, n).expect("cpu oracle mxfp4");

        let shape = Shape::new(vec![n]);
        let storage = upload_packed(
            &dev,
            &packed,
            &shape,
            DTypeStorage::FloatPack(FloatPackScheme::MxFp4),
        );
        let out = dev
            .dequantize_on_device(as_cuda_storage(storage.as_ref()))
            .expect("gpu dequant mxfp4");
        let actual = out.to_cpu_vec_f32().expect("readback mxfp4");
        assert_dequant_close("mxfp4", &actual, &expected);
    }
}

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

    fn dequantize_iq_host(
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
