//! CUDA backend with cuBLAS GEMM and device memory allocation.

pub mod kernels;

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

unsafe extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
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

    let cache_dir = std::env::current_dir()
        .unwrap_or_default()
        .join("target")
        .join("grim_cuda_cache");
    fs::create_dir_all(&cache_dir).ok();

    let cu_path = cache_dir.join(format!("{}.cu", hash));
    let ptx_path = cache_dir.join(format!("{}.ptx", hash));

    fs::write(&cu_path, src)
        .map_err(|e| Error::Backend(format!("Failed to write CUDA source: {e}")))?;

    let status = Command::new("nvcc")
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
        let status2 = Command::new("nvcc")
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
            return Err(Error::Backend("nvcc compilation failed".into()));
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
        // SAFETY: `cudaMalloc` allocates `byte_len` bytes on the selected CUDA
        // device. The pointer is initialized to null and checked on error.
        let res = unsafe { cudaMalloc(&mut dev_ptr, byte_len) };
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
        let bytes = shape.elem_count() * dtype_byte_size(&dtype);

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
        // SAFETY: `cudaMalloc` allocates `bytes` bytes on the current CUDA device.
        // The pointer is initialized to null and checked on error; the device was
        // selected by the preceding `cudaSetDevice`.
        let res = unsafe { cudaMalloc(&mut dev_ptr, bytes) };
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
        // holds packed codes smaller than `elem_count * 4` bytes. Download the
        // raw byte payload and dequantize on the host via grim-quant, mirroring
        // the ROCm `to_cpu_vec_f32` -> `dequant_cpu` contract so that
        // `transpose_last_two`/`Linear::load` (which call `to_vec_f32` on the
        // raw quantized weight) keep working now that CUDA materialization no
        // longer pre-dequantizes these formats.
        if self.dtype.is_quantized() {
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
#[derive(Debug, Clone, Copy)]
pub struct CublasHandle(pub *mut c_void);
// SAFETY: `CublasHandle` wraps a raw CUDA driver handle. `Send` is safe because
// the handle is bound to a specific CUDA context on one device, and the driver
// tracks it independently of the creating thread. `Sync` is safe because the
// cuBLAS API serializes concurrent calls through its internal stream/context lock.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

/// CUDA device handle.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub(crate) ordinal: usize,
    cublas_handle: Arc<Mutex<Option<CublasHandle>>>,
}

// SAFETY: `CudaDevice` contains only `usize` and `Arc<Mutex<Option<CublasHandle>>>`.
// Both fields are `Send + Sync` by construction, so `CudaDevice` is
// automatically `Send + Sync` — the explicit `unsafe impl` is retained
// only to document the invariant; it could be removed entirely.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Creates a device reference for the given ordinal; returns Err if cuBLAS init fails.
    pub fn new(ordinal: usize) -> Result<Self> {
        let mut handle_ptr: *mut c_void = std::ptr::null_mut();
        let cublas_handle = unsafe {
            if cublasCreate_v2(&mut handle_ptr) == CUBLAS_STATUS_SUCCESS {
                Some(CublasHandle(handle_ptr))
            } else {
                return Err(Error::Backend(format!(
                    "cublasCreate_v2 failed for CUDA device {ordinal}"
                )));
            }
        };
        Ok(Self {
            ordinal,
            cublas_handle: Arc::new(Mutex::new(cublas_handle)),
        })
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

    /// Returns the cuBLAS handle for this device, lazily initializing if needed.
    pub fn get_cublas_handle(&self) -> Result<CublasHandle> {
        let mut handle = self.cublas_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = *handle {
            Ok(h)
        } else {
            let mut handle_ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `cublasCreate_v2` initializes a new cuBLAS handle.
            // `handle_ptr` is a valid null pointer that receives the new handle.
            let res = unsafe { cublasCreate_v2(&mut handle_ptr) };
            if res == CUBLAS_STATUS_SUCCESS {
                let h = CublasHandle(handle_ptr);
                *handle = Some(h);
                Ok(h)
            } else {
                Err(Error::Backend(format!(
                    "cublasCreate failed with status {}",
                    res
                )))
            }
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
        out: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
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

        // 13 kernel args: q, k, v, out, out_max, out_sum, num_heads, num_kv_heads, head_dim, seq_len, kv_seq_len, cache_offset, inv_sqrt_d
        let mut args: [*mut c_void; 13] = [
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

        let handle = self.get_cublas_handle()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;

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

        // cuBLAS is column-major; grim is row-major. Transpose trick:
        // C_row(M,N) = A_row(M,K) * B_row(K,N) → C_col(N,M) = B_col(N,K) * A_col(K,M).
        // Pass b_ptr (K,N col-major) as cuBLAS's A with lda=K; a_ptr (M,K col-major) as B with ldb=M.
        // Result C_col(N,M) with ldc=N, read row-major as C_row(M,N).
        unsafe {
            let status = cublasSgemm_v2(
                handle.0,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32, // m_cublas = rows of op(A_c) = N
                m as i32, // n_cublas = cols of op(B_c) = M
                k as i32, // k_cublas = K (inner)
                &alpha,
                b_ptr as *const f32, // A_cublas ptr = B
                k as i32,            // lda = K   (B is (K,N) col-major)
                a_ptr as *const f32, // B_cublas ptr = A
                m as i32,            // ldb = M   (A is (M,K) col-major)
                &beta,
                out_ptr as *mut f32,
                n as i32, // ldc = N
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

        Ok((Box::new(out_storage), compute_handle))
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

    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            out_shape,
            out_max,
            out_sum,
        )
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
        dim: usize,
        base: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rope x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;

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

        let mut out = vec![0.0f32; batch * seq_len * dim_dinner];
        for b_idx in 0..batch {
            for d_idx in 0..dim_dinner {
                let mut h = vec![0.0f32; dim_dstate];
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
        let (out_storage, _h) =
            self.qkv_attention(q, k, v, num_kv_heads, seq_len, 0, out_shape, None, None)?;
        let _ = num_heads;
        let _ = head_dim;
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
        let (out_storage, _h) =
            self.qkv_attention(q, k, v, num_heads, kv_seq_len, 0, out_shape, None, None)?;
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
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // Fused Q8_0 GPU GEMM: requires K % 32 == 0 and GPU-resident operands.
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

                let handle =
                    self.launch_quantized_matmul_q8_0(a_ptr, b_ptr, scales_ptr, out_ptr, m, n, k)?;
                return Ok((Box::new(out_storage), handle));
            }
        }

        // CPU fallback: dequant with Q8_0 convention (scale * i8) for numerical agreement.
        tracing::warn!("CUDA quantized_matmul: falling back to CPU execution");
        let a_vec = a.to_cpu_vec_f32()?;
        let mut b_dequant = vec![0.0f32; k * n];
        let blocks_per_col = k / 32;

        let b_bytes = if let Some(ref c_s) = b_packed.as_any().downcast_ref::<CudaStorage>() {
            let mut host_bytes = vec![0u8; c_s.bytes()];
            if let Some(dev_ptr) = c_s.device_ptr {
                unsafe {
                    let res = cudaMemcpy(
                        host_bytes.as_mut_ptr() as *mut c_void,
                        dev_ptr as *const c_void,
                        c_s.bytes(),
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

        for col in 0..n {
            for block in 0..blocks_per_col {
                let scale_idx = col * blocks_per_col + block;
                let scale = if scale_idx < b_scales.len() {
                    b_scales[scale_idx]
                } else {
                    1.0f32
                };
                for i in 0..32 {
                    let byte_offset = (col * blocks_per_col + block) * 32 + i;
                    let byte_val = if byte_offset < b_bytes.len() {
                        b_bytes[byte_offset]
                    } else {
                        0u8
                    };
                    let q_val = (byte_val as i8) as f32;
                    let r = block * 32 + i;
                    if r < k {
                        b_dequant[r * n + col] = q_val * scale;
                    }
                }
            }
        }

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
                handle.0,
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
            .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &b_scales, &out_shape)
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
            .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], &out_shape)
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
            .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &b_scales, &out_shape)
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
