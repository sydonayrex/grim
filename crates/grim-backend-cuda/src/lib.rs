//! CUDA backend for Grim.
//!
//! Provides the `CudaDevice` and `CudaStorage` structs implementing the `BackendDevice`
//! and `BackendStorage` traits from `grim-tensor` by wrapping CUDA runtime APIs and cuBLAS.

pub mod kernels;

use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::sync::LazyLock;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{ArithType, DType, QuantProvenance, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

// ---------- CUDA FFI bindings ----------

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

#[allow(dead_code)]
unsafe extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;

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
unsafe impl Send for SendCmodule {}
unsafe impl Sync for SendCmodule {}

static JIT_CACHE: LazyLock<Mutex<HashMap<u64, SendCmodule>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn compile_and_load_kernel(src: &str, device_ordinal: usize) -> Result<CUmodule> {
    let hash = seahash::hash(src.as_bytes());
    {
        let cache = JIT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&module) = cache.get(&hash) {
            return Ok(module.0);
        }
    }

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

    fs::write(&cu_path, src).map_err(|e| Error::Backend(format!("Failed to write CUDA source: {e}")))?;

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
    unsafe {
        let select_res = cudaSetDevice(device_ordinal as i32);
        if select_res != 0 {
            return Err(Error::Backend(format!("cudaSetDevice failed: {}", select_res)));
        }

        let mut ptx_bytes = ptx_content.into_bytes();
        ptx_bytes.push(0); // Null-terminate the PTX string!
        let load_res = cuModuleLoadData(&mut module, ptx_bytes.as_ptr() as *const c_void);
        if load_res != 0 {
            return Err(Error::Backend(format!("cuModuleLoadData failed with error {}", load_res)));
        }
    }

    let mut cache = JIT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.insert(hash, SendCmodule(module));
    Ok(module)
}

/// A handle to a queued CUDA stream operation.
#[derive(Debug)]
pub struct CudaHandle {
    pub completed: Arc<Mutex<bool>>,
}

impl ComputeHandle for CudaHandle {
    /// Blocks the current host thread until all operations tracked by this handle
    /// have completed on the CUDA device.
    fn synchronize(&self) -> Result<()> {
        let mut completed = self.completed.lock().unwrap_or_else(|e| e.into_inner());
        if !*completed {
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

    /// Checks if the CUDA operations tracked by this handle have finished executing.
    fn is_ready(&self) -> bool {
        *self.completed.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// CUDA-side tensor storage.
#[derive(Debug)]
pub struct CudaStorage {
    /// Opaque pointer to the GPU memory block allocated for this storage.
    device_ptr: Option<u64>,
    /// Size of the allocated buffer in bytes.
    bytes: usize,
    /// Shape of the tensor stored.
    shape: Shape,
    /// Data type of the elements.
    dtype: DType,
    /// Provenance of the tensor data, identifying if it was externally quantized.
    provenance: QuantProvenance,
    /// Ordinal index of the GPU device where this buffer is allocated.
    ordinal: usize,
}

impl CudaStorage {
    /// Allocates raw GPU memory on a CUDA device.
    pub fn alloc_gpu(shape: &Shape, dtype: DType, device_ordinal: usize) -> Result<Self> {
        let bytes = shape.elem_count() * dtype_byte_size(&dtype);
        
        let select_res = unsafe { cudaSetDevice(device_ordinal as i32) };
        if select_res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                device_ordinal
            )));
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
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

    /// Copies data from host to GPU using `cudaMalloc` and `cudaMemcpy`.
    pub fn copy_from_host(
        host_data: &[f32],
        shape: &Shape,
        dtype: DType,
        device_ordinal: usize,
    ) -> Result<Self> {
        let storage = Self::alloc_gpu(shape, dtype, device_ordinal)?;
        let dev_ptr = storage.device_ptr.unwrap() as *mut c_void;

        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                host_data.as_ptr() as *const c_void,
                storage.bytes,
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
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

    /// Returns the shape of the tensor.
    pub fn shape_metadata(&self) -> &Shape {
        &self.shape
    }

    /// Returns the ordinal index of the device owning this storage.
    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns the raw device pointer if allocated.
    pub fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }

    /// Returns the size of the storage in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        if let Some(ptr_val) = self.device_ptr {
            if ptr_val != 0 {
                unsafe {
                    let _ = cudaFree(ptr_val as *mut c_void);
                }
            }
        }
    }
}

impl BackendStorage for CudaStorage {
    /// Gets the data type of the storage.
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Gets the quantization provenance of the storage.
    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    /// Gets the shape of the storage.
    fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copies the GPU device buffer content back to host memory as an F32 vector.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        let dev_ptr = self.device_ptr.ok_or_else(|| {
            Error::Backend("CudaStorage has no valid device pointer".into())
        })? as *mut c_void;

        let mut host_data = vec![0.0f32; self.shape.elem_count()];
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

    /// Returns `self` as `Any` to allow internal downcasting in the backend.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Wrapper to make cuBLAS FFI types Send + Sync.
#[derive(Debug, Clone, Copy)]
pub struct CublasHandle(pub *mut c_void);
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

/// CUDA device handle.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    /// Ordinal index of this CUDA device.
    pub(crate) ordinal: usize,
    cublas_handle: Arc<Mutex<Option<CublasHandle>>>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Constructs a new device reference for the given ordinal.
    pub fn new(ordinal: usize) -> Self {
        let mut handle_ptr: *mut c_void = std::ptr::null_mut();
        let mut cublas_handle = None;
        unsafe {
            if cublasCreate_v2(&mut handle_ptr) == CUBLAS_STATUS_SUCCESS {
                cublas_handle = Some(CublasHandle(handle_ptr));
            } else {
                eprintln!(
                    "[CudaDevice::new] Warning: cublasCreate_v2 failed for device {}. \
                     Operations will retry lazily on first matmul.",
                    ordinal
                );
            }
        }
        Self {
            ordinal,
            cublas_handle: Arc::new(Mutex::new(cublas_handle)),
        }
    }

    /// Probes the system for available CUDA GPUs and returns a device instance for each.
    pub fn probe() -> Result<Vec<CudaDevice>> {
        if let Ok(s) = std::env::var("GRIM_CUDA_ORDINAL_OVERRIDE") {
            if let Ok(n) = s.parse::<usize>() {
                return Ok(vec![CudaDevice::new(n)]);
            }
        }

        let mut count: i32 = 0;
        let res = unsafe { cudaGetDeviceCount(&mut count) };
        if res == cudaSuccess && count > 0 {
            let mut devices = Vec::with_capacity(count as usize);
            for i in 0..count {
                devices.push(CudaDevice::new(i as usize));
            }
            return Ok(devices);
        }

        Ok(vec![])
    }

    /// Gets the cuBLAS handle initialized for this device.
    pub fn get_cublas_handle(&self) -> Result<CublasHandle> {
        let mut handle = self.cublas_handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = *handle {
            Ok(h)
        } else {
            let mut handle_ptr: *mut c_void = std::ptr::null_mut();
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

    /// Gets the ordinal of the CUDA device.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}


impl BackendDevice for CudaDevice {
    /// Allocates a zero-initialized tensor buffer on the CUDA device.
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let storage = CudaStorage::alloc_gpu(shape, dtype, self.ordinal)?;
        let dev_ptr = storage.device_ptr.expect("zeros: device_ptr was null after alloc_gpu") as *mut c_void;

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

    /// Performs matrix multiplication on the CUDA device via cuBLAS.
    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("matmul a is not CudaStorage".into())
        })?;
        let b_storage = b.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("matmul b is not CudaStorage".into())
        })?;

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
            return Err(Error::Shape(format!("expected out [{m},{n}], got {out_shape:?}")));
        }

        let dtype_out = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        };
        let out_storage = CudaStorage::alloc_gpu(out_shape, dtype_out, self.ordinal)?;

        let handle = self.get_cublas_handle()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let a_ptr = a_storage.device_ptr.expect("matmul: A device_ptr was null") as *const c_void;
        let b_ptr = b_storage.device_ptr.expect("matmul: B device_ptr was null") as *const c_void;
        let out_ptr = out_storage.device_ptr.expect("matmul: out device_ptr was null") as *mut c_void;

        // Column-major cublasSgemm: A is KxM (lda=k), B is NxK (ldb=n), C is NxM (ldc=n).
        // For C = A @ B (where A is row-major MxK and B is row-major KxN),
        // cublas computes op(B) @ op(A) = B^T @ A^T (cublas is column-major).
        //
        // For square matrices (M=K=N), row-major flat == column-major flat indexing,
        // so this happens to compute the correct A @ B. For non-square matrices
        // (M≠K or K≠N), the dimension mapping is broken — a proper fix requires
        // explicit data reordering or different transposition flags.
        unsafe {
            let status = cublasSgemm_v2(
                handle.0,
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

        Ok((Box::new(out_storage), compute_handle))
    }

    /// Performs elementwise addition on the CUDA device.
    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("add a is not CudaStorage".into())
        })?;
        let b_storage = b.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("add b is not CudaStorage".into())
        })?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_add").unwrap();
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {}", res)));
            }

            let mut a_ptr = a_storage.device_ptr.unwrap() as *mut c_void;
            let mut b_ptr = b_storage.device_ptr.unwrap() as *mut c_void;
            let mut out_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
            let mut n_i = n as i32;

            let mut args = [
                &mut a_ptr as *mut *mut c_void as *mut c_void,
                &mut b_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut n_i as *mut i32 as *mut c_void,
            ];

            let block_size = 256;
            let grid_size = (n + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!("cuLaunchKernel failed: {}", launch_res)));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Performs elementwise multiplication on the CUDA device.
    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("mul a is not CudaStorage".into())
        })?;
        let b_storage = b.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("mul b is not CudaStorage".into())
        })?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_mul").unwrap();
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {}", res)));
            }

            let mut a_ptr = a_storage.device_ptr.unwrap() as *mut c_void;
            let mut b_ptr = b_storage.device_ptr.unwrap() as *mut c_void;
            let mut out_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
            let mut n_i = n as i32;

            let mut args = [
                &mut a_ptr as *mut *mut c_void as *mut c_void,
                &mut b_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut n_i as *mut i32 as *mut c_void,
            ];

            let block_size = 256;
            let grid_size = (n + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!("cuLaunchKernel failed: {}", launch_res)));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Performs elementwise SiLU-multiplication (SwiGLU gate) on the CUDA device.
    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_storage = gate.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("silu_mul gate is not CudaStorage".into())
        })?;
        let up_storage = up.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("silu_mul up is not CudaStorage".into())
        })?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_silu_mul").unwrap();
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {}", res)));
            }

            let mut gate_ptr = gate_storage.device_ptr.unwrap() as *mut c_void;
            let mut up_ptr = up_storage.device_ptr.unwrap() as *mut c_void;
            let mut out_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
            let mut n_i = n as i32;

            let mut args = [
                &mut gate_ptr as *mut *mut c_void as *mut c_void,
                &mut up_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut n_i as *mut i32 as *mut c_void,
            ];

            let block_size = 256;
            let grid_size = (n + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!("cuLaunchKernel failed: {}", launch_res)));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Performs RMS Normalization on the CUDA device.
    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("rms_norm x is not CudaStorage".into())
        })?;
        let w_storage = w.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("rms_norm w is not CudaStorage".into())
        })?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let total = out.elem_count();
        let row_len = out.dims()[out.dims().len() - 1];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_rms_norm").unwrap();
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {}", res)));
            }

            let mut x_ptr = x_storage.device_ptr.unwrap() as *mut c_void;
            let mut w_ptr = w_storage.device_ptr.unwrap() as *mut c_void;
            let mut out_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
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

            let block_size = 256;
            let grid_size = (total + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!("cuLaunchKernel failed: {}", launch_res)));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Performs Softmax along the last dimension on the CUDA device.
    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("softmax x is not CudaStorage".into())
        })?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let total = out.elem_count();
        let last_dim = out.dims()[out.dims().len() - 1];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_softmax").unwrap();
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {}", res)));
            }

            let mut x_ptr = x_storage.device_ptr.unwrap() as *mut c_void;
            let mut out_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
            let mut last_dim_i = last_dim as i32;
            let mut total_i = total as i32;

            let mut args = [
                &mut x_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut last_dim_i as *mut i32 as *mut c_void,
                &mut total_i as *mut i32 as *mut c_void,
            ];

            let block_size = 256;
            let grid_size = (total + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!("cuLaunchKernel failed: {}", launch_res)));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Performs embedding lookup on the CUDA device.
    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let weight_storage = weight.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("embedding weight is not CudaStorage".into())
        })?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let num_indices = indices.len();
        let embedding_dim = out.dims()[out.dims().len() - 1];

        // Allocate device memory for indices
        let mut dev_indices_ptr: *mut c_void = std::ptr::null_mut();
        let size_indices = num_indices * 4;
        unsafe {
            let res = cudaMalloc(&mut dev_indices_ptr, size_indices);
            if res != cudaSuccess {
                return Err(Error::Backend(format!("cudaMalloc for indices failed: {}", res)));
            }
            let res = cudaMemcpy(
                dev_indices_ptr,
                indices.as_ptr() as *const c_void,
                size_indices,
                cudaMemcpyHostToDevice,
            );
            if res != cudaSuccess {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!("cudaMemcpy for indices failed: {}", res)));
            }
        }

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_embedding").unwrap();
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {}", res)));
            }

            let mut w_ptr = weight_storage.device_ptr.unwrap() as *mut c_void;
            let mut out_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
            let mut emb_dim_i = embedding_dim as i32;
            let mut num_idx_i = num_indices as i32;

            let mut args = [
                &mut w_ptr as *mut *mut c_void as *mut c_void,
                &mut dev_indices_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut emb_dim_i as *mut i32 as *mut c_void,
                &mut num_idx_i as *mut i32 as *mut c_void,
            ];

            let block_size = 256;
            let total_threads = num_indices * embedding_dim;
            let grid_size = (total_threads + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32, 1, 1,
                block_size as u32, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!("cuLaunchKernel failed: {}", launch_res)));
            }
        }

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

    fn advise(&self, _storage: &dyn BackendStorage, _advice: grim_tensor::backend::MemAdvice) -> Result<()> {
        // CUDA backend: MemAdvice is currently a no-op
        Ok(())
    }
}

/// Helper function to retrieve the size in bytes of a data type.
fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 | ArithType::BF16 => 2,
        ArithType::I64 => 8,
        ArithType::U8 => 1,
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

        let (out_storage, handle) = dev.matmul(a_storage.as_ref(), b_storage.as_ref(), &out_shape).unwrap();
        handle.synchronize().unwrap();

        let res = out_storage.to_cpu_vec_f32().unwrap();
        // [1 2] @ [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
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
        assert_eq!(out_add.to_cpu_vec_f32().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);

        // 2. Mul
        let (out_mul, h) = dev.mul(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        assert_eq!(out_mul.to_cpu_vec_f32().unwrap(), vec![5.0, 12.0, 21.0, 32.0]);

        // 3. SiLU Mul
        let (out_silu, h) = dev.silu_mul(a.as_ref(), b.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        let res_silu = out_silu.to_cpu_vec_f32().unwrap();
        let expected_silu0 = (1.0f32 / (1.0f32 + (-1.0f32).exp())) * 5.0f32;
        assert!((res_silu[0] - expected_silu0).abs() < 1e-4);

        // 4. RMS Norm
        let weight_data = vec![1.0, 1.0, 1.0, 1.0];
        let weight = dev.from_cpu(&weight_data, &shape, DType::F32).unwrap();
        let (out_rms, h) = dev.rms_norm(a.as_ref(), weight.as_ref(), 1e-5, &shape).unwrap();
        h.synchronize().unwrap();
        let res_rms = out_rms.to_cpu_vec_f32().unwrap();
        // RMS of [1, 2, 3, 4] is sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386
        let rms_val = 7.5f32.sqrt();
        assert!((res_rms[0] - 1.0 / rms_val).abs() < 1e-4);

        // 5. Softmax
        let (out_sm, h) = dev.softmax(a.as_ref(), &shape).unwrap();
        h.synchronize().unwrap();
        let res_sm = out_sm.to_cpu_vec_f32().unwrap();
        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp() + 4.0f32.exp();
        assert!((res_sm[0] - 1.0f32.exp() / sum_exp).abs() < 1e-4);

        // 6. Embedding
        let weight_emb_data = vec![
            10.0, 20.0,
            30.0, 40.0,
            50.0, 60.0,
        ];
        let weight_emb = dev.from_cpu(&weight_emb_data, &Shape::new(vec![3, 2]), DType::F32).unwrap();
        let indices = vec![2u32, 0u32];
        let out_emb_shape = Shape::new(vec![2, 2]);
        let (out_emb, h) = dev.embedding(weight_emb.as_ref(), &indices, &out_emb_shape).unwrap();
        h.synchronize().unwrap();
        let res_emb = out_emb.to_cpu_vec_f32().unwrap();
        assert_eq!(res_emb, vec![50.0, 60.0, 10.0, 20.0]);
    }
}
