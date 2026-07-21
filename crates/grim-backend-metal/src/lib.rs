//! Metal compatibility backend for Grim.
//!
//! Provides the `MetalDevice` and `MetalStorage` structs implementing the `BackendDevice`
//! and `BackendStorage` traits from `grim-tensor`, enabling Metal device target support (MSL).
//! Implements a robust Unified Memory Architecture (UMA) zero-copy FFI and CPU-fallback execution
//! pipeline to ensure full capability compatibility on all supported targets.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{DType, QuantProvenance, Storage as DTypeStorage};
#[cfg(target_vendor = "apple")]
use grim_tensor::dtype::ArithType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

use grim_backend_cpu::{CpuDevice, CpuStorage};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

#[cfg(embed_metallib)]
const METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));

/// Typed errors specific to the Metal backend.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MetalError {
    /// Failure during device or command queue initialization.
    #[error("Metal initialization/FFI error: {0}")]
    Ffi(String),
    /// Failure when loading or compiling MSL code.
    #[error("Metal shader compilation failed: {0}")]
    Compilation(String),
    /// Data type not supported by the Metal shader kernels.
    #[error("Metal only supports F32 operations, got dtype: {0:?}")]
    UnsupportedDType(DType),
    /// Failure to allocate graphics memory.
    #[error("Metal buffer allocation failed: {0}")]
    AllocationFailed(String),
    /// Internal context state error.
    #[error("Metal context error: {0}")]
    Context(String),
    /// Retrieved null pointer from graphics memory mapping.
    #[error("Metal buffer contents is null")]
    NullBuffer,
    /// Storage size or layout mismatch.
    #[error("Metal storage data mismatch: {0}")]
    DataMismatch(String),
}

impl From<MetalError> for Error {
    /// Maps a backend-specific MetalError to the general tensor-level Error.
    fn from(err: MetalError) -> Self {
        Error::Backend(err.to_string())
    }
}

/// Target-agnostic GPU memory usage/residency profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// Memory visible to both CPU and GPU (Shared/Host-visible).
    Shared,
    /// Memory visible only to the GPU (Device-local/Private).
    Private,
}

#[cfg(target_vendor = "apple")]
impl BufferUsage {
    /// Maps our usage abstraction to the platform MTLResourceOptions.
    pub fn to_mtl_options(self) -> objc2_metal::MTLResourceOptions {
        match self {
            BufferUsage::Shared => objc2_metal::MTLResourceOptions::StorageModeShared,
            BufferUsage::Private => objc2_metal::MTLResourceOptions::StorageModePrivate,
        }
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
struct MetalPipelines {
    add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu_mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    embedding: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qkv_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    kv_dequant_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Global, lazy-initialized Metal compute context.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalContext {
    /// Main platform device handle.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Shared execution queue.
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Precompiled MSL compute pipeline kernels.
    pub pipelines: std::sync::Arc<MetalPipelines>,
}

#[cfg(target_vendor = "apple")]
static METAL_CONTEXT: std::sync::OnceLock<std::result::Result<MetalContext, MetalError>> = std::sync::OnceLock::new();

#[cfg(target_vendor = "apple")]
impl MetalContext {
    /// Returns a static reference to the shared context, lazy-initializing it if necessary.
    pub fn get() -> std::result::Result<&'static MetalContext, MetalError> {
        METAL_CONTEXT.get_or_init(|| {
            use objc2_metal::MTLCreateSystemDefaultDevice;
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| MetalError::Ffi("No default Metal device found".into()))?;
            let command_queue = device
                .newCommandQueue()
                .ok_or_else(|| MetalError::Ffi("Failed to create MTLCommandQueue".into()))?;

            let msl_source = include_str!("kernels.msl");
            let hash = fnv1a_hash(msl_source);
            let mut library: Option<Retained<objc2_metal::MTLLibrary>> = None;

            if let Some(cache_dir) = get_cache_dir() {
                let _ = std::fs::create_dir_all(&cache_dir);
                let cached_path = cache_dir.join(format!("grim_metal_{:016x}.metallib", hash));
                #[cfg(embed_metallib)]
                {
                    if !cached_path.exists() {
                        let _ = std::fs::write(&cached_path, METALLIB_BYTES);
                    }
                }
                if cached_path.exists() {
                    unsafe {
                        use objc2::runtime::AnyObject;
                        use objc2::{msg_send, class};
                        let nsurl_class = class!(NSURL);
                        let path_str = objc2::ns_string!(cached_path.to_str().unwrap());
                        let url: *mut AnyObject = msg_send![nsurl_class, fileURLWithPath: path_str];
                        let mut error: *mut AnyObject = std::ptr::null_mut();
                        let loaded_lib: Option<Retained<objc2_metal::MTLLibrary>> = msg_send![&device, newLibraryWithURL: url, error: &mut error];
                        if let Some(lib) = loaded_lib {
                            library = Some(lib);
                        }
                    }
                }

                if library.is_none() {
                    if let Ok(temp_dir) = tempfile::tempdir() {
                        let air_path = temp_dir.path().join("kernel.air");
                        let msl_path = temp_dir.path().join("kernel.metal");
                        if std::fs::write(&msl_path, msl_source).is_ok() {
                            let status1 = std::process::Command::new("xcrun")
                                .args(&["-sdk", "macosx", "metal", "-c", "-o", air_path.to_str().unwrap(), msl_path.to_str().unwrap()])
                                .status();
                            if let Ok(s1) = status1 {
                                if s1.success() {
                                    let status2 = std::process::Command::new("xcrun")
                                        .args(&["-sdk", "macosx", "metallib", "-o", cached_path.to_str().unwrap(), air_path.to_str().unwrap()])
                                        .status();
                                    if let Ok(s2) = status2 {
                                        if s2.success() {
                                            unsafe {
                                                use objc2::runtime::AnyObject;
                                                use objc2::{msg_send, class};
                                                let nsurl_class = class!(NSURL);
                                                let path_str = objc2::ns_string!(cached_path.to_str().unwrap());
                                                let url: *mut AnyObject = msg_send![nsurl_class, fileURLWithPath: path_str];
                                                let mut error: *mut AnyObject = std::ptr::null_mut();
                                                let loaded_lib: Option<Retained<objc2_metal::MTLLibrary>> = msg_send![&device, newLibraryWithURL: url, error: &mut error];
                                                if let Some(lib) = loaded_lib {
                                                    library = Some(lib);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let library = if let Some(lib) = library {
                lib
            } else {
                device
                    .newLibraryWithSource_options_error(&objc2::ns_string!(msl_source), None)
                    .map_err(|e| MetalError::Compilation(format!("{:?}", e)))?
            };

            let get_pipeline = |name: &str| -> std::result::Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
                let function = library
                    .newFunctionWithName(&objc2::ns_string!(name))
                    .ok_or_else(|| MetalError::Compilation(format!("MSL function {} not found", name)))?;
                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| MetalError::Compilation(format!("Failed to create pipeline for {}: {:?}", name, e)))
            };

            let pipelines = std::sync::Arc::new(MetalPipelines {
                add: get_pipeline("grim_add")?,
                mul: get_pipeline("grim_mul")?,
                silu_mul: get_pipeline("grim_silu_mul")?,
                rms_norm: get_pipeline("grim_rms_norm")?,
                softmax: get_pipeline("grim_softmax")?,
                embedding: get_pipeline("grim_embedding")?,
                matmul: get_pipeline("grim_matmul")?,
                qkv_attn: get_pipeline("grim_qkv_attention")?,
                kv_dequant_attn: get_pipeline("grim_kv_dequant_attention")?,
            });

            Ok(MetalContext {
                device,
                command_queue,
                pipelines,
            })
        }).as_ref().map_err(|e| e.clone())
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalHandle {
    /// Command buffer containing operations.
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalHandle;


impl ComputeHandle for MetalHandle {
    /// Blocks the host thread until Metal operations tracked by this handle complete.
    fn synchronize(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    /// Checks if the Metal operations tracked by this handle have finished.
    fn is_ready(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandBufferStatus;
            self.command_buffer.status() == MTLCommandBufferStatus::Completed
        }
        #[cfg(not(target_vendor = "apple"))]
        true
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalStorage {
    buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    data: Option<std::sync::Mutex<Vec<u8>>>,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
}

#[cfg(target_vendor = "apple")]
impl Drop for MetalStorage {
    fn drop(&mut self) {
        self.buffer = None;
        self.data = None;
    }
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalStorage {
    data: std::sync::Mutex<Vec<u8>>,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
}

impl BackendStorage for MetalStorage {
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
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref buffer) = self.buffer {
                let contents = buffer.contents() as *const f32;
                if contents.is_null() {
                    return Err(Error::Backend("Metal buffer contents is null".into()));
                }
                let mut out = vec![0.0f32; self.shape.elem_count()];
                unsafe {
                    std::ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), out.len());
                }
                Ok(out)
            } else if let Some(ref data) = self.data {
                let data_guard = data.lock().unwrap();
                let elem_count = self.shape.elem_count();
                let mut out = vec![0.0f32; elem_count];
                let bytes = elem_count * dtype_byte_size(&self.dtype)?;
                if data_guard.len() < bytes {
                    return Err(Error::from(MetalError::DataMismatch("CPU storage buffer size mismatch".into())));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(data_guard.as_ptr(), out.as_mut_ptr() as *mut u8, bytes);
                }
                Ok(out)
            } else {
                Err(Error::Backend("MetalStorage has no buffer or fallback data".into()))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let data_guard = self.data.lock().unwrap();
            let elem_count = self.shape.elem_count();
            let mut out = vec![0.0f32; elem_count];
            let bytes = elem_count * dtype_byte_size(&self.dtype)?;
            if data_guard.len() < bytes {
                return Err(Error::from(MetalError::DataMismatch("CPU storage buffer size mismatch".into())));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(data_guard.as_ptr(), out.as_mut_ptr() as *mut u8, bytes);
            }
            Ok(out)
        }
    }

    /// Returns `self` as `Any` to allow internal downcasting in the backend.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    ordinal: usize,
    inner: Option<std::sync::Arc<MetalDeviceInner>>,
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
struct MetalDeviceInner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: std::sync::Arc<MetalPipelines>,
    active_command_buffer: std::sync::Mutex<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    ordinal: usize,
}

#[allow(dead_code)]
fn fnv1a_hash(s: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in s.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

#[allow(dead_code)]
fn get_cache_dir() -> Option<std::path::PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        Some(std::path::PathBuf::from(home).join(".cache").join("grim_metal_cache"))
    } else if let Ok(user_profile) = std::env::var("USERPROFILE") {
        Some(std::path::PathBuf::from(user_profile).join(".cache").join("grim_metal_cache"))
    } else {
        None
    }
}

impl MetalDevice {
    /// Constructs a new device reference for the given ordinal.
    ///
    /// If hardware initialization fails on Apple platforms, it logs a warning and returns
    /// a device in fallback mode rather than panicking.
    pub fn new(ordinal: usize) -> Self {
        #[cfg(target_vendor = "apple")]
        {
            match Self::try_new(ordinal) {
                Ok(dev) => dev,
                Err(e) => {
                    eprintln!("[MetalDevice::new] WARNING: Failed to initialize Metal hardware. Error: {:?}", e);
                    Self { ordinal, inner: None }
                }
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Self { ordinal }
        }
    }

    /// Fallible constructor propagating Metal FFI initialization errors.
    pub fn try_new(ordinal: usize) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            let ctx = MetalContext::get()?;
            let inner = std::sync::Arc::new(MetalDeviceInner {
                device: ctx.device.clone(),
                command_queue: ctx.command_queue.clone(),
                pipelines: ctx.pipelines.clone(),
                active_command_buffer: std::sync::Mutex::new(None),
            });
            Ok(Self {
                ordinal,
                inner: Some(inner),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Self { ordinal })
        }
    }

    #[cfg(target_vendor = "apple")]
    /// Acquires the active command buffer or creates a new one if none is active or it has been committed.
    pub fn get_or_create_command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let inner = self.inner.as_ref().ok_or_else(|| Error::from(MetalError::Context("Device inner is None".into())))?;
        let mut active = inner.active_command_buffer.lock().unwrap();
        if let Some(ref buf) = *active {
            use objc2_metal::MTLCommandBufferStatus;
            if buf.status() == MTLCommandBufferStatus::NotEnqueued {
                return Ok(buf.clone());
            }
        }
        let new_buf = inner.command_queue.commandBuffer()
            .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create command buffer".into())))?;
        *active = Some(new_buf.clone());
        Ok(new_buf)
    }

    /// Flushes (commits) any active deferred command buffer.
    pub fn flush(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let mut active = inner.active_command_buffer.lock().unwrap();
                if let Some(buf) = active.take() {
                    buf.commit();
                }
            }
        }
        Ok(())
    }

    #[cfg(target_vendor = "apple")]
    /// Allocates a new buffer by copying data from the host.
    pub fn new_buffer_with_bytes(&self, bytes: &[u8], usage: BufferUsage) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let inner = self.inner.as_ref().ok_or_else(|| Error::from(MetalError::Context("Device inner is None".into())))?;
        let options = usage.to_mtl_options();
        let buffer = unsafe {
            inner.device.newBufferWithBytes_length_options(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                options,
            )
        }.ok_or_else(|| Error::from(MetalError::AllocationFailed("Failed to allocate MTLBuffer with bytes".into())))?;
        Ok(buffer)
    }

    /// Returns the ordinal of this device.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Probes the system for available Metal GPUs.
    pub fn probe() -> Result<Vec<MetalDevice>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Ok(dev) = MetalDevice::try_new(0) {
                if dev.inner.is_some() {
                    return Ok(vec![dev]);
                }
            }
            Ok(vec![])
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok(vec![])
    }
}

impl BackendDevice for MetalDevice {
    /// Allocates a zero-initialized tensor buffer on the Metal device.
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let bytes = shape.elem_count()
            .checked_mul(dtype_byte_size(&dtype)?)
            .ok_or_else(|| Error::from(MetalError::AllocationFailed("Buffer size overflow".into())))?;
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use objc2_metal::MTLResourceOptions;
                let buffer = inner
                    .device
                    .newBufferWithLength_options(bytes as u64, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| Error::from(MetalError::AllocationFailed("Failed to allocate Metal buffer".into())))?;

                let contents = buffer.contents();
                if !contents.is_null() {
                    unsafe {
                        std::ptr::write_bytes(contents, 0, bytes);
                    }
                }

                Ok(Box::new(MetalStorage {
                    buffer: Some(buffer),
                    data: None,
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            } else {
                Ok(Box::new(MetalStorage {
                    buffer: None,
                    data: Some(std::sync::Mutex::new(vec![0u8; bytes])),
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(vec![0u8; bytes]),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }
    /// Performs matrix multiplication on the Metal device.
    /// Performs matrix multiplication on the Metal device.
    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            #[link(name = "Accelerate", kind = "framework")]
            extern "C" {
                fn cblas_sgemm(
                    layout: i32,
                    trans_a: i32,
                    trans_b: i32,
                    m: i32,
                    n: i32,
                    k: i32,
                    alpha: f32,
                    a: *const f32,
                    lda: i32,
                    b: *const f32,
                    ldb: i32,
                    beta: f32,
                    c: *mut f32,
                    ldc: i32,
                );
            }

            if self.inner.is_none() {
                // Device-absent fallback path via Accelerate framework sgemm
                let a_vec = a.to_cpu_vec_f32()?;
                let b_vec = b.to_cpu_vec_f32()?;
                let dims_a = a.shape().dims();
                let dims_b = b.shape().dims();
                let m = dims_a[0];
                let k = dims_a[1];
                let n = dims_b[1];
                let mut c_vec = vec![0.0f32; m * n];
                unsafe {
                    cblas_sgemm(
                        101, // RowMajor
                        111, // NoTrans
                        111, // NoTrans
                        m as i32,
                        n as i32,
                        k as i32,
                        1.0,
                        a.as_ptr() as *const f32,
                        k as i32,
                        b.as_ptr() as *const f32,
                        n as i32,
                        0.0,
                        c_vec.as_mut_ptr(),
                        n as i32,
                    );
                }
                let out_storage = self.from_cpu(&c_vec, out, a.dtype())?;
                let ctx = MetalContext::get()?;
                let dummy_cmd = ctx.command_queue.commandBuffer()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create dummy command buffer".into())))?;
                return Ok((out_storage, Box::new(MetalHandle { command_buffer: dummy_cmd })));
            }

            if let Some(ref inner) = self.inner {
                if a.dtype().arith != ArithType::F32 || b.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(a.dtype())));
                }

                let a_s = a.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal matmul: input a is not MetalStorage".into())
                })?;
                let b_s = b.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal matmul: input b is not MetalStorage".into())
                })?;
                let a_buf = a_s.buffer.as_ref().ok_or_else(|| Error::Backend("a has no GPU buffer".into()))?;
                let b_buf = b_s.buffer.as_ref().ok_or_else(|| Error::Backend("b has no GPU buffer".into()))?;

                let a_dims = a.shape().dims();
                let b_dims = b.shape().dims();
                if a_dims.len() != 2 || b_dims.len() != 2 {
                    return Err(Error::Shape("Metal matmul expects 2-D inputs".into()));
                }
                let (m, k) = (a_dims[0], a_dims[1]);
                let (k2, n) = (b_dims[0], b_dims[1]);
                if k != k2 {
                    return Err(Error::ShapeMismatch {
                        expected: a_dims.to_vec(),
                        got: b_dims.to_vec(),
                    });
                }

                let dtype_out = DType {
                    arith: grim_tensor::dtype::ArithType::F32,
                    storage: DTypeStorage::Native,
                };
                let out_storage = self.zeros(out, dtype_out.clone())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

                encoder.setComputePipelineState(&inner.pipelines.matmul);
                encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                let m_val = m as i32;
                let n_val = n as i32;
                let k_val = k as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &m_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &n_val as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                    encoder.setBytes_length_atIndex(
                        &k_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }

                let tuner = Tuner::new();
                let config = tuner.search_tile_config(m, n, k, inner);
                let config_data = [config.block_m as i32, config.block_n as i32, config.block_k as i32];
                unsafe {
                    encoder.setBytes_length_atIndex(
                        config_data.as_ptr() as *const std::ffi::c_void,
                        12,
                        6,
                    );
                }

                let threads_per_group = MTLSize::new(config.block_n as u64, config.block_m as u64, 1);
                let groups = MTLSize::new(
                    ((n + (config.block_n as usize) - 1) / (config.block_n as usize)) as u64,
                    ((m + (config.block_m as usize) - 1) / (config.block_m as usize)) as u64,
                    1,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
            } else {
                run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                    cpu_dev.matmul(a_cpu, b_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.matmul(a_cpu, b_cpu, out_shape)
            })
        }
    }

    /// Performs elementwise addition on the Metal device.
    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if a.dtype().arith != ArithType::F32 || b.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(a.dtype())));
                }
                self.run_elementwise(inner, &inner.pipelines.add, a, b, out)
            } else {
                run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                    cpu_dev.add(a_cpu, b_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.add(a_cpu, b_cpu, out_shape)
            })
        }
    }

    /// Performs elementwise multiplication on the Metal device.
    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if a.dtype().arith != ArithType::F32 || b.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(a.dtype())));
                }
                self.run_elementwise(inner, &inner.pipelines.mul, a, b, out)
            } else {
                run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                    cpu_dev.mul(a_cpu, b_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.mul(a_cpu, b_cpu, out_shape)
            })
        }
    }

    /// Performs elementwise SiLU-multiplication (SwiGLU gate) on the Metal device.
    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if gate.dtype().arith != ArithType::F32 || up.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(gate.dtype())));
                }
                self.run_elementwise(inner, &inner.pipelines.silu_mul, gate, up, out)
            } else {
                run_fallback_binary(self, gate, up, out, |cpu_dev, g_cpu, u_cpu, out_shape| {
                    cpu_dev.silu_mul(g_cpu, u_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, gate, up, out, |cpu_dev, g_cpu, u_cpu, out_shape| {
                cpu_dev.silu_mul(g_cpu, u_cpu, out_shape)
            })
        }
    }

    /// Performs RMS Normalization on the Metal device.
    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if x.dtype().arith != ArithType::F32 || w.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(x.dtype())));
                }

                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rms_norm: input x is not MetalStorage".into())
                })?;
                let w_s = w.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rms_norm: input w is not MetalStorage".into())
                })?;
                let x_buf = x_s.buffer.as_ref().ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;
                let w_buf = w_s.buffer.as_ref().ok_or_else(|| Error::Backend("w has no GPU buffer".into()))?;

                let out_storage = self.zeros(out, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let total = out.elem_count();
                let row_len = x.shape().dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

                encoder.setComputePipelineState(&inner.pipelines.rms_norm);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                let row_len_val = row_len;
                let eps_val = eps;
                let total_val = total as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &row_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &eps_val as *const f32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
            } else {
                run_fallback_binary(self, x, w, out, |cpu_dev, x_cpu, w_cpu, out_shape| {
                    cpu_dev.rms_norm(x_cpu, w_cpu, eps, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, x, w, out, |cpu_dev, x_cpu, w_cpu, out_shape| {
                cpu_dev.rms_norm(x_cpu, w_cpu, eps, out_shape)
            })
        }
    }

    /// Performs Softmax along the last dimension on the Metal device.
    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if x.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(x.dtype())));
                }

                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal softmax: input x is not MetalStorage".into())
                })?;
                let x_buf = x_s.buffer.as_ref().ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;

                let out_storage = self.zeros(out, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let total = out.elem_count();
                let last_dim = out.dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

                encoder.setComputePipelineState(&inner.pipelines.softmax);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);

                let last_dim_val = last_dim;
                let total_val = total as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &last_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
            } else {
                let x_vec = x.to_cpu_vec_f32()?;
                let cpu_dev = CpuDevice::new();
                let x_cpu = cpu_dev.from_cpu(&x_vec, x.shape(), x.dtype())?;
                let x_storage = x_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
                    Error::Backend("Failed to downcast input x to CpuStorage".into())
                })?;
                let (res_storage, handle) = cpu_dev.softmax(x_storage, out)?;
                let res_vec = res_storage.to_cpu_vec_f32()?;
                let out_metal = self.from_cpu(&res_vec, out, x.dtype())?;
                Ok((out_metal, handle))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let x_vec = x.to_cpu_vec_f32()?;
            let cpu_dev = CpuDevice::new();
            let x_cpu = cpu_dev.from_cpu(&x_vec, x.shape(), x.dtype())?;
            let x_storage = x_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
                Error::Backend("Failed to downcast input x to CpuStorage".into())
            })?;
            let (res_storage, handle) = cpu_dev.softmax(x_storage, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, x.dtype())?;
            Ok((out_metal, handle))
        }
    }

    /// Performs embedding lookup on the Metal device.
    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if weight.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(weight.dtype())));
                }

                let w_s = weight.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal embedding: weight is not MetalStorage".into())
                })?;
                let w_buf = w_s.buffer.as_ref().ok_or_else(|| Error::Backend("weight has no GPU buffer".into()))?;

                // Create a temporary buffer for indices and copy them using new_buffer_with_bytes
                let indices_bytes = indices.len()
                    .checked_mul(4)
                    .ok_or_else(|| Error::from(MetalError::AllocationFailed("Indices size overflow".into())))?;
                let indices_u8 = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices_bytes)
                };
                let indices_buffer = self.new_buffer_with_bytes(indices_u8, BufferUsage::Shared)?;

                let out_storage = self.zeros(out, weight.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let embedding_dim = out.dims().last().copied().unwrap_or(1) as i32;
                let num_indices = indices.len() as i32;
                let total = out.elem_count();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

                encoder.setComputePipelineState(&inner.pipelines.embedding);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&indices_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &embedding_dim as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_indices as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
            } else {
                let w_vec = weight.to_cpu_vec_f32()?;
                let cpu_dev = CpuDevice::new();
                let w_cpu = cpu_dev.from_cpu(&w_vec, weight.shape(), weight.dtype())?;
                let w_storage = w_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
                    Error::Backend("Failed to downcast weight to CpuStorage".into())
                })?;
                let (res_storage, handle) = cpu_dev.embedding(w_storage, indices, out)?;
                let res_vec = res_storage.to_cpu_vec_f32()?;
                let out_metal = self.from_cpu(&res_vec, out, weight.dtype())?;
                Ok((out_metal, handle))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let w_vec = weight.to_cpu_vec_f32()?;
            let cpu_dev = CpuDevice::new();
            let w_cpu = cpu_dev.from_cpu(&w_vec, weight.shape(), weight.dtype())?;
            let w_storage = w_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
                Error::Backend("Failed to downcast weight to CpuStorage".into())
            })?;
            let (res_storage, handle) = cpu_dev.embedding(w_storage, indices, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, weight.dtype())?;
            Ok((out_metal, handle))
        }
    }

    /// Copies a slice of F32 values from host memory to the device storage.
    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let bytes = shape.elem_count()
            .checked_mul(dtype_byte_size(&dtype)?)
            .ok_or_else(|| Error::from(MetalError::AllocationFailed("Buffer size overflow".into())))?;
        if data.len() * 4 < bytes {
            return Err(Error::from(MetalError::DataMismatch(format!(
                "from_cpu: source slice ({} bytes) too small for destination ({} bytes)",
                data.len() * 4,
                bytes
            ))));
        }

        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref _inner) = self.inner {
                let data_bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
                };
                let buffer = self.new_buffer_with_bytes(data_bytes, BufferUsage::Shared)?;

                Ok(Box::new(MetalStorage {
                    buffer: Some(buffer),
                    data: None,
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            } else {
                let mut fallback_data = vec![0u8; bytes];
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, fallback_data.as_mut_ptr(), bytes);
                }
                Ok(Box::new(MetalStorage {
                    buffer: None,
                    data: Some(std::sync::Mutex::new(fallback_data)),
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let mut fallback_data = vec![0u8; bytes];
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, fallback_data.as_mut_ptr(), bytes);
            }
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(fallback_data),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    /// Provide hints about memory usage/advice patterns to the device/system.
    fn advise(&self, _storage: &dyn BackendStorage, _advice: grim_tensor::backend::MemAdvice) -> Result<()> {
        Ok(())
    }

    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if q.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(q.dtype())));
                }

                let q_s = q.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("kv_dequant_attention q is not MetalStorage".into())
                })?;
                let k_s = k_tensor.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("kv_dequant_attention k_tensor is not MetalStorage".into())
                })?;
                let ks_s = k_scales.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("kv_dequant_attention k_scales is not MetalStorage".into())
                })?;
                let v_s = v_tensor.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("kv_dequant_attention v_tensor is not MetalStorage".into())
                })?;
                let vs_s = v_scales.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("kv_dequant_attention v_scales is not MetalStorage".into())
                })?;

                let q_buf = q_s.buffer.as_ref().ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
                let k_buf = k_s.buffer.as_ref().ok_or_else(|| Error::Backend("k_tensor has no GPU buffer".into()))?;
                let ks_buf = ks_s.buffer.as_ref().ok_or_else(|| Error::Backend("k_scales has no GPU buffer".into()))?;
                let v_buf = v_s.buffer.as_ref().ok_or_else(|| Error::Backend("v_tensor has no GPU buffer".into()))?;
                let vs_buf = vs_s.buffer.as_ref().ok_or_else(|| Error::Backend("v_scales has no GPU buffer".into()))?;

                let out_dims = out_shape.dims();
                if out_dims.len() != 3 {
                    return Err(Error::Backend("kv_dequant_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into()));
                }
                let seq_len = out_dims[0];
                let num_heads = out_dims[1];
                let head_dim = out_dims[2];

                let out_storage = self.zeros(out_shape, q.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

                encoder.setComputePipelineState(&inner.pipelines.kv_dequant_attn);
                encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(ks_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(vs_buf), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 5);

                let num_heads_val = num_heads as i32;
                let num_kv_heads_val = num_kv_heads as i32;
                let head_dim_val = head_dim as i32;
                let seq_len_val = seq_len as i32;
                let kv_seq_len_val = kv_seq_len as i32;
                let cache_offset_val = cache_offset as i32;
                let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
                let quant_bits_val = quant_bits as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(&num_heads_val as *const i32 as *const std::ffi::c_void, 4, 6);
                    encoder.setBytes_length_atIndex(&num_kv_heads_val as *const i32 as *const std::ffi::c_void, 4, 7);
                    encoder.setBytes_length_atIndex(&head_dim_val as *const i32 as *const std::ffi::c_void, 4, 8);
                    encoder.setBytes_length_atIndex(&seq_len_val as *const i32 as *const std::ffi::c_void, 4, 9);
                    encoder.setBytes_length_atIndex(&kv_seq_len_val as *const i32 as *const std::ffi::c_void, 4, 10);
                    encoder.setBytes_length_atIndex(&cache_offset_val as *const i32 as *const std::ffi::c_void, 4, 11);
                    encoder.setBytes_length_atIndex(&inv_sqrt_d as *const f32 as *const std::ffi::c_void, 4, 12);
                    encoder.setBytes_length_atIndex(&quant_bits_val as *const i32 as *const std::ffi::c_void, 4, 13);
                }

                let threads_per_group = MTLSize::new(1, 1, 1);
                let groups = MTLSize::new(seq_len as u64, num_heads as u64, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
            } else {
                Err(Error::Backend("Metal device inner is None (fallback mode)".into()))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (q, k_tensor, k_scales, v_tensor, v_scales, num_kv_heads, kv_seq_len, cache_offset, quant_bits, out_shape);
            Err(Error::Unimplemented("kv_dequant_attention not supported on non-Apple platform".into()))
        }
    }
}

impl MetalDevice {
    /// Fused QKV attention matching ROCm / CUDA signatures.
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

        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if q.dtype().arith != ArithType::F32 || k.dtype().arith != ArithType::F32 || v.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(q.dtype())));
                }

                let q_s = q.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("qkv_attention q is not MetalStorage".into())
                })?;
                let k_s = k.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("qkv_attention k is not MetalStorage".into())
                })?;
                let v_s = v.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("qkv_attention v is not MetalStorage".into())
                })?;

                let q_buf = q_s.buffer.as_ref().ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
                let k_buf = k_s.buffer.as_ref().ok_or_else(|| Error::Backend("k has no GPU buffer".into()))?;
                let v_buf = v_s.buffer.as_ref().ok_or_else(|| Error::Backend("v has no GPU buffer".into()))?;

                let max_s = match out_max {
                    Some(m) => {
                        let ms = m.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                            Error::Backend("qkv_attention out_max is not MetalStorage".into())
                        })?;
                        Some(ms.buffer.as_ref().ok_or_else(|| Error::Backend("out_max has no GPU buffer".into()))?)
                    }
                    None => None,
                };
                let sum_s = match out_sum {
                    Some(s) => {
                        let ss = s.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                            Error::Backend("qkv_attention out_sum is not MetalStorage".into())
                        })?;
                        Some(ss.buffer.as_ref().ok_or_else(|| Error::Backend("out_sum has no GPU buffer".into()))?)
                    }
                    None => None,
                };

                let out_storage = self.zeros(out, DType::F32)?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

                encoder.setComputePipelineState(&inner.pipelines.qkv_attn);
                encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(max_s.copied(), 0, 4);
                encoder.setBuffer_offset_atIndex(sum_s.copied(), 0, 5);

                let num_heads_val = num_heads as i32;
                let num_kv_heads_val = num_kv_heads as i32;
                let head_dim_val = head_dim as i32;
                let seq_len_val = seq_len as i32;
                let kv_seq_len_val = kv_seq_len as i32;
                let cache_offset_val = cache_offset as i32;
                let inv_sqrt_d_val = 1.0 / (head_dim as f32).sqrt();

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &num_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_kv_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        7,
                    );
                    encoder.setBytes_length_atIndex(
                        &head_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        8,
                    );
                    encoder.setBytes_length_atIndex(
                        &seq_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        9,
                    );
                    encoder.setBytes_length_atIndex(
                        &kv_seq_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        10,
                    );
                    encoder.setBytes_length_atIndex(
                        &cache_offset_val as *const i32 as *const std::ffi::c_void,
                        4,
                        11,
                    );
                    encoder.setBytes_length_atIndex(
                        &inv_sqrt_d_val as *const f32 as *const std::ffi::c_void,
                        4,
                        12,
                    );
                }

                let threads_per_group = MTLSize::new(32, 1, 1);
                let groups = MTLSize::new(seq_len as u64, num_heads as u64, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
            } else {
                let _ = out_max;
                let _ = out_sum;
                // Simple host-fallback simulation for unit tests
                let q_vec = q.to_cpu_vec_f32()?;
                let k_vec = k.to_cpu_vec_f32()?;
                let v_vec = v.to_cpu_vec_f32()?;

                let mut out_vec = vec![0.0f32; out.elem_count()];
                let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

                for i in 0..seq_len {
                    for h in 0..num_heads {
                        let q_per_kv = num_heads / num_kv_heads;
                        let kv_head = h / q_per_kv;
                        let q_offset = (i * num_heads + h) * head_dim;
                        let abs_i = cache_offset as usize + i;
                        let range_len = if abs_i < kv_seq_len { abs_i + 1 } else { kv_seq_len };

                        let mut running_max = -1e30_f32;
                        let mut running_sum = 0.0_f32;

                        let mut scores = vec![0.0f32; range_len];
                        for j in 0..range_len {
                            let mut score = 0.0_f32;
                            for d in 0..head_dim {
                                score += q_vec[q_offset + d] * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                            }
                            score *= inv_sqrt_d;
                            scores[j] = score;
                            if score > running_max {
                                running_max = score;
                            }
                        }

                        for j in 0..range_len {
                            running_sum += (scores[j] - running_max).exp();
                        }

                        for d in 0..head_dim {
                            let mut acc = 0.0_f32;
                            for j in 0..range_len {
                                let weight = (scores[j] - running_max).exp() / (if running_sum > 0.0_f32 { running_sum } else { 1.0_f32 });
                                acc += weight * v_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                            }
                            out_vec[q_offset + d] = acc;
                        }
                    }
                }

                let out_storage = self.from_cpu(&out_vec, out, DType::F32)?;
                Ok((out_storage, Box::new(MetalHandle)))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = out_max;
            let _ = out_sum;
            // Simple host-fallback simulation for unit tests
            let q_vec = q.to_cpu_vec_f32()?;
            let k_vec = k.to_cpu_vec_f32()?;
            let v_vec = v.to_cpu_vec_f32()?;

            let mut out_vec = vec![0.0f32; out.elem_count()];
            let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

            for i in 0..seq_len {
                for h in 0..num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let kv_head = h / q_per_kv;
                    let q_offset = (i * num_heads + h) * head_dim;
                    let abs_i = cache_offset as usize + i;
                    let range_len = if abs_i < kv_seq_len { abs_i + 1 } else { kv_seq_len };

                    let mut running_max = -1e30_f32;
                    let mut running_sum = 0.0_f32;

                    let mut scores = vec![0.0f32; range_len];
                    for j in 0..range_len {
                        let mut score = 0.0_f32;
                        for d in 0..head_dim {
                            score += q_vec[q_offset + d] * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                        }
                        score *= inv_sqrt_d;
                        scores[j] = score;
                        if score > running_max {
                            running_max = score;
                        }
                    }

                    for j in 0..range_len {
                        running_sum += (scores[j] - running_max).exp();
                    }

                    for d in 0..head_dim {
                        let mut acc = 0.0_f32;
                        for j in 0..range_len {
                            let weight = (scores[j] - running_max).exp() / (if running_sum > 0.0_f32 { running_sum } else { 1.0_f32 });
                            acc += weight * v_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                        }
                        out_vec[q_offset + d] = acc;
                    }
                }
            }

            let out_storage = self.from_cpu(&out_vec, out, DType::F32)?;
            Ok((out_storage, Box::new(MetalHandle)))
        }
    }

    #[cfg(target_vendor = "apple")]
    fn run_elementwise(
        &self,
        inner: &MetalDeviceInner,
        pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            Error::Backend("Metal elementwise: input a is not MetalStorage".into())
        })?;
        let b_s = b.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            Error::Backend("Metal elementwise: input b is not MetalStorage".into())
        })?;
        let a_buf = a_s.buffer.as_ref().ok_or_else(|| Error::Backend("a has no GPU buffer".into()))?;
        let b_buf = b_s.buffer.as_ref().ok_or_else(|| Error::Backend("b has no GPU buffer".into()))?;

        let out_storage = self.zeros(out, a.dtype())?;
        let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
        let out_buf = out_s.buffer.as_ref().unwrap();

        let total = out.elem_count();

        let cmd_buffer = self.get_or_create_command_buffer()?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Error::from(MetalError::Ffi("Failed to create compute encoder".into())))?;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

        let total_val = total as i32;
        unsafe {
            encoder.setBytes_length_atIndex(
                &total_val as *const i32 as *const std::ffi::c_void,
                4,
                3,
            );
        }

        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();

        Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })))
    }
}

/// Layout dimensions for Metal compute block threadgroups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalTileConfig {
    /// Dimension size of the tiled block in the M dimension.
    pub block_m: u32,
    /// Dimension size of the tiled block in the N dimension.
    pub block_n: u32,
    /// Dimension size of the tiled block in the K dimension.
    pub block_k: u32,
}

/// Runtime autotuner analyzing GPU execution constraints to optimize tile shapes.
pub struct Tuner;

impl Tuner {
    /// Creates a new Tuner instance.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates target GPU constraints to look up or benchmark the optimal threadgroup configuration.
    #[cfg(target_vendor = "apple")]
    pub fn search_tile_config(
        &self,
        m: usize,
        n: usize,
        k: usize,
        inner: &MetalDeviceInner,
    ) -> MetalTileConfig {
        let key = (m, n, k);
        self.with_persistent_cache(key, || {
            let candidates = vec![
                MetalTileConfig { block_m: 8, block_n: 8, block_k: 8 },
                MetalTileConfig { block_m: 16, block_n: 16, block_k: 16 },
                MetalTileConfig { block_m: 32, block_n: 16, block_k: 16 },
                MetalTileConfig { block_m: 16, block_n: 32, block_k: 16 },
            ];
            let mut best_config = candidates[1];
            let mut best_time = std::time::Duration::MAX;

            use objc2_metal::MTResourceOptions;
            let bytes_a = m * k * 4;
            let bytes_b = k * n * 4;
            let bytes_c = m * n * 4;
            let buf_a = inner.device.newBufferWithLength_options(bytes_a as u64, MTResourceOptions::StorageModeShared).unwrap();
            let buf_b = inner.device.newBufferWithLength_options(bytes_b as u64, MTResourceOptions::StorageModeShared).unwrap();
            let buf_c = inner.device.newBufferWithLength_options(bytes_c as u64, MTResourceOptions::StorageModeShared).unwrap();

            for &config in &candidates {
                let config_data = [config.block_m as i32, config.block_n as i32, config.block_k as i32];
                for _ in 0..2 {
                    if let Some(cmd) = inner.command_queue.commandBuffer() {
                        if let Some(enc) = cmd.computeCommandEncoder() {
                            enc.setComputePipelineState(&inner.pipelines.matmul);
                            enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                enc.setBytes_length_atIndex(&m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                enc.setBytes_length_atIndex(&n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                enc.setBytes_length_atIndex(&k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                                enc.setBytes_length_atIndex(config_data.as_ptr() as *const std::ffi::c_void, 12, 6);
                            }
                            let threads = MTLSize::new(config.block_n as u64, config.block_m as u64, 1);
                            let groups = MTLSize::new(((n + (config.block_n as usize) - 1) / (config.block_n as usize)) as u64, ((m + (config.block_m as usize) - 1) / (config.block_m as usize)) as u64, 1);
                            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            enc.endEncoding();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                        }
                    }
                }

                let start = std::time::Instant::now();
                let iters = 5;
                for _ in 0..iters {
                    if let Some(cmd) = inner.command_queue.commandBuffer() {
                        if let Some(enc) = cmd.computeCommandEncoder() {
                            enc.setComputePipelineState(&inner.pipelines.matmul);
                            enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                enc.setBytes_length_atIndex(&m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                enc.setBytes_length_atIndex(&n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                enc.setBytes_length_atIndex(&k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                                enc.setBytes_length_atIndex(config_data.as_ptr() as *const std::ffi::c_void, 12, 6);
                            }
                            let threads = MTLSize::new(config.block_n as u64, config.block_m as u64, 1);
                            let groups = MTLSize::new(((n + (config.block_n as usize) - 1) / (config.block_n as usize)) as u64, ((m + (config.block_m as usize) - 1) / (config.block_m as usize)) as u64, 1);
                            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            enc.endEncoding();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                        }
                    }
                }
                let elapsed = start.elapsed() / iters;
                if elapsed < best_time {
                    best_time = elapsed;
                    best_config = config;
                }
            }

            best_config
        })
    }

    #[cfg(target_vendor = "apple")]
    fn with_persistent_cache<F>(
        &self,
        key: (usize, usize, usize),
        benchmark: F,
    ) -> MetalTileConfig
    where
        F: FnOnce() -> MetalTileConfig,
    {
        use std::sync::Mutex;
        use std::collections::HashMap;
        use std::sync::OnceLock;

        static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), (u32, u32, u32)>>> = OnceLock::new();
        let cache_mutex = CACHE.get_or_init(|| {
            let mut map = HashMap::new();
            if let Some(cache_dir) = get_cache_dir() {
                let cache_file = cache_dir.join("grim_metal_autotune_cache.txt");
                if cache_file.exists() {
                    if let Ok(contents) = std::fs::read_to_string(cache_file) {
                        for line in contents.lines() {
                            let parts: Vec<&str> = line.split('=').collect();
                            if parts.len() == 2 {
                                let key_parts: Vec<&str> = parts[0].split(',').collect();
                                let val_parts: Vec<&str> = parts[1].split(',').collect();
                                if key_parts.len() == 3 && val_parts.len() == 3 {
                                    if let (Ok(km), Ok(kn), Ok(kk), Ok(vx), Ok(vy), Ok(vz)) = (
                                        key_parts[0].parse::<usize>(),
                                        key_parts[1].parse::<usize>(),
                                        key_parts[2].parse::<usize>(),
                                        val_parts[0].parse::<u32>(),
                                        val_parts[1].parse::<u32>(),
                                        val_parts[2].parse::<u32>(),
                                    ) {
                                        map.insert((km, kn, kk), (vx, vy, vz));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Mutex::new(map)
        });

        {
            let guard = cache_mutex.lock().unwrap();
            if let Some(&val) = guard.get(&key) {
                return MetalTileConfig {
                    block_m: val.0,
                    block_n: val.1,
                    block_k: val.2,
                };
            }
        }

        let config = benchmark();

        {
            let mut guard = cache_mutex.lock().unwrap();
            guard.insert(key, (config.block_m, config.block_n, config.block_k));
            if let Some(cache_dir) = get_cache_dir() {
                let cache_file = cache_dir.join("grim_metal_autotune_cache.txt");
                let mut lines = Vec::new();
                for (k, v) in guard.iter() {
                    lines.push(format!("{},{},{}={},{},{}", k.0, k.1, k.2, v.0, v.1, v.2));
                }
                let _ = std::fs::write(cache_file, lines.join("\n"));
            }
        }

        config
    }
}

/// Run binary operations on the CPU fallback pipeline.
#[cfg(not(target_vendor = "apple"))]
fn run_fallback_binary<F>(
    device: &MetalDevice,
    a: &dyn BackendStorage,
    b: &dyn BackendStorage,
    out: &Shape,
    op: F,
) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>
where
    F: FnOnce(&CpuDevice, &CpuStorage, &CpuStorage, &Shape) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>,
{
    let a_vec = a.to_cpu_vec_f32()?;
    let b_vec = b.to_cpu_vec_f32()?;

    let cpu_dev = CpuDevice::new();
    let a_cpu = cpu_dev.from_cpu(&a_vec, a.shape(), a.dtype())?;
    let b_cpu = cpu_dev.from_cpu(&b_vec, b.shape(), b.dtype())?;

    let a_storage = a_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
        Error::Backend("Failed to downcast input a to CpuStorage".into())
    })?;
    let b_storage = b_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
        Error::Backend("Failed to downcast input b to CpuStorage".into())
    })?;

    let (res_storage, handle) = op(&cpu_dev, a_storage, b_storage, out)?;

    let res_vec = res_storage.to_cpu_vec_f32()?;
    let out_metal = device.from_cpu(&res_vec, out, a.dtype())?;

    Ok((out_metal, handle))
}

/// Helper function to retrieve the size in bytes of a data type.
#[allow(dead_code)]
fn dtype_byte_size(dtype: &DType) -> Result<usize> {
    #[cfg(target_vendor = "apple")]
    {
        match dtype.arith {
            ArithType::F32 | ArithType::U32 => Ok(4),
            ArithType::F16 | ArithType::BF16 => Ok(2),
            ArithType::I64 => Ok(8),
            ArithType::U8 => Ok(1),
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = dtype;
        Ok(4)
    }
}

/// MLX integration bridge allowing zero-copy sharing of Metal allocations.
pub struct MlxBridge;

impl MlxBridge {
    /// Creates a new MlxBridge instance.
    pub fn new() -> Self {
        Self
    }

    /// Zero-copy maps a `MetalStorage` buffer to an MLX array.
    /// Returns the raw pointer to the underlying MTLBuffer.
    #[cfg(target_vendor = "apple")]
    pub unsafe fn to_mlx_array(&self, storage: &MetalStorage) -> Result<*mut std::ffi::c_void> {
        let buffer = storage.buffer.as_ref().ok_or_else(|| {
            Error::Backend("Storage lacks an active Metal buffer".into())
        })?;
        let raw_ptr = objc2::rc::Retained::as_ptr(buffer) as *mut std::ffi::c_void;
        Ok(raw_ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_probe() {
        let devices = MetalDevice::probe().unwrap();
        #[cfg(not(target_vendor = "apple"))]
        assert!(devices.is_empty());
        #[cfg(target_vendor = "apple")]
        {
            // If metal is supported on the testing mac:
            if let Ok(dev) = MetalDevice::try_new(0) {
                if dev.inner.is_some() {
                    assert!(!devices.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_metal_zeros() {
        let dev = MetalDevice::new(0);
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 4]);
        let vec = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(vec, vec![0.0f32; 8]);
    }

    #[test]
    fn test_metal_matmul() {
        let dev = MetalDevice::new(0);
        let a = dev.from_cpu(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
        let b = dev.from_cpu(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![2, 2]);
        let (out, handle) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_metal_add() {
        let dev = MetalDevice::new(0);
        let a = dev.from_cpu(&[1.0, 2.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let b = dev.from_cpu(&[3.0, 4.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let (out, handle) = dev.add(a.as_ref(), b.as_ref(), &Shape::new(vec![2])).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![4.0, 6.0]);
    }

    #[test]
    fn test_metal_qkv_attention() {
        let dev = MetalDevice::new(0);
        let q = dev.from_cpu(&[1.0, 0.0, 0.0, 1.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let k = dev.from_cpu(&[1.0, 0.0, 0.0, 1.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let v = dev.from_cpu(&[2.0, 3.0, 4.0, 5.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![1, 2, 2]);
        let (out, handle) = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), 2, 1, 0, &out_shape, None, None).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_dtype_guards_negative() {
        // GPU path tests for apple Silicon (only run if hardware is available)
        let dev = MetalDevice::new(0);
        if dev.inner.is_some() {
            // Attempt to run matmul with a non-F32 dtype (e.g. U8 or F16)
            let a = dev.from_cpu(&[1.0, 2.0], &Shape::new(vec![1, 2]), DType::U8).unwrap();
            let b = dev.from_cpu(&[3.0, 4.0], &Shape::new(vec![2, 1]), DType::U8).unwrap();
            let out_shape = Shape::new(vec![1, 1]);
            let res = dev.matmul(a.as_ref(), b.as_ref(), &out_shape);
            assert!(res.is_err(), "Expected matmul with non-F32 inputs to fail on GPU");
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_shape_mismatches_negative() {
        let dev = MetalDevice::new(0);
        if dev.inner.is_some() {
            let a = dev.from_cpu(&[1.0, 2.0], &Shape::new(vec![1, 2]), DType::F32).unwrap();
            let b = dev.from_cpu(&[3.0, 4.0], &Shape::new(vec![3, 1]), DType::F32).unwrap();
            let out_shape = Shape::new(vec![1, 1]);
            let res = dev.matmul(a.as_ref(), b.as_ref(), &out_shape);
            assert!(res.is_err(), "Expected shape mismatch to return error");
        }
    }

    #[test]
    fn test_metal_kv_dequant_attention() {
        let dev = MetalDevice::new(0);
        let q = dev.from_cpu(&[1.0, 0.0, 0.0, 1.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let k_tensor = dev.from_cpu(&[1.0, 0.0, 0.0, 1.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let k_scales = dev.from_cpu(&[1.0, 1.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let v_tensor = dev.from_cpu(&[2.0, 3.0, 4.0, 5.0], &Shape::new(vec![1, 2, 2]), DType::F32).unwrap();
        let v_scales = dev.from_cpu(&[1.0, 1.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![1, 2, 2]);
        let res = dev.kv_dequant_attention(
            q.as_ref(),
            k_tensor.as_ref(),
            k_scales.as_ref(),
            v_tensor.as_ref(),
            v_scales.as_ref(),
            1,
            2,
            0,
            8,
            &out_shape,
        );
        #[cfg(not(target_vendor = "apple"))]
        {
            assert!(res.is_err());
        }
        #[cfg(target_vendor = "apple")]
        {
            if dev.inner.is_some() {
                let (out, handle) = res.unwrap();
                handle.synchronize().unwrap();
                let data = out.to_cpu_vec_f32().unwrap();
                assert_eq!(data.len(), 4);
            }
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_gpu_compute_coverage() {
        let dev = MetalDevice::try_new(0).unwrap();
        if dev.inner.is_some() {
            let a = dev.from_cpu(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
            let b = dev.from_cpu(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
            let (out, handle) = dev.matmul(a.as_ref(), b.as_ref(), &Shape::new(vec![2, 2])).unwrap();
            handle.synchronize().unwrap();
            assert_eq!(out.to_cpu_vec_f32().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);

            let (out_add, handle_add) = dev.add(a.as_ref(), b.as_ref(), &Shape::new(vec![4])).unwrap();
            handle_add.synchronize().unwrap();
            assert_eq!(out_add.to_cpu_vec_f32().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_mlx_bridge() {
        let dev = MetalDevice::try_new(0).unwrap();
        if dev.inner.is_some() {
            let storage = dev.from_cpu(&[1.0, 2.0], &Shape::new(vec![2]), DType::F32).unwrap();
            let metal_storage = storage.as_any().downcast_ref::<MetalStorage>().unwrap();
            let bridge = MlxBridge::new();
            let raw_ptr = unsafe { bridge.to_mlx_array(metal_storage).unwrap() };
            assert!(!raw_ptr.is_null());
        }
    }
}
