//! Metal backend for Apple Silicon GPUs using MSL compute pipelines.

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
#[allow(unused_imports)]
use grim_tensor::dtype::{
    ArithType, DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance,
    Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, ScythePlacement, Shape};

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

#[derive(Debug, Clone, thiserror::Error)]
pub enum MetalError {
    #[error("Metal initialization/FFI error: {0}")]
    Ffi(String),
    #[error("Metal shader compilation failed: {0}")]
    Compilation(String),
    #[error("Metal only supports F32 operations, got dtype: {0:?}")]
    UnsupportedDType(DType),
    #[error("Metal buffer allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Metal context error: {0}")]
    Context(String),
    #[error("Metal buffer contents is null")]
    NullBuffer,
    #[error("Metal storage data mismatch: {0}")]
    DataMismatch(String),
}

impl From<MetalError> for Error {
    fn from(err: MetalError) -> Self {
        Error::Backend(err.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    Shared,
    Private,
}

#[cfg(target_vendor = "apple")]
impl BufferUsage {
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
    silu_mul_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    embedding: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qkv_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qkv_paged_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    tree_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    kv_dequant_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul_scalar: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    sqrt: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    recip: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rope: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Partial-rotary + YaRN RoPE (`grim_rope_yarn`).
    rope_yarn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quantized_matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residualpacked_matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quantized_matmul_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    all_reduce: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    comm_fuse_reduce: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_fp8: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_mxfp4: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_mxfp8: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_q4k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_q8_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq2xxs: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq2xs: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq2s: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq3xxs: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq3s: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq4nl: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dequant_iq4xs: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    moe_fused_dispatch: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add_rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quant_q8_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quant_fp8: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quant_mxfp4: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quant_mxfp8: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    quant_q4k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pipelines: std::sync::Arc<MetalPipelines>,
}

#[cfg(target_vendor = "apple")]
static METAL_CONTEXT: std::sync::OnceLock<std::result::Result<MetalContext, MetalError>> =
    std::sync::OnceLock::new();

#[cfg(target_vendor = "apple")]
impl MetalContext {
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
                silu_mul_backward: get_pipeline("grim_silu_mul_backward")?,
                rms_norm: get_pipeline("grim_rms_norm")?,
                softmax: get_pipeline("grim_softmax")?,
                embedding: get_pipeline("grim_embedding")?,
                matmul: get_pipeline("grim_matmul")?,
                qkv_attn: get_pipeline("grim_qkv_attention")?,
                qkv_paged_attn: get_pipeline("grim_qkv_attention_paged")?,
                tree_attn: get_pipeline("grim_tree_attention")?,
                kv_dequant_attn: get_pipeline("grim_kv_dequant_attention")?,
                mul_scalar: get_pipeline("grim_mul_scalar")?,
                sqrt: get_pipeline("grim_sqrt")?,
                recip: get_pipeline("grim_recip")?,
                rope: get_pipeline("grim_rope")?,
                rope_yarn: get_pipeline("grim_rope_yarn")?,
                quantized_matmul: get_pipeline("grim_quantized_matmul_q8_0")?,
                residualpacked_matmul: get_pipeline("grim_quantized_matmul_residualpacked")?,
                quantized_matmul_backward: get_pipeline("grim_quantized_matmul_backward_q8_0")?,
                all_reduce: get_pipeline("grim_all_reduce")?,
                comm_fuse_reduce: get_pipeline("grim_comm_fuse_reduce")?,
                dequant_fp8: get_pipeline("grim_dequant_fp8")?,
                dequant_mxfp4: get_pipeline("grim_dequant_mxfp4")?,
                dequant_mxfp8: get_pipeline("grim_dequant_mxfp8")?,
                dequant_q4k: get_pipeline("grim_dequant_q4k")?,
                dequant_q8_0: get_pipeline("grim_dequant_q8_0")?,
                dequant_iq2xxs: get_pipeline("grim_dequant_iq2xxs")?,
                dequant_iq2xs: get_pipeline("grim_dequant_iq2xs")?,
                dequant_iq2s: get_pipeline("grim_dequant_iq2s")?,
                dequant_iq3xxs: get_pipeline("grim_dequant_iq3xxs")?,
                dequant_iq3s: get_pipeline("grim_dequant_iq3s")?,
                dequant_iq4nl: get_pipeline("grim_dequant_iq4nl")?,
                dequant_iq4xs: get_pipeline("grim_dequant_iq4xs")?,
                moe_fused_dispatch: get_pipeline("grim_moe_fused_dispatch")?,
                add_rms_norm: get_pipeline("grim_add_rms_norm")?,
                quant_q8_0: get_pipeline("grim_quant_q8_0")?,
                quant_fp8: get_pipeline("grim_quant_fp8")?,
                quant_mxfp4: get_pipeline("grim_quant_mxfp4")?,
                quant_mxfp8: get_pipeline("grim_quant_mxfp8")?,
                quant_q4k: get_pipeline("grim_quant_q4k")?,
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
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalHandle;

impl ComputeHandle for MetalHandle {
    fn synchronize(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

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
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

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
                    return Err(Error::from(MetalError::DataMismatch(
                        "CPU storage buffer size mismatch".into(),
                    )));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data_guard.as_ptr(),
                        out.as_mut_ptr() as *mut u8,
                        bytes,
                    );
                }
                Ok(out)
            } else {
                Err(Error::Backend(
                    "MetalStorage has no buffer or fallback data".into(),
                ))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let data_guard = self.data.lock().unwrap();
            let elem_count = self.shape.elem_count();
            match self.dtype.storage {
                DTypeStorage::KQuant(KQuantScheme::Q80) => {
                    let dev = MetalDevice::new(0)?;
                    dev.dequantize_q8_0_host(&data_guard, elem_count)
                }
                DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                    let dev = MetalDevice::new(0)?;
                    dev.dequantize_fp8_host(&data_guard, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                    let dev = MetalDevice::new(0)?;
                    dev.dequantize_q4k_host(&data_guard, elem_count)
                }
                _ => {
                    let mut out = vec![0.0f32; elem_count];
                    let bytes = elem_count * dtype_byte_size(&self.dtype)?;
                    if data_guard.len() < bytes {
                        return Err(Error::from(MetalError::DataMismatch(
                            "CPU storage buffer size mismatch".into(),
                        )));
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data_guard.as_ptr(),
                            out.as_mut_ptr() as *mut u8,
                            bytes,
                        );
                    }
                    Ok(out)
                }
            }
        }
    }

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

#[cfg(target_vendor = "apple")]
fn fnv1a_hash(s: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in s.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

#[cfg(target_vendor = "apple")]
fn get_cache_dir() -> Option<std::path::PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        Some(
            std::path::PathBuf::from(home)
                .join(".cache")
                .join("grim_metal_cache"),
        )
    } else if let Ok(user_profile) = std::env::var("USERPROFILE") {
        Some(
            std::path::PathBuf::from(user_profile)
                .join(".cache")
                .join("grim_metal_cache"),
        )
    } else {
        None
    }
}

impl MetalDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            Self::try_new(ordinal)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Self { ordinal })
        }
    }

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
    pub fn get_or_create_command_buffer(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| Error::from(MetalError::Context("Device inner is None".into())))?;
        let mut active = inner.active_command_buffer.lock().unwrap();
        if let Some(ref buf) = *active {
            use objc2_metal::MTLCommandBufferStatus;
            if buf.status() == MTLCommandBufferStatus::NotEnqueued {
                return Ok(buf.clone());
            }
        }
        let new_buf = inner.command_queue.commandBuffer().ok_or_else(|| {
            Error::from(MetalError::Ffi("Failed to create command buffer".into()))
        })?;
        *active = Some(new_buf.clone());
        Ok(new_buf)
    }

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
    pub fn new_buffer_with_bytes(
        &self,
        bytes: &[u8],
        usage: BufferUsage,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| Error::from(MetalError::Context("Device inner is None".into())))?;
        let options = usage.to_mtl_options();
        let buffer = unsafe {
            inner.device.newBufferWithBytes_length_options(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                options,
            )
        }
        .ok_or_else(|| {
            Error::from(MetalError::AllocationFailed(
                "Failed to allocate MTLBuffer with bytes".into(),
            ))
        })?;
        Ok(buffer)
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn probe() -> Result<Vec<MetalDevice>> {
        #[cfg(target_vendor = "apple")]
        {
            let dev = MetalDevice::new(0)?;
            if dev.inner.is_some() {
                return Ok(vec![dev]);
            }
            Ok(vec![])
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok(vec![])
    }

    // ─── Standalone dequant host wrappers (q8_0, q4k, iq*, fp8, mxfp) ─────────

    /// Dequantize Q8_0 packed bytes to F32 on host/GPU.
    pub fn dequantize_q8_0_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Ok(ctx) = MetalContext::get() {
                let n_blocks = bytes.len() / 34;
                let packed_buf = self.new_buffer_with_bytes(bytes, BufferUsage::Shared)?;
                let out_buf = ctx
                    .device
                    .newBufferWithLength_options(
                        (elem_count * 4) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| Error::Backend("Metal dequant_q8_0: alloc out failed".into()))?;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Backend("Metal dequant: encoder failed".into()))?;
                encoder.setComputePipelineState(&ctx.pipelines.dequant_q8_0);
                encoder.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
                let n_b = n_blocks as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &n_b as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                }
                let grid = objc2_metal::MTLSize::new(((n_blocks * 32 + 255) / 256) as u64, 1, 1);
                let threads = objc2_metal::MTLSize::new(256, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                encoder.endEncoding();
                cmd_buffer.commit();
                cmd_buffer.waitUntilCompleted();

                let ptr = out_buf.contents() as *const f32;
                let mut values = vec![0.0f32; elem_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), elem_count);
                }
                return Ok(values);
            }
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

    /// Dequantize Q4_K packed bytes to F32 on host/GPU.
    pub fn dequantize_q4k_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Ok(ctx) = MetalContext::get() {
                let n_blocks = bytes.len() / 144;
                let packed_buf = self.new_buffer_with_bytes(bytes, BufferUsage::Shared)?;
                let out_buf = ctx
                    .device
                    .newBufferWithLength_options(
                        (elem_count * 4) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| Error::Backend("Metal dequant_q4k: alloc out failed".into()))?;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Backend("Metal dequant: encoder failed".into()))?;
                encoder.setComputePipelineState(&ctx.pipelines.dequant_q4k);
                encoder.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
                let n_b = n_blocks as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &n_b as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                }
                let grid = objc2_metal::MTLSize::new(((n_blocks * 256 + 255) / 256) as u64, 1, 1);
                let threads = objc2_metal::MTLSize::new(256, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                encoder.endEncoding();
                cmd_buffer.commit();
                cmd_buffer.waitUntilCompleted();

                let ptr = out_buf.contents() as *const f32;
                let mut values = vec![0.0f32; elem_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), elem_count);
                }
                return Ok(values);
            }
        }
        grim_quant::dequant_q4k(bytes, elem_count)
    }

    /// Helper for IQ host dequant dispatches.
    fn dequantize_iq_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        _block_bytes: usize,
        kernel_name: &str,
    ) -> Result<Vec<f32>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Ok(ctx) = MetalContext::get() {
                let n_blocks = bytes.len() / _block_bytes;
                let pipeline = match kernel_name {
                    "iq2xxs" => &ctx.pipelines.dequant_iq2xxs,
                    "iq2xs" => &ctx.pipelines.dequant_iq2xs,
                    "iq2s" => &ctx.pipelines.dequant_iq2s,
                    "iq3xxs" => &ctx.pipelines.dequant_iq3xxs,
                    "iq3s" => &ctx.pipelines.dequant_iq3s,
                    "iq4nl" => &ctx.pipelines.dequant_iq4nl,
                    "iq4xs" => &ctx.pipelines.dequant_iq4xs,
                    _ => return Err(Error::Backend(format!("Unknown iq kernel {kernel_name}"))),
                };
                let packed_buf = self.new_buffer_with_bytes(bytes, BufferUsage::Shared)?;
                let out_buf = ctx
                    .device
                    .newBufferWithLength_options(
                        (elem_count * 4) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::Backend(format!("Metal {kernel_name}: alloc out failed"))
                    })?;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Backend("Metal dequant: encoder failed".into()))?;
                encoder.setComputePipelineState(pipeline);
                encoder.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
                let n_b = n_blocks as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &n_b as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                }
                let grid = objc2_metal::MTLSize::new(((n_blocks * 256 + 255) / 256) as u64, 1, 1);
                let threads = objc2_metal::MTLSize::new(256, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                encoder.endEncoding();
                cmd_buffer.commit();
                cmd_buffer.waitUntilCompleted();

                let ptr = out_buf.contents() as *const f32;
                let mut values = vec![0.0f32; elem_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), elem_count);
                }
                return Ok(values);
            }
        }
        match kernel_name {
            "iq2xxs" => grim_quant::dequant_iq2xxs(bytes, elem_count),
            "iq2xs" => grim_quant::dequant_iq2xs(bytes, elem_count),
            "iq2s" => grim_quant::dequant_iq2s(bytes, elem_count),
            "iq3xxs" => grim_quant::dequant_iq3xxs(bytes, elem_count),
            "iq3s" => grim_quant::dequant_iq3s(bytes, elem_count),
            "iq4nl" => grim_quant::dequant_iq4nl(bytes, elem_count),
            "iq4xs" => grim_quant::dequant_iq4xs(bytes, elem_count),
            _ => Err(Error::Backend(format!("Unknown iq kernel {kernel_name}"))),
        }
    }

    pub fn dequantize_iq2xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 66, "iq2xxs")
    }
    pub fn dequantize_iq2xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 74, "iq2xs")
    }
    pub fn dequantize_iq2s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 82, "iq2s")
    }
    pub fn dequantize_iq3xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 96, "iq3xxs")
    }
    pub fn dequantize_iq3s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 110, "iq3s")
    }
    pub fn dequantize_iq4nl_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 170, "iq4nl")
    }
    pub fn dequantize_iq4xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 178, "iq4xs")
    }

    /// Dequantize packed FP8 bytes (4-byte f32 LE scale header + E4M3 codes).
    pub fn dequantize_fp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let _scale = if bytes.len() >= 4 {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            1.0
        };
        let _payload = if bytes.len() >= 4 { &bytes[4..] } else { bytes };

        #[cfg(target_vendor = "apple")]
        {
            if let Ok(ctx) = MetalContext::get() {
                let packed_buf = self.new_buffer_with_bytes(_payload, BufferUsage::Shared)?;
                let out_buf = ctx
                    .device
                    .newBufferWithLength_options(
                        (elem_count * 4) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| Error::Backend("Metal dequant_fp8: alloc out failed".into()))?;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Backend("Metal dequant: encoder failed".into()))?;
                encoder.setComputePipelineState(&ctx.pipelines.dequant_fp8);
                encoder.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
                let count_i32 = elem_count as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &count_i32 as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                }
                let grid = objc2_metal::MTLSize::new(((elem_count + 255) / 256) as u64, 1, 1);
                let threads = objc2_metal::MTLSize::new(256, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                encoder.endEncoding();
                cmd_buffer.commit();
                cmd_buffer.waitUntilCompleted();

                let ptr = out_buf.contents() as *const f32;
                let mut values = vec![0.0f32; elem_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), elem_count);
                }
                for v in values.iter_mut() {
                    *v *= _scale;
                }
                return Ok(values);
            }
        }
        grim_quant::dequant_fp8(bytes, elem_count)
    }

    /// Helper for MXFP single-buffer dequant.
    fn split_dequant_mxfp_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        is_mxfp4: bool,
    ) -> Result<Vec<f32>> {
        let mut cursor = 0usize;
        let read_segment = |buf: &[u8], cur: &mut usize| -> Result<Vec<u8>> {
            let len = u64::from_le_bytes(
                buf[*cur..*cur + 8]
                    .try_into()
                    .map_err(|_| Error::Backend("mxfp: bad length prefix".into()))?,
            ) as usize;
            *cur += 8;
            let seg = buf[*cur..*cur + len].to_vec();
            *cur += len;
            Ok(seg)
        };
        let _codes = read_segment(bytes, &mut cursor)?;
        let _exps = read_segment(bytes, &mut cursor)?;

        #[cfg(target_vendor = "apple")]
        {
            if let Ok(ctx) = MetalContext::get() {
                let codes_buf = self.new_buffer_with_bytes(&_codes, BufferUsage::Shared)?;
                let exps_buf = self.new_buffer_with_bytes(&_exps, BufferUsage::Shared)?;
                let out_buf = ctx
                    .device
                    .newBufferWithLength_options(
                        (elem_count * 4) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| Error::Backend("Metal dequant_mxfp: alloc out failed".into()))?;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Backend("Metal dequant: encoder failed".into()))?;
                let pipeline = if is_mxfp4 {
                    &ctx.pipelines.dequant_mxfp4
                } else {
                    &ctx.pipelines.dequant_mxfp8
                };
                encoder.setComputePipelineState(pipeline);
                encoder.setBuffer_offset_atIndex(Some(&codes_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&exps_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
                let count_i32 = elem_count as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &count_i32 as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                }
                let grid = objc2_metal::MTLSize::new(((elem_count + 255) / 256) as u64, 1, 1);
                let threads = objc2_metal::MTLSize::new(256, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                encoder.endEncoding();
                cmd_buffer.commit();
                cmd_buffer.waitUntilCompleted();

                let ptr = out_buf.contents() as *const f32;
                let mut values = vec![0.0f32; elem_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), elem_count);
                }
                return Ok(values);
            }
        }
        if is_mxfp4 {
            grim_quant::dequant_mxfp4(bytes, elem_count)
        } else {
            grim_quant::dequant_mxfp8(bytes, elem_count)
        }
    }

    pub fn dequantize_mxfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp_host(bytes, elem_count, true)
    }

    pub fn dequantize_mxfp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp_host(bytes, elem_count, false)
    }

    /// Fused Add + RMSNorm kernel for Metal.
    /// Computes `y = x + residual` and `norm_out = RMSNorm(y, weight, eps)` in a single Metal GPU pass.
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
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let x_s = x
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal x is not MetalStorage".into()))?;
                let res_s = residual
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal residual is not MetalStorage".into()))?;
                let w_s = weight
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal weight is not MetalStorage".into()))?;

                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x lacks buffer".into()))?;
                let res_buf = res_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("residual lacks buffer".into()))?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("weight lacks buffer".into()))?;

                let y_storage = self.zeros(out_shape, x.dtype())?;
                let norm_storage = self.zeros(out_shape, x.dtype())?;

                let y_s = y_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let norm_s = norm_storage
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .unwrap();

                let y_buf = y_s.buffer.as_ref().unwrap();
                let norm_buf = norm_s.buffer.as_ref().unwrap();

                let total = out_shape.elem_count();
                let row_len = x.shape().dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.add_rms_norm);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(res_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(y_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(norm_buf), 0, 4);

                let row_len_val = row_len;
                let eps_val = eps;
                let total_val = total as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &row_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                    encoder.setBytes_length_atIndex(
                        &eps_val as *const f32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        7,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                return Ok((
                    y_storage,
                    norm_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ));
            }
        }

        // CPU Fallback for non-Apple targets
        let (y_storage, h1) = self.add(x, residual, out_shape)?;
        h1.synchronize()?;
        let (norm_storage, h2) = self.rms_norm(y_storage.as_ref(), weight, eps, out_shape)?;
        h2.synchronize()?;
        Ok((y_storage, norm_storage, h2))
    }

    /// Quantize F32 tensor `x` on-device to `format`.
    pub fn quantize_on_device(
        &self,
        x: &dyn BackendStorage,
        format: QuantFormat,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let x_s = x
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal x is not MetalStorage".into()))?;
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x lacks buffer".into()))?;
                let total = x.shape().elem_count();

                let (pipeline, out_bytes, output_dtype) = match format {
                    QuantFormat::Q8_0 => {
                        let n_blocks = (total + 31) / 32;
                        (
                            &inner.pipelines.quant_q8_0,
                            n_blocks * 34,
                            DType {
                                arith: ArithType::F32,
                                storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                            },
                        )
                    }
                    QuantFormat::Fp8 => (
                        &inner.pipelines.quant_fp8,
                        4 + total,
                        DType {
                            arith: ArithType::F32,
                            storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                        },
                    ),
                    QuantFormat::MxFp4 => {
                        let n_groups = (total + 31) / 32;
                        let code_bytes = (total + 1) / 2;
                        let total_bytes = code_bytes + n_groups;
                        (
                            &inner.pipelines.quant_mxfp4,
                            total_bytes,
                            DType {
                                arith: ArithType::F32,
                                storage: DTypeStorage::FloatPack(FloatPackScheme::MxFp4),
                            },
                        )
                    }
                    QuantFormat::MxFp8 => {
                        let n_groups = (total + 31) / 32;
                        let total_bytes = total + n_groups;
                        (
                            &inner.pipelines.quant_mxfp8,
                            total_bytes,
                            DType {
                                arith: ArithType::F32,
                                storage: DTypeStorage::FloatPack(FloatPackScheme::MxFp8),
                            },
                        )
                    }
                    QuantFormat::Q4_K => {
                        let n_superblocks = (total + 255) / 256;
                        (
                            &inner.pipelines.quant_q4k,
                            n_superblocks * 144,
                            DType {
                                arith: ArithType::F32,
                                storage: DTypeStorage::KQuant(KQuantScheme::Q4K),
                            },
                        )
                    }
                    other => {
                        return Err(Error::Backend(format!(
                            "Metal quantize_on_device: unsupported format {:?}",
                            other
                        )));
                    }
                };

                let out_shape = Shape::from_slice(&[out_bytes]);
                let out_storage = self.zeros(&out_shape, output_dtype)?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(pipeline);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);

                let total_val = total as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                }

                match format {
                    QuantFormat::Q8_0 => {
                        let n_blocks = (total + 31) / 32;
                        let threads_per_group = MTLSize::new(32, 1, 1);
                        let groups = MTLSize::new(n_blocks as u64, 1, 1);
                        encoder
                            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                    }
                    QuantFormat::Fp8 => {
                        let threads_per_group = MTLSize::new(256, 1, 1);
                        let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                        encoder
                            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                    }
                    QuantFormat::MxFp4 | QuantFormat::MxFp8 => {
                        let n_groups = (total + 31) / 32;
                        let threads_per_group = MTLSize::new(32, 1, 1);
                        let groups = MTLSize::new(n_groups as u64, 1, 1);
                        encoder
                            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                    }
                    QuantFormat::Q4_K => {
                        let n_superblocks = (total + 255) / 256;
                        let threads_per_group = MTLSize::new(32, 1, 1);
                        let groups = MTLSize::new(n_superblocks as u64, 1, 1);
                        encoder
                            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                    }
                    _ => unreachable!(),
                }
                encoder.endEncoding();

                return Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ));
            }
        }

        let _total = x.shape().elem_count();
        let x_cpu = x.to_cpu_vec_f32()?;
        let (out_bytes, output_dtype) = match format {
            QuantFormat::Q8_0 => {
                let bytes = grim_quant::quant_q80(&x_cpu)?;
                (
                    bytes,
                    DType {
                        arith: ArithType::F32,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                )
            }
            QuantFormat::Fp8 => {
                let bytes = grim_quant::quant_fp8(&x_cpu)?;
                (
                    bytes,
                    DType {
                        arith: ArithType::F32,
                        storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                    },
                )
            }
            other => {
                return Err(Error::Backend(format!(
                    "quantize_on_device unsupported format {:?}",
                    other
                )));
            }
        };

        let out_shape = Shape::from_slice(&[out_bytes.len()]);
        let storage = self.from_cpu_bytes(&out_bytes, &out_shape, output_dtype)?;
        Ok((storage, Box::new(ReadyHandle)))
    }

    /// Fused grouped MoE dispatch (WI-M5). Mirrors `grim_moe_fused_dispatch` on
    /// ROCm and `moe_fused_dispatch` on Vulkan. The MSL kernel runs one thread
    /// per output element of `out` (`[batch, hidden]`) and accumulates the
    /// gated-MLP contributions of every routed (token, expert) pair targeting
    /// that token. The router arrays (`router_tokens`/`router_experts`) are
    /// f32-backed (Metal has no integer buffer storage in this crate) and are
    /// cast to `int` inside the shader.
    #[allow(unused_variables)]
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
        rsf: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let num_pairs = router_tokens.shape().elem_count();

        // GPU fast path
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if let (
                    Some(x_s), Some(gw_s), Some(uw_s), Some(dw_s),
                    Some(rt_s), Some(re_s), Some(rw_s),
                ) = (
                    x.as_any().downcast_ref::<MetalStorage>(),
                    gate_w.as_any().downcast_ref::<MetalStorage>(),
                    up_w.as_any().downcast_ref::<MetalStorage>(),
                    down_w.as_any().downcast_ref::<MetalStorage>(),
                    router_tokens.as_any().downcast_ref::<MetalStorage>(),
                    router_experts.as_any().downcast_ref::<MetalStorage>(),
                    router_weights.as_any().downcast_ref::<MetalStorage>(),
                ) {
                    let bufs = [
                        x_s.buffer.as_ref(), gw_s.buffer.as_ref(), uw_s.buffer.as_ref(),
                        dw_s.buffer.as_ref(), rt_s.buffer.as_ref(), re_s.buffer.as_ref(),
                        rw_s.buffer.as_ref(),
                    ];
                    if bufs.iter().all(|b| b.is_some()) {
                        if let Ok(out_storage) = self.zeros(out_shape, DType::F32) {
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd = self.get_or_create_command_buffer()?;
                            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;

                            encoder.setComputePipelineState(&inner.pipelines.moe_fused_dispatch);
                            for (i, b) in bufs.iter().enumerate() {
                                encoder.setBuffer_offset_atIndex(Some(b.unwrap()), 0, i as u64);
                            }
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 7);

                            let hidden_val = hidden as i32;
                            let inter_val = inter as i32;
                            let num_experts_val = num_experts as i32;
                            let batch_val = batch as i32;
                            let rsf_val = rsf;
                            let num_pairs_val = num_pairs as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(&hidden_val as *const i32 as *const std::ffi::c_void, 4, 8);
                                encoder.setBytes_length_atIndex(&inter_val as *const i32 as *const std::ffi::c_void, 4, 9);
                                encoder.setBytes_length_atIndex(&num_experts_val as *const i32 as *const std::ffi::c_void, 4, 10);
                                encoder.setBytes_length_atIndex(&batch_val as *const i32 as *const std::ffi::c_void, 4, 11);
                                encoder.setBytes_length_atIndex(&rsf_val as *const f32 as *const std::ffi::c_void, 4, 12);
                                encoder.setBytes_length_atIndex(&num_pairs_val as *const i32 as *const std::ffi::c_void, 4, 13);
                            }

                            let grid = MTLSize::new(batch as u64, hidden as u64, 1);
                            let threads = MTLSize::new(1, 1, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                            encoder.endEncoding();

                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd })));
                        }
                    }
                }
            }
        }

        // CPU fallback (also the path on non-Apple hosts)
        let xv = x.to_cpu_vec_f32()?;
        let gw = gate_w.to_cpu_vec_f32()?;
        let uw = up_w.to_cpu_vec_f32()?;
        let dw = down_w.to_cpu_vec_f32()?;
        let rt = router_tokens.to_cpu_vec_f32()?;
        let re = router_experts.to_cpu_vec_f32()?;
        let rw = router_weights.to_cpu_vec_f32()?;

        let hidden_us = hidden as usize;
        let inter_us = inter as usize;
        let batch_us = batch as usize;
        let mut out = vec![0.0f32; batch_us * hidden_us];
        for tok in 0..batch_us {
            let x_base = tok * hidden_us;
            for p in 0..num_pairs {
                if rt[p] as usize != tok {
                    continue;
                }
                let exp_id = re[p] as usize;
                let weight = rw[p];
                let gw_base = exp_id * inter_us * hidden_us;
                let uw_base = exp_id * inter_us * hidden_us;
                let dw_base = exp_id * hidden_us * inter_us;
                for h in 0..hidden_us {
                    let mut down = 0.0f32;
                    for i in 0..inter_us {
                        let mut g = 0.0f32;
                        let mut u = 0.0f32;
                        for j in 0..hidden_us {
                            let xvj = xv[x_base + j];
                            g += gw[gw_base + i * hidden_us + j] * xvj;
                            u += uw[uw_base + i * hidden_us + j] * xvj;
                        }
                        let a = (g / (1.0f32 + (-g).exp())) * u;
                        down += dw[dw_base + h * inter_us + i] * a;
                    }
                    out[tok * hidden_us + h] += rsf * weight * down;
                }
            }
        }
        let storage = self.from_cpu(&out, out_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            Ok((storage, Box::new(MetalHandle { command_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((storage, Box::new(MetalHandle)))
    }
}

impl BackendDevice for MetalDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype)?)
            .ok_or_else(|| {
                Error::from(MetalError::AllocationFailed("Buffer size overflow".into()))
            })?;
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use objc2_metal::MTLResourceOptions;
                let buffer = inner
                    .device
                    .newBufferWithLength_options(
                        bytes as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::from(MetalError::AllocationFailed(
                            "Failed to allocate Metal buffer".into(),
                        ))
                    })?;

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
                // Device-absent fallback via Accelerate framework sgemm
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
                        a_vec.as_ptr(),
                        k as i32,
                        b_vec.as_ptr(),
                        n as i32,
                        0.0,
                        c_vec.as_mut_ptr(),
                        n as i32,
                    );
                }
                let out_storage = self.from_cpu(&c_vec, out, a.dtype())?;
                let ctx = MetalContext::get()?;
                // Device-absent fallback: computation already done via Accelerate; no-op command buffer for MetalHandle.
                let fallback_cmd = ctx.command_queue.commandBuffer().ok_or_else(|| {
                    Error::from(MetalError::Ffi(
                        "Failed to create fallback command buffer".into(),
                    ))
                })?;
                return Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: fallback_cmd,
                    }),
                ));
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
                let a_buf = a_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("a has no GPU buffer".into()))?;
                let b_buf = b_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("b has no GPU buffer".into()))?;

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
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

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
                let config_data = [
                    config.block_m as i32,
                    config.block_n as i32,
                    config.block_k as i32,
                ];
                unsafe {
                    encoder.setBytes_length_atIndex(
                        config_data.as_ptr() as *const std::ffi::c_void,
                        12,
                        6,
                    );
                }

                let threads_per_group =
                    MTLSize::new(config.block_n as u64, config.block_m as u64, 1);
                let groups = MTLSize::new(
                    ((n + (config.block_n as usize) - 1) / (config.block_n as usize)) as u64,
                    ((m + (config.block_m as usize) - 1) / (config.block_m as usize)) as u64,
                    1,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
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

    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let e_s = e.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal silu_mul_backward: e is not MetalStorage".into())
                })?;
                let g_s = g.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal silu_mul_backward: g is not MetalStorage".into())
                })?;
                let dw_s = dw.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal silu_mul_backward: dw is not MetalStorage".into())
                })?;
                let e_buf = e_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("e has no GPU buffer".into()))?;
                let g_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("g has no GPU buffer".into()))?;
                let dw_buf = dw_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("dw has no GPU buffer".into()))?;
                let df_storage = self.zeros(out_shape, DType::F32)?;
                let de_storage = self.zeros(out_shape, DType::F32)?;
                let df_buf = df_storage
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .unwrap()
                    .buffer
                    .as_ref()
                    .unwrap();
                let de_buf = de_storage
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .unwrap()
                    .buffer
                    .as_ref()
                    .unwrap();
                let cmd = self.get_or_create_command_buffer()?;
                let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder.setComputePipelineState(&inner.pipelines.silu_mul_backward);
                encoder.setBuffer_offset_atIndex(Some(e_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(g_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dw_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(df_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(de_buf), 0, 4);
                let total = out_shape.elem_count() as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &total as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }
                let threads = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total as usize + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                encoder.endEncoding();
                return Ok((
                    df_storage,
                    de_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }
        let cpu = CpuDevice::new();
        let e_cpu = e.to_cpu_vec_f32()?;
        let g_cpu = g.to_cpu_vec_f32()?;
        let dw_cpu = dw.to_cpu_vec_f32()?;
        let mut df = vec![0.0f32; out_shape.elem_count()];
        let mut de = vec![0.0f32; out_shape.elem_count()];
        for i in 0..df.len() {
            let s = 1.0 / (1.0 + (-e_cpu[i]).exp());
            df[i] = dw_cpu[i] * g_cpu[i] * s * (1.0 + e_cpu[i] * (1.0 - s));
            de[i] = dw_cpu[i] * s * e_cpu[i];
        }
        let df_storage = cpu.from_cpu(&df, out_shape, DType::F32)?;
        let de_storage = cpu.from_cpu(&de, out_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            return Ok((
                df_storage,
                de_storage,
                Box::new(MetalHandle { command_buffer }),
            ));
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((df_storage, de_storage, Box::new(MetalHandle)))
    }

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
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("w has no GPU buffer".into()))?;

                let out_storage = self.zeros(out, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let total = out.elem_count();
                let row_len = x.shape().dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

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

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
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

    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let (out, _handle) = self.quantize_on_device(x, format)?;
        Ok(out)
    }

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
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;

                let out_storage = self.zeros(out, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let total = out.elem_count();
                let last_dim = out.dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

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

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                let x_vec = x.to_cpu_vec_f32()?;
                tracing::warn!(
                    "Metal softmax: GPU path unavailable, falling back to CPU execution"
                );
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
            tracing::warn!("Metal softmax: non-Apple target, falling back to CPU execution");
            let cpu_dev = CpuDevice::new();
            let x_cpu = cpu_dev.from_cpu(&x_vec, x.shape(), x.dtype())?;
            let x_storage = x_cpu
                .as_any()
                .downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::Backend("Failed to downcast input x to CpuStorage".into()))?;
            let (res_storage, handle) = cpu_dev.softmax(x_storage, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, x.dtype())?;
            Ok((out_metal, handle))
        }
    }

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

                let w_s = weight
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("Metal embedding: weight is not MetalStorage".into())
                    })?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("weight has no GPU buffer".into()))?;

                // Create a temporary buffer for indices.
                let indices_bytes = indices.len().checked_mul(4).ok_or_else(|| {
                    Error::from(MetalError::AllocationFailed("Indices size overflow".into()))
                })?;
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
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

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

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                let w_vec = weight.to_cpu_vec_f32()?;
                tracing::warn!(
                    "Metal embedding: GPU path unavailable, falling back to CPU execution"
                );
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
            tracing::warn!("Metal embedding: non-Apple target, falling back to CPU execution");
            let cpu_dev = CpuDevice::new();
            let w_cpu = cpu_dev.from_cpu(&w_vec, weight.shape(), weight.dtype())?;
            let w_storage = w_cpu
                .as_any()
                .downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::Backend("Failed to downcast weight to CpuStorage".into()))?;
            let (res_storage, handle) = cpu_dev.embedding(w_storage, indices, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, weight.dtype())?;
            Ok((out_metal, handle))
        }
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype)?)
            .ok_or_else(|| {
                Error::from(MetalError::AllocationFailed("Buffer size overflow".into()))
            })?;
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
                let data_bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
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
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        fallback_data.as_mut_ptr(),
                        bytes,
                    );
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
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    fallback_data.as_mut_ptr(),
                    bytes,
                );
            }
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(fallback_data),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
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
                let k_s = k_tensor
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention k_tensor is not MetalStorage".into())
                    })?;
                let ks_s = k_scales
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention k_scales is not MetalStorage".into())
                    })?;
                let v_s = v_tensor
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention v_tensor is not MetalStorage".into())
                    })?;
                let vs_s = v_scales
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("kv_dequant_attention v_scales is not MetalStorage".into())
                    })?;

                let q_buf = q_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
                let k_buf = k_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k_tensor has no GPU buffer".into()))?;
                let ks_buf = ks_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k_scales has no GPU buffer".into()))?;
                let v_buf = v_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v_tensor has no GPU buffer".into()))?;
                let vs_buf = vs_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v_scales has no GPU buffer".into()))?;

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
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

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
                        &inv_sqrt_d as *const f32 as *const std::ffi::c_void,
                        4,
                        12,
                    );
                    encoder.setBytes_length_atIndex(
                        &quant_bits_val as *const i32 as *const std::ffi::c_void,
                        4,
                        13,
                    );
                }

                let threads_per_group = MTLSize::new(1, 1, 1);
                let groups = MTLSize::new(seq_len as u64, num_heads as u64, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                Err(Error::Backend(
                    "Metal device inner is None (fallback mode)".into(),
                ))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (
                q,
                k_tensor,
                k_scales,
                v_tensor,
                v_scales,
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                quant_bits,
                out_shape,
            );
            Err(Error::Unimplemented(
                "kv_dequant_attention not supported on non-Apple platform".into(),
            ))
        }
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
        self.qkv_attention(
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

    #[allow(unused_variables)] // params only used on the cfg-gated Apple path
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
        // The Metal `grim_qkv_attention_paged` kernel accepts a `window_lo` +
        // `has_window` argument pair; SWA layers compute the lower bound
        // host-side and the kernel masks below it. No host fallback needed.

        #[cfg(target_vendor = "apple")]
        if let Some(ref inner) = self.inner {
            let dims = out_shape.dims();
            if dims.len() != 3 {
                return Err(Error::from(MetalError::DataMismatch(
                    "paged attention expects [batch, heads, dim]".into(),
                )));
            }
            let q_s = q
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged q is not MetalStorage".into()))?;
            if num_kv_heads == 0 || dims[1] % num_kv_heads != 0 {
                return Err(Error::from(MetalError::DataMismatch(
                    "paged attention requires num_heads divisible by num_kv_heads".into(),
                )));
            }
            let table_s = block_tables
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged block table is not MetalStorage".into()))?;
            let k_s = k_pages
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged k pages is not MetalStorage".into()))?;
            let v_s = v_pages
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("paged v pages is not MetalStorage".into()))?;
            let q_buf = q_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
            let table_buf = table_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("block table has no GPU buffer".into()))?;
            let k_buf = k_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("k pages has no GPU buffer".into()))?;
            let v_buf = v_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("v pages has no GPU buffer".into()))?;
            let out_storage = self.zeros(out_shape, DType::F32)?;
            let out_buf = out_storage
                .as_any()
                .downcast_ref::<MetalStorage>()
                .unwrap()
                .buffer
                .as_ref()
                .unwrap();
            let cmd = self.get_or_create_command_buffer()?;
            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
            })?;
            encoder.setComputePipelineState(&inner.pipelines.qkv_paged_attn);
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(table_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 4);
            // SWA: window_lo = max(0, cache_offset - window + 1).
            let abs_first = cache_offset as usize;
            let window_lo_val: i32 = match window {
                Some(w) => abs_first.saturating_sub(w.saturating_sub(1)) as i32,
                None => 0,
            };
            let has_window_val: i32 = if window.is_some() { 1 } else { 0 };
            let vals = [
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                page_size as i32,
                max_blocks as i32,
                kv_seq_len as i32,
                num_kv_heads as i32,
                window_lo_val,
                has_window_val,
            ];
            unsafe {
                for (i, value) in vals.iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        value as *const i32 as *const std::ffi::c_void,
                        4,
                        5 + i,
                    );
                }
            }
            encoder.dispatchThreads(
                MTLSize::new(dims[2] as u64, dims[1] as u64, dims[0] as u64),
                MTLSize::new(32, 1, 1),
            );
            encoder.endEncoding();
            return Ok((
                out_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }
        Err(Error::Unimplemented(
            "Metal paged attention requires Apple Metal GPU support".into(),
        ))
    }

    #[allow(unused_variables)] // params only used on the cfg-gated Apple path
    fn tree_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        tree_parents: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dims = out_shape.dims();
        if dims.len() != 4 {
            return Err(Error::from(MetalError::DataMismatch(
                "tree attention expects [batch, 1+gamma, heads, dim]".into(),
            )));
        }
        if num_kv_heads == 0 || dims[2] % num_kv_heads != 0 {
            return Err(Error::from(MetalError::DataMismatch(
                "tree attention requires num_heads divisible by num_kv_heads".into(),
            )));
        }
        #[cfg(target_vendor = "apple")]
        if let Some(ref inner) = self.inner {
            let q_s = q
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree q is not MetalStorage".into()))?;
            let k_s = k
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree k is not MetalStorage".into()))?;
            let v_s = v
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree v is not MetalStorage".into()))?;
            let p_s = tree_parents
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("tree parents is not MetalStorage".into()))?;
            let q_buf = q_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
            let k_buf = k_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("k has no GPU buffer".into()))?;
            let v_buf = v_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("v has no GPU buffer".into()))?;
            let p_buf = p_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("parents has no GPU buffer".into()))?;
            let out_storage = self.zeros(out_shape, DType::F32)?;
            let out_buf = out_storage
                .as_any()
                .downcast_ref::<MetalStorage>()
                .unwrap()
                .buffer
                .as_ref()
                .unwrap();
            let cmd = self.get_or_create_command_buffer()?;
            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
            })?;
            encoder.setComputePipelineState(&inner.pipelines.tree_attn);
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(p_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 4);
            let vals = [
                dims[0] as i32,
                dims[2] as i32,
                kv_seq_len as i32,
                dims[3] as i32,
                (dims[1] - 1) as i32,
                cache_offset as i32,
                num_kv_heads as i32,
            ];
            unsafe {
                for (i, value) in vals.iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        value as *const i32 as *const std::ffi::c_void,
                        4,
                        5 + i,
                    );
                }
            }
            encoder.dispatchThreads(
                MTLSize::new(dims[3] as u64, (dims[1] * dims[2]) as u64, dims[0] as u64),
                MTLSize::new(256, 1, 1),
            );
            encoder.endEncoding();
            return Ok((
                out_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }
        Err(Error::Unimplemented(
            "Metal tree attention requires Apple Metal GPU support".into(),
        ))
    }

    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                return self.run_unary(
                    inner,
                    &inner.pipelines.mul_scalar,
                    x,
                    Some(scalar),
                    out_shape,
                    Some(1),
                    3,
                );
            }
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let res: Vec<f32> = x_vec.into_iter().map(|v| v * scalar).collect();
        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                return self.run_unary(inner, &inner.pipelines.sqrt, x, None, out_shape, None, 2);
            }
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let res: Vec<f32> = x_vec.into_iter().map(|v| v.sqrt()).collect();
        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                return self.run_unary(inner, &inner.pipelines.recip, x, None, out_shape, None, 2);
            }
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let res: Vec<f32> = x_vec.into_iter().map(|v| 1.0 / v).collect();
        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
        #[cfg(target_vendor = "apple")]

        {
            if let Some(ref inner) = self.inner {
                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rope: input x is not MetalStorage".into())
                })?;
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;

                let out_storage = self.zeros(out_shape, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let num_tokens = positions.len();
                let num_heads = (out_shape.elem_count() / (num_tokens * dim)) as i32;
                let head_dim = dim as i32;

                // Upload positions to a temporary GPU buffer.
                let pos_data = std::sync::Arc::new(positions.to_vec());
                let pos_buf = inner
                    .device
                    .newBufferWithLength_options(
                        (positions.len() * std::mem::size_of::<u32>()) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::from(MetalError::AllocationFailed(
                            "Failed to allocate pos buffer for rope".into(),
                        ))
                    })?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pos_data.as_ptr() as *const u8,
                        pos_buf.contents() as *mut u8,
                        positions.len() * std::mem::size_of::<u32>(),
                    );
                }

                let total = out_shape.elem_count();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                let num_tokens_val = num_tokens as i32;
                let num_heads_val = num_heads;
                let head_dim_val = head_dim;
                let base_val = base;

                if !cfg.is_plain() {
                    // Partial-rotary / YaRN: dispatch grim_rope_yarn. The YaRN
                    // ramp + mscale are recomputed inside the kernel.
                    let rotary_dim = cfg.rotary_dim.min(dim) as i32;
                    let rotary_half = (rotary_dim / 2) as usize;
                    let (has_yarn, yarn_factor, yarn_orig_max, yarn_beta_fast, yarn_beta_slow, mscale) =
                        match cfg.yarn {
                            Some(y) => (
                                1i32,
                                y.factor,
                                y.original_max_pos as f32,
                                y.beta_fast,
                                y.beta_slow,
                                y.attention_factor,
                            ),
                            None => (0i32, 1.0f32, 8192.0f32, 32.0f32, 1.0f32, 1.0f32),
                        };
                    encoder.setComputePipelineState(&inner.pipelines.rope_yarn);
                    encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&pos_buf), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                    // Scalar buffers 3..14.
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            &num_tokens_val as *const i32 as *const std::ffi::c_void, 4, 3,
                        );
                        encoder.setBytes_length_atIndex(
                            &num_heads_val as *const i32 as *const std::ffi::c_void, 4, 4,
                        );
                        encoder.setBytes_length_atIndex(
                            &head_dim_val as *const i32 as *const std::ffi::c_void, 4, 5,
                        );
                        encoder.setBytes_length_atIndex(
                            &rotary_dim as *const i32 as *const std::ffi::c_void, 4, 6,
                        );
                        encoder.setBytes_length_atIndex(
                            &has_yarn as *const i32 as *const std::ffi::c_void, 4, 7,
                        );
                        encoder.setBytes_length_atIndex(
                            &base_val as *const f32 as *const std::ffi::c_void, 4, 8,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_factor as *const f32 as *const std::ffi::c_void, 4, 9,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_orig_max as *const f32 as *const std::ffi::c_void, 4, 10,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_beta_fast as *const f32 as *const std::ffi::c_void, 4, 11,
                        );
                        encoder.setBytes_length_atIndex(
                            &yarn_beta_slow as *const f32 as *const std::ffi::c_void, 4, 12,
                        );
                        encoder.setBytes_length_atIndex(
                            &mscale as *const f32 as *const std::ffi::c_void, 4, 13,
                        );
                    }
                    // Grid covers max(num_tokens*num_heads*rotary_half, *copy_len).
                    let copy_len = dim - 2 * rotary_half;
                    let total_pairs = (num_tokens
                        * num_heads as usize
                        * rotary_half.max(if copy_len > 0 { copy_len } else { 0 }).max(1))
                    as u64;
                    let threads_per_group = MTLSize::new(256, 1, 1);
                    let groups = MTLSize::new(((total_pairs + 255) / 256), 1, 1);
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                    encoder.endEncoding();
                    return Ok((
                        out_storage,
                        Box::new(MetalHandle {
                            command_buffer: cmd_buffer,
                        }),
                    ));
                }

                // Plain full-rotary RoPE.
                encoder.setComputePipelineState(&inner.pipelines.rope);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&pos_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &num_tokens_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                    encoder.setBytes_length_atIndex(
                        &head_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                    encoder.setBytes_length_atIndex(
                        &base_val as *const f32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                }

                let half_dim = dim / 2;
                let total_pairs = (total / (half_dim * 2)).max(1);
                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total_pairs + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                return Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ));
            }
        }

        // CPU fallback: non-Apple target or Metal unavailable. Plain RoPE only;
        // refuse non-plain configs so the block-level CPU Rope::forward runs.
        if !cfg.is_plain() {
            return Err(Error::Unimplemented(
                "Metal rope: partial-rotary / YaRN not available on non-Apple CPU fallback; falling back to CPU rope module"
                    .into(),
            ));
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let num_tokens = positions.len();
        let num_heads = out_shape.elem_count() / (num_tokens * dim);
        let half_dim = dim / 2;

        let mut res = x_vec.clone();
        for t in 0..num_tokens {
            let p = positions[t] as f32;
            for h in 0..num_heads {
                for i in 0..half_dim {
                    let freq = 1.0f32 / base.powf((2 * i) as f32 / dim as f32);
                    let val = p * freq;
                    let cos_v = val.cos();
                    let sin_v = val.sin();

                    let base_idx = (t * num_heads + h) * dim;
                    let idx0 = base_idx + i;
                    let idx1 = base_idx + i + half_dim;

                    let v0 = x_vec[idx0];
                    let v1 = x_vec[idx1];

                    res[idx0] = v0 * cos_v - v1 * sin_v;
                    res[idx1] = v0 * sin_v + v1 * cos_v;
                }
            }
        }

        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use objc2_metal::MTLResourceOptions;
                let buffer = inner
                    .device
                    .newBufferWithLength_options(
                        data.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::from(MetalError::AllocationFailed(
                            "Failed to allocate Metal buffer".into(),
                        ))
                    })?;

                let contents = buffer.contents();
                if !contents.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            contents as *mut u8,
                            data.len(),
                        );
                    }
                }

                return Ok(Box::new(MetalStorage {
                    buffer: Some(buffer),
                    data: None,
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }));
            }
        }
        #[cfg(target_vendor = "apple")]
        {
            Ok(Box::new(MetalStorage {
                buffer: None,
                data: Some(std::sync::Mutex::new(data.to_vec())),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(data.to_vec()),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
            self.qkv_attention(q, k, v, num_kv_heads, seq_len, 0, None, out_shape, None, None)?;
        let _ = num_heads;
        let _ = head_dim;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
            self.qkv_attention(q, k, v, num_heads, kv_seq_len, 0, None, out_shape, None, None)?;

        let _ = head_dim;
        let _ = seq_len;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let w_vec = w.to_cpu_vec_f32()?;
        tracing::warn!("Metal rwkv_time_mix: falling back to CPU execution");
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let r_vec = r.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        tracing::warn!("Metal rwkv_channel_mix: falling back to CPU execution");
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        _format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // --- Apple Silicon GPU fast-path ----------------------------------------
        // Each thread computes one output element [row, col] by dequantizing
        // its column of B on-the-fly inside the kernel.  Both A and B-packed
        // must be device-resident MetalStorage buffers.
        #[cfg(target_vendor = "apple")]
        {
            let a_s = a.as_any().downcast_ref::<MetalStorage>();
            let b_s = b_packed.as_any().downcast_ref::<MetalStorage>();
            if let (Some(a_s), Some(b_s)) = (a_s, b_s) {
                if let (Some(a_buf), Some(b_buf)) = (a_s.buffer.as_ref(), b_s.buffer.as_ref()) {
                    if let Ok(ctx) = MetalContext::get() {
                        if let DTypeStorage::ResidualPacked(cfg) = b_packed.dtype().storage {
                            let residuals =
                                match b_packed.provenance() {
                                    QuantProvenance::WithResiduals {
                                        outlier_count,
                                        outlier_indices_offset,
                                        outlier_values_offset,
                                        backup1_bpw,
                                        backup1_codes_offset,
                                        backup1_scale_offset,
                                        backup2_bpw,
                                        backup2_codes_offset,
                                        backup2_scale_offset,
                                    } => (
                                        outlier_count,
                                        outlier_indices_offset,
                                        outlier_values_offset,
                                        backup1_bpw,
                                        backup1_codes_offset,
                                        backup1_scale_offset,
                                        backup2_bpw,
                                        backup2_codes_offset,
                                        backup2_scale_offset,
                                    ),
                                    _ => return Err(Error::Unimplemented(
                                        "Metal ResidualPacked requires WithResiduals provenance"
                                            .into(),
                                    )),
                                };
                            if cfg.bpw == 0 || cfg.bpw > 8 || k == 0 || n == 0 {
                                return Err(Error::Shape(
                                    "invalid ResidualPacked dimensions or bitwidth".into(),
                                ));
                            }
                            let row_stride = ((k * cfg.bpw as usize).div_ceil(8) + 255) / 256 * 256;
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    b_buf.contents() as *const u8,
                                    b_buf.length() as usize,
                                )
                            };
                            let decode_scales = |offset: usize| -> Vec<f32> {
                                if offset == 0 {
                                    vec![1.0; n]
                                } else {
                                    (0..n)
                                        .map(|i| {
                                            bytes.get(offset + i).copied().unwrap_or(255) as f32
                                                / 255.0
                                        })
                                        .collect()
                                }
                            };
                            let (
                                outlier_count,
                                oi_off,
                                ov_off,
                                b1,
                                b1_off,
                                b1_scale,
                                b2,
                                b2_off,
                                b2_scale,
                            ) = residuals;
                            let mut scales = vec![1.0f32; n];
                            scales[..b_scales.len().min(n)]
                                .copy_from_slice(&b_scales[..b_scales.len().min(n)]);
                            scales.extend(decode_scales(b1_scale));
                            scales.extend(decode_scales(b2_scale));
                            let mut indices = Vec::<u32>::with_capacity(outlier_count);
                            let mut values = Vec::<f32>::with_capacity(outlier_count);
                            for i in 0..outlier_count {
                                let p = oi_off + i * 6;
                                let q = ov_off + i * 6;
                                if p + 4 > bytes.len() || q + 2 > bytes.len() {
                                    return Err(Error::Backend(
                                        "ResidualPacked outlier region exceeds Metal buffer".into(),
                                    ));
                                }
                                indices
                                    .push(u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()));
                                let h = u16::from_le_bytes(bytes[q..q + 2].try_into().unwrap());
                                let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
                                let exp = ((h >> 10) & 0x1f) as i32;
                                let mant = (h & 0x3ff) as u32;
                                values.push(if exp == 0 {
                                    sign * (mant as f32) * 2.0f32.powi(-24)
                                } else {
                                    sign * (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 25)
                                });
                            }
                            let make_buf = |ptr: *const std::ffi::c_void, len: usize| {
                                ctx.device
                                    .newBufferWithBytes_length_options(
                                        ptr,
                                        len as u64,
                                        MTLResourceOptions::StorageModeShared,
                                    )
                                    .ok()
                                    .ok_or_else(|| {
                                        Error::from(MetalError::AllocationFailed(
                                            "ResidualPacked auxiliary buffer allocation failed"
                                                .into(),
                                        ))
                                    })
                            };
                            let scales_buf =
                                make_buf(scales.as_ptr() as *const _, scales.len() * 4)?;
                            let idx_buf =
                                make_buf(indices.as_ptr() as *const _, indices.len().max(1) * 4)?;
                            let val_buf =
                                make_buf(values.as_ptr() as *const _, values.len().max(1) * 4)?;
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_buf = out_storage
                                .as_any()
                                .downcast_ref::<MetalStorage>()
                                .unwrap()
                                .buffer
                                .as_ref()
                                .unwrap();
                            let cmd = self.get_or_create_command_buffer()?;
                            let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;
                            enc.setComputePipelineState(&ctx.pipelines.residualpacked_matmul);
                            for (buf, idx) in [
                                Some(a_buf),
                                Some(b_buf),
                                Some(&scales_buf),
                                Some(&idx_buf),
                                Some(&val_buf),
                                Some(out_buf),
                            ]
                            .iter()
                            .enumerate()
                            {
                                enc.setBuffer_offset_atIndex(*buf, 0, idx as usize);
                            }
                            let vals = [
                                m as i32,
                                n as i32,
                                k as i32,
                                cfg.bpw as i32,
                                row_stride as i32,
                                0,
                                b1 as i32,
                                b1_off as i32,
                                b1_scale as i32,
                                b2 as i32,
                                b2_off as i32,
                                b2_scale as i32,
                                outlier_count as i32,
                            ];
                            unsafe {
                                for (i, v) in vals.iter().enumerate() {
                                    enc.setBytes_length_atIndex(
                                        v as *const i32 as *const _,
                                        4,
                                        6 + i,
                                    );
                                }
                            }
                            enc.dispatchThreadgroups(
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1),
                                MTLSize::new(16, 16, 1),
                            );
                            enc.endEncoding();
                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd,
                                }),
                            ));
                        }
                        if k >= 32 && k % 32 == 0 {
                            // Pad / truncate scales to exactly n * (k/32) entries.
                            let blocks_per_col = k / 32;
                            let scales_len = n * blocks_per_col;
                            let mut scales_f32 = vec![1.0f32; scales_len];
                            let copy_len = b_scales.len().min(scales_len);
                            scales_f32[..copy_len].copy_from_slice(&b_scales[..copy_len]);

                            let scales_buf = ctx
                                .device
                                .newBufferWithBytes_length_options(
                                    scales_f32.as_ptr() as *const std::ffi::c_void,
                                    (scales_f32.len() * 4) as u64,
                                    MTLResourceOptions::StorageModeShared,
                                )
                                .ok_or_else(|| {
                                    Error::from(MetalError::AllocationFailed(
                                        "Failed to allocate scales buffer".into(),
                                    ))
                                })?;

                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&ctx.pipelines.quantized_matmul);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(&scales_buf), 0, 2);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);

                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    6,
                                );
                            }

                            // 16×16 threadgroup = 256 threads, matching CUDA block size.
                            let threads_per_group = MTLSize::new(16, 16, 1);
                            let groups =
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                groups,
                                threads_per_group,
                            );
                            encoder.endEncoding();

                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd_buffer,
                                }),
                            ));
                        }
                    }
                }
            }
        }
        // --- end GPU fast-path ---------------------------------------------------

        // CPU fallback: dequant b and compute matmul on host.
        tracing::warn!("Metal quantized_matmul: falling back to CPU execution");
        let a_vec = a.to_cpu_vec_f32()?;
        let mut b_dequant = vec![0.0f32; k * n];
        let blocks_per_col = k / 32;

        #[cfg(target_vendor = "apple")]
        let b_bytes = if let Some(ref m_s) = b_packed.as_any().downcast_ref::<MetalStorage>() {
            if let Some(ref buf) = m_s.buffer {
                let ptr = buf.contents() as *const u8;
                let len = m_s.shape.elem_count();
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            } else if let Some(ref d) = m_s.data {
                d.lock().unwrap().clone()
            } else {
                vec![0u8; k * n]
            }
        } else {
            vec![0u8; k * n]
        };

        #[cfg(not(target_vendor = "apple"))]
        let b_bytes = if let Some(ref m_s) = b_packed.as_any().downcast_ref::<MetalStorage>() {
            m_s.data.lock().unwrap().clone()
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
                        128u8
                    };
                    let q_val = (byte_val as i16 - 128) as f32 / 127.0f32;
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        residuals: Option<&grim_tensor::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Basic validation - only Q8_0 (8-bit) supported, requires k >= 32 and block-aligned
        if default_bpw != 8 || k < 32 || k % 32 != 0 {
            return Err(Error::Unimplemented(
                "Metal Q8_0 backward supports only 8-bit block-aligned tensors".into(),
            ));
        }

        // For residuals (outliers, backup layers), fall back to CPU
        // This matches ROCm behavior where residuals cause a fallback path
        if let Some(res) = residuals {
            if res.outlier_count > 0 || res.backup1_bpw > 0 || res.backup2_bpw > 0 {
                let dy_vec = dy.to_cpu_vec_f32()?;
                let b_bytes = b_packed.to_cpu_vec_f32()?;
                let mut dx = vec![0.0f32; m * k];
                let blocks_per_col = k / 32;

                for row in 0..m {
                    for ki in 0..k {
                        let block = ki / 32;
                        let in_block = ki % 32;
                        let mut sum = 0.0f32;
                        for col in 0..n {
                            let idx = (col * blocks_per_col + block) * 32 + in_block;
                            let q = b_bytes.get(idx).copied().unwrap_or(0.0);
                            let scale = b_scales
                                .get(col * blocks_per_col + block)
                                .copied()
                                .unwrap_or(1.0);
                            sum += dy_vec[row * n + col] * q * scale;
                        }
                        dx[row * k + ki] = sum;
                    }
                }
                return Ok((
                    self.from_cpu(&dx, out_shape, DType::F32)?,
                    Box::new(MetalHandle),
                ));
            }
        }

        // Apple Metal GPU fast-path
        #[cfg(target_vendor = "apple")]
        if let Some(ref inner) = self.inner {
            let dy_s = dy.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal Q8_0 backward dy is not MetalStorage".into())
            })?;
            let b_s = b_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend("Metal Q8_0 backward b is not MetalStorage".into())
                })?;

            let dy_buf = dy_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("Metal Q8_0 backward dy has no GPU buffer".into()))?;
            let b_buf = b_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("Metal Q8_0 backward b has no GPU buffer".into()))?;

            let ctx = MetalContext::get()?;
            let scale_count = n * (k / 32);
            let mut scales = vec![1.0f32; scale_count];
            let copy_len = b_scales.len().min(scale_count);
            scales[..copy_len].copy_from_slice(&b_scales[..copy_len]);

            let scales_buf = ctx
                .device
                .newBufferWithBytes_length_options(
                    scales.as_ptr() as *const std::ffi::c_void,
                    (scales.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    Error::from(MetalError::Ffi(
                        "Failed to allocate Q8_0 scale buffer".into(),
                    ))
                })?;

            let dx_storage = self.zeros(out_shape, DType::F32)?;
            let dx_s = dx_storage
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("dx_storage is not MetalStorage".into()))?;
            let dx_buf = dx_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("dx storage has no GPU buffer".into()))?;

            let cmd = self.get_or_create_command_buffer()?;
            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
            })?;

            encoder.setComputePipelineState(&inner.pipelines.quantized_matmul_backward);
            encoder.setBuffer_offset_atIndex(Some(dy_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&scales_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(dx_buf), 0, 3);

            let m_i = m as i32;
            let n_i = n as i32;
            let k_i = k as i32;
            unsafe {
                encoder.setBytes_length_atIndex(
                    &m_i as *const i32 as *const std::ffi::c_void,
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    &n_i as *const i32 as *const std::ffi::c_void,
                    4,
                    5,
                );
                encoder.setBytes_length_atIndex(
                    &k_i as *const i32 as *const std::ffi::c_void,
                    4,
                    6,
                );
            }

            // Grid: (k/16) threadgroups in x, (m/16) in y, 1 in z
            // Each thread Computes dx[row, k_idx] for row < m, k_idx < k
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize::new(((k + 15) / 16) as u64, ((m + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1),
            );
            encoder.endEncoding();

            return Ok((
                dx_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }

        // Non-Apple fallback to CPU
        // CPU fallback implementation for quantized matmul backward
        let dy_vec = dy.to_cpu_vec_f32()?;
        let b_bytes = b_packed.to_cpu_vec_f32()?;
        let mut dx = vec![0.0f32; m * k];
        let blocks_per_col = k / 32;

        for row in 0..m {
            for ki in 0..k {
                let block = ki / 32;
                let in_block = ki % 32;
                let mut sum = 0.0f32;
                for col in 0..n {
                    let idx = (col * blocks_per_col + block) * 32 + in_block;
                    let q = b_bytes.get(idx).copied().unwrap_or(0.0);
                    let scale = b_scales
                        .get(col * blocks_per_col + block)
                        .copied()
                        .unwrap_or(1.0);
                    sum += dy_vec[row * n + col] * q * scale;
                }
                dx[row * k + ki] = sum;
            }
        }
        Ok((
            self.from_cpu(&dx, out_shape, DType::F32)?,
            Box::new(MetalHandle),
        ))
    }

    #[allow(unused_variables)] // locals only used on the cfg-gated Apple path
    fn all_reduce(
        &self,
        inputs: &[&dyn BackendStorage],
        op: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if inputs.is_empty() {
            return Err(Error::Backend("all_reduce: no inputs".into()));
        }
        if op != "sum" {
            return Err(Error::Backend(format!(
                "all_reduce: only 'sum' supported, got '{op}'"
            )));
        }
        let shape = inputs[0].shape().clone();
        let dtype = inputs[0].dtype();
        let total = shape.elem_count();
        let is_f32 = dtype.arith == ArithType::F32;

        // All inputs must share the same shape.
        for s in inputs {
            if s.shape() != &shape {
                return Err(Error::Backend("all_reduce: input shape mismatch".into()));
            }
        }

        // ── GPU fast path: zero the output, then accumulate each input in turn.
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if is_f32 && total > 0 {
                    // Validate that every input is GPU-backed before dispatching.
                    let mut input_bufs: Vec<&Retained<ProtocolObject<dyn MTLBuffer>>> =
                        Vec::with_capacity(inputs.len());
                    let mut valid = true;
                    for input in inputs {
                        match input.as_any().downcast_ref::<MetalStorage>() {
                            Some(s) => match &s.buffer {
                                Some(b) => input_bufs.push(b),
                                None => {
                                    valid = false;
                                    break;
                                }
                            },
                            None => {
                                valid = false;
                                break;
                            }
                        }
                    }
                    if valid {
                        if let Ok(out_storage) = self.zeros(&shape, DType::F32) {
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd = self.get_or_create_command_buffer()?;
                            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&inner.pipelines.all_reduce);
                            let n_val = total as i32;
                            let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                            let threads = MTLSize::new(256, 1, 1);
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    2,
                                );
                            }
                            for in_buf in &input_bufs {
                                encoder.setBuffer_offset_atIndex(Some(*in_buf), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            }
                            encoder.endEncoding();

                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd,
                                }),
                            ));
                        }
                    }
                }
            }
        }

        // ── CPU fallback ─────────────────────────────────────────────────
        let mut acc = inputs[0].to_cpu_vec_f32()?;
        for other in &inputs[1..] {
            let v = other.to_cpu_vec_f32()?;
            if v.len() != acc.len() {
                return Err(Error::Backend(
                    "all_reduce: input length mismatch during fallback".into(),
                ));
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        let storage = self.from_cpu(&acc, &shape, dtype)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            Ok((storage, Box::new(MetalHandle { command_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((storage, Box::new(MetalHandle)))
    }

    #[allow(unused_variables)] // locals only used on the cfg-gated Apple path
    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        if partials.is_empty() {
            return Err(Error::Backend("comm_fuse_reduce: no partials".into()));
        }
        let dims0 = partials[0].0.shape().dims();
        let m = dims0[0];
        let n_total: usize = partials
            .iter()
            .map(|(s, _)| s.shape().dims().get(1).copied().unwrap_or(0))
            .sum();
        let dtype = partials[0].0.dtype();
        let is_f32 = dtype.arith == ArithType::F32;
        let out_shape = Shape::new(vec![m, n_total]);

        // ── GPU fast path: zero the output, then scatter-copy each shard.
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if is_f32 && n_total > 0 {
                    // Validate that every shard is GPU-backed before dispatching.
                    let mut entries: Vec<(&Retained<ProtocolObject<dyn MTLBuffer>>, usize)> =
                        Vec::with_capacity(partials.len());
                    let mut valid = true;
                    for (storage, _placement) in partials {
                        match storage.as_any().downcast_ref::<MetalStorage>() {
                            Some(s) => match &s.buffer {
                                Some(b) => {
                                    let n_src = s.shape().dims().get(1).copied().unwrap_or(0);
                                    entries.push((b, n_src));
                                }
                                None => {
                                    valid = false;
                                    break;
                                }
                            },
                            None => {
                                valid = false;
                                break;
                            }
                        }
                    }
                    if valid {
                        if let Ok(out_storage) = self.zeros(&out_shape, DType::F32) {
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd = self.get_or_create_command_buffer()?;
                            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&inner.pipelines.comm_fuse_reduce);
                            let m_val = m as i32;
                            let n_total_val = n_total as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    2,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_total_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let mut col_offset = 0usize;
                            for (in_buf, n_src) in &entries {
                                encoder.setBuffer_offset_atIndex(Some(*in_buf), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                                let n_src_val = *n_src as i32;
                                let col_offset_val = col_offset as i32;
                                unsafe {
                                    encoder.setBytes_length_atIndex(
                                        &n_src_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        3,
                                    );
                                    encoder.setBytes_length_atIndex(
                                        &col_offset_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        4,
                                    );
                                }
                                let groups = MTLSize::new(
                                    ((*n_src + 15) / 16) as u64,
                                    ((m + 15) / 16) as u64,
                                    1,
                                );
                                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                                col_offset += *n_src;
                            }
                            encoder.endEncoding();

                            return Ok(Box::new(out_storage));
                        }
                    }
                }
            }
        }

        // ── CPU fallback ─────────────────────────────────────────────────
        let mut assembled = vec![0.0f32; m * n_total];
        let mut col_offset = 0usize;
        for (storage, _placement) in partials {
            let data = storage.to_cpu_vec_f32()?;
            let n_cols = storage.shape().dims().get(1).copied().unwrap_or(0);
            for row in 0..m {
                for col in 0..n_cols {
                    assembled[row * n_total + col_offset + col] += data[row * n_cols + col];
                }
            }
            col_offset += n_cols;
        }
        let storage = self.from_cpu(&assembled, &out_shape, dtype)?;
        Ok(storage)
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
            ArithType::F16 | ArithType::BF16 => 200.0,
            ArithType::F32 => 100.0,
            _ => 50.0,
        };
        (flops / (tflops * 1e12) * 1000.0).max(0.01)
    }
}

impl MetalDevice {
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
        // The Metal `grim_qkv_attention` kernel accepts a `window_lo` +
        // `has_window` argument pair; SWA layers compute the lower bound
        // host-side and the kernel masks below it. No host fallback needed.
        // (On non-Apple targets the Apple dispatch block is cfg'd out, so
        // reference `window` here to keep the binding live.)
        let _ = &window;

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
                if q.dtype().arith != ArithType::F32
                    || k.dtype().arith != ArithType::F32
                    || v.dtype().arith != ArithType::F32
                {
                    return Err(Error::from(MetalError::UnsupportedDType(q.dtype())));
                }

                let q_s = q
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("qkv_attention q is not MetalStorage".into()))?;
                let k_s = k
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("qkv_attention k is not MetalStorage".into()))?;
                let v_s = v
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("qkv_attention v is not MetalStorage".into()))?;

                let q_buf = q_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
                let k_buf = k_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k has no GPU buffer".into()))?;
                let v_buf = v_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v has no GPU buffer".into()))?;

                let max_s = match out_max {
                    Some(m) => {
                        let ms = m.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                            Error::Backend("qkv_attention out_max is not MetalStorage".into())
                        })?;
                        Some(
                            ms.buffer.as_ref().ok_or_else(|| {
                                Error::Backend("out_max has no GPU buffer".into())
                            })?,
                        )
                    }
                    None => None,
                };
                let sum_s = match out_sum {
                    Some(s) => {
                        let ss = s.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                            Error::Backend("qkv_attention out_sum is not MetalStorage".into())
                        })?;
                        Some(
                            ss.buffer.as_ref().ok_or_else(|| {
                                Error::Backend("out_sum has no GPU buffer".into())
                            })?,
                        )
                    }
                    None => None,
                };

                let out_storage = self.zeros(out, DType::F32)?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

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
                    // SWA: window_lo = max(0, cache_offset - window + 1);
                    // has_window = window.is_some().
                    let abs_first = cache_offset as usize;
                    let window_lo_val: i32 = match window {
                        Some(w) => abs_first.saturating_sub(w.saturating_sub(1)) as i32,
                        None => 0,
                    };
                    let has_window_val: i32 = if window.is_some() { 1 } else { 0 };
                    encoder.setBytes_length_atIndex(
                        &window_lo_val as *const i32 as *const std::ffi::c_void,
                        4,
                        13,
                    );
                    encoder.setBytes_length_atIndex(
                        &has_window_val as *const i32 as *const std::ffi::c_void,
                        4,
                        14,
                    );
                }

                let threads_per_group = MTLSize::new(32, 1, 1);
                let groups = MTLSize::new(seq_len as u64, num_heads as u64, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                let _ = out_max;
                let _ = out_sum;
                // Host-fallback for unit tests without Apple hardware.
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
                        let range_len = if abs_i < kv_seq_len {
                            abs_i + 1
                        } else {
                            kv_seq_len
                        };

                        let mut running_max = -1e30_f32;
                        let mut running_sum = 0.0_f32;

                        let mut scores = vec![0.0f32; range_len];
                        for j in 0..range_len {
                            let mut score = 0.0_f32;
                            for d in 0..head_dim {
                                score += q_vec[q_offset + d]
                                    * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
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
                                let weight = (scores[j] - running_max).exp()
                                    / (if running_sum > 0.0_f32 {
                                        running_sum
                                    } else {
                                        1.0_f32
                                    });
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
            // Host-fallback for unit tests without Apple hardware.
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
                    let range_len = if abs_i < kv_seq_len {
                        abs_i + 1
                    } else {
                        kv_seq_len
                    };

                    let mut running_max = -1e30_f32;
                    let mut running_sum = 0.0_f32;

                    let mut scores = vec![0.0f32; range_len];
                    for j in 0..range_len {
                        let mut score = 0.0_f32;
                        for d in 0..head_dim {
                            score += q_vec[q_offset + d]
                                * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
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
                            let weight = (scores[j] - running_max).exp()
                                / (if running_sum > 0.0_f32 {
                                    running_sum
                                } else {
                                    1.0_f32
                                });
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
        let a_buf = a_s
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("a has no GPU buffer".into()))?;
        let b_buf = b_s
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("b has no GPU buffer".into()))?;

        let out_storage = self.zeros(out, a.dtype())?;
        let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
        let out_buf = out_s.buffer.as_ref().unwrap();

        let total = out.elem_count();

        let cmd_buffer = self.get_or_create_command_buffer()?;
        let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
            Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
        })?;

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

        Ok((
            out_storage,
            Box::new(MetalHandle {
                command_buffer: cmd_buffer,
            }),
        ))
    }

    #[cfg(target_vendor = "apple")]
    fn run_unary(
        &self,
        inner: &MetalDeviceInner,
        pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        input: &dyn BackendStorage,
        scalar: Option<f32>,
        out: &Shape,
        scalar_binding: Option<usize>,
        n_binding: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let input_s = input
            .as_any()
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| Error::Backend("Metal unary: input is not MetalStorage".into()))?;
        let input_buf = input_s
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("input has no GPU buffer".into()))?;

        let out_storage = self.zeros(out, input.dtype())?;
        let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
        let out_buf = out_s.buffer.as_ref().unwrap();

        let total = out.elem_count();

        let cmd_buffer = self.get_or_create_command_buffer()?;
        let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
            Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
        })?;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);

        let total_val = total as i32;
        if let Some(s_val) = scalar {
            if let Some(sb) = scalar_binding {
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &s_val as *const f32 as *const std::ffi::c_void,
                        4,
                        sb as u64,
                    );
                }
            }
        }
        unsafe {
            encoder.setBytes_length_atIndex(
                &total_val as *const i32 as *const std::ffi::c_void,
                4,
                n_binding as u64,
            );
        }

        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();

        Ok((
            out_storage,
            Box::new(MetalHandle {
                command_buffer: cmd_buffer,
            }),
        ))
    }

    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code, unused_variables)] // stub for non-Apple builds
    fn run_unary(
        &self,
        _input: &dyn BackendStorage,
        _scalar: Option<f32>,
        out: &Shape,
        _scalar_binding: Option<usize>,
        _n_binding: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = _scalar;
        let _ = _scalar_binding;
        let _ = _n_binding;
        // Non-Apple target — callers should fall back to CPU.
        Err(Error::Backend(
            "Metal unary kernel not available on this platform".into(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
}

pub struct Tuner;

impl Tuner {
    pub fn new() -> Self {
        Self
    }

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
                MetalTileConfig {
                    block_m: 8,
                    block_n: 8,
                    block_k: 8,
                },
                MetalTileConfig {
                    block_m: 16,
                    block_n: 16,
                    block_k: 16,
                },
                MetalTileConfig {
                    block_m: 32,
                    block_n: 16,
                    block_k: 16,
                },
                MetalTileConfig {
                    block_m: 16,
                    block_n: 32,
                    block_k: 16,
                },
            ];
            let mut best_config = candidates[1];
            let mut best_time = std::time::Duration::MAX;

            use objc2_metal::MTResourceOptions;
            let bytes_a = m * k * 4;
            let bytes_b = k * n * 4;
            let bytes_c = m * n * 4;
            let buf_a = inner
                .device
                .newBufferWithLength_options(bytes_a as u64, MTResourceOptions::StorageModeShared)
                .unwrap();
            let buf_b = inner
                .device
                .newBufferWithLength_options(bytes_b as u64, MTResourceOptions::StorageModeShared)
                .unwrap();
            let buf_c = inner
                .device
                .newBufferWithLength_options(bytes_c as u64, MTResourceOptions::StorageModeShared)
                .unwrap();

            for &config in &candidates {
                let config_data = [
                    config.block_m as i32,
                    config.block_n as i32,
                    config.block_k as i32,
                ];
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
                                enc.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                enc.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                enc.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                                enc.setBytes_length_atIndex(
                                    config_data.as_ptr() as *const std::ffi::c_void,
                                    12,
                                    6,
                                );
                            }
                            let threads =
                                MTLSize::new(config.block_n as u64, config.block_m as u64, 1);
                            let groups = MTLSize::new(
                                ((n + (config.block_n as usize) - 1) / (config.block_n as usize))
                                    as u64,
                                ((m + (config.block_m as usize) - 1) / (config.block_m as usize))
                                    as u64,
                                1,
                            );
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
                                enc.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                enc.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                enc.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                                enc.setBytes_length_atIndex(
                                    config_data.as_ptr() as *const std::ffi::c_void,
                                    12,
                                    6,
                                );
                            }
                            let threads =
                                MTLSize::new(config.block_n as u64, config.block_m as u64, 1);
                            let groups = MTLSize::new(
                                ((n + (config.block_n as usize) - 1) / (config.block_n as usize))
                                    as u64,
                                ((m + (config.block_m as usize) - 1) / (config.block_m as usize))
                                    as u64,
                                1,
                            );
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
    fn with_persistent_cache<F>(&self, key: (usize, usize, usize), benchmark: F) -> MetalTileConfig
    where
        F: FnOnce() -> MetalTileConfig,
    {
        use std::collections::HashMap;
        use std::sync::Mutex;
        use std::sync::OnceLock;

        static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), (u32, u32, u32)>>> =
            OnceLock::new();
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

#[cfg(not(target_vendor = "apple"))]
fn run_fallback_binary<F>(
    device: &MetalDevice,
    a: &dyn BackendStorage,
    b: &dyn BackendStorage,
    out: &Shape,
    op: F,
) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>
where
    F: FnOnce(
        &CpuDevice,
        &CpuStorage,
        &CpuStorage,
        &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>,
{
    let a_vec = a.to_cpu_vec_f32()?;
    let b_vec = b.to_cpu_vec_f32()?;

    let cpu_dev = CpuDevice::new();
    let a_cpu = cpu_dev.from_cpu(&a_vec, a.shape(), a.dtype())?;
    let b_cpu = cpu_dev.from_cpu(&b_vec, b.shape(), b.dtype())?;

    let a_storage = a_cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Backend("Failed to downcast input a to CpuStorage".into()))?;
    let b_storage = b_cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Backend("Failed to downcast input b to CpuStorage".into()))?;

    let (res_storage, handle) = op(&cpu_dev, a_storage, b_storage, out)?;

    let res_vec = res_storage.to_cpu_vec_f32()?;
    let out_metal = device.from_cpu(&res_vec, out, a.dtype())?;

    Ok((out_metal, handle))
}

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

pub struct MlxBridge;

impl MlxBridge {
    pub fn new() -> Self {
        Self
    }

    /// Zero-copy maps a MetalStorage buffer to an MLX array.
    #[cfg(target_vendor = "apple")]
    pub unsafe fn to_mlx_array(&self, storage: &MetalStorage) -> Result<*mut std::ffi::c_void> {
        let buffer = storage
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("Storage lacks an active Metal buffer".into()))?;
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
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 4]);
        let vec = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(vec, vec![0.0f32; 8]);
    }

    #[test]
    fn test_metal_matmul() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let a = dev
            .from_cpu(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), DType::F32)
            .unwrap();
        let b = dev
            .from_cpu(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), DType::F32)
            .unwrap();
        let out_shape = Shape::new(vec![2, 2]);
        let (out, handle) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_metal_add() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let a = dev
            .from_cpu(&[1.0, 2.0], &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let b = dev
            .from_cpu(&[3.0, 4.0], &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let (out, handle) = dev
            .add(a.as_ref(), b.as_ref(), &Shape::new(vec![2]))
            .unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![4.0, 6.0]);
    }

    #[test]
    fn test_metal_qkv_attention() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let q = dev
            .from_cpu(
                &[1.0, 0.0, 0.0, 1.0],
                &Shape::new(vec![1, 2, 2]),
                DType::F32,
            )
            .unwrap();
        let k = dev
            .from_cpu(
                &[1.0, 0.0, 0.0, 1.0],
                &Shape::new(vec![1, 2, 2]),
                DType::F32,
            )
            .unwrap();
        let v = dev
            .from_cpu(
                &[2.0, 3.0, 4.0, 5.0],
                &Shape::new(vec![1, 2, 2]),
                DType::F32,
            )
            .unwrap();
        let out_shape = Shape::new(vec![1, 2, 2]);
        let (out, handle) = dev
            .qkv_attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                2,
                1,
                0,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_dtype_guards_negative() {
        // GPU path tests for apple Silicon (only run if hardware is available)
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        if dev.inner.is_some() {
            // Attempt to run matmul with a non-F32 dtype (e.g. U8 or F16)
            let a = dev
                .from_cpu(&[1.0, 2.0], &Shape::new(vec![1, 2]), DType::U8)
                .unwrap();
            let b = dev
                .from_cpu(&[3.0, 4.0], &Shape::new(vec![2, 1]), DType::U8)
                .unwrap();
            let out_shape = Shape::new(vec![1, 1]);
            let res = dev.matmul(a.as_ref(), b.as_ref(), &out_shape);
            assert!(
                res.is_err(),
                "Expected matmul with non-F32 inputs to fail on GPU"
            );
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_shape_mismatches_negative() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        if dev.inner.is_some() {
            let a = dev
                .from_cpu(&[1.0, 2.0], &Shape::new(vec![1, 2]), DType::F32)
                .unwrap();
            let b = dev
                .from_cpu(&[3.0, 4.0], &Shape::new(vec![3, 1]), DType::F32)
                .unwrap();
            let out_shape = Shape::new(vec![1, 1]);
            let res = dev.matmul(a.as_ref(), b.as_ref(), &out_shape);
            assert!(res.is_err(), "Expected shape mismatch to return error");
        }
    }

    #[test]
    fn test_metal_math_ops() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
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
    fn test_metal_kv_dequant_attention() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let q = dev
            .from_cpu(
                &[1.0, 0.0, 0.0, 1.0],
                &Shape::new(vec![1, 2, 2]),
                DType::F32,
            )
            .unwrap();
        let k_tensor = dev
            .from_cpu(
                &[1.0, 0.0, 0.0, 1.0],
                &Shape::new(vec![1, 2, 2]),
                DType::F32,
            )
            .unwrap();
        let k_scales = dev
            .from_cpu(&[1.0, 1.0], &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let v_tensor = dev
            .from_cpu(
                &[2.0, 3.0, 4.0, 5.0],
                &Shape::new(vec![1, 2, 2]),
                DType::F32,
            )
            .unwrap();
        let v_scales = dev
            .from_cpu(&[1.0, 1.0], &Shape::new(vec![2]), DType::F32)
            .unwrap();
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
            let a = dev
                .from_cpu(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), DType::F32)
                .unwrap();
            let b = dev
                .from_cpu(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), DType::F32)
                .unwrap();
            let (out, handle) = dev
                .matmul(a.as_ref(), b.as_ref(), &Shape::new(vec![2, 2]))
                .unwrap();
            handle.synchronize().unwrap();
            assert_eq!(out.to_cpu_vec_f32().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);

            let (out_add, handle_add) = dev
                .add(a.as_ref(), b.as_ref(), &Shape::new(vec![4]))
                .unwrap();
            handle_add.synchronize().unwrap();
            assert_eq!(
                out_add.to_cpu_vec_f32().unwrap(),
                vec![6.0, 8.0, 10.0, 12.0]
            );
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn test_metal_mlx_bridge() {
        let dev = MetalDevice::try_new(0).unwrap();
        if dev.inner.is_some() {
            let storage = dev
                .from_cpu(&[1.0, 2.0], &Shape::new(vec![2]), DType::F32)
                .unwrap();
            let metal_storage = storage.as_any().downcast_ref::<MetalStorage>().unwrap();
            let bridge = MlxBridge::new();
            let raw_ptr = unsafe { bridge.to_mlx_array(metal_storage).unwrap() };
            assert!(!raw_ptr.is_null());
        }
    }

    // ===== Golden Mutation-Resistant Op Tests =====

    fn close(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-4,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    #[test]
    fn test_metal_add_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let a_data = vec![1.5f32, -2.5, 0.0, 3.14159];
        let b_data = vec![2.5f32, 3.5, -1.0, 1.0];
        let a = dev
            .from_cpu(&a_data, &Shape::new(vec![4]), DType::F32)
            .unwrap();
        let b = dev
            .from_cpu(&b_data, &Shape::new(vec![4]), DType::F32)
            .unwrap();
        let (out, handle) = dev
            .add(a.as_ref(), b.as_ref(), &Shape::new(vec![4]))
            .unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 4);
        close(res[0], 4.0, "add w0");
        close(res[1], 1.0, "add w1");
        close(res[2], -1.0, "add w2");
        close(res[3], 4.14159, "add w3");
    }

    #[test]
    fn test_metal_mul_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let a_data = vec![2.0f32, -3.0, 0.5];
        let b_data = vec![4.0f32, 2.0, -8.0];
        let a = dev
            .from_cpu(&a_data, &Shape::new(vec![3]), DType::F32)
            .unwrap();
        let b = dev
            .from_cpu(&b_data, &Shape::new(vec![3]), DType::F32)
            .unwrap();
        let (out, handle) = dev
            .mul(a.as_ref(), b.as_ref(), &Shape::new(vec![3]))
            .unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 3);
        close(res[0], 8.0, "mul w0");
        close(res[1], -6.0, "mul w1");
        close(res[2], -4.0, "mul w2");
    }

    #[test]
    fn test_metal_silu_mul_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let gate_data = vec![1.0f32, -1.0];
        let up_data = vec![2.0f32, 3.0];
        let gate = dev
            .from_cpu(&gate_data, &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let up = dev
            .from_cpu(&up_data, &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let (out, handle) = dev
            .silu_mul(gate.as_ref(), up.as_ref(), &Shape::new(vec![2]))
            .unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 2);

        let sig_1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
        let expected_0 = sig_1 * 1.0 * 2.0;

        let sig_neg1 = 1.0f32 / (1.0f32 + (1.0f32).exp());
        let expected_1 = (-1.0f32 * sig_neg1) * 3.0;

        close(res[0], expected_0, "silu_mul w0");
        close(res[1], expected_1, "silu_mul w1");
    }

    #[test]
    fn test_metal_rms_norm_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let x_data = vec![3.0f32, 4.0];
        let w_data = vec![1.0f32, 2.0];
        let shape = Shape::new(vec![2]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let w = dev.from_cpu(&w_data, &shape, DType::F32).unwrap();
        let (out, handle) = dev.rms_norm(x.as_ref(), w.as_ref(), 1e-6, &shape).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 2);

        let rms_val = (12.5f32 + 1e-6).sqrt();
        let expected_0 = (3.0 / rms_val) * 1.0;
        let expected_1 = (4.0 / rms_val) * 2.0;
        close(res[0], expected_0, "rms_norm w0");
        close(res[1], expected_1, "rms_norm w1");
    }

    #[test]
    fn test_metal_softmax_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let x_data = vec![1.0f32, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let (out, handle) = dev.softmax(x.as_ref(), &shape).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 3);

        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        close(res[0], 1.0f32.exp() / sum_exp, "softmax w0");
        close(res[1], 2.0f32.exp() / sum_exp, "softmax w1");
        close(res[2], 3.0f32.exp() / sum_exp, "softmax w2");
    }

    #[test]
    fn test_metal_embedding_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let table = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let weight = dev
            .from_cpu(&table, &Shape::new(vec![3, 2]), DType::F32)
            .unwrap();
        let indices = vec![2u32, 0];
        let out_shape = Shape::new(vec![2, 2]);
        let (out, handle) = dev
            .embedding(weight.as_ref(), &indices, &out_shape)
            .unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![50.0, 60.0, 10.0, 20.0]);
    }
}

pub fn vram_info(_ordinal: usize) -> Option<(u64, u64)> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        if let Some(dev) = MTLCreateSystemDefaultDevice() {
            let max_bytes = dev.recommendedMaxWorkingSetSize();
            let used_bytes = dev.currentAllocatedSize();
            let free_bytes = max_bytes.saturating_sub(used_bytes);
            return Some((free_bytes as u64, max_bytes as u64));
        }
    }
    None
}

impl grim_format::convert::GpuDequant for MetalDevice {
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
