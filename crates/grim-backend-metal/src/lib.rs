pub mod autotune;
pub mod caps;
pub mod kernels;

pub use autotune::{GemmOp, MetalAutotuner, MetalTileConfig, ShapeClass};
pub use caps::MetalCaps;

mod ops {
    pub mod core_tensor_ops;
    pub mod elementwise_ops;
    pub mod attention_ops;
    pub mod autograd_ops;
    pub mod optimizer_ops;
    pub mod quant_ops;
    pub mod recurrent_ops;
    pub mod collective_ops;
    pub mod memory_ops;
    pub mod device_ops;
    pub mod gpu_dequant;
}

#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::core_tensor_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::elementwise_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::attention_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::autograd_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::optimizer_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::quant_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::recurrent_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::collective_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::memory_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::device_ops::*;
#[allow(unused_imports)] // API surface: ops methods stay on MetalDevice
pub use ops::gpu_dequant::*;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
pub use grim_tensor::{
    ArithType, AttentionOps, AutogradOps, BackendDevice, BackendStorage, CollectiveOps,
    CoreTensorOps, ElementwiseOps, FusionOps, GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps,
    RecurrentOps, SamplingOps, ScythePlacement, Shape,
};

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

/// SIMDgroup GEMM dispatch gate (audit Metal-track).
/// Returns the chosen variant for an (m, n, k) GEMM, or `None` to use the.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(crate) fn simdgroup_gemm_variant(m: usize, n: usize, k: usize) -> Option<SimdgroupGemmVariant> {
    const MIN_DIM: usize = 64;
    if m % 8 != 0 || n % 8 != 0 || k % 8 != 0 {
        return None;
    }
    if m < MIN_DIM || n < MIN_DIM || k < MIN_DIM {
        return None;
    }
    // 16x16 threadgroups amortize scheduling best when the output is wide;
    // narrow-N GEMMs would leave quadrants idle.
    if n % 16 == 0 && m % 16 == 0 && n >= 128 && m >= 128 {
        return Some(SimdgroupGemmVariant::Tile16);
    }
    Some(SimdgroupGemmVariant::Tile8)
}

/// Which simdgroup kernel the gate selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(crate) enum SimdgroupGemmVariant {
    /// 32-row × 8-col blocks, one 8x8 MMA per simdgroup.
    Tile8,
    /// 16x16 output per threadgroup, one 8x8 quadrant per simdgroup.
    Tile16,
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
    rmsnorm_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rope_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    embedding_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    embedding_scatter_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
    fused_dequant_gemm_q4k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fused_dequant_gemm_fp8: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fused_dequant_gemm_mxfp4: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fused_rmsnorm_mxfp4_gemm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fused_rmsnorm_mxfp4_gemm_rope_kv: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_split_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    reduce_split_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qkv_paged_dequant_attn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    sub: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    reduce_sum: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    reduce_max: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    argmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose_2d: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_simdgroup_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_simdgroup_f32_16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    zeros_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fused_linear_ce: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fused_linear_ce_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    flash_decode_split_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax_merge: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
                fused_dequant_gemm_q4k: get_pipeline("grim_fused_dequant_gemm_q4k")?,
                fused_dequant_gemm_fp8: get_pipeline("grim_fused_dequant_gemm_fp8")?,
                fused_dequant_gemm_mxfp4: get_pipeline("grim_fused_dequant_gemm_mxfp4")?,
                fused_rmsnorm_mxfp4_gemm: get_pipeline("grim_fused_rmsnorm_mxfp4_gemm")?,
                fused_rmsnorm_mxfp4_gemm_rope_kv: get_pipeline("grim_fused_rmsnorm_mxfp4_gemm_rope_kv")?,
                matmul_split_k: get_pipeline("grim_matmul_split_k")?,
                reduce_split_k: get_pipeline("grim_reduce_split_k")?,
                qkv_paged_dequant_attn: get_pipeline("grim_qkv_attention_paged_dequant")?,
                sub: get_pipeline("grim_sub")?,
                reduce_sum: get_pipeline("grim_reduce_sum")?,
                reduce_max: get_pipeline("grim_reduce_max")?,
                argmax: get_pipeline("grim_argmax")?,
                transpose_2d: get_pipeline("grim_transpose_2d")?,
                matmul_simdgroup_f32: get_pipeline("grim_matmul_simdgroup_f32")?,
                matmul_simdgroup_f32_16: get_pipeline("grim_matmul_simdgroup_f32_16")?,
                zeros_f32: get_pipeline("grim_zeros_f32")?,
                rmsnorm_backward: get_pipeline("grim_rmsnorm_backward")?,
                rope_backward: get_pipeline("grim_rope_backward")?,
                softmax_backward: get_pipeline("grim_softmax_backward")?,
                embedding_backward: get_pipeline("grim_embedding_backward")?,
                fused_linear_ce: get_pipeline("grim_fused_linear_ce")?,
                fused_linear_ce_backward: get_pipeline("grim_fused_linear_ce_backward")?,
                flash_decode_split_k: get_pipeline("grim_flash_decode_split_k")?,
                softmax_merge: get_pipeline("grim_softmax_merge")?,
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
                let data_guard = data.lock().unwrap_or_else(|e| e.into_inner());
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
            let data_guard = self.data.lock().unwrap_or_else(|e| e.into_inner());
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

#[cfg(target_vendor = "apple")]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    ordinal: usize,
    pub caps: MetalCaps,
    pub autotuner: std::sync::Arc<MetalAutotuner>,
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
    pub caps: MetalCaps,
    pub autotuner: std::sync::Arc<MetalAutotuner>,
}

impl MetalDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            Self::try_new(ordinal)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Self {
                ordinal,
                caps: MetalCaps::probe_default(
                    ordinal as u64,
                    format!("Metal Device {ordinal}"),
                    7,
                ),
                autotuner: std::sync::Arc::new(MetalAutotuner::new()),
            })
        }
    }

    pub fn try_new(ordinal: usize) -> Result<Self> {
        let caps =
            MetalCaps::probe_default(ordinal as u64, format!("Apple Metal GPU {ordinal}"), 8);
        let autotuner = std::sync::Arc::new(MetalAutotuner::new());
        autotuner.load_cache(&caps);
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
                caps,
                autotuner,
                inner: Some(inner),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Self {
                ordinal,
                caps,
                autotuner,
            })
        }
    }

    pub fn caps(&self) -> &MetalCaps {
        &self.caps
    }

    pub fn hw_fingerprint(&self) -> u64 {
        self.caps.cache_key_hash()
    }

    pub fn save_autotune_cache(&self) {
        self.autotuner.save_cache(&self.caps);
    }

    #[cfg(target_vendor = "apple")]
    pub fn get_or_create_command_buffer(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| Error::from(MetalError::Context("Device inner is None".into())))?;
        let mut active = inner
            .active_command_buffer
            .lock()
            .map_err(|_| Error::Backend("Metal active_command_buffer mutex poisoned".into()))?;
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
                let mut active = inner
                    .active_command_buffer
                    .lock()
                    .map_err(|_| Error::Backend("Metal active_command_buffer mutex poisoned".into()))?;
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
    #[allow(clippy::type_complexity)]
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

    /// Fused LM-head + cross-entropy forward pass for Metal.
    /// Computes `logits = hidden @ lm_head^T` and cross-entropy loss + LSE in a single Metal.
    #[allow(clippy::type_complexity)]
    pub fn fused_linear_cross_entropy_forward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        _v_tile_size: i32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let h_s = hidden
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal hidden is not MetalStorage".into()))?;
                let w_s = lm_head
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal lm_head is not MetalStorage".into()))?;
                let t_s = targets
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal targets is not MetalStorage".into()))?;

                let h_buf = h_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("hidden lacks buffer".into()))?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("lm_head lacks buffer".into()))?;
                let t_buf = t_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("targets lacks buffer".into()))?;

                let hd = hidden.shape().dims();
                let wd = lm_head.shape().dims();
                let td = targets.shape().dims();
                if hd.len() != 2
                    || wd.len() != 2
                    || td.len() != 1
                    || td[0] != hd[0]
                    || wd[1] != hd[1]
                {
                    return Err(Error::Shape(
                        "fused_linear_ce: incompatible input shapes".into(),
                    ));
                }
                if v_tile_size <= 0 {
                    return Err(Error::Backend(
                        "fused_linear_ce: v_tile_size must be positive".into(),
                    ));
                }

                let batch = hd[0];
                let loss_storage = self.zeros(&Shape::new(vec![batch]), DType::F32)?;
                let lse_storage = self.zeros(&Shape::new(vec![batch]), DType::F32)?;
                let loss_s = loss_storage
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .unwrap();
                let lse_s = lse_storage.as_any().downcast_ref::<MetalStorage>().unwrap();

                let loss_buf = loss_s.buffer.as_ref().unwrap();
                let lse_buf = lse_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.fused_linear_ce);
                encoder.setBuffer_offset_atIndex(Some(h_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(t_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(loss_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(lse_buf), 0, 4);

                let hidden_dim_val = hd[1] as i32;
                let vocab_size_val = wd[0] as i32;
                let batch_val = batch as i32;
                let v_tile_val = v_tile_size as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &batch_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                    encoder.setBytes_length_atIndex(
                        &hidden_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &vocab_size_val as *const i32 as *const std::ffi::c_void,
                        4,
                        7,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(
                    (batch.max(1) as u64 + threads_per_group.width - 1) / threads_per_group.width,
                    1,
                    1,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                return Ok((
                    loss_storage,
                    lse_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ));
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            let cpu = CpuDevice::new();
            let hidden_cpu = hidden.to_cpu_vec_f32()?;
            let lm_head_cpu = lm_head.to_cpu_vec_f32()?;
            let targets_cpu = targets.to_cpu_vec_f32()?;
            let hd = hidden.shape().dims();
            let wd = lm_head.shape().dims();
            let batch = hd[0];
            let hidden_dim = hd[1];
            let vocab_size = wd[0];
            let mut loss = vec![0.0f32; batch];
            let mut lse = vec![0.0f32; batch];
            for b in 0..batch {
                let target = targets_cpu[b] as usize;
                let h_base = b * hidden_dim;
                // compute logits row and run CE + LSE
                let mut max_val = f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;
                let mut target_logit = 0.0f32;
                for v in 0..vocab_size {
                    let mut logit = 0.0f32;
                    for d in 0..hidden_dim {
                        logit += hidden_cpu[h_base + d] * lm_head_cpu[v * hidden_dim + d];
                    }
                    if v == target {
                        target_logit = logit;
                    }
                    if logit > max_val {
                        sum_exp = sum_exp * max_val.exp() + 1.0;
                        max_val = logit;
                    } else {
                        sum_exp += (logit - max_val).exp();
                    }
                }
                let lse_val = max_val + sum_exp.ln();
                loss[b] = lse_val - target_logit;
                lse[b] = lse_val;
            }
            let loss_storage = cpu.from_cpu(&loss, &Shape::new(vec![batch]), DType::F32)?;
            let lse_storage = cpu.from_cpu(&lse, &Shape::new(vec![batch]), DType::F32)?;
            Ok((loss_storage, lse_storage, Box::new(MetalHandle)))
        }
    }

    /// Fused LM-head + cross-entropy backward pass for Metal.
    /// Computes `grad_h = d(logits)/d(hidden)` using the LSE and targets from the forward pass.
    pub fn fused_linear_cross_entropy_backward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        lse: &dyn BackendStorage,
        _grad_h: &dyn BackendStorage,
        _v_tile_size: i32,
    ) -> Result<Box<dyn ComputeHandle>> {
        #[cfg(target_vendor = "apple")]
        {
            let grad_h = _grad_h;
            if let Some(ref inner) = self.inner {
                let h_s = hidden
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal hidden is not MetalStorage".into()))?;
                let w_s = lm_head
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal lm_head is not MetalStorage".into()))?;
                let t_s = targets
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal targets is not MetalStorage".into()))?;
                let l_s = lse
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal lse is not MetalStorage".into()))?;
                let g_s = grad_h
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("Metal grad_h is not MetalStorage".into()))?;

                let h_buf = h_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("hidden lacks buffer".into()))?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("lm_head lacks buffer".into()))?;
                let t_buf = t_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("targets lacks buffer".into()))?;
                let lse_buf = l_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("lse lacks buffer".into()))?;
                let grad_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("grad_h lacks buffer".into()))?;

                let hd = hidden.shape().dims();
                let wd = lm_head.shape().dims();
                let td = targets.shape().dims();
                let ld = lse.shape().dims();

                if hd.len() != 2
                    || wd.len() != 2
                    || td.len() != 1
                    || ld.len() != 1
                    || td[0] != hd[0]
                    || wd[1] != hd[1]
                    || ld[0] != hd[0]
                {
                    return Err(Error::Shape(
                        "fused_linear_ce_backward: incompatible input shapes".into(),
                    ));
                }
                let _v_tile_size = v_tile_size;

                let batch = hd[0];
                let hidden_dim_val = hd[1] as i32;
                let vocab_size_val = wd[0] as i32;
                let batch_val = batch as i32;
                let v_tile_val = v_tile_size as i32;
                let inv_batch_val = (1.0f32 / batch_val as f32) as f32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.fused_linear_ce_backward);
                encoder.setBuffer_offset_atIndex(Some(h_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(t_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(lse_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(grad_buf), 0, 4);

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &hidden_dim_val as *const i32 as *const std::ffi::c_void,
                        5,
                        5,
                    );
                    encoder.setBytes_length_atIndex(
                        &vocab_size_val as *const i32 as *const std::ffi::c_void,
                        5,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &v_tile_val as *const i32 as *const std::ffi::c_void,
                        5,
                        7,
                    );
                    encoder.setBytes_length_atIndex(
                        &inv_batch_val as *const f32 as *const std::ffi::c_void,
                        5,
                        8,
                    );
                    encoder.setBytes_length_atIndex(
                        &batch_val as *const i32 as *const std::ffi::c_void,
                        5,
                        9,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(
                    (batch.max(1) as u64 + threads_per_group.width - 1) / threads_per_group.width,
                    1,
                    1,
                );
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                return Ok(Box::new(MetalHandle {
                    command_buffer: cmd_buffer,
                }));
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            let cpu = CpuDevice::new();
            let hidden_cpu = hidden.to_cpu_vec_f32()?;
            let lm_head_cpu = lm_head.to_cpu_vec_f32()?;
            let targets_cpu = targets.to_cpu_vec_f32()?;
            let lse_cpu = lse.to_cpu_vec_f32()?;
            let hd = hidden.shape().dims();
            let wd = lm_head.shape().dims();
            let batch = hd[0];
            let hidden_dim = hd[1];
            let vocab_size = wd[0];
            let mut grad = vec![0.0f32; batch * hidden_dim];
            let inv_batch = 1.0f32 / batch as f32;
            for b in 0..batch {
                let target = targets_cpu[b] as usize;
                let _lse_row = lse_cpu[b];
                let h_base = b * hidden_dim;
                for d in 0..hidden_dim {
                    grad[h_base + d] = 0.0f32;
                }
                for v in 0..vocab_size {
                    let mut logit = 0.0f32;
                    let w_base = v * hidden_dim;
                    for d in 0..hidden_dim {
                        logit += hidden_cpu[h_base + d] * lm_head_cpu[w_base + d];
                    }
                    let dl = (logit.exp() - if v == target { 1.0f32 } else { 0.0f32 }) * inv_batch;
                    for d in 0..hidden_dim {
                        grad[h_base + d] += dl * lm_head_cpu[w_base + d];
                    }
                }
            }
            let _grad_storage =
                cpu.from_cpu(&grad, &Shape::new(vec![batch, hidden_dim]), DType::F32)?;
            Ok(Box::new(MetalHandle))
        }
    }

    /// Fused RMSNorm + MXFP4 GEMM. Mirrors ROCm's `fused_rmsnorm_mxfp4_gemm`: x @ MXFP4(W) with fused RMSNorm.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_rmsnorm_mxfp4_gemm(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        w_packed: &dyn BackendStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("fused_rmsnorm_mxfp4_gemm: x not MetalStorage".into())
            })?;
            let gamma_s = gamma
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend("fused_rmsnorm_mxfp4_gemm: gamma not MetalStorage".into())
                })?;
            let w_s = w_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend("fused_rmsnorm_mxfp4_gemm: w_packed not MetalStorage".into())
                })?;
            let x_buf = x_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("x no buffer".into()))?;
            let gamma_buf = gamma_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("gamma no buffer".into()))?;
            let w_buf = w_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("w_packed no buffer".into()))?;
            let out_storage = self.zeros(&Shape::new(vec![m, n]), DType::F32)?;
            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
            let out_buf = out_s.buffer.as_ref().unwrap();
            let ctx = MetalContext::get()?;
            let cmd = self.get_or_create_command_buffer()?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| Error::from(MetalError::Ffi("compute encoder".into())))?;
            enc.setComputePipelineState(&ctx.pipelines.fused_rmsnorm_mxfp4_gemm);
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(w_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 3);
            let m_val = m as i32;
            let n_val = n as i32;
            let k_val = k as i32;
            unsafe {
                enc.setBytes_length_atIndex(&m_val as *const i32 as *const std::ffi::c_void, 4, 4);
                enc.setBytes_length_atIndex(&n_val as *const i32 as *const std::ffi::c_void, 4, 5);
                enc.setBytes_length_atIndex(&k_val as *const i32 as *const std::ffi::c_void, 4, 6);
                enc.setBytes(&eps as *const f32 as *const std::ffi::c_void, 4, 7);
            }
            let tpg = MTLSize::new(16, 16, 1);
            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            return Ok((
                out_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let cpu = CpuDevice::new();
            let x_cpu = x.to_cpu_vec_f32()?;
            let gamma_cpu = gamma.to_cpu_vec_f32()?;
            // w_packed is MXFP4: raw u8 bytes (codes + shared exps), not f32.
            // Downcast to MetalStorage and read the data Mutex directly.
            let w_s = w_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend(
                        "fused_rmsnorm_mxfp4_gemm CPU fallback: w_packed not MetalStorage".into(),
                    )
                })?;
            let w_data = w_s.data.lock().unwrap_or_else(|e| e.into_inner());
            let w_bytes: &[u8] = w_data.as_slice();
            let codes_bytes = k.div_ceil(2);
            let exps_bytes = k / 32;
            let row_bytes = codes_bytes + exps_bytes;
            let eps = eps.max(1e-5f32);
            let mut out = vec![0.0f32; m * n];
            for row in 0..m {
                let x_row = &x_cpu[row * k..(row + 1) * k];
                let mut mean = 0.0f32;
                for &v in x_row {
                    mean += v;
                }
                mean /= k as f32;
                let mut sum_sq = 0.0f32;
                for &v in x_row {
                    let d = v - mean;
                    sum_sq += d * d;
                }
                let inv_rms = 1.0f32 / (sum_sq / k as f32 + eps).sqrt();
                let _w_row = &w_bytes[row * row_bytes..(row + 1) * row_bytes];
                for col in 0..n {
                    let w_col = &w_bytes[col * row_bytes..(col + 1) * row_bytes];
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        let byte_idx = i / 2;
                        let packed = w_col[byte_idx];
                        let nib = if (i % 2) == 0 {
                            packed & 0x0F
                        } else {
                            packed >> 4
                        };
                        let shared_exp = w_col[codes_bytes + i / 32];
                        // Replicate metal_mxfp4_to_float in host-side fallback:
                        // MXFP4 = 4-bit mantissa (bits 3:0), shared exponent from exps table.
                        let mant = (nib & 0x0F) as f32 / 15.0f32;
                        let exp_delta = (shared_exp as i32) - 124; // E4M3 bias 124 → exponent補
                        let w = (if (nib & 0x08) != 0 { -mant } else { mant })
                            * (1.0f32 + mant)
                            * (1u32 << (exp_delta.max(0) as u32)) as f32;
                        acc += (x_row[i] * inv_rms * gamma_cpu[i]) * w;
                    }
                    out[row * n + col] = acc;
                }
            }
            let os = cpu.from_cpu(&out, &Shape::new(vec![m, n]), DType::F32)?;
            Ok((os, Box::new(MetalHandle)))
        }
    }

    /// Fused RMSNorm + MXFP4 GEMM + RoPE + KV cache scatter.
    /// Mirrors ROCm's `fused_rmsnorm_mxfp4_gemm_rope_kv`.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        wq_packed: &dyn BackendStorage,
        wk_packed: &dyn BackendStorage,
        wv_packed: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        q_dim: usize,
        kv_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        #[cfg(target_vendor = "apple")]
        {
            let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("fused_rmsnorm_mxfp4_gemm_rope_kv: x not MetalStorage".into())
            })?;
            let gamma_s = gamma
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend(
                        "fused_rmsnorm_mxfp4_gemm_rope_kv: gamma not MetalStorage".into(),
                    )
                })?;
            let wq_s = wq_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend(
                        "fused_rmsnorm_mxfp4_gemm_rope_kv: wq_packed not MetalStorage".into(),
                    )
                })?;
            let wk_s = wk_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend(
                        "fused_rmsnorm_mxfp4_gemm_rope_kv: wk_packed not MetalStorage".into(),
                    )
                })?;
            let wv_s = wv_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend(
                        "fused_rmsnorm_mxfp4_gemm_rope_kv: wv_packed not MetalStorage".into(),
                    )
                })?;
            let x_buf = x_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("x no buffer".into()))?;
            let gamma_buf = gamma_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("gamma no buffer".into()))?;
            let wq_buf = wq_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("wq_packed no buffer".into()))?;
            let wk_buf = wk_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("wk_packed no buffer".into()))?;
            let wv_buf = wv_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("wv_packed no buffer".into()))?;
            let q_out_buf = q_out
                .and_then(|o| o.as_any().downcast_ref::<MetalStorage>()?.buffer.as_ref())
                .ok_or_else(|| Error::Backend("q_out no buffer".into()))?;
            let k_cache_buf = k_cache
                .and_then(|o| o.as_any().downcast_ref::<MetalStorage>()?.buffer.as_ref())
                .ok_or_else(|| Error::Backend("k_cache no buffer".into()))?;
            let v_cache_buf = v_cache
                .and_then(|o| o.as_any().downcast_ref::<MetalStorage>()?.buffer.as_ref())
                .ok_or_else(|| Error::Backend("v_cache no buffer".into()))?;
            let pos_buf = positions
                .and_then(|o| o.as_any().downcast_ref::<MetalStorage>()?.buffer.as_ref())
                .ok_or_else(|| Error::Backend("positions no buffer".into()))?;
            let ctx = MetalContext::get()?;
            let cmd = self.get_or_create_command_buffer()?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| Error::from(MetalError::Ffi("compute encoder".into())))?;
            enc.setComputePipelineState(&ctx.pipelines.fused_rmsnorm_mxfp4_gemm_rope_kv);
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(wq_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(wk_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(wv_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(q_out_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(k_cache_buf), 0, 6);
            enc.setBuffer_offset_atIndex(Some(v_cache_buf), 0, 7);
            enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 8);
            let m_val = m as i32;
            let k_val = k as i32;
            let q_dim_val = q_dim as i32;
            let kv_dim_val = kv_dim as i32;
            let rotary_dim_val = rotary_dim as i32;
            let rope_theta_val = rope_theta;
            let num_kv_heads_val = num_kv_heads as i32;
            let head_dim_val = head_dim as i32;
            unsafe {
                enc.setBytes_length_atIndex(&m_val as *const i32 as *const std::ffi::c_void, 4, 9);
                enc.setBytes_length_atIndex(&k_val as *const i32 as *const std::ffi::c_void, 4, 10);
                enc.setBytes_length_atIndex(
                    &q_dim_val as *const i32 as *const std::ffi::c_void,
                    4,
                    11,
                );
                enc.setBytes_length_atIndex(
                    &kv_dim_val as *const i32 as *const std::ffi::c_void,
                    4,
                    12,
                );
                enc.setBytes_length_atIndex(
                    &rotary_dim_val as *const i32 as *const std::ffi::c_void,
                    4,
                    13,
                );
                enc.setBytes(
                    &rope_theta_val as *const f32 as *const std::ffi::c_void,
                    4,
                    14,
                );
                enc.setBytes_length_atIndex(
                    &num_kv_heads_val as *const i32 as *const std::ffi::c_void,
                    4,
                    15,
                );
                enc.setBytes_length_atIndex(
                    &head_dim_val as *const i32 as *const std::ffi::c_void,
                    4,
                    16,
                );
            }
            let out_dim = q_dim.max(kv_dim);
            let tpg = MTLSize::new(16, 16, 1);
            let groups = MTLSize::new(((out_dim + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            return Ok(Box::new(MetalHandle {
                command_buffer: cmd,
            }));
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (
                x,
                gamma,
                wq_packed,
                wk_packed,
                wv_packed,
                q_out,
                k_cache,
                v_cache,
                positions,
                m,
                k,
                q_dim,
                kv_dim,
                rotary_dim,
                rope_theta,
                num_kv_heads,
                head_dim,
            );
            Err(Error::Backend(
                "fused_rmsnorm_mxfp4_gemm_rope_kv: CPU fallback not implemented".into(),
            ))
        }
    }

    /// Flash-decode (split-KV parallel attention) for Metal.
    /// Wraps `grim_flash_decode_split_k` + `grim_softmax_merge` kernels.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn flash_decode(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        num_splits: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let q_s = q
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("flash_decode: q not MetalStorage".into()))?;
                let k_s = k
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("flash_decode: k not MetalStorage".into()))?;
                let v_s = v
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("flash_decode: v not MetalStorage".into()))?;

                let q_buf = q_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("q lacks buffer".into()))?;
                let k_buf = k_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k lacks buffer".into()))?;
                let v_buf = v_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v lacks buffer".into()))?;

                let ns = num_splits.max(1);
                let mid_out = self.zeros(&Shape::new(vec![ns, num_heads, head_dim]), DType::F32)?;
                let mid_max = self.zeros(&Shape::new(vec![ns, num_heads]), DType::F32)?;
                let mid_sum = self.zeros(&Shape::new(vec![ns, num_heads]), DType::F32)?;

                let mout_s = mid_out.as_any().downcast_ref::<MetalStorage>().unwrap();
                let mmax_s = mid_max.as_any().downcast_ref::<MetalStorage>().unwrap();
                let msum_s = mid_sum.as_any().downcast_ref::<MetalStorage>().unwrap();

                let mout_buf = mout_s.buffer.as_ref().unwrap();
                let mmax_buf = mmax_s.buffer.as_ref().unwrap();
                let msum_buf = msum_s.buffer.as_ref().unwrap();

                let cmd = self.get_or_create_command_buffer()?;
                let enc = cmd
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::from(MetalError::Ffi("compute encoder".into())))?;

                enc.setComputePipelineState(&inner.pipelines.flash_decode_split_k);
                enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(mout_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(mmax_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(msum_buf), 0, 5);

                let scale = (1.0f32 / (head_dim.max(1) as f32).sqrt()) as f32;
                unsafe {
                    enc.setBytes_length_atIndex(
                        &(num_heads as i32) as *const i32 as *const std::ffi::c_void,
                        6,
                        6,
                    );
                    enc.setBytes_length_atIndex(
                        &(num_kv_heads as i32) as *const i32 as *const std::ffi::c_void,
                        6,
                        7,
                    );
                    enc.setBytes_length_atIndex(
                        &(head_dim as i32) as *const i32 as *const std::ffi::c_void,
                        6,
                        8,
                    );
                    enc.setBytes_length_atIndex(
                        &(kv_seq_len as i32) as *const i32 as *const std::ffi::c_void,
                        6,
                        9,
                    );
                    enc.setBytes_length_atIndex(
                        &(ns as i32) as *const i32 as *const std::ffi::c_void,
                        6,
                        10,
                    );
                    enc.setBytes_length_atIndex(
                        &scale as *const f32 as *const std::ffi::c_void,
                        6,
                        11,
                    );
                }

                let tpg = MTLSize::new(256, 1, 1);
                let gr = MTLSize::new((num_heads.max(1) as u64) * (ns as u64), 1, 1);
                enc.dispatchThreadgroups_threadsPerThreadgroup(gr, tpg);

                // Stage 2: softmax merge
                enc.setComputePipelineState(&inner.pipelines.softmax_merge);
                enc.setBuffer_offset_atIndex(Some(mout_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(mmax_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(msum_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    &(num_heads as i32) as *const i32 as *const std::ffi::c_void,
                    4,
                    4,
                );
                enc.setBytes_length_atIndex(
                    &(ns as i32) as *const i32 as *const std::ffi::c_void,
                    4,
                    5,
                );
                enc.setBytes_length_atIndex(
                    &(head_dim as i32) as *const i32 as *const std::ffi::c_void,
                    4,
                    6,
                );
                let gr2 = MTLSize::new(num_heads.max(1) as u64, 1, 1);
                enc.dispatchThreadgroups_threadsPerThreadgroup(gr2, tpg);
                enc.endEncoding();

                cmd.commit();
                cmd.waitUntilCompleted();
                return Ok((
                    mid_out,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            let cpu = CpuDevice::new();
            let qv = q.to_cpu_vec_f32()?;
            let kv = k.to_cpu_vec_f32()?;
            let vv = v.to_cpu_vec_f32()?;

            let nh = num_heads.max(1);
            let nkh = num_kv_heads.max(1);
            let hd = head_dim.max(1);
            let sl = kv_seq_len.max(1);
            let ns = num_splits.max(1);
            let cl = sl.div_ceil(ns);
            let scale = 1.0f32 / (hd as f32).sqrt();

            let mut mo = vec![0.0f32; ns * nh * hd];
            let mut mm = vec![-1e20f32; ns * nh];
            let mut ms = vec![0.0f32; ns * nh];

            for h in 0..nh {
                let kh = h / nkh.max(1);
                for s in 0..ns {
                    let st = s * cl;
                    let en = (st + cl).min(sl);
                    let ll = en - st;
                    let split_off = h * ns + s;
                    let mut sf = vec![0.0f32; ll];
                    let mut mx = -1e20f32;
                    for i in 0..ll {
                        let pos = st + i;
                        let kbase = (pos * nkh + kh) * hd;
                        let mut d = 0.0f32;
                        for dd in 0..hd {
                            d += qv[h * hd + dd] * kv[kbase + dd];
                        }
                        sf[i] = d * scale;
                        if sf[i] > mx {
                            mx = sf[i];
                        }
                    }
                    let mut se = 0.0f32;
                    for i in 0..ll {
                        sf[i] = (sf[i] - mx).exp();
                        se += sf[i];
                    }
                    mm[split_off] = mx;
                    ms[split_off] = se;
                    let ob = split_off * hd;
                    for dd in 0..hd {
                        let mut a = 0.0f32;
                        for i in 0..ll {
                            let pos = st + i;
                            let vbase = (pos * nkh + kh) * hd;
                            a += sf[i] * vv[vbase + dd];
                        }
                        mo[ob + dd] = a;
                    }
                }
            }

            let mut out = vec![0.0f32; nh * hd];
            for h in 0..nh {
                let gm: f32 = (0..ns).map(|s| mm[h * ns + s]).fold(-1e20, f32::max);
                let gs = (0..ns)
                    .map(|s| (mm[h * ns + s] - gm).exp() * ms[h * ns + s])
                    .sum::<f32>();
                let inv = 1.0f32 / gs.max(1e-38f32);
                for dd in 0..hd {
                    let mut a = 0.0f32;
                    for s in 0..ns {
                        let w = (mm[h * ns + s] - gm).exp() * ms[h * ns + s] * inv;
                        a += w * mo[(h * ns + s) * hd + dd];
                    }
                    out[h * hd + dd] = a;
                }
            }

            let os = cpu.from_cpu(&out, &Shape::new(vec![nh, hd]), DType::F32)?;
            Ok((os, Box::new(MetalHandle)))
        }
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
                    other => {
                        return Err(Error::Backend(format!(
                            "Metal quantize_on_device: unsupported format for dispatch {:?}",
                            other
                        )));
                    }
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

    /// Fused grouped MoE dispatch (WI-M5). Mirrors `grim_moe_fused_dispatch` on ROCm and `moe_fused_dispatch` on Vulkan.
    #[allow(unused_variables)]
    #[allow(clippy::too_many_arguments)]
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
                    Some(x_s),
                    Some(gw_s),
                    Some(uw_s),
                    Some(dw_s),
                    Some(rt_s),
                    Some(re_s),
                    Some(rw_s),
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
                        x_s.buffer.as_ref(),
                        gw_s.buffer.as_ref(),
                        uw_s.buffer.as_ref(),
                        dw_s.buffer.as_ref(),
                        rt_s.buffer.as_ref(),
                        re_s.buffer.as_ref(),
                        rw_s.buffer.as_ref(),
                    ];
                    if bufs.iter().all(|b| b.is_some()) {
                        if let Ok(out_storage) = self.zeros(out_shape, DType::F32) {
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd = self.get_or_create_command_buffer()?;
                            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
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
                                encoder.setBytes_length_atIndex(
                                    &hidden_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    8,
                                );
                                encoder.setBytes_length_atIndex(
                                    &inter_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    9,
                                );
                                encoder.setBytes_length_atIndex(
                                    &num_experts_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    10,
                                );
                                encoder.setBytes_length_atIndex(
                                    &batch_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    11,
                                );
                                encoder.setBytes_length_atIndex(
                                    &rsf_val as *const f32 as *const std::ffi::c_void,
                                    4,
                                    12,
                                );
                                encoder.setBytes_length_atIndex(
                                    &num_pairs_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    13,
                                );
                            }

                            let grid = MTLSize::new(batch as u64, hidden as u64, 1);
                            let threads = MTLSize::new(1, 1, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
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

    pub fn matmul_with_op(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
        op: GemmOp,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_with_op_internal(a, b, out, Some(op))
    }

    /// Convenience alias for the LM-head projection GEMM.
    /// Mirrors ROCm's `matmul_lm_head` (roc_device.rs:13879) - routes through `matmul_with_op(GemmOp::LmHead)`, which selects the `TLOLog` shape class in.
    pub fn matmul_lm_head(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_with_op(a, b, out_shape, GemmOp::LmHead)
    }

    pub fn matmul_with_op_internal(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
        _op: Option<GemmOp>,
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
                // SPEED-ROC-16: `b` is the natural weight (N, K); matmul computes C = A @ B^T.
                let n = dims_b[0];
                let mut c_vec = vec![0.0f32; m * n];
                unsafe {
                    // RowMajor C[M,N] = A[M,K] * B[N,K]^T. trans_b=Trans (112); ldb=K.
                    cblas_sgemm(
                        101, // RowMajor
                        111, // NoTrans (A)
                        112, // Trans (B)
                        m as i32,
                        n as i32,
                        k as i32,
                        1.0,
                        a_vec.as_ptr(),
                        k as i32,
                        b_vec.as_ptr(),
                        k as i32,
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
                // SPEED-ROC-16: `b` is the natural weight (N, K); matmul computes C = A @ B^T.
                let (n, k2) = (b_dims[0], b_dims[1]);
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

                // SIMDgroup dispatch gate (audit Metal-track): eligible f32 GEMMs route to
                // the hardware-MMA kernels; everything else keeps the autotuned naive path below.
                if inner.caps.supports_simdgroup_matrix {
                    if let Some(variant) = simdgroup_gemm_variant(m, n, k) {
                        let pipeline = match variant {
                            SimdgroupGemmVariant::Tile8 => &inner.pipelines.matmul_simdgroup_f32,
                            SimdgroupGemmVariant::Tile16 => {
                                &inner.pipelines.matmul_simdgroup_f32_16
                            }
                        };
                        encoder.setComputePipelineState(pipeline);
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
                        let (groups, threads_per_group) = match variant {
                            // 4 simdgroups × 32 lanes; each threadgroup owns a
                            // 32(row) × 8(col) output block.
                            SimdgroupGemmVariant::Tile8 => (
                                MTLSize::new(((n + 7) / 8) as u64, ((m + 31) / 32) as u64, 1),
                                MTLSize::new(128, 1, 1),
                            ),
                            // 4 simdgroups; each threadgroup owns a 16×16 block.
                            SimdgroupGemmVariant::Tile16 => (
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1),
                                MTLSize::new(128, 1, 1),
                            ),
                        };
                        encoder
                            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                        encoder.endEncoding();
                        return Ok((
                            out_storage,
                            Box::new(MetalHandle {
                                command_buffer: cmd_buffer,
                            }),
                        ));
                    }
                }

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

                let config = self.autotuner.search_tile_config_measured(
                    &self.caps,
                    m,
                    n,
                    k,
                    _op,
                    Some(|cfg: &MetalTileConfig| measure_pipeline_timing(inner, m, n, k, cfg)),
                );
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
}



impl SamplingOps for MetalDevice {}


impl FusionOps for MetalDevice {}







impl GraphCaptureOps for MetalDevice {}

impl grim_tensor::BackendDevice for MetalDevice {}


#[cfg(target_vendor = "apple")]
pub(crate) fn measure_pipeline_timing(
    inner: &MetalDeviceInner,
    m: usize,
    n: usize,
    k: usize,
    cfg: &MetalTileConfig,
) -> Option<f64> {
    use objc2_metal::MTResourceOptions;
    let bytes_a = m * k * 4;
    let bytes_b = k * n * 4;
    let bytes_c = m * n * 4;
    let buf_a = inner
        .device
        .newBufferWithLength_options(bytes_a as u64, MTResourceOptions::StorageModeShared)?;
    let buf_b = inner
        .device
        .newBufferWithLength_options(bytes_b as u64, MTResourceOptions::StorageModeShared)?;
    let buf_c = inner
        .device
        .newBufferWithLength_options(bytes_c as u64, MTResourceOptions::StorageModeShared)?;

    let config_data = [cfg.block_m as i32, cfg.block_n as i32, cfg.block_k as i32];

    // Warm-up passes
    for _ in 0..2 {
        let cmd = inner.command_queue.commandBuffer()?;
        let enc = cmd.computeCommandEncoder()?;
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
        let threads = MTLSize::new(cfg.block_n as u64, cfg.block_m as u64, 1);
        let groups = MTLSize::new(
            ((n + (cfg.block_n as usize) - 1) / (cfg.block_n as usize)) as u64,
            ((m + (cfg.block_m as usize) - 1) / (cfg.block_m as usize)) as u64,
            1,
        );
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    // Timed passes
    let start = std::time::Instant::now();
    let iters = 5;
    for _ in 0..iters {
        let cmd = inner.command_queue.commandBuffer()?;
        let enc = cmd.computeCommandEncoder()?;
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
        let threads = MTLSize::new(cfg.block_n as u64, cfg.block_m as u64, 1);
        let groups = MTLSize::new(
            ((n + (cfg.block_n as usize) - 1) / (cfg.block_n as usize)) as u64,
            ((m + (cfg.block_m as usize) - 1) / (cfg.block_m as usize)) as u64,
            1,
        );
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let elapsed = start.elapsed() / iters;
    Some(elapsed.as_secs_f64() * 1000.0)
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

impl Default for MlxBridge {
    fn default() -> Self {
        Self
    }
}

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

/// WI-1: live compute utilization for `ordinal`.
/// Scope note (per WI-1): Metal has no cross-vendor utilization API.
pub fn compute_utilization(_ordinal: usize) -> Option<u32> {
    None
}


#[cfg(test)]
mod tests {
    fn close(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-4,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

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
        // Under SPEED-ROC-16: matmul computes C = A @ B^T.
        // [1, 2; 3, 4] @ [5, 6; 7, 8]^T = [1*5+2*6, 1*7+2*8; 3*5+4*6, 3*7+4*8] = [17, 23, 39, 53]
        assert_eq!(res, vec![17.0, 23.0, 39.0, 53.0]);
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

    #[test]
    fn test_metal_add_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let a_data = vec![1.5f32, -2.5, 0.0, std::f32::consts::PI];
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
        let expected_1 = (-sig_neg1) * 3.0;

        close(res[0], expected_0, "silu_mul w0");
        close(res[1], expected_1, "silu_mul w1");
    }

    #[test]
    fn test_metal_rms_norm_golden_exact() {
        let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
        let x_data = vec![3.0f32, 4.0];
        let w_data = vec![1.0f32, 2.0];
        // `rms_norm`'s contract takes a rank-2 (rows, dim) out_shape; the
        // weight stays 1-D.
        let shape = Shape::new(vec![1, 2]);
        let w_shape = Shape::new(vec![2]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let w = dev.from_cpu(&w_data, &w_shape, DType::F32).unwrap();
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

#[cfg(test)]
mod simdgroup_gate_tests {
    use super::*;

    /// The simdgroup kernels use unclipped 8x8 tile loads: any dimension not
    /// a multiple of 8 must fall back to the autotuned naive kernel.
    #[test]
    fn non_multiple_of_eight_falls_back() {
        assert_eq!(simdgroup_gemm_variant(63, 128, 128), None);
        assert_eq!(simdgroup_gemm_variant(128, 127, 128), None);
        assert_eq!(
            simdgroup_gemm_variant(128, 128, 8),
            None,
            "k=8 is a multiple but below threshold? no — k=8 < 64"
        );
        assert_eq!(simdgroup_gemm_variant(128, 128, 63), None);
    }

    /// Below-threshold GEMMs pay more in scheduling than the MMA saves.
    #[test]
    fn small_gemm_falls_back() {
        assert_eq!(simdgroup_gemm_variant(8, 8, 8), None);
        assert_eq!(simdgroup_gemm_variant(32, 32, 32), None);
        assert_eq!(simdgroup_gemm_variant(64, 64, 56), None);
    }

    /// Eligible shapes select the expected variant: Tile8 by default,
    /// Tile16 for wide-square outputs where quadrants stay populated.
    #[test]
    fn eligible_shapes_select_variant() {
        assert_eq!(
            simdgroup_gemm_variant(64, 64, 64),
            Some(SimdgroupGemmVariant::Tile8)
        );
        assert_eq!(
            simdgroup_gemm_variant(4096, 4096, 4096),
            Some(SimdgroupGemmVariant::Tile16)
        );
        // Wide-N but narrow-M: Tile8 (Tile16 quadrants would idle rows).
        assert_eq!(
            simdgroup_gemm_variant(64, 512, 128),
            Some(SimdgroupGemmVariant::Tile8)
        );
        // n % 16 != 0 forces Tile8 even at large sizes.
        assert_eq!(
            simdgroup_gemm_variant(256, 4104, 256),
            Some(SimdgroupGemmVariant::Tile8)
        );
    }
}

/// Apple-hardware parity gate: the simdgroup path must agree with the Accelerate/CPU fallback within f32 accumulation tolerance for an eligible (m,n,k).
/// Runs wherever Metal hardware exists; a no-op elsewhere (the dispatch gate keeps non-eligible shapes off.
#[cfg(all(test, target_vendor = "apple"))]
mod simdgroup_parity_tests {
    use super::*;

    #[test]
    fn simdgroup_matmul_matches_naive_path() {
        let dev = MetalDevice::new(0);
        if dev.inner.is_none() {
            return; // no Metal device attached
        }
        let (m, k, n) = (128usize, 128usize, 128usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.05 - 0.4).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
        let dtype = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        };
        let a = dev
            .from_cpu(&a_data, &Shape::new(vec![m, k]), dtype.clone())
            .unwrap();
        let b = dev
            .from_cpu(&b_data, &Shape::new(vec![k, n]), dtype.clone())
            .unwrap();
        assert!(simdgroup_gemm_variant(m, n, k).is_some());

        let (out, handle) =
            CoreTensorOps::matmul(&dev, a.as_ref(), b.as_ref(), &Shape::new(vec![m, n])).unwrap();
        handle.synchronize().unwrap();
        let got = out.to_cpu_vec_f32().unwrap();

        // Independent host reference (single-precision, same accumulation
        // order class — tolerance covers fma-vs-separate rounding).
        for i in 0..m {
            for j in 0..n {
                let mut want = 0.0f32;
                for kk in 0..k {
                    want += a_data[i * k + kk] * b_data[kk * n + j];
                }
                assert!(
                    (got[i * n + j] - want).abs() < 1e-2 * want.abs().max(1.0),
                    "[{i},{j}] got {} want {want}",
                    got[i * n + j]
                );
            }
        }
    }
}
