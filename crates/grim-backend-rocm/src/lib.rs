//! ROCm backend for Grim  primary GPU target per architecture §4.
//!
//! Replicates core architectural design concepts from the `rocm-rs` library ecosystem:
//! - Safe RAII allocation handles (Drop-on-scope, zero leaks) mimicking `DeviceMemoryExt`.
//! - Modular FFI layer designed for drop-in bindings to AMD's rocBLAS and HIP runtime.
//! - Explicit attribute-probing correctness gates mapping device traits.
//!
//! This crate provides the `RocmDevice` and `RocmStorage` implementations with FFI bindings to:
//! - HIP runtime (`libamdhip64.so`): `hipMalloc`, `hipFree`, `hipMemcpy`
//! - rocBLAS (`librocblas.so`): `rocblas_create_handle`, `rocblas_sgemm`, etc.

pub use grim_tensor::Shape;
use grim_tensor::dtype::{DType, QuantProvenance};
pub use grim_tensor::error::Error;
pub use grim_tensor::error::Result;

pub use grim_tensor::Storage as DTypeStorage;
pub use grim_tensor::backend::{BackendDevice, ComputeHandle};
pub use grim_tensor::{ArithType, BackendStorage};
pub use std::ffi::c_void;
pub use std::sync::Arc;

// ----- Crate-wide module declarations ---------------------------
// Per spec: lib.rs holds the cross-cutting re-export surface plus
// sub-module decls. Each sub-module owns its own implementation.

pub mod autotune;
pub mod device;
pub mod fsdp;
pub mod fusion;
pub mod gptq_kernel;
pub mod graph_capture;
pub mod kernels;
pub mod memory;
pub mod p2p_route;
pub mod peer_access;
pub mod perf_gate;
pub mod quantization;
pub mod rccl;
pub mod rocm_detect;
pub mod speculative;

/// SCYTHE-2 WI-2: capability profiler re-export.
pub use device::capability_profiler::{
    CAPABILITY_EPOCH, CapabilityProfiler, bump_epoch, current_epoch, vram_info,
};

// ----- Crate-root re-exports ------------------------------------
// Existing callers (lib_internal_tests.rs + external crates) see
// these names without needing to know which sub-module they live in.

pub use crate::device::handles::{
    HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC, HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    HIP_DEVICE_ATTRIBUTE_WARP_SIZE, HIP_MEM_ADVISE_SET_ACCESSED_BY,
    HIP_MEM_ADVISE_SET_COARSE_GRAIN, HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
    HIP_MEM_ADVISE_SET_READ_MOSTLY, HIP_MEM_ADVISE_UNSET_ACCESSED_BY,
    HIP_MEM_ADVISE_UNSET_COARSE_GRAIN, HIP_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
    HIP_MEM_ADVISE_UNSET_READ_MOSTLY, HipDim3, HipErrorT, HipGraphKernelNodeParams,
    HipGraphMemcpyNodeParams, HipMemcpyKind, HiprtcProgram, RocmDeviceProps, RocmHandle,
    WavefrontSize, hipDeviceGetAttribute, hipDeviceSynchronize, hipFree, hipGetDeviceCount,
    hipGetDeviceProperties, hipGraphCreate, hipGraphDestroy, hipGraphExecDestroy,
    hipGraphExtendFromGlobalStream, hipGraphInstantiate, hipGraphLaunch, hipGraphUpload,
    hipHostFree, hipHostMalloc, hipMalloc, hipMallocManaged, hipMemAdvise, hipMemGetInfo,
    hipMemPrefetchAsync, hipMemcpy, hipMemcpyAsync, hipMemset, hipMemsetAsync,
    hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad, hipModuleUnload, hipSetDevice,
    hipStreamBeginCapture, hipStreamCreate, hipStreamDestroy, hipStreamEndCapture,
    hipStreamSynchronize, hipSuccess, hiprtcAddNameExpression, hiprtcCompileProgram,
    hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode, hiprtcGetCodeSize,
    hiprtcGetErrorString, hiprtcGetProgramLog, hiprtcGetProgramLogSize,
};

pub use crate::device::rocblas::{
    ROCBLAS_GEMM_FLAGS_NONE,
    RocblasInt,
    RocblasOperation,
    Rocblstatus,
    RoclabsHandle,
    arith_to_compute_dtype,
    arith_to_rocblas_dtype,
    rocblas_create_handle,
    rocblas_datatype,
    rocblas_destroy_handle,
    rocblas_gemm_algo,
    rocblas_gemm_ex,
    rocblas_gemm_flags,
    rocblas_gemm_strided_batched_ex,
    rocblas_set_stream,
    rocblas_sgemm,
    rocblas_status_success,
    // gemm-tuning dispatch helper — picks the right `algo` enum from a
    // non-zero `solution_index` lookup table entry, falling back to
    // `rocblas_gemm_algo::standard` for `solution_index == 0`.
    select_gemm_algo,
};

pub use crate::device::layout::{
    KvLayout, WavefrontTiledLayout, WeightLayout, align_quantized_tensor_for_rocm_gemm,
    align_tensor_for_rocm_gemm, attention_min_bpw, enforce_attention_precision,
    is_attention_projection, kv_from_block_major, kv_to_block_major, resolve_weight_layout,
    select_kv_layout,
};

pub use crate::device::gemm_tuning::{GemmTileConfig, lookup_gemm_config, lookup_solution_index};

pub use crate::graph_capture::{CapturedGraph, HipGraphExecutor, hip_graph_launch};

pub use crate::gptq_kernel::wavefront_size_for_gcn;

pub use crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE;
pub use crate::kernels::jit_cache::HsacoKernelCache;
pub use crate::kernels::source_asm::compute_kernel_source;

pub use crate::memory::allocator::RocmCachingAllocator;
pub use crate::memory::pinned::RocmPinnedBuffer;
pub use crate::memory::storage::RocmStorage;

pub use crate::device::helpers::{
    check_hip, jit_compile_hsaco, memcpy_with_xnack_fallback, upload_device_buffer,
};
pub use crate::device::probe::{probe_host_gpu, probe_system_rocm, probe_xnack};

pub use crate::device::util::{
    ROCM_COMPUTE_BLOCK, arg, as_rocm, detect_gpu_arch, dev_ptr, dtype_byte_size, dtype_f32,
    gpu_target_arch, gpu_target_flag, linear_launch,
};

// ROCmDevice itself: large struct + every impl lives in
// `device::roc_device`. Re-exported here so existing callers can
// keep using `RocmDevice::new(...)` etc. unchanged.
pub use crate::device::roc_device::{FUSED_FORWARD_DISPATCH_STATS, RocmDevice};

pub use crate::rccl::RcclAllReduce;

pub use fusion::{
    DecodeGemmConfig, FusedDequantGemmConfig, HipKernelLaunch, KvDequantAttentionConfig,
    QkvAttentionFusionConfig, RmsNormMatMulFusionConfig, SplitKGemmConfig, WmmaGemmConfig, hipDim3,
};

pub use kernels::qkv_attention::{BlockTableEntry, launch_paged_attention, launch_tree_attention};

pub use quantization::QuantMode;

#[cfg(test)]
mod lib_internal_tests;
