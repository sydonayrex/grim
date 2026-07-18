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

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use std::fs;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{DType, QuantProvenance, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendDevice, BackendStorage, Shape};
use grim_format::gguf::{GrimMetadata, GrimLayoutHint};

// ----- Crate-wide module declarations ---------------------------
// Per spec: lib.rs holds the cross-cutting re-export surface plus
// sub-module decls. Each sub-module owns its own implementation.

pub mod autotune;
pub mod device;
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

// ----- Crate-root re-exports ------------------------------------
// Existing callers (lib_internal_tests.rs + external crates) see
// these names without needing to know which sub-module they live in.

pub use crate::device::handles::{
    hipDeviceGetAttribute, hipDeviceSynchronize, hipFree, hipGetDeviceCount,
    hipGetDeviceProperties, hipGraphCreate, hipGraphDestroy,
    hipGraphExecDestroy, hipGraphExtendFromGlobalStream, hipGraphInstantiate,
    hipGraphLaunch, hipGraphUpload, hipHostFree, hipHostMalloc,
    hipMemAdvise, hipMemcpy, hipMemcpyAsync, hipMemset, hipMemsetAsync,
    hipMalloc, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad,
    hipModuleUnload, hipSetDevice, hipStreamBeginCapture, hipStreamCreate,
    hipStreamDestroy, hipStreamEndCapture, hipStreamSynchronize,
    hiprtcAddNameExpression, hiprtcCompileProgram, hiprtcCreateProgram,
    hiprtcDestroyProgram, hiprtcGetCode, hiprtcGetCodeSize,
    hiprtcGetErrorString, hiprtcGetProgramLog, hiprtcGetProgramLogSize,
    HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC, HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    HIP_DEVICE_ATTRIBUTE_WARP_SIZE, HIP_MEM_ADVISE_SET_ACCESSED_BY,
    HIP_MEM_ADVISE_SET_COARSE_GRAIN, HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
    HIP_MEM_ADVISE_SET_READ_MOSTLY, HIP_MEM_ADVISE_UNSET_ACCESSED_BY,
    HIP_MEM_ADVISE_UNSET_COARSE_GRAIN, HIP_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
    HIP_MEM_ADVISE_UNSET_READ_MOSTLY, HipDim3, HipErrorT, HipGraphKernelNodeParams,
    HipGraphMemcpyNodeParams, HipMemcpyKind, HiprtcProgram, RocmDeviceProps, RocmHandle,
    WavefrontSize, hipSuccess,
};

pub use crate::device::rocblas::{
    arith_to_compute_dtype, arith_to_rocblas_dtype,
    rocblas_create_handle, rocblas_destroy_handle, rocblas_gemm_ex,
    rocblas_gemm_strided_batched_ex, rocblas_set_stream, rocblas_sgemm,
    rocblas_status_success, RocblasInt, RocblasOperation, Rocblstatus,
    RoclabsHandle, rocblas_datatype, rocblas_gemm_algo, rocblas_gemm_flags,
    ROCBLAS_GEMM_FLAGS_NONE,
    // gemm-tuning dispatch helper — picks the right `algo` enum from a
    // non-zero `solution_index` lookup table entry, falling back to
    // `rocblas_gemm_algo::standard` for `solution_index == 0`.
    select_gemm_algo,
};

pub use crate::device::layout::{
    align_quantized_tensor_for_rocm_gemm, align_tensor_for_rocm_gemm,
    attention_min_bpw, enforce_attention_precision, is_attention_projection,
    kv_from_block_major, kv_to_block_major, resolve_weight_layout,
    select_kv_layout, KvLayout, WeightLayout, WavefrontTiledLayout,
};

pub use crate::device::gemm_tuning::{
    GemmTileConfig, lookup_gemm_config, lookup_solution_index,
};

pub use crate::graph_capture::{CapturedGraph, HipGraphExecutor, hip_graph_launch};

pub use crate::gptq_kernel::wavefront_size_for_gcn;

pub use crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE;
pub use crate::kernels::source_asm::compute_kernel_source;
pub use crate::kernels::jit_cache::HsacoKernelCache;

pub use crate::memory::allocator::RocmCachingAllocator;
pub use crate::memory::storage::RocmStorage;
pub use crate::memory::pinned::RocmPinnedBuffer;

pub use crate::device::probe::probe_xnack;
pub use crate::device::helpers::{
    jit_compile_hsaco, memcpy_with_xnack_fallback, upload_device_buffer,
};

pub use crate::device::util::{
    arg, as_rocm, dev_ptr, detect_gpu_arch, dtype_byte_size, dtype_f32,
    gpu_target_arch, gpu_target_flag, linear_launch, ROCM_COMPUTE_BLOCK,
};

// ROCmDevice itself: large struct + every impl lives in
// `device::roc_device`. Re-exported here so existing callers can
// keep using `RocmDevice::new(...)` etc. unchanged.
pub use crate::device::roc_device::RocmDevice;

pub use fusion::{DecodeGemmConfig, FusedDequantGemmConfig, SplitKGemmConfig, HipKernelLaunch, QkvAttentionFusionConfig, RmsNormMatMulFusionConfig, KvDequantAttentionConfig, hipDim3};

pub use kernels::qkv_attention::{BlockTableEntry, launch_paged_attention, launch_tree_attention};

pub use quantization::QuantMode;

#[cfg(test)]
mod lib_internal_tests;
