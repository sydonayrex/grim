//! ROCm backend for Grim — primary GPU target per architecture §4.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::not_unsafe_ptr_arg_deref
)]
//!
//! Replicates core architectural design concepts from the `rocm-rs` library ecosystem:
//! - Safe RAII allocation handles (Drop-on-scope, zero leaks) mimicking `DeviceMemoryExt`.
//! - Modular FFI layer designed for drop-in bindings to AMD's rocBLAS and HIP runtime.
//! - Explicit attribute-probing correctness gates mapping device traits.
//!
//! This crate provides the `RocmDevice` and `RocmStorage` implementations with FFI bindings to:
//! - HIP runtime (`libamdhip64.so`): `hipMalloc`, `hipFree`, `hipMemcpy`
//! - rocBLAS (`librocblas.so`): `rocblas_create_handle`, `rocblas_sgemm`, etc.
//!
//! # Feature gates (jit-mgpu.md §10)
//!
//! - **`jit-hw-adaptive`** *(default on)* — hardware-adaptive JIT. `launch_compute_kernel`
//!   injects hardware-discovered `#define`s (wavefront/LDS/CU + tile geometry from
//!   [`device::hardware_spec::HardwareSpec`] + [`kernels::tile_picker::pick_tiles`]) into the
//!   kernel source before hiprtc compile, and keys the `.hsaco` cache by a hardware
//!   fingerprint. Disable to fall back to the static source path.
//! - **`multi-gpu-kernel`** *(default off)* — multi-GPU kernel launch
//!   ([`multi_gpu_launch::launch_multi_gpu_kernel`]): splits the M dimension across devices,
//!   JIT-compiles a per-device kernel, and RCCL-all-reduces the shards. Off by default because
//!   it requires RCCL initialization and P2P setup not all deployments need.

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
#[cfg(feature = "multi-gpu-kernel")]
pub mod multi_gpu_launch;
pub mod p2p_route;

pub mod peer_access;
pub mod perf_gate;
pub mod quantization;
pub mod rccl;
pub mod rocm_detect;
pub mod speculative;
pub mod trace;

/// SCYTHE-2 WI-2: capability profiler re-export.
pub use device::capability_profiler::{
    CAPABILITY_EPOCH, CapabilityProfiler, bump_epoch, compute_utilization, current_epoch, vram_info,
};
pub use device::moe_hybrid_exec::{MoeGraphSyncFlag, MoeHybridExecutionPlan, MoeHybridExecutor};

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
    WavefrontSize, hipDeviceGetAttribute, hipDeviceSynchronize, hipEventCreate, hipEventDestroy,
    hipEventRecord, hipEventSynchronize, hipFree, hipFreeAsync, hipGetDeviceCount,
    hipGetDeviceProperties, hipGraphCreate, hipGraphDestroy, hipGraphExecDestroy,
    hipGraphExtendFromGlobalStream, hipGraphInstantiate, hipGraphLaunch, hipGraphUpload,
    hipHostFree, hipHostMalloc, hipMalloc, hipMallocManaged, hipMemAdvise, hipMemGetInfo,
    hipMemPrefetchAsync, hipMemcpy, hipMemcpyAsync, hipMemset, hipMemsetAsync,
    hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad, hipModuleUnload, hipSetDevice,
    hipStreamBeginCapture, hipStreamCreate, hipStreamDestroy, hipStreamEndCapture,
    hipStreamSynchronize, hipStreamWaitEvent, hipSuccess, hiprtcAddNameExpression,
    hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode,
    hiprtcGetCodeSize, hiprtcGetErrorString, hiprtcGetLoweredName, hiprtcGetProgramLog,
    hiprtcGetProgramLogSize,
};

pub use crate::device::rocblas::{
    ROCBLAS_GEMM_FLAGS_NONE,
    RocblasHandle,
    RocblasInt,
    RocblasOperation,
    Rocblstatus,
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
pub use crate::kernels::fused_linear_ce::FUSED_LINEAR_CE_KERNEL_SOURCE;
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
    gpu_target_arch, gpu_target_flag, gpu_test_enabled, linear_launch, prefill_in_flight,
    raw_set_device, set_prefill_in_flight, warp_rows_launch,
};

// ROCmDevice itself: large struct + every impl lives in
// `device::roc_device`. Re-exported here so existing callers can
// keep using `RocmDevice::new(...)` etc. unchanged.
pub use crate::device::roc_device::{
    CharonBackwardResult, FUSED_FORWARD_DISPATCH_STATS, RocmDevice,
};

pub use crate::graph_capture::{
    DecodeBatchBucket, DecodeBucketGraphPool, DecodeGraph, DecodeGraphKey, GraphCaptureManager,
};
pub use crate::rccl::{RcclAllReduce, RocmMultiNodeGroup};

pub use fusion::{
    DecodeGemmConfig, FusedDequantGemmConfig, HipKernelLaunch, KvDequantAttentionConfig,
    KvQuantFormat, QkvAttentionFusionConfig, RmsNormMatMulFusionConfig, SplitKGemmConfig,
    WmmaGemmConfig, concat_qkv_weights, hipDim3,
};

pub use kernels::qkv_attention::{
    BlockTableEntry, KvCacheQuantFormat, launch_paged_attention, launch_paged_attention_quant,
    launch_qkv_attention_wmma, launch_tree_attention,
};
pub use kernels::tile_picker::run_install_tune;

/// WI-X3: GPU-native stochastic sampling (`grim_sample_logits_stochastic`,
/// defined in `kernels::device_sampler`). Single-block JIT kernel applying
/// temperature scaling, top-k and top-p filtering entirely on device, then
/// drawing a token with the Gumbel-max trick (multinomial without a cumsum
/// scan). Only the 4-byte token id crosses PCIe.
///
/// **Call-site hook**: the greedy counterpart `grim_sample_logits_argmax`
/// (`kernels/speculative_sampler.rs`) currently has no Rust-side wrapper
/// caller, and production sampling still reads the full logits row to the host
/// before CPU sampling in `grim-server/src/lib.rs::sample_next_token`
/// (~lines 465-489: `outcome.logits.to_vec_f32()` followed by
/// `sampler.sample(...)` on a CPU tensor). Callers that want device-side
/// sampling should invoke [`sample_logits_on_device`] /
/// [`sample_logits_on_device_at`] there — with the ROCm storage backing
/// `outcome.logits`, the model `vocab`, and the request's sampling params —
/// BEFORE that readback, and fall back to the existing CPU sampler whenever
/// this returns `Ok(None)` (unsupported shape/vocab) or `Err` (any HIP
/// failure). Stochastic requests (`temperature > 0 || top_k > 0 || top_p <
/// 1.0`) route here; pure-greedy requests may keep the CPU argmax path.
pub use crate::kernels::device_sampler::{
    DEVICE_SAMPLER_KERNEL_SOURCE, MAX_DEVICE_SAMPLER_VOCAB, sample_logits_on_device,
    sample_logits_on_device_at,
};

pub use quantization::QuantMode;

/// WI-2: arch-gate helpers re-exported so callers outside
/// `grim-backend-rocm` (e.g. `grim-cli::doctor`) can classify a
/// `WeightFormat` against detected hardware without reimplementing the
/// capability table. Prerequisite change per the WI-2 plan: these were
/// `pub` in their module but invisible at the crate boundary.
pub use quantization::{arch_capability, gcn_arch, resolve_quant_mode};

/// WI-2: coarse-grained arch bin, re-exported so callers outside
/// `grim-backend-rocm` (e.g. `grim-cli::doctor`) can classify a model
/// against detected hardware without reaching into the `quantization`
/// module.
pub use quantization::GcnArch;

#[cfg(test)]
mod lib_internal_tests;

// WI-M3: context-correctness gates for the multi-GPU HIP context-drift
// fault (gguf_multigpu_context_plan.md). Device-gated; needs >=2 GPUs.
#[cfg(test)]
mod context_drift_tests;

// Root-cause probe: first-JIT zero-logits through the production sampler
// (scythe2 plan validation log 2026-08-23e). Device-gated.
#[cfg(test)]
mod sampler_zero_probe;
