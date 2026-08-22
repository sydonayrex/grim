//! `RocmDevice` — the ROCm-side GPU device. Constructed via [see: `RocmDevice::new(ordinal)`, `.hsaco`, `HsacoKernelCache`, `BackendDevice`]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use grim_tensor::backend::{ComputeHandle, ReadyHandle, ScythePlacement};
use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendDevice, BackendStorage, Shape};

/// Statistics for `BackendDevice::quantized_matmul_backward_dx` dispatch (WI-F5-close). [see: `attempts`, `grim-autograd::matmul_backward`]
#[derive(Debug, Default)]
pub struct FusedBackwardDispatchStats {
    pub attempts: AtomicUsize,
    pub kernel_calls: AtomicUsize,
    pub fallback_calls: AtomicUsize,
}

/// Process-wide counter shared by every `RocmDevice` instance. Read with [see: `#[cfg(test)]`, `take()`]
pub static FUSED_BACKWARD_DISPATCH_STATS: FusedBackwardDispatchStats = FusedBackwardDispatchStats {
    attempts: AtomicUsize::new(0),
    kernel_calls: AtomicUsize::new(0),
    fallback_calls: AtomicUsize::new(0),
};

#[derive(Debug, Default)]
pub struct FusedForwardDispatchStats {
    pub attempts: AtomicUsize,
    pub kernel_calls: AtomicUsize,
    pub fallback_calls: AtomicUsize,
    pub last_backup2_bpw: AtomicUsize,
    pub last_backup2_codes_offset: AtomicUsize,
    pub last_backup2_scale_offset: AtomicUsize,
}

pub static FUSED_FORWARD_DISPATCH_STATS: FusedForwardDispatchStats = FusedForwardDispatchStats {
    attempts: AtomicUsize::new(0),
    kernel_calls: AtomicUsize::new(0),
    fallback_calls: AtomicUsize::new(0),
    last_backup2_bpw: AtomicUsize::new(0),
    last_backup2_codes_offset: AtomicUsize::new(0),
    last_backup2_scale_offset: AtomicUsize::new(0),
};

// Symbols that lib.rs re-exports publicly. They live in sub-modules [see: `crate::*`, `pub use`]
use crate::{
    CapturedGraph,
    DecodeGemmConfig,
    FusedDequantGemmConfig,
    // HIP types / constants
    HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
    HipDim3,
    HipErrorT,
    HipMemcpyKind,
    // kernel cache + graph capture
    HsacoKernelCache,
    QkvAttentionFusionConfig,
    QuantMode,
    ROCBLAS_GEMM_FLAGS_NONE,
    RmsNormMatMulFusionConfig,
    RocblasHandle,
    RocblasInt,
    RocblasOperation,
    RocmCachingAllocator,
    RocmDeviceProps,
    RocmHandle,
    RocmPinnedBuffer,
    // Misc types
    RocmStorage,
    SplitKGemmConfig,
    WavefrontSize,
    WmmaGemmConfig,
    // lib.rs helpers (re-exported from memory/, device::util/, etc.)
    arg,
    // rocBLAS FFI
    arith_to_compute_dtype,
    arith_to_rocblas_dtype,
    as_rocm,
    // device helpers
    check_hip,
    detect_gpu_arch,
    dev_ptr,
    dtype_f32,
    // HIP runtime FFI
    hipDeviceGetAttribute,
    hipDeviceSynchronize,
    hipFree,
    hipFreeAsync,
    hipGetDeviceCount,
    hipGraphDestroy,
    hipGraphExecDestroy,
    hipGraphInstantiate,
    hipGraphLaunch,
    hipMemAdvise,
    hipMemGetInfo,
    hipMemcpyAsync,
    hipMemset,
    hipMemsetAsync,
    hipModuleGetFunction,
    hipModuleLaunchKernel,
    hipModuleLoad,
    hipModuleUnload,
    hipSetDevice,
    hipStreamBeginCapture,
    hipStreamCreate,
    hipStreamDestroy,
    hipStreamEndCapture,
    hipStreamSynchronize,
    hipSuccess,
    jit_compile_hsaco,
    linear_launch,
    rocblas_create_handle,
    rocblas_destroy_handle,
    rocblas_gemm_ex,
    rocblas_gemm_strided_batched_ex,
    rocblas_set_stream,
    rocblas_sgemm,
    rocblas_status_success,
    select_gemm_algo,
    upload_device_buffer,
    warp_rows_launch,
};

/// Return type for [`RocmDevice::charon_grouped_backward_roundtrip`].
///
/// Holds the four named gradient buffers from the Charon MoE backward kernel,
/// each as a flat `Vec<f32>` in the device layout (expert-outermost for weight
/// grads, batch-outermost for `d_x`).
#[derive(Debug, Clone)]
pub struct CharonBackwardResult {
    pub d_gate_w: Vec<f32>,
    pub d_up_w: Vec<f32>,
    pub d_down_w: Vec<f32>,
    pub d_x: Vec<f32>,
}

#[derive(Debug)]
pub struct RocmDevice {
    pub(crate) ordinal: usize,
    pub(crate) props: RocmDeviceProps,
    handle_cache: Mutex<Option<RocblasHandle>>,
    pub(crate) stream_pool: Mutex<Vec<*mut c_void>>,
    pub(crate) hsaco_cache: HsacoKernelCache,
    /// WI 2.4.4-2 — opt-in switch for the JIT `grim_decode_gemm_f16` [see: `false`, `fusion::DecodeGemmConfig`, `Mutex`, `handle_cache`]
    pub(crate) decode_gemm_config: Mutex<DecodeGemmConfig>,
    pub(crate) fused_dequant_gemm_config: Mutex<FusedDequantGemmConfig>,
    pub(crate) split_k_config: Mutex<SplitKGemmConfig>,
    pub(crate) wmma_gemm_config: Mutex<WmmaGemmConfig>,
    /// AtomicBool shadow of `decode_gemm_config.enabled` — read lock-free on every matmul
    /// dispatch. Written by `set_decode_gemm_enabled`. [see: `decode_gemm_config`]
    pub(crate) decode_gemm_enabled: AtomicBool,
    /// AtomicBool shadow of `fused_dequant_gemm_config.enabled` — read lock-free on every
    /// quantized_matmul dispatch. Written by `set_fused_dequant_gemm_enabled`.
    pub(crate) fused_dequant_gemm_enabled: AtomicBool,
    /// Opt-in gate for the Jay-Tier MXFP4 fused dequant-GEMM kernel
    /// (`launch_fused_dequant_gemm_mxfp4`). Defaults to `false` so the proven
    /// tiled MXFP4 path stays the default until parity with the F32 oracle is
    /// confirmed on a target GPU. Written by `set_mxfp4_fused_dequant_gemm_enabled`.
    pub(crate) mxfp4_fused_dequant_gemm_enabled: AtomicBool,
    /// AtomicBool shadow of `wmma_gemm_config.enabled` — read lock-free by
    /// `should_use_wmma_path`. Written by `set_wmma_gemm_enabled`.
    pub(crate) wmma_gemm_enabled: AtomicBool,
    /// Caching device-memory allocator (size-bucketed free-list). See `RocmCachingAllocator`.
    pub(crate) allocator: Arc<RocmCachingAllocator>,
    /// Pinned host buffers backing in-flight stream-ordered H2D copies. A
    /// `hipMemcpyAsync` reads these pages on the copy engine *after* the CPU
    /// returns, so the page-locked source must outlive the enqueue. Each
    /// stream-ordered upload retains its pin here; `synchronize()` drains the
    /// list only after the device has completed all queued copies. This is the
    /// lifetime-safety half of the async H2D pipeline.
    pub(crate) retained_pins: Mutex<Vec<RocmPinnedBuffer<f32>>>,
    /// Phase-3 §3.1: device scratch pool — a thread-safe, power-of-2-bucketed [see: `hipMalloc`, `get_scratch`]
    pub(crate) scratch_pool: Arc<crate::memory::pool::DeviceScratchPool>,
    /// Loaded HIP modules + resolved entry functions, cached per unique kernel entry. [see: `hipModuleLoad`, `hipModuleGetFunction`]
    pub(crate) autotuner: Mutex<crate::autotune::Autotuner>,
    /// Tuning-mode + occupancy + tuning-solution store for this device.
    /// [salamander.md §3.6: TuningMode, BlockSizeBand, OccupancyTuning,
    /// tuning solution storage]
    #[allow(dead_code)]
    pub(crate) tuning: Mutex<crate::autotune::AutotunerConfig>,

    pub(crate) module_cache: Mutex<HashMap<String, (*mut c_void, *mut c_void)>>,
    /// Resolved-function fast path for `launch_compute_kernel_with_solution`:
    /// (entry, grid_x, grid_y) -> hipFunction. Skips the per-launch kernel
    /// source regeneration + seahash + CString work for repeat launches (the
    /// overwhelming case on the decode hot path). Grid dims are part of the
    /// key because `jit-hw-adaptive` bakes tile geometry derived from them
    /// into the source.
    pub(crate) resolved_kernel_cache: Mutex<HashMap<(String, u32, u32, Option<i32>), *mut c_void>>,
    /// Interner for `&'static str` autotune keys (entry / arch). Each unique
    /// string is leaked EXACTLY ONCE; repeat `get_or_tune_tiles` /
    /// `store_tune_cache` calls reuse it instead of leaking per call.
    pub(crate) str_interner: Mutex<std::collections::HashSet<&'static str>>,
    /// Real `hipModuleLoad` call count (cache hits excluded). Item 2 acceptance.
    pub(crate) module_load_count: AtomicUsize,
    /// Total kernel + GEMM launches since the last `reset_launch_count`.
    /// Instrumentation for fusion-boundary launch-count gates (WI-F1 etc.);
    /// counts every `hipModuleLaunchKernel` and every rocBLAS GEMM enqueued
    /// through `matmul_op`.
    pub(crate) launch_counter: AtomicUsize,
    /// GPU target this device was created for, captured at construction. Used to [see: `temp_env::with_var("GRIM_GPU_TARGET", ..)`]
    pub(crate) gpu_target: String,
    /// Whether graph capture/replay is enabled. Keyed off the `GRIM_CAPTURE_GRAPH`
    capture_enabled: bool,
    /// The dedicated capture stream, owned for the device's lifetime. Created lazily on [see: `begin_graph_capture`, `Drop`]
    capture_stream: RwLock<Option<*mut c_void>>,
    /// True only between `begin_graph_capture` and `end_graph_capture`. Gates the [see: `active_stream`, `active_capture_stream`]
    capture_active: AtomicBool,
    /// Keyed cache of captured + instantiated graphs. A graph is recorded exactly once [see: `replay_graph`]
    captured_graphs: Mutex<HashMap<String, CapturedGraph>>,
    /// GraphCaptureManager for decode-step graph capture/replay. Lazily initialized.
    graph_capture_mgr: Mutex<Option<crate::graph_capture::GraphCaptureManager>>,
    /// Whether batched GEMM rocBLAS handle has been warmed up.
    pub(crate) batched_gemm_warmed: AtomicBool,
    /// Optional NCCL/RCCL communicator for multi-GPU all-reduce.
    pub(crate) rccl: Mutex<Option<Arc<crate::rccl::RcclAllReduce>>>,
    /// Upload completion event for async H2D pipeline.
    pub(crate) upload_event: Mutex<Option<*mut c_void>>,
}

// SAFETY: `RocmDevice` wraps HIP device state (context, stream pool, handle
// caches) that is process-local and accessed only through the owning thread's
// HIP context. Moving the device to another thread (Send) is safe because HIP
// contexts are thread-local but the device ordinal remains valid. The type is
// Sync because all mutable state is behind interior mutability (Mutex,
// AtomicBool) that serializes access.
//
// Current enforcement: all live call paths into this type pass through
// `AppState.engine: Mutex<Engine>` in grim-server, so no concurrent access is
// possible through the server's actual API today. Do NOT remove that lock or add
// a second concurrent access path (e.g. worker pool, background prefetch thread)
// without auditing the interior mutability here first.
unsafe impl Send for RocmDevice {}
unsafe impl Sync for RocmDevice {}

impl RocmDevice {
    /// Allocate raw device bytes for an engine-owned persistent dispatch ring.
    pub fn alloc_scythe_ring_bytes(&self, bytes: usize) -> Result<RocmStorage> {
        RocmStorage::alloc_gpu_with_bytes(
            &Shape::from_slice(&[bytes]),
            dtype_f32(),
            bytes,
            &self.allocator,
            self.ordinal,
        )
    }

    /// Enqueue one descriptor upload on the device's active stream.
    pub fn copy_scythe_descriptor_async(
        &self,
        dst: u64,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> Result<()> {
        check_hip("hipMemcpyAsync(ScytheRing H2D)", unsafe {
            hipMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src,
                bytes,
                HipMemcpyKind::HostToDevice,
                self.active_stream(),
            )
        })
    }

    /// Create a new ROCm device instance and initialize its handle caches and stream pool. [see: `RocmDevice::try_new`, `hipSetDevice`]
    pub fn new(ordinal: usize) -> Self {
        match Self::try_new(ordinal) {
            Ok(dev) => dev,
            Err(e) => {
                // Surface the failure loudly so a misconfigured host is [see: `Error::Backend`]
                eprintln!(
                    "[RocmDevice::new] hipSetDevice({ordinal}) failed: {e}; \
                     constructing a no-stream fallback device"
                );
                Self::fallback(ordinal)
            }
        }
    }

    /// Select the best available ROCm device.
    pub fn new_best() -> Self {
        Self::new(0)
    }

    /// Fallible constructor that propagates the `hipSetDevice` error. [see: `probe()`, `RocmDevice::new`]
    pub fn try_new(ordinal: usize) -> Result<Self> {
        let detected = detect_gpu_arch(ordinal as i32);
        crate::rocm_detect::auto_configure_hsa_override(&detected);

        unsafe {
            let set_status = hipSetDevice(ordinal as i32);
            if set_status != hipSuccess {
                return Err(Error::Backend(format!(
                    "hipSetDevice({ordinal}) failed with code {set_status} \
                     (is the ordinal out of range?)"
                )));
            }
        }

        let mut handle_cache = None;
        // Attempt to create rocblas handle lazily on first op if needed.
        unsafe {
            let mut h: RocblasHandle = RocblasHandle(std::ptr::null_mut());
            let status = rocblas_create_handle(&mut h);
            if status == rocblas_status_success {
                handle_cache = Some(h);
            }
        }

        // Query device attributes for Wavefront size correctness gate.
        let mut warp_size = 32; // Default to W32 (RDNA) fallback
        let mut xnack_val = 0;
        let mut streams = Vec::new();
        unsafe {
            let mut val = 0;
            let status =
                hipDeviceGetAttribute(&mut val, HIP_DEVICE_ATTRIBUTE_WARP_SIZE, ordinal as i32);
            if status == hipSuccess {
                warp_size = val;
            }
            let status_xnack = hipDeviceGetAttribute(
                &mut xnack_val,
                HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
                ordinal as i32,
            );
            if status_xnack != hipSuccess {
                xnack_val = 0;
            }

            // Create a pool of 4 streams for reusing across dispatches
            for _ in 0..4 {
                let mut stream: *mut c_void = std::ptr::null_mut();
                let status = hipStreamCreate(&mut stream);
                if status == hipSuccess && !stream.is_null() {
                    streams.push(stream);
                }
            }
        }
        let dev = Self::build(ordinal, warp_size, xnack_val, handle_cache, streams);
        // Auto-init RCCL when multi-process TP is active — this rank process
        // builds its own RcclAllReduce over the full ordinal list so
        // `RowParallelLinear::forward`'s all_reduce has a live comm handle.
        dev.auto_init_rccl();
        Ok(dev)
    }
    fn fallback(ordinal: usize) -> Self {
        Self::build(ordinal, 32, 0, None, Vec::new())
    }

    /// Attach (or detach) an RCCL multi-GPU collective handle. Called by the
    /// training orchestrator after constructing `RcclAllReduce` so that
    /// [`BackendDevice::all_reduce`] and [`BackendDevice::comm_fuse_reduce`]
    /// can dispatch device-side collectives instead of falling back to the
    /// CPU fan-in path. [see: `RcclAllReduce::try_new`]
    pub fn set_rccl_handle(&self, handle: Option<Arc<crate::rccl::RcclAllReduce>>) {
        *self.rccl.lock().unwrap_or_else(|e| e.into_inner()) = handle;
    }

    /// Borrow the live RCCL handle (if any) for diagnostic / external use.
    pub fn rccl_handle(&self) -> Option<Arc<crate::rccl::RcclAllReduce>> {
        self.rccl.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Auto-init the RCCL handle from `GRIM_TP_*` env vars when multi-process
    /// TP is active. Each rank process calls this after construction; the
    /// handle covers the full ordinal list so every rank's `ncclAllReduce`
    /// rendezvous with its peers. No-op when `GRIM_TP_SIZE <= 1` or when
    /// RCCL is unavailable.
    ///
    /// Reads the env inline (mirrors `TensorParallelConfig::from_env`) because
    /// `grim-backend-rocm` cannot depend on `grim-nn` (grim-nn → grim-backend-
    /// rocm would form a cycle). Must agree with the ordinal-resolution logic
    /// in `grim-engine`'s `model_loader` and `Engine::new`.
    pub fn auto_init_rccl(&self) {
        // Inline TensorParallelConfig::from_env — returns None when GRIM_TP_SIZE
        // is unset or 1 (single-device).
        let world_size = std::env::var("GRIM_TP_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&w| w > 1);
        let Some(world_size) = world_size else {
            return;
        };
        let rank = std::env::var("GRIM_TP_RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if rank >= world_size {
            eprintln!(
                "[RocmDevice] invalid TP config: rank {rank} >= world_size {world_size}; \
                 skipping RCCL init"
            );
            return;
        }
        // Build the full ordinal list: explicit GRIM_GPUS (one per rank)
        // or fall back to 0..world_size.
        let gpus: Vec<usize> = std::env::var("GRIM_GPUS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default();
        let all_ordinals: Vec<usize> = if !gpus.is_empty() && gpus.len() >= world_size {
            gpus.iter().take(world_size).copied().collect()
        } else {
            (0..world_size).collect()
        };
        match crate::rccl::RcclAllReduce::try_new(&all_ordinals) {
            Ok(rccl) => {
                self.set_rccl_handle(Some(Arc::new(rccl)));
                eprintln!(
                    "[RocmDevice] auto-init RCCL: rank {rank}/{world_size} on ordinal {ordinal}, \
                     comm over {ordinals:?}",
                    rank = rank,
                    world_size = world_size,
                    ordinal = self.ordinal,
                    ordinals = all_ordinals
                );
            }
            Err(e) => {
                eprintln!(
                    "[RocmDevice] RCCL init failed for rank {rank}/{world_size}: {e}; \
                     RowParallelLinear::forward will fall back to partial output",
                    rank = rank,
                    world_size = world_size
                );
            }
        }
    }

    /// P2P memcpy that routes via direct peer DMA or host-bounce staging,
    /// bridging the typed routing decision (`P2PStatus` → `RouteLink`) to the
    /// actual memcpy primitives.
    ///
    /// This is the bridge that `p2p_route.rs` defers: it calls
    /// `peer_access::peer_status` to classify the link, `to_route_link` to
    /// pick the route strategy, then `copy_route` to execute either
    /// `hipMemcpyPeerAsync` (PeerDirect) or a D2H→H2D host-pin `hipMemcpyAsync`
    /// pair (HostBounce). The stream is pulled from this device's stream pool.
    pub fn copy_via_route(
        &self,
        src_device: i32,
        dst_device: i32,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
        len: usize,
    ) -> Result<()> {
        let status = crate::peer_access::peer_status(src_device, dst_device)?;
        let route = crate::p2p_route::to_route_link(status, len as u64, u64::MAX);
        let stream = self
            .get_stream_from_pool(0)
            .ok_or_else(|| Error::Backend("copy_via_route: no stream available in pool".into()))?;
        crate::p2p_route::copy_route(src_device, dst_device, src_ptr, dst_ptr, len, route, stream)
    }

    /// Process-wide cache of constructed devices, keyed by ordinal.
    ///
    /// Constructing a `RocmDevice` is *not* cheap: it creates a rocBLAS handle
    /// (which itself hipMallocs a 32–128 MiB internal workspace), four HIP
    /// streams and a fresh caching allocator. Hot paths (weight materialisation,
    /// per-token tensor uploads) used to call `RocmDevice::new` per tensor,
    /// which on small-VRAM parts (e.g. gfx1036 with a 2 GiB carve-out) exhausts
    /// device memory and makes `rocblas_create_handle` fail with status 5
    /// (`rocblas_status_memory_error`). Use `shared` on those paths.
    fn device_cache() -> &'static Mutex<HashMap<usize, Arc<RocmDevice>>> {
        static CACHE: std::sync::OnceLock<Mutex<HashMap<usize, Arc<RocmDevice>>>> =
            std::sync::OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Return the process-wide shared device for `ordinal`, constructing it on
    /// first use. Prefer this over `RocmDevice::new` anywhere a device is
    /// obtained repeatedly (per tensor, per token, per layer). [see: `RocmDevice::new`]
    pub fn shared(ordinal: usize) -> Arc<RocmDevice> {
        let cache = Self::device_cache();
        let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(dev) = guard.get(&ordinal) {
            return Arc::clone(dev);
        }
        let dev = Arc::new(Self::new(ordinal));
        guard.insert(ordinal, Arc::clone(&dev));
        dev
    }

    /// Probe the total amount of device memory reported by the driver, in bytes. [see: `hipMemGetInfo`, `hipDeviceProp_t`]
    pub fn query_device_vram_bytes(_ordinal: usize) -> usize {
        unsafe {
            let mut free_mem: usize = 0;
            let mut total_mem: usize = 0;
            let status = hipMemGetInfo(&mut free_mem, &mut total_mem);
            if status == hipSuccess && total_mem > 0 {
                return total_mem;
            }
        }
        4usize * 1024 * 1024 * 1024 // probing failed: assume 4 GiB
    }

    /// Shared tail of `try_new` / `fallback`: assemble the struct from
    fn build(
        ordinal: usize,
        warp_size: i32,
        xnack_val: i32,
        handle_cache: Option<RocblasHandle>,
        streams: Vec<*mut c_void>,
    ) -> Self {
        let xnack_enabled = xnack_val == 1;

        let gpu_target = detect_gpu_arch(ordinal as i32);
        let wavefront_size = if let Ok(s) = std::env::var("GRIM_WAVEFRONT_SIZE") {
            if s == "64" {
                WavefrontSize::W64
            } else {
                WavefrontSize::W32
            }
        } else {
            match crate::quantization::gcn_arch(&gpu_target) {
                crate::quantization::GcnArch::CDNA1
                | crate::quantization::GcnArch::CDNA2
                | crate::quantization::GcnArch::CDNA3
                | crate::quantization::GcnArch::CDNA4 => WavefrontSize::W64,
                crate::quantization::GcnArch::RDNA1
                | crate::quantization::GcnArch::RDNA2
                | crate::quantization::GcnArch::RDNA3
                | crate::quantization::GcnArch::RDNA4
                | crate::quantization::GcnArch::UDNA => WavefrontSize::W32,
                _ => {
                    if warp_size == 64 && gpu_target.starts_with("gfx9") {
                        WavefrontSize::W64
                    } else if warp_size == 32 {
                        WavefrontSize::W32
                    } else {
                        WavefrontSize::W32 // Default RDNA fallback
                    }
                }
            }
        };
        let wf_u32 = match wavefront_size {
            WavefrontSize::W32 => 32,
            WavefrontSize::W64 => 64,
        };

        // Phase-aware cache cap: when the env override is absent, derive a
        let cap_bytes: usize = std::env::var("GRIM_ALLOC_POOL_CAP_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                let total_vram = Self::query_device_vram_bytes(ordinal);
                let derived = total_vram / 6; // ≈ 16.7 % of VRAM
                derived
            })
            // Pool floor raised to 512 MB (a 512 MB cap on a 16 GB card
            // forced real hipMalloc/hipFree churn for every transient once
            // the cap was hit — i.e. always); ceiling 4 GB keeps runaway
            // env overrides bounded.
            .clamp(512 * 1024 * 1024, 4 * 1024 * 1024 * 1024);

        let arch_leak: &'static str = Box::leak(gpu_target.clone().into_boxed_str());
        let mut autotuner = crate::autotune::Autotuner::for_device(ordinal, arch_leak);

        let cache_path = std::path::PathBuf::from(format!(".autotune_cache/{gpu_target}.json"));
        if cache_path.exists() {
            if let Ok(bytes) = std::fs::read(&cache_path) {
                if let Ok(t) =
                    crate::autotune::Autotuner::from_json_bytes(ordinal, arch_leak, &bytes)
                {
                    autotuner = t;
                }
            }
        }

        Self {
            ordinal,
            props: RocmDeviceProps {
                wavefront_size,
                xnack_enabled,
            },
            handle_cache: Mutex::new(handle_cache),
            stream_pool: Mutex::new(streams),
            hsaco_cache: HsacoKernelCache::new(),
            allocator: Arc::new(RocmCachingAllocator::new(ordinal, cap_bytes)),
            retained_pins: Mutex::new(Vec::new()),
            scratch_pool: crate::memory::pool::DeviceScratchPool::new(),
            autotuner: Mutex::new(autotuner),
            // Tuning-mode + occupancy + tuning-solution store for this device.
            // [salamander.md §3.6: TuningMode, BlockSizeBand, OccupancyTuning,
            // tuning solution storage]
            tuning: Mutex::new(crate::autotune::AutotunerConfig::default()),

            module_cache: Mutex::new(HashMap::new()),
            resolved_kernel_cache: Mutex::new(HashMap::new()),
            str_interner: Mutex::new(std::collections::HashSet::new()),
            module_load_count: AtomicUsize::new(0),
            launch_counter: AtomicUsize::new(0),
            gpu_target: gpu_target.clone(),
            capture_enabled: std::env::var("GRIM_CAPTURE_GRAPH").is_ok(),
            capture_stream: RwLock::new(None),
            capture_active: AtomicBool::new(false),
            captured_graphs: Mutex::new(HashMap::new()),
            batched_gemm_warmed: AtomicBool::new(false),
            decode_gemm_config: Mutex::new(DecodeGemmConfig {
                enabled: true,
                wavefront_size: wf_u32,
            }),
            fused_dequant_gemm_config: Mutex::new(FusedDequantGemmConfig {
                enabled: true,
                wavefront_size: wf_u32,
            }),
            split_k_config: Mutex::new(SplitKGemmConfig { enabled: true }),
            wmma_gemm_config: Mutex::new(WmmaGemmConfig {
                enabled: matches!(
                    crate::quantization::gcn_arch(&gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                ),
                wavefront_size: wf_u32,
            }),
            decode_gemm_enabled: AtomicBool::new(true),
            fused_dequant_gemm_enabled: AtomicBool::new(true),
            mxfp4_fused_dequant_gemm_enabled: AtomicBool::new(
                match std::env::var("GRIM_MXFP4_FUSED_GEMM") {
                    Ok(v) => {
                        // Explicit operator override: "1"/"true" forces the fused
                        // path on, "0"/"false" forces it off. Any other value
                        // falls back to the arch-confirmed default.
                        !matches!(v.as_str(), "0" | "false" | "off" | "no")
                    }
                    Err(_) => {
                        // GPU-confirmation guard: RDNA4 (gfx12x), UDNA (gfx13x), and CDNA4 (gfx95x)
                        // are the architectures with native FP4/MXFP4 matrix hardware.
                        matches!(
                            crate::quantization::gcn_arch(&gpu_target),
                            crate::quantization::GcnArch::RDNA4
                                | crate::quantization::GcnArch::UDNA
                                | crate::quantization::GcnArch::CDNA4
                        )
                    }
                },
            ),
            wmma_gemm_enabled: AtomicBool::new(matches!(
                crate::quantization::gcn_arch(&gpu_target),
                crate::quantization::GcnArch::RDNA3
                    | crate::quantization::GcnArch::RDNA4
                    | crate::quantization::GcnArch::UDNA
            )),
            rccl: Mutex::new(None),
            upload_event: Mutex::new(None),
            graph_capture_mgr: Mutex::new(None),
        }
    }

    /// Release all pooled device buffers back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        self.allocator.empty_cache();
    }

    /// Return the GCN target architecture string (e.g. "gfx1036", "gfx1100").
    pub fn gcn_arch(&self) -> &str {
        &self.gpu_target
    }

    /// P1-WI-1 dispatch probe: should this GEMM route through the WMMA path [see: `GrimTensorExt`, `true`, `wmma_gemm_config`, `layout_hint`]
    pub fn should_use_wmma_path(
        &self,
        ext: Option<&grim_format::spec::GrimTensorExt>,
        out_arith: ArithType,
    ) -> bool {
        // Lock-free read via AtomicBool shadow; the full Mutex<WmmaGemmConfig> is only
        // consulted by the setter. [see: `wmma_gemm_enabled`, `set_wmma_gemm_enabled`]
        let cfg_enabled = self.wmma_gemm_enabled.load(Ordering::Relaxed);
        wmma_route_decision(ext, out_arith, cfg_enabled)
    }

    /// WI 2.4.4-2 — opt-in flag for the JIT `grim_decode_gemm_f16`. [see: `true`, `QkvAttentionFusionConfig::enabled`]
    pub fn set_decode_gemm_enabled(&self, enabled: bool) {
        // Write the AtomicBool shadow first (lock-free hot-path reads this).
        self.decode_gemm_enabled.store(enabled, Ordering::Relaxed);
        let mut cfg = self.decode_gemm_config.lock().unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// Set whether fused dequantization GEMM is enabled (WI-C).
    pub fn set_fused_dequant_gemm_enabled(&self, enabled: bool) {
        self.fused_dequant_gemm_enabled
            .store(enabled, Ordering::Relaxed);
        let mut cfg = self.fused_dequant_gemm_config.lock().unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// Set whether the Jay-Tier MXFP4 fused dequant-GEMM kernel is enabled.
    /// Defaults to `false` (tiled fallback) until parity with the F32 oracle is
    /// confirmed on a target GPU.
    pub fn set_mxfp4_fused_dequant_gemm_enabled(&self, enabled: bool) {
        self.mxfp4_fused_dequant_gemm_enabled
            .store(enabled, Ordering::Relaxed);
    }

    /// Set whether SplitK GEMM is enabled (WI-D).
    pub fn set_split_k_enabled(&self, enabled: bool) {
        let mut cfg = self.split_k_config.lock().unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// Set whether the JIT compiled WMMA GEMM kernel is enabled (WI-G). [see: `grim_wmma_gemm`]
    pub fn set_wmma_gemm_enabled(&self, enabled: bool) {
        self.wmma_gemm_enabled.store(enabled, Ordering::Relaxed);
        let mut cfg = self.wmma_gemm_config.lock().unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// `(hipMalloc_count, hipFree_count)` since this device was created — real driver
    pub fn allocator_stats(&self) -> (usize, usize) {
        self.allocator.stats()
    }

    /// Number of real `hipModuleLoad` calls since device creation. Cache hits are [see: `module_cache_loads_each_kernel_once`]
    pub fn module_load_stats(&self) -> usize {
        self.module_load_count.load(Ordering::SeqCst)
    }

    /// Reset the kernel/GEMM launch counter (fusion-gate instrumentation).
    pub fn reset_launch_count(&self) {
        self.launch_counter.store(0, Ordering::SeqCst);
    }

    /// Kernel + GEMM launches enqueued since the last `reset_launch_count`.
    pub fn launch_count(&self) -> usize {
        self.launch_counter.load(Ordering::SeqCst)
    }

    /// `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` for a kernel entry that
    /// has already been launched (and thus resolved) on this device. Returns
    /// `None` if the entry is not in the resolved-kernel cache yet. Occupancy
    /// regression harness for fusion gates (WI-F2/F4).
    pub fn kernel_max_blocks_per_cu(&self, entry: &str, block_size: u32) -> Option<i32> {
        let func = self
            .resolved_kernel_cache
            .lock()
            .ok()
            .and_then(|c| c.iter().find(|(k, _)| k.0 == entry).map(|(_, &f)| f))?;
        if func.is_null() {
            return None;
        }
        let mut blocks: i32 = 0;
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let res = unsafe {
            crate::device::handles::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut blocks,
                func,
                block_size as i32,
                0,
            )
        };
        if res != hipSuccess {
            return None;
        }
        Some(blocks)
    }

    /// Phase-3 §3.1: get a pooled scratch buffer. [see: `hipMalloc`, `Result`]
    pub fn get_scratch(
        &self,
        size: usize,
        align: usize,
    ) -> Result<crate::memory::pool::PooledBuffer> {
        self.scratch_pool.get(size, align)
    }

    /// Phase-3 §3.1: peek at the live pool's tracked size (for ops/tests).
    pub fn scratch_pool_current_bytes(&self) -> usize {
        self.scratch_pool.current_bytes()
    }

    /// Phase-3 §3.1: peak in-flight bytes since pool creation.
    pub fn scratch_pool_peak_bytes(&self) -> usize {
        self.scratch_pool.peak_bytes()
    }

    /// Phase-3 §3.1 (REFACTOR): upload `data` into a pooled scratch buffer [see: `hipMalloc`, `hipFree`]
    pub fn upload_to_scratch(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<crate::memory::pool::PooledBuffer> {
        let _ = shape;
        let elem_size: usize = match dtype {
            DType::F32 => 4,
            DType::BF16 => 2,
            _ => {
                return Err(Error::Backend(format!(
                    "upload_to_scratch: unsupported dtype {:?}; only F32/BF16 in this revision",
                    dtype
                )));
            }
        };
        let bytes = data.len() * elem_size;
        let align = elem_size.max(16); // safe default; matches element boundaries.
        let buf = self.scratch_pool.get(bytes, align)?;
        // Copy host → device. We do a synchronous `hipMemcpy` here; the
        let res: HipErrorT = unsafe {
            crate::hipMemcpy(
                buf.as_ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                bytes,
                crate::HipMemcpyKind::HostToDevice,
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "upload_to_scratch: hipMemcpy failed: code={}",
                res
            )));
        }
        Ok(buf)
    }

    /// If a graph-capture session is active, returns the dedicated capture stream. [see: `None`]
    fn active_capture_stream(&self) -> Option<*mut c_void> {
        if self.capture_active.load(Ordering::SeqCst) {
            *self.capture_stream.read().unwrap_or_else(|e| e.into_inner())
        } else {
            None
        }
    }

    /// The stream an op should dispatch onto: the capture stream when a session is
    fn active_stream(&self) -> *mut c_void {
        let stream = if self.capture_active.load(Ordering::SeqCst) {
            self.capture_stream
                .read()
                .unwrap()
                .unwrap_or_else(|| self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut()))
        } else {
            self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut())
        };
        // SPEED-ROC-1: if a stream-ordered upload is in flight on the transfer
        // stream, fence this (compute) stream on its completion event so the
        // prefetch can overlap the prior decode-step GEMM instead of racing it.
        // `hipStreamWaitEvent` is a no-op ordering edge; it does not block the
        // host. The event is recorded by `upload_from_host_stream_ordered`.
        if let Ok(guard) = self.upload_event.lock() {
            if let Some(ev) = *guard {
                if !ev.is_null() {
                    unsafe {
                        let _ = crate::hipStreamWaitEvent(stream, ev, 0);
                    }
                }
            }
        }
        stream
    }

    /// Block until all previously issued work on all streams of this device
    pub fn synchronize(&self) {
        // Pin the correct device before synchronizing — hipDeviceSynchronize()
        // synchronizes the calling thread's current device, not necessarily
        // self.ordinal. [P1-7 fix: DeviceGuard before sync.]
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let _ = unsafe { hipDeviceSynchronize() };
        // All stream-ordered H2D copies are complete, so the pinned host
        // sources they read are now safe to release. Draining here (rather than
        // at each upload) is what allows consecutive uploads to queue on the
        // stream pool and overlap with one another.
        if let Ok(mut pins) = self.retained_pins.lock() {
            pins.clear();
        }
    }

    /// Begin a generic graph-capture session keyed by `key`. Until `end_graph_capture` [see: `key`, `GRIM_CAPTURE_GRAPH`]
    pub fn begin_graph_capture(&self, _key: &str) -> Result<()> {
        if !self.capture_enabled {
            return Ok(());
        }
        if self.capture_active.load(Ordering::SeqCst) {
            return Err(Error::Backend(
                "begin_graph_capture: a capture session is already active".into(),
            ));
        }
        // Lazily create the capture stream; it lives for the device lifetime so rocblas
        let mut cs = self.capture_stream.write().unwrap_or_else(|e| e.into_inner());
        if cs.is_none() {
            let mut stream: *mut c_void = std::ptr::null_mut();
            let res = unsafe { hipStreamCreate(&mut stream) };
            if res != hipSuccess {
                return Err(Error::Backend(format!(
                    "hipStreamCreate (capture) failed: {}",
                    res
                )));
            }
            *cs = Some(stream);
        }
        let stream = cs.unwrap();
        // Canonical rocBLAS graph-capture pattern: bind the handle to the capture
        if let Ok(h) = self.get_rocblas_handle() {
            unsafe {
                let _ = rocblas_set_stream(h, stream);
            }
        }
        // Relaxed capture mode: allocations (hipMalloc for op outputs, rocblas
        let res = unsafe { hipStreamBeginCapture(stream, 2) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamBeginCapture failed: {}",
                res
            )));
        }
        self.capture_active.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// End the capture session started with `key`, instantiate the recorded graph, [see: `key`, `replay_graph`]
    pub fn end_graph_capture(&self, key: &str) -> Result<()> {
        if !self.capture_enabled {
            return Ok(());
        }
        if !self.capture_active.load(Ordering::SeqCst) {
            return Err(Error::Backend(
                "end_graph_capture: no capture session is active".into(),
            ));
        }
        let stream = self
            .capture_stream
            .read()
            .unwrap()
            .unwrap_or(std::ptr::null_mut());
        let mut graph: *mut c_void = std::ptr::null_mut();
        let res = unsafe { hipStreamEndCapture(stream, &mut graph) };
        if res != hipSuccess {
            self.capture_active.store(false, Ordering::SeqCst);
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(Error::Backend(format!(
                "hipStreamEndCapture failed: {}",
                res
            )));
        }
        // Clear the stream so it is ready to be reused by a later capture session.
        unsafe {
            let _ = hipStreamSynchronize(stream);
        }
        let mut exec: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if res != hipSuccess {
            self.capture_active.store(false, Ordering::SeqCst);
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(Error::Backend(format!(
                "hipGraphInstantiate failed: {}",
                res
            )));
        }
        let mut cache = self.captured_graphs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = cache.insert(key.to_string(), CapturedGraph { graph, exec }) {
            unsafe {
                let _ = hipGraphExecDestroy(old.exec);
                let _ = hipGraphDestroy(old.graph);
            }
        }
        // Do NOT reset the rocBLAS handle to the null stream here. Every eager
        // GEMM dispatch re-binds the handle to `active_stream()` before use
        // (P0-17 fix), so leaving it bound to the capture stream is harmless
        // and avoids the footgun where a later GEMM that forgets to set the
        // stream silently lands on the default stream instead of the active
        // one. The next dispatch's `rocblas_set_stream(handle, active_stream())`
        // overwrites this binding regardless.
        self.capture_active.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Replay the graph previously captured under `key`. Returns `Ok(false)` when no [see: `key`, `Ok(true)`]
    pub fn replay_graph(&self, key: &str) -> Result<bool> {
        if !self.capture_enabled {
            return Ok(false);
        }
        // Replay on the same capture stream the graph was recorded on, so the rocblas
        let stream = {
            let cs = self.capture_stream.read().unwrap_or_else(|e| e.into_inner());
            cs.unwrap_or_else(|| self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut()))
        };
        let cache = self.captured_graphs.lock().unwrap_or_else(|e| e.into_inner());
        match cache.get(key) {
            Some(g) => {
                // Bind rocblas to the replay stream so its captured GEMM node executes there.
                if let Ok(h) = self.get_rocblas_handle() {
                    unsafe {
                        let _ = rocblas_set_stream(h, stream);
                    }
                }
                let res = unsafe { hipGraphLaunch(g.exec, stream) };
                if res != hipSuccess {
                    return Err(Error::Backend(format!("hipGraphLaunch failed: {}", res)));
                }
                // No post-replay sync: replay is async on `stream`; callers
                // that need the result sync (or read back) at their boundary.
                // The rocblas handle binding is an enqueue-time setting, so we
                // leave it bound to `stream` rather than resetting to null:
                // the next eager GEMM re-binds to `active_stream()` anyway, and
                // resetting to the default stream would be a footgun (P0-17
                // class) if any future dispatch forgets to set the stream.
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// True if a graph is cached under `key` (useful for callers deciding whether to
    pub fn has_captured_graph(&self, key: &str) -> bool {
        self.captured_graphs.lock().unwrap_or_else(|e| e.into_inner()).contains_key(key)
    }

    // =============================================================================
    // WRECK-9: decode-step graph capture via GraphCaptureManager.
    // =============================================================================

    /// Lazily-initialized GraphCaptureManager for decode-step graph capture.
    fn ensure_graph_capture_mgr(&self) {
        let mut mgr = self.graph_capture_mgr.lock().unwrap_or_else(|e| e.into_inner());
        if mgr.is_none() {
            *mgr = Some(crate::graph_capture::GraphCaptureManager::for_device(self));
        }
    }

    /// Capture the decode-step GEMM (`launch_decode_gemm_f16`) under a shape key via the
    /// GraphCaptureManager, then replay it. Collapses per-step launch+dispatch overhead for
    /// repeated decode steps at the same shape.
    ///
    /// Returns Ok(true) if captured and replayed; Ok(false) if manager not initialized
    /// (callers fall back to eager `launch_decode_gemm_f16`).
    pub fn decode_graph_capture_and_replay(
        &self,
        key: crate::graph_capture::DecodeGraphKey,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<bool> {
        self.ensure_graph_capture_mgr();
        let mgr = self.graph_capture_mgr.lock().unwrap_or_else(|e| e.into_inner());
        let mgr = mgr.as_ref().ok_or_else(|| {
            Error::Backend(
                "decode_graph_capture_and_replay: graph capture manager not initialized".into(),
            )
        })?;
        mgr.get_or_capture(key, |stream| {
            if let Ok(h) = self.get_rocblas_handle() {
                unsafe {
                    let _ = rocblas_set_stream(h, stream);
                }
            }
            self.launch_decode_gemm_f16(a, b, out, m, n, k)?;
            Ok(())
        })?;
        mgr.replay(key)?;
        Ok(true)
    }

    // =============================================================================
    pub fn matmul_batched(
        &self,
        a: &[&dyn BackendStorage],
        b: &[&dyn BackendStorage],
        out_shape: &Shape,
    ) -> Result<Vec<Box<dyn BackendStorage>>> {
        if a.len() != b.len() {
            return Err(Error::Shape(
                "matmul_batched: a and b batch counts differ".into(),
            ));
        }
        let batch = a.len();
        if batch == 0 {
            return Ok(Vec::new());
        }

        // One-time warm-up of the rocBLAS `gemm_strided_batched_ex` kernel.
        if !self.batched_gemm_warmed.swap(true, Ordering::SeqCst) {
            let warm_a = self.from_cpu(
                &[1.0f32, 2.0, 3.0, 4.0],
                &Shape::from_slice(&[2, 2]),
                DType::F32,
            )?;
            let warm_b = self.from_cpu(
                &[1.0f32, 2.0, 3.0, 4.0],
                &Shape::from_slice(&[2, 2]),
                DType::F32,
            )?;
            let wa: Vec<&dyn BackendStorage> = vec![warm_a.as_ref(), warm_a.as_ref()];
            let wb: Vec<&dyn BackendStorage> = vec![warm_b.as_ref(), warm_b.as_ref()];
            let _ = self.matmul_batched(&wa, &wb, &Shape::from_slice(&[2, 2]));
        }

        let a0 = as_rocm(a[0])?;
        let b0 = as_rocm(b[0])?;
        let a_dims = a0.shape().dims();
        let b_dims = b0.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape("matmul_batched expects 2-D inputs".into()));
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
                "expected out [{m},{n}], got {:?}",
                out_shape.dims()
            )));
        }
        let dtype_out = DType {
            arith: a0.dtype.arith,
            storage: DTypeStorage::Native,
        };
        for i in 1..batch {
            let ai = as_rocm(a[i])?;
            let bi = as_rocm(b[i])?;
            if ai.shape().dims() != &[m, k] || bi.shape().dims() != &[k, n] {
                return Err(Error::Shape(
                    "matmul_batched: all batch entries must share shape [m,k]/[k,n]".into(),
                ));
            }
            if ai.dtype != a0.dtype || bi.dtype != b0.dtype {
                return Err(Error::Shape(
                    "matmul_batched: all batch entries must share dtype".into(),
                ));
            }
        }

        let stride_a = (m * k) as usize;
        let stride_b = (k * n) as usize;
        let stride_d = (m * n) as usize;

        // Pack inputs into contiguous device buffers (device-to-device copies).
        let a_packed = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[batch * stride_a]),
            dtype_out.clone(),
            &self.allocator,
            self.ordinal,
        )?;
        let b_packed = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[batch * stride_b]),
            dtype_out.clone(),
            &self.allocator,
            self.ordinal,
        )?;
        let d_packed = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[batch * stride_d]),
            dtype_out.clone(),
            &self.allocator,
            self.ordinal,
        )?;
        let stream = self.active_stream();
        let handle = self.get_rocblas_handle()?;
        // Bind rocBLAS to the same stream the D2D input copies use, so the copies
        unsafe {
            let _ = rocblas_set_stream(handle, stream);
        }
        let a_elem_size = a0.dtype.arith.byte_size();
        let b_elem_size = b0.dtype.arith.byte_size();

        for i in 0..batch {
            let ai = as_rocm(a[i])?;
            let bi = as_rocm(b[i])?;
            check_hip("matmul_batched: hipMemcpyDtoD a", unsafe {
                hipMemcpyAsync(
                    (a_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_a * a_elem_size),
                    ai.device_ptr.unwrap() as *mut c_void,
                    ai.bytes,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            })?;
            check_hip("matmul_batched: hipMemcpyDtoD b", unsafe {
                hipMemcpyAsync(
                    (b_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_b * b_elem_size),
                    bi.device_ptr.unwrap() as *mut c_void,
                    bi.bytes,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            })?;
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let a_type = arith_to_rocblas_dtype(a0.dtype.arith);
        let b_type = arith_to_rocblas_dtype(b0.dtype.arith);
        let out_type = arith_to_rocblas_dtype(dtype_out.arith);
        let compute_type = arith_to_compute_dtype(dtype_out.arith);

        // Look up the offline-tuned solution index for this shape/dtype, so
        // matmul_batched routes through the same autotune table as matmul.
        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, dtype_out.arith);

        // Row-major C[M,N] = A[M,K] @ B[K,N] via rocBLAS column-major recipe [see: `matmul`]
        unsafe {
            let status = rocblas_gemm_strided_batched_ex(
                handle,
                RocblasOperation::None,
                RocblasOperation::None,
                n as RocblasInt,
                m as RocblasInt,
                k as RocblasInt,
                &alpha as *const f32 as *const c_void,
                b_packed.device_ptr.unwrap() as *const c_void,
                b_type,
                n as RocblasInt,
                (stride_b) as i64,
                a_packed.device_ptr.unwrap() as *const c_void,
                a_type,
                k as RocblasInt,
                (stride_a) as i64,
                &beta as *const f32 as *const c_void,
                d_packed.device_ptr.unwrap() as *const c_void,
                out_type,
                n as RocblasInt,
                (stride_d) as i64,
                d_packed.device_ptr.unwrap() as *mut c_void,
                out_type,
                n as RocblasInt,
                (stride_d) as i64,
                batch as RocblasInt,
                compute_type,
                // Wire `lookup_solution_index` to `algo` via `select_gemm_algo`
                // so rocBLAS honors the autotuned solution index. [see: `select_gemm_algo`, `standard`]
                select_gemm_algo(solution_index),
                solution_index as RocblasInt,
                ROCBLAS_GEMM_FLAGS_NONE,
            );
            // NOTE: do NOT reset the handle to the null (default) stream here.
            // Every eager GEMM dispatch re-binds the handle to `active_stream()`
            // before its call (P0-17 fix), so leaving the binding as-is is
            // correct and avoids the footgun where a later GEMM that omits the
            // set_stream would silently dispatch on the default stream.
            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_strided_batched_ex failed with status {status}"
                )));
            }
        }

        // Split the packed device-resident result into per-batch storages via
        // device-to-device strided copies — no D2H/H2D round-trip. [see: `active_stream`]
        let d_element_size = dtype_out.arith.byte_size();
        let mut out = Vec::with_capacity(batch);
        for i in 0..batch {
            let batch_storage = RocmStorage::alloc_gpu(
                out_shape,
                dtype_out.clone(),
                &self.allocator,
                self.ordinal,
            )?;
            check_hip("matmul_batched: hipMemcpyDtoD d split", unsafe {
                hipMemcpyAsync(
                    batch_storage.device_ptr.unwrap() as *mut c_void,
                    (d_packed.device_ptr.unwrap() as *mut c_void)
                        .add(i * stride_d * d_element_size),
                    stride_d * d_element_size,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            })?;
            out.push(Box::new(batch_storage) as Box<dyn BackendStorage>);
        }
        // No trailing sync: the unpacked D2D copies share the GEMM's stream,
        // so callers that read the outputs observe completed data through
        // stream order (or their own sync). The previous per-call drain
        // serialized every batched QKV projection.
        Ok(out)
    }

    /// Pinned-memory + async host→device upload for the per-token decode hot path. [see: `data`, `hipMemcpy`, `RocmDevice::from_cpu`, `Vec`]
    pub fn copy_from_host_async(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let pinned = RocmPinnedBuffer::<f32>::from_slice(data)?;
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }
        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;
        // Use a pooled compute stream so the copy can overlap with other queued
        let stream = self.active_stream();
        let res = unsafe {
            hipMemcpyAsync(
                dev_ptr_void,
                pinned.as_ptr() as *const c_void,
                storage.bytes,
                HipMemcpyKind::HostToDevice,
                stream,
            )
        };
        if res != hipSuccess {
            // Return the buffer to the caching allocator (not bare hipFree) so
            // pool accounting stays correct under repeated errors. [see: `RocmCachingAllocator::free`]
            self.allocator.free(dev_ptr_void, storage.bytes);
            return Err(Error::Backend(format!(
                "hipMemcpyAsync(H2D) failed with error code {}",
                res
            )));
        }
        // Retain the pin until the next device-wide synchronize — never free a
        // page-locked source while a stream-ordered copy may still read it.
        // This matches the correct pattern in `upload_from_host_stream_ordered`.
        if let Ok(mut pins) = self.retained_pins.lock() {
            pins.push(pinned);
        }
        Ok(Box::new(storage))
    }

    /// Stream-ordered f32 H2D upload that does NOT synchronize before returning.
    ///
    /// Pins the host data, allocates device storage, and issues `hipMemcpyAsync`
    /// on a dedicated **transfer stream** (distinct from the compute stream the
    /// GEMMs use). After the copy it records a reusable completion event into
    /// [`RocmDevice::upload_event`]; [`active_stream`] fences the compute stream
    /// on that event, so a weight prefetch enqueued here can overlap the prior
    /// decode-step GEMM on the compute stream instead of serializing behind it
    /// (SPEED-ROC-1). The pinned source is retained in [`RocmDevice::retained_pins`]
    /// so it outlives the enqueue; [`synchronize`] releases every retained pin
    /// after the device completes the copies.
    ///
    /// [`synchronize`]: RocmDevice::synchronize
    /// [`active_stream`]: RocmDevice::active_stream
    pub fn upload_from_host_stream_ordered(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let pinned = RocmPinnedBuffer::<f32>::from_slice(data)?;
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }
        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;
        // Distinct transfer stream (pool index 1) so H2D copy-engine work runs
        // concurrently with compute on the active stream (pool index 0).
        let xfer = self
            .get_stream_from_pool(1)
            .or_else(|| self.get_stream_from_pool(0))
            .unwrap_or(std::ptr::null_mut());
        let status = unsafe {
            hipMemcpyAsync(
                dev_ptr_void,
                pinned.as_ptr() as *const c_void,
                storage.bytes,
                HipMemcpyKind::HostToDevice,
                xfer,
            )
        };
        // On failure the async copy was never enqueued, so the pin is safe to
        // drop immediately; return the device buffer to the caching allocator.
        if status != hipSuccess {
            self.allocator.free(dev_ptr_void, storage.bytes);
            return Err(Error::Backend(format!(
                "hipMemcpyAsync(H2D, stream-ordered) failed with error code {status}"
            )));
        }
        // Record a completion event on the transfer stream so the next
        // compute-dispatch on the active stream (via `active_stream()`) can
        // wait on it. Reuse a single event across uploads.
        let event = {
            let mut guard = self
                .upload_event
                .lock()
                .map_err(|_| Error::Backend("upload_event mutex poisoned".into()))?;
            match *guard {
                Some(e) => e,
                None => {
                    let mut ev: *mut c_void = std::ptr::null_mut();
                    let r = unsafe { crate::hipEventCreate(&mut ev) };
                    if r != hipSuccess {
                        self.allocator.free(dev_ptr_void, storage.bytes);
                        return Err(Error::Backend(format!(
                            "hipEventCreate failed with code {r}"
                        )));
                    }
                    *guard = Some(ev);
                    ev
                }
            }
        };
        let r = unsafe { crate::hipEventRecord(event, xfer) };
        if r != hipSuccess {
            self.allocator.free(dev_ptr_void, storage.bytes);
            return Err(Error::Backend(format!(
                "hipEventRecord failed with code {r}"
            )));
        }
        // Retain the pin until the next device-wide synchronize (never free a
        // page-locked source while a stream-ordered copy may still read it).
        if let Ok(mut pins) = self.retained_pins.lock() {
            pins.push(pinned);
        }
        Ok(Box::new(storage))
    }

    /// Like [`RocmDevice::copy_from_host_async`] but uploads from a caller-owned [see: `hipHostMalloc`]
    pub fn upload_from_pinned(
        &self,
        src: &RocmPinnedBuffer<f32>,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }
        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;
        let stream = self.active_stream();
        check_hip("hipMemcpyAsync(H2D)", unsafe {
            hipMemcpyAsync(
                dev_ptr_void,
                src.as_ptr() as *const c_void,
                storage.bytes,
                HipMemcpyKind::HostToDevice,
                stream,
            )
        })?;
        check_hip("hipStreamSynchronize(H2D)", unsafe {
            hipStreamSynchronize(stream)
        })?;
        Ok(Box::new(storage))
    }

    /// In-memory D2D transpose of a contiguous `[a, b]` f32 tensor into a fresh
    /// `[b, a]` device buffer via `grim_transpose_2d_f32`.
    ///
    /// This replaces the DtoH + transpose + H2D round trip that the host
    /// fallback performs for F32 weights on GPU: the input storage is read and
    /// written entirely in device memory, so transposing a weight that is
    /// already resident on the device costs no host transfer.
    pub fn transpose_f32_2d(
        &self,
        src: &dyn BackendStorage,
        a: usize,
        b: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let src_s = src
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("transpose_f32_2d: src is not RocmStorage".into()))?;
        let out_shape = Shape::new(vec![b, a]);
        let storage =
            RocmStorage::alloc_gpu(&out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut in_ptr = dev_ptr(src_s)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let total = a
            .checked_mul(b)
            .ok_or_else(|| Error::Backend("transpose_f32_2d: a*b overflow".into()))?;
        let (grid, block) = linear_launch(total);
        let mut a_i = a as i32;
        let mut b_i = b as i32;
        let stream = self.launch_compute_kernel(
            "grim_transpose_2d_f32",
            grid,
            block,
            &mut [
                arg(&mut in_ptr),
                arg(&mut out_ptr),
                arg(&mut a_i),
                arg(&mut b_i),
            ],
        )?;
        if self.active_capture_stream().is_none() {
            check_hip("hipStreamSynchronize(transpose_2d_f32)", unsafe {
                hipStreamSynchronize(stream)
            })?;
        }
        Ok(Box::new(storage))
    }

    /// Upload f32 data into HIP managed memory. Managed allocations remain
    /// valid to ordinary ROCm kernels while HIP may migrate cold pages to
    /// system RAM, providing a transparent overflow tier for large weights.
    pub fn from_cpu_managed(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        RocmStorage::copy_from_host_managed(data, shape, dtype, &self.allocator, self.ordinal)
            .map(|storage| Box::new(storage) as Box<dyn BackendStorage>)
    }

    /// Pinned-memory + async device→host download for the per-token decode hot path. [see: `hipMemcpy`, `Vec<f32>`]
    pub fn read_to_host_async(&self, storage: &dyn BackendStorage) -> Result<Vec<f32>> {
        let elem_count = storage.shape().elem_count();
        let mut pinned = RocmPinnedBuffer::<f32>::alloc(elem_count)?;
        let dev_ptr_void = match storage.as_any().downcast_ref::<RocmStorage>() {
            Some(rs) => match rs.device_ptr {
                Some(p) => p as *mut c_void,
                None => {
                    return Err(Error::Backend(
                        "RocmStorage has no valid device pointer".into(),
                    ));
                }
            },
            None => {
                return Err(Error::Backend(
                    "read_to_host_async only supports RocmStorage".into(),
                ));
            }
        };
        let stream = self.active_stream();
        check_hip("hipMemcpyAsync(D2H)", unsafe {
            hipMemcpyAsync(
                pinned.as_mut_ptr() as *mut c_void,
                dev_ptr_void,
                elem_count * std::mem::size_of::<f32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        })?;
        // MAJ-3 fix: synchronize the stream before reading pinned memory — the
        check_hip("hipStreamSynchronize(D2H)", unsafe {
            hipStreamSynchronize(stream)
        })?;
        let mut out = vec![0.0f32; elem_count];
        out.copy_from_slice(pinned.as_slice());
        Ok(out)
    }

    /// Same as [`RocmDevice::read_to_host_async`] but downloads into a caller-owned [see: `elem_count`]
    pub fn read_into_pinned(
        &self,
        storage: &dyn BackendStorage,
        dst: &mut RocmPinnedBuffer<f32>,
    ) -> Result<()> {
        let elem_count = storage.shape().elem_count();
        if dst.len() != elem_count {
            *dst = RocmPinnedBuffer::<f32>::alloc(elem_count)?;
        }
        let dev_ptr_void = match storage.as_any().downcast_ref::<RocmStorage>() {
            Some(rs) => match rs.device_ptr {
                Some(p) => p as *mut c_void,
                None => {
                    return Err(Error::Backend(
                        "RocmStorage has no valid device pointer".into(),
                    ));
                }
            },
            None => {
                return Err(Error::Backend(
                    "read_into_pinned only supports RocmStorage".into(),
                ));
            }
        };
        let stream = self.active_stream();
        check_hip("hipMemcpyAsync(D2H)", unsafe {
            hipMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                dev_ptr_void,
                elem_count * std::mem::size_of::<f32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        })?;
        check_hip("hipStreamSynchronize(D2H)", unsafe {
            hipStreamSynchronize(stream)
        })?;
        Ok(())
    }
}

impl Drop for RocmDevice {
    fn drop(&mut self) {
        // Drain any in-flight kernels on the pooled streams before recycling or
        // freeing. Pin the device first (P1-7 discipline): hipDeviceSynchronize
        // targets the calling thread's current device, which may not be
        // `self.ordinal` if another device's Drop ran on this thread.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        unsafe {
            let _ = hipDeviceSynchronize();
        }
        // Return all pooled buffers to the driver before the allocator Arc is dropped,
        self.allocator.empty_cache();
        // Unload every cached HIP module (they were loaded exactly once per
        if let Ok(mut cache) = self.module_cache.lock() {
            for (_, (module, _func)) in cache.drain() {
                unsafe {
                    let _ = hipModuleUnload(module);
                }
            }
        }
        if let Ok(mut pool) = self.stream_pool.lock() {
            for stream in pool.drain(..) {
                unsafe {
                    let _ = hipStreamDestroy(stream);
                }
            }
        }
        // Destroy the reusable upload-completion event (SPEED-ROC-1 overlap).
        if let Ok(mut guard) = self.upload_event.lock() {
            if let Some(ev) = guard.take() {
                if !ev.is_null() {
                    unsafe {
                        let _ = crate::hipEventDestroy(ev);
                    }
                }
            }
        }
        if let Ok(mut cache) = self.handle_cache.lock() {
            if let Some(handle) = cache.take() {
                unsafe {
                    let _ = rocblas_destroy_handle(handle);
                }
            }
        }
        // Destroy the capture stream (owned for the device lifetime). By now the
        if let Some(stream) = self.capture_stream.write().unwrap_or_else(|e| e.into_inner()).take() {
            unsafe {
                let _ = hipStreamDestroy(stream);
            }
        }
    }
}

impl RocmDevice {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn wavefront_size(&self) -> WavefrontSize {
        self.props.wavefront_size
    }

    pub fn xnack_enabled(&self) -> bool {
        self.props.xnack_enabled
    }

    pub fn props(&self) -> &RocmDeviceProps {
        &self.props
    }

    /// Retrieve a stream from the persistent pool (round-robin checkouts).
    pub fn get_stream_from_pool(&self, idx: usize) -> Option<*mut c_void> {
        let pool = self.stream_pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.is_empty() {
            None
        } else {
            Some(pool[idx % pool.len()])
        }
    }

    /// Liveness check for a single ordinal without constructing a full [see: `Ok(true)`, `hipGetDeviceCount`, `ordinal + 1`, `Ok(false)`]
    pub fn probe_one(ordinal: usize) -> Result<bool> {
        if let Ok(s) = std::env::var("GRIM_ROCM_ORDINAL_OVERRIDE") {
            if let Ok(n) = s.parse::<usize>() {
                return Ok(n == ordinal);
            }
        }
        let mut count: i32 = 0;
        let count_status = unsafe { hipGetDeviceCount(&mut count) };
        if count_status != hipSuccess {
            return Err(Error::Backend(format!(
                "hipGetDeviceCount failed with code {count_status}"
            )));
        }
        Ok((count as usize) > ordinal)
    }

    pub fn probe() -> Result<Vec<RocmDevice>> {
        if let Ok(s) = std::env::var("GRIM_ROCM_ORDINAL_OVERRIDE") {
            if let Ok(n) = s.parse::<usize>() {
                return Ok(vec![RocmDevice::new(n)]);
            }
        }
        // Attempt to enumerate via HIP.
        let mut count: i32 = 0;
        let count_status = unsafe { hipGetDeviceCount(&mut count) };
        if count_status != hipSuccess {
            // If the HIP runtime isn't present or call fails, return empty vec
            return Ok(vec![]);
        }
        let mut devices = Vec::with_capacity(count as usize);
        for i in 0..count {
            devices.push(RocmDevice::new(i as usize));
        }
        Ok(devices)
    }

    pub fn get_rocblas_handle(&self) -> Result<RocblasHandle> {
        let mut cache = self
            .handle_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(h) = *cache {
            return Ok(h);
        }

        // Pin the calling thread's device before creating the rocBLAS handle —
        // rocBLAS inherits whatever device is current, and a handle created on
        // the wrong device produces silent wrong-answer GEMMs.
        // [P1-3 fix: DeviceGuard::set before rocblas_create_handle.]
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        unsafe {
            let mut h: RocblasHandle = RocblasHandle(std::ptr::null_mut());
            let mut status = rocblas_create_handle(&mut h);
            if status != rocblas_status_success {
                // rocBLAS hipMallocs a 32-128 MiB internal workspace when the
                // handle is created; on small-VRAM parts or under high allocator pressure,
                // that fails with status 5 (rocblas_status_memory_error).
                // Drain allocator memory pool and synchronize device before retrying.
                let _ = crate::hipDeviceSynchronize();
                self.allocator.empty_cache();
                h = RocblasHandle(std::ptr::null_mut());
                status = rocblas_create_handle(&mut h);
            }
            if status == rocblas_status_success {
                *cache = Some(h);
                return Ok(h);
            }

            // Fallback: If rocBLAS workspace creation fails due to VRAM memory pressure,
            // return a zeroed handle — our custom HIP fused GEMM kernels handle matmuls
            // without requiring rocBLAS internal workspace allocations.
            //
            // IMPORTANT: callers MUST null-check the handle before use. Passing a
            // null RocblasHandle to rocblas_gemm_ex will SIGSEGV. The existing
            // matmul_op/matmul_with_solution call sites already guard with
            // `!h.0.is_null()` before falling back to the WMMA/custom path.
            // [P1-6: documented null-handle risk; existing callers already guard.]
            if status == 5 {
                eprintln!(
                    "[grim-backend-rocm] rocblas_create_handle failed with memory error (status 5); \
                     falling back to custom HIP fused GEMM kernels"
                );
                let fallback_handle = RocblasHandle(std::ptr::null_mut());
                *cache = Some(fallback_handle);
                return Ok(fallback_handle);
            }

            let (free_b, total_b) = {
                let mut free_mem: usize = 0;
                let mut total_mem: usize = 0;
                let s = hipMemGetInfo(&mut free_mem, &mut total_mem);
                if s == hipSuccess {
                    (free_mem, total_mem)
                } else {
                    (0, 0)
                }
            };
            Err(Error::Backend(format!(
                "rocblas_create_handle failed with status {status} \
                 (5 = rocblas_status_memory_error; device {} has {} MiB free of {} MiB — \
                 rocBLAS needs a 32-128 MiB internal workspace, lower it with \
                 ROCBLAS_DEVICE_MEMORY_SIZE)",
                self.ordinal,
                free_b / (1024 * 1024),
                total_b / (1024 * 1024)
            )))
        }
    }

    /// Device-side element-wise sum of multiple F32 storages via the
    /// `grim_all_reduce_accum` kernel. Each input must have the same shape.
    /// The result is written into the pre-allocated `out_ptr` (a `RocmStorage`
    /// device pointer u64). No host round-trip occurs. [see: `grim_all_reduce_accum`]
    pub(crate) fn device_accumulate_f32(
        &self,
        inputs: &[&dyn BackendStorage],
        out_ptr: u64,
    ) -> Result<()> {
        let total = inputs[0].shape().elem_count();

        // Collect host-side device pointers, upload them as a device array.
        let host_ptrs: Vec<u64> = inputs
            .iter()
            .map(|&s| as_rocm(s).and_then(dev_ptr))
            .collect::<Result<Vec<_>>>()?;
        let ptr_bytes: Vec<u8> = host_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let ptr_storage = RocmStorage::copy_from_host_raw_bytes(
            &ptr_bytes,
            &Shape::from_slice(&[host_ptrs.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let ptrs_dev = dev_ptr(&ptr_storage)?;

        let (grid, block) = linear_launch(total);
        let mut out_ptr = out_ptr;
        let mut ptrs_dev = ptrs_dev;
        let mut n_inputs = inputs.len() as i32;
        let mut n_elements = total as i32;
        self.launch_compute_kernel(
            "grim_all_reduce_accum",
            grid,
            block,
            &mut [
                arg(&mut out_ptr),
                arg(&mut ptrs_dev),
                arg(&mut n_inputs),
                arg(&mut n_elements),
            ],
        )?;
        Ok(())
    }
}

impl BackendDevice for RocmDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: zeros");

        // `hipMemset` zeroes bytes, which is only correct when the dtype's zero
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;

        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }

        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;

        // If a graph-capture session is active, record an async memset on the
        // capture stream; otherwise enqueue async on the active stream so
        // zeroing stays stream-ordered instead of blocking the host (the old
        // default-stream hipMemset was a device-wide serialization point).
        let res = match self.active_capture_stream() {
            Some(capture_stream) => unsafe {
                hipMemsetAsync(dev_ptr_void, 0, storage.bytes, capture_stream)
            },
            None => unsafe { hipMemsetAsync(dev_ptr_void, 0, storage.bytes, self.active_stream()) },
        };

        if res != hipSuccess {
            // Free on failure
            if storage.device_ptr.is_some() {
                let ptr_void = storage.device_ptr.unwrap() as *mut c_void;
                unsafe {
                    _ = hipFree(ptr_void);
                }
            }
            return Err(Error::Backend(format!(
                "hipMemset for zeros failed with error code {}",
                res
            )));
        }

        Ok(Box::new(storage))
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: from_cpu");

        RocmStorage::copy_from_host(data, shape, dtype, &self.allocator, self.ordinal)
            .map(|s| Box::new(s) as Box<dyn BackendStorage>)
    }

    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        RocmStorage::copy_from_host_raw_bytes(data, shape, dtype, &self.allocator, self.ordinal)
            .map(|s| Box::new(s) as Box<dyn BackendStorage>)
    }

    fn alloc_storage(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        RocmStorage::alloc_gpu(shape, dtype, &self.allocator, self.ordinal)
            .map(|s| Box::new(s) as Box<dyn BackendStorage>)
    }

    fn copy_slice_into(
        &self,
        dst: &dyn BackendStorage,
        src: &dyn BackendStorage,
        dst_elem_offset: usize,
        count: usize,
    ) -> Result<()> {
        let dst_s = as_rocm(dst)?;
        let src_s = as_rocm(src)?;
        if !dst_s.device_ptr_is_valid() || !src_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "copy_slice_into: inputs lack a valid device pointer".into(),
            ));
        }
        if dst_elem_offset + count > dst_s.shape().elem_count() {
            return Err(Error::Shape(format!(
                "copy_slice_into: overflow (dst_elem_offset={dst_elem_offset} + count={count} > dst elems={}",
                dst_s.shape().elem_count()
            )));
        }
        let bytes = count * std::mem::size_of::<f32>();
        let dst_ptr = unsafe {
            (dst_s.device_ptr_u64().unwrap() as *mut c_void)
                .add(dst_elem_offset * std::mem::size_of::<f32>())
        };
        let src_ptr = src_s.device_ptr_u64().unwrap() as *const c_void;
        check_hip("copy_slice_into: hipMemcpyAsync D2D", unsafe {
            hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                bytes,
                HipMemcpyKind::DeviceToDevice,
                self.active_stream(),
            )
        })?;
        Ok(())
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, crate::autotune::GemmOp::Other)
    }

    fn matmul_with_solution(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        solution_index: i32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: matmul_with_solution");

        // For matmul on GPU, both inputs must be RocmStorage (or we need to copy them to the device first)
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => {
                return Err(Error::Backend(
                    "matmul_with_solution: input a is not RocmStorage".into(),
                ));
            }
        };

        let b_storage = match b.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => {
                return Err(Error::Backend(
                    "matmul_with_solution: input b is not RocmStorage".into(),
                ));
            }
        };

        if !a_storage.device_ptr_is_valid() || !b_storage.device_ptr_is_valid() {
            return Err(Error::Backend(
                "matmul_with_solution: inputs must have valid GPU device pointers".into(),
            ));
        }

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape(
                "matmul_with_solution expects 2-D inputs".into(),
            ));
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
                "expected out [{m},{n}], got {:?}",
                out_shape.dims()
            )));
        }

        // Allocate output GPU storage with the actual input precision
        let dtype_out = DType {
            arith: a_storage.dtype.arith,
            storage: DTypeStorage::Native,
        };
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

        // Shape-indexed GEMM dispatch lookup (Tensile-inspired layout resolution)
        let _tile_config = lookup_gemm_config(m, n, k, self.props.wavefront_size);
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}",
            m, n, k, _tile_config, self.props.wavefront_size
        );

        // Get rocBLAS handle and execute gemm_ex with the provided solution_index
        let handle = self.get_rocblas_handle()?;
        // Bind the rocBLAS handle to the active stream so GEMM executes on the
        // correct stream and the returned ComputeHandle synchronizes correctly.
        // [P0-17 fix: previously missing — caused sync-lie and split-K race.]
        let _ = unsafe { rocblas_set_stream(handle, self.active_stream()) };

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;

        let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
        let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
        let out_ptr_void = out_storage.device_ptr.unwrap() as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is

        let use_gemm_ex = cfg!(feature = "rocm-aiter")
            || self.gpu_target == "gfx90a"
            || self.gpu_target == "gfx942";

        unsafe {
            let status = if use_gemm_ex
                || dtype_out.arith == ArithType::F16
                || dtype_out.arith == ArithType::BF16
            {
                let a_type = arith_to_rocblas_dtype(a_storage.dtype.arith);
                let b_type = arith_to_rocblas_dtype(b_storage.dtype.arith);
                let out_type = arith_to_rocblas_dtype(dtype_out.arith);
                let compute_type = arith_to_compute_dtype(dtype_out.arith);
                let alpha_ptr = &alpha as *const f32 as *const c_void;
                let beta_ptr = &beta as *const f32 as *const c_void;
                rocblas_gemm_ex(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    n as RocblasInt,
                    a_ptr_void,
                    a_type,
                    k as RocblasInt,
                    beta_ptr,
                    out_ptr_void,
                    out_type,
                    n as RocblasInt,
                    out_ptr_void,
                    out_type,
                    n as RocblasInt,
                    compute_type,
                    // Wire `lookup_solution_index` to `algo` so rocBLAS actually [see: `select_gemm_algo(0)`, `standard`]
                    select_gemm_algo(solution_index),
                    solution_index as RocblasInt,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            } else {
                rocblas_sgemm(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    &alpha,
                    b_ptr_void as *const f32,
                    n as RocblasInt,
                    a_ptr_void as *const f32,
                    k as RocblasInt,
                    &beta,
                    out_ptr_void as *mut f32,
                    n as RocblasInt,
                )
            };

            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas matmul_with_solution execution failed with error status {}",
                    status
                )));
            }
        };

        let compute_handle = Box::new(RocmHandle::new(Some(self.active_stream())));
        Ok((Box::new(out_storage), compute_handle))
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "add: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;

        let mut out_ptr = dev_ptr(&storage)?;
        let mut a_ptr = dev_ptr(a_s)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_add",
            grid,
            block,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "mul: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut a_ptr = dev_ptr(a_s)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_mul",
            grid,
            block,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "mul_scalar: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let mut s = scalar;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_mul_scalar",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut s), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn add_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "add_scalar: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let mut s = scalar;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_add_scalar",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut s), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn sub_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.add_scalar(x, -scalar, out)
    }

    fn div_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.mul_scalar(x, 1.0 / scalar, out)
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "sqrt: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_sqrt",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "recip: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_recip",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_s = as_rocm(gate)?;
        let up_s = as_rocm(up)?;
        if !gate_s.device_ptr_is_valid() || !up_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut gate_ptr = dev_ptr(gate_s)?;
        let mut up_ptr = dev_ptr(up_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_silu_mul",
            grid,
            block,
            &mut [
                arg(&mut gate_ptr),
                arg(&mut up_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SwiGLU backward: `(df, de) = silu_mul_backward(e, g, dw)`.
    /// `df` = gradient w.r.t. `g` (up), `de` = gradient w.r.t. `e` (gate).
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
        let e_s = as_rocm(e)?;
        let g_s = as_rocm(g)?;
        let dw_s = as_rocm(dw)?;
        if !e_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() || !dw_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul_backward: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out_shape.elem_count();
        let df_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let de_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut e_ptr = dev_ptr(e_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut dw_ptr = dev_ptr(dw_s)?;
        let mut df_ptr = dev_ptr(&df_storage)?;
        let mut de_ptr = dev_ptr(&de_storage)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_silu_mul_backward",
            grid,
            block,
            &mut [
                arg(&mut e_ptr),
                arg(&mut g_ptr),
                arg(&mut dw_ptr),
                arg(&mut df_ptr),
                arg(&mut de_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(df_storage),
            Box::new(de_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn fused_adamw_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let m_s = as_rocm(m)?;
        let v_s = as_rocm(v)?;
        if !p_s.device_ptr_is_valid()
            || !g_s.device_ptr_is_valid()
            || !m_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_adamw_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut m_ptr = dev_ptr(m_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_adamw_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut m_ptr),
                arg(&mut v_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn fused_lion_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        exp_avg: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let exp_s = as_rocm(exp_avg)?;
        if !p_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() || !exp_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_lion_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut exp_ptr = dev_ptr(exp_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut weight_decay = weight_decay;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_lion_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut exp_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut weight_decay),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn fused_madam_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        gamma: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let m_s = as_rocm(m)?;
        let v_s = as_rocm(v)?;
        if !p_s.device_ptr_is_valid()
            || !g_s.device_ptr_is_valid()
            || !m_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_madam_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut m_ptr = dev_ptr(m_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut gamma = gamma;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_madam_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut m_ptr),
                arg(&mut v_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut gamma),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(weight)?;
        if !x_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rms_norm: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("rms_norm: empty input".into()));
        }
        let row_len = *out.dims().last().unwrap();
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        // grim_rms_norm is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_rms_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut out_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn rmsnorm_backward(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        out_grad: &dyn BackendStorage,
        eps: f32,
        x_shape: &Shape,
        w_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(weight)?;
        let g_s = as_rocm(out_grad)?;
        if !x_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rmsnorm_backward: missing device pointer".into(),
            ));
        }
        let row_len = *w_shape.dims().last().unwrap_or(&1);
        let total = x_shape.elem_count();
        let dx_storage =
            RocmStorage::alloc_gpu(x_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let dw_storage =
            RocmStorage::alloc_gpu(w_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut dx_ptr = dev_ptr(&dx_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;

        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_rmsnorm_backward",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut g_ptr),
                arg(&mut dx_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(dx_storage),
            Box::new(dw_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn rope_backward(
        &self,
        out_grad: &dyn BackendStorage,
        cos: &dyn BackendStorage,
        sin: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = as_rocm(out_grad)?;
        let c_s = as_rocm(cos)?;
        let s_s = as_rocm(sin)?;
        if !g_s.device_ptr_is_valid() || !c_s.device_ptr_is_valid() || !s_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope_backward: missing device pointer".into(),
            ));
        }
        let half_dim = cos.shape().elem_count();
        let head_dim = half_dim * 2;
        let total_tokens = out_shape.elem_count() / head_dim.max(1);
        let dx_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut c_ptr = dev_ptr(c_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut dx_ptr = dev_ptr(&dx_storage)?;
        let mut half_dim_i = half_dim as i32;
        let mut total_tokens_i = total_tokens as i32;

        let total_pairs = (total_tokens * head_dim) / 2;
        let (grid, block) = linear_launch(total_pairs);
        self.launch_compute_kernel(
            "grim_rope_backward",
            grid,
            block,
            &mut [
                arg(&mut g_ptr),
                arg(&mut c_ptr),
                arg(&mut s_ptr),
                arg(&mut dx_ptr),
                arg(&mut half_dim_i),
                arg(&mut total_tokens_i),
            ],
        )?;
        Ok((
            Box::new(dx_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn softmax_backward(
        &self,
        out_grad: &dyn BackendStorage,
        softmax_out: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = as_rocm(out_grad)?;
        let s_s = as_rocm(softmax_out)?;
        if !g_s.device_ptr_is_valid() || !s_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "softmax_backward: missing device pointer".into(),
            ));
        }
        let row_len = *out_shape.dims().last().unwrap_or(&1);
        let total = out_shape.elem_count();
        let dx_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut dx_ptr = dev_ptr(&dx_storage)?;
        let mut row_len_i = row_len as i32;
        let mut total_i = total as i32;

        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_softmax_backward",
            grid,
            block,
            &mut [
                arg(&mut g_ptr),
                arg(&mut s_ptr),
                arg(&mut dx_ptr),
                arg(&mut row_len_i),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(dx_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// P3 (4th fused backward kernel): scatter-add embedding gradient on
    /// device — `dweight[token_ids[t], :] += out_grad[t, :]`. Token ids are
    /// uploaded as a small U32 buffer; dweight is zero-filled first, then
    /// atomically accumulated.
    fn embedding_backward(
        &self,
        out_grad: &dyn BackendStorage,
        token_ids: &[u32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = as_rocm(out_grad)?;
        if !g_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "embedding_backward: missing device pointer".into(),
            ));
        }
        let num_tokens = token_ids.len();
        if num_tokens == 0 || hidden_dim == 0 || vocab_size == 0 {
            return Err(Error::Shape(
                "embedding_backward: empty vocab/hidden/tokens".into(),
            ));
        }

        let dw_shape = Shape::new(vec![vocab_size, hidden_dim]);
        let dw_storage =
            RocmStorage::alloc_gpu(&dw_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let ids_shape = Shape::new(vec![num_tokens]);
        let ids_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let ids_storage = RocmStorage::copy_from_host_raw_bytes(
            &ids_bytes,
            &ids_shape,
            DType::U32,
            &self.allocator,
            self.ordinal,
        )?;

        let mut g_ptr = dev_ptr(g_s)?;
        let mut ids_ptr = dev_ptr(&ids_storage)?;
        let mut dw_ptr = dev_ptr(&dw_storage)?;
        let mut dw_total_i = (vocab_size * hidden_dim) as i32;
        let mut num_tokens_i = num_tokens as i32;
        let mut hidden_dim_i = hidden_dim as i32;
        let mut vocab_size_i = vocab_size as i32;

        // 1) zero-fill dweight.
        let (grid, block) = linear_launch(vocab_size * hidden_dim);
        self.launch_compute_kernel(
            "grim_zero_f32",
            grid,
            block,
            &mut [arg(&mut dw_ptr), arg(&mut dw_total_i)],
        )?;
        // 2) atomic scatter-add.
        let (grid, block) = linear_launch(num_tokens * hidden_dim);
        self.launch_compute_kernel(
            "grim_embedding_backward",
            grid,
            block,
            &mut [
                arg(&mut g_ptr),
                arg(&mut ids_ptr),
                arg(&mut dw_ptr),
                arg(&mut num_tokens_i),
                arg(&mut hidden_dim_i),
                arg(&mut vocab_size_i),
            ],
        )?;
        Ok((
            Box::new(dw_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "softmax: input lacks a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("softmax: empty input".into()));
        }
        let row_len = *x_dims.last().unwrap();
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut row_len_i = row_len as i32;
        let mut total_i = total as i32;
        // grim_softmax is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_softmax",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut out_ptr),
                arg(&mut row_len_i),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let w_s = match as_rocm(weight) {
            Ok(s) => s,
            Err(_) => {
                return Err(Error::Backend(
                    "embedding: weight is not RocmStorage".into(),
                ));
            }
        };
        if !w_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "embedding: weight lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out.dims();
        if out_dims.len() < 2 {
            return Err(Error::Shape("embedding: out must be [n, dim]".into()));
        }
        let n = out_dims[0];
        let dim = out_dims[1];
        if n != indices.len() {
            return Err(Error::Shape(format!(
                "embedding: indices len {} != out leading dim {}",
                indices.len(),
                n
            )));
        }

        // materialize() already dequantizes Q8_0 to F32 before returning
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut idx_ptr = upload_device_buffer(indices)?;
        let mut dim_i = dim as i32;
        let mut total_i = total as i32;
        let (grid, block) = linear_launch(total);
        let stream = self.launch_compute_kernel(
            "grim_embedding",
            grid,
            block,
            &mut [
                arg(&mut w_ptr),
                arg(&mut out_ptr),
                arg(&mut idx_ptr),
                arg(&mut dim_i),
                arg(&mut total_i),
            ],
        )?;
        // The fused kernel reads idx_ptr from the GPU. Free stream-ordered so
        // the release happens after the kernel's reads; this is also
        // graph-capturable (the capture path previously leaked the buffer).
        unsafe {
            let free_stream = stream
                .as_ref()
                .map(|_| self.active_stream())
                .unwrap_or(std::ptr::null_mut());
            let _ = hipFreeAsync(idx_ptr, free_stream);
        }
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let (out, _handle) = self.quantize_on_device(x, format)?;
        Ok(out)
    }

    fn advise(
        &self,
        storage: &dyn BackendStorage,
        advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: advise");

        let rocm_storage = storage
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("advise: storage is not RocmStorage".into()))?;

        let dev_ptr = match rocm_storage.device_ptr {
            Some(ptr) => ptr as *const c_void,
            None => return Ok(()), // Unallocated or CPU-side: no-op
        };

        // Correctness Gate: Probe XNACK. If disabled, pageable unified memory
        // migrations fail — and there is nothing useful to substitute: the old
        // fallback issued a whole-tensor self-copy on the null stream, which
        // was a no-op for data but a device-wide serialization point.
        if !self.props.xnack_enabled {
            return Ok(());
        }

        let raw_advice = match advice {
            grim_tensor::MemAdvice::ReadMostly => {
                crate::device::handles::HIP_MEM_ADVISE_SET_READ_MOSTLY
            }
            grim_tensor::MemAdvice::PreferredLocation { device_id: _ } => {
                crate::device::handles::HIP_MEM_ADVISE_SET_PREFERRED_LOCATION
            }
            grim_tensor::MemAdvice::AccessedBy { device_id: _ } => {
                crate::device::handles::HIP_MEM_ADVISE_SET_ACCESSED_BY
            }
            grim_tensor::MemAdvice::CoarseGrain => {
                crate::device::handles::HIP_MEM_ADVISE_SET_COARSE_GRAIN
            }
            grim_tensor::MemAdvice::FineGrain => {
                crate::device::handles::HIP_MEM_ADVISE_UNSET_COARSE_GRAIN
            }
            // OS-level hints (madvise) are ignored on the GPU memory space
            _ => return Ok(()),
        };

        unsafe {
            check_hip(
                "hipMemAdvise",
                hipMemAdvise(dev_ptr, rocm_storage.bytes, raw_advice, self.ordinal as i32),
            )?;
        }
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
        let quant_format =
            crate::fusion::KvQuantFormat::from_legacy_quant_bits(quant_bits as u8, true);
        self.kv_dequant_attention_impl(
            q,
            k_tensor,
            k_scales,
            v_tensor,
            v_scales,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            quant_format,
            quant_bits,
            out_shape,
        )
    }

    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        conv_state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(weight)?;
        let st_s = as_rocm(conv_state)?;
        if !x_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid() || !st_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "short_conv1d: inputs lack valid device ptr".into(),
            ));
        }
        let total = out_shape.elem_count();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut b_ptr = match bias {
            Some(b) => dev_ptr(as_rocm(b)?)?,
            None => 0u64,
        };
        let mut st_ptr = dev_ptr(st_s)?;

        let dims = out_shape.dims();
        let mut batch = dims[0] as i32;
        let mut channels = dims[2] as i32;
        let mut k_size = (w_s.bytes / (channels as usize * 4)) as i32;

        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_short_conv1d_causal_step",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut b_ptr),
                arg(&mut st_ptr),
                arg(&mut out_ptr),
                arg(&mut batch),
                arg(&mut channels),
                arg(&mut k_size),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn kda_gated_delta_rule_step(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        beta: &dyn BackendStorage,
        a_gate: &dyn BackendStorage,
        recurrent_state: &dyn BackendStorage,
        d_k: usize,
        d_v: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let beta_s = as_rocm(beta)?;
        let gate_s = as_rocm(a_gate)?;
        let s_s = as_rocm(recurrent_state)?;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut q_ptr = dev_ptr(q_s)?;
        let mut k_ptr = dev_ptr(k_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut beta_ptr = dev_ptr(beta_s)?;
        let mut gate_ptr = dev_ptr(gate_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut dk_i = d_k as i32;
        let mut dv_i = d_v as i32;

        let (grid, block) = linear_launch(d_v);
        self.launch_compute_kernel(
            "grim_kda_gated_delta_rule_step",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut k_ptr),
                arg(&mut v_ptr),
                arg(&mut beta_ptr),
                arg(&mut gate_ptr),
                arg(&mut s_ptr),
                arg(&mut out_ptr),
                arg(&mut dk_i),
                arg(&mut dv_i),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn mla_q_kv_norm_split(
        &self,
        q_raw: &dyn BackendStorage,
        kv_raw: &dyn BackendStorage,
        q_norm_w: &dyn BackendStorage,
        kv_norm_w: &dyn BackendStorage,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_dim: usize,
        eps: f32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let q_s = as_rocm(q_raw)?;
        let kv_s = as_rocm(kv_raw)?;
        let qw_s = as_rocm(q_norm_w)?;
        let kvw_s = as_rocm(kv_norm_w)?;

        let q_nope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_nope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let q_rope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_rope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let kv_nope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_nope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let kv_rope_st = RocmStorage::alloc_gpu(
            &Shape::new(vec![qk_rope_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut q_ptr = dev_ptr(q_s)?;
        let mut kv_ptr = dev_ptr(kv_s)?;
        let mut qw_ptr = dev_ptr(qw_s)?;
        let mut kvw_ptr = dev_ptr(kvw_s)?;
        let mut q_nope_ptr = dev_ptr(&q_nope_st)?;
        let mut q_rope_ptr = dev_ptr(&q_rope_st)?;
        let mut kv_nope_ptr = dev_ptr(&kv_nope_st)?;
        let mut kv_rope_ptr = dev_ptr(&kv_rope_st)?;
        let mut nope_i = qk_nope_dim as i32;
        let mut rope_i = qk_rope_dim as i32;
        let mut v_i = v_dim as i32;
        let mut eps_f = eps;

        let total = qk_nope_dim + qk_rope_dim;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_mla_q_kv_norm_split",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut kv_ptr),
                arg(&mut qw_ptr),
                arg(&mut kvw_ptr),
                arg(&mut q_nope_ptr),
                arg(&mut q_rope_ptr),
                arg(&mut kv_nope_ptr),
                arg(&mut kv_rope_ptr),
                arg(&mut nope_i),
                arg(&mut rope_i),
                arg(&mut v_i),
                arg(&mut eps_f),
            ],
        )?;

        Ok((
            Box::new(q_nope_st),
            Box::new(q_rope_st),
            Box::new(kv_nope_st),
            Box::new(kv_rope_st),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        FUSED_FORWARD_DISPATCH_STATS
            .attempts
            .fetch_add(1, Ordering::Relaxed);
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return self.matmul(a, b_packed, out_shape),
        };
        let b_storage = match b_packed.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return self.matmul(a, b_packed, out_shape),
        };

        let dims = out_shape.dims();
        let (m, n) = match dims.len() {
            2 => (dims[0], dims[1]),
            _ => (
                dims[..dims.len() - 1].iter().product(),
                dims[dims.len() - 1],
            ),
        };
        let k = a_storage.shape().dims().last().copied().unwrap_or(0);

        let out_storage = RocmStorage::alloc_gpu(
            out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;

        use grim_tensor::{BlockDtype, FloatPackScheme, KQuantScheme};
        match b_storage.dtype().storage {
            DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                self.launch_fused_dequant_gemm_q4k(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q5K) => {
                self.launch_fused_dequant_gemm_q5k(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q6K) => {
                self.launch_fused_dequant_gemm_q6k(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q2K) => {
                self.launch_fused_dequant_gemm_q2k(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q3K) => {
                self.launch_fused_dequant_gemm_q3k(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => {
                self.launch_fused_dequant_gemm_iq2xxs(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XS) => {
                self.launch_fused_dequant_gemm_iq2xs(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2S) => {
                self.launch_fused_dequant_gemm_iq2s(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => {
                self.launch_fused_dequant_gemm_iq3xxs(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3S) => {
                self.launch_fused_dequant_gemm_iq3s(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4NL) => {
                self.launch_fused_dequant_gemm_iq4nl(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4XS) => {
                self.launch_fused_dequant_gemm_iq4xs(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                // Q8_0 uses the fused dequant+GEMM kernel (34-byte blocks → F32), matching
                // the other KQuant schemes rather than falling back to dequant+matmul.
                self.launch_fused_dequant_gemm_q8_0(a_storage, b_storage, &out_storage, m, n, k)?;
            }
            DTypeStorage::Block(BlockDtype::Fp8)
            | DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                // gfx1200+ uses MFMA for FP8 throughput; other architectures use scalar.
                if self.gpu_target.starts_with("gfx12") {
                    self.launch_fused_dequant_gemm_fp8_mfma(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_fp8(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::FloatPack(FloatPackScheme::MxFp4) => {
                // The MXFP4 kernel reads one E8M0 exponent per 32-element block
                // (block_idx = (col*K+k)/32) and expects B_codes / B_exps as
                // separate device buffers. Weight tensors carry the
                // length-prefixed framing [u64 codes_len][codes][u64
                // exps_len][exps]; both segment lengths are derivable from the
                // shape. The framed segments are addressed IN PLACE via
                // interior pointers — the weight blob is immutable, so the
                // former per-call split (two allocations + two synchronous
                // DtoD copies of the whole weight per GEMM per layer per token)
                // was pure overhead. Alignment: device allocations are
                // >=256B-aligned and codes_len = N*K/2 is a multiple of 16 for
                // any K multiple of 32, so both interior pointers stay
                // 16B-aligned for vectorized kernel loads. Legacy codes-only
                // buffers (no framing) keep the _b_scales / dummy exponent
                // path and pass B through unchanged.
                let elems = k * n;
                let codes_len = elems / 2;
                let exps_len = elems.div_ceil(32);
                let framed_len = 16 + codes_len + exps_len;
                let base = b_storage
                    .device_ptr_u64()
                    .ok_or_else(|| Error::Backend("mxfp4 gemm: b has no device ptr".into()))?;

                // Keep any transient exponent storage alive until after the
                // kernel launch below (single-stream ordering makes pooled
                // reuse safe once this binding drops).
                let exps_storage: Option<RocmStorage>;
                let (codes_ptr, exps_ptr): (u64, u64) = if b_storage.bytes == framed_len {
                    exps_storage = None;
                    (base + 8, base + 16 + codes_len as u64)
                } else if !_b_scales.is_empty() {
                    // Caller-supplied f32 E8M0 byte values as exponents.
                    let exps_u8: Vec<u8> = _b_scales
                        .iter()
                        .map(|s| s.round().clamp(0.0, 255.0) as u8)
                        .collect();
                    let storage = RocmStorage::copy_from_host_raw_bytes(
                        &exps_u8,
                        &Shape::new(vec![exps_u8.len()]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let ptr = storage
                        .device_ptr_u64()
                        .ok_or_else(|| Error::Backend("mxfp4 gemm: exps upload failed".into()))?;
                    exps_storage = Some(storage);
                    (base, ptr)
                } else {
                    // Legacy/empty path: zeroed dummy exponents.
                    let storage = RocmStorage::alloc_gpu(
                        &Shape::new(vec![exps_len.max(1)]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let ptr = storage
                        .device_ptr_u64()
                        .ok_or_else(|| Error::Backend("mxfp4 gemm: dummy exps failed".into()))?;
                    exps_storage = Some(storage);
                    (base, ptr)
                };

                let use_fused = self
                    .mxfp4_fused_dequant_gemm_enabled
                    .load(Ordering::Relaxed);
                if use_fused {
                    self.launch_fused_dequant_gemm_mxfp4(
                        a_storage,
                        codes_ptr,
                        exps_ptr,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_mxfp4_gemm_tiled(
                        a_storage,
                        codes_ptr,
                        exps_ptr,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
                // Keep any transient exponent storage alive until the kernel
                // launch(es) above are enqueued on the active stream. Pooled
                // reuse is only safe once the transitive storage drops.
                drop(exps_storage);
            }
            DTypeStorage::FloatPack(FloatPackScheme::MxFp8) => {
                let dummy_exps = RocmStorage::alloc_gpu(
                    &Shape::new(vec![(k * n).max(32) / 32]),
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::Native,
                    },
                    &self.allocator,
                    self.ordinal,
                )?;
                self.launch_fused_dequant_gemm_mxfp8(
                    a_storage,
                    b_storage,
                    &dummy_exps,
                    &out_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::ResidualPacked(cfg) => {
                // Generic variable-bitwidth packed + residual layout (WI-C / WI-T8): [see: `grim_fused_dequant_gemm_f16`, `enabled`]
                // Lock-free enabled check via AtomicBool shadow. [see: `fused_dequant_gemm_enabled`, `set_fused_dequant_gemm_enabled`]
                if !self.fused_dequant_gemm_enabled.load(Ordering::Relaxed) {
                    FUSED_FORWARD_DISPATCH_STATS
                        .fallback_calls
                        .fetch_add(1, Ordering::Relaxed);
                    return self.matmul(a, b_packed, out_shape);
                }
                let out_f32 =
                    RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
                let residuals = grim_tensor::QuantizedMatmulBackwardResiduals::from_provenance(
                    &b_storage.provenance(),
                );
                let provenance = b_storage.provenance();
                let (primary_bytes, outlier_indices, outlier_values) = match provenance {
                    grim_tensor::QuantProvenance::WithResiduals {
                        primary_scale_bytes,
                        outlier_indices,
                        outlier_values_bits,
                        ..
                    } => (
                        primary_scale_bytes,
                        outlier_indices,
                        outlier_values_bits
                            .into_iter()
                            .map(f32::from_bits)
                            .collect::<Vec<_>>(),
                    ),
                    _ => (Vec::new(), Vec::new(), Vec::new()),
                };
                let scales_storage = if primary_bytes.is_empty() {
                    None
                } else {
                    Some(RocmStorage::copy_from_host_raw_bytes(
                        &primary_bytes,
                        &Shape::from_slice(&[primary_bytes.len()]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?)
                };
                let index_bytes: Vec<u8> = outlier_indices
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                let value_bytes: Vec<u8> = outlier_values
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                let indices_storage = if outlier_indices.is_empty() {
                    None
                } else {
                    Some(RocmStorage::copy_from_host_raw_bytes(
                        &index_bytes,
                        &Shape::from_slice(&[outlier_indices.len()]),
                        DType {
                            arith: ArithType::U32,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?)
                };
                let values_storage = if outlier_values.is_empty() {
                    None
                } else {
                    Some(RocmStorage::copy_from_host_raw_bytes(
                        &value_bytes,
                        &Shape::from_slice(&[outlier_values.len()]),
                        DType::F32,
                        &self.allocator,
                        self.ordinal,
                    )?)
                };
                let scale_ptr = scales_storage
                    .as_ref()
                    .and_then(|s| s.device_ptr)
                    .map(|p| p as *const c_void)
                    .unwrap_or(std::ptr::null());
                let index_ptr = indices_storage
                    .as_ref()
                    .and_then(|s| s.device_ptr)
                    .map(|p| p as *const c_void)
                    .unwrap_or(std::ptr::null());
                let value_ptr = values_storage
                    .as_ref()
                    .and_then(|s| s.device_ptr)
                    .map(|p| p as *const c_void)
                    .unwrap_or(std::ptr::null());
                let stream = self.launch_fused_dequant_gemm_f16(
                    a_storage,
                    b_storage,
                    scale_ptr,
                    &out_f32,
                    m,
                    n,
                    k,
                    cfg.bpw,
                    residuals.outlier_count,
                    index_ptr,
                    value_ptr,
                    residuals.backup1_bpw,
                    residuals.backup1_codes_offset,
                    residuals.backup1_scale_offset,
                    residuals.backup2_bpw,
                    residuals.backup2_codes_offset,
                    residuals.backup2_scale_offset,
                )?;
                FUSED_FORWARD_DISPATCH_STATS
                    .kernel_calls
                    .fetch_add(1, Ordering::Relaxed);
                FUSED_FORWARD_DISPATCH_STATS
                    .last_backup2_bpw
                    .store(residuals.backup2_bpw as usize, Ordering::Relaxed);
                FUSED_FORWARD_DISPATCH_STATS
                    .last_backup2_codes_offset
                    .store(residuals.backup2_codes_offset, Ordering::Relaxed);
                FUSED_FORWARD_DISPATCH_STATS
                    .last_backup2_scale_offset
                    .store(residuals.backup2_scale_offset, Ordering::Relaxed);
                let handle: Box<dyn ComputeHandle> = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_f32), handle));
            }
            _ => {
                return self.matmul(a, b_packed, out_shape);
            }
        }

        let handle: Box<dyn ComputeHandle> = Box::new(ReadyHandle);
        Ok((Box::new(out_storage), handle))
    }

    /// WI-F5-close: fused dequant backward dispatch (the lattice point [see: `grim-autograd::matmul_backward`]
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
        FUSED_BACKWARD_DISPATCH_STATS
            .attempts
            .fetch_add(1, Ordering::Relaxed);

        // Both operands must already be ROCm-resident for the kernel to run
        let dy_storage = match dy.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => {
                return Err(Error::Backend(
                    "quantized_matmul_backward_dx: dy not ROCm-resident; CPU fallback expected"
                        .into(),
                ));
            }
        };
        let b_storage = match b_packed.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend(
                "quantized_matmul_backward_dx: b_packed not ROCm-resident; CPU fallback expected"
                    .into(),
            )),
        };

        // Extract residual metadata or use defaults when absent.
        let outlier_count = residuals.map(|r| r.outlier_count).unwrap_or(0);
        let outlier_indices_ptr = residuals
            .and_then(|r| {
                if r.outlier_count > 0 {
                    Some(r.outlier_indices_ptr)
                } else {
                    None
                }
            })
            .unwrap_or(std::ptr::null());
        let outlier_values_ptr = residuals
            .and_then(|r| {
                if r.outlier_count > 0 {
                    Some(r.outlier_values_ptr)
                } else {
                    None
                }
            })
            .unwrap_or(std::ptr::null());
        let backup1_bpw = residuals.map(|r| r.backup1_bpw).unwrap_or(0);
        let backup1_codes_offset = residuals.map(|r| r.backup1_codes_offset).unwrap_or(0);
        let backup1_scale_offset = residuals.map(|r| r.backup1_scale_offset).unwrap_or(0);
        let backup2_bpw = residuals.map(|r| r.backup2_bpw).unwrap_or(0);
        let backup2_codes_offset = residuals.map(|r| r.backup2_codes_offset).unwrap_or(0);
        let backup2_scale_offset = residuals.map(|r| r.backup2_scale_offset).unwrap_or(0);

        // Allocate the dX output buffer (f32 row-major [M, K]).
        let dx_storage = match out_shape.dims() {
            &[mm, kk] if mm == m && kk == k => RocmStorage::alloc_gpu(
                out_shape,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?,
            other => {
                return Err(Error::Shape(format!(
                    "quantized_matmul_backward_dx: out_shape must be [{m},{k}], got {:?}",
                    other
                )));
            }
        };

        // Pack scales into a temporary ROCm buffer so the kernel can reach them.
        // `grim_fused_dequant_backward_gemm_f16` (and the matching forward
        // `grim_fused_dequant_gemm_f16`) read scales as `const unsigned char*`
        // and divide by 255.0f to recover a [0, 1] per-column scale. The
        // caller (`grim-autograd::matmul_backward`) hands us float scales —
        // for formats like `ResidualPacked`/`GroupInt` these are arbitrary
        // positive reals (`with_quant_scales(vec![2.5f32, 2.5f32])`). Uploading
        // them as raw F32 and casting to `unsigned char*` would have the kernel
        // read the IEEE-754 exponent byte as the scale, producing garbage. We
        // quantize F32→U8 here so the kernel contract holds: `scale_byte / 255.0f`
        // is a normalized factor. Values > 1.0 saturate at 255; this is the
        // intended contract for the WI-C residual-packed fallback path.
        let scales_storage = if b_scales.is_empty() {
            None
        } else {
            let byte_scales: Vec<u8> = b_scales
                .iter()
                .map(|&s| {
                    let n = s.clamp(0.0f32, 1.0f32) * 255.0f32;
                    n.round().clamp(0.0f32, 255.0f32) as u8
                })
                .collect();
            Some(RocmStorage::copy_from_host_raw_bytes(
                &byte_scales,
                &Shape::new(vec![byte_scales.len()]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?)
        };
        let b_scales_ptr = match &scales_storage {
            Some(s) => match s.device_ptr {
                Some(raw) => raw as *const c_void,
                None => {
                    return Err(Error::Backend(
                        "quantized_matmul_backward_dx: scales alloc missing gpu ptr".into(),
                    ));
                }
            },
            None => std::ptr::null(),
        };

        // Call the actual kernel based on quantization storage type (not bpw,
        use grim_tensor::{BlockDtype, FloatPackScheme, KQuantScheme};
        match b_storage.dtype().storage {
            DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                self.launch_fused_dequant_backward_gemm_q4k(
                    dy_storage,
                    b_storage,
                    b_scales_ptr,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                self.launch_fused_dequant_backward_gemm_q8_0(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q5K) => {
                self.launch_fused_dequant_backward_gemm_q5k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q6K) => {
                self.launch_fused_dequant_backward_gemm_q6k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q2K) => {
                self.launch_fused_dequant_backward_gemm_q2k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q3K) => {
                self.launch_fused_dequant_backward_gemm_q3k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => {
                self.launch_fused_dequant_backward_gemm_iq2xxs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XS) => {
                self.launch_fused_dequant_backward_gemm_iq2xs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2S) => {
                self.launch_fused_dequant_backward_gemm_iq2s(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => {
                self.launch_fused_dequant_backward_gemm_iq3xxs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3S) => {
                self.launch_fused_dequant_backward_gemm_iq3s(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4NL) => {
                self.launch_fused_dequant_backward_gemm_iq4nl(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4XS) => {
                self.launch_fused_dequant_backward_gemm_iq4xs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::Block(BlockDtype::Fp8)
            | DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                if self.gpu_target.starts_with("gfx12") {
                    self.launch_fused_dequant_backward_gemm_fp8_mfma(
                        dy_storage,
                        b_storage,
                        &dx_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_backward_gemm_fp8(
                        dy_storage,
                        b_storage,
                        &dx_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::FloatPack(FloatPackScheme::MxFp4) => {
                let exps_storage = if !b_scales.is_empty() {
                    let exps_u8: Vec<u8> = b_scales
                        .iter()
                        .map(|s| s.round().clamp(0.0, 255.0) as u8)
                        .collect();
                    RocmStorage::copy_from_host_raw_bytes(
                        &exps_u8,
                        &Shape::new(vec![exps_u8.len()]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?
                } else {
                    RocmStorage::alloc_gpu(
                        &Shape::new(vec![(k * n).max(32) / 32]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?
                };
                self.launch_mxfp4_backward_gemm(
                    dy_storage,
                    b_storage,
                    &exps_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::ResidualPacked(cfg) => {
                // Mirror the forward `enabled` gate: when the fused backward path
                // is disabled, fall back to a standard matmul of dY against the
                // transposed dequantized B (same behavior as the forward fallback
                // at line ~2252). This fixes the asymmetry where the forward
                // dispatch honors `FusedDequantGemmConfig::enabled` but the
                // backward dispatch unconditionally calls the fused kernel.
                if !self.fused_dequant_gemm_enabled.load(Ordering::Relaxed) {
                    FUSED_BACKWARD_DISPATCH_STATS
                        .fallback_calls
                        .fetch_add(1, Ordering::Relaxed);
                    return self.matmul(dy, b_packed, out_shape);
                }
                self.launch_fused_dequant_backward_gemm_f16(
                    dy_storage,
                    b_storage,
                    b_scales_ptr,
                    &dx_storage,
                    m,
                    n,
                    k,
                    cfg.bpw,
                    outlier_count,
                    outlier_indices_ptr,
                    outlier_values_ptr,
                    backup1_bpw,
                    backup1_codes_offset,
                    backup1_scale_offset,
                    backup2_bpw,
                    backup2_codes_offset,
                    backup2_scale_offset,
                )?;
            }
            DTypeStorage::Native => {
                // Unquantized weights: no dequant needed. Use straight matmul
                // (dY @ B^T). This was previously falling through to the fused
                // backward kernel, which is incorrect for native FP16/BF16 weights.
                return self.matmul(dy, b_packed, out_shape);
            }
            _ => {
                self.launch_fused_dequant_backward_gemm_f16(
                    dy_storage,
                    b_storage,
                    b_scales_ptr,
                    &dx_storage,
                    m,
                    n,
                    k,
                    default_bpw,
                    outlier_count,
                    outlier_indices_ptr,
                    outlier_values_ptr,
                    backup1_bpw,
                    backup1_codes_offset,
                    backup1_scale_offset,
                    backup2_bpw,
                    backup2_codes_offset,
                    backup2_scale_offset,
                )?;
            }
        }

        FUSED_BACKWARD_DISPATCH_STATS
            .kernel_calls
            .fetch_add(1, Ordering::Relaxed);

        let handle: Box<dyn ComputeHandle> = Box::new(ReadyHandle);
        Ok((Box::new(dx_storage), handle))
    }

    fn mla_absorbed_decode(
        &self,
        q_absorbed: &dyn BackendStorage,
        q_rope: &dyn BackendStorage,
        kv_cache: &dyn BackendStorage,
        w_uv: Option<&dyn BackendStorage>,
        out: &dyn BackendStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        seq_len: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let q_abs = q_absorbed
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: q_absorbed is not RocmStorage".into()))?;
        let q_r = q_rope
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: q_rope is not RocmStorage".into()))?;
        let kv = kv_cache
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: kv_cache is not RocmStorage".into()))?;
        let o = out
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: out is not RocmStorage".into()))?;
        let w = w_uv
            .map(|s| {
                s.as_any()
                    .downcast_ref::<RocmStorage>()
                    .ok_or_else(|| Error::Backend("mla_absorbed_decode: w_uv is not RocmStorage".into()))
            })
            .transpose()?;
        self.launch_mla_absorbed_decode(
            q_abs, q_r, kv, w, o, num_heads, kv_lora_rank, qk_rope_dim, v_head_dim, seq_len,
        )?;
        Ok(Box::new(crate::device::handles::RocmHandle::new(Some(
            self.active_stream(),
        ))))
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
        // Compute host-side window_lo per-query position:
        // For full causal attention (window == None), window_lo = 0 for all queries.
        // For sliding-window (window == Some(w)), window_lo = max(0, abs_i - w + 1)
        // is constant across all query positions in this call only when seq_len == 1
        // (decode step). For prefill (seq_len > 1) the kernel receives the per-block
        // minimum window_lo = max(0, cache_offset - w + 1); each query thread then
        // computes its own abs_i = cache_offset + i and the KV range is
        // [window_lo_block, abs_i + 1). This is a conservative lower bound:
        // threads whose abs_i > cache_offset attend to slightly more KV than they
        // should, but the causal upper bound (abs_i + 1) is still enforced.
        // Laguna-S-2.1 uses seq_len == 1 for all decode calls so the bound is exact.
        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => {
                let abs_first = cache_offset as usize;
                abs_first.saturating_sub(w.saturating_sub(1)) as i32
            }
        };
        let config = {
            let out_dims = out_shape.dims();
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
                let num_heads = if head_dim > 0 {
                    hidden_dim / head_dim
                } else {
                    1
                };
                (seq_len, num_heads, head_dim)
            } else {
                return Err(Error::Shape(
                    "qkv_attention expects 2-D [seq_len, hidden_dim] or 3-D [seq_len, num_heads, head_dim] output shape".into(),
                ));
            };
            QkvAttentionFusionConfig {
                enabled: true,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len: seq_len,
                wavefront_size: self.props.wavefront_size as u32,
                quant_mode: QuantMode::Fp32,
            }
        };
        if !config.enabled {
            return Err(Error::Backend(
                "qkv_attention: kernel is gated (QkvAttentionFusionConfig.enabled=false)".into(),
            ));
        }

        // ─── structural validation ──────────────────────────────────────
        if config.num_heads == 0 || config.num_kv_heads == 0 || config.head_dim == 0 {
            return Err(Error::Shape(
                "qkv_attention: zero-sized num_heads / num_kv_heads / head_dim".into(),
            ));
        }
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "qkv_attention: num_heads ({}) must be a multiple of num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            )));
        }
        if config.head_dim > 256 {
            return Err(Error::Shape(format!(
                "qkv_attention Phase 2 supports head_dim <= 256 (got {})",
                config.head_dim
            )));
        }

        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid() || !k_s.device_ptr_is_valid() || !v_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "qkv_attention: inputs lack a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];

        // ─── allocate output + launch ────────────────────────────────────
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;
        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let v_ptr = dev_ptr(v_s)?;

        let mut max_ptr: u64 = 0;
        if let Some(m) = out_max {
            let m_s = as_rocm(m)?;
            max_ptr = dev_ptr(m_s)?;
        }
        let mut sum_ptr: u64 = 0;
        if let Some(s) = out_sum {
            let s_s = as_rocm(s)?;
            sum_ptr = dev_ptr(s_s)?;
        }

        let num_heads_i = config.num_heads as i32;
        let num_kv_heads_i = config.num_kv_heads as i32;
        let head_dim_i = config.head_dim as i32;
        let seq_len_i = seq_len as i32;
        let kv_seq_len_i = kv_seq_len as i32;
        let cache_offset_i = cache_offset as i32;
        let inv_sqrt_d: f32 = 1.0 / (config.head_dim as f32).sqrt();

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut nh = num_heads_i;
        let mut nkv = num_kv_heads_i;
        let mut hd = head_dim_i;
        let mut sl = seq_len_i;
        let mut ksl = kv_seq_len_i;
        let mut co = cache_offset_i;
        let mut isd = inv_sqrt_d;
        let mut wlo = window_lo_i;
        let mut oproj_ptr: u64 = 0;
        let mut odim: i32 = 0;
        let mut fuseo: i32 = 0;
        let mut alibi_ptr: u64 = 0;
        let mut has_alibi: i32 = 0;

        // Prior RoPE / cache ops were enqueued on the same stream, so stream
        // ordering already guarantees they complete before this kernel reads
        // q/k/v — no host sync needed (each removed sync stalls the whole
        // per-token pipeline).

        // Split-KV FlashDecoding acceleration for long-context single-token decode
        if seq_len == 1
            && kv_seq_len >= 1024
            && window.is_none()
            && out_max.is_none()
            && out_sum.is_none()
        {
            let num_splits = self.flash_decode_split_count(
                q_s,
                k_s,
                v_s,
                &storage,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                kv_seq_len,
            );
            let stream = self.launch_flash_decode(
                q_s,
                k_s,
                v_s,
                &storage,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                kv_seq_len,
                num_splits,
            )?;
            return Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))));
        }

        let mut launch_block_dim = launch.block_dim;
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: "grim_qkv_attention",
            gpu_arch: arch_leak,
            m: config.num_heads,
            n: config.head_dim,
            k: kv_seq_len.clamp(1, 1 << 16),
        };
        if let Ok(tuner) = self.autotuner.lock() {
            if let Some(cfg) = tuner.lookup(key) {
                if cfg.block_dim > 0 {
                    launch_block_dim.x = cfg.block_dim;
                }
            }
        }

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch_block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;

        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, oproj_ptr,
            odim, fuseo, alibi_ptr, has_alibi,
        );

        // No post-launch sync: the output storage is returned to the caller
        // and any readback (or same-stream reuse of pooled scratch) is
        // ordered by the single active stream. A sync here would serialize
        // the CPU against every attention of every layer of every token.

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    fn qkv_attention_alibi(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        alibi_slopes: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let slopes_s = as_rocm(alibi_slopes)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !slopes_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "qkv_attention_alibi: inputs lack a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_alibi: out_shape must be [seq, heads, head_dim]".into(),
            ));
        }
        let (seq_len, num_heads, head_dim) = (out_dims[0], out_dims[1], out_dims[2]);
        if slopes_s.shape().elem_count() < num_heads {
            return Err(Error::Shape(
                "qkv_attention_alibi: alibi_slopes must hold num_heads entries".into(),
            ));
        }

        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => (cache_offset as usize)
                .saturating_sub(w.saturating_sub(1)) as i32,
        };
        let config = QkvAttentionFusionConfig {
            enabled: true,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: seq_len,
            wavefront_size: self.props.wavefront_size as u32,
            quant_mode: QuantMode::Fp32,
        };
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;

        let mut qptr = dev_ptr(q_s)?;
        let mut kptr = dev_ptr(k_s)?;
        let mut vptr = dev_ptr(v_s)?;
        let mut optr = out_ptr;
        let mut max_ptr: u64 = 0;
        let mut sum_ptr: u64 = 0;
        let mut nh = num_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sl = seq_len as i32;
        let mut ksl = kv_seq_len as i32;
        let mut co = cache_offset as i32;
        let mut isd: f32 = 1.0 / (head_dim as f32).sqrt();
        let mut wlo = window_lo_i;
        let mut oproj_ptr: u64 = 0;
        let mut odim: i32 = 0;
        let mut fuseo: i32 = 0;
        let mut alibi_ptr = dev_ptr(slopes_s)?;
        let mut has_alibi: i32 = 1;

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;
        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, wlo,
            oproj_ptr, odim, fuseo, alibi_ptr, has_alibi,
        );
        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
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
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope: input lacks a valid device pointer".into(),
            ));
        }

        // Partial-rotary / YaRN path: dispatch to grim_rope_yarn which accepts a
        // pre-uploaded inv_freq[] buffer and handles both partial rotary_dim and
        // YaRN magnitude correction entirely on-GPU.
        if !cfg.is_plain() {
            return self.rope_launch_yarn(x_s, positions, cfg, out_shape);
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "RoPE expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let d = dim as i32;
        let half = d / 2;
        if positions.len() != s as usize {
            return Err(Error::Shape(
                "rope: positions length must match seq_len".into(),
            ));
        }

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut pos_ptr = upload_device_buffer(positions)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_rope",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
            ],
        )?;

        // pos_ptr is kernel input; release it stream-ordered after the launch
        // (graph-capturable, no host stall).
        unsafe {
            let _ = hipFreeAsync(pos_ptr, stream);
        }

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma_q: &dyn BackendStorage,
        gamma_k: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        self.fused_mxfp4_gemm_qk_norm_rope_kv(
            x,
            gamma_q,
            gamma_k,
            w_codes,
            w_exps,
            q_out,
            k_cache,
            v_cache,
            out_all,
            positions,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq,
            mscale,
            eps,
            max_seq_len,
        )
    }

    /// Broadcast 1-D bias tensor `[out_dim]` into 2-D storage `[batch, out_dim]` via `grim_broadcast_bias`.
    fn broadcast_bias(
        &self,
        bias: &dyn BackendStorage,
        batch: usize,
        out_dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let b_s = as_rocm(bias)?;
        if !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "broadcast_bias: bias lacks a valid device pointer".into(),
            ));
        }
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut batch_i = batch as i32;
        let mut out_dim_i = out_dim as i32;
        let total = batch * out_dim;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_broadcast_bias",
            grid,
            block,
            &mut [
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut batch_i),
                arg(&mut out_dim_i),
            ],
        )?;

        let _ = stream; // no post-launch sync: output consumed via stream order

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// In-place scale+bias epilogue on a `[batch, out_dim]` GEMM output via
    /// `grim_scale_bias_epilogue`. Plain rocBLAS has no epilogue-fusion API, so
    /// this standalone kernel is the required post-GEMM step for W8A8-style
    /// per-token × per-channel scaling. `a_scale`/`b_scale`/`bias` may be
    /// `None`; kernel treats absent scale as 1.0 and absent bias as 0.0.
    fn scale_bias_epilogue(
        &self,
        out: &dyn BackendStorage,
        a_scale: Option<&dyn BackendStorage>,
        b_scale: Option<&dyn BackendStorage>,
        bias: Option<&dyn BackendStorage>,
        batch: usize,
        out_dim: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let o_s = as_rocm(out)?;
        if !o_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "scale_bias_epilogue: out lacks a valid device pointer".into(),
            ));
        }
        // `_a_s` / `_b_s` / `_bt_s` hold borrows that keep the underlying storage
        // allocations alive until after the kernel launch; only the raw pointers are
        // forwarded into the kernel args.
        let (_a_s, a_ptr): (Option<&dyn BackendStorage>, Option<*mut c_void>) = match a_scale {
            Some(s) => {
                let s = as_rocm(s)?;
                (Some(s), Some(dev_ptr(s)? as *mut c_void))
            }
            None => (None, None),
        };
        let (_b_s, b_ptr): (Option<&dyn BackendStorage>, Option<*mut c_void>) = match b_scale {
            Some(s) => {
                let s = as_rocm(s)?;
                (Some(s), Some(dev_ptr(s)? as *mut c_void))
            }
            None => (None, None),
        };
        let (_bt_s, b_ptr2): (Option<&dyn BackendStorage>, Option<*mut c_void>) = match bias {
            Some(s) => {
                let s = as_rocm(s)?;
                (Some(s), Some(dev_ptr(s)? as *mut c_void))
            }
            None => (None, None),
        };

        let mut out_ptr = dev_ptr(o_s)?;
        let mut a_p = a_ptr.unwrap_or(std::ptr::null_mut());
        let mut b_p = b_ptr.unwrap_or(std::ptr::null_mut());
        let mut bpt = b_ptr2.unwrap_or(std::ptr::null_mut());
        let mut batch_i = batch as i32;
        let mut out_dim_i = out_dim as i32;
        let total = batch * out_dim;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_scale_bias_epilogue",
            grid,
            block,
            &mut [
                arg(&mut out_ptr),
                arg(&mut a_p),
                arg(&mut b_p),
                arg(&mut bpt),
                arg(&mut batch_i),
                arg(&mut out_dim_i),
            ],
        )?;

        let _ = stream; // no post-launch sync: output consumed via stream order

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
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
        let x_s = as_rocm(x)?;
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        let c_s = as_rocm(c)?;
        let d_s = as_rocm(d)?;
        let state_s = as_rocm(state)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_selective_scan(
            x_s,
            a_s,
            b_s,
            c_s,
            d_s,
            &state_s,
            &out_storage,
            batch,
            dim_dstate,
            dim_dinner,
            seq_len,
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
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
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_cross_attention(
            q_s,
            k_s,
            v_s,
            &out_storage,
            num_heads,
            head_dim,
            seq_len,
            kv_seq_len,
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
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
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(w)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let g_s = as_rocm(g)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_rwkv_time_mix(x_s, w_s, k_s, v_s, g_s, &out_storage, batch, dim, seq_len)?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
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
        let x_s = as_rocm(x)?;
        let k_s = as_rocm(k)?;
        let r_s = as_rocm(r)?;
        let v_s = as_rocm(v)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_rwkv_channel_mix(x_s, k_s, r_s, v_s, &out_storage, batch, dim)?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SCYTHE-2 WI-5: BackendDevice::all_reduce for RocmDevice. [see: `RowParallelLinear::forward`, `BackendDevice::all_reduce`]
    ///
    /// Performs the sum collective entirely on the ROCm device:
    /// - Cross-GPU: when an RCCL handle is attached and `num_gpus > 1`, uses
    ///   `RcclAllReduce::sum_gradients_device` for a device-side `ncclAllReduce`.
    /// - Intra-process: sums multiple partial shards on-device via the
    ///   `grim_all_reduce_accum` kernel (F32), avoiding the D2H/H2D round-trip.
    /// - Fallback: CPU fan-in for non-F32 dtypes or mismatched shard shapes.
    fn all_reduce(
        &self,
        inputs: &[&dyn grim_tensor::BackendStorage],
        op: &str,
    ) -> grim_tensor::error::Result<(
        Box<dyn grim_tensor::BackendStorage>,
        Box<dyn grim_tensor::backend::ComputeHandle>,
    )> {
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
        let stream = self.active_stream();
        let stream_u64 = stream as u64;
        let rccl = self.rccl.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let is_f32 = dtype.arith == ArithType::F32;
        // TP activations arrive as F16/BF16 single tensors per rank; routing
        // them through RCCL (instead of the old host round-trip) needs the
        // matching NCCL dtype.
        let rccl_dtype = match dtype.arith {
            ArithType::F32 => Some(crate::rccl::NCCL_FLOAT32),
            ArithType::F16 => Some(crate::rccl::NCCL_FLOAT16),
            ArithType::BF16 => Some(crate::rccl::NCCL_BFLOAT16),
            _ => None,
        };

        // ── Cross-GPU all-reduce via RCCL (device-side) ───────────────────
        // When an RCCL handle is attached and we have multiple GPUs, perform
        // the collective directly on device memory via ncclAllReduce.
        if let Some(rccl_handle) = &rccl {
            if rccl_handle.num_gpus > 1 && is_f32 {
                let out_storage =
                    RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                let out_ptr = dev_ptr(&out_storage)?;

                if inputs.len() == 1 {
                    // Single tensor: direct cross-GPU all-reduce.
                    let send_ptr = dev_ptr(as_rocm(inputs[0])?)?;
                    rccl_handle.sum_gradients_device(
                        send_ptr,
                        out_ptr,
                        total,
                        stream_u64,
                        self.ordinal,
                    )?;
                } else {
                    // Multiple shards: accumulate on-device first, then all-reduce.
                    let temp_storage =
                        RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                    let temp_ptr = dev_ptr(&temp_storage)?;
                    self.device_accumulate_f32(inputs, temp_ptr)?;
                    rccl_handle.sum_gradients_device(
                        temp_ptr,
                        out_ptr,
                        total,
                        stream_u64,
                        self.ordinal,
                    )?;
                }

                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
            }

            // F16/BF16 single-shard TP activations: all-reduce in the native
            // dtype — previously this fell through to a full D2H→CPU-sum→H2D
            // round trip per RowParallel layer per token.
            if rccl_handle.num_gpus > 1 && !is_f32 && inputs.len() == 1 {
                if let Some(nccl_dt) = rccl_dtype {
                    let out_storage = RocmStorage::alloc_gpu(
                        &shape,
                        dtype.clone(),
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let out_ptr = dev_ptr(&out_storage)?;
                    let send_ptr = dev_ptr(as_rocm(inputs[0])?)?;
                    rccl_handle.all_reduce_device(
                        send_ptr,
                        out_ptr,
                        total,
                        nccl_dt,
                        stream_u64,
                        self.ordinal,
                    )?;
                    return Ok((
                        Box::new(out_storage),
                        Box::new(RocmHandle::new(Some(stream))),
                    ));
                }
            }
        }

        // ── Intra-process device-side fan-in (no RCCL) ────────────────────
        // Avoid the CPU round-trip: sum partials directly on the GPU.
        if is_f32 && total > 0 {
            if inputs.len() == 1 {
                // Identity: device-to-device copy (no D2H + H2D round-trip).
                let bytes = total * crate::dtype_byte_size(&dtype);
                let out_storage =
                    RocmStorage::alloc_gpu(&shape, dtype.clone(), &self.allocator, self.ordinal)?;
                let src_ptr = dev_ptr(as_rocm(inputs[0])?)? as *const c_void;
                let dst_ptr = out_storage.device_ptr.unwrap() as *mut c_void;
                check_hip("hipMemcpyAsync(D2D) all_reduce", unsafe {
                    hipMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        bytes,
                        HipMemcpyKind::DeviceToDevice,
                        stream,
                    )
                })?;
                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
            }

            // Multi-input: device-side element-wise sum via grim_all_reduce_accum.
            let all_same = inputs.iter().all(|s| s.shape() == inputs[0].shape());
            if all_same {
                let out_storage =
                    RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                let out_ptr = dev_ptr(&out_storage)?;
                self.device_accumulate_f32(inputs, out_ptr)?;
                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
            }
        }

        // ── CPU fallback ───────────────────────────────────────────────────
        // Used for non-F32 dtypes or mismatched shard shapes where the device
        // accum kernel cannot apply.
        let mut acc = inputs[0].to_cpu_vec_f32()?;
        for other in &inputs[1..] {
            let v = other.to_cpu_vec_f32()?;
            if v.len() != acc.len() {
                return Err(Error::Backend(format!(
                    "all_reduce: input shape mismatch (first {} != other {})",
                    acc.len(),
                    v.len()
                )));
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        let storage = self.from_cpu(&acc, &shape, dtype)?;
        Ok((storage, Box::new(ReadyHandle)))
    }

    /// SCYTHE-2 WI-1/WI-6: WaveTune bilinear latency predictor for RocmDevice. [see: `(M, N, K)`, `2604.10187`]
    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        _placement: &grim_tensor::backend::ScythePlacement,
    ) -> f64 {
        // Peak TFLOPS from arch string (same table as capability_profiler.rs).
        let arch = detect_gpu_arch(self.ordinal as i32);
        let tflops_fp16: f64 = if arch.starts_with("gfx1100") {
            61.4
        } else if arch.starts_with("gfx1102") {
            26.0
        } else if arch.starts_with("gfx12") {
            80.0
        } else if arch.starts_with("gfx9") {
            190.0
        } else {
            20.0 // conservative unknown
        };
        // Apply dtype factor: FP8 is 2× FP16 on RDNA4+; FP32 is 0.5×.
        let dtype_factor = match dtype.arith {
            ArithType::F32 => 0.5,
            _ => 1.0,
        };
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let peak = tflops_fp16 * dtype_factor * 1e12; // FLOPS/s
        if peak <= 0.0 {
            return f64::INFINITY;
        }
        flops / peak * 1e3 // ms
    }

    /// SCYTHE-2 WI-6: CommFuse decomposed P2P fan-in override. [see: `crate::comm_fuse::comm_fuse_fan_in`, `to_cpu_vec_f32`]
    ///
    /// Assembles column-shard partials entirely on the ROCm device:
    /// - Device-side: places each partial at its column offset via row-by-row
    ///   `hipMemcpy` D2D, avoiding the D2H/H2D round-trip. When an RCCL handle
    ///   is attached and `num_gpus > 1`, a cross-GPU `ncclAllReduce` is issued
    ///   after assembly.
    /// - Fallback: CPU fan-in for non-F32 dtypes.
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
        let stream = self.active_stream();
        let stream_u64 = stream as u64;
        let rccl = self.rccl.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let elem_bytes = crate::dtype_byte_size(&dtype);

        // ── Device-side assembly + optional RCCL all-reduce ────────────────
        if is_f32 {
            let out_shape = Shape::from_slice(&[m, n_total]);
            let out_storage =
                RocmStorage::alloc_gpu(&out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
            let out_ptr_val = dev_ptr(&out_storage)?;
            let out_ptr_usize = out_ptr_val as usize;

            // Place each partial at its column offset, row by row (D2D memcpy).
            let mut col_offset = 0usize;
            for (storage, _placement) in partials {
                let s = as_rocm(*storage)?;
                let partial_ptr = dev_ptr(s)? as usize;
                let n_cols = s.shape().dims().get(1).copied().unwrap_or(0);
                for row in 0..m {
                    let src = (partial_ptr + row * n_cols * elem_bytes) as *const c_void;
                    let dst =
                        (out_ptr_usize + (row * n_total + col_offset) * elem_bytes) as *mut c_void;
                    check_hip("hipMemcpy(D2D) comm_fuse", unsafe {
                        crate::hipMemcpy(
                            dst,
                            src,
                            n_cols * elem_bytes,
                            crate::HipMemcpyKind::DeviceToDevice,
                        )
                    })?;
                }
                col_offset += n_cols;
            }

            // Optional RCCL cross-GPU all-reduce on the assembled buffer.
            let total_elems = m * n_total;
            if let Some(rccl_handle) = &rccl {
                if rccl_handle.num_gpus > 1 {
                    rccl_handle.sum_gradients_device(
                        out_ptr_val,
                        out_ptr_val,
                        total_elems,
                        stream_u64,
                        self.ordinal,
                    )?;
                }
            }

            return Ok(Box::new(out_storage));
        }

        // ── CPU fallback (non-F32 dtypes) ──────────────────────────────────
        let mut host_data: Vec<Vec<f32>> = Vec::with_capacity(partials.len());
        let mut n_cols_list: Vec<usize> = Vec::with_capacity(partials.len());
        for (storage, _placement) in partials {
            let data = storage.to_cpu_vec_f32()?;
            let n_cols = storage.shape().dims().get(1).copied().unwrap_or(0);
            host_data.push(data);
            n_cols_list.push(n_cols);
        }
        let slice_refs: Vec<(&[f32], usize)> = host_data
            .iter()
            .zip(n_cols_list.iter())
            .map(|(d, &nc)| (d.as_slice(), nc))
            .collect();

        let result =
            crate::kernels::comm_fuse::comm_fuse_fan_in(&slice_refs, m, n_total, &partials[0].1)?;

        let out_shape = Shape::from_slice(&[result.shape.0, result.shape.1]);
        let out_storage = self.from_cpu(&result.data, &out_shape, DType::F32)?;
        Ok(out_storage)
    }

    /// SCYTHE-2 WI-5: Paged attention override. [see: `crate::launch_paged_attention`, `grim_qkv_attention_paged`]
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
        // Sliding-window lower bound:
        //   None       -> 0 (full causal)
        //   Some(w)    -> max(0, cache_offset - (w - 1)). For decode (seq_len==1)
        //                 this is exact; for prefill it is the per-block conservative
        //                 lower bound (each query thread uses abs_i = cache_offset + i
        //                 and the causal upper bound abs_i + 1 is still enforced in
        //                 the kernel).
        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => {
                let abs_first = cache_offset as usize;
                abs_first.saturating_sub(w.saturating_sub(1)) as i32
            }
        };

        let q_s = as_rocm(q)?;
        let bt_s = as_rocm(block_tables)?;
        let k_s = as_rocm(k_pages)?;
        let v_s = as_rocm(v_pages)?;

        if !q_s.device_ptr_is_valid()
            || !bt_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "qkv_attention_paged: inputs lack a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let batch = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        crate::launch_paged_attention(
            self,
            q_s,
            bt_s,
            k_s,
            v_s,
            &mut storage,
            batch as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            max_blocks as u32,
            page_size as u32,
            kv_seq_len as u32,
            cache_offset,
            window_lo_i,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SCYTHE-2 WI-5: Tree attention override. [see: `crate::launch_tree_attention`, `grim_tree_attention`]
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
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let tp_s = as_rocm(tree_parents)?;

        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !tp_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "tree_attention: an input lacks a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let batch = out_dims[0];
        let one_plus_gamma = out_dims[1];
        let num_heads = out_dims[2];
        let head_dim = out_dims[3];

        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        crate::launch_tree_attention(
            self,
            q_s,
            k_s,
            v_s,
            tp_s,
            &mut storage,
            batch as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            one_plus_gamma as u32,
            kv_seq_len as u32,
            cache_offset,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}

// to `device::gemm_tuning` — see that module.
pub use crate::device::gemm_tuning::{
    GemmTileConfig, lookup_gemm_config, lookup_gemm_config_for_shape, lookup_solution_index,
};

// Re-exports that pulled up `pub use crate::graph_capture::*` etc. in [see: `pub use`]

impl RocmDevice {
    /// GPU-side YaRN / partial-rotary RoPE: computes `inv_freq[]` on the host,
    /// uploads it once per call, then dispatches `grim_rope_yarn` entirely on-device.
    ///
    /// # Contract
    /// - `x_s` must have a valid device pointer (caller checks `device_ptr_is_valid`).
    /// - `out_shape` must be `[B, S, D]` with `D == cfg.dim`.
    /// - `cfg.rotary_dim <= cfg.dim`; the non-rotary tail `[rotary_dim, D)` is copied verbatim.
    /// - Positions slice length must equal `S`.
    pub(crate) fn rope_launch_yarn(
        &self,
        x_s: &RocmStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dims = out_shape.dims();
        if dims.len() != 3 || dims[2] != cfg.dim {
            return Err(Error::Shape(format!(
                "rope_launch_yarn: expected [B,S,D={}], got {:?}",
                cfg.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let rotary_dim = cfg.rotary_dim.min(d);
        let rotary_half = rotary_dim / 2;
        let yarn = cfg.yarn;

        if positions.len() != s {
            return Err(Error::Shape(
                "rope_launch_yarn: positions length must match seq_len".into(),
            ));
        }

        // Build the YaRN-ramp-corrected inv_freq[] on the host — O(rotary_half) work,
        // negligible vs kernel launch overhead. This avoids storing per-layer buffers.
        let inv_freq: Vec<f32> = (0..rotary_half)
            .map(|i| {
                let freq = 1.0_f32 / cfg.base.powf((2 * i) as f32 / d as f32);
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

        // Upload positions and inv_freq to device-resident scratch buffers.
        // These are temporary allocations freed after the stream synchronises.
        let mut pos_ptr = upload_device_buffer(positions)?;
        let mut freq_ptr = upload_device_buffer(&inv_freq)?;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut b_i = b as i32;
        let mut s_i = s as i32;
        let mut d_i = d as i32;
        let mut rh_i = rotary_half as i32;
        let mut ms_f = mscale;

        // Launch grid covers max(b*s*rotary_half, b*s*copy_len) threads to
        // handle both the rotate pass and the verbatim-copy pass in one kernel launch.
        let copy_len = d - 2 * rotary_half;
        let total = b
            * s
            * rotary_half
                .max(if copy_len > 0 { copy_len } else { 0 })
                .max(1);
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_rope_yarn",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut freq_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut rh_i),
                arg(&mut ms_f),
            ],
        );

        // Free scratch device buffers stream-ordered (after the kernel's
        // reads); graph-capturable and no host stall.
        unsafe {
            let free_stream = stream
                .as_ref()
                .map(|_| self.active_stream())
                .unwrap_or(std::ptr::null_mut());
            let _ = hipFreeAsync(pos_ptr, free_stream);
            let _ = hipFreeAsync(freq_ptr, free_stream);
        }

        let stream = stream?;

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// WI 2.4.4-2c — dispatch `grim_decode_gemm_f16` and return the [see: `launch_compute_kernel`, `DecodeGemmConfig::enabled`]
    pub(crate) fn launch_decode_gemm_f16(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("decode_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("decode_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("decode_gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems = m * n;
        let grid_x = ((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        // Row-major strides in fp16 elements (not bytes).
        let stride_a = k; // A[M, K]
        let stride_b = n; // B[K, N]
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_decode_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Enqueues the JIT-compiled WMMA matrix-core GEMM kernel (WI-G).
    pub(crate) fn launch_wmma_gemm(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm: out has no device ptr".into()))?;

        let is_native_wmma = matches!(
            crate::quantization::gcn_arch(&self.gpu_target),
            crate::quantization::GcnArch::RDNA3
                | crate::quantization::GcnArch::RDNA4
                | crate::quantization::GcnArch::UDNA
        );

        let (grid_dim, block_dim) = if is_native_wmma {
            // Native rocWMMA path: 16x16 tile per block, 1 wavefront (32 threads for W32).
            let grid_x = ((n + 15) / 16) as u32;
            let grid_y = ((m + 15) / 16) as u32;
            (HipDim3::new(grid_x, grid_y, 1), HipDim3::new(32, 1, 1))
        } else {
            // Scalar fallback path: 1D grid of 256 threads.
            const BLOCK_SIZE: usize = 256;
            let total_elems = m * n;
            let grid_x = ((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
            (
                HipDim3::new(grid_x, 1, 1),
                HipDim3::new(BLOCK_SIZE as u32, 1, 1),
            )
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let stride_a = k; // A[M, K]
        let stride_b = n; // B[K, N]
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_wmma_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Launch the standalone FP8 GEMM kernel (gfx1200+ native MFMA,
    #[allow(dead_code)]
    pub(crate) fn launch_fp8_gemm_rdna4(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_gemm_rdna4: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_gemm_rdna4: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_gemm_rdna4: out has no device ptr".into()))?;

        // 16×16 tiling: one thread per output element, tile = 16 threads
        const TILE: usize = 16;
        let grid_x = ((n + TILE - 1) / TILE) as u32;
        let grid_y = ((m + TILE - 1) / TILE) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(TILE as u32, TILE as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fp8_gemm_rdna4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled fused dequantization GEMM kernel for [see: `b_storage`, `Storage::ResidualPacked`]
    pub(crate) fn launch_fused_dequant_gemm_f16(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        default_bpw: u8,
        outlier_count: usize,
        outlier_indices_ptr: *const c_void,
        outlier_values_ptr: *const c_void,
        backup_bpw: u8,
        backup_codes_offset: usize,
        backup_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: m*n overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_gemm: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        let stride_a = k; // A[M, K]
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sc = stride_c as i32;

        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;

        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;
        let mut b2_bpw = backup2_bpw as i32;
        let mut b2_codes_off = backup2_codes_offset as i32;
        let mut b2_scale_off = backup2_scale_offset as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_fused_dequant_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut bsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sc),
                arg(&mut bpw_val),
                arg(&mut out_cnt),
                arg(&mut out_idx_ptr),
                arg(&mut out_val_ptr),
                arg(&mut b_bpw),
                arg(&mut b_codes_off),
                arg(&mut b_scale_off),
                arg(&mut b2_bpw),
                arg(&mut b2_codes_off),
                arg(&mut b2_scale_off),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Launch the Charon fused MoE dispatch kernel (`rocm_kernel_plan.md`
    /// WI-A). Single sortless launch: each block reads its (token, expert)
    /// pair from the uploaded routing arrays and performs the fused
    /// gate+up→SiLU→down→weighted-accumulate.
    ///
    /// Device-gated: only callable with real `RocmStorage` device buffers.
    /// The host-logic half (routing flatten + grid/block plan + arg
    /// marshalling) is unit-tested without a device in
    /// `kernels::charon::tests` (G-A2). Parity vs the CPU oracle
    /// (`MoeFfn::forward`) is G-A4 — a device-verify TODO in this sandbox.
    ///
    /// Caller wiring lives in `grim_nn::moe::MoeFfn::forward_rocm`
    /// (gated on the `rocm-mem` feature), reached when the activation is on
    /// `Device::Rocm`. That path routes through `moe_fused_dispatch` below.
    pub fn launch_charon_fused_dispatch(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        assignment: &crate::kernels::charon::RoutingAssignment,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_fused_dispatch: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_fused_dispatch: out has no device ptr".into()))?;

        // Validate shapes + null pointers before any HIP dereference (G-A2
        // host-logic path, unit-tested in kernels::charon::tests).
        crate::kernels::charon::validate_launch_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_ptr as *mut c_void,
            expert_up_w_ptr as *mut c_void,
            expert_down_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            assignment,
            hidden,
            inter,
        )?;

        // Zero the output buffer before launch: the kernel accumulates per-
        // expert contributions via `atomicAdd`, so any stale bytes in the
        // output storage would be added into the result. This mirrors the
        // `BackendDevice::zeros` path (hipMemset, roc_device.rs:1363).
        check_hip("charon hipMemset(output, 0)", unsafe {
            hipMemset(out_ptr as *mut c_void, 0, out_storage.bytes())
        })?;

        // Plan the launch (wave-aligned block, grid over pairs). Pass the
        // device's real wavefront size (W32 on gfx1036, W64 on CDNA).
        let wave = self.wavefront_size() as u32;
        let mut tuner_guard = self.autotuner.lock().ok();
        let plan = crate::kernels::charon::plan_fused_dispatch_with_autotuner(
            assignment,
            wave,
            tuner_guard.as_deref_mut(),
            &self.gpu_target,
            hidden,
            inter,
        );

        if plan.grid_x == 0 {
            // No pairs → nothing to launch; return the active stream.
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        // Upload the routing arrays (tokens, experts, weights) to the device.
        // These are freed after the launch synchronizes (mirroring the
        // embedding path's transient-buffer discipline).
        let mut tok_ptr = upload_device_buffer(&assignment.tokens)?;
        let mut exp_ptr = upload_device_buffer(&assignment.experts)?;
        let mut w_ptr = upload_device_buffer(&assignment.weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = assignment.num_pairs() as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_pairs_i),
                arg(&mut rsf),
            ],
        )?;

        // Free the transient routing buffers after the kernel completes.
        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    /// Fused MoE dispatch helper: allocates output buffer, uploads flat expert weights,
    /// and launches `grim_moe_fused_dispatch`.
    pub fn moe_fused_dispatch(
        &self,
        activations: &RocmStorage,
        gate_flat: &[f32],
        up_flat: &[f32],
        down_flat: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_buf = self.from_cpu(gate_flat, &Shape::new(vec![gate_flat.len()]), DType::F32)?;
        let up_buf = self.from_cpu(up_flat, &Shape::new(vec![up_flat.len()]), DType::F32)?;
        let down_buf = self.from_cpu(down_flat, &Shape::new(vec![down_flat.len()]), DType::F32)?;

        self.moe_fused_dispatch_resident(
            activations,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            assignment,
            out_shape,
            hidden,
            inter,
            routed_scaling_factor,
        )
    }

    /// Fused MoE dispatch against weight buffers that are already resident on
    /// the device. Unlike [`Self::moe_fused_dispatch`], no host `&[f32]`
    /// weight arrays are uploaded per call — callers keep the flattened
    /// gate/up/down buffers resident across forward calls (see
    /// `grim_nn::moe::MoeFfn::forward_rocm`), so the per-call cost is limited
    /// to the routing arrays and the output allocation.
    pub fn moe_fused_dispatch_resident(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        assignment: &crate::kernels::charon::RoutingAssignment,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("gate_buf downcast failed".into()))?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("up_buf downcast failed".into()))?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("down_buf downcast failed".into()))?;

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let stream = self.launch_charon_fused_dispatch(
            activations,
            gate_r.device_ptr.unwrap(),
            up_r.device_ptr.unwrap(),
            down_r.device_ptr.unwrap(),
            assignment,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Device launcher for the #1 token-sorted (grouped) fused MoE dispatch.

    ///
    /// Mirrors `launch_charon_fused_dispatch` but feeds the sorted routing
    /// layout (`SortedRouting`) produced by `moe_align_block_size` and calls
    /// `grim_moe_fused_grouped`. The in-kernel math is identical to the
    /// sortless path (gate+up fused → SiLU → down → weighted accumulate), so
    /// the high-perf fused structure is preserved across quantizations — only
    /// the work ordering changes (grouped by expert, no per-pair atomics
    /// contention beyond the necessary top-k>1 accumulation).
    ///
    /// Device-gated: only callable with real `RocmStorage` device buffers.
    #[allow(dead_code)]
    pub(crate) fn launch_charon_grouped_dispatch(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        num_experts: usize,
    ) -> Result<*mut c_void> {
        self.launch_charon_grouped_dispatch_entry(
            activations,
            expert_gate_w_ptr,
            expert_up_w_ptr,
            expert_down_w_ptr,
            sorted,
            out_storage,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
            "grim_moe_fused_grouped",
        )
    }

    /// WI-F3 — grouped dispatch against a caller-selected kernel entry, so
    /// `CharonSelector` variants can route to the WMMA grouped kernel
    /// (`grim_moe_fused_grouped_wmma`) or the scalar grouped kernel via
    /// `kernels::charon::grouped_dispatch_entry`. Same host/sort contract.
    pub(crate) fn launch_charon_grouped_dispatch_entry(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        num_experts: usize,
        entry: &str,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_dispatch: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_dispatch: out has no device ptr".into())
        })?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_ptr as *mut c_void,
            expert_up_w_ptr as *mut c_void,
            expert_down_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        // Output is accumulated via atomicAdd in-kernel; zero first.
        check_hip("charon_grouped hipMemset(output, 0)", unsafe {
            hipMemset(out_ptr as *mut c_void, 0, out_storage.bytes())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            entry,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    /// Device launcher for the #2 FP8 W8A8 token-sorted grouped dispatch.
    ///
    /// Mirrors `launch_charon_grouped_dispatch` (same sorted layout + grid/block)
    /// but weights are FP8 E4M3 bytes with per-block-16 scales + per-token act
    /// scale. Calls `grim_moe_fused_grouped_fp8`, reusing the identical
    /// in-register fused math so the high-perf structure is preserved across
    /// quantization (vLLM W8A8 contract).
    #[allow(dead_code)]
    pub(crate) fn launch_charon_grouped_dispatch_fp8(
        &self,
        activations: &RocmStorage,
        expert_gate_w_fp8_ptr: u64,
        expert_up_w_fp8_ptr: u64,
        expert_down_w_fp8_ptr: u64,
        expert_gate_scale_ptr: u64,
        expert_up_scale_ptr: u64,
        expert_down_scale_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_fp8: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_fp8: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_fp8_ptr as *mut c_void,
            expert_up_w_fp8_ptr as *mut c_void,
            expert_down_w_fp8_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_fp8 hipMemset(output, 0)", unsafe {
            hipMemset(out_ptr as *mut c_void, 0, out_storage.bytes())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_fp8_ptr as *mut c_void;
        let mut uw = expert_up_w_fp8_ptr as *mut c_void;
        let mut dw = expert_down_w_fp8_ptr as *mut c_void;
        let mut gs = expert_gate_scale_ptr as *mut c_void;
        let mut us = expert_up_scale_ptr as *mut c_void;
        let mut ds = expert_down_scale_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_fp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut gs),
                arg(&mut us),
                arg(&mut ds),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped_fp8 hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    pub(crate) fn launch_charon_grouped_dispatch_mxfp4(
        &self,
        activations: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        egate_e_ptr: u64,
        eup_e_ptr: u64,
        edown_e_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_mxfp4: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_mxfp4: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            egate_w_ptr as *mut c_void,
            eup_w_ptr as *mut c_void,
            edown_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_mxfp4 hipMemset(output, 0)", unsafe {
            hipMemset(out_ptr as *mut c_void, 0, out_storage.bytes())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = egate_w_ptr as *mut c_void;
        let mut uw = eup_w_ptr as *mut c_void;
        let mut dw = edown_w_ptr as *mut c_void;
        let mut ge = egate_e_ptr as *mut c_void;
        let mut ue = eup_e_ptr as *mut c_void;
        let mut de = edown_e_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ge),
                arg(&mut ue),
                arg(&mut de),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped_mxfp4 hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    pub(crate) fn launch_charon_grouped_dispatch_mxfp8(
        &self,
        activations: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        egate_e_ptr: u64,
        eup_e_ptr: u64,
        edown_e_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_mxfp8: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_mxfp8: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            egate_w_ptr as *mut c_void,
            eup_w_ptr as *mut c_void,
            edown_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_mxfp8 hipMemset(output, 0)", unsafe {
            hipMemset(out_ptr as *mut c_void, 0, out_storage.bytes())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = egate_w_ptr as *mut c_void;
        let mut uw = eup_w_ptr as *mut c_void;
        let mut dw = edown_w_ptr as *mut c_void;
        let mut ge = egate_e_ptr as *mut c_void;
        let mut ue = eup_e_ptr as *mut c_void;
        let mut de = edown_e_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ge),
                arg(&mut ue),
                arg(&mut de),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped_mxfp8 hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    pub(crate) fn launch_charon_grouped_dispatch_q80(
        &self,
        activations: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_q80: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_q80: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            egate_w_ptr as *mut c_void,
            eup_w_ptr as *mut c_void,
            edown_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_q80 hipMemset(output, 0)", unsafe {
            hipMemset(out_ptr as *mut c_void, 0, out_storage.bytes())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = egate_w_ptr as *mut c_void;
        let mut uw = eup_w_ptr as *mut c_void;
        let mut dw = edown_w_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_q80",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped_q80 hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    /// Launcher for the unified IQ/K-quant grouped MoE kernel
    /// (`grim_moe_fused_grouped_iqk`). `format_id` selects the super-block
    /// decode (0 iq4nl .. 11 q3k); each expert's weights occupy one 256-weight
    /// super-block of `BLOCK_BYTES[format_id]` bytes. Mirrors
    /// `launch_charon_grouped_dispatch_q80` otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_charon_grouped_dispatch_iqk(
        &self,
        act_storage: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        format_id: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        use crate::kernels::charon::plan_grouped_dispatch;
        let _ = num_experts; // validated by caller; kernel reads per-expert super-blocks

        check_hip("charon_grouped_iqk hipMemset(output, 0)", unsafe {
            hipMemset(
                out_storage.device_ptr.ok_or_else(|| {
                    Error::Backend("charon_grouped_iqk: out has no device ptr".into())
                })? as *mut c_void,
                0,
                out_storage.bytes(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        let mut a = act_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_iqk: activations has no device ptr".into())
        })? as *mut c_void;
        let mut gw = egate_w_ptr;
        let mut uw = eup_w_ptr;
        let mut dw = edown_w_ptr;
        let mut ascale = a_scale_ptr;
        let mut optr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_iqk: out has no device ptr".into()))?
            as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut format_i = format_id as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_iqk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut format_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped_iqk hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    /// (`tests/golden_charon_moe_gpu.rs`, G-A4). Takes plain host `f32`
    /// buffers + a routing assignment, uploads them, zeros the output,
    /// launches `grim_moe_fused_dispatch`, and reads the result back.
    ///
    /// Expert weight layout: `[num_experts, inter*hidden]` (gate/up) and
    /// `[num_experts, hidden*inter]` (down), expert outermost — matching
    /// `ExpertBank::gate[e].weight` / `down[e].weight` row-major layout so
    /// the kernel's `gw + exp*inter*hidden` stride is correct.
    ///
    /// This is the only public surface the integration tests need; it does
    /// not expose `RocmStorage` or raw device pointers.
    pub fn charon_fused_dispatch_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        let act_shape = Shape::new(vec![batch, hidden]);
        let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
        let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
        let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        // Upload host buffers to the device.
        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_gate_w, &exp_gate_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_up_w, &exp_up_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_down_w, &exp_down_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        // Downcast to RocmStorage to reach device pointers + the launcher.
        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        self.launch_charon_fused_dispatch(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            assignment,
            out_s,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Host-to-host roundtrip for the #1 token-sorted (grouped) fused MoE
    /// dispatch. Mirrors `charon_fused_dispatch_roundtrip` but token-sorts the
    /// routing (vLLM `moe_align_block_size`) and launches
    /// `grim_moe_fused_grouped`. Used by the cross-kernel parity golden test
    /// that proves the grouped path produces identical numerics to the
    /// sortless path on gfx1036 (G-A4 extension for WI-A).
    pub fn charon_grouped_dispatch_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        let num_experts = expert_gate_w.len() / (inter * hidden);
        let block_size = 128usize; // token-block the grouped kernel strides across

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
        let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
        let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_gate_w, &exp_gate_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_up_w, &exp_up_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_down_w, &exp_down_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        self.launch_charon_grouped_dispatch(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// WI-F3 — WMMA grouped MoE dispatch roundtrip: sorts via
    /// `moe_align_block_size`, then launches the WMMA/tensor-core grouped
    /// kernel (`grim_moe_fused_grouped_wmma`, the
    /// `CharonVariant::LargeGroupPrefill` dispatch target via
    /// `kernels::charon::grouped_dispatch_entry`) and reads the result back.
    /// On non-WMMA arches (gfx1036/RDNA2) the kernel compiles to the scalar
    /// fallback, so this roundtrip is parity-safe everywhere.
    pub fn charon_grouped_dispatch_wmma_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        let num_experts = expert_gate_w.len() / (inter * hidden);
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
        let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
        let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_gate_w, &exp_gate_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_up_w, &exp_up_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_down_w, &exp_down_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let entry = crate::kernels::charon::grouped_dispatch_entry(
            crate::kernels::charon::CharonVariant::LargeGroupPrefill,
        );
        self.launch_charon_grouped_dispatch_entry(
            act_s,
            dev_ptr(gw_s)?,
            dev_ptr(uw_s)?,
            dev_ptr(dw_s)?,
            &sorted,
            out_s,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
            entry,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    // -----------------------------------------------------------------------
    // Charon MoE backward launcher (P2 — WI-Charon-1 device dispatch)
    // -----------------------------------------------------------------------

    /// Device launcher for the FP32 Charon MoE backward kernel
    /// (`grim_moe_fused_grouped_backward`). Mirrors
    /// `launch_charon_grouped_dispatch`: validates inputs, zero-initialises the
    /// four atomicAdd output buffers, plans the grouped grid/block from the
    /// sorted routing arrays, uploads sorted routing arrays, and launches.
    pub(crate) fn launch_charon_grouped_backward(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        d_y: &RocmStorage,
        d_gate_w: &RocmStorage,
        d_up_w: &RocmStorage,
        d_down_w: &RocmStorage,
        d_x: &RocmStorage,
        sorted: &crate::kernels::charon::SortedRouting,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        _num_experts: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: activations has no device ptr".into())
        })?;
        let dy_ptr = d_y.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_y has no device ptr".into())
        })?;
        let dgw_ptr = d_gate_w.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_gate_w has no device ptr".into())
        })?;
        let duw_ptr = d_up_w.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_up_w has no device ptr".into())
        })?;
        let ddw_ptr = d_down_w.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_down_w has no device ptr".into())
        })?;
        let dx_ptr = d_x.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_x has no device ptr".into())
        })?;

        crate::kernels::charon_backward::validate_backward_inputs(
            expert_gate_w_ptr as *const c_void,
            expert_up_w_ptr as *const c_void,
            expert_down_w_ptr as *const c_void,
            dy_ptr as *const c_void,
            dgw_ptr as *mut c_void,
            duw_ptr as *mut c_void,
            ddw_ptr as *mut c_void,
            dx_ptr as *mut c_void,
            hidden,
            inter,
            sorted.num_tokens_post_padded,
            sorted.block_size,
        )?;

        // All four output buffers are accumulated via atomicAdd; zero first.
        check_hip("charon_backward hipMemset(d_gate_w, 0)", unsafe {
            hipMemset(dgw_ptr as *mut c_void, 0, d_gate_w.bytes())
        })?;
        check_hip("charon_backward hipMemset(d_up_w, 0)", unsafe {
            hipMemset(duw_ptr as *mut c_void, 0, d_up_w.bytes())
        })?;
        check_hip("charon_backward hipMemset(d_down_w, 0)", unsafe {
            hipMemset(ddw_ptr as *mut c_void, 0, d_down_w.bytes())
        })?;
        check_hip("charon_backward hipMemset(d_x, 0)", unsafe {
            hipMemset(dx_ptr as *mut c_void, 0, d_x.bytes())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(&sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(&sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(&sorted.sorted_weights)?;

        // Kernel arg order matches grim_moe_fused_grouped_backward signature:
        // activations, gate_w, up_w, down_w, d_y, d_gate_w, d_up_w, d_down_w,
        // d_x, sorted_token_ids, sorted_expert_ids, sorted_weights,
        // hidden, inter, num_tokens, block_size, routed_scaling_factor
        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut dy = dy_ptr as *mut c_void;
        let mut dgw = dgw_ptr as *mut c_void;
        let mut duw = duw_ptr as *mut c_void;
        let mut ddw = ddw_ptr as *mut c_void;
        let mut dx = dx_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_backward",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut dy),
                arg(&mut dgw),
                arg(&mut duw),
                arg(&mut ddw),
                arg(&mut dx),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(tok_ptr);
                    hipFree(exp_ptr);
                    hipFree(w_ptr);
                    return Err(Error::Backend(format!(
                        "charon_grouped_backward hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(tok_ptr);
                hipFree(exp_ptr);
                hipFree(w_ptr);
            }
        }
        Ok(stream)
    }

    /// Host-to-host roundtrip for the Charon MoE backward kernel.
    ///
    /// Uploads all inputs (activations, expert weights, d_y) and the sorted
    /// routing arrays to the device, launches `grim_moe_fused_grouped_backward`,
    /// and downloads the four gradient buffers (`d_gate_w`, `d_up_w`,
    /// `d_down_w`, `d_x`). Used by the P2 device verifier test.
    pub fn charon_grouped_backward_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        d_y: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<CharonBackwardResult> {
        let num_experts = expert_gate_w.len() / (inter * hidden);
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w.len()]);
        let uw_shape = Shape::new(vec![expert_up_w.len()]);
        let dw_shape = Shape::new(vec![expert_down_w.len()]);
        let dy_shape = Shape::new(vec![d_y.len()]);

        let dgw_shape = Shape::new(vec![num_experts * inter * hidden]);
        let duw_shape = Shape::new(vec![num_experts * inter * hidden]);
        let ddw_shape = Shape::new(vec![num_experts * hidden * inter]);
        let dx_shape = Shape::new(vec![batch * hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_gate_w, &gw_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_up_w, &uw_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_down_w, &dw_shape, DType::F32)?;
        let dy_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, d_y, &dy_shape, DType::F32)?;

        // Output grad buffers — allocated (not uploaded), kernel fills them.
        let dgw_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &dgw_shape, DType::F32)?;
        let duw_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &duw_shape, DType::F32)?;
        let ddw_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &ddw_shape, DType::F32)?;
        let dx_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &dx_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let dy_s = as_rocm(dy_storage.as_ref())?;
        let dgw_s = as_rocm(dgw_storage.as_ref())?;
        let duw_s = as_rocm(duw_storage.as_ref())?;
        let ddw_s = as_rocm(ddw_storage.as_ref())?;
        let dx_s = as_rocm(dx_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        self.launch_charon_grouped_backward(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            dy_s,
            dgw_s,
            duw_s,
            ddw_s,
            dx_s,
            &sorted,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
        )?;
        self.synchronize();

        Ok(CharonBackwardResult {
            d_gate_w: dgw_storage.to_cpu_vec_f32()?,
            d_up_w: duw_storage.to_cpu_vec_f32()?,
            d_down_w: ddw_storage.to_cpu_vec_f32()?,
            d_x: dx_storage.to_cpu_vec_f32()?,
        })
    }

    /// Host-to-host roundtrip for the #3 MXFP4 (E2M1 + E8M0) token-sorted grouped
    /// dispatch. Takes packed E2M1 weight codes + E8M0 shared-exponent bytes
    /// (one exp per 32-element group along the contraction dim) for gate/up/down,
    /// token-sorts via `moe_align_block_size`, and launches
    /// `grim_moe_fused_grouped_mxfp4`. Used by the MXFP4-vs-FP32 KAT golden test.
    pub fn charon_grouped_dispatch_roundtrip_mxfp4(
        &self,
        activations: &[f32],
        expert_gate_w_codes: &[u8], // packed E2M1, [num_experts, inter*hidden/2]
        expert_up_w_codes: &[u8],
        expert_down_w_codes: &[u8],
        expert_gate_e8m0: &[u8], // [num_experts, inter*hidden/32]
        expert_up_e8m0: &[u8],
        expert_down_e8m0: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // num_experts from the packed gate-code layout: [num_experts, inter*hidden/2].
        let num_experts = expert_gate_w_codes.len() / ((inter * hidden / 2).max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_codes.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_codes.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_codes.len()]);
        let ge_shape = Shape::new(vec![expert_gate_e8m0.len()]);
        let ue_shape = Shape::new(vec![expert_up_e8m0.len()]);
        let de_shape = Shape::new(vec![expert_down_e8m0.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_w_codes,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_w_codes,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_w_codes,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ge_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_e8m0,
            &ge_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ue_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_e8m0,
            &ue_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let de_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_e8m0,
            &de_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let ge_s = as_rocm(ge_storage.as_ref())?;
        let ue_s = as_rocm(ue_storage.as_ref())?;
        let de_s = as_rocm(de_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let ge_ptr = dev_ptr(ge_s)?;
        let ue_ptr = dev_ptr(ue_s)?;
        let de_ptr = dev_ptr(de_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_mxfp4(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            ge_ptr,
            ue_ptr,
            de_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Host-to-host roundtrip for the #4 MXFP8 (E4M3 + E8M0) token-sorted grouped
    /// dispatch. Takes E4M3 weight codes (1 byte each, NOT packed) + one E8M0
    /// shared-exponent byte per 32-element group, reusing the identical in-register
    /// gate/up/SiLU/down math. Used by the MXFP8-vs-FP32 KAT (WI-A / G-A4 extension).
    pub fn charon_grouped_dispatch_roundtrip_mxfp8(
        &self,
        activations: &[f32],
        expert_gate_w_codes: &[u8], // E4M3, [num_experts, inter*hidden]
        expert_up_w_codes: &[u8],
        expert_down_w_codes: &[u8],
        expert_gate_e8m0: &[u8], // [num_experts, inter*hidden/32]
        expert_up_e8m0: &[u8],
        expert_down_e8m0: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // num_experts from the E4M3 gate-code layout: [num_experts, inter*hidden].
        let num_experts = expert_gate_w_codes.len() / ((inter * hidden).max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_codes.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_codes.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_codes.len()]);
        let ge_shape = Shape::new(vec![expert_gate_e8m0.len()]);
        let ue_shape = Shape::new(vec![expert_up_e8m0.len()]);
        let de_shape = Shape::new(vec![expert_down_e8m0.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_w_codes,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_w_codes,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_w_codes,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ge_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_e8m0,
            &ge_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ue_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_e8m0,
            &ue_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let de_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_e8m0,
            &de_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let ge_s = as_rocm(ge_storage.as_ref())?;
        let ue_s = as_rocm(ue_storage.as_ref())?;
        let de_s = as_rocm(de_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let ge_ptr = dev_ptr(ge_s)?;
        let ue_ptr = dev_ptr(ue_s)?;
        let de_ptr = dev_ptr(de_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_mxfp8(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            ge_ptr,
            ue_ptr,
            de_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Takes Q8_0 weight bytes (f16 scale + i8 per 32 weights) + per-token act
    /// scale, token-sorts via `moe_align_block_size`, and launches
    /// `grim_moe_fused_grouped_q80` reusing the identical in-register math. Used
    /// by the Q8_0-vs-FP32 KAT golden test (WI-5 / G-A4 extension).
    pub fn charon_grouped_dispatch_roundtrip_q80(
        &self,
        activations: &[f32],
        expert_gate_w_q80: &[u8],
        expert_up_w_q80: &[u8],
        expert_down_w_q80: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // Q8_0 layout: per 32 weights a 2-byte f16 scale + 32 i8 => 34 bytes.
        let bytes_per_block = 34usize;
        let weights_per_expert = inter * hidden;
        let num_experts =
            expert_gate_w_q80.len() / (bytes_per_block * weights_per_expert.div_ceil(32));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_q80.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_q80.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_q80.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_w_q80,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_w_q80,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_w_q80,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_q80(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Generic host-to-host roundtrip for the unified IQ/K-quant token-sorted
    /// grouped dispatch. `format_id` selects the super-block decode (0 iq4nl ..
    /// 11 q3k); `block_bytes` is `BLOCK_BYTES[format_id]`. Each expert's weights
    /// occupy one super-block of `block_bytes` bytes. Used by the IQ/K-vs-FP32
    /// KAT golden tests.
    pub fn charon_grouped_dispatch_roundtrip_iqk(
        &self,
        activations: &[f32],
        expert_gate_w_q: &[u8],
        expert_up_w_q: &[u8],
        expert_down_w_q: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        format_id: usize,
        block_bytes: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // Each expert occupies one 256-weight super-block of `block_bytes`.
        let weights_per_expert = (inter * hidden).div_ceil(256) * 256;
        let num_experts = expert_gate_w_q.len() / (block_bytes * (weights_per_expert / 256).max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_q.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_q.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_q.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_w_q,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_w_q,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_w_q,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_iqk(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            format_id,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// token-sorts via `moe_align_block_size`, and launches
    /// `grim_moe_fused_grouped_fp8` reusing the identical in-register math. Used
    /// by the FP8-vs-FP32 KAT golden test (WI-A / G-A4 extension for WI-2).
    pub fn charon_grouped_dispatch_roundtrip_fp8(
        &self,
        activations: &[f32],
        expert_gate_w_fp8: &[u8],
        expert_up_w_fp8: &[u8],
        expert_down_w_fp8: &[u8],
        expert_gate_scale: &[f32],
        expert_up_scale: &[f32],
        expert_down_scale: &[f32],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // num_experts is derived from the gate-scale layout produced by the test:
        //   gate_scale is [num_experts, inter, hidden/16]  (block_size=16 along hidden)
        let h16 = (hidden + 15) / 16;
        let num_experts = expert_gate_scale.len() / (inter * h16.max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_fp8.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_fp8.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_fp8.len()]);
        let gs_shape = Shape::new(vec![expert_gate_scale.len()]);
        let us_shape = Shape::new(vec![expert_up_scale.len()]);
        let ds_shape = Shape::new(vec![expert_down_scale.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, activations, &act_shape, DType::F32)?;
        // FP8 weights are uploaded as raw U8 blobs (no DType::F8 on this path).
        let gw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_gate_w_fp8,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_up_w_fp8,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = BackendDevice::from_cpu_bytes(
            self,
            expert_down_w_fp8,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let gs_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_gate_scale, &gs_shape, DType::F32)?;
        let us_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_up_scale, &us_shape, DType::F32)?;
        let ds_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, expert_down_scale, &ds_shape, DType::F32)?;
        let as_storage: Box<dyn BackendStorage> =
            BackendDevice::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            BackendDevice::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let gs_s = as_rocm(gs_storage.as_ref())?;
        let us_s = as_rocm(us_storage.as_ref())?;
        let ds_s = as_rocm(ds_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let gs_ptr = dev_ptr(gs_s)?;
        let us_ptr = dev_ptr(us_s)?;
        let ds_ptr = dev_ptr(ds_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_fp8(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            gs_ptr,
            us_ptr,
            ds_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_f16(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        default_bpw: u8,
        outlier_count: usize,
        outlier_indices_ptr: *const c_void,
        outlier_values_ptr: *const c_void,
        backup_bpw: u8,
        backup_codes_offset: usize,
        backup_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: B has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        // Grid covers M*K output elements (one thread per element of dX[M,K]).
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward: m*k overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_backward: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        // dY is [M, N] row-major → stride_dy = N
        let mut sdy = n as i32;
        let mut sdx = k as i32;

        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;

        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;

        let mut b2_bpw = backup2_bpw as i32;
        let mut b2_codes_off = backup2_codes_offset as i32;
        let mut b2_scale_off = backup2_scale_offset as i32;

        // STE: grad_scale = 1.0 for pure identity (straight-through estimator).
        // The quantize→dequantize step receives zero gradient — the upstream
        // gradient flows straight through to the dequantized weight values.
        let mut grad_scale: f32 = 1.0;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut bsptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sdy),
                arg(&mut sdx),
                arg(&mut bpw_val),
                arg(&mut out_cnt),
                arg(&mut out_idx_ptr),
                arg(&mut out_val_ptr),
                arg(&mut b_bpw),
                arg(&mut b_codes_off),
                arg(&mut b_scale_off),
                arg(&mut b2_bpw),
                arg(&mut b2_codes_off),
                arg(&mut b2_scale_off),
                arg(&mut grad_scale),
            ],
        )
    }

    /// FUSED-QUANT-BWD §4: Launch the M+Adam fused optimizer-step kernel.
    ///
    /// Runs AFTER the backward GEMM kernel so all tile-level gradients in `dX`
    /// are fully accumulated before scale-bump propagation begins (fixes the
    /// stale-scale one-step concern from new_methods.md §Caveats).
    ///
    /// Updates `weight` and `scale` in-place using M+Adam's additive-multiplicative
    /// split: momentum in FP8-simulated precision, scale-bump propagation in BF16.
    #[allow(dead_code)] // kernel launcher, not yet wired into this build's call graph
    pub(crate) fn launch_madam_update_f32(
        &self,
        dx_storage: &RocmStorage,
        weight_storage: &RocmStorage,
        scale_storage: Option<&RocmStorage>,
        m_buffer: &RocmStorage,
        v_buffer: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<*mut c_void> {
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: dX has no device ptr".into()))?;
        let w_ptr = weight_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: weight has no device ptr".into()))?;
        let m_ptr = m_buffer
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: m_buffer has no device ptr".into()))?;
        let v_ptr = v_buffer
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: v_buffer has no device ptr".into()))?;
        let scale_ptr: *const std::ffi::c_void = scale_storage
            .and_then(|s| s.device_ptr)
            .map(|p| p as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null());

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("madam_update: m*k overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "madam_update: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;

        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dxptr = dx_ptr;
        let mut wptr = w_ptr;
        let mut sptr = scale_ptr;
        let mut mptr = m_ptr;
        let mut vptr = v_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut lr_f = lr as f32;
        let mut b1 = beta1 as f32;
        let mut b2 = beta2 as f32;
        let mut ep = eps as f32;
        let mut stp = step as i32;

        self.launch_compute_kernel(
            "grim_madam_update_f32",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dxptr),
                arg(&mut wptr),
                arg(&mut sptr),
                arg(&mut mptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut lr_f),
                arg(&mut b1),
                arg(&mut b2),
                arg(&mut ep),
                arg(&mut stp),
            ],
        )?;
        Ok(std::ptr::null_mut())
    }

    /// Launch the JIT compiled Q4_K fused dequantization matmul kernel (Crow Tier).
    pub(crate) fn launch_fused_dequant_gemm_q4k(
        &self,
        a_storage: &RocmStorage,
        b_q4k_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q4k: a has no device ptr".into()))?;
        let b_ptr = b_q4k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q4k: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_q4k: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q4k: m*n overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!("fused_dequant_gemm_q4k: grid too large for u32"))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_q4k",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled Q4_K fused dequantization backward matmul kernel (Crow Tier).
    /// `b_scales_ptr` is accepted for interface parity with the f16 fallback;
    /// KQuant blocks carry their own per-block scales inline and the kernel
    /// does not consume it (passing null here is the KQuant contract).
    pub(crate) fn launch_fused_dequant_backward_gemm_q4k(
        &self,
        dy_storage: &RocmStorage,
        b_q4k_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k: dY has no device ptr".into())
        })?;
        let b_ptr = b_q4k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k: B has no device ptr".into())
        })?;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k: dX has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward_q4k: m*k overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_backward_q4k: grid too large for u32"
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let bsptr = b_scales_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let _ = bsptr;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_q4k",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Generic forward fused dequant GEMM launcher for simple kernels
    fn launch_fused_deq_gemm_simple(
        &self,
        name: &str,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: a has no device ptr", name)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", name)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", name)))?;
        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend(format!("{}: m*n overflow", name)))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Generic backward fused dequant GEMM launcher.
    fn launch_fused_deq_backward_gemm_simple(
        &self,
        name: &str,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dY has no device ptr", name)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: B has no device ptr", name)))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dX has no device ptr", name)))?;
        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend(format!("{}: m*k overflow", name)))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    // ─── Standalone dequant launchers ──────────────────────────────────────────

    /// Dequantize Q4_K packed bytes to F32. `n_blocks` is derived [see: `packed.bytes / 144`]
    pub(crate) fn launch_dequant_q4k(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q4k: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q4k: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_blocks as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("dequant_q4k: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_blk = n_blocks as i32;
        self.launch_compute_kernel(
            "grim_dequant_q4k",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_blk)],
        )
    }

    /// Standalone FP8 dequant: convert FP8 E4M3 bytes to F32.
    pub(crate) fn launch_dequant_fp8(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_fp8: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_fp8: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("dequant_fp8: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_fp8",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_w)],
        )
    }

    /// Standalone MXFP4 dequant: decompress MXFP4 codes + shared exponents to F32.
    pub(crate) fn launch_dequant_mxfp4(
        &self,
        codes_storage: &RocmStorage,
        exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let codes_ptr = codes_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: codes has no device ptr".into()))?;
        let exps_ptr = exps_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: exps has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("dequant_mxfp4: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut codes = codes_ptr;
        let mut exps = exps_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut codes),
                arg(&mut exps),
                arg(&mut out),
                arg(&mut n_w),
            ],
        )
    }

    /// Standalone MXFP8 dequant: decompress MXFP8 codes + shared exponents to F32.
    pub(crate) fn launch_dequant_mxfp8(
        &self,
        codes_storage: &RocmStorage,
        exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let codes_ptr = codes_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: codes has no device ptr".into()))?;
        let exps_ptr = exps_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: exps has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("dequant_mxfp8: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut codes = codes_ptr;
        let mut exps = exps_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut codes),
                arg(&mut exps),
                arg(&mut out),
                arg(&mut n_w),
            ],
        )
    }

    // ─── Standalone IQ dequant launchers ──────────────────────────

    pub(crate) fn launch_dequant_iq2xxs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq2xxs", packed_storage, out_storage, n_blocks)
    }
    pub(crate) fn launch_dequant_iq2xs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq2xs", packed_storage, out_storage, n_blocks)
    }
    pub(crate) fn launch_dequant_iq2s(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq2s", packed_storage, out_storage, n_blocks)
    }
    pub(crate) fn launch_dequant_iq3xxs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq3xxs", packed_storage, out_storage, n_blocks)
    }
    pub(crate) fn launch_dequant_iq3s(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq3s", packed_storage, out_storage, n_blocks)
    }
    pub(crate) fn launch_dequant_iq4nl(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq4nl", packed_storage, out_storage, n_blocks)
    }
    pub(crate) fn launch_dequant_iq4xs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq4xs", packed_storage, out_storage, n_blocks)
    }

    /// Generic helper for standalone dequant kernels that take (packed, out, n_blocks).
    fn launch_generic_dequant(
        &self,
        name: &str,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: packed has no device ptr", name)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", name)))?;
        // The IQ dequant kernels use one 64-thread block per quant block
        // (each thread decodes 4 elements with a float4 store).
        const BLOCK_SIZE: usize = 64;
        let grid_x: u32 = n_blocks
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_blk = n_blocks as i32;
        self.launch_compute_kernel(
            name,
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_blk)],
        )
    }

    // ─── Fused dequant+GEMM kernels: Q5K ──────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_q5k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q5k", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_q5k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q5k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: Q6K ──────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_q6k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q6k", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_q6k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q6k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: Q2K ──────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_q2k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q2k", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_q2k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q2k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: Q3K ──────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_q3k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q3k", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_q3k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q3k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ2_XXS ──────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq2xxs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2xxs", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq2xxs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq2xxs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ2_XS ───────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq2xs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2xs", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq2xs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq2xs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ2_S ────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq2s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2s", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq2s(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq2s",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ3_XXS ──────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq3xxs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq3xxs", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq3xxs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq3xxs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ3_S ────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq3s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq3s", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq3s(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq3s",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ4_NL ──────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq4nl(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq4nl", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq4nl(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq4nl",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: IQ4_XS ──────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_iq4xs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq4xs", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_iq4xs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq4xs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    // ─── Fused dequant+GEMM kernels: Q8_0 ────────────────────────────────────

    pub(crate) fn launch_fused_dequant_gemm_q8_0(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q8_0", a, b, out, m, n, k)
    }
    pub(crate) fn launch_fused_dequant_backward_gemm_q8_0(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q8_0",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    /// Launch the JIT compiled Q8_0 dequantization kernel.  Reads packed [see: `packed_storage`, `n_weights`, `out_storage`, `materialize()`]
    pub fn dequantize_q8_0(&self, packed: &RocmStorage) -> Result<RocmStorage> {
        const QK8_0: usize = 32;
        // Q8_0 stores weights as packed bytes: each block is 34 bytes (2-byte [see: `n_blocks * 32`]
        let packed_bytes = packed.bytes;
        let n_blocks = packed_bytes / (QK8_0 + 2);
        let n_weights = n_blocks * QK8_0;
        let f32_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_weights]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_q8_0(packed, &f32_storage, n_blocks)?;
        Ok(f32_storage)
    }

    /// Dequantize Q8_0 packed bytes to an f32 host Vec via the ROCm kernel.
    pub fn dequantize_q8_0_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let f32_storage = self.dequantize_q8_0(&packed)?;
        let mut values = self.read_to_host_async(&f32_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize Q4_K packed bytes to F32 on the GPU. [see: `block_q4_K`]
    /// `packed` must hold `n_blocks` × 144-byte super-blocks; `out` must hold
    /// `n_blocks` × 256 F32 values.
    pub fn dequantize_q4k(&self, packed: &RocmStorage) -> Result<RocmStorage> {
        const QK4_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        let packed_bytes = packed.bytes;
        let n_blocks = packed_bytes / BLOCK_BYTES;
        let n_weights = n_blocks * QK4_K;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_weights]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_q4k(packed, &out_storage, n_blocks)?;
        Ok(out_storage)
    }

    /// Dequantize Q4_K packed bytes to an f32 host Vec via the ROCm kernel.
    pub fn dequantize_q4k_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let f32_storage = self.dequantize_q4k(&packed)?;
        let mut values = self.read_to_host_async(&f32_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    // ─── Standalone dequant host wrappers (iq/fp8/mxfp) ────────────────────────

    /// Run any standalone IQ dequant kernel against `bytes` and return `elem_count` f32 values.
    fn dequantize_iq_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        block_bytes: usize,
        kernel: &str,
    ) -> Result<Vec<f32>> {
        const QK: usize = 256;
        let n_blocks = bytes.len() / block_bytes;
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_blocks * QK]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        match kernel {
            "grim_dequant_iq2xxs" => {
                self.launch_dequant_iq2xxs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq2xs" => {
                self.launch_dequant_iq2xs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq2s" => {
                self.launch_dequant_iq2s(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq3xxs" => {
                self.launch_dequant_iq3xxs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq3s" => {
                self.launch_dequant_iq3s(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq4nl" => {
                self.launch_dequant_iq4nl(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq4xs" => {
                self.launch_dequant_iq4xs(&packed, &out_storage, n_blocks)?;
            }
            other => {
                return Err(Error::Backend(format!(
                    "dequantize_iq_host: unknown kernel {other}"
                )));
            }
        }
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize IQ2_XXS packed bytes via the ROCm kernel. 66 bytes / 256-elem super-block.
    pub fn dequantize_iq2xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 66, "grim_dequant_iq2xxs")
    }
    /// Dequantize IQ2_XS packed bytes via the ROCm kernel. 74 bytes / 256-elem super-block.
    pub fn dequantize_iq2xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 74, "grim_dequant_iq2xs")
    }
    /// Dequantize IQ2_S packed bytes via the ROCm kernel. 82 bytes / 256-elem super-block.
    pub fn dequantize_iq2s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 82, "grim_dequant_iq2s")
    }
    /// Dequantize IQ3_XXS packed bytes via the ROCm kernel. 96 bytes / 256-elem super-block.
    pub fn dequantize_iq3xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 96, "grim_dequant_iq3xxs")
    }
    /// Dequantize IQ3_S packed bytes via the ROCm kernel. 110 bytes / 256-elem super-block.
    pub fn dequantize_iq3s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 110, "grim_dequant_iq3s")
    }
    /// Dequantize IQ4_NL packed bytes via the ROCm kernel. 170 bytes / 256-elem super-block.
    pub fn dequantize_iq4nl_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 170, "grim_dequant_iq4nl")
    }
    /// Dequantize IQ4_XS packed bytes via the ROCm kernel. 178 bytes / 256-elem super-block.
    pub fn dequantize_iq4xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 178, "grim_dequant_iq4xs")
    }

    /// Dequantize packed FP8 bytes (4-byte f32 LE scale header, then one E4M3 code per element).
    pub fn dequantize_fp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let scale = if bytes.len() >= 4 {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            1.0
        };
        let payload = if bytes.len() >= 4 { &bytes[4..] } else { bytes };
        let packed = RocmStorage::copy_from_host_raw_bytes(
            payload,
            &Shape::new(vec![payload.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_fp8(&packed, &out_storage, elem_count)?;
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        for v in values.iter_mut() {
            *v *= scale;
        }
        Ok(values)
    }

    /// Helper to split an MXFP single-buffer (length-prefixed codes/exps segments) into two device buffers.
    /// Reuses the same framing as `grim_quant::dequant_mxfp4`/`dequant_mxfp8`.
    fn split_dequant_mxfp(
        &self,
        bytes: &[u8],
        elem_count: usize,
        kernel: &str,
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
        let codes = read_segment(bytes, &mut cursor)?;
        let exps = read_segment(bytes, &mut cursor)?;

        let codes_storage = RocmStorage::copy_from_host_raw_bytes(
            &codes,
            &Shape::new(vec![codes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let exps_storage = RocmStorage::copy_from_host_raw_bytes(
            &exps,
            &Shape::new(vec![exps.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;

        if kernel.contains("mxfp4") {
            self.launch_dequant_mxfp4(&codes_storage, &exps_storage, &out_storage, elem_count)?;
        } else {
            self.launch_dequant_mxfp8(&codes_storage, &exps_storage, &out_storage, elem_count)?;
        }
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize an MXFP4 single-buffer roster (length-prefixed codes/exps segments) to f32.
    pub fn dequantize_mxfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp(bytes, elem_count, "mxfp4")
    }
    /// Dequantize an MXFP8 single-buffer roster (length-prefixed codes/exps segments) to f32.
    pub fn dequantize_mxfp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp(bytes, elem_count, "mxfp8")
    }

    /// Dequantize Q8_0 packed bytes to F32. `n_blocks` is the number of [see: `packed`, `packed.bytes / 34`]
    pub(crate) fn launch_dequant_q8_0(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q8_0: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q8_0: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_blocks as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("dequant_q8_0: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_blk = n_blocks as i32;

        self.launch_compute_kernel(
            "grim_dequant_q8_0",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_blk)],
        )
    }

    /// Launch the JIT compiled FP8 fused dequantization matmul kernel (Raven Tier).
    pub(crate) fn launch_fused_dequant_gemm_fp8(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: a has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_fp8: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: m*n overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!("fused_dequant_gemm_fp8: grid too large for u32"))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_fp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled FP8 fused dequantization backward matmul kernel (Raven Tier).
    pub(crate) fn launch_fused_dequant_backward_gemm_fp8(
        &self,
        dy_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: dY has no device ptr".into())
        })?;
        let b_ptr = b_fp8_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: B has no device ptr".into())
        })?;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: dX has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward_fp8: m*k overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_backward_fp8: grid too large for u32"
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_fp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled MXFP4 fused dequantization matmul kernel (Jay Tier).
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_mxfp4(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: a has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_mxfp4: m*n overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!("fused_dequant_gemm_mxfp4: grid too large for u32"))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled MXFP8 fused dequantization matmul kernel (Magpie Tier).
    pub(crate) fn launch_fused_dequant_gemm_mxfp8(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: a has no device ptr".into())
        })?;
        let b_fp8_ptr = b_fp8_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: b_fp8 has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: b_exps has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_mxfp8: m*n overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| {
                Error::Backend(format!("fused_dequant_gemm_mxfp8: grid too large for u32"))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bfp8ptr = b_fp8_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bfp8ptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled tiled MXFP4 GEMM kernel.
    ///
    /// `b_codes_ptr` / `b_exps_ptr` are raw device pointers: either standalone
    /// storages or interior pointers into a framed weight blob (see the
    /// `MxFp4` arm of `quantized_matmul`).
    pub fn launch_mxfp4_gemm_tiled(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // The kernels read activations as float4 and codes as uint4; both
        // require K to be a multiple of 32 (one MXFP4 micro-block).
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "mxfp4_gemm_tiled: K must be a multiple of 32, got {k}"
            )));
        }
        // Skinny-M decode: a plain (n/16, m/16) grid leaves most CUs idle
        // (e.g. m=1, n=4096 -> 16 CTAs on a 28+ CU part). Route to the
        // split-K pair so the K dimension spreads work across the device.
        if m <= 8 && k >= 2048 {
            return self.launch_mxfp4_gemm_splitk(
                a_storage,
                b_codes_ptr,
                b_exps_ptr,
                out_storage,
                m,
                n,
                k,
            );
        }
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_tiled: a has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_tiled: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_x = ((n + 15) / 16) as u32;
        let grid_y = ((m + 15) / 16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_mxfp4_gemm_tiled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Split-K MXFP4 GEMM for skinny-M decode (M <= 8): slice K across CUs,
    /// reduce the partials deterministically. Kept as two launches so the
    /// result is bit-stable across runs (no float atomics).
    fn launch_mxfp4_gemm_splitk(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: a has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: out has no device ptr".into()))?;

        let num_splits: u32 = if k >= 8192 {
            8
        } else if k >= 4096 {
            4
        } else {
            2
        };
        let partials = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[num_splits as usize, m, n]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let partials_ptr = partials
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: partials alloc failed".into()))?;

        const SPLITK_BLOCK: usize = 64;
        let grid_dim = HipDim3::new(
            ((n + SPLITK_BLOCK - 1) / SPLITK_BLOCK) as u32,
            m as u32,
            num_splits,
        );
        let block_dim = HipDim3::new(SPLITK_BLOCK as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut pptr = partials_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut splits = num_splits as i32;

        let stream = self.launch_compute_kernel(
            "grim_mxfp4_gemm_splitk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut pptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut splits),
            ],
        )?;

        const REDUCE_BLOCK: usize = 256;
        let total = m * n;
        let reduce_grid = HipDim3::new(((total + REDUCE_BLOCK - 1) / REDUCE_BLOCK) as u32, 1, 1);
        let reduce_block = HipDim3::new(REDUCE_BLOCK as u32, 1, 1);

        let mut optr = out_ptr;
        let _ = self.launch_compute_kernel(
            "grim_mxfp4_splitk_reduce",
            reduce_grid,
            reduce_block,
            &mut [
                arg(&mut pptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut splits),
            ],
        )?;

        // `partials` drops to the pool here; same-stream reuse is ordered
        // after both kernels above.
        Ok(stream)
    }

    /// Launch the JIT compiled backward MXFP4 GEMM kernel (dA = dY @ B^T).
    pub(crate) fn launch_mxfp4_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_codes_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_backward_gemm: dy has no device ptr".into()))?;
        let b_codes_ptr = b_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4_backward_gemm: b_codes has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4_backward_gemm: b_exps has no device ptr".into())
        })?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_backward_gemm: dx has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_x = ((k + 15) / 16) as u32;
        let grid_y = ((m + 15) / 16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);

        let mut dyptr = dy_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_mxfp4_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the fused RMSNorm + MXFP4 GEMM kernel (e.g. for MLP projections).
    pub fn launch_fused_rmsnorm_mxfp4_gemm(
        &self,
        x_storage: &RocmStorage,
        gamma_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: x has no device ptr".into())
        })?;
        let gamma_ptr = gamma_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: gamma has no device ptr".into())
        })?;
        let w_codes_ptr = w_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: w_codes has no device ptr".into())
        })?;
        let w_exps_ptr = w_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: w_exps has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: out has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(64, 1, 1);
        let grid_dim = HipDim3::new(m as u32, ((n + 63) / 64) as u32, 1);

        let mut xptr = x_ptr;
        let mut gammaptr = gamma_ptr;
        let mut wcodesptr = w_codes_ptr;
        let mut wexpsptr = w_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut eps_val = eps;

        self.launch_compute_kernel_with_solution(
            "grim_fused_rmsnorm_mxfp4_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut gammaptr),
                arg(&mut wcodesptr),
                arg(&mut wexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut eps_val),
            ],
            None,
            64 * std::mem::size_of::<f32>(),
        )
    }

    /// Launch the fused RMSNorm + MXFP4 GEMM + RoPE + direct KV cache scatter kernel.
    pub fn launch_fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x_storage: &RocmStorage,
        gamma_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        q_out_storage: Option<&RocmStorage>,
        k_cache_storage: Option<&RocmStorage>,
        v_cache_storage: Option<&RocmStorage>,
        out_all_storage: Option<&RocmStorage>,
        positions_storage: Option<&RocmStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq_storage: Option<&RocmStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: x has no device ptr".into())
        })?;
        let gamma_ptr = gamma_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: gamma has no device ptr".into())
        })?;
        let w_codes_ptr = w_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: w_codes has no device ptr".into())
        })?;
        let w_exps_ptr = w_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: w_exps has no device ptr".into())
        })?;

        let q_out_ptr = q_out_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let k_cache_ptr = k_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let v_cache_ptr = v_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let out_all_ptr = out_all_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let positions_ptr = positions_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let inv_freq_ptr = inv_freq_storage.and_then(|s| s.device_ptr).unwrap_or(0);

        let n_total = (num_q_heads + 2 * num_kv_heads) * head_dim;
        let block_dim = HipDim3::new(64, 1, 1);
        let grid_dim = HipDim3::new(m as u32, ((n_total + 63) / 64) as u32, 1);

        let mut xptr = x_ptr;
        let mut gammaptr = gamma_ptr;
        let mut wcodesptr = w_codes_ptr;
        let mut wexpsptr = w_exps_ptr;
        let mut qptr = q_out_ptr;
        let mut kptr = k_cache_ptr;
        let mut vptr = v_cache_ptr;
        let mut allptr = out_all_ptr;
        let mut posptr = positions_ptr;
        let mut mm = m as i32;
        let mut kk = k as i32;
        let mut nq = num_q_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut theta = rope_theta;
        let mut invfreqptr = inv_freq_ptr;
        let mut mscale_val = mscale;
        let mut eps_val = eps;
        let mut max_seq = max_seq_len as i32;

        self.launch_compute_kernel_with_solution(
            "grim_fused_rmsnorm_mxfp4_gemm_rope_kv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut gammaptr),
                arg(&mut wcodesptr),
                arg(&mut wexpsptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut allptr),
                arg(&mut posptr),
                arg(&mut mm),
                arg(&mut kk),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut theta),
                arg(&mut invfreqptr),
                arg(&mut mscale_val),
                arg(&mut eps_val),
                arg(&mut max_seq),
            ],
            None,
            64 * std::mem::size_of::<f32>(),
        )
    }

    /// Launch LFM2-style fused QKV projection: MXFP4 GEMM (x @ W_qkv) followed by
    /// per-head QK-Norm + RoPE (YaRN-aware). The GEMM result is staged in a scratch
    /// buffer (or `out_all` if provided) and consumed by `grim_qk_norm_rope`.
    pub fn launch_fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x_storage: &RocmStorage,
        gamma_q_storage: &RocmStorage,
        gamma_k_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        q_out_storage: Option<&RocmStorage>,
        k_cache_storage: Option<&RocmStorage>,
        v_cache_storage: Option<&RocmStorage>,
        out_all_storage: Option<&RocmStorage>,
        positions_storage: Option<&RocmStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq_storage: Option<&RocmStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<*mut c_void> {
        let gamma_q_ptr = gamma_q_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gamma_q has no device ptr".into())
        })?;
        let gamma_k_ptr = gamma_k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gamma_k has no device ptr".into())
        })?;

        let n_q = num_q_heads * head_dim;
        let n_k = num_kv_heads * head_dim;
        let n_total = n_q + 2 * n_k;

        // Stage the raw QKV GEMM output. Reuse `out_all` if supplied; otherwise
        // allocate a transient scratch buffer freed after the stream syncs.
        let scratch = if out_all_storage.is_some() {
            None
        } else {
            Some(RocmStorage::alloc_gpu(
                &Shape::from_slice(&[m, n_total]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?)
        };
        let gemm_storage: &RocmStorage = match (out_all_storage, &scratch) {
            (Some(o), _) => o,
            (None, Some(s)) => s,
            (None, None) => unreachable!(),
        };
        let gemm_ptr = gemm_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gemm buffer has no device ptr".into())
        })?;

        // Phase 1: MXFP4 GEMM -> gemm_out (C = x @ W_qkv)
        self.launch_mxfp4_gemm_tiled(
            x_storage,
            w_codes_storage.device_ptr_u64().ok_or_else(|| {
                Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: codes ptr".into())
            })?,
            w_exps_storage.device_ptr_u64().ok_or_else(|| {
                Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: exps ptr".into())
            })?,
            gemm_storage,
            m,
            n_total,
            k,
        )?;

        // Phase 2: per-head QK-Norm + RoPE -> q_out / k_cache / v_cache
        let q_out_ptr = q_out_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let k_cache_ptr = k_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let v_cache_ptr = v_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let positions_ptr = positions_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let inv_freq_ptr = inv_freq_storage.and_then(|s| s.device_ptr).unwrap_or(0);

        let total = m * (num_q_heads + 2 * num_kv_heads);
        let (grid, block) = linear_launch(total);

        let mut gemmptr = gemm_ptr;
        let mut gqptr = gamma_q_ptr;
        let mut gkptr = gamma_k_ptr;
        let mut posptr = positions_ptr;
        let mut qptr = q_out_ptr;
        let mut kptr = k_cache_ptr;
        let mut vptr = v_cache_ptr;
        let mut mm = m as i32;
        let mut nq = num_q_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut theta = rope_theta;
        let mut invfreqptr = inv_freq_ptr;
        let mut mscale_val = mscale;
        let mut eps_val = eps;
        let mut max_seq = max_seq_len as i32;

        let stream = self.launch_compute_kernel(
            "grim_qk_norm_rope",
            grid,
            block,
            &mut [
                arg(&mut gemmptr),
                arg(&mut gqptr),
                arg(&mut gkptr),
                arg(&mut posptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut theta),
                arg(&mut invfreqptr),
                arg(&mut mscale_val),
                arg(&mut eps_val),
                arg(&mut max_seq),
            ],
        )?;

        // The transient scratch buffer returns to the caching allocator when
        // `scratch` drops at scope exit. The previous code additionally
        // hipFree'd the pointer by hand — a double free that also corrupted
        // the pool's bookkeeping (the allocator hands the same address to a
        // future allocation) — after a hipStreamSynchronize stall. Pooled
        // reuse is ordered by the single active stream, so neither the sync
        // nor the manual free is needed.
        drop(scratch);

        Ok(stream)
    }

    /// Launch FlashDecoding (Split-KV Parallel Attention) across sequence chunks + merge reduction.
    pub fn launch_flash_decode(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        num_splits: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: out has no device ptr".into()))?;

        let num_splits = num_splits.max(1);
        let mid_out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads, head_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mid_max_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mid_sum_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut mid_out_ptr = dev_ptr(&mid_out_storage)?;
        let mut mid_max_ptr = dev_ptr(&mid_max_storage)?;
        let mut mid_sum_ptr = dev_ptr(&mid_sum_storage)?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_stage1 = HipDim3::new(num_heads as u32, num_splits as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut nh = num_heads as i32;
        let mut nkvh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut slen = kv_seq_len as i32;
        let mut nsplits = num_splits as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        // Stage 1
        let lds_stage1_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_flash_decode_stage1",
            grid_stage1,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mid_out_ptr),
                arg(&mut mid_max_ptr),
                arg(&mut mid_sum_ptr),
                arg(&mut nh),
                arg(&mut nkvh),
                arg(&mut hd),
                arg(&mut slen),
                arg(&mut nsplits),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_stage1_bytes,
        )?;

        // Stage 2
        let grid_stage2 = HipDim3::new(num_heads as u32, 1, 1);
        let mut optr = out_ptr;
        self.launch_compute_kernel(
            "grim_flash_decode_stage2",
            grid_stage2,
            block_dim,
            &mut [
                arg(&mut mid_out_ptr),
                arg(&mut mid_max_ptr),
                arg(&mut mid_sum_ptr),
                arg(&mut optr),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut nsplits),
            ],
        )
    }

    /// Launch DeepSeek Multi-Head Latent Attention (MLA) Matrix-Absorbed Decode.
    pub fn launch_mla_absorbed_decode(
        &self,
        q_absorbed: &RocmStorage,
        q_rope: &RocmStorage,
        kv_cache: &RocmStorage,
        w_uv: Option<&RocmStorage>,
        out: &RocmStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        seq_len: usize,
    ) -> Result<*mut c_void> {
        let q_abs_ptr = q_absorbed.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: q_absorbed has no device ptr".into())
        })?;
        let q_rope_ptr = q_rope.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: q_rope has no device ptr".into())
        })?;
        let kv_ptr = kv_cache.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: kv_cache has no device ptr".into())
        })?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: out has no device ptr".into()))?;
        let w_uv_ptr = w_uv.and_then(|s| s.device_ptr).unwrap_or(0);
        let has_w_uv = if w_uv.is_some() { 1i32 } else { 0i32 };

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(num_heads as u32, 1, 1);

        let mut qabsptr = q_abs_ptr;
        let mut qropeptr = q_rope_ptr;
        let mut kvptr = kv_ptr;
        let mut wuvptr = w_uv_ptr;
        let mut optr = out_ptr;
        let mut nh = num_heads as i32;
        let mut lora_r = kv_lora_rank as i32;
        let mut rope_d = qk_rope_dim as i32;
        let mut v_dim = v_head_dim as i32;
        let mut slen = seq_len as i32;
        let mut inv_sqrt = 1.0f32 / ((kv_lora_rank + qk_rope_dim) as f32).sqrt();
        let mut has_w = has_w_uv;

        let lds_bytes = 256 * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_mla_absorbed_decode",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qabsptr),
                arg(&mut qropeptr),
                arg(&mut kvptr),
                arg(&mut wuvptr),
                arg(&mut optr),
                arg(&mut nh),
                arg(&mut lora_r),
                arg(&mut rope_d),
                arg(&mut v_dim),
                arg(&mut slen),
                arg(&mut inv_sqrt),
                arg(&mut has_w),
            ],
            None,
            lds_bytes,
        )
    }

    /// Split-KV count for FlashDecoding: consult the autotuner (persisted in
    /// `.autotune_cache/{gpu_target}.json`) keyed by
    /// `(num_heads, head_dim, kv_len)`; on miss return the static heuristic.
    /// With `GRIM_ATTENTION_AUTOTUNE=1` (and outside stream capture), a miss
    /// instead benchmarks candidate split counts with real launches and
    /// records the winner — same treatment as GEMM tiles (ADR 0001 §5).
    #[allow(clippy::too_many_arguments)]
    fn flash_decode_split_count(
        &self,
        q_s: &RocmStorage,
        k_s: &RocmStorage,
        v_s: &RocmStorage,
        out: &RocmStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
    ) -> usize {
        let heuristic = (kv_seq_len / 256).clamp(2, 64);
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: "grim_flash_decode",
            gpu_arch: arch_leak,
            m: num_heads,
            n: head_dim,
            k: kv_seq_len.clamp(1, 1 << 16),
        };
        let Ok(mut tuner) = self.autotuner.lock() else {
            return heuristic;
        };
        if let Some(cfg) = tuner.lookup(key) {
            return cfg.tile_kv.max(1) as usize;
        }
        let tune_enabled = std::env::var("GRIM_ATTENTION_AUTOTUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !tune_enabled || self.active_capture_stream().is_some() {
            return heuristic;
        }

        // Bench candidate splits: real launches on the active stream, timed
        // wall-clock (launch + synchronize). Each candidate runs 3 iterations
        // after 1 warmup; the minimum wins.
        let mut candidates: Vec<usize> = [2usize, 4, 8, 16, 32, 64]
            .into_iter()
            .filter(|&s| s <= kv_seq_len.max(2))
            .collect();
        if !candidates.contains(&heuristic) {
            candidates.push(heuristic);
        }
        let mut best = (heuristic, f64::INFINITY);
        for &splits in &candidates {
            let mut best_ms = f64::INFINITY;
            for iter in 0..4 {
                if let Err(e) = self.launch_flash_decode(
                    q_s, k_s, v_s, out, num_heads, num_kv_heads, head_dim, kv_seq_len, splits,
                ) {
                    // A failing candidate (e.g. LDS overflow at high split
                    // counts) is simply not viable; skip it.
                    let _ = e;
                    best_ms = f64::INFINITY;
                    break;
                }
                if iter == 0 {
                    continue; // warmup
                }
                let t = std::time::Instant::now();
                if let Err(e) = self.launch_flash_decode(
                    q_s, k_s, v_s, out, num_heads, num_kv_heads, head_dim, kv_seq_len, splits,
                ) {
                    let _ = e;
                    best_ms = f64::INFINITY;
                    break;
                }
                let stream = self.active_stream();
                if unsafe { hipStreamSynchronize(stream) } != hipSuccess {
                    best_ms = f64::INFINITY;
                    break;
                }
                let ms = t.elapsed().as_secs_f64() * 1e3;
                best_ms = best_ms.min(ms);
            }
            if best_ms < best.1 {
                best = (splits, best_ms);
            }
        }
        if best.1.is_finite() {
            let cfg = crate::autotune::AutotuneConfig {
                block_dim: 256,
                tile_kv: best.0 as u32,
                grid_stride: 1,
                cycles_per_invocation: (best.1 * 1e6) as u64,
                spec_gamma: 4,
                spec_acceptance_threshold: 0.6,
                spec_alpha: 0.0,
            };
            let _ = tuner.record(key, cfg);
            let _ = self.save_autotune_cache(std::path::Path::new(&format!(
                ".autotune_cache/{}.json",
                self.gpu_target
            )));
            best.0
        } else {
            heuristic
        }
    }

    /// Launch Marlin-style Interleaved W4A16 GEMM.
    pub fn launch_marlin_gemm_w4a16(
        &self,
        a_storage: &RocmStorage,
        b_w4_storage: &RocmStorage,
        scales_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: a has no device ptr".into()))?;
        let b_ptr = b_w4_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: b has no device ptr".into()))?;
        let scales_ptr = scales_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: scales has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut sptr = scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut gs = group_size as i32;

        let kernel_name = if out_storage.dtype.arith == grim_tensor::ArithType::F16 {
            "grim_marlin_gemm_w4a16"
        } else {
            "grim_marlin_gemm_w4a16_f32"
        };

        self.launch_compute_kernel(
            kernel_name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut gs),
            ],
        )
    }

    /// Launch BitNet b1.58 Ternary GEMM (W1.58A8).
    pub fn launch_bitnet_gemm_w158a8(
        &self,
        a_storage: &RocmStorage,
        b_ternary_storage: &RocmStorage,
        scale_b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        scale_a: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: a has no device ptr".into()))?;
        let b_ptr = b_ternary_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: b has no device ptr".into()))?;
        let scale_b_ptr = scale_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: scale_b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut sbptr = scale_b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut sa = scale_a;

        self.launch_compute_kernel(
            "grim_bitnet_gemm_w158a8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sbptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
            ],
        )
    }

    /// Launch Extend Attention Chunk kernel across context slice [chunk_start, chunk_end).
    pub fn launch_extend_attention_chunk(
        &self,
        q_storage: &RocmStorage,
        k_cache_storage: &RocmStorage,
        v_cache_storage: &RocmStorage,
        chunk_out_storage: &RocmStorage,
        chunk_lse_storage: &RocmStorage,
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: q has no device ptr".into()))?;
        let k_ptr = k_cache_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: k has no device ptr".into()))?;
        let v_ptr = v_cache_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: v has no device ptr".into()))?;
        let out_ptr = chunk_out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: out has no device ptr".into()))?;
        let lse_ptr = chunk_lse_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("extend_attention: lse has no device ptr".into()))?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, num_heads as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut lseptr = lse_ptr;
        let mut nt = num_tokens as i32;
        let mut nh = num_heads as i32;
        let mut nkvh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut cstart = chunk_start as i32;
        let mut cend = chunk_end as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let lds_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_extend_attention_chunk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut lseptr),
                arg(&mut nt),
                arg(&mut nh),
                arg(&mut nkvh),
                arg(&mut hd),
                arg(&mut cstart),
                arg(&mut cend),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_bytes,
        )
    }

    /// Launch Log-Sum-Exp Attention State Merging kernel.
    pub fn launch_merge_attn_states(
        &self,
        out_a_storage: &RocmStorage,
        lse_a_storage: &RocmStorage,
        out_b_storage: &RocmStorage,
        lse_b_storage: &RocmStorage,
        out_merged_storage: &RocmStorage,
        lse_merged_storage: &RocmStorage,
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<*mut c_void> {
        let out_a_ptr = out_a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: out_a has no device ptr".into()))?;
        let lse_a_ptr = lse_a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: lse_a has no device ptr".into()))?;
        let out_b_ptr = out_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: out_b has no device ptr".into()))?;
        let lse_b_ptr = lse_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("merge_attn_states: lse_b has no device ptr".into()))?;
        let out_m_ptr = out_merged_storage.device_ptr.ok_or_else(|| {
            Error::Backend("merge_attn_states: out_merged has no device ptr".into())
        })?;
        let lse_m_ptr = lse_merged_storage.device_ptr.ok_or_else(|| {
            Error::Backend("merge_attn_states: lse_merged has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, num_heads as u32, 1);

        let mut a_ptr = out_a_ptr;
        let mut la_ptr = lse_a_ptr;
        let mut b_ptr = out_b_ptr;
        let mut lb_ptr = lse_b_ptr;
        let mut m_ptr = out_m_ptr;
        let mut lm_ptr = lse_m_ptr;
        let mut nt = num_tokens as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;

        self.launch_compute_kernel(
            "grim_merge_attn_states",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_ptr),
                arg(&mut la_ptr),
                arg(&mut b_ptr),
                arg(&mut lb_ptr),
                arg(&mut m_ptr),
                arg(&mut lm_ptr),
                arg(&mut nt),
                arg(&mut nh),
                arg(&mut hd),
            ],
        )
    }

    /// Launch Reshape and Cache into Preshuffled Layout kernel.
    pub fn launch_reshape_and_cache_preshuffled(
        &self,
        key_storage: &RocmStorage,
        value_storage: &RocmStorage,
        k_cache_storage: &RocmStorage,
        v_cache_storage: &RocmStorage,
        slot_mapping_storage: &RocmStorage,
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<*mut c_void> {
        let k_ptr = key_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("reshape_preshuffled: key has no device ptr".into()))?;
        let v_ptr = value_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("reshape_preshuffled: value has no device ptr".into()))?;
        let kc_ptr = k_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("reshape_preshuffled: k_cache has no device ptr".into())
        })?;
        let vc_ptr = v_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("reshape_preshuffled: v_cache has no device ptr".into())
        })?;
        let sm_ptr = slot_mapping_storage.device_ptr.ok_or_else(|| {
            Error::Backend("reshape_preshuffled: slot_mapping has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(head_dim as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, num_heads as u32, 1);

        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut kcptr = kc_ptr;
        let mut vcptr = vc_ptr;
        let mut smptr = sm_ptr;
        let mut nt = num_tokens as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;

        self.launch_compute_kernel(
            "grim_reshape_and_cache_preshuffled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut kcptr),
                arg(&mut vcptr),
                arg(&mut smptr),
                arg(&mut nt),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut bs),
            ],
        )
    }

    /// Launch Preshuffled Paged Attention Decode kernel.
    pub fn launch_preshuffled_paged_attention(
        &self,
        q_storage: &RocmStorage,
        k_cache_storage: &RocmStorage,
        v_cache_storage: &RocmStorage,
        block_tables_storage: &RocmStorage,
        context_lens_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_seqs: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("preshuffled_paged_attn: q has no device ptr".into()))?;
        let kc_ptr = k_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: k_cache has no device ptr".into())
        })?;
        let vc_ptr = v_cache_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: v_cache has no device ptr".into())
        })?;
        let bt_ptr = block_tables_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: block_tables has no device ptr".into())
        })?;
        let cl_ptr = context_lens_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: context_lens has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("preshuffled_paged_attn: out has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_dim = HipDim3::new(num_seqs as u32, num_heads as u32, 1);

        let mut qptr = q_ptr;
        let mut kcptr = kc_ptr;
        let mut vcptr = vc_ptr;
        let mut btptr = bt_ptr;
        let mut clptr = cl_ptr;
        let mut optr = out_ptr;
        let mut nseqs = num_seqs as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut max_b = max_num_blocks_per_seq as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let lds_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_preshuffled_paged_attention",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kcptr),
                arg(&mut vcptr),
                arg(&mut btptr),
                arg(&mut clptr),
                arg(&mut optr),
                arg(&mut nseqs),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut bs),
                arg(&mut max_b),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_bytes,
        )
    }

    /// Launch Multimodal 3D Rotary Position Embedding (M-RoPE) for Q and K tensors.
    pub fn launch_mrope_qk(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        positions_storage: &RocmStorage,
        num_tokens: usize,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        section_t: usize,
        section_h: usize,
        section_w: usize,
        rope_theta: f32,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: k has no device ptr".into()))?;
        let pos_ptr = positions_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: positions has no device ptr".into()))?;

        let block_dim = HipDim3::new((rotary_dim / 2) as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, (num_q_heads + num_k_heads) as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut posptr = pos_ptr;
        let mut nt = num_tokens as i32;
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_k_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut st = section_t as i32;
        let mut sh = section_h as i32;
        let mut sw = section_w as i32;
        let mut theta = rope_theta;

        self.launch_compute_kernel(
            "grim_mrope_qk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut posptr),
                arg(&mut nt),
                arg(&mut nqh),
                arg(&mut nkh),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut st),
                arg(&mut sh),
                arg(&mut sw),
                arg(&mut theta),
            ],
        )
    }

    /// Launch GPU Speculative Rejection Sampling kernel.
    pub fn launch_speculative_rejection_sample(
        &self,
        target_probs_storage: &RocmStorage,
        draft_probs_storage: &RocmStorage,
        draft_tokens_storage: &RocmStorage,
        uniform_rands_storage: &RocmStorage,
        accepted_tokens_storage: &RocmStorage,
        accepted_lens_storage: &RocmStorage,
        batch_size: usize,
        num_draft_tokens: usize,
        vocab_size: usize,
    ) -> Result<*mut c_void> {
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
        let at_ptr = accepted_tokens_storage.device_ptr.ok_or_else(|| {
            Error::Backend("spec_sample: accepted_tokens has no device ptr".into())
        })?;
        let al_ptr = accepted_lens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: accepted_lens has no device ptr".into()))?;

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(batch_size as u32, 1, 1);

        let mut tpptr = tp_ptr;
        let mut dpptr = dp_ptr;
        let mut dtptr = dt_ptr;
        let mut urptr = ur_ptr;
        let mut atptr = at_ptr;
        let mut alptr = al_ptr;
        let mut bs = batch_size as i32;
        let mut ndt = num_draft_tokens as i32;
        let mut vs = vocab_size as i32;

        self.launch_compute_kernel(
            "grim_speculative_rejection_sample",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut tpptr),
                arg(&mut dpptr),
                arg(&mut dtptr),
                arg(&mut urptr),
                arg(&mut atptr),
                arg(&mut alptr),
                arg(&mut bs),
                arg(&mut ndt),
                arg(&mut vs),
            ],
        )
    }

    /// Compute dynamic Expert Parallel Load Balancing (EPLB) greedy LPT placement.
    pub fn eplb_balance_experts(
        &self,
        expert_frequencies: &[f32],
        num_ranks: usize,
        replication_slots: usize,
    ) -> crate::device::eplb::EplbPackingPlan {
        crate::device::eplb::EplbRouter::balance_experts(
            expert_frequencies,
            num_ranks,
            replication_slots,
        )
    }

    /// Plan continuous batch reordering into [Decode : Extend : Prefill] partitions.
    pub fn reorder_batch(
        &self,
        sequences: &[crate::device::batch_orchestrator::SequenceMeta],
    ) -> crate::device::batch_orchestrator::ReorderedBatch {
        crate::device::batch_orchestrator::BatchReorderer::plan(sequences)
    }

    /// Launch the JIT compiled SplitK reduction kernel (WI-D).
    pub(crate) fn launch_split_k_reduction(
        &self,
        partials_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        split_k: u32,
    ) -> Result<*mut c_void> {
        let partials_ptr = partials_storage.device_ptr.ok_or_else(|| {
            Error::Backend("split_k_reduction: partials has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("split_k_reduction: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems = m * n;
        let grid_x = ((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut p_ptr = partials_ptr;
        let mut o_ptr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut sk = split_k as i32;

        self.launch_compute_kernel(
            "grim_split_k_reduction",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut p_ptr),
                arg(&mut o_ptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut sk),
            ],
        )
    }

    /// JIT-compile or query the cache, then launch the specified kernel on a [see: `entry`, `module_cache`]
    pub(crate) fn launch_compute_kernel(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
    ) -> Result<*mut c_void> {
        self.launch_compute_kernel_with_solution(entry, grid, block, args, None, 0)
    }

    /// JIT compile source or fetch cached binary. When a `HardwareSpec` is supplied, the
    /// cache key incorporates the hardware fingerprint (wavefront/lds/cu/mp/threads) via
    /// `JitCacheKey::from_spec`, so parametrized kernels for different hardware don't
    /// collide. Without a spec, falls back to the legacy (entry, arch, hash) key.
    pub fn jit_compile_or_cache(
        &self,
        source: &str,
        entry: &str,
        spec: Option<&crate::device::hardware_spec::HardwareSpec>,
    ) -> Result<(std::path::PathBuf, String)> {
        let hash = seahash::hash(source.as_bytes());
        let cache_key = if let Some(spec) = spec {
            crate::kernels::jit_cache::JitCacheKey::from_spec(entry, &self.gpu_target, spec, hash)
                .to_key_string()
        } else {
            format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash)
        };

        if let Some((cached_path, cached_lowered)) = self.hsaco_cache.get_cached_kernel(&cache_key)
        {
            Ok((cached_path, cached_lowered))
        } else {
            let (code, lowered) = jit_compile_hsaco(source, entry, &self.gpu_target)?;
            let p = self
                .hsaco_cache
                .cache_kernel(&cache_key, source, &code, &lowered)?;
            Ok((p, lowered))
        }
    }

    /// Benchmark kernel execution time in milliseconds using HIP events.
    /// Loads the module, resolves the entry, launches once on the device stream bracketed by
    /// start/stop events, and returns the elapsed GPU time. Falls back to a conservative
    /// constant if any HIP call fails so the FCP search still returns a valid winner.
    pub fn time_kernel_ms(
        &self,
        hsaco: &std::path::Path,
        lowered: &str,
        dims: crate::kernels::tile_picker::ShapeDims,
        cand: &crate::kernels::tile_picker::TileConfig,
    ) -> f64 {
        use crate::device::handles::{
            hipEventCreate, hipEventDestroy, hipEventElapsedTime, hipEventRecord,
            hipEventSynchronize, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad,
            hipModuleUnload,
        };
        let mut start_event: *mut c_void = std::ptr::null_mut();
        let mut stop_event: *mut c_void = std::ptr::null_mut();

        unsafe {
            if hipEventCreate(&mut start_event) != hipSuccess {
                return 0.5;
            }
            if hipEventCreate(&mut stop_event) != hipSuccess {
                let _ = hipEventDestroy(start_event);
                return 0.5;
            }

            let _ = hipEventRecord(start_event, std::ptr::null_mut());

            let grid = HipDim3::new(
                (dims.m + cand.grid_stride_m - 1) / cand.grid_stride_m,
                (dims.n + cand.grid_stride_n - 1) / cand.grid_stride_n,
                1,
            );
            let block = HipDim3::new(cand.threads, 1, 1);

            let path_c = match std::ffi::CString::new(hsaco.to_str().unwrap_or("")) {
                Ok(c) => c,
                Err(_) => {
                    let _ = hipEventDestroy(start_event);
                    let _ = hipEventDestroy(stop_event);
                    return 0.5;
                }
            };
            let entry_c = match std::ffi::CString::new(lowered) {
                Ok(c) => c,
                Err(_) => {
                    let _ = hipEventDestroy(start_event);
                    let _ = hipEventDestroy(stop_event);
                    return 0.5;
                }
            };

            let mut module: *mut c_void = std::ptr::null_mut();
            if hipModuleLoad(&mut module, path_c.as_ptr()) == hipSuccess {
                let mut func: *mut c_void = std::ptr::null_mut();
                if hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) == hipSuccess {
                    let mut dummy_args: [*mut c_void; 0] = [];
                    let _ = hipModuleLaunchKernel(
                        func,
                        grid.x,
                        grid.y,
                        grid.z,
                        block.x,
                        block.y,
                        block.z,
                        0,
                        std::ptr::null_mut(),
                        dummy_args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    );
                }
                let _ = hipModuleUnload(module);
            }

            let _ = hipEventRecord(stop_event, std::ptr::null_mut());
            let _ = hipEventSynchronize(stop_event);

            let mut elapsed_ms: f32 = 0.0;
            let status = hipEventElapsedTime(&mut elapsed_ms, start_event, stop_event);

            let _ = hipEventDestroy(start_event);
            let _ = hipEventDestroy(stop_event);

            if status == hipSuccess && elapsed_ms > 0.0 {
                elapsed_ms as f64
            } else {
                0.5
            }
        }
    }

    /// Store empirically discovered winning tile configuration into the autotuner cache.
    /// `winner_ms` is the measured GPU time of the winning candidate; persisted as
    /// `cycles_per_invocation` (ns-scale u64) so cached entries carry the measurement.
    /// Intern `s` as a `&'static str`, leaking each unique value exactly once.
    fn intern_str(&self, s: &str) -> &'static str {
        if let Ok(mut set) = self.str_interner.lock() {
            if let Some(existing) = set.get(s) {
                return existing;
            }
            let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
            set.insert(leaked);
            leaked
        } else {
            // Poisoned interner: fall back to a one-shot leak (correctness
            // over boundedness).
            Box::leak(s.to_string().into_boxed_str())
        }
    }

    pub fn store_tune_cache(
        &self,
        entry: &str,
        _spec: &crate::device::hardware_spec::HardwareSpec,
        dims: crate::kernels::tile_picker::ShapeDims,
        winner: &crate::kernels::tile_picker::TileConfig,
        winner_ms: f64,
    ) {
        let mut autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
        // &'static str keys via the interner — one leak per unique
        // (entry, arch) pair, not per call.
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let entry_leak: &'static str = self.intern_str(entry);
        let key = crate::autotune::KernelKey {
            kernel: entry_leak,
            gpu_arch: arch_leak,
            m: dims.m as usize,
            n: dims.n as usize,
            k: dims.k as usize,
        };
        let config = crate::autotune::AutotuneConfig {
            block_dim: winner.threads,
            tile_kv: winner.block_k,
            grid_stride: winner.grid_stride_m,
            cycles_per_invocation: (winner_ms * 1e6) as u64,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
        };
        let _ = autotuner.record(key, config);
    }

    /// Persist the in-memory autotune cache to a JSON file at `path`.
    pub fn save_autotune_cache(&self, path: &std::path::Path) -> Result<()> {
        let autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
        autotuner.save_to_file(path)
    }

    /// Return a fresh HardwareSpec snapshot describing this device.
    pub fn hardware_spec(&self) -> crate::device::hardware_spec::HardwareSpec {
        crate::device::hardware_spec::HardwareSpec::from(self)
    }

    /// Read-through tile-cache lookup. On a hit, maps the stored `AutotuneConfig` back to a
    /// `TileConfig`. On a miss, runs `fcp_fallback_tile_search` (compile + GPU-time a small
    /// constrained candidate set, keep the fastest), which self-persists the winner via
    /// `store_tune_cache` so subsequent calls for the same shape are a table hit, not a
    /// re-measure. This is the Phase 5 wiring that makes the empirical FCP search fire.
    pub fn get_or_tune_tiles(
        &self,
        entry: &str,
        spec: &crate::device::hardware_spec::HardwareSpec,
        dims: crate::kernels::tile_picker::ShapeDims,
        shape_class: crate::autotune::ShapeClass,
    ) -> crate::kernels::tile_picker::TileConfig {
        // Same interning as store_tune_cache — one leak per unique (entry, arch).
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let entry_leak: &'static str = self.intern_str(entry);
        let key = crate::autotune::KernelKey {
            kernel: entry_leak,
            gpu_arch: arch_leak,
            m: dims.m as usize,
            n: dims.n as usize,
            k: dims.k as usize,
        };

        // 1. Hot path: in-memory table hit.
        {
            let autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cfg) = autotuner.lookup(key) {
                return crate::kernels::tile_picker::TileConfig {
                    block_m: 0,
                    block_n: 0,
                    block_k: cfg.tile_kv,
                    split_k: 1,
                    grid_stride_m: cfg.grid_stride,
                    grid_stride_n: cfg.grid_stride,
                    lds_double_buffer: 64 * 1024
                        >= 2 * (2
                            * (cfg.tile_kv * (spec.wavefront_size.max(16))
                                + cfg.tile_kv * (spec.wavefront_size.max(16))
                                + (spec.wavefront_size.max(16)) * (spec.wavefront_size.max(16)))),
                    use_wmma: spec.gcn_arch.starts_with("gfx11")
                        || spec.gcn_arch.starts_with("gfx12"),
                    use_mfma: spec.gcn_arch.starts_with("gfx12")
                        || spec.gcn_arch.starts_with("gfx9"),
                    threads: cfg.block_dim,
                }
                .with_block_geometry(spec, shape_class);
            }
        }

        // 2. Cold path: empirical FCP search. Self-persists, so the next call hits step 1.
        crate::kernels::tile_picker::fcp_fallback_tile_search(self, spec, entry, dims, shape_class)
    }

    /// Op-tagged GEMM. `op` drives the `ShapeClass` via `ShapeClass::from_op`: `LmHead`
    /// selects the TLOLog tile arm (wide block_n for the vocab-dominated output column);
    /// everything else bins by M as before (from_op(Other, m) == from_m(m)).
    pub(crate) fn matmul_op(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        op: crate::autotune::GemmOp,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: matmul");

        // For matmul on GPU, both inputs must be RocmStorage (or we need to copy them to the device first)
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend("matmul: input a is not RocmStorage".into())),
        };

        let b_storage = match b.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend("matmul: input b is not RocmStorage".into())),
        };

        if !a_storage.device_ptr_is_valid() || !b_storage.device_ptr_is_valid() {
            return Err(Error::Backend(
                "matmul: inputs must have valid GPU device pointers".into(),
            ));
        }

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(Error::Shape("matmul expects inputs with rank >= 2".into()));
        }

        let k = a_dims[a_dims.len() - 1];
        let m = a.shape().elem_count() / k;

        let n = b_dims[b_dims.len() - 1];
        let k2 = b.shape().elem_count() / n;

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        if out_shape.elem_count() != m * n {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                m * n,
                out_shape.dims()
            )));
        }

        // Allocate output GPU storage with the actual input precision
        let dtype_out = DType {
            arith: a_storage.dtype.arith,
            storage: DTypeStorage::Native,
        };
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

        // Shape-indexed GEMM dispatch lookup (Tensile-inspired layout resolution).
        // Op-identity classifier: LmHead -> TLOLog tile arm; everything else bins by m.
        let shape_class = crate::autotune::ShapeClass::from_op(op, m);
        let tile_config =
            lookup_gemm_config_for_shape(m, n, k, self.props.wavefront_size, shape_class);
        // Offline-tuned solution_index per (M,N,K) for FP32. Falls back to 0 for [see: `examples/tune_gemm.rs`]
        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, dtype_out.arith);
        // WI 2.4.3 — split_k clamp gate.
        let split_k_effective: u32 = {
            let split_k_enabled = self.split_k_config.lock().unwrap_or_else(|e| e.into_inner()).enabled;
            if split_k_enabled
                && tile_config.split_k > 1
                && (k % tile_config.split_k as usize == 0)
                && (m > 1 || k > 8192)
            {
                tile_config.split_k
            } else {
                1
            }
        };

        if split_k_effective > 1 {
            let k_part = k / split_k_effective as usize;
            let partials_shape = Shape::from_slice(&[split_k_effective as usize, m, n]);
            let partials_storage = RocmStorage::alloc_gpu(
                &partials_shape,
                dtype_out.clone(),
                &self.allocator,
                self.ordinal,
            )?;

            let handle = self.get_rocblas_handle()?;
            // Bind the rocBLAS handle to the active stream so GEMM executes on the
            // correct stream and the returned ComputeHandle synchronizes correctly.
            // [P0-17 fix: previously missing — caused sync-lie and split-K race.]
            let _ = unsafe { rocblas_set_stream(handle, self.active_stream()) };
            let alpha: f32 = 1.0f32;
            let beta: f32 = 0.0f32;

            let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
            let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
            let partials_ptr_void = partials_storage.device_ptr.unwrap() as *mut c_void;

            let status = unsafe {
                let a_type = arith_to_rocblas_dtype(a_storage.dtype.arith);
                let b_type = arith_to_rocblas_dtype(b_storage.dtype.arith);
                let out_type = arith_to_rocblas_dtype(dtype_out.arith);
                let compute_type = arith_to_compute_dtype(dtype_out.arith);
                let alpha_ptr = &alpha as *const f32 as *const c_void;
                let beta_ptr = &beta as *const f32 as *const c_void;

                rocblas_gemm_strided_batched_ex(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k_part as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    n as RocblasInt,
                    (k_part * n) as i64,
                    a_ptr_void,
                    a_type,
                    k as RocblasInt,
                    k_part as i64,
                    beta_ptr,
                    partials_ptr_void,
                    out_type,
                    n as RocblasInt,
                    (m * n) as i64,
                    partials_ptr_void,
                    out_type,
                    n as RocblasInt,
                    (m * n) as i64,
                    split_k_effective as RocblasInt,
                    compute_type,
                    select_gemm_algo(solution_index),
                    solution_index,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            };

            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_strided_batched_ex failed with status {status}"
                )));
            }
            self.launch_counter.fetch_add(1, Ordering::SeqCst);

            // Sum up the partials along the batch dimension using the hand-written reduction kernel
            let stream = self.launch_split_k_reduction(
                &partials_storage,
                &out_storage,
                m,
                n,
                split_k_effective,
            )?;
            let compute_handle = Box::new(RocmHandle::new(Some(stream)));
            return Ok((Box::new(out_storage), compute_handle));
        }
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}, solution_index={}",
            m, n, k, tile_config, self.props.wavefront_size, solution_index
        );

        // ─── WI 2.4.4-2 — decode GEMM dispatch (opt-in, F16-only, m ≤ 8) ───── [see: `ck_gemm.cpp`, `grim_decode_gemm_f16`]
        {
            // Lock-free read via AtomicBool shadow — avoids Mutex acquisition on every matmul.
            // [see: `decode_gemm_enabled`, `set_decode_gemm_enabled`]
            if self.decode_gemm_enabled.load(Ordering::Relaxed)
                && dtype_out.arith == ArithType::F16
                && m <= 8
            {
                // WI 2.4.4-2(a) — thread the *real* enqueued stream into the [see: `launch_compute_kernel`, `hipModuleLaunchKernel`]
                let stream =
                    self.launch_decode_gemm_f16(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        }

        // ─── WI-G — WMMA GEMM dispatch (opt-in, F16-only) ─────
        {
            if self.should_use_wmma_path(None, dtype_out.arith) {
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        }

        // Get rocBLAS handle and execute sgemm. If handle is null (due to memory error fallback),
        // execute using WMMA HIP GEMM kernel directly.
        let handle = match self.get_rocblas_handle() {
            Ok(h) if !h.0.is_null() => h,
            _ => {
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        };

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;

        let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
        let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
        let out_ptr_void = out_storage.device_ptr.unwrap() as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is

        let use_gemm_ex = cfg!(feature = "rocm-aiter")
            || self.gpu_target == "gfx90a"
            || self.gpu_target == "gfx942";

        unsafe {
            let status = if use_gemm_ex
                || dtype_out.arith == ArithType::F16
                || dtype_out.arith == ArithType::BF16
            {
                let a_type = arith_to_rocblas_dtype(a_storage.dtype.arith);
                let b_type = arith_to_rocblas_dtype(b_storage.dtype.arith);
                let out_type = arith_to_rocblas_dtype(dtype_out.arith);
                let compute_type = arith_to_compute_dtype(dtype_out.arith);
                let alpha_ptr = &alpha as *const f32 as *const c_void;
                let beta_ptr = &beta as *const f32 as *const c_void;
                rocblas_gemm_ex(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    n as RocblasInt,
                    a_ptr_void,
                    a_type,
                    k as RocblasInt,
                    beta_ptr,
                    out_ptr_void,
                    out_type,
                    n as RocblasInt,
                    out_ptr_void,
                    out_type,
                    n as RocblasInt,
                    compute_type,
                    // Wire `lookup_solution_index` to `algo` so rocBLAS actually [see: `select_gemm_algo(0)`, `standard`]
                    select_gemm_algo(solution_index),
                    solution_index as RocblasInt,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            } else {
                rocblas_sgemm(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    &alpha,
                    b_ptr_void as *const f32,
                    n as RocblasInt,
                    a_ptr_void as *const f32,
                    k as RocblasInt,
                    &beta,
                    out_ptr_void as *mut f32,
                    n as RocblasInt,
                )
            };

            if status != rocblas_status_success {
                // If rocBLAS matmul returns an error (e.g. status 1 = invalid handle),
                // fall back seamlessly to WMMA HIP GEMM kernel.
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
            self.launch_counter.fetch_add(1, Ordering::SeqCst);
        };

        let compute_handle = Box::new(RocmHandle::new(Some(self.active_stream())));
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Public hook for the engine layer to tag the lm_head / logit-projection GEMM, so the
    /// dispatch layer classifies it as `ShapeClass::TLOLog` (op-identity) and selects the
    /// distinct wide-N tile regardless of M. This is the jit-mgpu.md §4.2 dispatch-layer tag.
    pub fn matmul_lm_head(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, crate::autotune::GemmOp::LmHead)
    }

    /// WI-F1 — Fused QKV projection GEMM. `qkv_weight` must be the load-time
    /// concatenation of the per-layer Q/K/V projection weights along the
    /// output dim — row-major `[hidden, q_dim + k_dim + v_dim]`, built once at
    /// model load via [`crate::fusion::concat_qkv_weights`] (never per forward
    /// pass). Produces all three projections in a single GEMM launch;
    /// downstream attention reads q/k/v by offset into the fused output, so
    /// the projection stage drops from 3 launches to 1 for every layer.
    pub fn fused_qkv_proj(
        &self,
        x: &dyn BackendStorage,
        qkv_weight: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_dims = x.shape().dims();
        let w_dims = qkv_weight.shape().dims();
        if x_dims.len() < 2 || w_dims.len() < 2 {
            return Err(Error::Shape(
                "fused_qkv_proj expects rank >= 2 inputs".into(),
            ));
        }
        let hidden = x_dims[x_dims.len() - 1];
        let qkv_dim = w_dims[w_dims.len() - 1];
        let k2 = qkv_weight.shape().elem_count() / qkv_dim;
        if hidden != k2 {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_dims.to_vec(),
            });
        }
        let tokens = x.shape().elem_count() / hidden;
        if out_shape.elem_count() != tokens * qkv_dim {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                tokens * qkv_dim,
                out_shape.dims()
            )));
        }
        self.matmul_op(x, qkv_weight, out_shape, crate::autotune::GemmOp::Attention)
    }

    /// WI-F2 — Fused attention output projection. Runs the same fused QKV
    /// attention kernel (`grim_qkv_attention`) with the O-projection applied
    /// in the kernel epilogue: each head's normalized attention vector is
    /// multiplied by its slice of `o_proj` (row-major
    /// `[num_heads*head_dim, o_dim]`) and accumulated across heads in-kernel,
    /// avoiding the HBM round-trip of writing per-head attention output and
    /// re-reading it for a separate GEMM. `out_shape` is `[seq_len, o_dim]`.
    /// The unfused path (`qkv_attention` + matmul) remains the reference and
    /// fallback; arch gating of this fusion is a follow-up under green.
    pub fn fused_attn_o_proj(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        o_proj: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_dims = q.shape().dims();
        let o_dims = out_shape.dims();
        if q_dims.len() != 3 || o_dims.len() != 2 {
            return Err(Error::Shape(
                "fused_attn_o_proj expects q [seq, heads, head_dim] and out [seq, o_dim]".into(),
            ));
        }
        let seq_len = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];
        let o_dim = o_dims[1];
        if o_dims[0] != seq_len {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: out rows {} must equal seq_len {seq_len}",
                o_dims[0]
            )));
        }
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || o_dim == 0 {
            return Err(Error::Shape(
                "fused_attn_o_proj: zero-sized heads / head_dim / o_dim".into(),
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj supports head_dim <= 256 (got {head_dim})"
            )));
        }
        let o_s = as_rocm(o_proj)?;
        if o_proj.shape().elem_count() != num_heads * head_dim * o_dim {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: o_proj must be [num_heads*head_dim, o_dim] = {} elems (got {})",
                num_heads * head_dim * o_dim,
                o_proj.shape().elem_count()
            )));
        }
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !o_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_attn_o_proj: inputs lack a valid device pointer".into(),
            ));
        }

        let config = QkvAttentionFusionConfig {
            enabled: true,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: seq_len,
            wavefront_size: self.props.wavefront_size as u32,
            quant_mode: QuantMode::Fp32,
        };
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;

        // atomicAdd accumulation across heads requires a zeroed output;
        // async memset on the active stream keeps it stream-ordered.
        let res = unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                storage.bytes,
                self.active_stream(),
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "fused_attn_o_proj: hipMemsetAsync failed with status {res}"
            )));
        }

        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let v_ptr = dev_ptr(v_s)?;
        let o_proj_ptr = dev_ptr(o_s)?;

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut max_ptr: u64 = 0;
        let mut sum_ptr: u64 = 0;
        let mut nh = num_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sl = seq_len as i32;
        let mut ksl = kv_seq_len as i32;
        let mut co = cache_offset as i32;
        let mut isd: f32 = 1.0 / (head_dim as f32).sqrt();
        let mut wlo: i32 = 0;
        let mut oproj_ptr = o_proj_ptr;
        let mut odim = o_dim as i32;
        let mut fuseo: i32 = 1;
        let mut alibi_ptr: u64 = 0;
        let mut has_alibi: i32 = 0;

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;
        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, wlo,
            oproj_ptr, odim, fuseo, alibi_ptr, has_alibi,
        );
        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    pub(crate) fn launch_compute_kernel_with_solution(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
        solution_index: Option<i32>,
        shared_mem_bytes: usize,
    ) -> Result<*mut c_void> {
        // Fast path: a previously resolved hipFunction for this
        // (entry, grid-shape, solution_index) launches directly — no source
        // rebuild, no seahash, no CString, no module-cache walk. Same
        // solution_index is required because different indices map to
        // different on-disk hsaco files (cache_key includes _sol{N}).
        let fast_key = (entry.to_string(), grid.x, grid.y, solution_index);
        let cached_func: Option<*mut c_void> = self
            .resolved_kernel_cache
            .lock()
            .ok()
            .and_then(|c| c.get(&fast_key).copied());
        if let Some(func) = cached_func {
            if !func.is_null() {
                let stream = self.active_stream();
                let args_ptr = args.as_mut_ptr();
                check_hip("hipModuleLaunchKernel (cached)", unsafe {
                    hipModuleLaunchKernel(
                        func,
                        grid.x,
                        grid.y,
                        grid.z,
                        block.x,
                        block.y,
                        block.z,
                        shared_mem_bytes as u32,
                        stream,
                        args_ptr,
                        std::ptr::null_mut(),
                    )
                })?;
                return Ok(stream);
            }
        }

        // Build the kernel source. Under `jit-hw-adaptive`, inject hardware-specific #defines
        // (wavefront/LDS/CU + tile geometry) via `compute_kernel_source_with_spec` and route the
        // compile through the fingerprinted `jit_compile_or_cache`. Otherwise fall back to the
        // static source + legacy cache key.
        #[cfg(feature = "jit-hw-adaptive")]
        let (path, lowered_name, cache_key) = {
            let spec = self.hardware_spec();
            // `launch_compute_kernel` is a generic launcher (no GEMM M/N/K in its signature), so
            // infer a coarse (m, n) from the grid dims; the per-op TLOLog tagging is handled at
            // the `matmul_op` layer, not here. K is unknown to the generic launcher; use a
            // conservative default — split-K is derived from the real K inside `pick_tiles`.
            let (m_val, n_val) = if grid.y > 1 { (grid.x, grid.y) } else { (1, 1) };
            let shape_class = crate::autotune::ShapeClass::from_m(m_val as usize);
            let dims = crate::kernels::tile_picker::ShapeDims::new(m_val, n_val, 64);
            let kernel_source = crate::kernels::source_asm::compute_kernel_source_with_spec(
                &spec,
                entry,
                shape_class,
                dims,
                0,
                1,
                None,
            );
            let (p, lowered) = self.jit_compile_or_cache(&kernel_source, entry, Some(&spec))?;
            let mut key = format!(
                "grim_{}_{}_{:016x}",
                entry,
                self.gpu_target,
                seahash::hash(kernel_source.as_bytes())
            );
            if let Some(sol) = solution_index {
                key = format!("{}_sol{}", key, sol);
            }
            (p, lowered, key)
        };

        #[cfg(not(feature = "jit-hw-adaptive"))]
        let (path, lowered_name, cache_key) = {
            let kernel_source = crate::kernels::source_asm::compute_kernel_source();
            let hash = seahash::hash(kernel_source.as_bytes());
            let base_key = format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash);
            let cache_key = if let Some(sol) = solution_index {
                format!("{}_sol{}", base_key, sol)
            } else {
                base_key
            };
            let (path, lowered_name) = if let Some((cached_path, cached_lowered)) =
                self.hsaco_cache.get_cached_kernel(&cache_key)
            {
                (cached_path, cached_lowered)
            } else {
                let (code, lowered) = jit_compile_hsaco(&kernel_source, entry, &self.gpu_target)?;
                let p =
                    self.hsaco_cache
                        .cache_kernel(&cache_key, &kernel_source, &code, &lowered)?;
                (p, lowered)
            };
            (path, lowered_name, cache_key)
        };

        let path_c = std::ffi::CString::new(path.to_str().unwrap_or(""))
            .map_err(|e| Error::Backend(format!("hsaco path CString: {}", e)))?;
        let entry_c = std::ffi::CString::new(lowered_name.as_str())
            .map_err(|e| Error::Backend(format!("entry CString: {}", e)))?;

        // Load the HIP module once per unique kernel; reuse the cached module +
        // Pin the current device to self.ordinal before loading: the JIT pipeline
        // queries CapabilityProfiler which sweeps every device and can leave the
        // thread on a foreign ordinal. Loading a gfx1201 hsaco on gfx1200 yields
        // HIP error 209 (no binary for device).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut module_cache = self.module_cache.lock().unwrap_or_else(|e| e.into_inner());
        let (_module, func) = if let Some(cached) = module_cache.get(&cache_key) {
            let (m, f) = *cached;
            if let Ok(mut fast) = self.resolved_kernel_cache.lock() {
                fast.insert(fast_key, f);
            }
            (m, f)
        } else {
            let mut module: *mut c_void = std::ptr::null_mut();
            let load_res = unsafe { hipModuleLoad(&mut module, path_c.as_ptr()) };
            if load_res != hipSuccess {
                return Err(Error::Backend(format!(
                    "hipModuleLoad failed: {load_res} (entry={entry}, path={}, gpu_target={})",
                    path.display(),
                    self.gpu_target
                )));
            }
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = unsafe { hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) };
            if res != hipSuccess {
                unsafe {
                    hipModuleUnload(module);
                }
                return Err(Error::Backend(format!(
                    "hipModuleGetFunction failed: {}",
                    res
                )));
            }
            self.module_load_count.fetch_add(1, Ordering::SeqCst);
            module_cache.insert(cache_key, (module, func));
            if let Ok(mut fast) = self.resolved_kernel_cache.lock() {
                fast.insert(fast_key, func);
            }
            (module, func)
        };
        drop(module_cache);

        let stream = self.active_stream();

        let args_ptr = args.as_mut_ptr();
        check_hip("hipModuleLaunchKernel", unsafe {
            hipModuleLaunchKernel(
                func,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                shared_mem_bytes as u32,
                stream,
                args_ptr,
                std::ptr::null_mut(),
            )
        })?;
        self.launch_counter.fetch_add(1, Ordering::SeqCst);
        Ok(stream)
    }

    /// Dispatch a fused RMSNorm + MatMul operation onto the GPU.
    pub fn rmsnorm_matmul(
        &self,
        x: &dyn BackendStorage,
        w_norm: &dyn BackendStorage,
        weight_mat: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_norm_s = as_rocm(w_norm)?;
        let w_mat_s = as_rocm(weight_mat)?;
        if !x_s.device_ptr_is_valid()
            || !w_norm_s.device_ptr_is_valid()
            || !w_mat_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "rmsnorm_matmul: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        let w_mat_dims = weight_mat.shape().dims();
        if x_dims.len() < 2 || w_mat_dims.len() < 2 {
            return Err(Error::Shape(
                "rmsnorm_matmul expects rank >= 2 inputs".into(),
            ));
        }
        let k = x_dims[x_dims.len() - 1];
        let m = x.shape().elem_count() / k;
        let n = w_mat_dims[w_mat_dims.len() - 1];
        let k2 = weight_mat.shape().elem_count() / n;
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_mat_dims.to_vec(),
            });
        }
        if out_shape.elem_count() != m * n {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                m * n,
                out_shape.dims()
            )));
        }

        let config = RmsNormMatMulFusionConfig {
            hidden_size: k,
            intermediate_size: n,
            wavefront_size: self.props.wavefront_size as u32,
            lds_size: 65536,
        };
        let launch = config.hip_launch_params();

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_norm_ptr = dev_ptr(w_norm_s)?;
        let mut w_mat_ptr = dev_ptr(w_mat_s)?;
        let mut m_i = m as i32;
        let mut n_i = n as i32;
        let mut k_i = k as i32;
        let mut eps_f = eps;

        self.launch_compute_kernel(
            "grim_rmsnorm_matmul",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_norm_ptr),
                arg(&mut w_mat_ptr),
                arg(&mut out_ptr),
                arg(&mut m_i),
                arg(&mut n_i),
                arg(&mut k_i),
                arg(&mut eps_f),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// High-level fused RMSNorm + MXFP4 GEMM (e.g. for MLP gate/up/down projections).
    pub fn fused_rmsnorm_mxfp4_gemm(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let gamma_s = as_rocm(gamma)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;

        let out_shape = Shape::new(vec![m, n]);
        let out_storage =
            RocmStorage::alloc_gpu(&out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        self.launch_fused_rmsnorm_mxfp4_gemm(
            x_s,
            gamma_s,
            w_codes_s,
            w_exps_s,
            &out_storage,
            m,
            n,
            k,
            eps,
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// High-level fused RMSNorm + MXFP4 GEMM + RoPE + direct KV cache scatter.
    pub fn fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let x_s = as_rocm(x)?;
        let gamma_s = as_rocm(gamma)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;

        let q_out_s = match q_out {
            Some(q) => Some(as_rocm(q)?),
            None => None,
        };
        let k_cache_s = match k_cache {
            Some(k) => Some(as_rocm(k)?),
            None => None,
        };
        let v_cache_s = match v_cache {
            Some(v) => Some(as_rocm(v)?),
            None => None,
        };
        let out_all_s = match out_all {
            Some(a) => Some(as_rocm(a)?),
            None => None,
        };
        let positions_s = match positions {
            Some(p) => Some(as_rocm(p)?),
            None => None,
        };
        let inv_freq_s = match inv_freq {
            Some(f) => Some(as_rocm(f)?),
            None => None,
        };

        self.launch_fused_rmsnorm_mxfp4_gemm_rope_kv(
            x_s,
            gamma_s,
            w_codes_s,
            w_exps_s,
            q_out_s,
            k_cache_s,
            v_cache_s,
            out_all_s,
            positions_s,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq_s,
            mscale,
            eps,
            max_seq_len,
        )?;

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// LFM2-style fused QKV projection: MXFP4 GEMM (C = x @ W_qkv) followed by
    /// per-head QK-Norm + RoPE (YaRN-aware). Mirrors `fused_rmsnorm_mxfp4_gemm_rope_kv`
    /// but applies the normalization *after* the projection (QK-norm) instead of
    /// before it, matching QK-norm attention (e.g. LFM2). The input `x` is expected
    /// to already carry any pre-attention RMSNorm the model applies.
    pub fn fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma_q: &dyn BackendStorage,
        gamma_k: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let x_s = as_rocm(x)?;
        let gamma_q_s = as_rocm(gamma_q)?;
        let gamma_k_s = as_rocm(gamma_k)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;
        let q_out_s = q_out.map(|q| as_rocm(q)).transpose()?;
        let k_cache_s = k_cache.map(|k| as_rocm(k)).transpose()?;
        let v_cache_s = v_cache.map(|v| as_rocm(v)).transpose()?;
        let out_all_s = out_all.map(|a| as_rocm(a)).transpose()?;
        let positions_s = positions.map(|p| as_rocm(p)).transpose()?;
        let inv_freq_s = inv_freq.map(|f| as_rocm(f)).transpose()?;

        self.launch_fused_mxfp4_gemm_qk_norm_rope_kv(
            x_s,
            gamma_q_s,
            gamma_k_s,
            w_codes_s,
            w_exps_s,
            q_out_s,
            k_cache_s,
            v_cache_s,
            out_all_s,
            positions_s,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq_s,
            mscale,
            eps,
            max_seq_len,
        )?;

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// Fused Add + RMSNorm kernel.
    /// Computes `y = x + residual` and `norm_out = RMSNorm(y, weight, eps)` in a single HIP kernel pass.
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
        let x_s = as_rocm(x)?;
        let res_s = as_rocm(residual)?;
        let w_s = as_rocm(weight)?;
        if !x_s.device_ptr_is_valid() || !res_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_add_rms_norm: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("fused_add_rms_norm: empty input".into()));
        }
        let row_len = *x_dims.last().unwrap();
        let total = out_shape.elem_count();
        let y_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let norm_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut res_ptr = dev_ptr(res_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut y_out_ptr = dev_ptr(&y_storage)?;
        let mut norm_out_ptr = dev_ptr(&norm_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        // grim_add_rms_norm is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_add_rms_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut res_ptr),
                arg(&mut w_ptr),
                arg(&mut y_out_ptr),
                arg(&mut norm_out_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(y_storage),
            Box::new(norm_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch the Design-A on-device linear cross-entropy forward pass.
    pub fn fused_linear_cross_entropy_forward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        v_tile_size: i32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let h = as_rocm(hidden)?;
        let w = as_rocm(lm_head)?;
        let t = as_rocm(targets)?;
        if !h.device_ptr_is_valid() || !w.device_ptr_is_valid() || !t.device_ptr_is_valid() {
            return Err(Error::Backend(
                "fused_linear_ce: invalid input pointer".into(),
            ));
        }
        let hd = hidden.shape().dims();
        let wd = lm_head.shape().dims();
        let td = targets.shape().dims();
        if hd.len() != 2 || wd.len() != 2 || td.len() != 1 || td[0] != hd[0] || wd[1] != hd[1] {
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
        let loss = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let lse = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mut hp = dev_ptr(h)?;
        let mut wp = dev_ptr(w)?;
        let mut tp = dev_ptr(t)?;
        let mut lp = dev_ptr(&loss)?;
        let mut ep = dev_ptr(&lse)?;
        let mut k = hd[1] as i32;
        let mut v = wd[0] as i32;
        let mut tile = v_tile_size;
        let mut b = batch as i32;
        let block = crate::HipDim3 { x: 256, y: 1, z: 1 };
        let grid = crate::HipDim3 {
            x: batch as u32,
            y: 1,
            z: 1,
        };
        self.launch_compute_kernel(
            "grim_fused_linear_ce_forward",
            grid,
            block,
            &mut [
                arg(&mut hp),
                arg(&mut wp),
                arg(&mut tp),
                arg(&mut lp),
                arg(&mut ep),
                arg(&mut k),
                arg(&mut v),
                arg(&mut tile),
                arg(&mut b),
            ],
        )?;
        Ok((
            Box::new(loss),
            Box::new(lse),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch the Design-A on-device linear cross-entropy backward pass.
    pub fn fused_linear_cross_entropy_backward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        lse: &dyn BackendStorage,
        v_tile_size: i32,
        inv_batch: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let h = as_rocm(hidden)?;
        let w = as_rocm(lm_head)?;
        let t = as_rocm(targets)?;
        let e = as_rocm(lse)?;
        if !h.device_ptr_is_valid()
            || !w.device_ptr_is_valid()
            || !t.device_ptr_is_valid()
            || !e.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_linear_ce: invalid input pointer".into(),
            ));
        }
        let hd = hidden.shape().dims();
        let wd = lm_head.shape().dims();
        if hd.len() != 2 || wd.len() != 2 || wd[1] != hd[1] || targets.shape().elem_count() != hd[0]
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
        let grad =
            RocmStorage::alloc_gpu(hidden.shape(), dtype_f32(), &self.allocator, self.ordinal)?;
        let mut hp = dev_ptr(h)?;
        let mut wp = dev_ptr(w)?;
        let mut tp = dev_ptr(t)?;
        let mut ep = dev_ptr(e)?;
        let mut gp = dev_ptr(&grad)?;
        let mut k = hd[1] as i32;
        let mut v = wd[0] as i32;
        let mut tile = v_tile_size;
        let mut inv = inv_batch;
        let mut b = batch as i32;
        let block = crate::HipDim3 { x: 256, y: 1, z: 1 };
        let grid = crate::HipDim3 {
            x: batch as u32,
            y: 1,
            z: 1,
        };
        self.launch_compute_kernel(
            "grim_fused_linear_ce_backward",
            grid,
            block,
            &mut [
                arg(&mut hp),
                arg(&mut wp),
                arg(&mut tp),
                arg(&mut ep),
                arg(&mut gp),
                arg(&mut k),
                arg(&mut v),
                arg(&mut tile),
                arg(&mut inv),
                arg(&mut b),
            ],
        )?;
        Ok((
            Box::new(grad),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch one bounded Scythe persistent worker. The worker is intentionally
    /// launched as a single 128-thread block: the callable Charon device
    /// function uses that block cooperatively and processes the complete task.
    pub fn launch_scythe_persistent_dispatch(
        &self,
        slots: &dyn BackendStorage,
        capacity: u32,
        tail: &dyn BackendStorage,
        head: &dyn BackendStorage,
        stop: &dyn BackendStorage,
        max_tasks: u32,
    ) -> Result<Box<dyn ComputeHandle>> {
        let mut slots_ptr = dev_ptr(as_rocm(slots)?)?;
        let mut tail_ptr = dev_ptr(as_rocm(tail)?)?;
        let mut head_ptr = dev_ptr(as_rocm(head)?)?;
        let mut stop_ptr = dev_ptr(as_rocm(stop)?)?;
        let mut cap = capacity;
        let mut limit = max_tasks;
        self.launch_compute_kernel(
            "grim_scythe_persistent_dispatch",
            crate::HipDim3::new(1, 1, 1),
            crate::HipDim3::new(128, 1, 1),
            &mut [
                arg(&mut slots_ptr),
                arg(&mut cap),
                arg(&mut tail_ptr),
                arg(&mut head_ptr),
                arg(&mut stop_ptr),
                arg(&mut limit),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// Launch standalone Q8_0 quantization HIP kernel.
    pub fn launch_quant_q8_0(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let n_blocks = (total + 31) / 32;
        let grid = crate::HipDim3 {
            x: n_blocks as u32,
            y: 1,
            z: 1,
        };
        let block = crate::HipDim3 { x: 32, y: 1, z: 1 };
        let mut x_ptr = dev_ptr(x)?;
        let mut out_ptr = dev_ptr(out)?;
        let mut total_i = total as i32;

        self.launch_compute_kernel(
            "grim_quant_q8_0",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut total_i)],
        )
    }

    /// Launch standalone FP8 E4M3 quantization HIP kernel.
    pub fn launch_quant_fp8(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let (grid, block) = linear_launch(total);
        let mut x_ptr = dev_ptr(x)?;
        let mut out_ptr = dev_ptr(out)?;
        let mut total_i = total as i32;

        self.launch_compute_kernel(
            "grim_quant_fp8",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut total_i)],
        )
    }

    /// Quantize F32 tensor `x` on-device to `format`.
    pub fn quantize_on_device(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "quantize_on_device: input lacks valid device pointer".into(),
            ));
        }
        let total = x.shape().elem_count();
        use grim_tensor::{FloatPackScheme, KQuantScheme, QuantFormat};
        let (out_bytes, output_dtype) = match format {
            QuantFormat::Q8_0 => {
                let n_blocks = (total + 31) / 32;
                (
                    n_blocks * 34,
                    DType {
                        arith: ArithType::F32,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                )
            }
            QuantFormat::Fp8 => (
                4 + total,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                },
            ),
            other => {
                return Err(Error::Backend(format!(
                    "quantize_on_device: unsupported format {:?}",
                    other
                )));
            }
        };

        let out_shape = x.shape().clone();
        let out_storage = RocmStorage::alloc_gpu_with_bytes(
            &out_shape,
            output_dtype,
            out_bytes,
            &self.allocator,
            self.ordinal,
        )?;

        let stream = match format {
            QuantFormat::Q8_0 => self.launch_quant_q8_0(x_s, &out_storage, total)?,
            QuantFormat::Fp8 => self.launch_quant_fp8(x_s, &out_storage, total)?,
            _ => unreachable!(),
        };

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(stream))),
        ))
    }

    // NOTE: `qkv_attention` was promoted to the `BackendDevice` trait [see: `impl BackendDevice for RocmDevice`]

    /// Fused KV-dequant-attention (WI-R5). [see: `CompressedKvBlock`, `quant_bits`, `k_tensor`, `v_tensor`]
    pub fn kv_dequant_attention_impl(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_format: crate::fusion::KvQuantFormat,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let config = {
            let out_dims = out_shape.dims();
            if out_dims.len() != 3 {
                return Err(Error::Shape(
                    "kv_dequant_attention expects 3-D output shape [seq_len, num_heads, head_dim]"
                        .into(),
                ));
            }
            crate::fusion::KvDequantAttentionConfig {
                enabled: true,
                num_heads: out_dims[1],
                num_kv_heads,
                head_dim: out_dims[2],
                quant_format,
                wavefront_size: self.props.wavefront_size as u32,
            }
        };
        if !config.enabled {
            return Err(Error::Backend(
                "kv_dequant_attention: kernel is gated (KvDequantAttentionConfig.enabled=false)"
                    .into(),
            ));
        }

        if config.num_heads == 0 || config.num_kv_heads == 0 || config.head_dim == 0 {
            return Err(Error::Shape(
                "kv_dequant_attention: zero-sized num_heads / num_kv_heads / head_dim".into(),
            ));
        }
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "kv_dequant_attention: num_heads ({}) must be a multiple of num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            )));
        }
        if config.head_dim > 256 {
            return Err(Error::Shape(format!(
                "kv_dequant_attention supports head_dim <= 256 (got {})",
                config.head_dim
            )));
        }

        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k_tensor)?;
        let ks_s = as_rocm(k_scales)?;
        let v_s = as_rocm(v_tensor)?;
        let vs_s = as_rocm(v_scales)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !ks_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !vs_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "kv_dequant_attention: an input lacks a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];

        // One block per (seq_position, head); block dim 128 for wave32 or 256 for wave64
        let block_dim_x: u32 = if config.wavefront_size == 32 {
            128
        } else {
            256
        };
        let grid_x = (seq_len * config.num_heads) as u32;
        let grid_y = 1u32;
        let shared_mem_bytes = (config.head_dim * 4).min(32768);
        let launch = crate::fusion::HipKernelLaunch {
            grid_dim: HipDim3::new(grid_x, grid_y, 1),
            block_dim: HipDim3::new(block_dim_x, 1, 1),
            shared_mem_bytes,
        };

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;
        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let ks_ptr = dev_ptr(ks_s)?;
        let v_ptr = dev_ptr(v_s)?;
        let vs_ptr = dev_ptr(vs_s)?;

        let num_heads_i = config.num_heads as i32;
        let num_kv_heads_i = config.num_kv_heads as i32;
        let head_dim_i = config.head_dim as i32;
        let seq_len_i = seq_len as i32;
        let kv_seq_len_i = kv_seq_len as i32;
        let cache_offset_i = cache_offset as i32;
        let inv_sqrt_d: f32 = 1.0 / (config.head_dim as f32).sqrt();
        let mut inv_sqrt_d_bits = inv_sqrt_d.to_bits();
        let inv_sqrt_d_ptr = &mut inv_sqrt_d_bits as *mut u32 as *mut f32;
        let inv_sqrt_d_stable = inv_sqrt_d_ptr;
        let quant_bits_i = quant_bits as i32;
        let quant_format_i = config.quant_format.kernel_arg() as i32;

        let mut qp = q_ptr;
        let mut kp = k_ptr;
        let mut ksp = ks_ptr;
        let mut vp = v_ptr;
        let mut vsp = vs_ptr;
        let mut op = out_ptr;
        let mut nh = num_heads_i;
        let mut nkv = num_kv_heads_i;
        let mut hd = head_dim_i;
        let mut sl = seq_len_i;
        let mut ksl = kv_seq_len_i;
        let mut co = cache_offset_i;
        let mut isd = inv_sqrt_d;
        let mut qb = quant_bits_i;
        let mut qf = quant_format_i;

        let stream = self.launch_compute_kernel(
            "grim_kv_dequant_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qp),
                arg(&mut kp),
                arg(&mut ksp),
                arg(&mut vp),
                arg(&mut vsp),
                arg(&mut op),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut qb),
                arg(&mut qf),
            ],
        )?;

        let _ = (
            qp,
            kp,
            ksp,
            vp,
            vsp,
            op,
            nh,
            nkv,
            hd,
            sl,
            ksl,
            co,
            isd,
            qb,
            inv_sqrt_d_stable,
        );

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// Cross-entropy loss + softmax gradient, host-staged. Returns `(avg_loss, grad)`.
    pub fn cross_entropy_gpu(
        &self,
        logits: &dyn BackendStorage,
        targets: &[usize],
        label_smoothing: Option<f32>,
    ) -> Result<(f32, Box<dyn BackendStorage>)> {
        let l_s = as_rocm(logits)?;
        if !l_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "cross_entropy_gpu: logits lack a valid device pointer".into(),
            ));
        }

        let dims = logits.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Shape(
                "cross_entropy_gpu: logits must be 2-D [batch_size, vocab_size]".into(),
            ));
        }
        let batch_size = dims[0];
        let vocab_size = dims[1];
        if batch_size == 0 {
            return Err(Error::Backend(
                "cross_entropy_gpu: batch_size must be > 0".into(),
            ));
        }
        if targets.len() != batch_size {
            return Err(Error::Shape(format!(
                "cross_entropy_gpu: targets len {} != batch_size {}",
                targets.len(),
                batch_size
            )));
        }
        let smooth = label_smoothing.unwrap_or(0.0).clamp(0.0, 1.0);
        let uniform = smooth / (vocab_size as f32);
        let confident = 1.0 - smooth;

        let logits_vec = logits.to_cpu_vec_f32()?;
        if logits_vec.len() < batch_size * vocab_size {
            return Err(Error::Backend(format!(
                "cross_entropy_gpu: logits length {} < batch_size * vocab_size {}",
                logits_vec.len(),
                batch_size * vocab_size
            )));
        }

        let mut grad_vec = vec![0.0f32; batch_size * vocab_size];
        let mut total_loss = 0.0f32;
        let inv_batch = 1.0 / (batch_size as f32);

        for b in 0..batch_size {
            let target_token = targets[b];
            if target_token >= vocab_size {
                return Err(Error::Backend(format!(
                    "cross_entropy_gpu: target token {} out of bounds for vocab_size {}",
                    target_token, vocab_size
                )));
            }

            let row_start = b * vocab_size;
            let row_logits = &logits_vec[row_start..row_start + vocab_size];

            // Max trick for numerical stability.
            let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exp_logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let exp_val = (row_logits[v] - max_logit).exp();
                exp_logits[v] = exp_val;
                sum_exp += exp_val;
            }
            let log_sum_exp = max_logit + sum_exp.ln();

            // Cross-entropy with optional label smoothing:
            //   loss = -sum_v q(v) * log_softmax(v)
            // where log_softmax(v) = row_logits[v] - log_sum_exp, and the
            // target distribution is q(target) = confident, q(other) = uniform.
            // This collapses to: confident * (log_sum_exp - logit_target)
            //   + uniform * (vocab_size * log_sum_exp - sum(row_logits)).
            let log_target = log_sum_exp - row_logits[target_token];
            let sum_logits: f32 = row_logits.iter().sum();
            let smooth_loss = uniform * ((vocab_size as f32) * log_sum_exp - sum_logits);
            total_loss += confident * log_target + smooth_loss;

            // Gradient dL/dLogits = (softmax - q) / batch_size.
            for v in 0..vocab_size {
                let prob = exp_logits[v] / sum_exp;
                let target_q = if v == target_token {
                    confident + uniform
                } else {
                    uniform
                };
                grad_vec[row_start + v] = (prob - target_q) * inv_batch;
            }
        }

        let avg_loss = total_loss * inv_batch;
        let grad_shape = logits.shape().clone();
        let grad_storage = RocmStorage::copy_from_host(
            &grad_vec,
            &grad_shape,
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        Ok((avg_loss, Box::new(grad_storage) as Box<dyn BackendStorage>))
    }

    /// Fused 3-in-1 SwiGLU activation + dynamic scale quantization HIP kernel launch.
    pub fn silu_mul_quantize_gpu(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        _format: grim_tensor::dtype::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let g_s = as_rocm(gate)?;
        let u_s = as_rocm(up)?;
        if !g_s.device_ptr_is_valid() || !u_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul_quantize: inputs lack a valid device pointer".into(),
            ));
        }

        let total = out_shape.elem_count();
        let qout_storage = RocmStorage::alloc_gpu(
            out_shape,
            DType {
                arith: grim_tensor::dtype::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let scale_storage = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[1]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut gate_ptr = dev_ptr(g_s)?;
        let mut up_ptr = dev_ptr(u_s)?;
        let mut qout_ptr = dev_ptr(&qout_storage)?;
        let mut scale_ptr = dev_ptr(&scale_storage)?;
        let mut n_i = total as i32;

        let grid = HipDim3::new(1, 1, 1);
        let block = HipDim3::new(256, 1, 1);

        self.launch_compute_kernel(
            "grim_silu_mul_quantize",
            grid,
            block,
            &mut [
                arg(&mut gate_ptr),
                arg(&mut up_ptr),
                arg(&mut qout_ptr),
                arg(&mut scale_ptr),
                arg(&mut n_i),
            ],
        )?;

        Ok((
            Box::new(qout_storage),
            Box::new(scale_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Block-Quantized SageAttention HIP kernel launch.
    pub fn sage_attention_gpu(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid() || !k_s.device_ptr_is_valid() || !v_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "sage_attention: inputs lack a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let mut q_ptr = dev_ptr(q_s)?;
        let mut k_ptr = dev_ptr(k_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut out_ptr = dev_ptr(&out_storage)?;

        let mut nh_i = num_heads as i32;
        let mut nkv_i = num_kv_heads as i32;
        let mut hd_i = head_dim as i32;
        let mut sl_i = seq_len as i32;
        let mut ksl_i = kv_seq_len as i32;
        let mut sm_scale = 1.0f32 / (head_dim as f32).sqrt();

        let grid = HipDim3::new(num_heads as u32, ((seq_len + 127) / 128) as u32, 1);
        let block = HipDim3::new(128, 1, 1);

        self.launch_compute_kernel(
            "grim_sage_attention",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut k_ptr),
                arg(&mut v_ptr),
                arg(&mut out_ptr),
                arg(&mut nh_i),
                arg(&mut nkv_i),
                arg(&mut hd_i),
                arg(&mut sl_i),
                arg(&mut ksl_i),
                arg(&mut sm_scale),
            ],
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Tree-attention wrapper for speculative-decoding verification. [see: `1 + gamma`, `tree_parents`]
    pub fn qkv_attention_paged(
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
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_paged expects 3-D output shape [batch, num_heads, head_dim]".into(),
            ));
        }
        let batch = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let q_s = as_rocm(q)?;
        let bt_s = as_rocm(block_tables)?;
        let k_s = as_rocm(k_pages)?;
        let v_s = as_rocm(v_pages)?;

        if !q_s.device_ptr_is_valid()
            || !bt_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "qkv_attention_paged: inputs lack a valid device pointer".into(),
            ));
        }

        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        crate::launch_paged_attention(
            self,
            q_s,
            bt_s,
            k_s,
            v_s,
            &mut storage,
            batch as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            max_blocks as u32,
            page_size as u32,
            kv_seq_len as u32,
            cache_offset,
            0,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    pub fn tree_attention(
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
        // ─── structural validation ───────────────────────────────────────── [see: `qkv_attention`]
        let out_dims = out_shape.dims();
        if out_dims.len() != 4 {
            return Err(Error::Shape(
                "tree_attention requires 4-D output shape \
                 [batch, 1+gamma, num_heads, head_dim]"
                    .into(),
            ));
        }
        let batch = out_dims[0];
        let one_plus_gamma = out_dims[1];
        let num_heads = out_dims[2];
        let head_dim = out_dims[3];

        if batch == 0 || num_heads == 0 || head_dim == 0 {
            return Err(Error::Shape(
                "tree_attention: zero-sized batch / num_heads / head_dim".into(),
            ));
        }
        if one_plus_gamma == 0 {
            return Err(Error::Shape(
                "tree_attention: 1+gamma must be >= 1 (gamma == 0 still has a root)".into(),
            ));
        }
        // tree_parents must have at least 1+gamma entries.
        if tree_parents.shape().elem_count() < one_plus_gamma {
            return Err(Error::Shape(format!(
                "tree_attention: tree_parents must have >= {} entries (got {})",
                one_plus_gamma,
                tree_parents.shape().elem_count(),
            )));
        }
        // Block dim: 128 threads for Wave32 (gfx1036/RDNA2: 4 Wave32 wavefronts),
        // 256 threads for Wave64 (CDNA: 4 Wave64 wavefronts).
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "tree_attention Phase-3 supports head_dim <= 256 (got {})",
                head_dim
            )));
        }
        // GQA head-count sanity (same rule as `qkv_attention`).
        let gamma = one_plus_gamma - 1;
        if num_kv_heads == 0 || num_kv_heads > num_heads {
            return Err(Error::Shape(format!(
                "tree_attention: num_kv_heads ({}) must be within [1, num_heads] ({})",
                num_kv_heads, num_heads
            )));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "tree_attention: num_heads ({}) must be a multiple of num_kv_heads ({})",
                num_heads, num_kv_heads
            )));
        }

        // ─── input pointer validation ─────────────────────────────────────
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let tp_s = as_rocm(tree_parents)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !tp_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "tree_attention: an input lacks a valid device pointer".into(),
            ));
        }

        // ─── allocate output + launch ──────────────────────────────────
        let mut storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let gamma_u32 = gamma as u32;

        // The launcher takes `&dyn BackendStorage` for inputs and [see: `&mut dyn BackendStorage`, `RocmStorage`, `BackendStorage`, `tree_attention`]
        crate::launch_tree_attention(
            self,
            q_s,
            k_s,
            v_s,
            tp_s,
            &mut storage,
            batch as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            gamma_u32,
            kv_seq_len as u32,
            cache_offset,
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    // ─── Phase 2: Selective Scan ──────────────────────────────────

    /// Launch the JIT compiled Mamba selective scan kernel (Wave64,
    pub(crate) fn launch_selective_scan(
        &self,
        x_storage: &RocmStorage,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        c_storage: &RocmStorage,
        d_storage: &RocmStorage,
        state_storage: &RocmStorage,
        out_storage: &RocmStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        _seq_len: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: x has no device ptr".into()))?;
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: b has no device ptr".into()))?;
        let c_ptr = c_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: c has no device ptr".into()))?;
        let d_ptr = d_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: d has no device ptr".into()))?;
        let state_ptr = state_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: state has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (batch as u64)
            .checked_mul(dim_dinner as u64)
            .ok_or_else(|| Error::Backend("selective_scan: batch*dim_dinner overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("selective_scan: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        // Kernel signature: (a_log, b_tensor, c_tensor, d_tensor, dt_tensor,
        //                    h_in_out, x_tensor, y_data, batch_index, d_inner, d_state)
        let mut a_log_ptr = a_ptr;
        let mut b_tensor_ptr = b_ptr;
        let mut c_tensor_ptr = c_ptr;
        let mut d_tensor_ptr = d_ptr;
        let mut dt_tensor_ptr = d_ptr; // dt_bias passed as d_storage
        let mut h_in_out_ptr = state_ptr; // state buffer (read prev, write new)
        let mut x_tensor_ptr = x_ptr;
        let mut y_data_ptr = out_ptr; // output buffer
        let mut batch_index = batch as i32;
        let mut d_inner = dim_dinner as i32;
        let mut d_state = dim_dstate as i32;

        let shared_mem_bytes = dim_dstate * BLOCK_SIZE * std::mem::size_of::<f32>();

        self.launch_compute_kernel_with_solution(
            "grim_selective_scan",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_log_ptr),
                arg(&mut b_tensor_ptr),
                arg(&mut c_tensor_ptr),
                arg(&mut d_tensor_ptr),
                arg(&mut dt_tensor_ptr),
                arg(&mut h_in_out_ptr),
                arg(&mut x_tensor_ptr),
                arg(&mut y_data_ptr),
                arg(&mut batch_index),
                arg(&mut d_inner),
                arg(&mut d_state),
            ],
            None,
            shared_mem_bytes,
        )
    }

    // ─── Phase 2: Cross-Attention ─────────────────────────────────
    // ─── Phase 2: Cross-Attention ─────────────────────────────────

    /// Launch the JIT compiled Whisper cross-attention kernel
    pub(crate) fn launch_cross_attention(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 128;
        // One block per (query position, head) row.
        let total_rows = num_heads
            .checked_mul(seq_len)
            .ok_or_else(|| Error::Backend("cross_attention: rows overflow".into()))?;
        let grid_x: u32 = total_rows
            .try_into()
            .map_err(|_| Error::Backend("cross_attention: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        // Shared memory: scores[seq_len_k] + red_max[block_dim] + red_sum[block_dim].
        let shared_mem_bytes = (kv_seq_len + 2 * BLOCK_SIZE)
            .checked_mul(4)
            .ok_or_else(|| Error::Backend("cross_attention: shared mem overflow".into()))?;

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut sq = seq_len as i32;
        let mut sk = kv_seq_len as i32;
        let mut nh = num_heads as i32;
        let mut nkh = num_heads as i32; // cross-attention uses full GQA sharing
        let mut hd = head_dim as i32;
        let mut scale = 1.0f32 / (head_dim as f32).sqrt();

        self.launch_compute_kernel_with_solution(
            "grim_cross_attention",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut sq),
                arg(&mut sk),
                arg(&mut nh),
                arg(&mut nkh),
                arg(&mut hd),
                arg(&mut scale),
            ],
            None,
            shared_mem_bytes,
        )
    }

    // ─── Phase 2: RWKV Time-Mix ───────────────────────────────────

    /// Launch the JIT compiled RWKV time-mix kernel (recurrent
    pub(crate) fn launch_rwkv_time_mix(
        &self,
        x_storage: &RocmStorage,
        w_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        g_storage: &RocmStorage,
        out_storage: &RocmStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: x has no device ptr".into()))?;
        let w_ptr = w_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: w has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: v has no device ptr".into()))?;
        let g_ptr = g_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: g has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (batch as u64)
            .checked_mul(dim as u64)
            .ok_or_else(|| Error::Backend("rwkv_time_mix: batch*dim overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("rwkv_time_mix: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut xptr = x_ptr;
        let mut wptr = w_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut gptr = g_ptr;
        let mut optr = out_ptr;
        let mut b_val = batch as i32;
        let mut d_val = dim as i32;
        let mut s_val = seq_len as i32;

        self.launch_compute_kernel(
            "grim_rwkv_time_mix",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut wptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut gptr),
                arg(&mut optr),
                arg(&mut b_val),
                arg(&mut d_val),
                arg(&mut s_val),
            ],
        )
    }

    /// Launch the JIT compiled RWKV channel-mix kernel (RWKV-5/6
    pub(crate) fn launch_rwkv_channel_mix(
        &self,
        x_storage: &RocmStorage,
        k_storage: &RocmStorage,
        r_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        batch: usize,
        dim: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: x has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: k has no device ptr".into()))?;
        let r_ptr = r_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: r has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (batch as u64)
            .checked_mul(dim as u64)
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: batch*dim overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("rwkv_channel_mix: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut xptr = x_ptr;
        let mut kptr = k_ptr;
        let mut rptr = r_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut b_val = batch as i32;
        let mut d_val = dim as i32;

        self.launch_compute_kernel(
            "grim_rwkv_channel_mix",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut kptr),
                arg(&mut rptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut b_val),
                arg(&mut d_val),
            ],
        )
    }

    // ─── Phase 2: MFMA FP8 (gfx1200+) ────────────────────────────

    /// Launch the gfx1200 MFMA FP8 fused dequant GEMM kernel. [see: `should_use_wmma_path`, `rocm_device_props::gfx_level >= 12`]
    pub(crate) fn launch_fused_dequant_gemm_fp8_mfma(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: a has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: B_fp8 has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fp8_mfma: m*n overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("fp8_mfma: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_fp8_mfma",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the gfx1200 MFMA FP8 backward kernel.
    pub(crate) fn launch_fused_dequant_backward_gemm_fp8_mfma(
        &self,
        dy_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: dY has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: B_fp8 has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: m*k overflow".into()))?;
        let grid_x: u32 = ((total_elems + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            .try_into()
            .map_err(|_| Error::Backend("fp8_mfma_bwd: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_fp8_mfma",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }
}

/// P1-WI-1: pure routing decision for the WMMA GEMM path. Extracted from [see: `RocmDevice::should_use_wmma_path`]
pub(crate) fn wmma_route_decision(
    ext: Option<&grim_format::spec::GrimTensorExt>,
    out_arith: ArithType,
    cfg_enabled: bool,
) -> bool {
    // No extension ⇒ no per-tensor hint ⇒ stick with the existing dispatcher
    let Some(ext) = ext else {
        return false;
    };
    if !cfg_enabled {
        return false;
    }
    match ext.layout_hint {
        grim_format::spec::LayoutHintTag::PackedQuantWmma { bits, .. } => {
            matches!(bits, 2 | 3 | 4 | 8) && out_arith == ArithType::F16
        }
        _ => false,
    }
}

impl grim_format::convert::GpuDequant for RocmDevice {
    fn dequantize(
        &self,
        storage: &grim_tensor::dtype::Storage,
        bytes: &[u8],
        elem_count: usize,
    ) -> grim_tensor::error::Result<Option<Vec<f32>>> {
        use grim_tensor::dtype::{BlockDtype, FloatPackScheme, KQuantScheme, Storage};
        match storage {
            Storage::KQuant(KQuantScheme::Q80) => {
                Ok(Some(self.dequantize_q8_0_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::Q4K) => {
                Ok(Some(self.dequantize_q4k_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2XXS) => {
                Ok(Some(self.dequantize_iq2xxs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2XS) => {
                Ok(Some(self.dequantize_iq2xs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2S) => {
                Ok(Some(self.dequantize_iq2s_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ3XXS) => {
                Ok(Some(self.dequantize_iq3xxs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ3S) => {
                Ok(Some(self.dequantize_iq3s_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ4NL) => {
                Ok(Some(self.dequantize_iq4nl_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ4XS) => {
                Ok(Some(self.dequantize_iq4xs_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::Fp8) => {
                Ok(Some(self.dequantize_fp8_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::MxFp4) => {
                Ok(Some(self.dequantize_mxfp4_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::MxFp8) => {
                Ok(Some(self.dequantize_mxfp8_host(bytes, elem_count)?))
            }
            Storage::Block(BlockDtype::Fp8 | BlockDtype::Fp8Block16) => {
                Ok(Some(self.dequantize_fp8_host(bytes, elem_count)?))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod wmma_route_tests {
    use super::*;
    use grim_format::spec::{GrimTensorExt, LayoutHintTag};

    fn ext_packed(bits: u8) -> GrimTensorExt {
        GrimTensorExt {
            tensor_name: "test.weight".into(),
            layout_hint: LayoutHintTag::PackedQuantWmma {
                bits,
                frag_m: 16,
                frag_n: 16,
            },
            ..Default::default()
        }
    }

    #[test]
    fn no_extension_routes_to_default() {
        assert!(!wmma_route_decision(None, ArithType::F16, true));
    }

    #[test]
    fn disabled_config_skips_wmma() {
        assert!(!wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F16,
            false,
        ));
    }

    #[test]
    fn packed_4bit_f16_enabled() {
        assert!(wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn packed_2bit_supported_too() {
        assert!(wmma_route_decision(
            Some(&ext_packed(2)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn packed_unsupported_bpw_falls_back() {
        // 6-bit is not in {2,3,4,8} → must not dispatch to WMMA.
        assert!(!wmma_route_decision(
            Some(&ext_packed(6)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn non_f16_output_skips_wmma() {
        // WMMA path only registered for F16; F32 arch falls to rocBLAS.
        assert!(!wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F32,
            true,
        ));
    }

    #[test]
    fn default_hint_skips_wmma() {
        let ext = GrimTensorExt {
            layout_hint: LayoutHintTag::Default,
            ..Default::default()
        };
        assert!(!wmma_route_decision(Some(&ext), ArithType::F16, true));
    }

    #[test]
    fn wavefront_tiled_does_not_route_wmma() {
        // WavefrontTiled goes through a different (existing) tiled path; do
        let ext = GrimTensorExt {
            layout_hint: LayoutHintTag::WavefrontTiled,
            ..Default::default()
        };
        assert!(!wmma_route_decision(Some(&ext), ArithType::F16, true));
    }
}
