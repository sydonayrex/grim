//! `RocmDevice` — the ROCm-side GPU device. Constructed via [see: `RocmDevice::new(ordinal)`, `.hsaco`, `HsacoKernelCache`, `BackendDevice`]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::device::device_quant::wmma_route_decision;
use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendStorage, Shape};
pub use grim_tensor::{
    AttentionOps, AutogradOps, CollectiveOps, CoreTensorOps, ElementwiseOps, FusionOps,
    GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps, RecurrentOps, SamplingOps,
};

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
    HipErrorT,
    HipMemcpyKind,
    // kernel cache + graph capture
    HsacoKernelCache,
    ROCBLAS_GEMM_FLAGS_NONE,
    RocblasHandle,
    RocblasInt,
    RocblasOperation,
    RocmCachingAllocator,
    RocmDeviceProps,
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
    hipGetDeviceCount,
    hipGraphDestroy,
    hipGraphExecDestroy,
    hipGraphInstantiate,
    hipGraphLaunch,
    hipMemGetInfo,
    hipMemcpyAsync,
    hipModuleUnload,
    hipStreamBeginCapture,
    hipStreamCreate,
    hipStreamDestroy,
    hipStreamEndCapture,
    hipStreamSynchronize,
    hipSuccess,
    linear_launch,
    rocblas_create_handle,
    rocblas_destroy_handle,
    rocblas_gemm_strided_batched_ex,
    rocblas_set_stream,
    rocblas_status_success,
    select_gemm_algo,
};

/// Return type for [`RocmDevice::charon_grouped_backward_roundtrip`].
/// SPEED-ROC-9: holds the four Charon MoE backward gradient buffers as
/// DEVICE-RESIDENT storages — the old `Vec<f32>` fields forced a full D2H
/// round-trip of every gradient per step even when the optimizer/communicator
/// downstream works on device memory. Consumers that need host data call
/// [`CharonBackwardResult::to_cpu`].
pub struct CharonBackwardResult {
    pub d_gate_w: Box<dyn BackendStorage>,
    pub d_up_w: Box<dyn BackendStorage>,
    pub d_down_w: Box<dyn BackendStorage>,
    pub d_x: Box<dyn BackendStorage>,
}

impl std::fmt::Debug for CharonBackwardResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharonBackwardResult")
            .field("d_gate_w", &self.d_gate_w.shape().dims())
            .field("d_up_w", &self.d_up_w.shape().dims())
            .field("d_down_w", &self.d_down_w.shape().dims())
            .field("d_x", &self.d_x.shape().dims())
            .finish()
    }
}

/// Host-side mirror of [`CharonBackwardResult`] for consumers that read back.
#[derive(Debug, Clone)]
pub struct CharonBackwardHost {
    pub d_gate_w: Vec<f32>,
    pub d_up_w: Vec<f32>,
    pub d_down_w: Vec<f32>,
    pub d_x: Vec<f32>,
}

impl CharonBackwardResult {
    /// D2H readback of all four gradient buffers.
    pub fn to_cpu(&self) -> Result<CharonBackwardHost> {
        Ok(CharonBackwardHost {
            d_gate_w: self.d_gate_w.to_cpu_vec_f32()?,
            d_up_w: self.d_up_w.to_cpu_vec_f32()?,
            d_down_w: self.d_down_w.to_cpu_vec_f32()?,
            d_x: self.d_x.to_cpu_vec_f32()?,
        })
    }
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
    /// Opt-in gate for the Jay-Tier MXFP4 fused dequant-GEMM kernel (`launch_fused_dequant_gemm_mxfp4`).
    /// Defaults to `false` so the proven tiled MXFP4 path stays the default until parity with.
    pub(crate) mxfp4_fused_dequant_gemm_enabled: AtomicBool,
    /// AtomicBool shadow of `wmma_gemm_config.enabled` — read lock-free by
    /// `should_use_wmma_path`. Written by `set_wmma_gemm_enabled`.
    pub(crate) wmma_gemm_enabled: AtomicBool,
    /// Caching device-memory allocator (size-bucketed free-list). See `RocmCachingAllocator`.
    pub(crate) allocator: Arc<RocmCachingAllocator>,
    /// Pinned host buffers backing in-flight stream-ordered H2D copies.
    /// A `hipMemcpyAsync` reads these pages on the copy engine *after* the CPU returns, so the.
    pub(crate) retained_pins: Mutex<Vec<RocmPinnedBuffer<f32>>>,
    /// Phase-3 §3.1: device scratch pool — a thread-safe, power-of-2-bucketed [see: `hipMalloc`, `get_scratch`]
    pub(crate) scratch_pool: Arc<crate::memory::pool::DeviceScratchPool>,
    /// Loaded HIP modules + resolved entry functions, cached per unique kernel entry. [see: `hipModuleLoad`, `hipModuleGetFunction`]
    pub(crate) autotuner: Mutex<crate::autotune::Autotuner>,
    /// Tuning-mode + occupancy + tuning-solution store for this device.
    /// [salamander.md §3.6: TuningMode, BlockSizeBand, OccupancyTuning, tuning solution storage]
    #[allow(dead_code)]
    pub(crate) tuning: Mutex<crate::autotune::AutotunerConfig>,

    pub(crate) module_cache: Mutex<HashMap<String, (*mut c_void, *mut c_void)>>,
    /// Resolved-function fast path for `launch_compute_kernel_with_solution`: (entry, grid_x, grid_y) -> hipFunction.
    /// Skips the per-launch kernel source regeneration + seahash + CString work for repeat launches (the.
    pub(crate) resolved_kernel_cache: Mutex<HashMap<(String, u32, u32, Option<i32>), *mut c_void>>,
    /// Interner for `&'static str` autotune keys (entry / arch).
    /// Each unique string is leaked EXACTLY ONCE; repeat `get_or_tune_tiles` / `store_tune_cache` calls reuse it instead.
    pub(crate) str_interner: Mutex<std::collections::HashSet<&'static str>>,
    /// Real `hipModuleLoad` call count (cache hits excluded). Item 2 acceptance.
    pub(crate) module_load_count: AtomicUsize,
    /// Total kernel + GEMM launches since the last `reset_launch_count`.
    /// Instrumentation for fusion-boundary launch-count gates (WI-F1 etc.); counts every `hipModuleLaunchKernel` and every rocBLAS GEMM enqueued.
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

// SAFETY: `RocmDevice` wraps HIP device state (context, stream pool, handle caches) that is process-local and accessed only through the owning thread's HIP context.
// Moving the device to another thread (Send) is safe because HIP contexts are thread-local but.
unsafe impl Send for RocmDevice {}
unsafe impl Sync for RocmDevice {}

impl RocmDevice {
    /// Allocate raw device bytes for an engine-owned persistent dispatch ring.
    pub fn alloc_scythe_ring_bytes(&self, bytes: usize) -> Result<RocmStorage> {
        let storage = RocmStorage::alloc_gpu_with_bytes(
            &Shape::from_slice(&[bytes]),
            dtype_f32(),
            bytes,
            &self.allocator,
            self.ordinal,
        )?;
        // WI-SB6: ring slots MUST start zeroed - `status` byte-layout begins with PENDING(0).
        // Uninitialized VRAM let a resident worker claim phantom descriptors with garbage opcodes and wedge the.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        if let Some(ptr) = storage.device_ptr_u64() {
            let rc = unsafe { crate::device::handles::hipMemset(ptr as *mut c_void, 0, bytes) };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "alloc_scythe_ring_bytes: zeroing failed with hip status {rc}"
                )));
            }
            let _ = unsafe { crate::device::handles::hipDeviceSynchronize() };
        }
        Ok(storage)
    }

    /// WI-SB5: cross-device F32 copy via pinned staging with per-leg context pins.
    /// Independent of RCCL/peer-access features - safe default for small fan-in/gather transfers (decode-sized rows).
    pub fn copy_cross_device_bounce(
        &self,
        dst_ordinal: usize,
        dst_ptr: *mut c_void,
        src_ordinal: usize,
        src_ptr: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut staging = RocmPinnedBuffer::<u8>::alloc(bytes)?;
        // Leg 1: src device -> pinned host.
        {
            let _leg = crate::device::util::DeviceGuard::set(src_ordinal as i32);
            check_hip("cross-bounce D2H", unsafe {
                hipMemcpyAsync(
                    staging.as_mut_ptr() as *mut c_void,
                    src_ptr,
                    bytes,
                    HipMemcpyKind::DeviceToHost,
                    self.active_stream(),
                )
            })?;
            check_hip("cross-bounce D2H sync", unsafe {
                hipStreamSynchronize(self.active_stream())
            })?;
        }
        // Leg 2: pinned host -> dst device.
        {
            let _leg = crate::device::util::DeviceGuard::set(dst_ordinal as i32);
            check_hip("cross-bounce H2D", unsafe {
                hipMemcpyAsync(
                    dst_ptr,
                    staging.as_ptr() as *const c_void,
                    bytes,
                    HipMemcpyKind::HostToDevice,
                    self.active_stream(),
                )
            })?;
            check_hip("cross-bounce H2D sync", unsafe {
                hipStreamSynchronize(self.active_stream())
            })?;
        }
        Ok(())
    }

    /// Enqueue one descriptor upload on the device's active stream.
    pub fn copy_scythe_descriptor_async(
        &self,
        dst: u64,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> Result<()> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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

    /// WI-SB6: create a NON-BLOCKING stream. Unlike pool streams (created blocking-with-legacy), a non-blocking stream never serializes with other streams - required
    /// for the resident persistent wave so host control traffic (head publishes, tail polls, stop) is never queued behind an eternally-running kernel.
    pub fn create_non_blocking_stream(&self) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut stream: *mut c_void = std::ptr::null_mut();
        const HIP_STREAM_NON_BLOCKING: u32 = 0x1;
        check_hip("hipStreamCreateWithFlags(NonBlocking)", unsafe {
            crate::device::handles::hipStreamCreateWithFlags(
                &mut stream,
                HIP_STREAM_NON_BLOCKING,
                0,
            )
        })?;
        Ok(stream)
    }

    /// Destroy a stream previously returned by [`Self::create_non_blocking_stream`].
    pub fn destroy_stream(&self, stream: *mut c_void) {
        if !stream.is_null() {
            let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
            unsafe {
                let _ = crate::device::handles::hipStreamDestroy(stream);
            }
        }
    }

    /// [`Self::copy_scythe_descriptor_async`] on an explicit stream - the resident-wave control path must never enqueue behind the worker.
    /// WI-SB6 (2026-08-24): async enqueue + CONTROL-STREAM-ONLY sync.
    pub fn copy_scythe_descriptor_async_on(
        &self,
        dst: u64,
        src: *const std::ffi::c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        check_hip("hipMemcpyAsync(ScytheRing H2D, ctrl)", unsafe {
            hipMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src,
                bytes,
                HipMemcpyKind::HostToDevice,
                stream,
            )
        })?;
        check_hip("hipStreamSynchronize(ScytheRing H2D)", unsafe {
            crate::device::handles::hipStreamSynchronize(stream)
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

        // WI-M1/M2 context discipline: construction must run on the target device's context (streams and the rocBLAS handle bind to whatever device is current), but construction must be context-NEUTRAL for the caller.
        // This path used to park the constructing thread on `ordinal` permanently - first use of.
        let mut prev_dev: i32 = 0;
        unsafe {
            let _ = crate::device::handles::hipGetDevice(&mut prev_dev);
        }
        let set_status = crate::device::util::raw_set_device(ordinal as i32);
        if set_status != hipSuccess {
            return Err(Error::Backend(format!(
                "hipSetDevice({ordinal}) failed with code {set_status} \
                 (is the ordinal out of range?)"
            )));
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
        // Auto-init RCCL when multi-process TP is active - this rank process builds its own
        // RcclAllReduce over the full ordinal list so `RowParallelLinear::forward`'s all_reduce has a live comm handle.
        dev.auto_init_rccl();
        // Construction is context-neutral: hand the calling thread its
        // previous device back instead of parking it on `ordinal`.
        let _restore = crate::device::util::raw_set_device(prev_dev);
        Ok(dev)
    }
    fn fallback(ordinal: usize) -> Self {
        Self::build(ordinal, 32, 0, None, Vec::new())
    }

    /// Attach (or detach) an RCCL multi-GPU collective handle.
    /// Called by the training orchestrator after constructing `RcclAllReduce` so that [`BackendDevice::all_reduce`] and [`BackendDevice::comm_fuse_reduce`] can dispatch.
    pub fn set_rccl_handle(&self, handle: Option<Arc<crate::rccl::RcclAllReduce>>) {
        *self.rccl.lock().unwrap_or_else(|e| e.into_inner()) = handle;
    }

    /// Borrow the live RCCL handle (if any) for diagnostic / external use.
    pub fn rccl_handle(&self) -> Option<Arc<crate::rccl::RcclAllReduce>> {
        self.rccl.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Auto-init the RCCL handle from `GRIM_TP_*` env vars when multi-process TP is active.
    /// Each rank process calls this after construction; the handle covers the full ordinal list so.
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

    /// P2P memcpy that routes via direct peer DMA or host-bounce staging, bridging the typed routing decision (`P2PStatus` → `RouteLink`) to the actual memcpy primitives.
    /// This is the bridge that `p2p_route.rs` defers: it calls `peer_access::peer_status` to classify the link, `to_route_link`.
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
    /// Constructing a `RocmDevice` is *not* cheap: it creates a rocBLAS handle (which itself hipMallocs a.
    fn device_cache() -> &'static Mutex<HashMap<usize, Arc<RocmDevice>>> {
        static CACHE: std::sync::OnceLock<Mutex<HashMap<usize, Arc<RocmDevice>>>> =
            std::sync::OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Return the process-wide shared device for `ordinal`, constructing it on first use.
    /// Prefer this over `RocmDevice::new` anywhere a device is obtained repeatedly (per tensor, per token, per.
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
            // Pool floor raised to 512 MB (a 512 MB cap on a 16 GB card forced real hipMalloc/hipFree churn for every transient once the cap was hit - i.e.
            // always); ceiling 4 GB keeps runaway env overrides bounded.
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
            // [salamander.md §3.6: TuningMode, BlockSizeBand, OccupancyTuning, tuning solution storage]
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
                        // Explicit operator override: "1"/"true" forces the fused path on, "0"/"false" forces it off.
                        // Any other value falls back to the arch-confirmed default.
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
        let mut cfg = self
            .decode_gemm_config
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// Set whether fused dequantization GEMM is enabled (WI-C).
    pub fn set_fused_dequant_gemm_enabled(&self, enabled: bool) {
        self.fused_dequant_gemm_enabled
            .store(enabled, Ordering::Relaxed);
        let mut cfg = self
            .fused_dequant_gemm_config
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// Set whether the Jay-Tier MXFP4 fused dequant-GEMM kernel is enabled.
    /// Defaults to `false` (tiled fallback) until parity with the F32 oracle is confirmed on a.
    pub fn set_mxfp4_fused_dequant_gemm_enabled(&self, enabled: bool) {
        self.mxfp4_fused_dequant_gemm_enabled
            .store(enabled, Ordering::Relaxed);
    }

    /// Set whether SplitK GEMM is enabled (WI-D).
    pub fn set_split_k_enabled(&self, enabled: bool) {
        let mut cfg = self
            .split_k_config
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cfg.enabled = enabled;
    }

    /// Set whether the JIT compiled WMMA GEMM kernel is enabled (WI-G). [see: `grim_wmma_gemm`]
    pub fn set_wmma_gemm_enabled(&self, enabled: bool) {
        self.wmma_gemm_enabled.store(enabled, Ordering::Relaxed);
        let mut cfg = self
            .wmma_gemm_config
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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

    /// `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` for a kernel entry that has already been launched (and thus resolved) on this device.
    /// Returns `None` if the entry is not in the resolved-kernel cache yet.
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
        // WI-M1 context discipline: the pooled buffer lives on THIS device; pin the context
        // or a drifted thread's synchronous H2D copy lands the data on another device's memory.
        let _ctx = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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

    /// SPEED-ROC-4: whether HIP graph capture is enabled for this device
    /// (`GRIM_CAPTURE_GRAPH`). Read by hot-path dispatchers to opt into
    /// capture/replay without poking private fields across modules.
    pub(crate) fn graph_capture_enabled(&self) -> bool {
        self.capture_enabled
    }

    /// If a graph-capture session is active, returns the dedicated capture stream. [see: `None`]
    pub(crate) fn active_capture_stream(&self) -> Option<*mut c_void> {
        if self.capture_active.load(Ordering::SeqCst) {
            *self
                .capture_stream
                .read()
                .unwrap_or_else(|e| e.into_inner())
        } else {
            None
        }
    }

    /// The stream an op should dispatch onto: the capture stream when a session is
    pub(crate) fn active_stream(&self) -> *mut c_void {
        let stream = if self.capture_active.load(Ordering::SeqCst) {
            self.capture_stream
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(|| self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut()))
        } else {
            self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut())
        };
        // SPEED-ROC-1: if a stream-ordered upload is in flight on the transfer stream, fence this (compute) stream on its completion event so the prefetch can overlap the prior decode-step GEMM instead of racing it.
        // `hipStreamWaitEvent` is a no-op ordering edge; it does not block the host.
        if !stream.is_null() {
            if let Ok(guard) = self.upload_event.lock() {
                if let Some(ev) = *guard {
                    if !ev.is_null() {
                        unsafe {
                            let _ = crate::hipStreamWaitEvent(stream, ev, 0);
                        }
                    }
                }
            }
        }
        stream
    }

    /// Block until all previously issued work on all streams of this device
    pub fn synchronize(&self) {
        // Pin the correct device before synchronizing - hipDeviceSynchronize() synchronizes the calling thread's current device, not necessarily self.ordinal.
        // [P1-7 fix: DeviceGuard before sync.]
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let _ = unsafe { hipDeviceSynchronize() };
        // All stream-ordered H2D copies are complete, so the pinned host sources they read are now safe to release.
        // Draining here (rather than at each upload) is what allows consecutive uploads to queue on.
        if let Ok(mut pins) = self.retained_pins.lock() {
            pins.clear();
        }
    }

    /// Begin a generic graph-capture session keyed by `key`. Until `end_graph_capture` [see: `key`, `GRIM_CAPTURE_GRAPH`]
    pub fn begin_graph_capture(&self, _key: &str) -> Result<()> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        if !self.capture_enabled {
            return Ok(());
        }
        if self.capture_active.load(Ordering::SeqCst) {
            return Err(Error::Backend(
                "begin_graph_capture: a capture session is already active".into(),
            ));
        }
        // Lazily create the capture stream; it lives for the device lifetime so rocblas
        let mut cs = self
            .capture_stream
            .write()
            .unwrap_or_else(|e| e.into_inner());
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
            .unwrap_or_else(|e| e.into_inner())
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
        let mut cache = self
            .captured_graphs
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(old) = cache.insert(key.to_string(), CapturedGraph { graph, exec }) {
            unsafe {
                let _ = hipGraphExecDestroy(old.exec);
                let _ = hipGraphDestroy(old.graph);
            }
        }
        // Do NOT reset the rocBLAS handle to the null stream here.
        // Every eager GEMM dispatch re-binds the handle to `active_stream()` before use (P0-17 fix), so leaving.
        self.capture_active.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Replay the graph previously captured under `key`. Returns `Ok(false)` when no [see: `key`, `Ok(true)`]
    pub fn replay_graph(&self, key: &str) -> Result<bool> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        if !self.capture_enabled {
            return Ok(false);
        }
        // Replay on the same capture stream the graph was recorded on, so the rocblas
        let stream = {
            let cs = self
                .capture_stream
                .read()
                .unwrap_or_else(|e| e.into_inner());
            cs.unwrap_or_else(|| self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut()))
        };
        let cache = self
            .captured_graphs
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
                // No post-replay sync: replay is async on `stream`; callers that need the result sync (or read back) at their boundary.
                // The rocblas handle binding is an enqueue-time setting, so we leave it bound to `stream`.
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// True if a graph is cached under `key` (useful for callers deciding whether to
    pub fn has_captured_graph(&self, key: &str) -> bool {
        self.captured_graphs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(key)
    }

    // WRECK-9: decode-step graph capture via GraphCaptureManager.

    /// Lazily-initialized GraphCaptureManager for decode-step graph capture.
    fn ensure_graph_capture_mgr(&self) {
        let mut mgr = self
            .graph_capture_mgr
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if mgr.is_none() {
            *mgr = Some(crate::graph_capture::GraphCaptureManager::for_device(self));
        }
    }

    /// Capture the decode-step GEMM (`launch_decode_gemm_f16`) under a shape key via the GraphCaptureManager, then replay it.
    /// Collapses per-step launch+dispatch overhead for repeated decode steps at the same shape.
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
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        self.ensure_graph_capture_mgr();
        let mgr = self
            .graph_capture_mgr
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mgr = mgr.as_ref().ok_or_else(|| {
            Error::Backend(
                "decode_graph_capture_and_replay: graph capture manager not initialized".into(),
            )
        })?;
        // SPEED-ROC-4: bind the cached graph to the exact buffers captured —
        // see the pointer fields on `DecodeGraphKey`.
        let key = crate::graph_capture::DecodeGraphKey {
            a_ptr: a.device_ptr.unwrap_or(0) as usize,
            b_ptr: b.device_ptr.unwrap_or(0) as usize,
            out_ptr: out.device_ptr.unwrap_or(0) as usize,
            ..key
        };
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

        // P1-3: rocBLAS batched GEMM runs on the calling thread's current device - pin to the
        // owning ordinal or a drifted thread launches against foreign pointers (same class as the matmul_op fix).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

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
        if out_shape.dims() != [m, n] {
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
            if ai.shape().dims() != [m, k] || bi.shape().dims() != [k, n] {
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

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_d = m * n;

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
                    (a_packed.device_ptr_checked()? as *mut c_void).add(i * stride_a * a_elem_size),
                    ai.device_ptr_checked()? as *mut c_void,
                    ai.bytes,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            })?;
            check_hip("matmul_batched: hipMemcpyDtoD b", unsafe {
                hipMemcpyAsync(
                    (b_packed.device_ptr_checked()? as *mut c_void).add(i * stride_b * b_elem_size),
                    bi.device_ptr_checked()? as *mut c_void,
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
                b_packed.device_ptr_checked()? as *const c_void,
                b_type,
                n as RocblasInt,
                (stride_b) as i64,
                a_packed.device_ptr_checked()? as *const c_void,
                a_type,
                k as RocblasInt,
                (stride_a) as i64,
                &beta as *const f32 as *const c_void,
                d_packed.device_ptr_checked()? as *const c_void,
                out_type,
                n as RocblasInt,
                (stride_d) as i64,
                d_packed.device_ptr_checked()? as *mut c_void,
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
            // Every eager GEMM dispatch re-binds the handle to `active_stream()` before its call (P0-17 fix), so.
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
                    batch_storage.device_ptr_checked()? as *mut c_void,
                    (d_packed.device_ptr_checked()? as *mut c_void)
                        .add(i * stride_d * d_element_size),
                    stride_d * d_element_size,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            })?;
            out.push(Box::new(batch_storage) as Box<dyn BackendStorage>);
        }
        // No trailing sync: the unpacked D2D copies share the GEMM's stream, so callers that read the outputs observe completed data through stream order (or their own sync).
        // The previous per-call drain serialized every batched QKV projection.
        Ok(out)
    }

    /// Pinned-memory + async host→device upload for the per-token decode hot path. [see: `data`, `hipMemcpy`, `RocmDevice::from_cpu`, `Vec`]
    pub fn copy_from_host_async(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        // WI-M1 context discipline: an async H2D enqueues on a stream bound to the calling thread's current
        // context; pin the owning ordinal so a drifted caller cannot schedule the copy against foreign memory.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let pinned = RocmPinnedBuffer::<f32>::from_slice(data)?;
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }
        let dev_ptr_void = storage.device_ptr_checked()? as *mut c_void;
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
        // Retain the pin until the next device-wide synchronize - never free a page-locked source while a stream-ordered copy may still read it.
        // This matches the correct pattern in `upload_from_host_stream_ordered`.
        if let Ok(mut pins) = self.retained_pins.lock() {
            pins.push(pinned);
        }
        Ok(Box::new(storage))
    }

    /// Stream-ordered f32 H2D upload that does NOT synchronize before returning.
    /// Pins the host data, allocates device storage, and issues `hipMemcpyAsync` on a dedicated **transfer stream**.
    pub fn upload_from_host_stream_ordered(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        // WI-M1 context discipline: the async copy, the event CREATE and the event RECORD all bind to the calling thread's current context.
        // The cached `upload_event` is reused across uploads - if the first use happened under a.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let pinned = RocmPinnedBuffer::<f32>::from_slice(data)?;
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }
        let dev_ptr_void = storage.device_ptr_checked()? as *mut c_void;
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
        // Record a completion event on the transfer stream so the next compute-dispatch on the active stream (via `active_stream()`) can wait on it.
        // Reuse a single event across uploads.
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
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }
        let dev_ptr_void = storage.device_ptr_checked()? as *mut c_void;
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

    /// In-memory D2D transpose of a contiguous `[a, b]` f32 tensor into a fresh `[b, a]` device buffer via `grim_transpose_2d_f32`.
    /// This replaces the DtoH + transpose + H2D round trip that the host fallback performs.
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
        // SPEED-ROC-2: no trailing sync — src/dst are both allocator-owned
        // device buffers ordered on the same stream as every consumer.
        let _ = stream;
        Ok(Box::new(storage))
    }

    /// Upload f32 data into HIP managed memory.
    /// Managed allocations remain valid to ordinary ROCm kernels while HIP may migrate cold pages to.
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
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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
        // Drain any in-flight kernels on the pooled streams before recycling or freeing.
        // Pin the device first (P1-7 discipline): hipDeviceSynchronize targets the calling thread's current device, which may.
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
        if let Some(stream) = self
            .capture_stream
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
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

        // Pin the calling thread's device before creating the rocBLAS handle - rocBLAS inherits whatever device is current, and a handle created on the wrong device produces silent wrong-answer GEMMs.
        // [P1-3 fix: DeviceGuard::set before rocblas_create_handle.]
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        unsafe {
            let mut h: RocblasHandle = RocblasHandle(std::ptr::null_mut());
            let mut status = rocblas_create_handle(&mut h);
            if status != rocblas_status_success {
                // rocBLAS hipMallocs a 32-128 MiB internal workspace when the handle is created; on small-VRAM parts or under high allocator pressure, that fails with status 5 (rocblas_status_memory_error).
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

            // Fallback: If rocBLAS workspace creation fails due to VRAM memory pressure, return a zeroed handle - our custom HIP fused GEMM kernels handle matmuls without requiring rocBLAS internal workspace allocations.
            // IMPORTANT: callers MUST null-check the handle before use.
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

    /// Device-side element-wise sum of multiple storages via the appropriate `grim_all_reduce_accum*` kernel.
    /// Supports F32, F16, and BF16. Each input must have the same shape.
    pub(crate) fn device_accumulate(
        &self,
        inputs: &[&dyn BackendStorage],
        out_ptr: u64,
        dtype: &DType,
    ) -> Result<()> {
        let total = inputs[0].shape().elem_count();
        let kernel_name = match dtype.arith {
            ArithType::F32 => "grim_all_reduce_accum",
            ArithType::F16 => "grim_all_reduce_accum_f16",
            ArithType::BF16 => "grim_all_reduce_accum_bf16",
            other => {
                return Err(Error::Backend(format!(
                    "device_accumulate: unsupported dtype arith {other:?}"
                )));
            }
        };

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
            kernel_name,
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

    /// Device-side element-wise sum of multiple F32 storages via the `grim_all_reduce_accum` kernel.
    /// Each input must have the same shape.
    pub(crate) fn device_accumulate_f32(
        &self,
        inputs: &[&dyn BackendStorage],
        out_ptr: u64,
    ) -> Result<()> {
        self.device_accumulate(inputs, out_ptr, &dtype_f32())
    }
}

impl grim_tensor::BackendDevice for RocmDevice {}

// to `device::gemm_tuning` — see that module.
pub use crate::device::gemm_tuning::{
    GemmTileConfig, lookup_gemm_config, lookup_gemm_config_for_shape, lookup_solution_index,
};

// Re-exports that pulled up `pub use crate::graph_capture::*` etc. in [see: `pub use`]
