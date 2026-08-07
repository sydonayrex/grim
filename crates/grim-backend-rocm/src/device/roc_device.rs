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
    RocblasInt,
    RocblasOperation,
    RoclabsHandle,
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
    hipGetDeviceCount,
    hipGraphDestroy,
    hipGraphExecDestroy,
    hipGraphInstantiate,
    hipGraphLaunch,
    hipMemAdvise,
    hipMemGetInfo,
    hipMemcpy,
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
};

#[derive(Debug)]
pub struct RocmDevice {
    pub(crate) ordinal: usize,
    pub(crate) props: RocmDeviceProps,
    handle_cache: Mutex<Option<RoclabsHandle>>,
    pub(crate) stream_pool: Mutex<Vec<*mut c_void>>,
    pub(crate) hsaco_cache: HsacoKernelCache,
    /// WI 2.4.4-2 — opt-in switch for the JIT `grim_decode_gemm_f16` [see: `false`, `fusion::DecodeGemmConfig`, `Mutex`, `handle_cache`]
    pub(crate) decode_gemm_config: Mutex<DecodeGemmConfig>,
    pub(crate) fused_dequant_gemm_config: Mutex<FusedDequantGemmConfig>,
    pub(crate) split_k_config: Mutex<SplitKGemmConfig>,
    pub(crate) wmma_gemm_config: Mutex<WmmaGemmConfig>,
    /// Caching device-memory allocator (size-bucketed free-list). See `RocmCachingAllocator`.
    pub(crate) allocator: Arc<RocmCachingAllocator>,
    /// Phase-3 §3.1: device scratch pool — a thread-safe, power-of-2-bucketed [see: `hipMalloc`, `get_scratch`]
    pub(crate) scratch_pool: Arc<crate::memory::pool::DeviceScratchPool>,
    /// Loaded HIP modules + resolved entry functions, cached per unique kernel entry. [see: `hipModuleLoad`, `hipModuleGetFunction`]
    pub(crate) module_cache: Mutex<HashMap<String, (*mut c_void, *mut c_void)>>,
    /// Real `hipModuleLoad` call count (cache hits excluded). Item 2 acceptance.
    pub(crate) module_load_count: AtomicUsize,
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
    /// Once-flag: the first `matmul_batched` call in a process warms up the [see: `gemm_strided_batched_ex`]
    batched_gemm_warmed: AtomicBool,
    /// Optional RCCL collective handle for cross-GPU all-reduce (WI-R1/WI-R3).
    /// Set externally when multi-GPU RCCL training is active; `None` when
    /// single-GPU or RCCL not initialised. [see: `RcclAllReduce`, `set_rccl_handle`]
    pub rccl: Mutex<Option<Arc<crate::rccl::RcclAllReduce>>>,
}

unsafe impl Send for RocmDevice {}
unsafe impl Sync for RocmDevice {}

impl RocmDevice {
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
            let mut h: RoclabsHandle = RoclabsHandle(std::ptr::null_mut());
            let status = rocblas_create_handle(&mut h);
            if status == rocblas_status_success {
                handle_cache = Some(h);
            }
        }

        // Query device attributes for Wavefront size correctness gate.
        let mut warp_size = 64; // Default to W64 (MI200/MI300 CDNA) safety fallback
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
        Self::build(ordinal, 64, 0, None, Vec::new())
    }

    /// Attach (or detach) an RCCL multi-GPU collective handle. Called by the
    /// training orchestrator after constructing `RcclAllReduce` so that
    /// [`BackendDevice::all_reduce`] and [`BackendDevice::comm_fuse_reduce`]
    /// can dispatch device-side collectives instead of falling back to the
    /// CPU fan-in path. [see: `RcclAllReduce::try_new`]
    pub fn set_rccl_handle(&self, handle: Option<Arc<crate::rccl::RcclAllReduce>>) {
        *self.rccl.lock().unwrap() = handle;
    }

    /// Borrow the live RCCL handle (if any) for diagnostic / external use.
    pub fn rccl_handle(&self) -> Option<Arc<crate::rccl::RcclAllReduce>> {
        self.rccl.lock().unwrap().clone()
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

    /// Probe the total amount of device memory reported by the driver, in bytes. [see: `hipMemGetInfo`, `hipDeviceProp_t`]
    fn query_device_vram_bytes(_ordinal: usize) -> usize {
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
        handle_cache: Option<RoclabsHandle>,
        streams: Vec<*mut c_void>,
    ) -> Self {
        let wavefront_size = if warp_size == 32 {
            WavefrontSize::W32
        } else {
            WavefrontSize::W64
        };
        let xnack_enabled = xnack_val == 1;

        // Phase-aware cache cap: when the env override is absent, derive a
        let cap_bytes: usize = std::env::var("GRIM_ALLOC_POOL_CAP_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                let total_vram = Self::query_device_vram_bytes(ordinal);
                let derived = total_vram / 6; // ≈ 16.7 % of VRAM
                derived
            })
            .clamp(128 * 1024 * 1024, 512 * 1024 * 1024);

        let gpu_target = detect_gpu_arch(ordinal as i32);
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
            scratch_pool: crate::memory::pool::DeviceScratchPool::new(),
            module_cache: Mutex::new(HashMap::new()),
            module_load_count: AtomicUsize::new(0),
            gpu_target: gpu_target.clone(),
            capture_enabled: std::env::var("GRIM_CAPTURE_GRAPH").is_ok(),
            capture_stream: RwLock::new(None),
            capture_active: AtomicBool::new(false),
            captured_graphs: Mutex::new(HashMap::new()),
            batched_gemm_warmed: AtomicBool::new(false),
            decode_gemm_config: Mutex::new(DecodeGemmConfig {
                enabled: true,
                wavefront_size: warp_size as u32,
            }),
            fused_dequant_gemm_config: Mutex::new(FusedDequantGemmConfig {
                enabled: true,
                wavefront_size: warp_size as u32,
            }),
            split_k_config: Mutex::new(SplitKGemmConfig { enabled: true }),
            wmma_gemm_config: Mutex::new(WmmaGemmConfig {
                enabled: matches!(
                    crate::quantization::gcn_arch(&gpu_target),
                    crate::quantization::GcnArch::RDNA3 | crate::quantization::GcnArch::RDNA4
                ),
                wavefront_size: warp_size as u32,
            }),
            rccl: Mutex::new(None),
        }
    }

    /// Release all pooled device buffers back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        self.allocator.empty_cache();
    }

    /// P1-WI-1 dispatch probe: should this GEMM route through the WMMA path [see: `GrimTensorExt`, `true`, `wmma_gemm_config`, `layout_hint`]
    pub fn should_use_wmma_path(
        &self,
        ext: Option<&grim_format::spec::GrimTensorExt>,
        out_arith: ArithType,
    ) -> bool {
        let cfg_enabled = self.wmma_gemm_config.lock().unwrap().enabled;
        wmma_route_decision(ext, out_arith, cfg_enabled)
    }

    /// WI 2.4.4-2 — opt-in flag for the JIT `grim_decode_gemm_f16`. [see: `true`, `QkvAttentionFusionConfig::enabled`]
    pub fn set_decode_gemm_enabled(&self, enabled: bool) {
        let mut cfg = self.decode_gemm_config.lock().unwrap();
        cfg.enabled = enabled;
    }

    /// Set whether fused dequantization GEMM is enabled (WI-C).
    pub fn set_fused_dequant_gemm_enabled(&self, enabled: bool) {
        let mut cfg = self.fused_dequant_gemm_config.lock().unwrap();
        cfg.enabled = enabled;
    }

    /// Set whether SplitK GEMM is enabled (WI-D).
    pub fn set_split_k_enabled(&self, enabled: bool) {
        let mut cfg = self.split_k_config.lock().unwrap();
        cfg.enabled = enabled;
    }

    /// Set whether the JIT compiled WMMA GEMM kernel is enabled (WI-G). [see: `grim_wmma_gemm`]
    pub fn set_wmma_gemm_enabled(&self, enabled: bool) {
        let mut cfg = self.wmma_gemm_config.lock().unwrap();
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
            *self.capture_stream.read().unwrap()
        } else {
            None
        }
    }

    /// The stream an op should dispatch onto: the capture stream when a session is
    fn active_stream(&self) -> *mut c_void {
        if self.capture_active.load(Ordering::SeqCst) {
            return self
                .capture_stream
                .read()
                .unwrap()
                .unwrap_or_else(|| self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut()));
        }
        self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut())
    }

    /// Block until all previously issued work on all streams of this device
    pub fn synchronize(&self) {
        let _ = unsafe { hipDeviceSynchronize() };
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
        let mut cs = self.capture_stream.write().unwrap();
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
        let mut cache = self.captured_graphs.lock().unwrap();
        if let Some(old) = cache.insert(key.to_string(), CapturedGraph { graph, exec }) {
            unsafe {
                let _ = hipGraphExecDestroy(old.exec);
                let _ = hipGraphDestroy(old.graph);
            }
        }
        // Restore the rocBLAS handle to its default stream now that capture is done.
        if let Ok(h) = self.get_rocblas_handle() {
            unsafe {
                let _ = rocblas_set_stream(h, std::ptr::null_mut());
            }
        }
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
            let cs = self.capture_stream.read().unwrap();
            cs.unwrap_or_else(|| self.get_stream_from_pool(0).unwrap_or(std::ptr::null_mut()))
        };
        let cache = self.captured_graphs.lock().unwrap();
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
                unsafe {
                    let _ = hipStreamSynchronize(stream);
                }
                // Restore the default stream so later eager ops don't land on the capture stream.
                if let Ok(h) = self.get_rocblas_handle() {
                    unsafe {
                        let _ = rocblas_set_stream(h, std::ptr::null_mut());
                    }
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// True if a graph is cached under `key` (useful for callers deciding whether to
    pub fn has_captured_graph(&self, key: &str) -> bool {
        self.captured_graphs.lock().unwrap().contains_key(key)
    }

    /// Collapse a batch of `batch_count` same-shape GEMMs into one [see: `rocblas_gemm_strided_batched_ex`, `a[i]`, `[m, k]`, `b[i]`]
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
                hipMemcpy(
                    (a_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_a * a_elem_size),
                    ai.device_ptr.unwrap() as *mut c_void,
                    ai.bytes,
                    HipMemcpyKind::DeviceToDevice,
                )
            })?;
            check_hip("matmul_batched: hipMemcpyDtoD b", unsafe {
                hipMemcpy(
                    (b_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_b * b_elem_size),
                    bi.device_ptr.unwrap() as *mut c_void,
                    bi.bytes,
                    HipMemcpyKind::DeviceToDevice,
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
            // Restore the handle to the default (null) stream so other eager GEMMs
            let _ = rocblas_set_stream(handle, std::ptr::null_mut());
            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_strided_batched_ex failed with status {status}"
                )));
            }
        }

        // Read the packed results back, then split into per-batch device storages. [see: `active_stream`]
        if self.active_capture_stream().is_none() {
            unsafe {
                let _ = hipDeviceSynchronize();
            }
        }
        let d_host = d_packed.to_cpu_vec_f32()?;
        let mut out = Vec::with_capacity(batch);
        for i in 0..batch {
            let slice = &d_host[i * stride_d..(i + 1) * stride_d];
            out.push(self.from_cpu(slice, out_shape, dtype_out.clone())?);
        }
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
            if storage.device_ptr.is_some() {
                unsafe {
                    let _ = hipFree(storage.device_ptr.unwrap() as *mut c_void);
                }
            }
            return Err(Error::Backend(format!(
                "hipMemcpyAsync(H2D) failed with error code {}",
                res
            )));
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
        if let Ok(mut cache) = self.handle_cache.lock() {
            if let Some(handle) = cache.take() {
                unsafe {
                    let _ = rocblas_destroy_handle(handle);
                }
            }
        }
        // Destroy the capture stream (owned for the device lifetime). By now the
        if let Some(stream) = self.capture_stream.write().unwrap().take() {
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
        let pool = self.stream_pool.lock().unwrap();
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

    pub fn get_rocblas_handle(&self) -> Result<RoclabsHandle> {
        let mut cache = self
            .handle_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(h) = *cache {
            return Ok(h);
        }

        unsafe {
            let mut h: RoclabsHandle = RoclabsHandle(std::ptr::null_mut());
            let status = rocblas_create_handle(&mut h);
            if status == rocblas_status_success {
                *cache = Some(h);
                return Ok(h);
            } else {
                return Err(Error::Backend(format!(
                    "rocblas_create_handle failed with status {}",
                    status
                )));
            }
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
        let res = if let Some(capture_stream) = self.active_capture_stream() {
            unsafe { hipMemsetAsync(dev_ptr_void, 0, storage.bytes, capture_stream) }
        } else {
            // hipMemset on the default stream is synchronous (host blocks
            unsafe { hipMemset(dev_ptr_void, 0, storage.bytes) }
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

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
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
        let tile_config = lookup_gemm_config(m, n, k, self.props.wavefront_size);
        // Offline-tuned solution_index per (M,N,K) for FP32. Falls back to 0 for [see: `examples/tune_gemm.rs`]
        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, dtype_out.arith);
        // WI 2.4.3 — split_k clamp gate.
        let split_k_effective: u32 = {
            let split_k_enabled = self.split_k_config.lock().unwrap().enabled;
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
                    0,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            };

            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_strided_batched_ex failed with status {status}"
                )));
            }

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
            let cfg = self.decode_gemm_config.lock().unwrap();
            if cfg.enabled && dtype_out.arith == ArithType::F16 && m <= 8 {
                drop(cfg); // release the lock before the JIT launch
                // WI 2.4.4-2(a) — thread the *real* enqueued stream into the [see: `launch_compute_kernel`, `hipModuleLaunchKernel`]
                let stream =
                    self.launch_decode_gemm_f16(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        }

        // ─── WI-G — WMMA GEMM dispatch (opt-in, F16-only) ─────
        {
            let cfg = self.wmma_gemm_config.lock().unwrap();
            if cfg.enabled && dtype_out.arith == ArithType::F16 {
                drop(cfg); // release lock before JIT launch
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        }

        // Get rocBLAS handle and execute sgemm. The handle's stream was already bound [see: `begin_graph_capture`, `end_graph_capture`]
        let handle = self.get_rocblas_handle()?;

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;

        let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
        let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
        let out_ptr_void = out_storage.device_ptr.unwrap() as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is

        let use_gemm_ex = cfg!(feature = "rocm-aiter") || {
            let gcn = std::env::var("GRIM_GPU_TARGET").unwrap_or_else(|_| "gfx900".into());
            gcn == "gfx90a" || gcn == "gfx942"
        };

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
                    "rocblas matmul execution failed with error status {}",
                    status
                )));
            }
        };

        let compute_handle = Box::new(RocmHandle::new(Some(self.active_stream())));
        Ok((Box::new(out_storage), compute_handle))
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

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;

        let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
        let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
        let out_ptr_void = out_storage.device_ptr.unwrap() as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is

        let use_gemm_ex = cfg!(feature = "rocm-aiter") || {
            let gcn = std::env::var("GRIM_GPU_TARGET").unwrap_or_else(|_| "gfx900".into());
            gcn == "gfx90a" || gcn == "gfx942"
        };

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
        let row_len = *x_dims.last().unwrap();
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        let (grid, block) = linear_launch(total);
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
        let (grid, block) = linear_launch(total);
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
        // The fused kernel reads idx_ptr from the GPU. With the per-launch sync
        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(idx_ptr);
                    return Err(Error::Backend(format!(
                        "hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(idx_ptr);
            }
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

        // Correctness Gate: Probe XNACK. If disabled, pageable unified memory migrations fail.
        if !self.props.xnack_enabled {
            println!(
                "[RocmDevice] Warning: XNACK is disabled on GFX device {}. Unified page advising bypassed; falling back to asynchronous stream copy.",
                self.ordinal
            );
            // Simulate/fallback to a null stream async memcpy (using stream 0)
            unsafe {
                let null_stream: *mut c_void = std::ptr::null_mut();
                check_hip(
                    "hipMemcpyAsync (fallback D2D)",
                    hipMemcpyAsync(
                        dev_ptr as *mut c_void,
                        dev_ptr,
                        rocm_storage.bytes,
                        HipMemcpyKind::DeviceToDevice,
                        null_stream,
                    ),
                )?;
            }
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
        self.kv_dequant_attention_impl(
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
        )
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
                let dummy_exps = RocmStorage::alloc_gpu(
                    &Shape::new(vec![(k * n).max(32) / 32]),
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::Native,
                    },
                    &self.allocator,
                    self.ordinal,
                )?;
                self.launch_fused_dequant_gemm_mxfp4(
                    a_storage,
                    b_storage,
                    &dummy_exps,
                    &out_storage,
                    m,
                    n,
                    k,
                )?;
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
                if !self.fused_dequant_gemm_config.lock().unwrap().enabled {
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
            DTypeStorage::ResidualPacked(cfg) => {
                // Mirror the forward `enabled` gate: when the fused backward path
                // is disabled, fall back to a standard matmul of dY against the
                // transposed dequantized B (same behavior as the forward fallback
                // at line ~2252). This fixes the asymmetry where the forward
                // dispatch honors `FusedDequantGemmConfig::enabled` but the
                // backward dispatch unconditionally calls the fused kernel.
                if !self.fused_dequant_gemm_config.lock().unwrap().enabled {
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
        // ─── enabled gate ────────────────────────────────────────────────
        let config = {
            let out_dims = out_shape.dims();
            if out_dims.len() != 3 {
                return Err(Error::Shape(
                    "qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
                ));
            }
            let seq_len = out_dims[0];
            let num_heads = out_dims[1];
            let head_dim = out_dims[2];
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
            ],
        )?;

        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd,
        );

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        dim: usize,
        base: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope: input lacks a valid device pointer".into(),
            ));
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

        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(pos_ptr);
                    return Err(Error::Backend(format!(
                        "hipStreamSynchronize failed: {}",
                        sync
                    )));
                }
                hipFree(pos_ptr);
            }
        }

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
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
        let x_s = as_rocm(x)?;
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        let c_s = as_rocm(c)?;
        let d_s = as_rocm(d)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_selective_scan(
            x_s,
            a_s,
            b_s,
            c_s,
            d_s,
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
        let rccl = self.rccl.lock().unwrap().clone();
        let is_f32 = dtype.arith == ArithType::F32;

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
                    rccl_handle.sum_gradients_device(send_ptr, out_ptr, total, stream_u64)?;
                } else {
                    // Multiple shards: accumulate on-device first, then all-reduce.
                    let temp_storage =
                        RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                    let temp_ptr = dev_ptr(&temp_storage)?;
                    self.device_accumulate_f32(inputs, temp_ptr)?;
                    rccl_handle.sum_gradients_device(temp_ptr, out_ptr, total, stream_u64)?;
                }

                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
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
                check_hip("hipMemcpy(D2D) all_reduce", unsafe {
                    crate::hipMemcpy(
                        dst_ptr,
                        src_ptr,
                        bytes,
                        crate::HipMemcpyKind::DeviceToDevice,
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
        let rccl = self.rccl.lock().unwrap().clone();
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
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
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
pub use crate::device::gemm_tuning::{GemmTileConfig, lookup_gemm_config, lookup_solution_index};

// Re-exports that pulled up `pub use crate::graph_capture::*` etc. in [see: `pub use`]

impl RocmDevice {
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
            crate::quantization::GcnArch::RDNA3 | crate::quantization::GcnArch::RDNA4
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

    /// Launch the JIT compiled fused dequantization backward matmul kernel (WI-T3 / F5). [see: `dX[M, K] = dY[M, N] @ B^T`]
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
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_blocks as u64 + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
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
    pub(crate) fn launch_fused_dequant_gemm_mxfp4(
        &self,
        a_storage: &RocmStorage,
        b_codes_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: a has no device ptr".into())
        })?;
        let b_codes_ptr = b_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: b_codes has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: b_exps has no device ptr".into())
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

    pub(crate) fn launch_compute_kernel_with_solution(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
        solution_index: Option<i32>,
        shared_mem_bytes: usize,
    ) -> Result<*mut c_void> {
        // Build the kernel source fresh per dispatch so the live QKV kernel [see: `const`, `concat!`]
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let hash = seahash::hash(kernel_source.as_bytes());
        // Include the GPU target in the cache key so a binary compiled for one
        let base_key = format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash);
        let cache_key = if let Some(sol) = solution_index {
            format!("{}_sol{}", base_key, sol)
        } else {
            base_key
        };

        let path = if let Some(cached_path) = self.hsaco_cache.get_cached_kernel(&cache_key) {
            cached_path
        } else {
            let code = jit_compile_hsaco(&kernel_source, entry, &self.gpu_target)?;
            self.hsaco_cache
                .cache_kernel(&cache_key, &kernel_source, &code)?
        };

        let path_c = std::ffi::CString::new(path.to_str().unwrap_or(""))
            .map_err(|e| Error::Backend(format!("hsaco path CString: {}", e)))?;
        let entry_c = std::ffi::CString::new(entry)
            .map_err(|e| Error::Backend(format!("entry CString: {}", e)))?;

        // Load the HIP module once per unique kernel; reuse the cached module +
        let mut module_cache = self.module_cache.lock().unwrap();
        let (_module, func) = if let Some(cached) = module_cache.get(&cache_key) {
            *cached
        } else {
            let mut module: *mut c_void = std::ptr::null_mut();
            check_hip("hipModuleLoad", unsafe {
                hipModuleLoad(&mut module, path_c.as_ptr())
            })?;
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
        if x_dims.len() != 2 || w_mat_dims.len() != 2 {
            return Err(Error::Shape("rmsnorm_matmul expects 2-D inputs".into()));
        }
        let m = x_dims[0];
        let k = x_dims[1];
        let n = w_mat_dims[1];
        if w_mat_dims[0] != k {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_mat_dims.to_vec(),
            });
        }
        if out_shape.dims() != &[m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {:?}",
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
        let (grid, block) = linear_launch(total);
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

    /// Launch standalone Q8_0 quantization HIP kernel.
    pub fn launch_quant_q8_0(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let (grid, block) = linear_launch((total + 31) / 32);
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

        let out_shape = Shape::from_slice(&[out_bytes]);
        let out_storage =
            RocmStorage::alloc_gpu(&out_shape, output_dtype, &self.allocator, self.ordinal)?;

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
                quant_bits: quant_bits as u8,
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
        let quant_bits_i = config.quant_bits as i32;

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
        // Wave64 mandate: kernel block dim is 256 = 4 wavefronts of 64 on
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
        out_storage: &RocmStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
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

        let mut xptr = x_ptr;
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut cptr = c_ptr;
        let mut dptr = d_ptr;
        let mut optr = out_ptr;
        let mut b_val = batch as i32;
        let mut d_val = dim_dstate as i32;
        let mut dd_val = dim_dinner as i32;
        let mut s_val = seq_len as i32;

        self.launch_compute_kernel(
            "grim_selective_scan",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut cptr),
                arg(&mut dptr),
                arg(&mut optr),
                arg(&mut b_val),
                arg(&mut d_val),
                arg(&mut dd_val),
                arg(&mut s_val),
            ],
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
        match storage {
            // Q8_0 is bit-exact between CPU `dequant_q80` and the ROCm
            // `dequantize_q8_0` kernel (block-major f16 scale + 32 int8).
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80) => {
                Ok(Some(self.dequantize_q8_0_host(bytes, elem_count)?))
            }
            // Q4_K: bit-exact between CPU `dequant_q4k` and the ROCm
            // `dequantize_q4k` kernel (interleaved-pair nibble layout,
            // 6-bit packed sub-block scale/min). Routed through the GPU kernel
            // instead of the CPU fallback.
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K) => {
                Ok(Some(self.dequantize_q4k_host(bytes, elem_count)?))
            }
            // IQ grid-decode formats and other schemes whose GPU kernels do not
            // yet match the CPU reference layouts remain on the CPU fallback.
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
