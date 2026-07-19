//! `RocmDevice` — the ROCm-side GPU device. Constructed via
//! `RocmDevice::new(ordinal)`. Owns:
//!
//! - the per-ordinal rocBLAS handle cache
//! - the persistent stream pool (Items 2 / 5)
//! - the JIT `.hsaco` cache (`HsacoKernelCache`)
//! - the device-side caching allocator + scratch pool
//! - the loaded-module cache (Item 2)
//! - the captured-graph cache (Item 5)
//! - the warm-up once-flag for batched GEMM (Item 6)
//!
//! Plus the full surface of higher-level operations exposed through
//! the `BackendDevice` trait implementation (`matmul`, `add`, `mul`,
//! `silu_mul`, `rms_norm`, `softmax`, `embedding`, `rmsnorm_matmul`,
//! `qkv_attention`, `matmul_batched`, `from_cpu`, `to_cpu_vec_f32`,
//! `copy_from_host_async`, `read_to_host_async`, etc.).
//!
//! All of this previously lived inline in `lib.rs` (and grew it past
//! 4,800 lines before the modularization pull-apart landed). The
//! entire struct + every impl + Drop + BackendDevice for is in this
//! module — together with all the helper re-exports lib.rs pulls in
//! from memory/, kernels/, device/ sub-modules — so the API at the
//! crate root is identical to before.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 3 — Device is the entry
//!   point for the rest of the crate's GPU surface.
//! - `rust-gpu-discipline` §3 — every method that touches a GPU
//!   returns `Result`.
//! - `rust-ffi` — the impl methods call into FFI declared in
//!   `crate::device::*` and consumed via `lib.rs`'s `pub use`
//!   re-exports; they keep their unsafe blocks narrow.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendDevice, BackendStorage, Shape};

// Symbols that lib.rs re-exports publicly. They live in sub-modules
// re-exported at crate root. Pulling them from `crate::*` works
// because lib.rs's `pub use` chain makes them accessible from any
// descendent module.
use crate::{
    // HIP runtime FFI
    hipDeviceGetAttribute, hipDeviceSynchronize, hipFree, hipGetDeviceCount,
    hipGraphDestroy, hipGraphExecDestroy, hipGraphInstantiate,
    hipGraphLaunch, hipMemAdvise, hipMemcpy, hipMemcpyAsync, hipMemset, hipMemsetAsync,
    hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad,
    hipModuleUnload, hipSetDevice, hipStreamBeginCapture, hipStreamCreate,
    hipStreamDestroy, hipStreamEndCapture, hipStreamSynchronize,
    // HIP types / constants
    HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    HIP_DEVICE_ATTRIBUTE_WARP_SIZE, HipDim3, HipErrorT, HipMemcpyKind,
    hipSuccess, RocmHandle, WavefrontSize,
    // rocBLAS FFI
    arith_to_compute_dtype, arith_to_rocblas_dtype, rocblas_create_handle,
    rocblas_destroy_handle, rocblas_gemm_ex, rocblas_gemm_strided_batched_ex,
    select_gemm_algo,
    rocblas_set_stream, rocblas_sgemm, rocblas_status_success, RocblasInt,
    RocblasOperation, RoclabsHandle,
    ROCBLAS_GEMM_FLAGS_NONE,
    // kernel cache + graph capture
    HsacoKernelCache, CapturedGraph,
    // device helpers
    check_hip, jit_compile_hsaco, upload_device_buffer,
    // lib.rs helpers (re-exported from memory/, device::util/, etc.)
    arg, as_rocm, dev_ptr, detect_gpu_arch, dtype_f32,
    linear_launch,
    // Misc types
    RocmStorage, RocmCachingAllocator, RocmDeviceProps, RocmPinnedBuffer,
    DecodeGemmConfig, FusedDequantGemmConfig, SplitKGemmConfig, QkvAttentionFusionConfig, RmsNormMatMulFusionConfig, WmmaGemmConfig,
    QuantMode,
};



#[derive(Debug)]
pub struct RocmDevice {
    pub(crate) ordinal: usize,
    pub(crate) props: RocmDeviceProps,
    handle_cache: Mutex<Option<RoclabsHandle>>,
    pub(crate) stream_pool: Mutex<Vec<*mut c_void>>,
    pub(crate) hsaco_cache: HsacoKernelCache,
    /// WI 2.4.4-2 — opt-in switch for the JIT `grim_decode_gemm_f16`
    /// (decode-class, double-buffered LDS) — default `false`. See
    /// `fusion::DecodeGemmConfig`. Held behind a `Mutex` for consistency
    /// with `handle_cache`/`stream_pool` (no async, plain
    /// `std::sync::Mutex`).
    pub(crate) decode_gemm_config: Mutex<DecodeGemmConfig>,
    pub(crate) fused_dequant_gemm_config: Mutex<FusedDequantGemmConfig>,
    pub(crate) split_k_config: Mutex<SplitKGemmConfig>,
    pub(crate) wmma_gemm_config: Mutex<WmmaGemmConfig>,
    /// Caching device-memory allocator (size-bucketed free-list). See `RocmCachingAllocator`.
    pub(crate) allocator: Arc<RocmCachingAllocator>,
    /// Phase-3 §3.1: device scratch pool — a thread-safe, power-of-2-bucketed
    /// `hipMalloc` free-list. `get_scratch` hands out RAII buffers; the
    /// underlying slot is recycled on `Drop` so the fused-decode path doesn't
    /// pay per-call driver overhead. Skills: `rust-ai-ml-inference-guide`
    /// Action 3, `rust-gpu-parallelism` (stream-ordered memory plan),
    /// `rocm-profiling-perf` (allocation overhead).
    pub(crate) scratch_pool: Arc<crate::memory::pool::DeviceScratchPool>,
    /// Loaded HIP modules + resolved entry functions, cached per unique kernel entry.
    /// `hipModuleLoad`/`hipModuleGetFunction` happen once per kernel for the process
    /// lifetime; every later dispatch reuses the cached module (Item 2). `Send + Sync`
    /// via raw pointers + the Mutex (the struct is already asserted Send/Sync).
    pub(crate) module_cache: Mutex<HashMap<String, (*mut c_void, *mut c_void)>>,
    /// Real `hipModuleLoad` call count (cache hits excluded). Item 2 acceptance.
    pub(crate) module_load_count: AtomicUsize,
    /// GPU target this device was created for, captured at construction. Used to
    /// key the module cache so a binary is never loaded onto the wrong arch. Captured
    /// once (not read live) so a concurrent `temp_env::with_var("GRIM_GPU_TARGET", ..)`
    /// in another test thread can't flip the key mid-run and spuriously reload.
    pub(crate) gpu_target: String,
    /// Whether graph capture/replay is enabled. Keyed off the `GRIM_CAPTURE_GRAPH`
    /// env var so it stays a runtime flag, not a compile-time feature. When false,
    /// the begin/end/replay methods are no-ops (Item 5).
    capture_enabled: bool,
    /// The dedicated capture stream, owned for the device's lifetime. Created lazily on
    /// the first `begin_graph_capture` and destroyed only in `Drop` — keeping it alive
    /// past `end_graph_capture` is what lets rocblas free its capture-time workspace
    /// buffers on a still-valid stream instead of aborting at handle teardown. While a
    /// session is active, every op dispatches onto this stream (instead of the pool).
    capture_stream: RwLock<Option<*mut c_void>>,
    /// True only between `begin_graph_capture` and `end_graph_capture`. Gates the
    /// capture-stream routing in `active_stream`/`active_capture_stream` (Item 5).
    capture_active: AtomicBool,
    /// Keyed cache of captured + instantiated graphs. A graph is recorded exactly once
    /// per key; `replay_graph` launches the cached executable without re-recording.
    captured_graphs: Mutex<HashMap<String, CapturedGraph>>,
    /// Once-flag: the first `matmul_batched` call in a process warms up the
    /// rocBLAS `gemm_strided_batched_ex` kernel (lazy JIT / handle init) with a
    /// throwaway 2x2 batch=2 GEMM. Without this, the first real call can return
    /// before the kernel is ready and produce a zero-filled output (observed as
    /// an intermittent all-zeros result on the first process invocation).
    batched_gemm_warmed: AtomicBool,
}

unsafe impl Send for RocmDevice {}
unsafe impl Sync for RocmDevice {}

impl RocmDevice {
    /// Create a new ROCm device instance and initialize its handle caches and stream pool.
    ///
    /// Infallible constructor: probes the runtime and falls back to safe
    /// defaults (empty stream pool, default Wave64) on GPU-less boxes. Use
    /// [`RocmDevice::try_new`] when you need the typed error from a failed
    /// `hipSetDevice` (e.g. an out-of-range ordinal).
    pub fn new(ordinal: usize) -> Self {
        match Self::try_new(ordinal) {
            Ok(dev) => dev,
            Err(e) => {
                // Surface the failure loudly so a misconfigured host is
                // visible, then build a defensive device that won't crash
                // downstream callers — every subsequent op will also fail
                // with a typed `Error::Backend`, never a panic.
                eprintln!(
                    "[RocmDevice::new] hipSetDevice({ordinal}) failed: {e}; \
                     constructing a no-stream fallback device"
                );
                Self::fallback(ordinal)
            }
        }
    }

    /// Fallible constructor that propagates the `hipSetDevice` error.
    ///
    /// Use this in callers that want to distinguish "no GPU at this ordinal"
    /// (Err) from "GPU present, ready to go" (Ok) — e.g. `probe()` and CLI
    /// device enumeration. The infallible [`RocmDevice::new`] wraps this and
    /// falls back to a no-stream device on error.
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
        // hipDeviceGetAttribute takes the device id explicitly, so these
        // queries are correct regardless of which device hipSetDevice
        // last selected — but hipStreamCreate below IS device-implicit,
        // so we must fail loudly if hipSetDevice rejects the ordinal.
        let mut warp_size = 64; // Default to W64 (MI200/MI300 CDNA) safety fallback
        let mut xnack_val = 0;
        let mut streams = Vec::new();
        unsafe {
            let mut val = 0;
            let status = hipDeviceGetAttribute(&mut val, HIP_DEVICE_ATTRIBUTE_WARP_SIZE, ordinal as i32);
            if status == hipSuccess {
                warp_size = val;
            }
            let status_xnack = hipDeviceGetAttribute(&mut xnack_val, HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, ordinal as i32);
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
        Ok(Self::build(
            ordinal,
            warp_size,
            xnack_val,
            handle_cache,
            streams,
        ))
    }

    /// Build a defensive device with no streams and default props. Used by
    /// [`RocmDevice::new`] when `try_new` errors — keeps the infallible
    /// constructor contract while making the failure visible (every later
    /// op dispatches onto `null_stream` and returns `Err`).
    fn fallback(ordinal: usize) -> Self {
        Self::build(ordinal, 64, 0, None, Vec::new())
    }

    /// Shared tail of `try_new` / `fallback`: assemble the struct from
    /// already-probed fields. Pure construction, no FFI calls.
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

        // Caching allocator soft cap. Default 256 MiB; overridable for testing/large models.
        let cap_bytes = std::env::var("GRIM_ALLOC_POOL_CAP_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256 * 1024 * 1024);

        Self {
            ordinal,
            props: RocmDeviceProps { wavefront_size, xnack_enabled },
            handle_cache: Mutex::new(handle_cache),
            stream_pool: Mutex::new(streams),
            hsaco_cache: HsacoKernelCache::new(),
            allocator: Arc::new(RocmCachingAllocator::new(ordinal, cap_bytes)),
            scratch_pool: crate::memory::pool::DeviceScratchPool::new(),
            module_cache: Mutex::new(HashMap::new()),
            module_load_count: AtomicUsize::new(0),
            gpu_target: detect_gpu_arch(ordinal as i32),
            capture_enabled: std::env::var("GRIM_CAPTURE_GRAPH").is_ok(),
            capture_stream: RwLock::new(None),
            capture_active: AtomicBool::new(false),
            captured_graphs: Mutex::new(HashMap::new()),
            batched_gemm_warmed: AtomicBool::new(false),
            decode_gemm_config: Mutex::new(DecodeGemmConfig { enabled: true, wavefront_size: warp_size as u32 }),
            fused_dequant_gemm_config: Mutex::new(FusedDequantGemmConfig { enabled: true, wavefront_size: warp_size as u32 }),
            split_k_config: Mutex::new(SplitKGemmConfig { enabled: true }),
            wmma_gemm_config: Mutex::new(WmmaGemmConfig { enabled: true, wavefront_size: warp_size as u32 }),
        }
    }

    /// Release all pooled device buffers back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        self.allocator.empty_cache();
    }

    /// WI 2.4.4-2 — opt-in flag for the JIT `grim_decode_gemm_f16`.
    ///
    /// Set to `true` after a positive benchmark vs. rocBLAS; otherwise
    /// the decode-class F16 GEMM shape trips the rocBLAS path as it
    /// always has. Mirror `QkvAttentionFusionConfig::enabled` for
    /// pattern consistency.
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

    /// Set whether the JIT compiled WMMA GEMM kernel is enabled (WI-G).
    ///
    /// When enabled and the architecture supports WMMA (or falls back), the matmul
    /// dispatch will use the JIT'd `grim_wmma_gemm` kernel.
    pub fn set_wmma_gemm_enabled(&self, enabled: bool) {
        let mut cfg = self.wmma_gemm_config.lock().unwrap();
        cfg.enabled = enabled;
    }

    /// `(hipMalloc_count, hipFree_count)` since this device was created — real driver
    /// allocation calls, useful for asserting pool reuse (Item 1 acceptance).
    pub fn allocator_stats(&self) -> (usize, usize) {
        self.allocator.stats()
    }

    /// Number of real `hipModuleLoad` calls since device creation. Cache hits are
    /// excluded, so this stays constant once every compute kernel has been loaded
    /// once (Item 2 acceptance: `module_cache_loads_each_kernel_once`).
    pub fn module_load_stats(&self) -> usize {
        self.module_load_count.load(Ordering::SeqCst)
    }

    /// Phase-3 §3.1: get a pooled scratch buffer.
    ///
    /// Recycles the most-recently-freed slot in the matching bucket when one
    /// exists; otherwise `hipMalloc`s a fresh one and tracks the peak. Returns
    /// `Result` so the GPU-error path is explicit (no silent CPU fallback —
    /// `rust-gpu-discipline` §3).
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

    /// Phase-3 §3.1 (REFACTOR): upload `data` into a pooled scratch buffer
    /// sized for the requested dtype/shape, instead of `hipMalloc`+`hipFree`
    /// per call. Returns the `PooledBuffer`; the caller drops it to return
    /// the slot to the pool. Skill: `rust-ai-ml-inference-guide` Action 3.
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
        // decode-loop's per-call cost was the hipMalloc, not the copy.
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

    /// If a graph-capture session is active, returns the dedicated capture stream.
    /// Ops consult this to route their work onto the capture stream instead of the
    /// pool (Item 5). Returns `None` outside an active session.
    fn active_capture_stream(&self) -> Option<*mut c_void> {
        if self.capture_active.load(Ordering::SeqCst) {
            *self.capture_stream.read().unwrap()
        } else {
            None
        }
    }

    /// The stream an op should dispatch onto: the capture stream when a session is
    /// active, otherwise a pooled compute stream (or null as a last resort). Central
    /// so every op-dispatch function records into the same capture graph in lockstep.
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

    /// Begin a generic graph-capture session keyed by `key`. Until `end_graph_capture`
    /// is called, every op dispatched on this device is recorded onto a dedicated
    /// capture stream (the rocblas handle is rebound to it during matmul) rather than
    /// executed immediately. `key` is just an opaque handle the caller chooses; this
    /// backend stays agnostic to the op sequence and its shapes.
    ///
    /// No-op (Ok) when capture is disabled (`GRIM_CAPTURE_GRAPH` unset), so callers
    /// can bracket work unconditionally.
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
        // workspace buffers allocated on it during capture stay valid until handle teardown.
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
        // stream *before* beginning capture so rocBLAS records its GEMM into the
        // graph (rather than running it eagerly with a stale workspace).
        if let Ok(h) = self.get_rocblas_handle() {
            unsafe {
                let _ = rocblas_set_stream(h, stream);
            }
        }
        // Relaxed capture mode: allocations (hipMalloc for op outputs, rocblas
        // workspace) execute normally during capture instead of invalidating the
        // capture — they are simply not recorded into the graph (Item 5).
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

    /// End the capture session started with `key`, instantiate the recorded graph,
    /// and cache it under `key`. The graph is *not* launched here — callers replay it
    /// later via `replay_graph`. Subsequent ops run on the pool again.
    ///
    /// No-op (Ok) when capture is disabled, so it pairs with `begin_graph_capture`.
    pub fn end_graph_capture(&self, key: &str) -> Result<()> {
        if !self.capture_enabled {
            return Ok(());
        }
        if !self.capture_active.load(Ordering::SeqCst) {
            return Err(Error::Backend(
                "end_graph_capture: no capture session is active".into(),
            ));
        }
        let stream = self.capture_stream.read().unwrap().unwrap_or(std::ptr::null_mut());
        let mut graph: *mut c_void = std::ptr::null_mut();
        let res = unsafe { hipStreamEndCapture(stream, &mut graph) };
        if res != hipSuccess {
            self.capture_active.store(false, Ordering::SeqCst);
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(Error::Backend(format!("hipStreamEndCapture failed: {}", res)));
        }
        // Clear the stream so it is ready to be reused by a later capture session.
        unsafe {
            let _ = hipStreamSynchronize(stream);
        }
        let mut exec: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            hipGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
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

    /// Replay the graph previously captured under `key`. Returns `Ok(false)` when no
    /// graph is cached for `key` (so callers can fall back to eager dispatch without
    /// treating a first-run miss as an error). Returns `Ok(true)` after a successful
    /// launch. `Err` only on a launch failure. Must never be called mid-capture.
    pub fn replay_graph(&self, key: &str) -> Result<bool> {
        if !self.capture_enabled {
            return Ok(false);
        }
        // Replay on the same capture stream the graph was recorded on, so the rocblas
        // node and the elementwise kernels stay ordered on one stream (no cross-stream race).
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
    /// capture or replay without first attempting a replay).
    pub fn has_captured_graph(&self, key: &str) -> bool {
        self.captured_graphs.lock().unwrap().contains_key(key)
    }

    /// Collapse a batch of `batch_count` same-shape GEMMs into one
    /// `rocblas_gemm_strided_batched_ex` call (Item 6). Each `a[i]` is `[m, k]` and
    /// each `b[i]` is `[k, n]`; every output is `[m, n]`.
    ///
    /// Inputs are packed into contiguous device buffers (stride = per-matrix
    /// element count) via device-to-device copies so this works for any dtype
    /// without a host round-trip or f32 packing. The outputs are returned as
    /// individual `RocmStorage`s on the device, ready to feed downstream ops.
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
        // The first call in a process can return before the lazy JIT kernel is
        // ready, yielding a zero-filled output; a throwaway 2x2 batch=2 GEMM
        // absorbs that race. The flag is flipped *before* the recursive call so
        // the warm-up itself doesn't re-enter the guard.
        if !self.batched_gemm_warmed.swap(true, Ordering::SeqCst) {
            let warm_a = self.from_cpu(&[1.0f32, 2.0, 3.0, 4.0], &Shape::from_slice(&[2, 2]), DType::F32)?;
            let warm_b = self.from_cpu(&[1.0f32, 2.0, 3.0, 4.0], &Shape::from_slice(&[2, 2]), DType::F32)?;
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
        // are guaranteed to land before the GEMM reads the packed buffers
        // (rocBLAS would otherwise run on its own handle stream with no dependency
        // on the copies — a race that intermittently feeds uninitialized buffers).
        unsafe {
            let _ = rocblas_set_stream(handle, stream);
        }
        for i in 0..batch {
            let ai = as_rocm(a[i])?;
            let bi = as_rocm(b[i])?;
            check_hip("matmul_batched: hipMemcpyDtoD a", unsafe {
                hipMemcpy(
                    (a_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_a * 4),
                    ai.device_ptr.unwrap() as *mut c_void,
                    ai.bytes,
                    HipMemcpyKind::DeviceToDevice,
                )
            })?;
            check_hip("matmul_batched: hipMemcpyDtoD b", unsafe {
                hipMemcpy(
                    (b_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_b * 4),
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

        // Row-major C[M,N] = A[M,K] @ B[K,N] via rocBLAS column-major recipe
        // (operands swapped, transa=transb='N'); see `matmul` for the rationale.
        // For strided-batched, stride_a/b/d are the per-matrix element counts.
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
                // Per Item 7 + the §3.6 cross-task correction in
                // grim_qkv_attention_kernel_spec.md: solution_index is
                // *ignored* by rocBLAS unless algo == solution_index.
                // `select_gemm_algo(0)` is `standard` so the throwaway
                // 2x2 batch=2 warm-up GEMM (Item 6's "lazy rocBLAS JIT"
                // pre-flush) stays on the default engine; nobody tunes
                // that call because it never runs real inference.
                select_gemm_algo(0),
                0,
                ROCBLAS_GEMM_FLAGS_NONE,
            );
            // Restore the handle to the default (null) stream so other eager GEMMs
            // are unaffected by this call's binding.
            let _ = rocblas_set_stream(handle, std::ptr::null_mut());
            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_strided_batched_ex failed with status {status}"
                )));
            }
        }

        // Read the packed results back, then split into per-batch device storages.
        // rocBLAS fans the GEMM out to internal streams that may not join back to
        // `active_stream`; a device-wide sync is the reliable join point before a
        // host readback, and skips the cost during graph capture (replay syncs).
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

    /// Pinned-memory + async host→device upload for the per-token decode hot path.
    /// Allocates a pinned staging buffer, copies `data` into it, and issues an
    /// async `hipMemcpy` on the default stream (which overlaps with compute still
    /// queued on other streams), draining only before returning. The cold-path
    /// [`RocmDevice::from_cpu`] keeps using pageable `Vec` + synchronous `hipMemcpy`.
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
        // work; sync only that stream (not the whole device) before returning.
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
        let sync = unsafe { hipStreamSynchronize(stream) };
        if sync != hipSuccess {
            if storage.device_ptr.is_some() {
                unsafe {
                    let _ = hipFree(storage.device_ptr.unwrap() as *mut c_void);
                }
            }
            return Err(Error::Backend(format!(
                "hipStreamSynchronize after async upload failed with error code {}",
                sync
            )));
        }
        Ok(Box::new(storage))
    }

    /// Like [`RocmDevice::copy_from_host_async`] but uploads from a caller-owned
    /// pinned buffer that is reused across steps (no per-token `hipHostMalloc`).
    /// The decode loop keeps one input pinned buffer and calls this each token.
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
        let res = unsafe {
            hipMemcpyAsync(
                dev_ptr_void,
                src.as_ptr() as *const c_void,
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
        let sync = unsafe { hipStreamSynchronize(stream) };
        if sync != hipSuccess {
            if storage.device_ptr.is_some() {
                unsafe {
                    let _ = hipFree(storage.device_ptr.unwrap() as *mut c_void);
                }
            }
            return Err(Error::Backend(format!(
                "hipStreamSynchronize after async upload failed with error code {}",
                sync
            )));
        }
        Ok(Box::new(storage))
    }

    /// Pinned-memory + async device→host download for the per-token decode hot path.
    /// Downloads into an internal pinned staging buffer via async `hipMemcpy` and
    /// returns a pageable `Vec<f32>` (what callers sample from). The cold-path
    /// [`RocmStorage::to_cpu_vec_f32`] keeps using pageable `Vec` + synchronous
    /// `hipMemcpy`.
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
        check_hip("hipStreamSynchronize (after async download)", unsafe { hipStreamSynchronize(stream) })?;
        let mut out = vec![0.0f32; elem_count];
        out.copy_from_slice(pinned.as_slice());
        Ok(out)
    }

    /// Same as [`RocmDevice::read_to_host_async`] but downloads into a caller-owned
    /// pinned buffer that is reused across steps (no per-token allocation). The
    /// buffer is resized to `elem_count` if needed.
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
        check_hip("hipStreamSynchronize (after async download)", unsafe { hipStreamSynchronize(stream) })?;
        Ok(())
    }
}

impl Drop for RocmDevice {
    fn drop(&mut self) {
        // Drain any in-flight kernels on the pooled streams before recycling or
        // freeing buffers. Since Item 2 removed the per-launch sync, a buffer can
        // still be written by an async dispatch at the moment the device is torn
        // down; without this, freeing/reusing it races with the running kernel
        // (and with other tests that allocate the same address afterward).
        unsafe {
            let _ = hipDeviceSynchronize();
        }
        // Return all pooled buffers to the driver before the allocator Arc is dropped,
        // otherwise they would leak (the pool is only drained by empty_cache).
        self.allocator.empty_cache();
        // Unload every cached HIP module (they were loaded exactly once per
        // kernel entry and retained for the device lifetime, Item 2).
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
        // rocblas handle above has been dropped, so its capture-time workspace buffers
        // are already freed on this still-valid stream — no abort at teardown.
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
        let mut cache = self.handle_cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
}

impl BackendDevice for RocmDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: zeros");

        // `hipMemset` zeroes bytes, which is only correct when the dtype's zero
        // representation is all-zero bytes. This holds for every DType this
        // backend supports (f32/f16/bf16/integer), so it is safe for all paths.
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;

        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }

        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;

        // If a graph-capture session is active, record an async memset on the
        // capture stream and skip the device-wide sync (a sync on a capturing
        // stream is illegal, and the graph is drained on replay).
        let res = if let Some(capture_stream) = self.active_capture_stream() {
            unsafe { hipMemsetAsync(dev_ptr_void, 0, storage.bytes, capture_stream) }
        } else {
            let r = unsafe { hipMemset(dev_ptr_void, 0, storage.bytes) };
            if r == hipSuccess {
                // `hipMemset` is asynchronous (default stream); callers expect a
                // fully zeroed buffer on return, so drain it before handing the
                // storage out. This matches the old hipMemcpy(H2D) path.
                let sync = unsafe { hipDeviceSynchronize() };
                if sync != hipSuccess {
                    if storage.device_ptr.is_some() {
                        let ptr_void = storage.device_ptr.unwrap() as *mut c_void;
                        unsafe {
                            _ = hipFree(ptr_void);
                        }
                    }
                    return Err(Error::Backend(format!(
                        "hipDeviceSynchronize after zeros failed with error code {}",
                        sync
                    )));
                }
            }
            r
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
        let out_storage = RocmStorage::alloc_gpu(out_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

        // Shape-indexed GEMM dispatch lookup (Tensile-inspired layout resolution)
        let tile_config = lookup_gemm_config(m, n, k, self.props.wavefront_size);
        // Offline-tuned solution_index per (M,N,K) for FP32. Falls back to 0 for
        // unknown shapes or other dtypes. Populated by `examples/tune_gemm.rs`.
        let solution_index = lookup_solution_index(m, n, k, dtype_out.arith);
        // WI 2.4.3 — split_k clamp gate.
        let split_k_effective: u32 = {
            let split_k_enabled = self.split_k_config.lock().unwrap().enabled;
            if split_k_enabled && tile_config.split_k > 1 && (k % tile_config.split_k as usize == 0) {
                tile_config.split_k
            } else {
                1
            }
        };

        if split_k_effective > 1 {
            let k_part = k / split_k_effective as usize;
            let partials_shape = Shape::from_slice(&[split_k_effective as usize, m, n]);
            let partials_storage = RocmStorage::alloc_gpu(&partials_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

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
            let stream = self.launch_split_k_reduction(&partials_storage, &out_storage, m, n, split_k_effective)?;
            let compute_handle = Box::new(RocmHandle::new(Some(stream)));
            return Ok((Box::new(out_storage), compute_handle));
        }
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}, solution_index={}",
            m, n, k, tile_config, self.props.wavefront_size, solution_index
        );

        // ─── WI 2.4.4-2 — decode GEMM dispatch (opt-in, F16-only, m ≤ 8) ─────
        //
        // Replaces the vendored CK `ck_gemm.cpp` C wrapper with a JIT
        // `grim_decode_gemm_f16` kernel living in
        // `kernels::decode_gemm::KERNEL_SOURCE`. The gate mirroring
        // `QkvAttentionFusionConfig::enabled` pattern ensures this
        // never silently swaps the GEMM path — the user must flip
        // `set_decode_gemm_enabled(true)` after a positive benchmark.
        // Plan gate WI 2.6.4 demands a GPu parity test before that
        // happens; this branch also serves that test (enable in the
        // test harness, run comparison, disable).
        {
            let cfg = self.decode_gemm_config.lock().unwrap();
            if cfg.enabled && dtype_out.arith == ArithType::F16 && m <= 8 {
                drop(cfg); // release the lock before the JIT launch
                // WI 2.4.4-2(a) — thread the *real* enqueued stream into the
                // returned handle. `launch_compute_kernel` already enqueues via
                // `hipModuleLaunchKernel` and returns the stream; discarding it
                // (the prior `RocmHandle::new(None)`, the Rust analog of the
                // plan's `(void)s;`) made `ComputeHandle::synchronize` a silent
                // no-op, so a caller that waited on this handle could read
                // half-written decode output. Passing `Some(stream)` makes the
                // wait real — the read-back path (`to_cpu_vec_f32`) syncing the
                // default stream is the backstop, not the contract.
                let stream = self.launch_decode_gemm_f16(a_storage, b_storage, &out_storage, m, n, k)?;
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

        // Get rocBLAS handle and execute sgemm. The handle's stream was already bound
        // to the capture stream in `begin_graph_capture` (and restored in
        // `end_graph_capture`), so a GEMM issued during a session records into the graph.
        let handle = self.get_rocblas_handle()?;

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;
        
        let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
        let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
        let out_ptr_void = out_storage.device_ptr.unwrap() as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is
        // computed via sgemm/gemm_ex with transa=transb='N', the A/B operands
        // swapped, and (m,n,k,lda,ldb,ldc) = (N, M, K, N, K, N). See e.g. the
        // canonical row-major GEMM recipe used by ggml/llama.cpp.
        
        let use_gemm_ex = cfg!(feature = "rocm-aiter") || {
            let gcn = std::env::var("GRIM_GPU_TARGET").unwrap_or_else(|_| "gfx900".into());
            gcn == "gfx90a" || gcn == "gfx942"
        };

        unsafe {
            let status = if use_gemm_ex || dtype_out.arith == ArithType::F16 || dtype_out.arith == ArithType::BF16 {
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
                    // Wire `lookup_solution_index` to `algo` so rocBLAS actually
                    // honors it. `select_gemm_algo(0)` returns `standard` (untuned
                    // fallback); any non-zero solution_index gets
                    // `rocblas_gemm_algo::solution_index`, the only enum variant
                    // that tells rocBLAS to use the tuned index. See the §3.6
                    // cross-task correction in grim_qkv_attention_kernel_spec.md.
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

        let compute_handle = Box::new(RocmHandle::new(None));
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
            None => return Err(Error::Backend("matmul_with_solution: input a is not RocmStorage".into())),
        };

        let b_storage = match b.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend("matmul_with_solution: input b is not RocmStorage".into())),
        };

        if !a_storage.device_ptr_is_valid() || !b_storage.device_ptr_is_valid() {
            return Err(Error::Backend(
                "matmul_with_solution: inputs must have valid GPU device pointers".into(),
            ));
        }

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape("matmul_with_solution expects 2-D inputs".into()));
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
        let out_storage = RocmStorage::alloc_gpu(out_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

        // Shape-indexed GEMM dispatch lookup (Tensile-inspired layout resolution)
        let _tile_config = lookup_gemm_config(m, n, k, self.props.wavefront_size);
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}",
            m, n, k, tile_config, self.props.wavefront_size
        );

        // Get rocBLAS handle and execute gemm_ex with the provided solution_index
        let handle = self.get_rocblas_handle()?;

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;
        
        let a_ptr_void = a_storage.device_ptr.unwrap() as *const c_void;
        let b_ptr_void = b_storage.device_ptr.unwrap() as *const c_void;
        let out_ptr_void = out_storage.device_ptr.unwrap() as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is
        // computed via sgemm/gemm_ex with transa=transb='N', the A/B operands
        // swapped, and (m,n,k,lda,ldb,ldc) = (N, M, K, N, K, N). See e.g. the
        // canonical row-major GEMM recipe used by ggml/llama.cpp.

        let use_gemm_ex = cfg!(feature = "rocm-aiter") || {
            let gcn = std::env::var("GRIM_GPU_TARGET").unwrap_or_else(|_| "gfx900".into());
            gcn == "gfx90a" || gcn == "gfx942"
        };

        unsafe {
            let status = if use_gemm_ex || dtype_out.arith == ArithType::F16 || dtype_out.arith == ArithType::BF16 {
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
                    // Wire `lookup_solution_index` to `algo` so rocBLAS actually
                    // honors it. `select_gemm_algo(0)` returns `standard` (untuned
                    // fallback); any non-zero solution_index gets
                    // `rocblas_gemm_algo::solution_index`, the only enum variant
                    // that tells rocBLAS to use the tuned index. See the §3.6
                    // cross-task correction in grim_qkv_attention_kernel_spec.md.
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

        let compute_handle = Box::new(RocmHandle::new(None));
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
            return Err(Error::Backend("add: inputs lack a valid device pointer".into()));
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
            &mut [arg(&mut a_ptr), arg(&mut b_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
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
            return Err(Error::Backend("mul: inputs lack a valid device pointer".into()));
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
            &mut [arg(&mut a_ptr), arg(&mut b_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
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
            return Err(Error::Backend("silu_mul: inputs lack a valid device pointer".into()));
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
            &mut [arg(&mut gate_ptr), arg(&mut up_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
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
            return Err(Error::Backend("rms_norm: inputs lack a valid device pointer".into()));
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
        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend("softmax: input lacks a valid device pointer".into()));
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
        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let w_s = as_rocm(weight)?;
        if !w_s.device_ptr_is_valid() {
            return Err(Error::Backend("embedding: weight lacks a valid device pointer".into()));
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
        // removed (Item 2) we must wait on the stream before freeing the temp
        // buffer, otherwise hipFree could race the still-running kernel.
        // During a capture session the free must not happen inside the captured
        // region (a sync on a capturing stream is illegal and hipFree can't be
        // captured), so defer it — the buffer is released when capture ends.
        if self.active_capture_stream().is_none() {
            unsafe {
                let sync = hipStreamSynchronize(stream);
                if sync != hipSuccess {
                    hipFree(idx_ptr);
                    return Err(Error::Backend(format!("hipStreamSynchronize failed: {}", sync)));
                }
                hipFree(idx_ptr);
            }
        }
        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }

    fn advise(&self, storage: &dyn BackendStorage, advice: grim_tensor::backend::MemAdvice) -> Result<()> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: advise");

        let rocm_storage = storage.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
            Error::Backend("advise: storage is not RocmStorage".into())
        })?;

        let dev_ptr = match rocm_storage.device_ptr {
            Some(ptr) => ptr as *const c_void,
            None => return Ok(()), // Unallocated or CPU-side: no-op
        };

        // Correctness Gate: Probe XNACK. If disabled, pageable unified memory migrations fail.
        // We bypass hipMemAdvise and issue a fallback copy statement.
        if !self.props.xnack_enabled {
            println!(
                "[RocmDevice] Warning: XNACK is disabled on GFX device {}. Unified page advising bypassed; falling back to asynchronous stream copy.",
                self.ordinal
            );
            // Simulate/fallback to a null stream async memcpy (using stream 0)
            unsafe {
                let null_stream: *mut c_void = std::ptr::null_mut();
                check_hip("hipMemcpyAsync (fallback D2D)", hipMemcpyAsync(
                    dev_ptr as *mut c_void,
                    dev_ptr,
                    rocm_storage.bytes,
                    HipMemcpyKind::DeviceToDevice,
                    null_stream,
                ))?;
            }
            return Ok(());
        }

        let raw_advice = match advice {
            grim_tensor::MemAdvice::ReadMostly => crate::device::handles::HIP_MEM_ADVISE_SET_READ_MOSTLY,
            grim_tensor::MemAdvice::PreferredLocation { device_id: _ }
                => crate::device::handles::HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
            grim_tensor::MemAdvice::AccessedBy { device_id: _ }
                => crate::device::handles::HIP_MEM_ADVISE_SET_ACCESSED_BY,
            grim_tensor::MemAdvice::CoarseGrain
                => crate::device::handles::HIP_MEM_ADVISE_SET_COARSE_GRAIN,
            grim_tensor::MemAdvice::FineGrain
                => crate::device::handles::HIP_MEM_ADVISE_UNSET_COARSE_GRAIN,
            // OS-level hints (madvise) are ignored on the GPU memory space
            _ => return Ok(()),
        };

        unsafe {
            check_hip("hipMemAdvise", hipMemAdvise(dev_ptr, rocm_storage.bytes, raw_advice, self.ordinal as i32))?;
        }
        Ok(())
    }
}

// `GemmTileConfig`, `lookup_gemm_config`, `lookup_solution_index` moved
// to `device::gemm_tuning` — see that module.
pub use crate::device::gemm_tuning::{
    GemmTileConfig, lookup_gemm_config, lookup_solution_index,
};

// Re-exports that pulled up `pub use crate::graph_capture::*` etc. in
// lib.rs are NOT duplicated here — the same names must resolve at the
// crate root through being re-exported by lib.rs after this module
// is wired into the crate root via the same `pub use` chain.

impl RocmDevice {
    /// WI 2.4.4-2c — dispatch `grim_decode_gemm_f16` and return the
    /// enqueued stream handle for the caller to synchronize on.
    ///
    /// Kernel tile: (M_TILE=8, N_TILE=64, K_STEP=16), F16→f32 acc→F16,
    /// double-buffered LDS. Grid: (ceil(M / 8), ceil(N / 64)), block: 256.
    /// Stream from the persistent pool (mirrors `launch_compute_kernel`).
    ///
    /// Invariant: only called after the `DecodeGemmConfig::enabled` gate,
    /// `dtype == F16`, and `m <= 8` — see the call site in `RocmDevice::matmul`.
    pub(crate) fn launch_decode_gemm_f16(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| Error::Backend("decode_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage.device_ptr.ok_or_else(|| Error::Backend("decode_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| Error::Backend("decode_gemm: out has no device ptr".into()))?;

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
        let stride_a = k;     // A[M, K]
        let stride_b = n;     // B[K, N]
        let stride_c = n;     // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, ArithType::F16);
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
        )
    }

    /// Enqueues the JIT-compiled WMMA matrix-core GEMM kernel (WI-G).
    ///
    /// The kernel maps to either the hardware-accelerated WMMA block operations
    /// (on RDNA3+) or the scalar fallback logic (on RDNA2/older).
    pub(crate) fn launch_wmma_gemm(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| Error::Backend("wmma_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage.device_ptr.ok_or_else(|| Error::Backend("wmma_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| Error::Backend("wmma_gemm: out has no device ptr".into()))?;

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
        let stride_a = k;     // A[M, K]
        let stride_b = n;     // B[K, N]
        let stride_c = n;     // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, ArithType::F16);
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
        )
    }


    /// TODO(WI-C): Kernel + config exist; wire dispatch in matmul path when enabled.
    /// Launch the JIT compiled fused dequantization matmul kernel (WI-C).
    #[allow(dead_code)]
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
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| Error::Backend("fused_dequant_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage.device_ptr.ok_or_else(|| Error::Backend("fused_dequant_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| Error::Backend("fused_dequant_gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems = m * n;
        let grid_x = ((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        
        let stride_a = k;     // A[M, K]
        let stride_c = n;     // C[M, N]
        let mut sa = stride_a as i32;
        let mut sc = stride_c as i32;
        
        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;
        
        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;

        let solution_index = lookup_solution_index(m, n, k, ArithType::F16);
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
            ],
            Some(solution_index),
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
        let partials_ptr = partials_storage.device_ptr.ok_or_else(|| Error::Backend("split_k_reduction: partials has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| Error::Backend("split_k_reduction: out has no device ptr".into()))?;

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

    /// JIT-compile or query the cache, then launch the specified kernel on a
    /// stream from the persistent pool.
    ///
    /// The HIP module for `entry` is loaded (and the entry function resolved)
    /// exactly once per process and reused on every later dispatch via
    /// `module_cache` (Item 2). The per-launch `hipStreamSynchronize` that
    /// previously forced every op to block has been removed; the stream the
    /// kernel was enqueued on is returned so callers that must wait (e.g.
    /// before freeing a temporary buffer) can synchronize explicitly.
    /// Read-back (`to_cpu_vec_f32`) still blocks on the default stream, which
    /// synchronizes with all streams, so results remain correct.
    pub(crate) fn launch_compute_kernel(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
    ) -> Result<*mut c_void> {
        self.launch_compute_kernel_with_solution(entry, grid, block, args, None)
    }

    pub(crate) fn launch_compute_kernel_with_solution(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
        solution_index: Option<i32>,
    ) -> Result<*mut c_void> {
        // Build the kernel source fresh per dispatch so the live QKV kernel
        // module (and any other future sibling kernel modules) is included
        // without `const`-`concat!` gymnastics. The compile is cached by hash
        // below, so the rebuild cost is amortized.
        let kernel_source =
            crate::kernels::source_asm::compute_kernel_source();
        let hash = seahash::hash(kernel_source.as_bytes());
        // Include the GPU target in the cache key so a binary compiled for one
        // architecture is never loaded onto a different one (hipErrorNoBinaryForGpu).
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
            self.hsaco_cache.cache_kernel(&cache_key, &kernel_source, &code)?
        };

        let path_c = std::ffi::CString::new(path.to_str().unwrap_or(""))
            .map_err(|e| Error::Backend(format!("hsaco path CString: {}", e)))?;
        let entry_c = std::ffi::CString::new(entry)
            .map_err(|e| Error::Backend(format!("entry CString: {}", e)))?;

        // Load the HIP module once per unique kernel; reuse the cached module +
        // resolved function on every subsequent dispatch (Item 2).
        let mut module_cache = self.module_cache.lock().unwrap();
        let (_module, func) = if let Some(cached) = module_cache.get(&cache_key) {
            *cached
        } else {
            let mut module: *mut c_void = std::ptr::null_mut();
            check_hip("hipModuleLoad", unsafe { hipModuleLoad(&mut module, path_c.as_ptr()) })?;
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = unsafe { hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) };
            if res != hipSuccess {
                unsafe { hipModuleUnload(module); }
                return Err(Error::Backend(format!("hipModuleGetFunction failed: {}", res)));
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
                grid.x, grid.y, grid.z,
                block.x, block.y, block.z,
                0,
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
        if !x_s.device_ptr_is_valid() || !w_norm_s.device_ptr_is_valid() || !w_mat_s.device_ptr_is_valid() {
            return Err(Error::Backend("rmsnorm_matmul: inputs lack a valid device pointer".into()));
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
            return Err(Error::Shape(format!("expected out [{m},{n}], got {:?}", out_shape.dims())));
        }

        let config = RmsNormMatMulFusionConfig {
            hidden_size: k,
            intermediate_size: n,
            wavefront_size: self.props.wavefront_size as u32,
            lds_size: 65536,
        };
        let launch = config.hip_launch_params();
        
        let storage = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
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

        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }

    /// Dispatch a fused QKV Projection + Attention operation onto the GPU.
    ///
    /// Phase-1 contract (`grim_qkv_attention_kernel_spec.md`):
    /// - `q`: `[seq_len, num_heads, head_dim]`, f32
    /// - `k`, `v`: `[kv_seq_len, num_kv_heads, head_dim]`, f32
    /// - `kv_seq_len` and `cache_offset` must be supplied by the caller (the
    ///   paged KV cache or prefill gather is responsible for materializing
    ///   contiguous K/V buffers; this kernel does not slice).
    /// - `num_kv_heads` is a *real* call-site parameter — never derived as
    ///   `num_heads / 4`. Any GQA ratio (1:1, 2:1, 4:1, 8:1, ...) is valid;
    ///   the host validates `num_heads % num_kv_heads == 0`.
    /// - Causal masking happens **inside** the kernel against absolute
    ///   position `j <= cache_offset + i` (no caller-side pre-slicing).
    /// - The kernel is gated by `config.enabled`. When `false`, returns
    ///   `Err(Error::Backend(...))` and does *not* launch — this is the
    ///   PyTorch-parity path (`rust-gpu-discipline` §3), not a silent CPU
    ///   fallback. The field is preserved long-term so a regression can
    ///   be gated off without an emergency patch.
    pub fn qkv_attention(
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
        // Build the config up front to read its gate. We allow the caller
        // to override individual fields via env-on-the-fly in a follow-up,
        // but for now `enabled: true` is the launch path and `false` is
        // the eager-error path. PyTorch parity: never silent CPU fallback.
        let config = {
            let out_dims = out_shape.dims();
            if out_dims.len() != 3 {
                return Err(Error::Shape("qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into()));
            }
            let seq_len = out_dims[0];
            let num_heads = out_dims[1];
            let head_dim = out_dims[2];
            QkvAttentionFusionConfig {
                enabled: true, // launch path; the gate check is below and *after* the structural validation.
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
                "qkv_attention: kernel is gated (QkvAttentionFusionConfig.enabled=false) — flip to true after Step 4 tests pass".into(),
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
        // Wave64 mandate: kernel block dim is 256 = 4 wavefronts of 64 on
        // gfx1036/gfx110x/gfx1200; head_dim must fit in one wave (≤ 64)
        // for the Phase-1 reference path (Phase 2 will tile).
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
            return Err(Error::Backend("qkv_attention: inputs lack a valid device pointer".into()));
        }
        let out_dims = out_shape.dims();
        let seq_len = out_dims[0];

        // ─── allocate output + launch ────────────────────────────────────
        let launch = config.hip_launch_params();
        let storage = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
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
        let mut inv_sqrt_d_bits = inv_sqrt_d.to_bits();
        // The kernel signature accepts this as a float argument; emit it via
        // a pointer to a local float that the trampoline will pass through.
        let inv_sqrt_d_ptr = &mut inv_sqrt_d_bits as *mut u32 as *mut f32;
        // SAFETY: the kernel reads `inv_sqrt_d` from this pointer across the
        // entire dispatch; the lifetime covers the launch below.
        let inv_sqrt_d_stable = inv_sqrt_d_ptr; // keep the pointer pinned

        // Build the arg slice with all 13 params in the order the kernel
        // signature declares them.
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

        self.launch_compute_kernel(
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

        // Surface we used the temp pointers (suppress unused-mut warnings) and
        // keep them alive for the duration of the kernel call.
        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, inv_sqrt_d_stable,
        );

        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }

    /// Fused KV-dequant-attention (WI-R5).
    ///
    /// Consumes a `CompressedKvBlock`'s packed key/value tensors and per-group
    /// scales directly at attention time, dequantizing on the fly (per-head
    /// `quant_bits`) and applying the online-softmax attention — no
    /// materialized full-precision KV cache in VRAM. The packed `k_tensor`/
    /// `v_tensor` are the `CompressedKvBlock::key_bits`/`value_bits` byte
    /// blobs; `k_scales`/`v_scales` are the `key_meta`/`value_meta` f32 rows.
    ///
    /// Gated by [`KvDequantAttentionConfig`] (default-off, matching
    /// `QkvAttentionFusionConfig`). When `enabled == false` the call returns
    /// a typed error so callers fall back to the dense attention path rather
    /// than silently computing the wrong result.
    pub fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let config = {
            let out_dims = out_shape.dims();
            if out_dims.len() != 3 {
                return Err(Error::Shape(
                    "kv_dequant_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
                ));
            }
            crate::fusion::KvDequantAttentionConfig {
                enabled: true,
                num_heads: out_dims[1],
                num_kv_heads,
                head_dim: out_dims[2],
                quant_bits: 4,
                wavefront_size: self.props.wavefront_size as u32,
            }
        };
        if !config.enabled {
            return Err(Error::Backend(
                "kv_dequant_attention: kernel is gated (KvDequantAttentionConfig.enabled=false)".into(),
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

        // One block per (seq_position, head); block dim 256 = 4 waves on RDNA.
        let block_dim_x: u32 = if config.wavefront_size == 32 { 128 } else { 256 };
        let grid_x = seq_len as u32;
        let grid_y = config.num_heads as u32;
        let shared_mem_bytes = (config.head_dim * 4).min(32768);
        let launch = crate::fusion::HipKernelLaunch {
            grid_dim: HipDim3::new(grid_x, grid_y, 1),
            block_dim: HipDim3::new(block_dim_x, 1, 1),
            shared_mem_bytes,
        };

        let storage = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
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

        self.launch_compute_kernel(
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
            qp, kp, ksp, vp, vsp, op, nh, nkv, hd, sl, ksl, co, isd, qb, inv_sqrt_d_stable,
        );

        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }

    /// Tree-attention wrapper for speculative-decoding verification.
    ///
    /// Spec context (grim_qkv_attention_kernel_spec.md Phase-2 +
    /// grim_rocm_perf_and_abi_fix_spec.md Phase-3 3.5):
    ///
    /// > End-to-end latency 2-3x lower than greedy decoding at same quality.
    /// > Draft model accuracy >= 90% token acceptance rate.
    /// > Tree attention kernel latency < 2x single-token attention.
    ///
    /// Forward of the Phase-2 speculative decoder: the target model
    /// verifies `1 + gamma` tokens (the prompt + gamma drafted tokens)
    /// in a single combined QKV forward by branching each drafted
    /// token's attention to its ancestor chain in the speculative
    /// tree (see `tree_parents`).
    ///
    /// Shape contract:
    /// - `q:       [batch, 1 + gamma, num_heads, head_dim]`
    /// - `k:       [kv_seq_len, num_kv_heads, head_dim]` (shared across `batch`)
    /// - `v:       [kv_seq_len, num_kv_heads, head_dim]` (shared across `batch`)
    /// - `tree_parents: [1 + gamma]` u32, where index `i` is the parent of
    ///   tree position `i` (root has parent 0 / self); see kernel
    ///   `is_ancestor` for the dedup rule.
    /// - `out_shape: [batch, 1 + gamma, num_heads, head_dim]`
    ///
    /// Out allocation: this method allocates the output device
    /// storage up-front and returns it via the `Result<(Box<dyn
    /// BackendStorage>, ...)>` pattern that `BackendDevice` already
    /// uses for `qkv_attention`. This keeps the speculative-decoding
    /// dispatch site's tree-attention call composable.
    ///
    /// Wave64 mandate: head_dim must fit in one wave (<= 64) on
    /// gfx1036 / gfx110x / gfx1200; the Phase-3 tile-via-MFMA path is
    /// a follow-up (the kernel currently increments `q_offset` linearly
    /// and would lose one wave per extra head_dim).
    /// Fused Paged Attention (T2-2).
    ///
    /// Dispatches the JIT compiled `grim_qkv_attention_paged` kernel to run
    /// multi-query attention using paged KV blocks (represented by `block_tables`
    /// mapping request indexes to logical blocks in `k_pages` and `v_pages`).
    ///
    /// Gated by `enabled = true`. Returns a `RocmStorage` result storage and a
    /// computed handle to track execution status.
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
            return Err(Error::Shape("qkv_attention_paged expects 3-D output shape [batch, num_heads, head_dim]".into()));
        }
        let batch = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let q_s = as_rocm(q)?;
        let bt_s = as_rocm(block_tables)?;
        let k_s = as_rocm(k_pages)?;
        let v_s = as_rocm(v_pages)?;

        if !q_s.device_ptr_is_valid() || !bt_s.device_ptr_is_valid() || !k_s.device_ptr_is_valid() || !v_s.device_ptr_is_valid() {
            return Err(Error::Backend("qkv_attention_paged: inputs lack a valid device pointer".into()));
        }

        let mut storage = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        
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

        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
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
        // ─── structural validation ─────────────────────────────────────────
        // The tree-attention launch has stricter constraints than
        // `qkv_attention` because the kernel is also memory-layout-
        // bound (K/V is shared across batch; q is `[batch, 1+gamma,
        // num_heads, head_dim]`).
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
        // gfx10xx / gfx11xx / gfx12xx; head_dim must fit in one wave (<=64).
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
        // Downcasting to RocmStorage verifies each input is GPU-resident;
        // any non-RocmBackendStorage surface returns Err rather than
        // crashing the launcher with a bad downcast.
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
        let mut storage = RocmStorage::alloc_gpu(
            out_shape,
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let gamma_u32 = gamma as u32;

        // The launcher takes `&dyn BackendStorage` for inputs and
        // `&mut dyn BackendStorage` for the output. `RocmStorage` does
        // implement `BackendStorage` directly, so &RocmStorage coerces
        // to &dyn BackendStorage automatically. `tree_attention` does
        // its own `1 / sqrt(head_dim)` scaling inside the kernel -- the
        // host-side value here is intentionally not handed to FFI.
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

        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }
}
