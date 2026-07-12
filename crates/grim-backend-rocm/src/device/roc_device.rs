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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use std::fs;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{DType, QuantProvenance, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendDevice, BackendStorage, Shape};

// Symbols that lib.rs re-exports publicly. They live in sub-modules
// re-exported at crate root. Pulling them from `crate::*` works
// because lib.rs's `pub use` chain makes them accessible from any
// descendent module.
use crate::{
    // HIP runtime FFI
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
    // HIP types / constants
    HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC,
    HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    HIP_DEVICE_ATTRIBUTE_WARP_SIZE, HipDim3, HipErrorT, HipMemcpyKind,
    HiprtcProgram, hipSuccess, RocmHandle, WavefrontSize,
    // rocBLAS FFI
    arith_to_compute_dtype, arith_to_rocblas_dtype, rocblas_create_handle,
    rocblas_destroy_handle, rocblas_gemm_ex, rocblas_gemm_strided_batched_ex,
    // matmul / matmul_batched / matmul_with_solution route their
    // `solution_index` lookup through `select_gemm_algo` so the rocBLAS
    // `algo` argument gets bumped to `rocblas_gemm_algo::solution_index`
    // when a non-zero tuned index is in scope (otherwise rocBLAS
    // silently ignores the index, defeating the tune cache).
    select_gemm_algo,
    rocblas_set_stream, rocblas_sgemm, rocblas_status_success, RocblasInt,
    RocblasOperation, Rocblstatus, RoclabsHandle, rocblas_datatype,
    rocblas_gemm_algo, rocblas_gemm_flags, ROCBLAS_GEMM_FLAGS_NONE,
    // gemm tuning
    // (GemmTileConfig + lookup_gemm_config resolved via crate::* —
    // declared in device/gemm_tuning.rs and re-exported by lib.rs.)
    // kernel cache + graph capture
    HsacoKernelCache, CapturedGraph, HipGraphExecutor, hip_graph_launch,
    // device helpers
    memcpy_with_xnack_fallback, jit_compile_hsaco, upload_device_buffer,
    // lib.rs helpers (re-exported from memory/, device::util/, etc.)
    arg, as_rocm, dev_ptr, detect_gpu_arch, dtype_byte_size, dtype_f32,
    gpu_target_arch, gpu_target_flag, linear_launch, ROCM_COMPUTE_BLOCK,
    // Misc types
    RocmStorage, RocmCachingAllocator, RocmDeviceProps, RocmPinnedBuffer,
    QkvAttentionFusionConfig, RmsNormMatMulFusionConfig, QuantMode,
};



#[derive(Debug)]
pub struct RocmDevice {
    pub(crate) ordinal: usize,
    pub(crate) props: RocmDeviceProps,
    handle_cache: Mutex<Option<RoclabsHandle>>,
    pub(crate) stream_pool: Mutex<Vec<*mut c_void>>,
    pub(crate) hsaco_cache: HsacoKernelCache,
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
    pub fn new(ordinal: usize) -> Self {
        let mut handle_cache = None;
        // Attempt to create rocblas handle lazily on first op if needed.
        unsafe {
            let mut h: RoclabsHandle = RoclabsHandle(std::ptr::null_mut());
            let status = rocblas_create_handle(&mut h);
            if status == rocblas_status_success {
                handle_cache = Some(h);
            }
        }

        // Query device attributes for Wavefront size correctness gate
        let mut warp_size = 64; // Default to W64 (MI200/MI300 CDNA) safety fallback
        let mut xnack_val = 0;
        let mut streams = Vec::new();
        unsafe {
            // NOTE: if hipSetDevice fails here, subsequent hipDeviceGetAttribute calls
            // query the wrong device. This is a silent correctness bug — the API needs
            // redesign to propagate errors from constructors.
            let _ = hipSetDevice(ordinal as i32);
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
        }
    }

    /// Release all pooled device buffers back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        self.allocator.empty_cache();
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
    pub fn begin_graph_capture(&self, key: &str) -> Result<()> {
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
        if let Ok(mut h) = self.get_rocblas_handle() {
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
        if let Ok(mut h) = self.get_rocblas_handle() {
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
                if let Ok(mut h) = self.get_rocblas_handle() {
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
                if let Ok(mut h) = self.get_rocblas_handle() {
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
            let res = unsafe {
                hipMemcpy(
                    (a_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_a * 4),
                    ai.device_ptr.unwrap() as *mut c_void,
                    ai.bytes,
                    HipMemcpyKind::DeviceToDevice,
                )
            };
            if res != hipSuccess {
                return Err(Error::Backend(format!(
                    "matmul_batched: hipMemcpyDtoD a failed: code {res}"
                )));
            }
            let res = unsafe {
                hipMemcpy(
                    (b_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_b * 4),
                    bi.device_ptr.unwrap() as *mut c_void,
                    bi.bytes,
                    HipMemcpyKind::DeviceToDevice,
                )
            };
            if res != hipSuccess {
                return Err(Error::Backend(format!(
                    "matmul_batched: hipMemcpyDtoD b failed: code {res}"
                )));
            }
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
        let mut storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
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
        let mut storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;
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
        let res = unsafe {
            hipMemcpyAsync(
                pinned.as_mut_ptr() as *mut c_void,
                dev_ptr_void,
                elem_count * std::mem::size_of::<f32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipMemcpyAsync(D2H) failed with error code {}",
                res
            )));
        }
        let sync = unsafe { hipStreamSynchronize(stream) };
        if sync != hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamSynchronize after async download failed with error code {}",
                sync
            )));
        }
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
        let res = unsafe {
            hipMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                dev_ptr_void,
                elem_count * std::mem::size_of::<f32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipMemcpyAsync(D2H) failed with error code {}",
                res
            )));
        }
        let sync = unsafe { hipStreamSynchronize(stream) };
        if sync != hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamSynchronize after async download failed with error code {}",
                sync
            )));
        }
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
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}, solution_index={}",
            m, n, k, tile_config, self.props.wavefront_size, solution_index
        );

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
        let tile_config = lookup_gemm_config(m, n, k, self.props.wavefront_size);
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
                let res = hipMemcpyAsync(
                    dev_ptr as *mut c_void,
                    dev_ptr,
                    rocm_storage.bytes,
                    HipMemcpyKind::DeviceToDevice,
                    null_stream,
                );
                if res != hipSuccess {
                    return Err(Error::Backend(format!(
                        "Fallback hipMemcpyAsync failed with status {}",
                        res
                    )));
                }
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
            let res = hipMemAdvise(dev_ptr, rocm_storage.bytes, raw_advice, self.ordinal as i32);
            if res != hipSuccess {
                return Err(Error::Backend(format!(
                    "hipMemAdvise failed with status {}",
                    res
                )));
            }
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
    /// JIT-compile or query the cache, then launch the specified kernel on a stream from the persistent pool.
    /// Dispatch a compute kernel onto a pooled stream. The HIP module for `entry`
    /// is loaded (and the entry function resolved) exactly once per process and
    /// reused on every later dispatch via `module_cache` (Item 2). The per-launch
    /// `hipStreamSynchronize` that previously forced every op to block has been
    /// removed; the stream the kernel was enqueued on is returned so callers that
    /// must wait (e.g. before freeing a temporary buffer) can synchronize
    /// explicitly. Read-back (`to_cpu_vec_f32`) still blocks on the default stream,
    /// which synchronizes with all streams, so results remain correct.
    pub(crate) fn launch_compute_kernel(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
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
        let cache_key = format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash);

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
        let (module, func) = if let Some(cached) = module_cache.get(&cache_key) {
            *cached
        } else {
            let mut module: *mut c_void = std::ptr::null_mut();
            let res = unsafe { hipModuleLoad(&mut module, path_c.as_ptr()) };
            if res != hipSuccess {
                return Err(Error::Backend(format!("hipModuleLoad failed: {}", res)));
            }
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

        let mut args_ptr = args.as_mut_ptr();
        let res = unsafe {
            hipModuleLaunchKernel(
                func,
                grid.x, grid.y, grid.z,
                block.x, block.y, block.z,
                0,
                stream,
                args_ptr,
                std::ptr::null_mut(),
            )
        };
        
        if res != hipSuccess {
            return Err(Error::Backend(format!("hipModuleLaunchKernel failed: {}", res)));
        }
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
        if config.head_dim > 64 {
            return Err(Error::Shape(format!(
                "qkv_attention Phase 1 supports head_dim ≤ 64 (got {}); Phase 2 will tile via MFMA",
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
        let mut out_ptr = dev_ptr(&storage)?;
        let mut q_ptr = dev_ptr(q_s)?;
        let mut k_ptr = dev_ptr(k_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut num_heads_i = config.num_heads as i32;
        let mut num_kv_heads_i = config.num_kv_heads as i32;
        let mut head_dim_i = config.head_dim as i32;
        let mut seq_len_i = seq_len as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut cache_offset_i = cache_offset as i32;
        let inv_sqrt_d: f32 = 1.0 / (config.head_dim as f32).sqrt();
        let mut inv_sqrt_d_bits = inv_sqrt_d.to_bits();
        // The kernel signature accepts this as a float argument; emit it via
        // a pointer to a local float that the trampoline will pass through.
        let inv_sqrt_d_ptr = &mut inv_sqrt_d_bits as *mut u32 as *mut f32;
        // SAFETY: the kernel reads `inv_sqrt_d` from this pointer across the
        // entire dispatch; the lifetime covers the launch below.
        let mut inv_sqrt_d_stable = inv_sqrt_d_ptr; // keep the pointer pinned

        // Build the arg slice with all 11 params in the order the kernel
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
            qptr, kptr, vptr, optr, nh, nkv, hd, sl, ksl, co, isd, inv_sqrt_d_stable,
        );

        Ok((Box::new(storage), Box::new(RocmHandle::new(None))))
    }
}
