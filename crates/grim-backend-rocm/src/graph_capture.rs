//! Phase-3 §3.2 — HIP graph capture/replay for the fused decode step.
//!
//! `GraphCaptureManager` captures a closure's recorded stream into a
//! `DecodeGraph`, caches it per-shape key, and replays by `DecodegKey`.
//! Reuse hits the cache; the graph is created once per key and reused
//! on every replay, which is the source of the 15-30 µs/token gain the
//! spec calls out.
//!
//! Skill attribution:
//! - `rust-gpu-parallelism` — HIP graph capture API (`hipStreamBeginCapture`,
//!   `EndCapture`, `hipGraphInstantiate`, `hipGraphLaunch`).
//! - `rust-ai-ml-inference-guide` Action 9 — graph capture for the
//!   repeated decode step; invalidate via `hipGraphExecUpdate` on
//!   weight/shape change.
//! - `rocm-profiling-perf` — JIT warm-up guard: capture only after the
//!   underlying kernels are loaded (Item 2 of the original implementation).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::error::{Error, Result};

use crate::{
    hipGraphCreate, hipGraphDestroy, hipGraphExecDestroy,
    hipGraphInstantiate, hipGraphLaunch, hipStreamBeginCapture, hipStreamCreate,
    hipStreamDestroy, hipStreamEndCapture, hipStreamSynchronize, hipSuccess,
    HipErrorT, RocmDevice,
};

/// Key for the cached graph: every captured kernel sequence is keyed by
/// the runtime shape of the decoder. See `rocm-quantization-inference` /
/// `rust-ai-ml-inference-guide` for how to materialize per-arch variants
/// off the side-of-the-spec metadata (`target_gfx`).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DecodeGraphKey {
    pub batch: u32,
    pub seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub fused_dequant: bool,
}

/// A captured HIP graph plus its instantiated executable. Both handles
/// are reclaimed on `Drop` so the device teardown never leaks graph
/// resources even if a key is overwritten.
#[derive(Debug)]
pub struct DecodeGraph {
    graph: *mut c_void,
    exec: *mut c_void,
}

impl DecodeGraph {
    pub fn exec_handle(&self) -> *mut c_void {
        self.exec
    }
    pub fn graph_handle(&self) -> *mut c_void {
        self.graph
    }
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            unsafe { let _ = hipGraphExecDestroy(self.exec); }
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            unsafe { let _ = hipGraphDestroy(self.graph); }
            self.graph = std::ptr::null_mut();
        }
    }
}

/// Closure type invoked once per cache miss on the capture stream.
pub type CaptureFn = Box<dyn FnOnce(*mut c_void) -> Result<()> + Send>;

/// Cache of captured decode-step graphs, keyed by `DecodeGraphKey`.
#[derive(Debug)]
pub struct GraphCaptureManager {
    /// Owning the stream once and reusing it for every capture avoids
    /// the cost of allocating a fresh stream per capture.
    capture_stream: Mutex<Option<*mut c_void>>,
    cache: Mutex<HashMap<DecodeGraphKey, Arc<DecodeGraph>>>,
    /// Last-used order for cache eviction (optional; included for the
    /// hot-path LIFO reuse hint used by the fused-decode scheduler).
    lru: Mutex<Vec<DecodeGraphKey>>,
    /// Cache capacity — bounded by `(shape_cardinality)` in practice
    /// but exposed so callers can tune without surgery.
    pub max_entries: usize,
}

impl GraphCaptureManager {
    /// Bind the manager to a device. No HIP / ROCm call fires here;
    /// the capture stream is lazily allocated on the first
    /// `get_or_capture`. Skill: `rust-ai-ml-inference-guide` Action 9
    /// (graph capture for decode).
    pub fn for_device(_dev: &RocmDevice) -> Self {
        Self {
            capture_stream: Mutex::new(None),
            cache: Mutex::new(HashMap::new()),
            lru: Mutex::new(Vec::new()),
            max_entries: 64,
        }
    }

    /// Lazily create the capture stream. If creation fails, returns the
    /// error and keeps the cache empty; callers will get `Err` again on
    /// the next call (no silent CPU fallback per `rust-gpu-discipline` §3).
    fn ensure_capture_stream(&self) -> Result<*mut c_void> {
        if let Some(s) = *self.capture_stream.lock().map_err(|_| {
            Error::Backend("capture_stream mutex poisoned".into())
        })? {
            return Ok(s);
        }
        let mut stream: *mut c_void = std::ptr::null_mut();
        let res: HipErrorT = unsafe { hipStreamCreate(&mut stream) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "GraphCaptureManager: hipStreamCreate failed: {}",
                res
            )));
        }
        *self.capture_stream.lock().map_err(|_| {
            Error::Backend("capture_stream mutex poisoned".into())
        })? = Some(stream);
        Ok(stream)
    }

    /// Capture-once, replay-many read-through cache. The closure runs
    /// **at most once per key**; subsequent calls hand back the same
    /// `Arc<DecodeGraph>`. After `max_entries` unique keys are captured
    /// without reuse, older entries are evicted (LRU).
    pub fn get_or_capture<F>(
        &self,
        key: DecodeGraphKey,
        capture: F,
    ) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(*mut c_void) -> Result<()> + Send,
    {
        // Fast path: cache hit.
        if let Ok(cache) = self.cache.lock() {
            if let Some(g) = cache.get(&key) {
                // Track LRU.
                if let Ok(mut lru) = self.lru.lock() {
                    if let Some(pos) = lru.iter().position(|k| k == &key) {
                        lru.remove(pos);
                    }
                    lru.push(key);
                }
                return Ok(g.clone());
            }
        }

        // Slow path: capture. Snapshot the closure before locking so we
        // don't hold the cache mutex while we run the user's code.
        let stream = self.ensure_capture_stream()?;
        let mut graph: *mut c_void = std::ptr::null_mut();
        let mode = 2_u32; // hipStreamCaptureModeGlobal

        let res: HipErrorT = unsafe { hipGraphCreate(&mut graph, 0) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipGraphCreate failed: {}",
                res
            )));
        }
        let begin: HipErrorT = unsafe { hipStreamBeginCapture(stream, mode) };
        if begin != hipSuccess {
            unsafe { let _ = hipGraphDestroy(graph); }
            return Err(Error::Backend(format!(
                "hipStreamBeginCapture failed: {}",
                begin
            )));
        }

        let capture_result = capture(stream);

        let end_status: HipErrorT = unsafe { hipStreamEndCapture(stream, &mut graph) };
        if end_status != hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamEndCapture failed: {}",
                end_status
            )));
        }
        capture_result?;

        let mut exec: *mut c_void = std::ptr::null_mut();
        let inst_res: HipErrorT = unsafe {
            hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if inst_res != hipSuccess {
            unsafe { let _ = hipGraphDestroy(graph); }
            return Err(Error::Backend(format!(
                "hipGraphInstantiate failed: {}",
                inst_res
            )));
        }

        let g = Arc::new(DecodeGraph { graph, exec });

        // Insert + LRU bookkeeping + optional eviction.
        let mut cache = self.cache.lock().map_err(|_| {
            Error::Backend("graph cache mutex poisoned".into())
        })?;
        if cache.len() >= self.max_entries {
            // Evict the least recently used entry.
            if let Ok(mut lru) = self.lru.lock() {
                if let Some(stale) = lru.first().copied() {
                    lru.remove(0);
                    cache.remove(&stale);
                }
            }
        }
        if let Ok(mut lru) = self.lru.lock() {
            if let Some(pos) = lru.iter().position(|k| k == &key) {
                lru.remove(pos);
            }
            lru.push(key);
        }
        cache.insert(key, g.clone());
        Ok(g)
    }

    /// Replay the cached graph for `key`. Returns `Err` if no capture
    /// has yet been recorded for that key.
    pub fn replay(&self, key: DecodeGraphKey) -> Result<()> {
        let exec = self
            .cache
            .lock()
            .map_err(|_| Error::Backend("graph cache mutex poisoned".into()))?
            .get(&key)
            .map(|g| g.exec)
            .ok_or_else(|| Error::Backend("no captured graph for key (replay)".into()))?;
        let stream = self.ensure_capture_stream()?;
        let launch_res: HipErrorT = unsafe { hipGraphLaunch(exec, stream) };
        if launch_res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipGraphLaunch failed: {}",
                launch_res
            )));
        }
        Ok(())
    }
}

impl Drop for GraphCaptureManager {
    fn drop(&mut self) {
        // Tear down the capture stream if any was created.
        if let Ok(mut slot) = self.capture_stream.lock() {
            if let Some(s) = slot.take() {
                if !s.is_null() {
                    unsafe { let _ = hipStreamDestroy(s); }
                }
            }
        }
    }
}

// =============================================================================
// Legacy HIP graph wrapper (Item 5 first iteration — `hip_graph_launch` +
// `CapturedGraph` + `HipGraphExecutor`). Kept here so lib.rs stays small;
// the modern `GraphCaptureManager` above is the Phase-3 §3.2 implementation.
// =============================================================================

/// Thin `hipGraphLaunch` wrapper.
pub fn hip_graph_launch(graph_exec: *mut c_void, stream: *mut c_void) -> HipErrorT {
    unsafe { hipGraphLaunch(graph_exec, stream) }
}

/// A captured HIP graph plus its instantiated executable, owned under a key in
/// `RocmDevice::captured_graphs`. Frees both handles when dropped so a device
/// teardown never leaks graph resources even if a key is overwritten.
#[derive(Debug)]
pub struct CapturedGraph {
    pub(crate) graph: *mut c_void,
    pub(crate) exec: *mut c_void,
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            hipGraphExecDestroy(self.exec);
            hipGraphDestroy(self.graph);
        }
    }
}

/// HIP Graph capture and replay for optimized kernel execution.
/// §4.1: Build once, replay many pattern.
pub struct HipGraphExecutor {
    graph: *mut c_void,
    exec: Option<*mut c_void>,
    stream: Option<*mut c_void>,
    #[allow(dead_code)]
    device_ordinal: usize,
}

impl HipGraphExecutor {
    /// Create a new graph executor. The graph is instantiated on the current device.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let mut graph: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = hipGraphCreate(&mut graph, 0);
            if res != hipSuccess {
                return Err(Error::Backend(format!("hipGraphCreate failed: {}", res)));
            }
        }

        Ok(Self {
            graph,
            exec: None,
            stream: None,
            device_ordinal,
        })
    }

    /// Instantiate the graph for replay. Must be called after all nodes are added.
    pub fn instantiate(&mut self) -> Result<()> {
        let mut exec: *mut c_void = std::ptr::null_mut();
        unsafe {
            let mut stream: *mut c_void = std::ptr::null_mut();
            let res = hipStreamCreate(&mut stream);
            if res != hipSuccess {
                return Err(Error::Backend(format!("hipStreamCreate failed: {}", res)));
            }

            // Instantiate graph before taking ownership of the stream.
            // If instantiation fails, we destroy the stream here and return —
            // self.stream is never set, so Drop won't double-destroy.
            let res = hipGraphInstantiate(
                &mut exec,
                self.graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            if res != hipSuccess {
                let _ = hipStreamDestroy(stream);
                return Err(Error::Backend(format!("hipGraphInstantiate failed: {}", res)));
            }

            // Now safe to take ownership — both graph instantiation and stream
            // creation succeeded; if later steps fail, Drop cleans up.
            self.stream = Some(stream);
            self.exec = Some(exec);
        }
        Ok(())
    }

    /// Launch the graph. Safe to call after instantiate().
    pub fn launch(&mut self) -> Result<()> {
        let (stream, exec) = match (self.stream, self.exec) {
            (Some(s), Some(e)) => (s, e),
            _ => return Err(Error::Backend("Graph not instantiated".into())),
        };

        unsafe {
            let res = hipGraphLaunch(exec, stream);
            if res != hipSuccess {
                return Err(Error::Backend(format!("hipGraphLaunch failed: {}", res)));
            }
            let res = hipStreamSynchronize(stream);
            if res != hipSuccess {
                return Err(Error::Backend(format!("hipStreamSynchronize failed: {}", res)));
            }
            Ok(())
        }
    }
}

impl Drop for HipGraphExecutor {
    fn drop(&mut self) {
        unsafe {
            if let Some(exec) = self.exec {
                let _ = hipGraphExecDestroy(exec);
            }
            if let Some(stream) = self.stream {
                let _ = hipStreamDestroy(stream);
            }
            if !self.graph.is_null() {
                let _ = hipGraphDestroy(self.graph);
            }
        }
    }
}
