//! Phase-3 §3.2 - HIP graph capture/replay for the fused decode step.
//! `GraphCaptureManager` captures a closure's recorded stream into a `DecodeGraph`, caches it per-shape key, and replays.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::error::{Error, Result};

use crate::{
    HipErrorT, RocmDevice, hipGraphCreate, hipGraphDestroy, hipGraphExecDestroy,
    hipGraphInstantiate, hipGraphLaunch, hipStreamBeginCapture, hipStreamCreate, hipStreamDestroy,
    hipStreamEndCapture, hipStreamSynchronize, hipSuccess,
};

/// Key for the cached graph: every captured kernel sequence is keyed by the runtime shape of the decoder.
/// See `rocm-quantization-inference` / `rust-ai-ml-inference-guide` for how to materialize per-arch variants off the side-of-the-spec metadata (`target_gfx`).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DecodeGraphKey {
    pub batch: u32,
    pub seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub fused_dequant: bool,
    /// SPEED-ROC-4: a captured HIP graph bakes in the device pointers of the
    /// buffers it was recorded against. Keying replay on those pointers keeps
    /// it correct under the caching allocator: if any buffer is recycled to a
    /// different address the key misses and the graph is re-captured instead
    /// of replaying against stale memory.
    pub a_ptr: usize,
    pub b_ptr: usize,
    pub out_ptr: usize,
}

/// A captured HIP graph plus its instantiated executable.
/// Both handles are reclaimed on `Drop` so the device teardown never leaks graph resources even.
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
            unsafe {
                let _ = hipGraphExecDestroy(self.exec);
            }
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            unsafe {
                let _ = hipGraphDestroy(self.graph);
            }
            self.graph = std::ptr::null_mut();
        }
    }
}

/// Closure type invoked once per cache miss on the capture stream.
pub type CaptureFn = Box<dyn FnOnce(*mut c_void) -> Result<()> + Send>;

#[derive(Debug, Default)]
struct GraphCacheState {
    cache: HashMap<DecodeGraphKey, Arc<DecodeGraph>>,
    lru: Vec<DecodeGraphKey>,
}

/// Cache of captured decode-step graphs, keyed by `DecodeGraphKey`.
#[derive(Debug)]
pub struct GraphCaptureManager {
    /// Owning the stream once and reusing it for every capture avoids
    /// the cost of allocating a fresh stream per capture.
    capture_stream: Mutex<Option<*mut c_void>>,
    state: Mutex<GraphCacheState>,
    /// Cache capacity — bounded by `(shape_cardinality)` in practice
    /// but exposed so callers can tune without surgery.
    pub max_entries: usize,
}

impl GraphCaptureManager {
    /// Bind the manager to a device with default capacity (64).
    pub fn for_device(dev: &RocmDevice) -> Self {
        Self::for_device_with_capacity(dev, 64)
    }

    /// Bind the manager to a device with explicit capacity.
    pub(crate) fn for_device_with_capacity(_dev: &RocmDevice, max_entries: usize) -> Self {
        Self::with_capacity(max_entries)
    }

    /// Create a manager with explicit cache capacity (no device binding).
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            capture_stream: Mutex::new(None),
            state: Mutex::new(GraphCacheState::default()),
            max_entries,
        }
    }

    /// Lazily create the capture stream. If creation fails, returns the error and keeps the cache empty;
    /// callers will get `Err` again on the next call (no silent CPU fallback per `rust-gpu-discipline` §3).
    fn ensure_capture_stream(&self) -> Result<*mut c_void> {
        if let Some(s) = *self
            .capture_stream
            .lock()
            .map_err(|_| Error::Backend("capture_stream mutex poisoned".into()))?
        {
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
        *self
            .capture_stream
            .lock()
            .map_err(|_| Error::Backend("capture_stream mutex poisoned".into()))? = Some(stream);
        Ok(stream)
    }

    /// Capture-once, replay-many read-through cache.
    /// The closure runs **at most once per key**; subsequent calls hand back the same `Arc<DecodeGraph>`.
    pub fn get_or_capture<F>(&self, key: DecodeGraphKey, capture: F) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(*mut c_void) -> Result<()> + Send,
    {
        // Fast path: cache hit.
        if let Ok(mut state) = self.state.lock() {
            if let Some(g) = state.cache.get(&key).cloned() {
                // Track LRU.
                if let Some(pos) = state.lru.iter().position(|k| k == &key) {
                    state.lru.remove(pos);
                }
                state.lru.push(key);
                return Ok(g);
            }
        }

        // Slow path: capture. Snapshot the closure before locking so we
        // don't hold the cache mutex while we run the user's code.
        let stream = self.ensure_capture_stream()?;
        let mut graph: *mut c_void = std::ptr::null_mut();
        let mode = 2_u32; // hipStreamCaptureModeGlobal

        let res: HipErrorT = unsafe { hipGraphCreate(&mut graph, 0) };
        if res != hipSuccess {
            return Err(Error::Backend(format!("hipGraphCreate failed: {}", res)));
        }
        let begin: HipErrorT = unsafe { hipStreamBeginCapture(stream, mode) };
        if begin != hipSuccess {
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
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
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(Error::Backend(format!(
                "hipGraphInstantiate failed: {}",
                inst_res
            )));
        }

        // DecodeGraph holds raw HIP graph handles; access is serialized by the cache mutex,
        // so the Arc is deliberately shared despite the non-Send handle types.
        #[allow(clippy::arc_with_non_send_sync)]
        let g = Arc::new(DecodeGraph { graph, exec });

        // Insert + LRU bookkeeping + optional eviction.
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Backend("graph cache mutex poisoned".into()))?;
        if state.cache.len() >= self.max_entries {
            // Evict the least recently used entry.
            if !state.lru.is_empty() {
                let stale = state.lru.remove(0);
                state.cache.remove(&stale);
            }
        }
        if let Some(pos) = state.lru.iter().position(|k| k == &key) {
            state.lru.remove(pos);
        }
        state.lru.push(key);
        state.cache.insert(key, g.clone());
        Ok(g)
    }

    /// Check whether a graph is cached for `key` (without capturing).
    pub fn has_graph(&self, key: DecodeGraphKey) -> bool {
        self.state
            .lock()
            .map(|state| state.cache.contains_key(&key))
            .unwrap_or(false)
    }

    /// Replay the cached graph for `key`. Returns `Err` if no capture
    /// has yet been recorded for that key.
    pub fn replay(&self, key: DecodeGraphKey) -> Result<()> {
        let exec = self
            .state
            .lock()
            .map_err(|_| Error::Backend("graph cache mutex poisoned".into()))?
            .cache
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

/// Supported fixed decode batch buckets for HIP Graph capture.
/// Reduces kernel launch overhead by compiling static graphs for common batch cardinalities.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DecodeBatchBucket {
    /// Batch 1 (single-sequence decode).
    B1 = 1,
    /// Batch 2.
    B2 = 2,
    /// Batch 4.
    B4 = 4,
    /// Batch 8.
    B8 = 8,
    /// Batch 16.
    B16 = 16,
    /// Batch 32.
    B32 = 32,
}

impl DecodeBatchBucket {
    /// Return the raw integer batch size for this bucket.
    pub const fn batch_size(self) -> usize {
        self as usize
    }

    /// Map a runtime batch size to the smallest enclosing bucket $B \in \{1, 2, 4, 8, 16, 32\}$.
    /// Returns `None` if `batch == 0` or exceeds the maximum supported bucket (32).
    pub fn from_batch_size(batch: usize) -> Option<Self> {
        match batch {
            0 => None,
            1 => Some(Self::B1),
            2 => Some(Self::B2),
            3..=4 => Some(Self::B4),
            5..=8 => Some(Self::B8),
            9..=16 => Some(Self::B16),
            17..=32 => Some(Self::B32),
            _ => None,
        }
    }

    /// Return all supported fixed batch buckets in ascending order.
    pub const fn all_buckets() -> &'static [DecodeBatchBucket] {
        &[
            DecodeBatchBucket::B1,
            DecodeBatchBucket::B2,
            DecodeBatchBucket::B4,
            DecodeBatchBucket::B8,
            DecodeBatchBucket::B16,
            DecodeBatchBucket::B32,
        ]
    }
}

/// Bucket-specialized graph manager for fixed-size autoregressive decode execution.
/// P3: extended with batch capture/replay that owns the `DecodeGraphBuffers` per bucket
/// so device pointers stay stable across replays (required for graph-keyed replay).
#[derive(Debug, Default)]
pub struct DecodeBucketGraphPool {
    buckets: HashMap<DecodeBatchBucket, Arc<DecodeGraph>>,
    /// P3: persistent batch-sized buffers per bucket. Owned here so addresses
    /// stay stable across replays — the graph was captured against these exact
    /// device pointers, and the `DecodeGraphKey` includes them.
    buffers: HashMap<DecodeBatchBucket, DecodeGraphBuffers>,
    /// P3: per-bucket stream used for both capture and replay.
    streams: HashMap<DecodeBatchBucket, *mut c_void>,
}

impl DecodeBucketGraphPool {
    /// Create a new empty decode bucket graph pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a graph executable is captured and instantiated for a bucket.
    pub fn contains_bucket(&self, bucket: DecodeBatchBucket) -> bool {
        self.buckets.contains_key(&bucket)
    }

    /// Retrieve or capture a graph for the specified bucket.
    pub fn get_or_capture<F>(
        &mut self,
        bucket: DecodeBatchBucket,
        manager: &GraphCaptureManager,
        key_template: DecodeGraphKey,
        record_fn: F,
    ) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(*mut c_void) -> Result<()> + Send,
    {
        if let Some(graph) = self.buckets.get(&bucket) {
            return Ok(graph.clone());
        }

        let mut bucket_key = key_template;
        bucket_key.batch = bucket.batch_size() as u32;

        let graph = manager.get_or_capture(bucket_key, record_fn)?;
        self.buckets.insert(bucket, graph.clone());
        Ok(graph)
    }

    /// Launch a bucket graph on a specified HIP stream with zero kernel launch host overhead.
    pub fn launch(&self, bucket: DecodeBatchBucket, stream: *mut c_void) -> Result<()> {
        let graph = self.buckets.get(&bucket).ok_or_else(|| {
            Error::Backend(format!("No captured HIP graph for bucket {:?}", bucket))
        })?;

        let res = unsafe { hipGraphLaunch(graph.exec, stream) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipGraphLaunch failed on bucket {:?}: {}",
                bucket, res
            )));
        }
        Ok(())
    }

    /// P3: Allocate persistent batch-sized `DecodeGraphBuffers` for a bucket.
    /// These are owned by the pool so device pointers stay stable across replays.
    /// `model` provides the weight shapes; `batch` must match `bucket.batch_size()`.
    pub fn allocate_buffers_for_bucket(
        &mut self,
        bucket: DecodeBatchBucket,
        dev: &RocmDevice,
        num_layers: usize,
        hidden_size: usize,
        n_q: usize,
        n_k: usize,
        n_v: usize,
        intermediate_size: usize,
        max_ctx: usize,
        vocab_size: usize,
        num_heads: usize,
    ) -> Result<()> {
        let batch = bucket.batch_size();
        if self.buffers.contains_key(&bucket) {
            return Ok(());
        }
        let buffers = DecodeGraphBuffers::allocate(
            dev,
            num_layers,
            hidden_size,
            n_q,
            n_k,
            n_v,
            intermediate_size,
            max_ctx,
            vocab_size,
            num_heads,
            batch,
        )?;
        self.buffers.insert(bucket, buffers);
        // Allocate a dedicated stream per bucket for capture + replay.
        let mut stream: *mut c_void = std::ptr::null_mut();
        let res: HipErrorT = unsafe { hipStreamCreate(&mut stream) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "DecodeBucketGraphPool: hipStreamCreate for bucket {:?} failed: {}",
                bucket, res
            )));
        }
        self.streams.insert(bucket, stream);
        Ok(())
    }

    /// P3: Get the persistent buffers for a bucket (for input writes / logits reads).
    pub fn get_buffers(&self, bucket: DecodeBatchBucket) -> Option<&DecodeGraphBuffers> {
        self.buffers.get(&bucket)
    }

    /// P3: Get the stream for a bucket.
    pub fn get_stream(&self, bucket: DecodeBatchBucket) -> Option<*mut c_void> {
        self.streams.get(&bucket).copied()
    }

    /// P3: Capture a batch decode graph for a bucket. The closure records the
    /// full model forward (all layers + output) on the bucket's stream. The
    /// `buffers` must already be allocated via `allocate_buffers_for_bucket`.
    ///
    /// `token_ids` is a one-time seed input used during capture to drive the
    /// model through all kernels — the graph records the kernel topology, not
    /// the data, so replay with different token IDs is correct.
    pub fn capture_batch_graph<F>(
        &mut self,
        bucket: DecodeBatchBucket,
        dev: &RocmDevice,
        model_fn: F,
    ) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(&mut DecodeGraph, &RocmDevice) -> Result<()> + Send,
    {
        let stream = self
            .streams
            .get(&bucket)
            .copied()
            .ok_or_else(|| {
                Error::Backend(format!(
                    "No stream for bucket {:?} — call allocate_buffers_for_bucket first",
                    bucket
                ))
            })?;

        let buffers = self
            .buffers
            .get(&bucket)
            .cloned()
            .ok_or_else(|| {
                Error::Backend(format!(
                    "No buffers for bucket {:?} — call allocate_buffers_for_bucket first",
                    bucket
                ))
            })?;

        let mut graph = DecodeGraph::new(dev, buffers, stream);
        // Record the model forward on the bucket's stream.
        model_fn(&mut graph, dev)?;
        // End capture: instantiate the graph.
        graph.end_capture()?;
        let arc = Arc::new(graph);
        self.buckets.insert(bucket, arc.clone());
        Ok(arc)
    }

    /// P3: Replay a batch decode graph. Writes `token_ids` into the bucket's
    /// input buffer, updates `pos_dev`, then launches the graph in one call.
    /// Returns the logits device storage for D2H or GPU sampling.
    pub fn replay_batch(
        &self,
        bucket: DecodeBatchBucket,
        dev: &RocmDevice,
        token_ids: &[u32],
    ) -> Result<&RocmStorage> {
        let graph = self.buckets.get(&bucket).ok_or_else(|| {
            Error::Backend(format!("No captured graph for bucket {:?}", bucket))
        })?;
        let buffers = self.buffers.get(&bucket).ok_or_else(|| {
            Error::Backend(format!("No buffers for bucket {:?}", bucket))
        })?;
        let stream = self.streams.get(&bucket).copied().ok_or_else(|| {
            Error::Backend(format!("No stream for bucket {:?}", bucket))
        })?;

        if token_ids.len() != bucket.batch_size() {
            return Err(Error::Backend(format!(
                "replay_batch: token_ids len {} != bucket batch_size {}",
                token_ids.len(),
                bucket.batch_size()
            )));
        }

        // Write token IDs into layer 0 input.
        write_embeddings_to_buffer_batch(dev, &buffers.layer_input[0], token_ids)?;

        // Update position: pos_dev gets current_pos, then bump after replay.
        buffers
            .write_pos_async(dev, buffers.current_pos, stream)?;

        // Launch the captured graph.
        let res = unsafe { hipGraphLaunch(graph.exec, stream) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipGraphLaunch failed on bucket {:?}: {}",
                bucket, res
            )));
        }

        // Bump position by batch size for next replay.
        let new_pos = buffers.current_pos.wrapping_add(bucket.batch_size() as u32);
        buffers.current_pos = new_pos;

        Ok(&buffers.head_output)
    }
}

impl Drop for GraphCaptureManager {
    fn drop(&mut self) {
        // Tear down the capture stream if any was created.
        if let Ok(mut slot) = self.capture_stream.lock() {
            if let Some(s) = slot.take() {
                if !s.is_null() {
                    unsafe {
                        let _ = hipStreamDestroy(s);
                    }
                }
            }
        }
    }
}

// Legacy HIP graph wrapper (Item 5 first iteration - `hip_graph_launch` + `CapturedGraph` + `HipGraphExecutor`).
// Kept here so lib.rs stays small; the modern `GraphCaptureManager` above is the Phase-3 §3.2 implementation.

/// Thin `hipGraphLaunch` wrapper.
pub fn hip_graph_launch(graph_exec: *mut c_void, stream: *mut c_void) -> HipErrorT {
    unsafe { hipGraphLaunch(graph_exec, stream) }
}

/// A captured HIP graph plus its instantiated executable, owned under a key in `RocmDevice::captured_graphs`.
/// Frees both handles when dropped so a device teardown never leaks graph resources even if.
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
            // If instantiation fails, we destroy the stream here and return - self.stream is never set,.
            let res = hipGraphInstantiate(
                &mut exec,
                self.graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            if res != hipSuccess {
                let _ = hipStreamDestroy(stream);
                return Err(Error::Backend(format!(
                    "hipGraphInstantiate failed: {}",
                    res
                )));
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

        // Pin the device (P1-7 discipline): the graph and its stream were created against `device_ordinal`'s context.
        // If the calling thread's current device differs, hipStreamSynchronize would target the wrong context and either.
        let _guard = crate::device::util::DeviceGuard::set(self.device_ordinal as i32);
        unsafe {
            let res = hipGraphLaunch(exec, stream);
            if res != hipSuccess {
                return Err(Error::Backend(format!("hipGraphLaunch failed: {}", res)));
            }
            let res = hipStreamSynchronize(stream);
            if res != hipSuccess {
                return Err(Error::Backend(format!(
                    "hipStreamSynchronize failed: {}",
                    res
                )));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_batch_bucket_mapping() {
        assert_eq!(DecodeBatchBucket::from_batch_size(0), None);
        assert_eq!(
            DecodeBatchBucket::from_batch_size(1),
            Some(DecodeBatchBucket::B1)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(2),
            Some(DecodeBatchBucket::B2)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(3),
            Some(DecodeBatchBucket::B4)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(4),
            Some(DecodeBatchBucket::B4)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(5),
            Some(DecodeBatchBucket::B8)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(8),
            Some(DecodeBatchBucket::B8)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(12),
            Some(DecodeBatchBucket::B16)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(16),
            Some(DecodeBatchBucket::B16)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(24),
            Some(DecodeBatchBucket::B32)
        );
        assert_eq!(
            DecodeBatchBucket::from_batch_size(32),
            Some(DecodeBatchBucket::B32)
        );
        assert_eq!(DecodeBatchBucket::from_batch_size(64), None);

        assert_eq!(DecodeBatchBucket::B1.batch_size(), 1);
        assert_eq!(DecodeBatchBucket::B2.batch_size(), 2);
        assert_eq!(DecodeBatchBucket::B4.batch_size(), 4);
        assert_eq!(DecodeBatchBucket::B8.batch_size(), 8);
        assert_eq!(DecodeBatchBucket::B16.batch_size(), 16);
        assert_eq!(DecodeBatchBucket::B32.batch_size(), 32);

        let buckets = DecodeBatchBucket::all_buckets();
        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[0], DecodeBatchBucket::B1);
        assert_eq!(buckets[5], DecodeBatchBucket::B32);
    }

    #[test]
    fn test_decode_bucket_graph_pool_init() {
        let pool = DecodeBucketGraphPool::new();
        assert!(!pool.contains_bucket(DecodeBatchBucket::B1));
        assert!(!pool.contains_bucket(DecodeBatchBucket::B8));
    }
}
