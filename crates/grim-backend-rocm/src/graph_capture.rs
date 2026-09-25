// Phase-3 §3.2 - HIP graph capture/replay for the fused decode step.
//! `GraphCaptureManager` captures a closure's recorded stream into a `DecodeGraph`, caches it per-shape key, and replays.
//! P3: `DecodeBucketGraphPool` extends this with batch-sized fixed buffers per bucket,
//! capturing one graph per bucket and replaying the whole batch in one `hipGraphLaunch`.

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
    /// who-dat.md P3-12: hit/miss counters so allocator thrash (a transient
    /// buffer moving to a new address between steps) shows up as a falling
    /// hit ratio instead of silently re-capturing graphs.
    hits: u64,
    misses: u64,
}

/// who-dat.md P3-10: decode-graph failure was previously permanent — one
/// transient error (JIT miss, OOM blip, hipModuleLaunchKernel 901 during
/// capture) disabled graph decode for the whole run. `GraphRetryPolicy`
/// allows re-entering capture after `retry_interval` eager steps, up to
/// `max_retries` times per run. Defaults are conservative; both knobs are
/// env-overridable (`GRIM_GRAPH_MAX_RETRIES`, `GRIM_GRAPH_RETRY_INTERVAL`).
#[derive(Debug, Clone)]
pub struct GraphRetryPolicy {
    failures: u32,
    last_fail_step: Option<usize>,
    max_retries: u32,
    retry_interval: usize,
}

impl GraphRetryPolicy {
    pub fn new(max_retries: u32, retry_interval: usize) -> Self {
        Self {
            failures: 0,
            last_fail_step: None,
            max_retries,
            retry_interval,
        }
    }

    /// Default policy: 3 retries, at least 32 eager steps apart.
    pub fn default_policy() -> Self {
        let max_retries = std::env::var("GRIM_GRAPH_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let retry_interval = std::env::var("GRIM_GRAPH_RETRY_INTERVAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        Self::new(max_retries, retry_interval)
    }

    /// Total failures observed so far.
    pub fn failures(&self) -> u32 {
        self.failures
    }

    /// Step of the most recent failure, if any.
    pub fn last_fail_step(&self) -> Option<usize> {
        self.last_fail_step
    }

    /// Record a capture/seed/replay failure at `step`.
    pub fn note_fail(&mut self, step: usize) {
        self.failures = self.failures.saturating_add(1);
        self.last_fail_step = Some(step);
    }

    /// Whether graph decode may be (re)attempted at `step`.
    /// Fresh runs have no failures and always attempt; after a failure the
    /// policy requires `retry_interval` eager steps to pass and caps the
    /// number of retries at `max_retries`.
    pub fn should_attempt(&self, step: usize) -> bool {
        match self.last_fail_step {
            None => true,
            Some(last) => {
                self.failures <= self.max_retries
                    && step.saturating_sub(last) >= self.retry_interval
            }
        }
    }

    /// Clear failure state (e.g. after a successful capture), keeping the
    /// budget configuration.
    pub fn note_success(&mut self) {
        self.failures = 0;
        self.last_fail_step = None;
    }
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
                state.hits = state.hits.saturating_add(1);
                return Ok(g);
            }
            state.misses = state.misses.saturating_add(1);
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
        let (hits, misses) = (state.hits, state.misses);
        drop(state);
        // who-dat.md P3-12: one line per capture showing the running hit
        // ratio — repeated misses on a stable model mean pointer-key thrash.
        let total = hits + misses;
        eprintln!(
            "[grim] graph-capture: key miss (capture #{}); hit ratio {:.1}% ({}/{})",
            misses,
            100.0 * hits as f64 / total.max(1) as f64,
            hits,
            total,
        );
        Ok(g)
    }

    /// who-dat.md P3-12: cumulative (hits, misses) for the graph-key cache.
    pub fn cache_stats(&self) -> (u64, u64) {
        self.state
            .lock()
            .map(|s| (s.hits, s.misses))
            .unwrap_or((0, 0))
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
/// P3: each bucket owns one `FullDecodeGraph` (fixed `DecodeGraphBuffers` at stable
/// device addresses + the captured `hipGraphExec` recorded against them). Replay
/// writes the batch's token IDs + position into the fixed slots and launches once.
#[derive(Debug, Default)]
pub struct DecodeBucketGraphPool {
    graphs: HashMap<DecodeBatchBucket, crate::decode_graph_buffers::DecodeGraph>,
}

impl DecodeBucketGraphPool {
    /// Create a new empty decode bucket graph pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a graph executable is captured and instantiated for a bucket.
    pub fn contains_bucket(&self, bucket: DecodeBatchBucket) -> bool {
        self.graphs.get(&bucket).is_some_and(|g| g.is_captured)
    }

    /// Allocate a batch-sized decode graph (buffers + stream, not yet captured)
    /// for a bucket. Idempotent: existing entries are kept untouched.
    #[allow(clippy::too_many_arguments)]
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
        if self.graphs.contains_key(&bucket) {
            return Ok(());
        }
        use std::ffi::c_void as RawVoid;
        let mut stream: *mut RawVoid = std::ptr::null_mut();
        let res: HipErrorT = unsafe { hipStreamCreate(&mut stream) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "DecodeBucketGraphPool: stream create for bucket {bucket:?} failed: {res}"
            )));
        }
        let graph = crate::decode_graph_buffers::DecodeGraph::allocate(
            dev,
            stream,
            num_layers,
            hidden_size,
            n_q,
            n_k,
            n_v,
            intermediate_size,
            max_ctx,
            vocab_size,
            num_heads,
            bucket.batch_size(),
        )?;
        self.graphs.insert(bucket, graph);
        Ok(())
    }

    /// Get the persistent buffers for a bucket (for input writes / logits reads).
    pub fn get_buffers(
        &self,
        bucket: DecodeBatchBucket,
    ) -> Option<&crate::decode_graph_buffers::DecodeGraphBuffers> {
        self.graphs.get(&bucket).map(|g| &g.buffers)
    }

    /// Get the stream for a bucket.
    pub fn get_stream(&self, bucket: DecodeBatchBucket) -> Option<*mut c_void> {
        self.graphs.get(&bucket).map(|g| g.stream)
    }

    /// Mutable access to a bucket's (not yet captured) graph — used to run
    /// eager warmup forwards through the pool's fixed buffers so JIT kernels
    /// are resolved before `capture_batch_graph` records (module load inside
    /// capture fails with hipModuleLaunchKernel 901).
    pub fn graph_mut(
        &mut self,
        bucket: DecodeBatchBucket,
    ) -> Option<&mut crate::decode_graph_buffers::DecodeGraph> {
        self.graphs.get_mut(&bucket)
    }

    /// Capture the batch decode graph for a bucket. The closure records the
    /// full model forward (all layers + output) against the bucket's fixed
    /// buffers on its stream. Buffers are only stable because the pool owns them.
    ///
    /// The recorded graph binds the exact device pointers of the bucket's
    /// buffers at capture time; replay with different token IDs rewrites only
    /// input slots, never the topology.
    pub fn capture_batch_graph<F>(&mut self, bucket: DecodeBatchBucket, model_fn: F) -> Result<()>
    where
        F: FnOnce(&mut crate::decode_graph_buffers::DecodeGraph) -> Result<()>,
    {
        let graph = self.graphs.get_mut(&bucket).ok_or_else(|| {
            Error::Backend(format!(
                "No buffers for bucket {bucket:?} — call allocate_buffers_for_bucket first"
            ))
        })?;
        graph.begin_capture()?;
        if let Err(e) = model_fn(graph) {
            let _ = graph.abort_capture();
            return Err(e);
        }
        graph.end_capture()?;
        Ok(())
    }

    /// Replay a batch decode graph: writes `token_ids` (len == bucket batch)
    /// into layer-0 input slots, refreshes the device position scalar, then
    /// launches the whole decode step in one `hipGraphLaunch`. Returns the
    /// batch logits storage `[batch, vocab]` for D2H or GPU sampling.
    pub fn replay_batch(
        &mut self,
        bucket: DecodeBatchBucket,
        dev: &RocmDevice,
        token_ids: &[u32],
    ) -> Result<&crate::RocmStorage> {
        let graph = self
            .graphs
            .get_mut(&bucket)
            .ok_or_else(|| Error::Backend(format!("No captured graph for bucket {bucket:?}")))?;
        if token_ids.len() != bucket.batch_size() {
            return Err(Error::Backend(format!(
                "replay_batch: token_ids len {} != bucket batch_size {}",
                token_ids.len(),
                bucket.batch_size()
            )));
        }
        crate::decode_graph_buffers::write_embeddings_to_buffer_batch(
            dev,
            &graph.buffers.token_ids_dev,
            token_ids,
        )?;
        // Refresh the device position scalar with the host mirror; the graph's
        // bump node advances it on device, so the mirror must not double-count.
        let pos = graph.buffers.current_pos;
        graph.buffers.write_pos_async(dev, pos, graph.stream)?;
        graph.replay()?;
        Ok(&graph.buffers.head_output)
    }

    /// P3 replay with per-slot positions: requests in a decode bucket may sit
    /// at different KV offsets; `positions[slot]` is written into
    /// `pos_dev[slot]` before launch. `positions.len()` must equal the bucket
    /// batch size.
    pub fn replay_batch_with_pos(
        &mut self,
        bucket: DecodeBatchBucket,
        dev: &RocmDevice,
        token_ids: &[u32],
        positions: &[u32],
    ) -> Result<&crate::memory::storage::RocmStorage> {
        let graph = self
            .graphs
            .get_mut(&bucket)
            .ok_or_else(|| Error::Backend(format!("No captured graph for bucket {bucket:?}")))?;
        if token_ids.len() != bucket.batch_size() || positions.len() != bucket.batch_size() {
            return Err(Error::Backend(format!(
                "replay_batch_with_pos: token_ids {} / positions {} != batch {}",
                token_ids.len(),
                positions.len(),
                bucket.batch_size()
            )));
        }
        crate::decode_graph_buffers::write_embeddings_to_buffer_batch(
            dev,
            &graph.buffers.token_ids_dev,
            token_ids,
        )?;
        graph
            .buffers
            .write_pos_slots_async(dev, positions, graph.stream)?;
        graph.replay()?;
        Ok(&graph.buffers.head_output)
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
    fn test_graph_retry_policy_fresh_run_always_attempts() {
        let p = GraphRetryPolicy::new(3, 32);
        assert!(p.should_attempt(0));
        assert!(p.should_attempt(1));
        assert_eq!(p.failures(), 0);
        assert_eq!(p.last_fail_step(), None);
    }

    #[test]
    fn test_graph_retry_policy_blocks_until_interval() {
        let mut p = GraphRetryPolicy::new(3, 32);
        p.note_fail(10);
        assert!(!p.should_attempt(10), "same step must not retry");
        assert!(!p.should_attempt(41), "inside retry interval");
        assert!(p.should_attempt(42), "retry allowed after 32 eager steps");
    }

    #[test]
    fn test_graph_retry_policy_respects_max_retries() {
        let mut p = GraphRetryPolicy::new(2, 8);
        p.note_fail(0);
        assert!(p.should_attempt(8));
        p.note_fail(8);
        assert!(p.should_attempt(16));
        p.note_fail(16);
        assert!(
            !p.should_attempt(24),
            "budget exhausted (3 failures > 2 retries)"
        );
        assert!(!p.should_attempt(1000));
    }

    #[test]
    fn test_graph_retry_policy_success_resets_budget() {
        // ponytail: reset on success — a transient blip (e.g. first-run JIT
        // miss) must not burn the whole run's retry budget.
        let mut p = GraphRetryPolicy::new(1, 4);
        p.note_fail(0);
        assert!(p.should_attempt(4));
        p.note_success();
        assert_eq!(p.failures(), 0);
        assert_eq!(p.last_fail_step(), None);
        // Behaves like a fresh run again, and the budget is restored.
        p.note_fail(9);
        assert!(p.should_attempt(13));
    }

    #[test]
    fn test_graph_retry_policy_zero_retries_is_one_shot() {
        // GRIM_GRAPH_MAX_RETRIES=0 semantics: first failure disables forever.
        let mut p = GraphRetryPolicy::new(0, 1);
        p.note_fail(0);
        assert!(!p.should_attempt(1));
        assert!(!p.should_attempt(10_000));
    }

    #[test]
    fn test_graph_cache_stats_start_zero() {
        let mgr = GraphCaptureManager::with_capacity(4);
        assert_eq!(mgr.cache_stats(), (0, 0));
    }

    #[test]
    fn test_graph_cache_state_tracks_hits_and_misses() {
        let mut state = GraphCacheState::default();
        assert_eq!((state.hits, state.misses), (0, 0));
        state.misses = state.misses.saturating_add(1);
        state.hits = state.hits.saturating_add(3);
        assert_eq!((state.hits, state.misses), (3, 1));
    }

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
