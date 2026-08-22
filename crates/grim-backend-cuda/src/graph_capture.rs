//! CUDA graph capture and replay for the fused decode step (F10).
//!
//! GraphCaptureManager captures a closure recorded stream into a
//! DecodeGraph, caches it per-shape key, and replays by DecodeGraphKey.
//! Fixed decode batch buckets ( in {1, 2, 4, 8, 16, 32}$) enable zero-overhead
//! graph launches without re-instantiation across dynamic batch workloads.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::error::{Error, Result};

use crate::{
    CUstream, cudaGraphCreate, cudaGraphDestroy, cudaGraphExecDestroy, cudaGraphInstantiate,
    cudaGraphLaunch, cudaStreamBeginCapture, cudaStreamCreate, cudaStreamDestroy,
    cudaStreamEndCapture, cudaStreamSynchronize, cudaSuccess,
};

/// Key for the cached graph: every captured kernel sequence is keyed by
/// the runtime shape of the decoder.
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

/// A captured CUDA graph plus its instantiated executable.
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
                let _ = cudaGraphExecDestroy(self.exec);
            }
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            unsafe {
                let _ = cudaGraphDestroy(self.graph);
            }
            self.graph = std::ptr::null_mut();
        }
    }
}

pub type CaptureFn = Box<dyn FnOnce(CUstream) -> Result<()> + Send>;

#[derive(Debug, Default)]
struct GraphCacheState {
    cache: HashMap<DecodeGraphKey, Arc<DecodeGraph>>,
    lru: Vec<DecodeGraphKey>,
}

/// Cache of captured decode-step CUDA graphs, keyed by DecodeGraphKey.
#[derive(Debug)]
pub struct GraphCaptureManager {
    capture_stream: Mutex<Option<CUstream>>,
    state: Mutex<GraphCacheState>,
    pub max_entries: usize,
}

impl Default for GraphCaptureManager {
    fn default() -> Self {
        Self::new(64)
    }
}

impl GraphCaptureManager {
    pub fn new(max_entries: usize) -> Self {
        Self {
            capture_stream: Mutex::new(None),
            state: Mutex::new(GraphCacheState::default()),
            max_entries,
        }
    }

    fn ensure_capture_stream(&self) -> Result<CUstream> {
        if let Some(s) = *self
            .capture_stream
            .lock()
            .map_err(|_| Error::Backend("GraphCaptureManager: capture_stream mutex poisoned".into()))?
        {
            return Ok(s);
        }

        let mut stream: CUstream = std::ptr::null_mut();
        let status = unsafe { cudaStreamCreate(&mut stream) };
        if status != cudaSuccess {
            return Err(Error::Backend(format!(
                "GraphCaptureManager: cudaStreamCreate failed ({status})"
            )));
        }

        let mut slot = self
            .capture_stream
            .lock()
            .map_err(|_| Error::Backend("GraphCaptureManager: capture_stream mutex poisoned".into()))?;
        *slot = Some(stream);
        Ok(stream)
    }

    pub fn get_or_capture<F>(&self, key: DecodeGraphKey, record_fn: F) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(CUstream) -> Result<()> + Send,
    {
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::Backend("GraphCaptureManager: state mutex poisoned".into()))?;
            if let Some(hit) = state.cache.get(&key).cloned() {
                if let Some(pos) = state.lru.iter().position(|k| k == &key) {
                    let k = state.lru.remove(pos);
                    state.lru.push(k);
                }
                return Ok(hit);
            }
        }

        let stream = self.ensure_capture_stream()?;

        let status = unsafe { cudaStreamBeginCapture(stream, 0) };
        if status != cudaSuccess {
            return Err(Error::Backend(format!(
                "GraphCaptureManager: cudaStreamBeginCapture failed ({status})"
            )));
        }

        let record_result = record_fn(stream);

        let mut raw_graph: *mut c_void = std::ptr::null_mut();
        let end_status = unsafe { cudaStreamEndCapture(stream, &mut raw_graph) };

        if let Err(e) = record_result {
            if !raw_graph.is_null() {
                unsafe {
                    let _ = cudaGraphDestroy(raw_graph);
                }
            }
            return Err(Error::Backend(format!(
                "GraphCaptureManager: record_fn failed during capture: {e}"
            )));
        }

        if end_status != cudaSuccess || raw_graph.is_null() {
            if !raw_graph.is_null() {
                unsafe {
                    let _ = cudaGraphDestroy(raw_graph);
                }
            }
            return Err(Error::Backend(format!(
                "GraphCaptureManager: cudaStreamEndCapture failed ({end_status})"
            )));
        }

        let mut raw_exec: *mut c_void = std::ptr::null_mut();
        let mut error_node: *mut c_void = std::ptr::null_mut();
        let inst_status = unsafe {
            cudaGraphInstantiate(
                &mut raw_exec,
                raw_graph,
                &mut error_node,
                std::ptr::null_mut(),
                0,
            )
        };

        if inst_status != cudaSuccess || raw_exec.is_null() {
            unsafe {
                let _ = cudaGraphDestroy(raw_graph);
            }
            return Err(Error::Backend(format!(
                "GraphCaptureManager: cudaGraphInstantiate failed ({inst_status})"
            )));
        }

        let graph = Arc::new(DecodeGraph {
            graph: raw_graph,
            exec: raw_exec,
        });

        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::Backend("GraphCaptureManager: state mutex poisoned".into()))?;
            while state.cache.len() >= self.max_entries && !state.lru.is_empty() {
                let evict_key = state.lru.remove(0);
                state.cache.remove(&evict_key);
            }
            state.cache.insert(key, graph.clone());
            state.lru.push(key);
        }

        Ok(graph)
    }

    pub fn replay(&self, key: &DecodeGraphKey, stream: CUstream) -> Result<()> {
        let graph = {
            let state = self
                .state
                .lock()
                .map_err(|_| Error::Backend("GraphCaptureManager: state mutex poisoned".into()))?;
            state
                .cache
                .get(key)
                .cloned()
                .ok_or_else(|| Error::Backend(format!("GraphCaptureManager: key {key:?} not in cache")))?
        };

        let status = unsafe { cudaGraphLaunch(graph.exec, stream) };
        if status != cudaSuccess {
            return Err(Error::Backend(format!(
                "GraphCaptureManager: cudaGraphLaunch failed ({status})"
            )));
        }
        Ok(())
    }
}

impl Drop for GraphCaptureManager {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.capture_stream.lock() {
            if let Some(s) = slot.take() {
                if !s.is_null() {
                    unsafe {
                        let _ = cudaStreamDestroy(s);
                    }
                }
            }
        }
    }
}

/// Fixed decode batch buckets for zero-overhead graph replay.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DecodeBatchBucket {
    B1 = 1,
    B2 = 2,
    B4 = 4,
    B8 = 8,
    B16 = 16,
    B32 = 32,
}

impl DecodeBatchBucket {
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

    pub fn batch_size(self) -> usize {
        self as usize
    }

    pub fn all_buckets() -> &'static [DecodeBatchBucket] {
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

/// Bucket-specialized graph manager for fixed-size autoregressive decode execution on CUDA.
#[derive(Debug, Default)]
pub struct DecodeBucketGraphPool {
    buckets: HashMap<DecodeBatchBucket, Arc<DecodeGraph>>,
}

impl DecodeBucketGraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains_bucket(&self, bucket: DecodeBatchBucket) -> bool {
        self.buckets.contains_key(&bucket)
    }

    pub fn get_or_capture<F>(
        &mut self,
        bucket: DecodeBatchBucket,
        manager: &GraphCaptureManager,
        key_template: DecodeGraphKey,
        record_fn: F,
    ) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(CUstream) -> Result<()> + Send,
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

    pub fn launch(&self, bucket: DecodeBatchBucket, stream: CUstream) -> Result<()> {
        let graph = self
            .buckets
            .get(&bucket)
            .ok_or_else(|| Error::Backend(format!("No captured CUDA graph for bucket {:?}", bucket)))?;

        let res = unsafe { cudaGraphLaunch(graph.exec, stream) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaGraphLaunch failed on bucket {:?}: {}",
                bucket, res
            )));
        }
        Ok(())
    }
}

/// CUDA Graph executor for explicit graph creation and instantiation.
pub struct CudaGraphExecutor {
    graph: *mut c_void,
    exec: Option<*mut c_void>,
    stream: Option<CUstream>,
}

impl Default for CudaGraphExecutor {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            graph: std::ptr::null_mut(),
            exec: None,
            stream: None,
        })
    }
}

impl CudaGraphExecutor {
    pub fn new() -> Result<Self> {
        let mut graph: *mut c_void = std::ptr::null_mut();
        let res = unsafe { cudaGraphCreate(&mut graph, 0) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!("cudaGraphCreate failed: {}", res)));
        }

        Ok(Self {
            graph,
            exec: None,
            stream: None,
        })
    }

    pub fn instantiate(&mut self) -> Result<()> {
        let mut exec: *mut c_void = std::ptr::null_mut();
        let mut stream: CUstream = std::ptr::null_mut();
        let res = unsafe { cudaStreamCreate(&mut stream) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!("cudaStreamCreate failed: {}", res)));
        }

        let res = unsafe {
            cudaGraphInstantiate(
                &mut exec,
                self.graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if res != cudaSuccess {
            unsafe {
                let _ = cudaStreamDestroy(stream);
            }
            return Err(Error::Backend(format!("cudaGraphInstantiate failed: {}", res)));
        }

        self.stream = Some(stream);
        self.exec = Some(exec);
        Ok(())
    }

    pub fn launch(&mut self) -> Result<()> {
        let (stream, exec) = match (self.stream, self.exec) {
            (Some(s), Some(e)) => (s, e),
            _ => return Err(Error::Backend("Graph not instantiated".into())),
        };

        let res = unsafe { cudaGraphLaunch(exec, stream) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!("cudaGraphLaunch failed: {}", res)));
        }
        let res = unsafe { cudaStreamSynchronize(stream) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!("cudaStreamSynchronize failed: {}", res)));
        }
        Ok(())
    }
}

impl Drop for CudaGraphExecutor {
    fn drop(&mut self) {
        if let Some(exec) = self.exec {
            unsafe {
                let _ = cudaGraphExecDestroy(exec);
            }
        }
        if let Some(stream) = self.stream {
            unsafe {
                let _ = cudaStreamDestroy(stream);
            }
        }
        if !self.graph.is_null() {
            unsafe {
                let _ = cudaGraphDestroy(self.graph);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_decode_batch_bucket_mapping() {
        assert_eq!(DecodeBatchBucket::from_batch_size(0), None);
        assert_eq!(DecodeBatchBucket::from_batch_size(1), Some(DecodeBatchBucket::B1));
        assert_eq!(DecodeBatchBucket::from_batch_size(2), Some(DecodeBatchBucket::B2));
        assert_eq!(DecodeBatchBucket::from_batch_size(3), Some(DecodeBatchBucket::B4));
        assert_eq!(DecodeBatchBucket::from_batch_size(4), Some(DecodeBatchBucket::B4));
        assert_eq!(DecodeBatchBucket::from_batch_size(5), Some(DecodeBatchBucket::B8));
        assert_eq!(DecodeBatchBucket::from_batch_size(8), Some(DecodeBatchBucket::B8));
        assert_eq!(DecodeBatchBucket::from_batch_size(12), Some(DecodeBatchBucket::B16));
        assert_eq!(DecodeBatchBucket::from_batch_size(16), Some(DecodeBatchBucket::B16));
        assert_eq!(DecodeBatchBucket::from_batch_size(24), Some(DecodeBatchBucket::B32));
        assert_eq!(DecodeBatchBucket::from_batch_size(32), Some(DecodeBatchBucket::B32));
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
    fn test_cuda_decode_bucket_graph_pool_init() {
        let pool = DecodeBucketGraphPool::new();
        assert!(!pool.contains_bucket(DecodeBatchBucket::B1));
        assert!(!pool.contains_bucket(DecodeBatchBucket::B8));
    }
}
