//! Full decode-step HIP graph capture/replay.
//!
//! Wraps [`GraphCaptureManager`] to capture the *entire* decode step (all
//! transformer layers, not just a single GEMM) as a single HIP graph, then
//! replay it with one `hipGraphLaunch` per decode step. This collapses the
//! per-layer kernel launch overhead into a single driver call.
//!
//! The capture semantics are verified by `tests/test_hip_graph_capture.rs`:
//! preallocate → warmup → capture → multi-step replay with in-place input
//! updates → parity check against eager execution.
//!
//! Keying: graphs are keyed by [`DecodeGraphKey`] which includes the device
//! pointers of the input/output buffers (SPEED-ROC-4). If the caching
//! allocator recycles a buffer to a different address, the key misses and the
//! graph is re-captured instead of replaying against stale memory.
//!
//! Gated by `GRIM_CAPTURE_GRAPH=1` (same flag as the GEMM-level capture).

use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::error::Result;

use crate::graph_capture::{DecodeGraph, DecodeGraphKey, GraphCaptureManager};

/// A high-level wrapper for full decode-step graph capture/replay.
///
/// One instance per device, typically created lazily and stored in the engine's
/// model state. The [`DecodeGraphKey`] is derived from the runtime shape plus
/// the device pointers of the input hidden-state and output logits buffers.
#[derive(Debug)]
pub struct DecodeGraphCapture {
    manager: GraphCaptureManager,
}

impl DecodeGraphCapture {
    /// Create a new capture helper with default capacity (64 cached graphs).
    pub fn new() -> Self {
        Self {
            manager: GraphCaptureManager::with_capacity(64),
        }
    }

    /// Create with explicit cache capacity.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            manager: GraphCaptureManager::with_capacity(max_entries),
        }
    }

    /// Capture-once, replay-many for a full decode step.
    ///
    /// `key` must include the device pointers of all buffers the graph reads
    /// or writes (input hidden state, output logits, KV cache pages, weights).
    /// `record` is the closure that runs the decode step (e.g. `model.decode_one`)
    /// — it is invoked exactly once per cache miss, on the capture stream.
    ///
    /// On cache hit, the graph is replayed directly (the closure is NOT called).
    pub fn get_or_capture<F>(&self, key: DecodeGraphKey, record: F) -> Result<Arc<DecodeGraph>>
    where
        F: FnOnce(*mut c_void) -> Result<()> + Send,
    {
        self.manager.get_or_capture(key, record)
    }

    /// Replay a previously captured decode step graph.
    ///
    /// Returns `Err` if no graph has been captured for `key`. The replay is
    /// asynchronous on the capture stream — callers that need the result must
    /// synchronize the stream themselves.
    pub fn replay(&self, key: DecodeGraphKey) -> Result<()> {
        self.manager.replay(key)
    }

    /// Check whether a graph is cached for `key`.
    pub fn has_graph(&self, key: DecodeGraphKey) -> bool {
        self.manager.has_graph(key)
    }
}

impl Default for DecodeGraphCapture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_graph_capture_default() {
        let capture = DecodeGraphCapture::default();
        // No graphs captured initially
        let key = DecodeGraphKey {
            batch: 1,
            seq_len: 1,
            kv_seq_len: 128,
            head_dim: 64,
            num_heads: 8,
            num_kv_heads: 8,
            fused_dequant: false,
            a_ptr: 0x1000,
            b_ptr: 0x2000,
            out_ptr: 0x3000,
        };
        assert!(!capture.has_graph(key));
    }

    #[test]
    fn test_decode_graph_capture_with_capacity() {
        let capture = DecodeGraphCapture::with_capacity(128);
        let key = DecodeGraphKey {
            batch: 1,
            seq_len: 1,
            kv_seq_len: 256,
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: 8,
            fused_dequant: true,
            a_ptr: 0xABCD,
            b_ptr: 0xBCDE,
            out_ptr: 0xCDEF,
        };
        assert!(!capture.has_graph(key));
    }
}
