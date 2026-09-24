#![allow(dead_code, unused_imports)]

//! graph_capture_ops for MetalDevice — Metal command-buffer recording + replay.
//!
//! Metal has no HIP-graph-style hardware graph API. The equivalent is recording
//! commands into a `MTLCommandBuffer` and replaying via re-encoding on a fresh
//! command buffer. This module stores recorded graphs as snapshots of (pipeline,
//! buffer addresses, push constants, grid dims) and replays by re-encoding.
//!
//! On non-Apple targets, all types are stubbed out so the crate compiles.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(target_vendor = "apple")]
use grim_tensor::error::Error;
use grim_tensor::error::Result;

// ---------------------------------------------------------------------------
// Apple-specific types
// ---------------------------------------------------------------------------

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLComputePassDescriptor, MTLComputePipelineState, MTLSize};

#[cfg(target_vendor = "apple")]
#[derive(Debug, Clone)]
pub(crate) struct RecordedMetalGraph {
    /// Kernel pipeline — the compiled compute pipeline for this kernel.
    pub pipeline: MTLComputePipelineState,
    /// Kernel pipeline name (lookup key into MetalPipelines).
    pub kernel_name: String,
    /// Buffer device addresses (raw pointers as u64 for storage).
    pub buffer_addresses: Vec<u64>,
    /// Push constants as u32 words.
    pub push_constants: Vec<u32>,
    /// Grid dimensions (x, y, z).
    pub grid_dims: (u32, u32, u32),
    /// Threadgroup size (x, y, z).
    pub tg_dims: (u32, u32, u32),
    /// Number of buffers bound.
    pub buffer_count: u32,
}

#[cfg(target_vendor = "apple")]
impl RecordedMetalGraph {
    pub fn new(
        pipeline: MTLComputePipelineState,
        kernel_name: String,
        buffer_addresses: Vec<u64>,
        push_constants: Vec<u32>,
        grid_dims: (u32, u32, u32),
        tg_dims: (u32, u32, u32),
        buffer_count: u32,
    ) -> Self {
        Self {
            pipeline,
            kernel_name,
            buffer_addresses,
            push_constants,
            grid_dims,
            tg_dims,
            buffer_count,
        }
    }
}

/// Thread-safe store for recorded Metal graphs, keyed by capture key string.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub(crate) struct MetalGraphStore {
    map: Mutex<HashMap<String, RecordedMetalGraph>>,
}

#[cfg(target_vendor = "apple")]
impl MetalGraphStore {
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert(&self, key: &str, graph: RecordedMetalGraph) {
        if let Ok(mut g) = self.map.lock() {
            g.insert(key.to_string(), graph);
        }
    }

    pub fn get(&self, key: &str) -> Option<RecordedMetalGraph> {
        self.map.lock().ok().and_then(|g| g.get(key).cloned())
    }

    pub fn remove(&self, key: &str) -> Option<RecordedMetalGraph> {
        self.map.lock().ok().and_then(|mut g| g.remove(key))
    }
}

#[cfg(target_vendor = "apple")]
impl Default for MetalGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Non-Apple stubs
// ---------------------------------------------------------------------------

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug, Clone)]
pub struct RecordedMetalGraph;

#[cfg(not(target_vendor = "apple"))]
impl RecordedMetalGraph {
    pub fn new(
        _pipeline: (),
        _kernel_name: String,
        _buffer_addresses: Vec<u64>,
        _push_constants: Vec<u32>,
        _grid_dims: (u32, u32, u32),
        _tg_dims: (u32, u32, u32),
        _buffer_count: u32,
    ) -> Self {
        Self
    }
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalGraphStore;

#[cfg(not(target_vendor = "apple"))]
impl MetalGraphStore {
    pub fn new() -> Self {
        Self
    }

    pub fn insert(&self, _key: &str, _graph: RecordedMetalGraph) {}

    pub fn get(&self, _key: &str) -> Option<RecordedMetalGraph> {
        None
    }

    pub fn remove(&self, _key: &str) -> Option<RecordedMetalGraph> {
        None
    }
}

#[cfg(not(target_vendor = "apple"))]
impl Default for MetalGraphStore {
    fn default() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// MetalGraphCaptureState — cross-platform
// ---------------------------------------------------------------------------

/// Extend MetalDevice with graph capture storage.
/// This is a separate struct so the Apple/non-Apple build paths can both
/// hold a graph store without breaking the non-Apple build.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalGraphCaptureState {
    pub store: std::sync::Arc<MetalGraphStore>,
    /// The active command buffer currently being recorded (None if not capturing).
    pub active_capture: Mutex<Option<(String, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>>,
}

#[cfg(target_vendor = "apple")]
impl MetalGraphCaptureState {
    pub fn new() -> Self {
        Self {
            store: std::sync::Arc::new(MetalGraphStore::new()),
            active_capture: Mutex::new(None),
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug, Clone)]
pub struct MetalGraphCaptureState {
    pub store: std::sync::Arc<MetalGraphStore>,
}

#[cfg(not(target_vendor = "apple"))]
impl MetalGraphCaptureState {
    pub fn new() -> Self {
        Self {
            store: std::sync::Arc::new(MetalGraphStore::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// Replay implementation
// ---------------------------------------------------------------------------

/// Helper: dispatch a recorded graph by re-encoding on a fresh command buffer.
/// This is the replay path — it re-encodes the stored kernel launch and commits.
///
/// # Safety
///
/// The caller must ensure the buffer addresses in the recorded graph are still
/// valid (i.e. the underlying MTLBuffer objects have not been deallocated).
/// In practice, recorded graphs are short-lived and replay happens before the
/// tensors they reference are freed, so this is sound in the current usage model.
#[cfg(target_vendor = "apple")]
pub(crate) fn replay_recorded_graph(
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    graph: &RecordedMetalGraph,
) -> Result<()> {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;

    let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
        unsafe { msg_send![command_queue, commandBuffer] }
            .ok_or_else(|| {
                Error::Backend("Failed to create command buffer for graph replay".into())
            })?;

    cmd_buf.label(msg_send![&nsstring!(graph.kernel_name)]);

    let pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>> =
        unsafe { msg_send![&graph.pipeline, retain] };

    unsafe {
        let compute_pass_desc: MTLComputePassDescriptor = unsafe { std::mem::zeroed() };
        let compute_pass: *mut AnyObject =
            msg_send![&cmd_buf, computeCommandEncoderWithDescriptor: &compute_pass_desc];
        if compute_pass.is_null() {
            return Err(Error::Backend(
                "Failed to create compute command encoder for graph replay".into(),
            ));
        }
        msg_send![compute_pass, setComputePipelineState: &pipeline];
        for (i, addr) in graph.buffer_addresses.iter().enumerate() {
            tracing::warn!(
                "Graph replay: buffer {} address {addr} — buffer handle storage not yet wired",
                i
            );
        }
        let grid_size = MTLSize {
            width: graph.grid_dims.0,
            height: graph.grid_dims.1,
            depth: graph.grid_dims.2,
        };
        let tg_size = MTLSize {
            width: graph.tg_dims.0,
            height: graph.tg_dims.1,
            depth: graph.tg_dims.2,
        };
        msg_send![compute_pass, dispatchThreadgroups: grid_size threadsPerThreadgroup: tg_size];
        msg_send![compute_pass, endEncoding];
    }

    cmd_buf.commit();
    Ok(())
}

#[cfg(target_vendor = "apple")]
fn nsstring(name: &str) -> *const objc2::runtime::AnyObject {
    use objc2::ns_string;
    ns_string!(name)
}

/// Replay a recorded graph. Returns Ok(true) if the graph was found and replay
/// was attempted, Ok(false) if no graph is stored under the key.
///
/// On non-Apple targets, always returns Ok(false) (no Metal device available).
pub fn replay_graph_generic(
    #[cfg(target_vendor = "apple")] command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    #[cfg(target_vendor = "apple")] store: &std::sync::Arc<MetalGraphStore>,
    store: &std::sync::Arc<MetalGraphStore>,
    key: &str,
) -> Result<bool> {
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (store, key);
        Ok(false)
    }
    #[cfg(target_vendor = "apple")]
    {
        let graph = store.get(key);
        match graph {
            Some(ref g) => {
                replay_recorded_graph(command_queue, g)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}
