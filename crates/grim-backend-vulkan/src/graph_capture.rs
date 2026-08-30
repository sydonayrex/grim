//! Vulkan graph-capture bookkeeping.
//!
//! `VkGraphCache` records captured-graph names so the engine's capture/replay
//! branches execute without hitting `Err(Unimplemented)`. Current state:
//! structural scaffolding that makes the `GraphCaptureOps` API non-failing.
//! True command-buffer replay (recording a `VkCommandBuffer` via
//! `vkBeginCommandBuffer`/`vkEndCommandBuffer` and replaying with
//! `vkQueueSubmit`) requires `VK_EXT_graph_capture`, which is not wired here.
//!
//! Replace `replay()` with real `VkCommandBuffer` recording/replay when the
//! extension lands — the capture/replay win is §4.3 decode throughput.

use std::collections::HashMap;
use std::sync::Mutex;
use grim_tensor::error::{Error, Result};

/// Key for a captured command-buffer graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub name: String,
}

/// A recorded command-buffer graph.
///
/// Currently a placeholder: the real implementation wraps a `VkCommandBuffer`
/// recorded via `vkBeginCommandBuffer`/`vkEndCommandBuffer` and replayed via
/// `vkQueueSubmit`.
pub struct CapturedGraph;

/// Process-wide graph cache. Records capture keys and reports replay hits.
pub struct VkGraphCache {
    graphs: Mutex<HashMap<String, CapturedGraph>>,
}

impl VkGraphCache {
    pub fn new() -> Self {
        Self {
            graphs: Mutex::new(HashMap::new()),
        }
    }

    /// Begin capturing a graph named `key`. Currently a no-op; a real
    /// implementation would start `VkCommandBuffer` recording here.
    pub fn begin(&self, key: &str) -> Result<()> {
        let _ = key;
        Ok(())
    }

    /// End capturing a graph named `key`. Records the key so subsequent
    /// `replay()` calls report a hit.
    pub fn end(&self, key: &str) -> Result<()> {
        self.graphs
            .lock()
            .map_err(|e| Error::Backend(format!("{e}")))?
            .insert(key.to_string(), CapturedGraph);
        Ok(())
    }

    /// Report whether a graph named `key` has been captured. A real
    /// implementation would replay the recorded `VkCommandBuffer` here.
    pub fn replay(&self, key: &str) -> Result<bool> {
        Ok(self
            .graphs
            .lock()
            .map_err(|e| Error::Backend(format!("{e}")))?
            .contains_key(key))
    }

    /// Check whether a graph named `key` has been captured.
    pub fn has(&self, key: &str) -> bool {
        self.graphs.lock().map(|g| g.contains_key(key)).unwrap_or(false)
    }
}
