//! CPU graph capture and zero-allocation decode graph replay executor.

use grim_tensor::error::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Thread-safe operation closure for recorded CPU execution graphs.
pub type CpuGraphOp = Arc<dyn Fn() -> Result<()> + Send + Sync>;

/// Executable recorded CPU computation graph.
#[derive(Clone)]
pub struct CpuCapturedGraph {
    /// Graph key identifier.
    pub key: String,
    /// Recorded operation closure sequence.
    pub ops: Vec<CpuGraphOp>,
}

impl CpuCapturedGraph {
    /// Replay all recorded graph operation closures in sequence.
    pub fn replay(&self) -> Result<()> {
        for op in &self.ops {
            op()?;
        }
        Ok(())
    }
}

/// Manager and registry for CPU decode graph capture and replay.
#[derive(Default)]
pub struct CpuGraphRegistry {
    capture_active: Mutex<Option<String>>,
    recording_ops: Mutex<Vec<CpuGraphOp>>,
    graphs: Mutex<HashMap<String, CpuCapturedGraph>>,
}

impl CpuGraphRegistry {
    /// Create a new empty CPU graph registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin capturing operations under `key`.
    pub fn begin_capture(&self, key: &str) -> Result<()> {
        let mut active = self
            .capture_active
            .lock()
            .map_err(|_| Error::Backend("CpuGraphRegistry capture_active mutex poisoned".into()))?;
        if active.is_some() {
            return Err(Error::Backend(
                "begin_graph_capture: capture session already active".into(),
            ));
        }
        *active = Some(key.to_string());
        let mut ops = self
            .recording_ops
            .lock()
            .map_err(|_| Error::Backend("CpuGraphRegistry recording_ops mutex poisoned".into()))?;
        ops.clear();
        Ok(())
    }

    /// Check if a graph capture session is currently active.
    pub fn is_capturing(&self) -> bool {
        self.capture_active
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Record an operation closure into the current active graph capture session.
    pub fn record_op<F>(&self, op: F)
    where
        F: Fn() -> Result<()> + Send + Sync + 'static,
    {
        if let Ok(mut ops) = self.recording_ops.lock() {
            ops.push(Arc::new(op));
        }
    }

    /// End the graph capture session for `key` and save the recorded graph.
    pub fn end_capture(&self, key: &str) -> Result<()> {
        let mut active = self
            .capture_active
            .lock()
            .map_err(|_| Error::Backend("CpuGraphRegistry capture_active mutex poisoned".into()))?;
        match active.take() {
            Some(k) if k == key => {
                let mut ops = self
                    .recording_ops
                    .lock()
                    .map_err(|_| Error::Backend("CpuGraphRegistry recording_ops mutex poisoned".into()))?;
                let recorded = std::mem::take(&mut *ops);
                let graph = CpuCapturedGraph {
                    key: key.to_string(),
                    ops: recorded,
                };
                let mut graphs = self
                    .graphs
                    .lock()
                    .map_err(|_| Error::Backend("CpuGraphRegistry graphs mutex poisoned".into()))?;
                graphs.insert(key.to_string(), graph);
                Ok(())
            }
            _ => Err(Error::Backend(format!(
                "end_graph_capture: session key mismatch for {key}"
            ))),
        }
    }

    /// Replay the graph recorded under `key`. Returns `Ok(true)` if replayed, `Ok(false)` if key not found.
    pub fn replay(&self, key: &str) -> Result<bool> {
        let graphs = self
            .graphs
            .lock()
            .map_err(|_| Error::Backend("CpuGraphRegistry graphs mutex poisoned".into()))?;
        if let Some(graph) = graphs.get(key) {
            graph.replay()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check whether a graph executable is stored under `key`.
    pub fn has_captured(&self, key: &str) -> bool {
        self.graphs
            .lock()
            .map(|g| g.contains_key(key))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn graph_capture_and_replay_modifies_state() {
        let registry = CpuGraphRegistry::new();
        let counter = Arc::new(AtomicU32::new(0));

        registry.begin_capture("test_layer").expect("begin");
        assert!(registry.is_capturing());

        let c1 = Arc::clone(&counter);
        registry.record_op(move || {
            c1.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        let c2 = Arc::clone(&counter);
        registry.record_op(move || {
            c2.fetch_add(10, Ordering::SeqCst);
            Ok(())
        });

        registry.end_capture("test_layer").expect("end");
        assert!(!registry.is_capturing());

        let replayed = registry.replay("test_layer").expect("replay");
        assert!(replayed);
        assert_eq!(counter.load(Ordering::SeqCst), 11);

        let replayed2 = registry.replay("test_layer").expect("replay2");
        assert!(replayed2);
        assert_eq!(counter.load(Ordering::SeqCst), 22);
    }
}
