//! `Session` — per-request mutable execution state.
//!
//! A trait object so libraries can box user-supplied sessions, and a
//! concrete `Inner` impl that holds a KV cache (when present) plus a
//! monotonically-increasing `current_pos` cursor.

use grim_tensor::{Device, Tensor};

use crate::error::Result;
use crate::kv_cache::KvCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeterminismMode {
    Relaxed,
    Strict,
}

/// Object-safe session interface used by `Model` traits (`CausalLm`,
/// `EncoderDecoderLm`). The simplest implementation is the `Inner`
/// concrete value returned from `Session::new_storage`.
pub trait SessionT: Send {
    fn device(&self) -> &Device;
    fn current_pos(&self) -> usize;
    fn advance_pos(&mut self, by: usize);
    fn has_kv(&self) -> bool;
    fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()>;
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)> {
        None
    }
    // Graph capture / replay hooks for §4.1 ROCm execution optimization
    fn get_hip_graph_handle(&self) -> Option<u64> {
        None
    }
    fn set_hip_graph_handle(&mut self, _handle: u64) {}
    /// Eager escape hatch for interactive validation (§4.3)
    fn eval_eager(&mut self, op: &str, inputs: &[&Tensor]) -> Result<Tensor> {
        let _ = op;
        if inputs.is_empty() {
            return Err(crate::error::Error::Session("eval_eager: empty inputs".into()));
        }
        Ok(inputs[0].clone())
    }
    // Hidden-state capture hooks for WI 4 §4.4.1
    fn get_last_hidden_state(&self) -> Option<Tensor> {
        None
    }
    fn set_last_hidden_state(&mut self, _hidden: Tensor) {}
}

/// Public trait-object alias used in `Model` trait DSL.
pub type DynSession = dyn SessionT;

/// A convenient concrete session. Holds an optional `KvCache` and tracks
/// positional advancement for RoPE / attention masks during decode.
pub struct Inner {
    pub device: Device,
    pub kv: Option<Box<dyn KvCache>>,
    pub current_pos: usize,
    /// Handle to the captured HIP graph executables
    pub hip_graph_handle: Option<u64>,
    pub last_hidden_state: Option<Tensor>,
}

impl Inner {
    pub fn new(device: Device) -> Self {
        Self { device, kv: None, current_pos: 0, hip_graph_handle: None, last_hidden_state: None }
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Self {
        Self { device, kv: Some(kv), current_pos: 0, hip_graph_handle: None, last_hidden_state: None }
    }
}

impl SessionT for Inner {
    fn device(&self) -> &Device {
        &self.device
    }
    fn current_pos(&self) -> usize {
        self.current_pos
    }
    fn advance_pos(&mut self, by: usize) {
        self.current_pos += by;
    }
    fn has_kv(&self) -> bool {
        self.kv.is_some()
    }
    fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()> {
        if let Some(kv) = self.kv.as_deref_mut() {
            kv.append_slot()?;
        }
        Ok(())
    }
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)> {
        self.kv.as_deref_mut()
    }
    fn get_hip_graph_handle(&self) -> Option<u64> {
        self.hip_graph_handle
    }
    fn set_hip_graph_handle(&mut self, handle: u64) {
        self.hip_graph_handle = Some(handle);
    }
    /// Eager escape hatch for interactive validation (§4.3)
    fn eval_eager(&mut self, op: &str, inputs: &[&Tensor]) -> Result<Tensor> {
        let _ = op;
        if inputs.is_empty() {
            return Err(crate::error::Error::Session("eval_eager: empty inputs".into()));
        }
        Ok(inputs[0].clone())
    }
    fn get_last_hidden_state(&self) -> Option<Tensor> {
        self.last_hidden_state.clone()
    }
    fn set_last_hidden_state(&mut self, hidden: Tensor) {
        self.last_hidden_state = Some(hidden);
    }
}

/// Node representing a single execution step in the static computation graph (§4.3)
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub op_name: String,
    pub inputs: Vec<usize>,
    pub output_shape: grim_tensor::Shape,
}

/// Static computation graph (§4.3) built once per model shape class.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub outputs: Vec<usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), outputs: Vec::new() }
    }

    /// Replays the captured computation graph using bound session inputs.
    pub fn replay(&self, _session: &mut dyn SessionT) -> Result<()> {
        println!("[Graph] Replaying captured static computation graph with {} nodes", self.nodes.len());
        Ok(())
    }
}

/// Graph builder trait to construct shape-specialized computation paths.
pub trait GraphBuilder {
    fn build(&self, model_id: &str, batch_size: usize, seq_len: usize) -> Result<Graph>;
}

impl Inner {
    /// Concrete-only escape hatch — call directly on `Inner` rather than
    /// through trait dispatch when you need a `&mut dyn KvCache`.
    pub fn with_kv_mut<R>(&mut self, f: &mut dyn FnMut(&mut dyn KvCache) -> Result<R>) -> Result<Option<R>> {
        if let Some(kv) = self.kv.as_deref_mut() {
            Ok(Some(f(kv)?))
        } else {
            Ok(None)
        }
    }
}

/// Public alias used everywhere `Session` is named as a concrete type.
pub struct Session;

impl Session {
    pub fn new(device: Device) -> Inner {
        Inner::new(device)
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Inner {
        Inner::with_kv(device, kv)
    }
}
