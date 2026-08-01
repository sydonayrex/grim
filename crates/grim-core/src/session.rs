//! `Session` — per-request mutable execution state.
//!
//! A trait object so libraries can box user-supplied sessions, and a
//! concrete `Inner` impl that holds a KV cache (when present) plus a
//! monotonically-increasing `current_pos` cursor.

use grim_tensor::{Device, Tensor};

use crate::error::{Error, Result};
use crate::kv_cache::KvCache;
use crate::rng::SimpleRng;

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
    fn rollback_kv_to(&mut self, len: usize);
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

    /// Model-specific per-request state (typed slot).
    fn model_state(&self) -> Option<&(dyn std::any::Any + Send)> {
        None
    }
    fn model_state_mut(&mut self) -> Option<&mut (dyn std::any::Any + Send)> {
        None
    }
    fn set_model_state(&mut self, _state: Box<dyn std::any::Any + Send>) {}
    
    /// Per-request RNG for deterministic sampling (e.g., speculative rejection).
    fn request_rng(&self) -> Option<&SimpleRng> {
        None
    }
    fn request_rng_mut(&mut self) -> Option<&mut SimpleRng> {
        None
    }

    /// Current live GPU utilization estimate (0.0–1.0). Used by the
    /// confidence scheduler to pick a dynamic verify length.
    fn live_gpu_utilization(&self) -> f32 {
        0.5
    }

    /// Current batch pressure (queued tokens awaiting compute).
    fn batch_pressure(&self) -> usize {
        0
    }

    /// Last speculative accept count from the most recent `decode_one`.
    /// Default 1 (non-speculative path always accepts exactly one token).
    fn last_accepted_tokens(&self) -> usize {
        1
    }
    fn set_last_accepted_tokens(&mut self, _n: usize) {}
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
    /// Model-specific per-request state (e.g. LFM2 layer caches).
    /// Typed slot — each model downcasts to its own cache type.  Lives on
    /// the session so different requests against the same model get
    /// independent caches, matching bebelm-main's ownership model.
    pub model_state: Option<Box<dyn std::any::Any + Send>>,
    /// Per-request RNG for deterministic sampling (e.g., speculative rejection).
    pub request_rng: Option<SimpleRng>,
    /// Last speculative accept count from decode_one.
    pub last_accepted_tokens: usize,
}

impl Inner {
    pub fn new(device: Device) -> Self {
        Self { device, kv: None, current_pos: 0, hip_graph_handle: None, last_hidden_state: None, model_state: None, request_rng: None, last_accepted_tokens: 1 }
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Self {
        Self { device, kv: Some(kv), current_pos: 0, hip_graph_handle: None, last_hidden_state: None, model_state: None, request_rng: None, last_accepted_tokens: 1 }
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
    fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        if let Some(kv) = self.kv.as_deref_mut() {
            kv.append_slot()?;
            kv.store_kv(k, v)?;
        }
        Ok(())
    }
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)> {
        self.kv.as_deref_mut()
    }
    fn rollback_kv_to(&mut self, len: usize) {
        if let Some(kv) = self.kv.as_deref_mut() {
            let _ = kv.rollback_to(len);
        }
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
    fn model_state(&self) -> Option<&(dyn std::any::Any + Send)> {
        self.model_state.as_deref()
    }
    fn model_state_mut(&mut self) -> Option<&mut (dyn std::any::Any + Send)> {
        self.model_state.as_deref_mut()
    }
    fn set_model_state(&mut self, state: Box<dyn std::any::Any + Send>) {
        self.model_state = Some(state);
    }
    fn request_rng(&self) -> Option<&SimpleRng> {
        self.request_rng.as_ref()
    }
    fn request_rng_mut(&mut self) -> Option<&mut SimpleRng> {
        self.request_rng.as_mut()
    }
    fn last_accepted_tokens(&self) -> usize {
        self.last_accepted_tokens
    }
    fn set_last_accepted_tokens(&mut self, n: usize) {
        self.last_accepted_tokens = n;
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
    ///
    /// NOTE: Graph replay is not yet implemented (sims.md issue #10). The
    /// previous implementation printed the node count and returned `Ok(())`,
    /// silently making every shape-specialized computation path a no-op. We now
    /// surface an explicit `Unimplemented` error so callers cannot mistake a
    /// successful return for a successful replay.
    pub fn replay(&self, _session: &mut dyn SessionT) -> Result<()> {
        Err(Error::Unimplemented(format!(
            "Graph::replay: graph replay is not yet implemented ({} nodes captured).              No computation graph nodes were executed.",
            self.nodes.len()
        )))
    }
}

/// Graph builder trait to construct shape-specialized computation paths.
pub trait GraphBuilder {
    /// Constructs a shape-specialized computation graph for the specified model ID and sequence dimensions.
    fn build(&self, model_id: &str, batch_size: usize, seq_len: usize) -> Result<Graph>;
}

impl GraphBuilder for Inner {
    /// Builds a static computation graph representation tuned for the current execution device and shape parameters.
    fn build(&self, model_id: &str, batch_size: usize, seq_len: usize) -> Result<Graph> {
        let mut graph = Graph::new();
        let input_node = GraphNode {
            id: 0,
            op_name: format!("input_tokens[{model_id}]"),
            inputs: vec![],
            output_shape: grim_tensor::Shape::from([batch_size, seq_len]),
        };
        graph.nodes.push(input_node);
        graph.outputs.push(0);
        Ok(graph)
    }
}

impl GraphBuilder for Session {
    /// Convenience static builder delegate for creating computation graphs without holding an active session instance.
    fn build(&self, model_id: &str, batch_size: usize, seq_len: usize) -> Result<Graph> {
        Inner::new(Device::Cpu).build(model_id, batch_size, seq_len)
    }
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
