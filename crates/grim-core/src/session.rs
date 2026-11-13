//! `Session` - per-request mutable execution state.
//! A trait object so libraries can box user-supplied sessions, and a concrete `Inner` impl that.

use grim_tensor::{Device, Tensor};

use crate::error::{Error, Result};
use crate::kv_cache::KvCache;
use crate::rng::SimpleRng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeterminismMode {
    Relaxed,
    Strict,
}

/// Model-installed recurrent-state hooks (T2 hybrid prefix reuse).
///
/// The engine's session is the model-agnostic [`Inner`]; a model with
/// recurrent state installs these hooks on first forward so the engine's
/// `SessionT::snapshot_recurrent_state` / `restore_recurrent_state_ref` calls
/// dispatch to model-specific logic without needing a model-specific session
/// type. Shared as `Arc` so `Inner` can hold the hooks while mutably borrowing
/// the model state they operate on.
pub trait RecurrentHooks: Send + Sync {
    /// Capture the recurrent state from the session's model-state payload at
    /// `current_pos`. `Ok(None)` when the payload carries no recurrent state.
    fn snapshot_recurrent(
        &self,
        state: &(dyn std::any::Any + Send),
        current_pos: usize,
    ) -> Result<Option<Box<dyn std::any::Any + Send>>>;

    /// Restore a snapshot into the model-state payload. Returns the session
    /// position the snapshot was taken at.
    fn restore_recurrent(
        &self,
        state: &mut (dyn std::any::Any + Send),
        snap: &dyn std::any::Any,
    ) -> Result<usize>;
}

/// Object-safe session interface used by `Model` traits (`CausalLm`, `EncoderDecoderLm`).
/// The simplest implementation is the `Inner` concrete value returned from `Session::new_storage`.
pub trait SessionT: Send {
    fn device(&self) -> &Device;
    fn current_pos(&self) -> usize;
    fn advance_pos(&mut self, by: usize);
    fn has_kv(&self) -> bool;
    fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()>;
    /// Append K/V for one transformer layer into the session's paged store.
    fn append_kv_layer(&mut self, _layer: usize, _k: &Tensor, _v: &Tensor) -> Result<()> {
        Ok(())
    }
    /// Stage a radix-seeded prefix into `layer`'s paged pages without appending
    /// a token (T2 KV bridge). See [`KvCache::hydrate_seeded_prefix`].
    fn hydrate_seeded_prefix(&mut self, _layer: usize, _stride: usize) {}
    fn kv_cache(&self) -> Option<&(dyn KvCache + 'static)> {
        None
    }
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)> {
        None
    }
    /// True when the session's KV cache is a paged store usable by the
    /// paged-attention kernel.
    fn has_paged_kv(&self) -> bool {
        false
    }
    /// Return the assigned block table for paged attention execution, if active.
    fn block_table(&self) -> Option<&[u32]> {
        None
    }
    /// Return the physical K/V page tensors for `layer` plus the page size,
    /// if a paged store is active.
    fn paged_kv_handles(&self, _layer: usize) -> Option<(Tensor, Tensor, usize)> {
        None
    }
    /// Device-resident block table in the `grim_qkv_attention_paged` kernel's `BlockTableEntry { block_id, page_size }` ABI, cached across decode steps.
    /// Perf (WI-kv): uploading the table per layer per token was a redundant H2D per forward;.
    fn block_table_gpu_handle(
        &self,
    ) -> Option<std::sync::Arc<dyn grim_tensor::backend::BackendStorage>> {
        None
    }
    fn rollback_kv_to(&mut self, len: usize) -> Result<()>;
    // Graph capture / replay hooks for §4.1 ROCm execution optimization
    fn get_hip_graph_handle(&self) -> Option<u64> {
        None
    }
    fn set_hip_graph_handle(&mut self, _handle: u64) {}
    /// Eager escape hatch for interactive validation (§4.3)
    fn eval_eager(&mut self, _op: &str, _inputs: &[&Tensor]) -> Result<Tensor> {
        Err(crate::error::Error::Unimplemented(
            "eval_eager: eager evaluation is not implemented".into(),
        ))
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

    /// Install the model's recurrent-state hooks (T2). See [`RecurrentHooks`].
    fn set_recurrent_hooks(&mut self, _hooks: std::sync::Arc<dyn RecurrentHooks>) {}
    /// The installed recurrent-state hooks, if any.
    fn recurrent_hooks(&self) -> Option<std::sync::Arc<dyn RecurrentHooks>> {
        None
    }

    // NOTE: a trait method returning two mutable borrows cannot be called on
    // `&mut dyn SessionT` — the `&mut (dyn KvCache + 'static)` return makes the
    // compiler require `Self: 'static`, which `dyn SessionT + '_` does not
    // satisfy. Models that need both (the T2 KV bridge) take the owned tensors
    // out of the model state first, then borrow the KV. See `Qwen35::forward`'s
    // post-loop append.

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

    /// Hint to skip output LM head projection (useful for intermediate prefill chunks).
    fn skip_output_projection(&self) -> bool {
        false
    }
    fn set_skip_output_projection(&mut self, _skip: bool) {}

    /// Return full per-position logits for multi-token forwards (e.g. speculative verify).
    /// Default false preserves decode/prefill slicing optimizations.
    fn full_output_logits(&self) -> bool {
        false
    }
    fn set_full_output_logits(&mut self, _full: bool) {}

    /// Greedy/argmax-only sampling (temperature <= 0). Speculative verification should use
    /// argmax-equality in greedy mode for bit-identical MTP streams.
    fn sampling_greedy(&self) -> bool {
        false
    }
    fn set_sampling_greedy(&mut self, _greedy: bool) {}

    /// M2 (speculative verify): snapshot model/recurrent state for rollback.
    /// Returns an opaque snapshot owned by the caller; `None` if no recurrent state.
    fn snapshot_recurrent_state(&self) -> Result<Option<Box<dyn std::any::Any + Send>>> {
        Ok(None)
    }

    /// M2 (speculative verify): restore a snapshot previously returned by
    /// `snapshot_recurrent_state`.
    fn restore_recurrent_state(&mut self, _snapshot: Box<dyn std::any::Any + Send>) -> Result<()> {
        Ok(())
    }

    /// T2 (hybrid prefix reuse): restore from a snapshot the caller still owns.
    ///
    /// The radix checkpoint pool shares one snapshot among every request that
    /// matches the prefix, so it cannot hand ownership to each of them. This
    /// variant lets the session clone the concrete snapshot out of the shared
    /// reference. The default no-op keeps existing implementors source-compatible;
    /// a session that does not override it simply cannot be reused via T2.
    fn restore_recurrent_state_ref(&mut self, _snapshot: &dyn std::any::Any) -> Result<()> {
        Ok(())
    }
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
    pub model_state: Option<Box<dyn std::any::Any + Send>>,
    /// Per-request RNG for deterministic sampling (e.g., speculative rejection).
    pub request_rng: Option<SimpleRng>,
    /// Last speculative accept count from decode_one.
    pub last_accepted_tokens: usize,
    /// Hint to skip output LM head projection on non-terminal prefill chunks.
    pub skip_output_projection: bool,
    /// Return full per-position logits (e.g. speculative verify). Default false.
    pub full_output_logits: bool,
    /// Greedy/argmax-only sampling mode (temperature <= 0).
    pub sampling_greedy: bool,
    /// Model-installed recurrent-state hooks (T2 hybrid prefix reuse). `None`
    /// means the session has no recurrent state to snapshot or restore.
    pub recurrent_hooks: Option<std::sync::Arc<dyn RecurrentHooks>>,
}

impl Inner {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            kv: None,
            current_pos: 0,
            hip_graph_handle: None,
            last_hidden_state: None,
            model_state: None,
            request_rng: None,
            last_accepted_tokens: 1,
            skip_output_projection: false,
            full_output_logits: false,
            sampling_greedy: false,
            recurrent_hooks: None,
        }
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Self {
        Self {
            device,
            kv: Some(kv),
            current_pos: 0,
            hip_graph_handle: None,
            last_hidden_state: None,
            model_state: None,
            request_rng: None,
            last_accepted_tokens: 1,
            skip_output_projection: false,
            full_output_logits: false,
            sampling_greedy: false,
            recurrent_hooks: None,
        }
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
    fn append_kv_layer(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        if let Some(kv) = self.kv.as_deref_mut() {
            kv.append_kv_layer(layer, k, v)?;
        }
        Ok(())
    }
    fn hydrate_seeded_prefix(&mut self, layer: usize, stride: usize) {
        if let Some(kv) = self.kv.as_deref_mut() {
            kv.hydrate_seeded_prefix(layer, stride);
        }
    }
    fn kv_cache(&self) -> Option<&(dyn KvCache + 'static)> {
        self.kv.as_deref()
    }
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)> {
        self.kv.as_deref_mut()
    }
    fn has_paged_kv(&self) -> bool {
        self.kv.as_deref().is_some_and(|kv| kv.has_paged_kv())
    }
    fn block_table(&self) -> Option<&[u32]> {
        self.kv.as_deref().and_then(|kv| kv.block_table())
    }
    fn paged_kv_handles(&self, layer: usize) -> Option<(Tensor, Tensor, usize)> {
        self.kv.as_deref().and_then(|kv| kv.paged_kv_handles(layer))
    }
    fn block_table_gpu_handle(
        &self,
    ) -> Option<std::sync::Arc<dyn grim_tensor::backend::BackendStorage>> {
        self.kv
            .as_deref()
            .and_then(|kv| kv.block_table_gpu_handle())
    }
    fn rollback_kv_to(&mut self, len: usize) -> Result<()> {
        if let Some(kv) = self.kv.as_deref_mut() {
            kv.rollback_to(len)
        } else {
            Ok(())
        }
    }
    fn get_hip_graph_handle(&self) -> Option<u64> {
        self.hip_graph_handle
    }
    fn set_hip_graph_handle(&mut self, handle: u64) {
        self.hip_graph_handle = Some(handle);
    }
    /// Eager escape hatch for interactive validation (§4.3)
    fn eval_eager(&mut self, _op: &str, _inputs: &[&Tensor]) -> Result<Tensor> {
        Err(crate::error::Error::Unimplemented(
            "eval_eager: eager evaluation is not implemented".into(),
        ))
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
    fn skip_output_projection(&self) -> bool {
        self.skip_output_projection
    }
    fn set_skip_output_projection(&mut self, skip: bool) {
        self.skip_output_projection = skip;
    }
    fn full_output_logits(&self) -> bool {
        self.full_output_logits
    }
    fn set_full_output_logits(&mut self, full: bool) {
        self.full_output_logits = full;
    }
    fn sampling_greedy(&self) -> bool {
        self.sampling_greedy
    }
    fn set_sampling_greedy(&mut self, greedy: bool) {
        self.sampling_greedy = greedy;
    }

    fn set_recurrent_hooks(&mut self, hooks: std::sync::Arc<dyn RecurrentHooks>) {
        self.recurrent_hooks = Some(hooks);
    }
    fn recurrent_hooks(&self) -> Option<std::sync::Arc<dyn RecurrentHooks>> {
        self.recurrent_hooks.clone()
    }

    fn snapshot_recurrent_state(&self) -> Result<Option<Box<dyn std::any::Any + Send>>> {
        // `Inner` is model-agnostic; the model installs [`RecurrentHooks`] on
        // first forward. No hooks (or no model state yet) means no recurrent
        // state to snapshot.
        match (self.recurrent_hooks.as_ref(), self.model_state.as_deref()) {
            (Some(hooks), Some(state)) => hooks.snapshot_recurrent(state, self.current_pos),
            _ => Ok(None),
        }
    }

    fn restore_recurrent_state(&mut self, snapshot: Box<dyn std::any::Any + Send>) -> Result<()> {
        self.restore_recurrent_state_ref(&*snapshot)
    }

    fn restore_recurrent_state_ref(&mut self, snapshot: &dyn std::any::Any) -> Result<()> {
        // Clone the Arc so the hooks are not borrowed while the model state is
        // mutably borrowed.
        let Some(hooks) = self.recurrent_hooks.clone() else {
            return Ok(());
        };
        let Some(state) = self.model_state.as_deref_mut() else {
            return Ok(());
        };
        let pos = hooks.restore_recurrent(state, snapshot)?;
        self.current_pos = pos;
        if let Some(kv) = self.kv.as_deref_mut() {
            let _ = kv.rollback_to(pos);
        }
        Ok(())
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

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Replays the captured computation graph using bound session inputs.
    /// NOTE: Graph replay is not yet implemented (sims.md issue #10).
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
    pub fn with_kv_mut<R>(
        &mut self,
        f: &mut dyn FnMut(&mut dyn KvCache) -> Result<R>,
    ) -> Result<Option<R>> {
        if let Some(kv) = self.kv.as_deref_mut() {
            Ok(Some(f(kv)?))
        } else {
            Ok(None)
        }
    }
}

/// Public alias used everywhere `Session` is named as a concrete type.
/// The `Session::new` name is kept stable for call sites even though the runtime handle lives.
pub struct Session;

/// RAII guard that temporarily enables `skip_output_projection` on a `SessionT`,
/// restoring the prior setting upon Drop.
pub struct SkipOutputProjectionGuard<'a> {
    session: &'a mut dyn SessionT,
    prev_state: bool,
}

impl<'a> SkipOutputProjectionGuard<'a> {
    pub fn new(session: &'a mut dyn SessionT, skip: bool) -> Self {
        let prev_state = session.skip_output_projection();
        session.set_skip_output_projection(skip);
        Self {
            session,
            prev_state,
        }
    }

    pub fn current_skip(&self) -> bool {
        self.session.skip_output_projection()
    }
}

impl<'a> Drop for SkipOutputProjectionGuard<'a> {
    fn drop(&mut self) {
        self.session.set_skip_output_projection(self.prev_state);
    }
}

impl Session {
    // Deliberate alias design: `Session` is a stable name and `new` returns the
    // `Inner` handle type on purpose, so silence new_ret_no_self here.
    #[allow(clippy::should_implement_trait)]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(device: Device) -> Inner {
        Inner::new(device)
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Inner {
        Inner::with_kv(device, kv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_output_guard_resets_flag() {
        let mut session = Inner::new(Device::Cpu);
        assert!(!session.skip_output_projection());

        {
            let guard = SkipOutputProjectionGuard::new(&mut session, true);
            assert!(guard.current_skip());
        }
        assert!(!session.skip_output_projection());

        // Test nested or true-to-false restoration
        session.set_skip_output_projection(true);
        {
            let guard = SkipOutputProjectionGuard::new(&mut session, false);
            assert!(!guard.current_skip());
        }
        assert!(session.skip_output_projection());
    }
}
