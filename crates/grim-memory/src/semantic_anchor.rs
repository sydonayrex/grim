//! Semantic-aware state caching for recurrent and hybrid-attention models.
//! Frontier models with hybrid-attention (e.g.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Unique identifier for an allocated recurrent checkpoint.
pub type CheckpointId = usize;

/// Layer-wise recurrent state data payload.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentLayerState {
    /// Layer index in the model.
    pub layer_idx: usize,
    /// Flat float buffer containing the recurrent state (e.g., hidden state / SSM matrix).
    pub state_data: Vec<f32>,
    /// State tensor dimensions (e.g., `[num_heads, head_dim, head_dim]` or `[hidden_dim]`).
    pub shape: Vec<usize>,
}

/// Opaque session snapshot shared between a checkpoint and the engine.
///
/// `Mutex` makes the `Arc` `Sync` (the pool lives behind a `Mutex<KvBlockPool>`,
/// and `Arc<dyn Any + Send>` is not `Sync`). The engine takes a read lock and
/// passes `&dyn Any` to [`SessionT::restore_recurrent_state_ref`], so a checkpoint
/// can be restored by any number of requests without consuming it.
pub type SharedSessionSnapshot = Arc<std::sync::Mutex<Box<dyn std::any::Any + Send>>>;

/// A full-model snapshot of all recurrent and hybrid-attention layers at a semantic boundary.
///
/// `session_snapshot` is the opaque [`SessionT::snapshot_recurrent_state`] payload
/// (device GDN+conv state for Qwen35) captured at `token_offset`. It is `None` for
/// checkpoints registered before the session snapshot was available, and for the
/// synthetic layer-state checkpoints the pool tests build. The engine restores it
/// via [`SessionT::restore_recurrent_state_ref`]; the pool never inspects it,
/// which is what keeps grim-memory free of model-specific types.
#[derive(Clone)]
pub struct RecurrentStateCheckpoint {
    /// Unique checkpoint ID.
    pub id: CheckpointId,
    /// Token offset (absolute index in the sequence) where the checkpoint was captured.
    pub token_offset: usize,
    /// Radix node index this checkpoint is anchored to.
    pub radix_node_id: usize,
    /// Per-layer recurrent states.
    pub layer_states: Vec<RecurrentLayerState>,
    /// Last access timestamp for LRU eviction.
    pub last_access: Instant,
    /// Opaque session snapshot captured at `token_offset` (T2 hybrid prefix reuse).
    pub session_snapshot: Option<SharedSessionSnapshot>,
}

impl std::fmt::Debug for RecurrentStateCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecurrentStateCheckpoint")
            .field("id", &self.id)
            .field("token_offset", &self.token_offset)
            .field("radix_node_id", &self.radix_node_id)
            .field("layer_states", &self.layer_states)
            .field("last_access", &self.last_access)
            .field(
                "session_snapshot",
                &self.session_snapshot.as_ref().map(|_| "<opaque>"),
            )
            .finish()
    }
}

/// Registry of semantic boundary token IDs for detecting anchor points.
#[derive(Debug, Clone)]
pub struct SemanticAnchorRegistry {
    /// Token IDs that indicate semantic boundaries (e.g. thinking tags, tool call tokens).
    anchor_tokens: HashSet<u32>,
}

impl SemanticAnchorRegistry {
    /// Create a new registry with default or provided anchor token IDs.
    /// # Contract Any token ID passed here will be treated as an anchor candidate during.
    pub fn new(anchor_token_ids: impl IntoIterator<Item = u32>) -> Self {
        Self {
            anchor_tokens: anchor_token_ids.into_iter().collect(),
        }
    }

    /// Register additional semantic anchor token ID.
    pub fn register_anchor(&mut self, token_id: u32) {
        self.anchor_tokens.insert(token_id);
    }

    /// Returns `true` if `token_id` is a recognized semantic anchor boundary.
    pub fn is_anchor(&self, token_id: u32) -> bool {
        self.anchor_tokens.contains(&token_id)
    }

    /// Find all 0-based token offsets in `tokens` that end at or immediately follow an anchor token.
    pub fn find_anchors(&self, tokens: &[u32]) -> Vec<usize> {
        tokens
            .iter()
            .enumerate()
            .filter_map(|(idx, &tok)| {
                if self.is_anchor(tok) {
                    Some(idx + 1)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Bounded LRU pool managing recurrent-state checkpoints anchored in memory.
pub struct RecurrentCheckpointPool {
    /// Maximum number of checkpoints allowed in memory.
    max_checkpoints: usize,
    /// Storage for active checkpoints.
    checkpoints: HashMap<CheckpointId, Arc<RecurrentStateCheckpoint>>,
    /// Map from RadixTree node ID to Checkpoint ID.
    node_to_checkpoint: HashMap<usize, CheckpointId>,
    /// Counter for generating monotonically increasing checkpoint IDs.
    next_id: usize,
}

impl RecurrentCheckpointPool {
    /// Create a new checkpoint pool with bounded capacity.
    /// # Contract `max_checkpoints` must be greater than zero.
    pub fn new(max_checkpoints: usize) -> Self {
        assert!(max_checkpoints > 0, "max_checkpoints must be > 0");
        Self {
            max_checkpoints,
            checkpoints: HashMap::with_capacity(max_checkpoints),
            node_to_checkpoint: HashMap::new(),
            next_id: 1,
        }
    }

    /// Store a recurrent checkpoint anchored to `radix_node_id`.
    /// # Contract If pool is at capacity, the least recently used checkpoint is evicted.
    pub fn store_checkpoint(
        &mut self,
        radix_node_id: usize,
        token_offset: usize,
        layer_states: Vec<RecurrentLayerState>,
    ) -> Arc<RecurrentStateCheckpoint> {
        // If this node already had a checkpoint, remove it first
        if let Some(old_id) = self.node_to_checkpoint.remove(&radix_node_id) {
            self.checkpoints.remove(&old_id);
        }

        // Evict LRU if capacity exceeded
        if self.checkpoints.len() >= self.max_checkpoints {
            self.evict_coldest();
        }

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);

        let checkpoint = Arc::new(RecurrentStateCheckpoint {
            id,
            token_offset,
            radix_node_id,
            layer_states,
            last_access: Instant::now(),
            session_snapshot: None,
        });

        self.checkpoints.insert(id, Arc::clone(&checkpoint));
        self.node_to_checkpoint.insert(radix_node_id, id);
        checkpoint
    }

    /// Store a checkpoint that carries an opaque session snapshot (T2).
    ///
    /// Same contract as [`Self::store_checkpoint`], plus the snapshot is kept
    /// verbatim and handed back by [`Self::get_by_node`]. `layer_states` is
    /// still accepted so a caller can record both representations; the engine
    /// passes an empty vec and relies on `session_snapshot`.
    pub fn store_session_checkpoint(
        &mut self,
        radix_node_id: usize,
        token_offset: usize,
        layer_states: Vec<RecurrentLayerState>,
        session_snapshot: Option<Box<dyn std::any::Any + Send>>,
    ) -> Arc<RecurrentStateCheckpoint> {
        // If this node already had a checkpoint, remove it first
        if let Some(old_id) = self.node_to_checkpoint.remove(&radix_node_id) {
            self.checkpoints.remove(&old_id);
        }

        // Evict LRU if capacity exceeded
        if self.checkpoints.len() >= self.max_checkpoints {
            self.evict_coldest();
        }

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);

        let checkpoint = Arc::new(RecurrentStateCheckpoint {
            id,
            token_offset,
            radix_node_id,
            layer_states,
            last_access: Instant::now(),
            session_snapshot: session_snapshot.map(|s| Arc::new(std::sync::Mutex::new(s))),
        });

        self.checkpoints.insert(id, Arc::clone(&checkpoint));
        self.node_to_checkpoint.insert(radix_node_id, id);
        checkpoint
    }

    /// Get checkpoint attached to `radix_node_id`, refreshing its LRU recency.
    pub fn get_by_node(&mut self, radix_node_id: usize) -> Option<Arc<RecurrentStateCheckpoint>> {
        let &cp_id = self.node_to_checkpoint.get(&radix_node_id)?;
        if let Some(cp) = self.checkpoints.get_mut(&cp_id) {
            // Update last access timestamp
            let mut updated = (**cp).clone();
            updated.last_access = Instant::now();
            let new_arc = Arc::new(updated);
            *cp = Arc::clone(&new_arc);
            Some(new_arc)
        } else {
            None
        }
    }

    /// Invalidate checkpoint attached to `radix_node_id` (e.g. when the Radix node is pruned).
    pub fn invalidate_node(&mut self, radix_node_id: usize) -> Option<CheckpointId> {
        if let Some(cp_id) = self.node_to_checkpoint.remove(&radix_node_id) {
            self.checkpoints.remove(&cp_id);
            Some(cp_id)
        } else {
            None
        }
    }

    /// Evict the least recently accessed checkpoint from the pool.
    fn evict_coldest(&mut self) -> Option<CheckpointId> {
        let mut coldest: Option<(CheckpointId, Instant)> = None;
        for (&id, cp) in &self.checkpoints {
            match coldest {
                None => coldest = Some((id, cp.last_access)),
                Some((_, cold_time)) if cp.last_access < cold_time => {
                    coldest = Some((id, cp.last_access))
                }
                _ => {}
            }
        }
        if let Some((cold_id, _)) = coldest {
            if let Some(cp) = self.checkpoints.remove(&cold_id) {
                self.node_to_checkpoint.remove(&cp.radix_node_id);
                return Some(cold_id);
            }
        }
        None
    }

    /// Current number of stored checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Returns `true` if no checkpoints are stored.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_anchor_registry() {
        let think_open = 1001;
        let think_close = 1002;
        let tool_call = 1003;

        let registry = SemanticAnchorRegistry::new([think_open, think_close, tool_call]);
        assert!(registry.is_anchor(think_open));
        assert!(!registry.is_anchor(9999));

        let prompt = vec![10, 20, think_open, 30, 40, think_close, 50, tool_call, 60];
        let anchors = registry.find_anchors(&prompt);
        // Offsets immediately following anchor tokens: index 2 -> 3, index 5 -> 6, index 7 -> 8
        assert_eq!(anchors, vec![3, 6, 8]);
    }

    #[test]
    fn test_checkpoint_pool_lru_eviction() {
        let mut pool = RecurrentCheckpointPool::new(2);

        let state1 = vec![RecurrentLayerState {
            layer_idx: 0,
            state_data: vec![1.0, 2.0],
            shape: vec![2],
        }];
        let state2 = vec![RecurrentLayerState {
            layer_idx: 0,
            state_data: vec![3.0, 4.0],
            shape: vec![2],
        }];
        let state3 = vec![RecurrentLayerState {
            layer_idx: 0,
            state_data: vec![5.0, 6.0],
            shape: vec![2],
        }];

        pool.store_checkpoint(10, 16, state1);
        pool.store_checkpoint(20, 32, state2);
        assert_eq!(pool.len(), 2);

        // Access node 10 so node 20 becomes coldest
        assert!(pool.get_by_node(10).is_some());

        // Storing 3rd must evict node 20
        pool.store_checkpoint(30, 48, state3);
        assert_eq!(pool.len(), 2);
        assert!(pool.get_by_node(10).is_some());
        assert!(pool.get_by_node(20).is_none());
        assert!(pool.get_by_node(30).is_some());
    }
}

/// Dual-copy linear/GDN recurrent state slot for speculative decoding / MTP (`RAD_KV_LINEAR` + `RAD_KV_CONV`).
/// Holds both `committed` and `draft` state buffers plus a rolling `conv_width - 1 + n_spec` conv buffer
/// so draft rejection (`rollback_draft`) and acceptance (`commit_draft`) require zero layer replay.
#[derive(Debug, Clone, PartialEq)]
pub struct DualRecurrentSlot {
    committed: Vec<f32>,
    draft: Vec<f32>,
    conv_ring: Vec<f32>,
    conv_window_len: usize,
}

impl DualRecurrentSlot {
    /// Allocate a dual-slot recurrent state of `state_len` floats and conv rolling window `(conv_width - 1) + n_spec`.
    pub fn new(state_len: usize, conv_width: usize, n_spec: usize) -> Self {
        let conv_window_len = conv_width.saturating_sub(1) + n_spec;
        Self {
            committed: vec![0.0; state_len],
            draft: vec![0.0; state_len],
            conv_ring: vec![0.0; conv_window_len],
            conv_window_len,
        }
    }

    /// Number of rolling conv slots (`conv_width - 1 + n_spec`).
    pub fn conv_window_len(&self) -> usize {
        self.conv_window_len
    }

    /// Seed the committed state and synchronize the draft copy.
    pub fn seed_committed(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.committed.len(), "seed_committed length mismatch");
        self.committed.copy_from_slice(data);
        self.draft.copy_from_slice(data);
    }

    /// Immutable view of the verified/committed recurrent state.
    pub fn committed_state(&self) -> &[f32] {
        &self.committed
    }

    /// Immutable view of the speculative draft recurrent state.
    pub fn draft_state(&self) -> &[f32] {
        &self.draft
    }

    /// Mutable view of the speculative draft recurrent state.
    pub fn draft_state_mut(&mut self) -> &mut [f32] {
        &mut self.draft
    }

    /// Promote the verified draft state to committed and sync draft copy.
    pub fn commit_draft(&mut self) {
        std::mem::swap(&mut self.committed, &mut self.draft);
        self.draft.copy_from_slice(&self.committed);
    }

    /// Discard rejected draft mutations by restoring from `committed`.
    pub fn rollback_draft(&mut self) {
        self.draft.copy_from_slice(&self.committed);
    }

    /// Mutable access to the rolling conv window (`conv_width - 1 + n_spec`).
    pub fn conv_ring_mut(&mut self) -> &mut [f32] {
        &mut self.conv_ring
    }
}
