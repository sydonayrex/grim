//! Telemetry, metrics and state accessors.

use crate::*;
use std::sync::atomic::Ordering;

/// D5: serializable config summary — the scalar subset of `EngineConfig`
/// (Arc handles are reported as booleans; the snapshot must stay one JSON
/// artifact).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigSummary {
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub block_pool_capacity: usize,
    pub determinism_mode: String,
    pub tp_size: usize,
    pub kv_compressor: bool,
    pub disagg_router: bool,
}

/// D5: dispatch-fallback counters at snapshot time — the QKV shared-attention
/// dispatch stats plus the fused quantized-matmul forward/backward stats.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FallbackCounters {
    pub qkv_attempts: u64,
    pub qkv_arena_fallbacks: u64,
    pub fused_forward_attempts: usize,
    pub fused_forward_fallbacks: usize,
    pub fused_backward_attempts: usize,
    pub fused_backward_fallbacks: usize,
}

/// D5: decode-graph poison state — capture opt-in plus the slot keys that
/// currently hold a stored graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphState {
    /// Decode-graph capture opt-in (`GRIM_CAPTURE_GRAPH`).
    pub capture_enabled: bool,
    /// Slot keys with a stored decode graph.
    pub stored_decode_graphs: Vec<String>,
}

/// D5: one per-request debug bundle — the artifact a failing request dumps
/// (config, fallback counters, sticky set, graph state) in one shot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineDebugSnapshot {
    pub request_id: u64,
    pub config: ConfigSummary,
    pub fallback_counters: FallbackCounters,
    pub sticky_failed_configs: usize,
    pub graph_state: GraphState,
}

impl Engine {
    /// D5: per-request debug bundle — one artifact a failing request can
    /// dump to stderr (or serve over `/metrics`) that names the engine
    /// config summary, the dispatch-fallback counters, the sticky
    /// failed-config set size, and the decode-graph state. Serializable, so
    /// the whole bundle renders as one machine-readable JSON line.
    pub fn debug_snapshot(&self, request_id: u64) -> EngineDebugSnapshot {
        let (qkv_attempts, qkv_arena_fallbacks, sticky) =
            grim_models_transformer::shared_attention::qkv_arena_fallback_stats();
        let fwd = &grim_backend_rocm::FUSED_FORWARD_DISPATCH_STATS;
        let bwd = &grim_backend_rocm::FUSED_BACKWARD_DISPATCH_STATS;
        EngineDebugSnapshot {
            request_id,
            config: ConfigSummary {
                max_batched_tokens: self.config.max_batched_tokens,
                max_num_seqs: self.config.max_num_seqs,
                block_pool_capacity: self.config.block_pool_capacity,
                determinism_mode: format!("{:?}", self.config.determinism_mode),
                tp_size: self.config.tp_size,
                kv_compressor: self.config.kv_compressor.is_some(),
                disagg_router: self.config.disagg_router.is_some(),
            },
            fallback_counters: FallbackCounters {
                qkv_attempts,
                qkv_arena_fallbacks,
                fused_forward_attempts: fwd.attempts.load(Ordering::Relaxed),
                fused_forward_fallbacks: fwd.fallback_calls.load(Ordering::Relaxed),
                fused_backward_attempts: bwd.attempts.load(Ordering::Relaxed),
                fused_backward_fallbacks: bwd.fallback_calls.load(Ordering::Relaxed),
            },
            sticky_failed_configs: sticky,
            graph_state: GraphState {
                capture_enabled: grim_backend_rocm::RocmDevice::shared(0).graph_capture_enabled(),
                stored_decode_graphs: self.decode_graphs.keys().cloned().collect(),
            },
        }
    }

    /// Return live snapshot of visible GPU capabilities if profiler is active.
    pub fn capabilities(&self) -> Option<Vec<grim_tensor::backend::GpuCapability>> {
        self.capability_profiler.as_ref().map(|p| p.capabilities())
    }

    /// Return topology link matrix between visible GPUs.
    pub fn link_matrix(&self, num_gpus: usize) -> Vec<grim_tensor::backend::ScytheLink> {
        grim_backend_rocm::CapabilityProfiler::link_matrix(num_gpus)
    }

    /// Resolve the replica id a pinned request executes on. Unpinned requests
    /// (or pins without a matching farm) resolve to the base id unchanged.
    pub(crate) fn effective_model_id(&self, request_id: u64, base: &str) -> String {
        match self.scythe_pin.get(&request_id).copied() {
            Some(rank) => self
                .scythe_replicas
                .get(base)
                .and_then(|ids| ids.get(rank))
                .cloned()
                .unwrap_or_else(|| base.to_string()),
            None => base.to_string(),
        }
    }

    /// Replica id a request currently routes to (`None` when the request has
    /// no tracked model). Status/telemetry surface for farm-mode routing.
    pub fn resolved_model_id(&self, request_id: u64) -> Option<String> {
        let base = self.request_model_ids.get(&request_id)?;
        if base.is_empty() {
            return None;
        }
        Some(self.effective_model_id(request_id, base))
    }

    /// Returns the exponential moving average of generated tokens per second.
    /// Returns None if no model is loaded or no tokens have been generated yet.
    pub fn tokens_per_sec(&self) -> Option<f32> {
        if self.models.is_empty() || self.total_tokens_generated == 0 {
            None
        } else {
            Some(self.tokens_per_sec_ema)
        }
    }

    /// Returns the exponential moving average of prefilled prompt tokens per second.
    /// Returns None if no model is loaded or no prompt tokens have been prefilled yet.
    pub fn prefill_tokens_per_sec(&self) -> Option<f32> {
        if self.models.is_empty() || self.total_tokens_prefilled == 0 {
            None
        } else {
            Some(self.prefill_tokens_per_sec_ema)
        }
    }

    /// Most recent measured prefill time in milliseconds.
    /// `None` means no completed prefill has been observed yet; callers must not invent a latency.
    pub fn last_ttft_ms(&self) -> Option<f64> {
        self.last_ttft_ms
    }

    /// Most recent measured inter-token latency in milliseconds.
    /// `None` means no completed decode step has been observed yet; callers must not invent a.
    pub fn last_itl_ms(&self) -> Option<f64> {
        self.last_itl_ms
    }

    /// Clear the TTFT/ITL trace so a caller measuring request-by-request (the WI-SB3 A/B harness) can distinguish a fresh measurement from the previous request's stale one.
    /// Without this, `last_ttft_ms()` stays `Some` forever and every later sample records the earlier value.
    pub fn clear_latency_trace(&mut self) {
        self.last_ttft_ms = None;
        self.last_itl_ms = None;
    }

    /// Snapshot of scheduler queues for status and metrics consumers.
    pub fn scheduler_snapshot(&self) -> grim_scheduler::SchedulerSnapshot {
        self.scheduler.snapshot()
    }

    /// Returns KV cache telemetry stats `(used_bytes, total_bytes, blocks_used, blocks_total)`.
    pub fn kv_cache_telemetry(&self) -> (u64, u64, u64, u64) {
        if let Ok(pool) = self.block_pool.lock() {
            let cap = pool.capacity() as u64;
            let used = pool.used_count() as u64;
            let b_bytes = pool.block_bytes() as u64;
            (used * b_bytes, cap * b_bytes, used, cap)
        } else {
            (0, 0, 0, 0)
        }
    }

    /// Layer 1.5: radix prefix-cache hit-rate telemetry —
    /// `(prefix_lookups, requests_with_hits, reused_tokens_total)`.
    pub fn radix_cache_telemetry(&self) -> (u64, u64, u64) {
        (
            self.radix_lookups,
            self.radix_hit_requests,
            self.radix_hit_tokens,
        )
    }

    /// Layer 1.5 gate: hybrid models (LFM2's ShortConv state, Mamba variants)
    /// keep per-layer state that a paged-KV prefix seed cannot reconstruct —
    /// prefix reuse for those requires recurrent checkpoints, which Layer 1.5
    /// deliberately declines (no anchor state → no seed). Returns true when
    /// the request's model must NOT receive a radix seed.
    pub(crate) fn model_uses_hybrid_kv(&self, id: u64) -> bool {
        self.model_for_request(id)
            .and_then(|(mid, _)| self.models.get(&mid))
            .map(|m| {
                m.model
                    .target()
                    .as_any()
                    .downcast_ref::<grim_models_transformer::Lfm2>()
                    .is_some()
            })
            // Unknown model: do not seed — skipping the shared prefix would be
            // a correctness risk we cannot rule out.
            .unwrap_or(true)
    }

    /// Maximum context limit (max batched tokens) configured for this engine.
    pub fn context_limit(&self) -> usize {
        self.config.max_batched_tokens
    }

    /// Total count of tokens generated since engine startup.
    pub fn total_tokens_generated(&self) -> u64 {
        self.total_tokens_generated
    }

    /// Total count of prompt tokens prefilled since engine startup.
    pub fn total_tokens_prefilled(&self) -> u64 {
        self.total_tokens_prefilled
    }

    /// Invalidate radix prefix cache and reclaim unreferenced KV blocks.
    /// Returns the number of reclaimed blocks.
    pub fn reset_prefix_cache(&mut self) -> usize {
        if let Ok(mut pool) = self.block_pool.lock() {
            pool.reset_prefix_cache()
        } else {
            0
        }
    }

    /// The model a new adapter should attach to by default: the engine's
    /// default model if set, else the first registered one.
    pub fn default_model_name(&self) -> Option<&str> {
        if let Some(default) = self.models.get("default") {
            let _ = default;
            return Some("default");
        }
        self.models.keys().next().map(String::as_str)
    }

    /// Returns a list of loaded model names.
    pub fn loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Strategy the model is operating under right now (Plain / NativeMtp /
    /// DSpark). `None` if the model id isn't registered.
    pub fn strategy_for(&self, id: &str) -> Option<Strategy> {
        self.models.get(id).map(|m| m.model.strategy())
    }

    /// Check if a model is registered by name.
    pub fn has_model(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    /// Deterministic RNG snapshot for a request, used by the speculative verifier when the engine's determinism mode is `Strict`.
    /// Returns `None` when the request isn't tracked.
    pub fn request_rng_state(&self, id: u64) -> Option<u64> {
        self.request_rng.get(&id).map(|r| r.state())
    }

    /// Replay: deterministically rewind a request's RNG by `steps`.
    /// Strict mode exposes this so re-running a tick with the same input reproduces the same.
    pub fn advance_request_rng(&mut self, id: u64, steps: usize) {
        if let Some(r) = self.request_rng.get_mut(&id) {
            for _ in 0..steps {
                r.next_u64();
            }
        }
    }

    /// Last captured outcome for the request, if any.
    pub fn last_outcome(&self, id: u64) -> Option<&StepOutcome> {
        self.last_outcomes.get(&id)
    }

    /// Record the token that was sampled for a request.
    /// Called by the server after sampling so the next decode step feeds the real token.
    pub fn record_generated_token(&mut self, id: u64, token: u32) {
        self.request_last_token.insert(id, token);
    }

    /// Pause a running request - §5.2.1. KV blocks are
    /// retained in the block pool at zero scheduling priority.
    pub fn pause_request(&mut self, id: u64) -> bool {
        let moved = self.scheduler.pause(id);
        if moved {
            // The session is kept; KV blocks remain ref-counted.
            // The speculative wrapper's mid-step tentative state stays anchored to the cache and resumes from where.
            if let Some(s) = self.sessions.get_mut(&id) {
                let _ = s;
            }
        }
        moved
    }

    /// Resume a previously-paused request - §5.2.1.
    /// The request continues from the exact token position where it was paused.
    pub fn resume_request(&mut self, id: u64) -> bool {
        self.scheduler.resume(id)
    }

    /// True if the request is currently paused.
    pub fn is_paused(&self, id: u64) -> bool {
        self.scheduler.is_paused(id)
    }

    pub fn model(&self, id: &str) -> Option<&LoadedModel> {
        self.models.get(id)
    }

    /// `(model_id, priority)` lookup for the request - a request is bound to exactly one model in v1.
    /// Under SCYTHE-2 farm mode the returned id is the pinned replica, so every caller (prefill.
    pub(crate) fn model_for_request(&self, id: u64) -> Option<(String, i32)> {
        let model_id = self.request_model_ids.get(&id)?;
        let base = if model_id.is_empty() {
            // Fallback: pick the first registered model.
            self.models.iter().next()?.0.clone()
        } else {
            model_id.clone()
        };
        Some((self.effective_model_id(id, &base), 0))
    }
}

#[cfg(test)]
mod debug_snapshot_tests {
    use super::*;

    /// D5: debug_snapshot bundles config summary, fallback counters, sticky
    /// set size and graph state in one artifact — and the JSON round-trips,
    /// so a failing request can dump it as one machine-readable line.
    #[test]
    fn debug_snapshot_bundles_request_debug_state() {
        let engine = Engine::new(EngineConfig::default());
        let snap = engine.debug_snapshot(42);
        assert_eq!(snap.request_id, 42);
        assert_eq!(
            snap.config.max_batched_tokens,
            EngineConfig::default().max_batched_tokens
        );
        // The sticky count must match the live stats at snapshot time
        // (exact, not assumed-zero — parallel GPU tests may poison it).
        let (_, _, sticky_now) =
            grim_models_transformer::shared_attention::qkv_arena_fallback_stats();
        assert_eq!(snap.sticky_failed_configs, sticky_now);
        // Fresh engine: no decode graph stored, no counters inflated by this call.
        assert!(snap.graph_state.stored_decode_graphs.is_empty());
        // The bundle is machine-readable: every section survives a JSON parse.
        let line = serde_json::to_string(&snap).unwrap();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["request_id"], 42);
        assert!(v["config"]["max_batched_tokens"].is_u64());
        assert!(v["config"]["determinism_mode"].is_string());
        assert!(v["fallback_counters"]["qkv_attempts"].is_u64());
        assert!(v["fallback_counters"]["fused_forward_fallbacks"].is_u64());
        assert!(v["fallback_counters"]["fused_backward_fallbacks"].is_u64());
        assert!(v["graph_state"]["capture_enabled"].is_boolean());
        assert!(v["graph_state"]["stored_decode_graphs"].is_array());
    }
}
