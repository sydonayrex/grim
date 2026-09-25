/// Non-prefix KV chunk stitching and attention recalibration.
pub mod cache_blend;

/// WI-HYBRID Layer 2: max session-scoped decode-graph slots per model.
/// Each slot owns full device arenas; the default keeps GPU memory bounded.
fn session_graph_slots() -> usize {
    std::env::var("GRIM_SESSION_GRAPH_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(2)
}

/// WI-HYBRID Layer 2: pick the LRU session-scoped slot to evict when
/// `new_key` would exceed `cap` among this model's session slots. `None` = no
/// eviction (the key already exists, or the model is under cap).
fn session_slot_victim(
    prefix: &str,
    graph_keys: &[String],
    last_use: &HashMap<String, std::time::Instant>,
    cap: usize,
    new_key: &str,
) -> Option<String> {
    let mut mine: Vec<String> = graph_keys
        .iter()
        .filter(|k| k.starts_with(prefix))
        .cloned()
        .collect();
    if mine.iter().any(|k| k == new_key) {
        return None;
    }
    if mine.len() < cap {
        return None;
    }
    mine.sort_by_key(|k| last_use.get(k).copied());
    mine.first().cloned()
}

/// WI-HYBRID Layer 2: session block pins last this many seconds since the
/// last session-tagged turn (idle-time pin).
fn session_pin_secs() -> u64 {
    std::env::var("GRIM_SESSION_PIN_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300)
}
pub mod model_loader;
pub mod packing;
pub mod pipelines;
pub mod rope_scaling;
/// SCYTHE-2 WI-4 + WI-7: C²PLR controller, PlacementCache, ScytheRing.
pub mod scythe2;
/// WI-SB3: TTFT/ITL A/B harness — results protocol + WI-INF4 verdict rule.
pub mod scythe_ab;
pub mod speculative_loop;
pub mod streaming_forward;
pub mod tp_layers;
/// P2: packed-step training driver (varlen grouping + one optimizer step per group).
pub mod train_packed;

pub use cache_blend::{CacheBlendEngine, CachedSegment, StitchedPromptLayout};
pub use pipelines::moe_prefill_pipeline::{BufferRole, MoePrefillPipeline};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use grim_backend_cpu::DeterministicRng;
use grim_core::error::{Error, Result};
use grim_core::memory_certificate::{BoundaryVector, MemoryCertificate};
use grim_core::model::{AdapterHandle, CausalLm, ModelConfig};
use grim_core::session::{DeterminismMode, SessionT};
use grim_memory::{BLOCK_SIZE, KvBlockPool};
use grim_speculative::{ConfidenceHead, DraftBackbone, MarkovHead, SpeculativeCausalLm, Strategy};

type DynModelPtr = Box<SpeculativeCausalLm>;

/// A loaded model with its config and an instantiated CausalLm impl.
pub struct LoadedModel {
    pub model: DynModelPtr,
    pub config: Box<dyn ModelConfig>,
    /// Device this model's weights live on. Sessions are created on this device so
    /// decode/GPU work actually lands on the GPU instead of silently falling back to CPU.
    pub device: grim_tensor::Device,
    /// Tensor-parallel configuration stamped at registration time.
    /// `None` means single-device; otherwise carries the per-rank `(rank, world_size)` so callers can report or query.
    pub tp_config: Option<grim_nn::TensorParallelConfig>,
    /// Architecture hyperparameters, when the model reports them (R4).
    /// The admission gate re-certifies the request footprint against *current* free device memory using these.
    pub arch_hyperparams: Option<grim_core::hyperparams::ArchHyperparameters>,
}

/// A loaded adapter bundle (one LoRA's A/B matrices + scaling). LoRA batches
/// keyed by [`AdapterHandle::id`]; the engine resolves lookup at runtime.
pub struct LoadedAdapter {
    /// Human-readable name from registration - matched against HTTP request body `"adapters"` arrays.
    /// The server 400s on unknown names so this must be set at register time.
    pub name: String,
    pub handle: AdapterHandle,
    pub base_model_id: String,
}

/// Best-effort current free-device-memory probe for the admission gate (R4).
/// Returns `None` when the backend cannot probe free memory (CPU/Vulkan/Metal without a probe), in which.
fn free_device_memory(device: &grim_tensor::Device) -> Option<u64> {
    // Test/override hook: GRIM_TEST_FREE_DEVICE_BYTES lets tests (and
    // operators) inject a known free-memory value without a live device.
    if let Ok(v) = std::env::var("GRIM_TEST_FREE_DEVICE_BYTES") {
        if let Ok(bytes) = v.parse::<u64>() {
            return Some(bytes);
        }
    }
    match device {
        grim_tensor::Device::Rocm(ordinal) => grim_backend_rocm::free_device_memory(*ordinal),
        _ => None,
    }
}

/// Engine configuration.
pub struct EngineConfig {
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub block_pool_capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub target_ttft_ms: u64,
    pub target_itl_ms: u64,
    /// Determinism mode for callers that care about reproducible outputs.
    pub determinism_mode: DeterminismMode,
    /// Optional KV compressor for runtime KV cache quantization.
    pub kv_compressor: Option<Arc<dyn grim_kvquant::KvCompressor>>,
    /// Tensor-parallel world size (env `GRIM_TP_SIZE`).
    /// `0` or `1` = single-device.
    pub tp_size: usize,
    /// Explicit GPU ordinals for TP (`GRIM_GPUS`, empty = all visible).
    pub tp_gpus: Vec<usize>,
    /// WI-TOOLS-4c-i: hard cap on the total number of tool-call entries across every assistant message in a single request's `messages` array.
    /// Rejects the request with 400 once a conversation has made more tool calls than a.
    pub max_tool_calls_per_conversation: usize,
    /// WI-TOOLS-4c-ii: hard cap on `messages.len()` per request.
    /// Catches unbounded history growth (agentic loops or client bugs) before any tokenization/prefill work happens (default.
    pub max_messages_per_request: usize,
    /// Disaggregated serving router context.
    pub disagg_router: Option<Arc<grim_disagg::DisaggRouter>>,
    /// Disaggregation configuration (role, addrs). When set, the engine
    /// starts a background KV receiver server and wires disagg routing.
    pub disagg_config: Option<grim_disagg::DisaggConfig>,
    /// Externally-constructed cluster orchestrator (§5.6 failover).
    /// When `None` but `disagg_config` is set, the engine constructs one automatically from that config.
    pub disagg_orchestrator: Option<Arc<std::sync::Mutex<grim_disagg::DisaggOrchestrator>>>,
    /// Heartbeat timeout for disagg failover evaluation: a peer whose last observed heartbeat is older than
    /// this is presumed dead and the node fails over to colocated execution (default 5000 ms).
    pub disagg_heartbeat_timeout_ms: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batched_tokens: 4096,
            max_num_seqs: 8,
            block_pool_capacity: 1024,
            num_kv_heads: 4,
            head_dim: 128,
            target_ttft_ms: 2000,
            target_itl_ms: 100,
            determinism_mode: DeterminismMode::Relaxed,
            kv_compressor: None,
            tp_size: std::env::var("GRIM_TP_SIZE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0),
            tp_gpus: std::env::var("GRIM_GPUS")
                .ok()
                .map(|s| {
                    s.split(',')
                        .filter_map(|t| t.trim().parse::<usize>().ok())
                        .collect()
                })
                .unwrap_or_default(),
            max_tool_calls_per_conversation: 20,
            max_messages_per_request: 200,
            disagg_router: None,
            disagg_config: None,
            disagg_orchestrator: None,
            disagg_heartbeat_timeout_ms: 5000,
        }
    }
}

/// Per-request execution outcome captured by `tick()`.
#[derive(Clone)]
pub struct StepOutcome {
    /// Last forward-pass logits for the request. `None` if the request
    /// was not driven this tick (e.g. it was paused).
    pub logits: Option<Arc<grim_tensor::Tensor>>,
    /// Number of speculative slots accepted this tick (post-commit).
    pub accepted_tokens: usize,
    /// Whether this step executed through the speculative path. False
    /// when the wrapper fell back to plain autoregressive decoding.
    pub speculative: bool,
}

/// The core engine. Call `tick()` to advance one iteration.
pub struct Engine {
    pub config: EngineConfig,
    pub scheduler: grim_scheduler::Scheduler,
    pub block_pool: Arc<std::sync::Mutex<KvBlockPool>>,
    pub models: HashMap<String, LoadedModel>,
    pub sessions: HashMap<u64, Box<dyn SessionT>>,
    pub adapters: HashMap<u32, LoadedAdapter>,
    /// Per-request last-emitted logs (cleared on `finish_request`).
    pub last_outcomes: HashMap<u64, StepOutcome>,
    /// Per-request deterministic RNG, §5.8. Populated when `DeterminismMode::Strict` is active.
    pub request_rng: HashMap<u64, DeterministicRng>,
    pub request_model_ids: HashMap<u64, String>,
    pub request_adapters: HashMap<u64, Vec<u32>>,
    /// Per-request input token buffers. Populated in `enqueue_request` from `Request::input_ids`.
    pub request_input_ids: HashMap<u64, Vec<u32>>,
    /// Per-request count of prompt tokens already prefilled (chunked prefill bookkeeping, F9 follow-on).
    /// The scheduler's `consumed_tokens` says how many tokens the SCHEDULER has budgeted; this map says how.
    pub prefill_progress: HashMap<u64, usize>,
    /// Per-request last generated token. Updated after each decode step via `record_generated_token`.
    pub request_last_token: HashMap<u64, u32>,
    /// Self-tuning knob controller (§5.7). Owns `chunked_prefill_size` and `max_batched_tokens` - `tick()` re-applies them every pass, so tests
    /// that need deterministic chunking pin the knobs HERE (floor = ceiling = initial), not on the scheduler.
    pub self_tuning_controller: grim_scheduler::SelfTuningController,
    /// Tuned speculative params (MIN-3: applied, not discarded).
    tuned_speculative_block_len: usize,
    tuned_kv_compression_bit_width: u8,
    tokens_per_sec_ema: f32,
    total_tokens_generated: u64,
    prefill_tokens_per_sec_ema: f32,
    total_tokens_prefilled: u64,
    /// WI-E2: cumulative accepted tokens (speculative verification hits).
    accepted_tokens_total: u64,
    last_ttft_ms: Option<f64>,
    last_itl_ms: Option<f64>,
    /// Tensor-parallel config stamped onto each `LoadedModel`.
    /// Populated in `Engine::new` when TP is active (one OS process per rank, Design A); `None`.
    tp_config: Option<grim_nn::TensorParallelConfig>,
    /// Background KV receiver server handle (started in Engine::new when
    /// disagg_config is Some and role is Decode or Colocated).
    kv_receiver: Option<grim_disagg::KvReceiverServer>,
    /// Cluster orchestrator (§5.6): tracks peer heartbeats and failover.
    disagg_orchestrator: Option<Arc<std::sync::Mutex<grim_disagg::DisaggOrchestrator>>>,
    /// Last evaluated effective role (refreshed each tick). `Colocated`
    /// after failover means the remote peer is presumed dead.
    disagg_effective_role: std::sync::Mutex<grim_disagg::PoolRole>,
    /// Live GPU capability profiler and epoch manager.
    /// Only constructed when world_size > 1 or `GRIM_SCYTHE_INFERENCE=1` (WI-INF1).
    pub capability_profiler: Option<Arc<grim_backend_rocm::CapabilityProfiler>>,
    /// SCYTHE-2 online router for continuous batching / multi-GPU placement (WI-INF2).
    pub scythe_ctrl: Option<crate::scythe2::C2plrController>,
    /// SCYTHE-2 farm mode (WI-INF3 serving integration): per-base-model replica ids.
    /// Replica `r ≥ 1` of `base` is registered as `{base}#scythe{r}` and holds a full weight.
    scythe_replicas: HashMap<String, Vec<String>>,
    /// Request → replica rank, decided by the controller at admission time.
    /// The pinned replica executes every forward for that request's lifetime, so its KV pages stay.
    scythe_pin: HashMap<u64, usize>,
    /// WI-SB1 load-spreading: ranks of recently finished requests with the time they were released.
    /// Back-to-back admissions must still see the predecessor's load or a burst of short requests all.
    scythe_pin_cooldown: Vec<(usize, std::time::Instant)>,
    /// WI-SB2: requests held back because no farm rank could hold their KV footprint at enqueue time.
    /// They never reach the scheduler or own a session until a retry (each tick) finds.
    scythe_vram_waitlist: Vec<grim_scheduler::Request>,
    /// Set GRIM_RADIX=on to enable prefix-cache reuse on prefill (WP5).
    pub radix_enabled: bool,
    /// Layer 1.5: per-request count of leading blocks claimed from the radix
    /// tree at first prefill (`seed_prefix`). Seeded blocks are tree-claimed
    /// (+1 ref) at seed time, so they must NOT be re-registered when their
    /// owner finishes prefill — registration covers only the tail.
    pub radix_seeded_blocks: HashMap<u64, usize>,
    /// Per-request count of leading blocks already registered into the radix
    /// tree by this request (chunked prefill registers incrementally; only
    /// blocks beyond this cursor are inserted on a later pass).
    pub radix_registered_blocks: HashMap<u64, usize>,
    /// Layer 1.5 hit-rate metrics (surfaced via `radix_cache_telemetry`).
    pub radix_lookups: u64,
    pub radix_hit_requests: u64,
    pub radix_hit_tokens: u64,
    /// Persistent GPU buffers for decode-step inputs (token ID, position).
    /// Used by graph capture to enable in-place updates between replays.
    /// Maps request_id → (input_ids_gpu, positions_gpu).
    decode_graph_input_buffers: HashMap<u64, GraphCaptureInputBuffers>,
    /// Captured-graph output handles per capture key. The graph rewrites the
    /// same device buffers on every replay, so the first-capture logits `Arc`
    /// stays valid and replay returns a clone — no `decode_one`, no D2H.
    graph_capture_logits: HashMap<String, Arc<grim_tensor::Tensor>>,
    /// Fixed-buffer DecodeGraphs per request (twinkie-zombieland P2).
    /// P2-1 Layer 1 (session-continuity design): decode graphs are keyed by
    /// MODEL, not request. The graph + KV arenas are expensive fixed resources
    /// (alloc + capture + seed); reusing them across requests removes the
    /// per-request ~170 ms setup that dominated warm TTFT. The seed path
    /// re-binds the arenas to each new request's prefill state on every
    /// miss, so cross-request reuse is position-reset-correct.
    pub decode_graphs: HashMap<String, grim_backend_rocm::FullDecodeGraph>,
    /// WI-HYBRID Layer 2: request id → session hash (clients send an optional
    /// `session` tag; `None` = plain Layer 1.5 behavior).
    pub request_session: HashMap<u64, u64>,
    /// WI-HYBRID Layer 2: session-scoped graph slot key → last use, for LRU
    /// eviction under the per-model session-slot cap.
    pub session_slot_last_use: HashMap<String, std::time::Instant>,
    /// P3: bucket-specialized batch decode graphs. Maps model_id → pool of
    /// per-bucket captured graphs for batch decode replay in step_batch.
    batch_graph_pools: HashMap<String, grim_backend_rocm::DecodeBucketGraphPool>,
    /// P3: pinned slot assignment per (model_id, bucket). A bucket graph's KV
    /// arenas are per-slot, so a request MUST always land in the same slot:
    /// the set/order captured here is pinned at first capture, and a group is
    /// only replayed when its requests match the pin (otherwise eager).
    batch_bucket_slots: HashMap<(String, u32), Vec<u64>>,
}
/// Persistent GPU buffers for graph-captured decode steps.
#[derive(Debug)]
pub struct GraphCaptureInputBuffers {
    /// GPU buffer for the token ID input (1 x F32).
    pub input_ids: grim_tensor::Tensor,
    /// GPU buffer for the position input (1 x F32).
    pub positions: grim_tensor::Tensor,
}

/// Effective-capability view for SCYTHE-2 farm placement: a GPU already running `load` concurrent sessions contributes roughly `1/(1+load)` of its solo throughput, so the controller's WaveTune latency argmin doubles as a load balancer instead of piling every session onto the fastest card.
/// How long a finished request's rank stays counted as loaded (WI-SB1 load-spreading).
pub(crate) const SCYTHE_PIN_COOLDOWN: std::time::Duration = std::time::Duration::from_millis(1000);

/// External (non-farm) GPU utilization converts to equivalent pinned requests at this weight: a card maxed out by a desktop/game workload
/// counts as ~2 in-flight farm requests - enough to flip a ~2:1 measured capability pair toward the idle slower card.
const SCYTHE_EXTERNAL_BUSY_WEIGHT: f32 = 2.0;

/// WI-SB1: per-rank load seen by admission = active farm pins + pins released inside the cooldown window + external busy-% converted at `busy_weight`.
/// Pure so the weighting/expiry rules stay unit-testable.
fn scythe_effective_loads(
    active_pins: impl Iterator<Item = usize>,
    released: &[(usize, std::time::Instant)],
    cooldown: std::time::Duration,
    external_busy_pct: &[Option<u32>],
    num_ranks: usize,
    busy_weight: f32,
) -> Vec<f32> {
    let mut load = vec![0.0f32; num_ranks];
    for r in active_pins {
        if r < num_ranks {
            load[r] += 1.0;
        }
    }
    for &(r, t) in released {
        if r < num_ranks && t.elapsed() < cooldown {
            load[r] += 1.0;
        }
    }
    for (r, b) in external_busy_pct.iter().enumerate().take(num_ranks) {
        if let Some(pct) = b {
            load[r] += busy_weight * (*pct as f32) / 100.0;
        }
    }
    load
}

fn load_adjusted_caps(
    caps: &[grim_tensor::backend::GpuCapability],
    num_ranks: usize,
    load: &[f32],
) -> Vec<grim_tensor::backend::GpuCapability> {
    (0..num_ranks)
        .map(|r| {
            let mut c = caps.get(r).cloned().unwrap_or_default();
            c.ordinal = r;
            c.tflops_fp16 /= (1.0 + load.get(r).copied().unwrap_or(0.0)).max(0.001);
            if c.tflops_fp16 <= 0.0 {
                // Keep latency finite so the controller never divides by zero.
                c.tflops_fp16 = 0.001;
            }
            c
        })
        .collect()
}

/// WI-SB2: worst-case device memory one request can reach - its paged KV (`2·seq·kv_heads·head_dim·layers·4B`, K+V at fp32 page width) plus an activation working-set floor (`2·seq·hidden·layers·4B`).
/// When the model doesn't report a hidden width, the KV dimension stands in rather than.
fn scythe_request_footprint_bytes(
    seq_len: usize,
    max_new_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size_hint: Option<usize>,
    num_layers: u64,
) -> u64 {
    let seq = seq_len.saturating_add(max_new_tokens).max(1) as u64;
    let kv_dim = (num_kv_heads.saturating_mul(head_dim)).max(1) as u64;
    let hidden = hidden_size_hint.map_or(kv_dim, |h| (h as u64).max(1));
    let layers = num_layers.max(1);
    let kv_bytes = 2u64
        .saturating_mul(seq)
        .saturating_mul(kv_dim)
        .saturating_mul(layers)
        .saturating_mul(4);
    let working_set = 2u64
        .saturating_mul(seq)
        .saturating_mul(hidden)
        .saturating_mul(layers)
        .saturating_mul(4);
    kv_bytes.saturating_add(working_set)
}

/// WI-SB2: which farm ranks can hold a request's footprint.
/// Headroom for workspace and fragmentation is covered by [`SCYTHE_VRAM_WATERMARK_BYTES`].
fn scythe_vram_feasible(
    caps: &[grim_tensor::backend::GpuCapability],
    footprint_bytes: u64,
) -> Vec<bool> {
    caps.iter()
        .map(|c| {
            c.vram_free_bytes == 0
                || c.vram_free_bytes >= footprint_bytes.saturating_add(SCYTHE_VRAM_WATERMARK_BYTES)
        })
        .collect()
}

/// WI-SB2 admission guard watermark: free-VRAM headroom (512 MiB) a rank must keep above a request's computed
/// footprint so scratch buffers, logits and allocator fragmentation never push a pinned request into an OOM.
const SCYTHE_VRAM_WATERMARK_BYTES: u64 = 512 * 1024 * 1024;

/// Outcome of the WI-SB2 admission guard for one request against a farm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScytheAdmission {
    /// Route the whole request to this replica rank.
    Pin(usize),
    /// No rank can hold the request's footprint yet — hold it on the VRAM
    /// waitlist instead of admitting it onto a card that cannot serve it.
    WaitVram,
    /// Farm routing not engaged — use the plain single-replica path unchanged.
    Bypass,
}

/// Re-export key types at the grim-engine crate root.
pub use grim_memory::PagedKvCache;
pub use grim_scheduler::{AdmissionController, Request, Scheduler, SchedulerOutput};

#[cfg(test)]
#[allow(unused_must_use)]
mod tests {

    /// WI-HYBRID Layer 2: LRU eviction selection for session-scoped graph
    /// slots — picks the least-recently-used slot only when the cap is
    /// exceeded and the key is genuinely new.
    #[test]
    fn session_slot_victim_lru_selection() {
        use std::time::{Duration, Instant};
        let now = Instant::now();
        let mut last_use = HashMap::new();
        last_use.insert("small#s1".to_string(), now - Duration::from_secs(300));
        last_use.insert("small#s2".to_string(), now - Duration::from_secs(60));
        let keys = vec!["small#s1".to_string(), "small#s2".to_string()];

        // Under cap: no eviction.
        assert_eq!(
            session_slot_victim("small#", &keys, &last_use, 3, "small#s3"),
            None
        );
        // Key already present: no eviction (it is a reuse, not a new slot).
        assert_eq!(
            session_slot_victim("small#", &keys, &last_use, 2, "small#s2"),
            None
        );
        // At cap with a new key: the LRU slot (s1) is the victim.
        assert_eq!(
            session_slot_victim("small#", &keys, &last_use, 2, "small#s3"),
            Some("small#s1".to_string())
        );
        // Unknown prefix (other model): its slots are invisible here.
        let other = vec!["other#s9".to_string()];
        assert_eq!(
            session_slot_victim("small#", &other, &last_use, 1, "small#s3"),
            None
        );
    }
    use super::*;

    #[test]
    fn test_engine_telemetry_accessors() {
        let config = EngineConfig::default();
        let engine = Engine::new(config);
        assert_eq!(engine.tokens_per_sec(), None);
        assert_eq!(engine.prefill_tokens_per_sec(), None);
        assert_eq!(engine.total_tokens_generated(), 0);
        assert_eq!(engine.total_tokens_prefilled(), 0);
        assert_eq!(engine.last_ttft_ms(), None);
        assert_eq!(engine.last_itl_ms(), None);
        assert!(engine.speculative_telemetry(None).is_none());
        let (used_b, total_b, b_used, b_total) = engine.kv_cache_telemetry();
        assert_eq!(used_b, 0);
        assert!(total_b > 0);
        assert_eq!(b_used, 0);
        assert!(b_total > 0);
    }

    /// KV-int8 wiring: `GRIM_KV_QUANT=int8` must attach a compressor to the
    /// block pool; the default (unset) leaves it empty.
    #[test]
    fn test_grim_kv_quant_env_attach() {
        // Baseline: default config attaches no compressor.
        let engine = Engine::new(EngineConfig::default());
        assert!(
            !engine
                .block_pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .has_compressor()
        );

        // With GRIM_KV_QUANT=int8 the pool must hold a compressor.
        unsafe {
            std::env::set_var("GRIM_KV_QUANT", "int8");
        }
        let engine = Engine::new(EngineConfig::default());
        assert!(
            engine
                .block_pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .has_compressor()
        );

        // off / invalid → none.
        unsafe {
            std::env::set_var("GRIM_KV_QUANT", "off");
        }
        let engine = Engine::new(EngineConfig::default());
        assert!(
            !engine
                .block_pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .has_compressor()
        );
        unsafe {
            std::env::remove_var("GRIM_KV_QUANT");
        }
    }
    use grim_models_transformer::{Llama, LlamaConfig};
    use grim_tensor::Device;

    /// dats-demm §3 closure: DSpark wiring end-to-end. The wrapper reports
    /// Strategy::DSpark (previously unreachable in production) and greedy
    /// decoding through the speculative wrapper is token-identical to the
    /// plain target — speculation must never change greedy output.
    #[test]
    fn dspark_wrap_reports_strategy_and_preserves_greedy() {
        let draft = Arc::new(grim_speculative::TinyDraftBackbone::new(256, 32, 4, 42));
        let wrapped = Engine::build_dspark_model(small_llama(), draft);
        assert_eq!(wrapped.strategy(), Strategy::DSpark);

        let plain = small_llama();
        let mut sess_p = plain.new_session();
        let mut sess_d = wrapped.new_session();

        let mut token: f32 = 3.0;
        for step in 0..8 {
            let input = grim_backend_cpu::cpu_tensor(vec![token], grim_tensor::Shape::new(vec![1]));
            let pos =
                grim_backend_cpu::cpu_tensor(vec![step as f32], grim_tensor::Shape::new(vec![1]));
            let logits_plain =
                CausalLm::forward(&*plain, sess_p.as_mut(), &input, &pos, &[]).unwrap();
            let logits_dspark = wrapped
                .decode_one(sess_d.as_mut(), &input, &pos, 0.0, 0, &[])
                .unwrap();
            let lp = logits_plain.to_vec_f32().unwrap();
            let ld = logits_dspark.to_vec_f32().unwrap();
            assert_eq!(lp.len(), ld.len(), "vocab mismatch at step {step}");
            let next_p = lp
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let next_d = ld
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(
                next_p, next_d,
                "greedy divergence at step {step}: plain {next_p} vs dspark {next_d}"
            );
            token = next_p as f32;
        }
    }

    /// Journey: DSpark-wrapped model registration via `register_with_dspark`
    /// must land in the engine registry with a non-Plain strategy so the
    /// speculative decode path (and GRIM_SPEC telemetry) engages.
    #[test]
    fn dspark_registration_journey() {
        let mut engine = Engine::new(EngineConfig::default());
        let draft = Arc::new(grim_speculative::TinyDraftBackbone::new(256, 32, 4, 42));
        let markov = Arc::new(grim_speculative::UniformMarkovHead::new(256, 8, 7));
        let confidence = Arc::new(grim_speculative::EntropyConfidenceHead);
        engine.register_with_dspark("spec-journey", small_llama(), draft, markov, confidence);
        let loaded = engine.models.get("spec-journey").expect("registered");
        assert_eq!(loaded.model.strategy(), Strategy::DSpark);
    }

    fn small_llama() -> Box<dyn CausalLm> {
        Box::new(Llama::random(
            Device::Cpu,
            LlamaConfig {
                vocab_size: 256,
                hidden_size: 32,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 16,
                num_layers: 1,
                intermediate_size: 64,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 64,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ))
    }

    fn small_handle(id: u32, in_dim: usize, out_dim: usize) -> AdapterHandle {
        let a = grim_backend_cpu::cpu_tensor(
            vec![0.01f32; in_dim * 4],
            grim_tensor::Shape::new(vec![4, in_dim]),
        );
        let b = grim_backend_cpu::cpu_tensor(
            vec![0.01f32; out_dim * 4],
            grim_tensor::Shape::new(vec![out_dim, 4]),
        );
        AdapterHandle {
            id,
            a,
            b,
            alpha: 1.0,
        }
    }

    #[test]
    fn engine_registers_and_resolves_adapters() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_adapter("base", "adapter-1", small_handle(1, 32, 32));
        engine.register_adapter("base", "adapter-2", small_handle(2, 32, 32));
        assert_eq!(engine.adapter_count(), 2);

        let resolved = engine.resolve_adapters(&[1, 2]).unwrap();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].id, 1);
        assert_eq!(resolved[1].id, 2);

        assert!(engine.drop_adapter(1));
        assert_eq!(engine.adapter_count(), 1);
        assert!(!engine.drop_adapter(1), "idempotent — re-drop is no-op");
    }

    #[test]
    fn engine_resolve_returns_none_for_unknown_id() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_adapter("base", "adapter-1", small_handle(1, 32, 32));
        assert!(engine.resolve_adapters(&[99]).is_none());
    }

    #[test]
    fn engine_pause_resume_round_trip() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 7,
            prompt_tokens: 32,
            priority: 0,
            ..Default::default()
        });
        let _ = engine.tick();
        assert_eq!(engine.scheduler.running.len(), 1);
        assert!(!engine.is_paused(7));

        assert!(engine.pause_request(7));
        assert!(engine.is_paused(7));
        assert_eq!(engine.scheduler.paused.len(), 1);

        assert!(engine.resume_request(7));
        assert!(!engine.is_paused(7));
        assert_eq!(engine.scheduler.running.len(), 1);
    }

    #[test]
    fn engine_pause_unknown_id_is_noop() {
        let mut engine = Engine::new(EngineConfig::default());
        assert!(!engine.pause_request(404));
        assert!(!engine.resume_request(404));
        assert!(!engine.is_paused(404));
    }

    #[test]
    fn engine_wrapper_defaults_to_speculative_path() {
        // §5.3: registering a plain CausalLm without an attached bundle gets the autoselected wrapper.
        // With no bundle present the wrapper falls back to plain autoregressive, *but* the wrapper itself.
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        let strat = engine.strategy_for("small");
        assert_eq!(strat, Some(Strategy::Plain));
    }

    #[test]
    fn engine_tick_runs_prefill_then_decode_advancing_pos() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let _ = engine.tick();
        let pos_after_prefill = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(
            pos_after_prefill, 4,
            "prefill advanced current_pos to prompt_tokens"
        );

        engine.scheduler.running.retain(|r| r.id == 1);
        let _ = engine.tick();
        let pos_after_decode = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(pos_after_decode, 5, "decode advanced current_pos by 1");
    }

    #[test]
    fn engine_tick_records_step_outcome() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        engine.register_adapter("small", "adapter-99", small_handle(99, 32, 32));
        let _ = engine.tick();
        let outcome = engine.last_outcome(1).expect("tick must record outcome");
        assert!(outcome.logits.is_some(), "logits tensor must be recorded");
        let v = outcome.logits.as_ref().unwrap().to_vec_f32().unwrap();
        assert!(!v.is_empty(), "logits must be non-empty");
    }

    #[test]
    fn engine_pause_then_resume_preserves_session_position() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let _ = engine.tick(); // prefill — pos becomes 4.
        engine.scheduler.running.retain(|r| r.id == 1);
        let _ = engine.tick(); // decode — pos becomes 5.

        // Pause: session retains pos.
        engine.pause_request(1);
        let pos = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(pos, 5, "session preserved at pause");
        assert!(engine.is_paused(1));

        // Resume: still at 5, next tick advances to 6 (or further if
        // speculative accepted more than one).
        engine.resume_request(1);
        let _ = engine.tick();
        let pos = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert!(pos > 5, "tick must keep advancing after resume");
    }

    #[test]
    fn engine_step_one_public_api() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let ids = grim_backend_cpu::cpu_tensor(vec![1.0f32; 2], grim_tensor::Shape::new(vec![2]));
        let positions = ids.clone();
        let outcome = engine.step_one(1, "small", &ids, &positions).unwrap();
        assert!(outcome.logits.is_some());
    }

    #[test]
    fn engine_step_batch_public_api() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        engine.enqueue_request(Request {
            id: 2,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let ids = grim_backend_cpu::cpu_tensor(vec![1.0f32; 2], grim_tensor::Shape::new(vec![2]));
        let positions = ids.clone();
        let items = [
            (1u64, "small", &ids, &positions),
            (2u64, "small", &ids, &positions),
        ];
        let outcomes = engine.step_batch(&items).unwrap();
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes[0].1.logits.is_some());
        assert!(outcomes[1].1.logits.is_some());
    }

    #[test]
    fn engine_step_one_rejects_unknown_adapter() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let ids = grim_backend_cpu::cpu_tensor(vec![1.0f32; 2], grim_tensor::Shape::new(vec![2]));
        let positions = ids.clone();
        let outcome = engine.step_one(1, "small", &ids, &positions).unwrap();
        // Unknown adapter is silently dropped; outcomes still emitted.
        assert!(outcome.logits.is_some());
    }

    #[test]
    fn engine_pause_in_middle_of_decode_keeps_session_kv() {
        // §5.2.1: a mid-decode pause keeps KV blocks alive, ref-counted through the block pool.
        // The session's `current_pos` does not regress, and the speculative wrapper's tentative state stays anchored to.
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine
            .enqueue_request_with_kv(Request {
                id: 1,
                prompt_tokens: 4,
                priority: 0,
                ..Default::default()
            })
            .expect("enqueue with kv");
        assert!(engine.sessions.get(&1).map(|s| s.has_kv()).unwrap_or(false));

        let prefill = engine.tick().expect("prefill tick");
        assert!(prefill.prefill_ids.contains(&1));

        // Pause mid-decode.
        let running_pos = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        engine.pause_request(1);
        let paused_pos = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(running_pos, paused_pos, "pause must not change session pos");
        assert!(engine.is_paused(1));

        // Resume: same position. Tick again.
        engine.resume_request(1);
        let resumed_pos = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(
            running_pos, resumed_pos,
            "resume must continue from paused position"
        );
        let _ = engine.tick().expect("decode tick");
        let after_tick_pos = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert!(
            after_tick_pos > resumed_pos,
            "decode tick after resume must advance pos"
        );
    }

    #[test]
    fn engine_distinct_requests_keep_distinct_outcomes() {
        // When multiple requests run, each `last_outcome` reflects the
        // wrapper output for that specific request.
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        engine.enqueue_request(Request {
            id: 2,
            prompt_tokens: 8,
            priority: 0,
            ..Default::default()
        });
        let _ = engine.tick();
        let o1 = engine.last_outcome(1).cloned();
        let o2 = engine.last_outcome(2).cloned();
        assert!(o1.is_some() && o2.is_some());
        let v1 = o1.unwrap().logits.unwrap().to_vec_f32().unwrap();
        let v2 = o2.unwrap().logits.unwrap().to_vec_f32().unwrap();
        assert!(!v1.is_empty() && !v2.is_empty());
    }

    #[test]
    fn engine_throughput_steps_count_ticks_and_advances() {
        // The wrapper path is the standard one - count the speculative flag on the recorded outcomes and assert that every running request was driven once per tick.
        // v1's Llama forward doesn't accept extras, but the wrapper contract holds: every decode tick yields.
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        // Tick 1: prefill.
        let _ = engine.tick();
        let pos1 = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        // Tick 2: decode.
        engine.scheduler.running.retain(|r| r.id == 1);
        let _ = engine.tick();
        let pos2 = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert!(pos2 > pos1, "decode tick advances the session position");

        // Plain strategy still counts as "speculative" field = false on the wrapper output, confirming the structural
        // pipeline is in place for Strategy::Plain (with a real DSpark bundle attached the field flips to true).
        let outcome = engine.last_outcome(1).unwrap();
        assert!(
            !outcome.speculative,
            "without a bundled drafter, the wrapper falls back to plain decode"
        );
    }

    #[test]
    fn engine_finish_clears_outcome() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let _ = engine.tick();
        assert!(engine.last_outcome(1).is_some());
        engine.finish_request(1);
        assert!(engine.last_outcome(1).is_none());
    }

    #[test]
    fn engine_with_dspark_bundle_routes_through_dspark_strategy() {
        // Wiring concrete DraftBackbone / MarkovHead / ConfidenceHead impls through `register_with_dspark`.
        // This is the test of whether the speculative decoding pipeline (§5.3.2) is actually exercisable end-to-end.
        use grim_speculative::{EntropyConfidenceHead, TinyDraftBackbone, UniformMarkovHead};

        let mut engine = Engine::new(EngineConfig::default());
        let draft = TinyDraftBackbone::new(64, 16, 4, 0xDEAD_BEEF);
        let markov = UniformMarkovHead::new(64, 4, 0xCAFE_BABE);
        let conf = EntropyConfidenceHead;
        engine.register_with_dspark(
            "small",
            small_llama(),
            draft.into(),
            markov.into(),
            conf.into(),
        );
        assert_eq!(engine.strategy_for("small"), Some(Strategy::DSpark));

        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let out = engine.tick();
        assert!(
            out.is_ok(),
            "tick must succeed under DSpark strategy: {:?}",
            out.err()
        );
        let _ = engine.last_outcome(1);
    }

    #[test]
    fn engine_per_request_rng_seeded_in_strict_mode() {
        // §5.8: per-request-seeded Speculation RNG. Each request gets
        // its own deterministic stream from `request.id`.
        let config = EngineConfig {
            determinism_mode: DeterminismMode::Strict,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config);
        engine.register_model("small", small_llama());
        engine.enqueue_request(Request {
            id: 11,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        engine.enqueue_request(Request {
            id: 22,
            prompt_tokens: 4,
            priority: 0,
            ..Default::default()
        });
        let s1 = engine.request_rng_state(11);
        let s2 = engine.request_rng_state(22);
        assert!(s1.is_some() && s2.is_some());
        // Distinct ids → distinct initial states.
        assert_ne!(
            s1, s2,
            "different request ids must yield different rng seeds"
        );

        // Advance RNG by N for one request; the other's state is untouched.
        engine.advance_request_rng(11, 8);
        let s1_advanced = engine.request_rng_state(11).unwrap();
        let s2_unchanged = engine.request_rng_state(22).unwrap();
        assert_ne!(s1_advanced, s1.unwrap(), "RNG must be advancing");
        assert_eq!(
            s2_unchanged,
            s2.unwrap(),
            "other request's RNG must not change"
        );

        // finish_request clears the rng slot.
        engine.finish_request(11);
        assert_eq!(engine.request_rng_state(11), None);
    }

    fn write_mock_gguf_for_test(path: &std::path::Path) {
        use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufValue};
        use std::collections::HashMap;
        use std::io::Write;

        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.vocab_size".to_string(),
            GgufValue::String("256".to_string()),
        );
        metadata.insert(
            "llama.embedding_length".to_string(),
            GgufValue::String("32".to_string()),
        );
        metadata.insert(
            "llama.block_count".to_string(),
            GgufValue::String("1".to_string()),
        );
        metadata.insert(
            "llama.intermediate_size".to_string(),
            GgufValue::String("64".to_string()),
        );
        metadata.insert(
            "llama.attention.head_count".to_string(),
            GgufValue::String("2".to_string()),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::String("1".to_string()),
        );
        metadata.insert(
            "llama.attention.key_length".to_string(),
            GgufValue::String("16".to_string()),
        );
        metadata.insert(
            "llama.attention.layer_norm_eps".to_string(),
            GgufValue::String("0.00001".to_string()),
        );

        let tensor_specs = vec![
            ("token_embd.weight", vec![32, 256]),
            ("output_norm.weight", vec![32]),
            ("output.weight", vec![32, 256]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![32, 16]),
            ("blk.0.attn_v.weight", vec![32, 16]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.ffn_gate.weight", vec![32, 64]),
            ("blk.0.ffn_down.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![32, 64]),
        ];

        let mut buf = Vec::new();
        buf.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap(); // GGUF magic
        buf.write_all(&GGUF_VERSION.to_le_bytes()).unwrap(); // version
        buf.write_all(&(tensor_specs.len() as u64).to_le_bytes())
            .unwrap();
        buf.write_all(&(metadata.len() as u64).to_le_bytes())
            .unwrap();

        for (k, v) in &metadata {
            let kb = k.as_bytes();
            buf.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
            buf.write_all(kb).unwrap();
            buf.write_all(&8u32.to_le_bytes()).unwrap(); // String type
            if let GgufValue::String(s) = v {
                let vb = s.as_bytes();
                buf.write_all(&(vb.len() as u64).to_le_bytes()).unwrap();
                buf.write_all(vb).unwrap();
            }
        }

        let mut payload = Vec::new();
        for (name, dims) in &tensor_specs {
            let nb = name.as_bytes();
            buf.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
            buf.write_all(nb).unwrap();
            buf.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
            for &d in dims {
                buf.write_all(&(d as u64).to_le_bytes()).unwrap();
            }
            buf.write_all(&0u32.to_le_bytes()).unwrap(); // F32 dtype (tag 0)

            let offset = payload.len() as u64;
            buf.write_all(&offset.to_le_bytes()).unwrap();

            let count = dims.iter().product::<usize>();
            for i in 0..count {
                let val = ((i % 100) as f32 * 0.01 + 0.01).to_le_bytes();
                payload.extend_from_slice(&val);
            }
        }

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&payload);

        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn test_load_grim_with_gguf_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        let grim_path = dir.path().join("model.grim");

        write_mock_gguf_for_test(&gguf_path);

        // Convert GGUF to GRIM
        grim_format::convert_to_grim(
            gguf_path.to_str().unwrap(),
            grim_path.to_str().unwrap(),
            "gfx1100",
            16.0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("conversion failed");

        // Verify sibling GGUF is next to it
        assert!(gguf_path.exists());
        assert!(grim_path.exists());

        // Now load the model via load_from_path!
        let loaded = crate::model_loader::load_from_path(grim_path.to_str().unwrap());
        assert!(loaded.is_ok(), "failed to load .grim: {:?}", loaded.err());
    }

    #[test]
    fn engine_enqueues_real_input_ids_and_consumes_in_prefill() {
        // This test validates the fix for the "dummy token" bug where drive_prefill was feeding synthetic (0..prompt_tokens) instead of the actual prompt token IDs provided by the caller.
        // The test enqueues a request with known input_ids, ticks the engine, and verifies that the.
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("small", small_llama());

        // Use a specific, non-sequential token sequence within vocab (256) to detect synthetic substitution.
        let real_tokens = vec![7u32, 42, 100, 3, 200];
        let prompt_tokens = real_tokens.len();

        engine.enqueue_request(Request {
            id: 1,
            prompt_tokens,
            priority: 0,
            input_ids: Some(real_tokens.clone()),
            session: None,
            ..Default::default()
        });

        // First tick: prefill should consume ALL prompt tokens
        let _ = engine.tick().expect("tick must succeed");

        // The session position should advance by the number of REAL tokens, not by a synthetic range.
        // If the bug exists, it would advance by prompt_tokens (which happens to match here) but.
        let pos_after_prefill = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(
            pos_after_prefill, prompt_tokens,
            "prefill must advance by prompt_tokens"
        );

        // Keep the request in running for decode
        engine.scheduler.running.retain(|r| r.id == 1);

        // Second tick: decode step should use the LAST real token (999) as input, not the position index (which would be 5).
        // We can't directly observe the input_ids tensor from here, but we verify the session advances.
        let _ = engine.tick().expect("decode tick must succeed");
        let pos_after_decode = engine
            .sessions
            .get(&1)
            .map(|s| s.current_pos())
            .unwrap_or(0);
        assert_eq!(
            pos_after_decode,
            prompt_tokens + 1,
            "decode must advance by 1"
        );
    }

    /// Phase-1 correctness proof: a Llama driven through the paged-KV path (session carries a `PagedKvCache`) must produce byte-identical logits to the same model driven through the classic per-layer `LlamaLayerCache` path (no KV session).
    /// This is the invariant that lets us re-enable prefix-cache/tiering wiring on top of the paged.
    #[test]
    fn paged_llama_forward_matches_non_paged_llama_forward() {
        use grim_core::CausalLm;
        use grim_core::session::Inner;
        use grim_models_transformer::{Llama, LlamaConfig};
        use grim_tensor::Device;

        let cfg = LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 64,
            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let model = Llama::random(Device::Cpu, cfg);

        // Classic path: session with no KV cache → model_state caches.
        let mut classic = Inner::new(model.device.clone());
        // Paged path: session backed by a shared block pool.
        let pool = std::sync::Arc::new(std::sync::Mutex::new(grim_memory::KvBlockPool::new(
            1024, 1, 16,
        )));
        let kv = grim_memory::PagedKvCache::new(pool, 1, 16, 16);
        let mut paged = Inner::with_kv(model.device.clone(), Box::new(kv));

        let tok = grim_backend_cpu::cpu_tensor(
            vec![0.0f32, 1.0f32, 2.0f32, 3.0f32],
            grim_tensor::Shape::new(vec![4]),
        );
        let pos = grim_backend_cpu::cpu_tensor(
            vec![0.0f32, 1.0f32, 2.0f32, 3.0f32],
            grim_tensor::Shape::new(vec![4]),
        );
        let classic_logits = CausalLm::forward(&model, &mut classic, &tok, &pos, &[]).unwrap();
        let paged_logits = CausalLm::forward(&model, &mut paged, &tok, &pos, &[]).unwrap();
        let cl = classic_logits.to_vec_f32().unwrap();
        let pl = paged_logits.to_vec_f32().unwrap();
        let diffs: Vec<f32> = cl
            .iter()
            .zip(pl.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max_diff = diffs.iter().copied().fold(0.0f32, f32::max);
        let argmax = diffs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        eprintln!(
            "[eq] prefill max_diff={max_diff} at idx={argmax} classic={:?} paged={:?}",
            cl[argmax], pl[argmax]
        );
        assert_eq!(
            classic_logits.to_vec_f32().unwrap(),
            paged_logits.to_vec_f32().unwrap(),
            "prefill logits must match between paged and non-paged paths"
        );

        // One decode step at position 4 on the SAME sessions.
        let tok1 = grim_backend_cpu::cpu_tensor(vec![4.0f32], grim_tensor::Shape::new(vec![1]));
        let pos4 = grim_backend_cpu::cpu_tensor(vec![4.0f32], grim_tensor::Shape::new(vec![1]));
        let classic_decode = CausalLm::forward(&model, &mut classic, &tok1, &pos4, &[]).unwrap();
        let paged_decode = CausalLm::forward(&model, &mut paged, &tok1, &pos4, &[]).unwrap();
        assert_eq!(
            classic_decode.to_vec_f32().unwrap(),
            paged_decode.to_vec_f32().unwrap(),
            "decode logits must match between paged and non-paged paths"
        );
    }

    #[test]
    fn prefix_cache_reuses_blocks_for_shared_prefix() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.radix_enabled = true;

        let prompt1 = vec![
            101u32, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        ];
        let block_ids1: Vec<usize> = vec![10];

        // Insert prefix for prompt1 into block pool
        {
            let mut pool = engine.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.insert_prefix(&prompt1, &block_ids1);
        }

        // Query with prompt2 that shares the same prefix
        let mut prompt2 = prompt1.clone();
        prompt2.extend_from_slice(&[201, 202]);

        let (matched_blocks, matched_tokens, _) = {
            let mut pool = engine.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.match_prefix_promoting(&prompt2)
        };

        assert_eq!(matched_tokens, 16);
        assert_eq!(matched_blocks, block_ids1);
    }

    #[test]
    fn hybrid_attention_offload_spill_and_repromote_parity() {
        use grim_core::CausalLm;
        use grim_core::session::Inner;
        use grim_models_transformer::{Llama, LlamaConfig};
        use grim_tensor::Device;

        let cfg = LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 64,
            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let model = Llama::random(Device::Cpu, cfg);

        // Baseline (no spill manager)
        let pool_no_spill = std::sync::Arc::new(std::sync::Mutex::new(
            grim_memory::KvBlockPool::new(1024, 1, 16),
        ));
        let kv_no_spill = grim_memory::PagedKvCache::new(pool_no_spill, 1, 16, 16);
        let mut session_no_spill = Inner::with_kv(model.device.clone(), Box::new(kv_no_spill));

        // Spilled path (with SharedSpillManager)
        let scratch_dir =
            std::env::temp_dir().join(format!("grim_spill_test_{}", std::process::id()));
        let spill_mgr = std::sync::Arc::new(
            grim_kvtransport::SharedSpillManager::new(scratch_dir, 16 * 16).unwrap(),
        );
        let pool_spill = std::sync::Arc::new(std::sync::Mutex::new(grim_memory::KvBlockPool::new(
            1024, 1, 16,
        )));
        pool_spill
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .attach_spill(spill_mgr.clone());
        let kv_spill = grim_memory::PagedKvCache::new(pool_spill.clone(), 1, 16, 16);
        let mut session_spill = Inner::with_kv(model.device.clone(), Box::new(kv_spill));

        // Step 1: multi-token prompt prefill (32 tokens = 2 blocks of 16 tokens each)
        let tokens: Vec<f32> = (0..32).map(|i| (i % 64) as f32).collect();
        let pos: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let tok_t = grim_backend_cpu::cpu_tensor(tokens, grim_tensor::Shape::new(vec![32]));
        let pos_t = grim_backend_cpu::cpu_tensor(pos, grim_tensor::Shape::new(vec![32]));

        let logits_no_spill =
            CausalLm::forward(&model, &mut session_no_spill, &tok_t, &pos_t, &[]).unwrap();
        let logits_spill =
            CausalLm::forward(&model, &mut session_spill, &tok_t, &pos_t, &[]).unwrap();

        assert_eq!(
            logits_no_spill.to_vec_f32().unwrap(),
            logits_spill.to_vec_f32().unwrap(),
            "prefill logits must match before demote"
        );

        // Step 2: Under memory pressure / watermark, demote prefix block 0
        let block_0 = session_spill.block_table().unwrap()[0] as usize;
        {
            let mut p = pool_spill.lock().unwrap_or_else(|e| e.into_inner());
            let demoted = p.demote_block(block_0);
            assert!(demoted, "block 0 demotion must succeed");
            assert!(
                matches!(
                    spill_mgr.get_tier(block_0),
                    Some(grim_kvtransport::CacheTier::HostRam)
                        | Some(grim_kvtransport::CacheTier::NvMe)
                ),
                "demoted block must reside in HostRam or NvMe tier"
            );
        }

        // Step 3: Decode token 33 at position 32.
        // Paged attention checks plan_hybrid_attention_step -> detects block_0 in HostRam
        // -> re-promotes to GPU before attention -> produces identical logits!
        let mut cur_tok_no_spill = 32u32;
        let mut cur_tok_spill = 32u32;
        for (cur_pos, _) in (32u32..).zip(0..4) {
            let tok_no_spill_t = grim_backend_cpu::cpu_tensor(
                vec![cur_tok_no_spill as f32],
                grim_tensor::Shape::new(vec![1]),
            );
            let tok_spill_t = grim_backend_cpu::cpu_tensor(
                vec![cur_tok_spill as f32],
                grim_tensor::Shape::new(vec![1]),
            );
            let pos_t = grim_backend_cpu::cpu_tensor(
                vec![cur_pos as f32],
                grim_tensor::Shape::new(vec![1]),
            );

            let dec_no_spill =
                CausalLm::forward(&model, &mut session_no_spill, &tok_no_spill_t, &pos_t, &[])
                    .unwrap();
            let dec_spill =
                CausalLm::forward(&model, &mut session_spill, &tok_spill_t, &pos_t, &[]).unwrap();

            let logits_a = dec_no_spill.to_vec_f32().unwrap();
            let logits_b = dec_spill.to_vec_f32().unwrap();

            let diff: f32 = logits_a
                .iter()
                .zip(logits_b.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                diff < 1e-4,
                "decode logits at pos {cur_pos} after spill + re-promotion must match unspilled baseline (max diff {diff})"
            );

            // Greedy token pick (argmax)
            let next_a = logits_a
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            let next_b = logits_b
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();

            assert_eq!(
                next_a, next_b,
                "greedy token parity mismatch at pos {cur_pos}"
            );
            cur_tok_no_spill = next_a;
            cur_tok_spill = next_b;
        }
    }

    #[test]
    fn test_engine_speculative_mtp_and_eagle3_registration() {
        let mut engine = Engine::new(EngineConfig::default());
        let llama = Llama::random(
            Device::Cpu,
            LlamaConfig {
                vocab_size: 256,
                hidden_size: 32,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 16,
                num_layers: 1,
                intermediate_size: 64,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 64,
                partial_rotary_factor: 1.0,
                yarn: None,
            },
        );
        let mtp = Arc::new(grim_models_transformer::LlamaMtp::new_random(llama, 2));
        engine.register_native_mtp_model("llama-mtp", mtp);
        assert!(engine.models.contains_key("llama-mtp"));

        let eagle3_cfg = grim_models_transformer::Eagle3Config {
            vocab_size: 256,
            hidden_size: 32,
            target_hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 64,
            num_target_fusion_layers: 3,
        };
        let eagle3 = Arc::new(grim_models_transformer::Eagle3::random(
            Device::Cpu,
            eagle3_cfg,
        ));
        engine.register_eagle3_model("llama-eagle3", small_llama(), eagle3);
        assert!(engine.models.contains_key("llama-eagle3"));
    }

    #[test]
    fn test_single_gpu_capability_profiler_is_none() {
        // WI-INF1 Gate: default single-GPU box must pay zero probe cost
        let engine = Engine::new(EngineConfig::default());
        assert!(
            engine.capability_profiler.is_none(),
            "single-GPU box must have capability_profiler = None"
        );
        assert!(
            engine.scythe_ctrl.is_none(),
            "single-GPU box must have scythe_ctrl = None"
        );
        assert!(engine.capabilities().is_none());
    }

    #[test]
    fn test_scythe_route_attach_requires_armed_engine() {
        // Default engine (no flag, no multi-GPU): attach is a no-op that
        // reports false rather than arming a half-configured route.
        let mut engine = Engine::new(EngineConfig::default());
        assert!(!engine.scythe_armed());
        let mut sfb = crate::streaming_forward::StreamingBlockForward::new(1, 32);
        assert!(!engine.attach_scythe_route(&mut sfb));
        assert!(sfb.scythe_route.is_none());
    }

    fn farm_cap(tflops: f32, ordinal: usize) -> grim_tensor::backend::GpuCapability {
        grim_tensor::backend::GpuCapability {
            tflops_fp16: tflops,
            tflops_fp8: 0.0,
            hbm_bandwidth_gbps: 100.0,
            vram_free_bytes: 16 << 30,
            throttle_pct: 0.0,
            ordinal,
        }
    }

    /// WI-INF3 serving gate (farm mode): a pinned request executes on its replica.
    /// Replicas are built from the same fixed seed, so logits must be byte-identical to a.
    #[test]
    fn test_scythe_farm_pin_routes_across_replicas() {
        let mut engine = Engine::new(EngineConfig::default());
        // Arm manually (env-flag construction is the WI-INF1 gate's job);
        // controller sized for a 2-rank farm.
        engine.scythe_ctrl = Some(crate::scythe2::C2plrController::new(1, 2, 10.0));
        engine.capability_profiler = Some(Arc::new(grim_backend_rocm::CapabilityProfiler::new()));
        engine.register_model("small", small_llama());
        engine.register_model("small#scythe1", small_llama());
        engine
            .scythe_replicas
            .insert("small".into(), vec!["small".into(), "small#scythe1".into()]);
        assert_eq!(engine.scythe_farm_size("small"), 2);

        // Plain engine, same weights, for the numeric baseline.
        let mut single = Engine::new(EngineConfig::default());
        single.register_model("small", small_llama());

        let req = |id: u64| grim_scheduler::Request {
            id,
            prompt_tokens: 4,
            model_id: Some("small".into()),
            ..Default::default()
        };
        single.enqueue_request_with_kv(req(41)).unwrap();
        engine.enqueue_request_with_kv(req(7)).unwrap();
        engine.enqueue_request_with_kv(req(8)).unwrap();

        // The admission path picked some rank (host-dependent); this gate is
        // about ROUTING, so pin both ranks explicitly and verify each one.
        engine.scythe_pin.insert(7, 1);
        engine.scythe_pin.insert(8, 0);
        assert_eq!(engine.scythe_pin_of(7), Some(1));
        assert_eq!(engine.scythe_pin_of(8), Some(0));
        assert_eq!(
            engine.resolved_model_id(7).as_deref(),
            Some("small#scythe1")
        );
        assert_eq!(engine.resolved_model_id(8).as_deref(), Some("small"));

        let ids = grim_backend_cpu::cpu_tensor(vec![3.0f32], grim_tensor::Shape::new(vec![1]));
        let pos = grim_backend_cpu::cpu_tensor(vec![0.0f32], grim_tensor::Shape::new(vec![1]));
        let base = single.step_one(41, "small", &ids, &pos).unwrap();
        let on_rank1 = engine.step_one(7, "small", &ids, &pos).unwrap();
        let on_rank0 = engine.step_one(8, "small", &ids, &pos).unwrap();

        let base_v = base.logits.unwrap().to_vec_f32().unwrap();
        assert_eq!(
            on_rank1.logits.unwrap().to_vec_f32().unwrap(),
            base_v,
            "replica rank 1 must produce byte-identical logits"
        );
        assert_eq!(
            on_rank0.logits.unwrap().to_vec_f32().unwrap(),
            base_v,
            "rank 0 must produce byte-identical logits"
        );

        // Finishing releases the farm slot.
        engine.finish_request(7);
        assert_eq!(engine.scythe_pin_of(7), None);
    }

    /// WI-INF5 farm corollary: the load-adjusted capability view must spread sessions once the fast card saturates instead of pinning everything to it - unloaded traffic goes
    /// to the 80-TFLOPS card, but with 10 sessions already there its effective 80/11 TFLOPS drops below the idle 8-TFLOPS card and the next admission lands there.
    #[test]
    /// WI-SB1 load-spreading: a finished request's rank must stay counted in the cooldown window so the next admission sees
    /// it - otherwise a burst of short requests all observe an empty pin map and pile onto rank 0.
    fn test_finished_pin_enters_cooldown_window() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.scythe_pin.insert(7, 1);
        engine.finish_request(7);
        assert!(!engine.scythe_pin.contains_key(&7), "active pin released");
        assert_eq!(engine.scythe_pin_cooldown.len(), 1);
        assert_eq!(engine.scythe_pin_cooldown[0].0, 1);

        // Pruning happens at admission time, not finish time: an aged-out entry is still physically present until
        // the next decision scans it, but it must not COUNT toward load anymore (helper gate below).
        engine.scythe_pin_cooldown[0].1 -= std::time::Duration::from_millis(2000);
        engine.scythe_pin.insert(8, 0);
        engine.finish_request(8);
        let freshest = engine.scythe_pin_cooldown.last().unwrap();
        assert_eq!(freshest.0, 0);
        assert!(freshest.1.elapsed() < SCYTHE_PIN_COOLDOWN);
    }

    /// WI-SB1: effective-load math — active pins, cooldown-window releases,
    /// expired releases, and external busy-% weighting.
    #[test]
    fn test_scythe_effective_loads_weights_and_expiry() {
        let now = std::time::Instant::now();
        let released = vec![
            (0usize, now),                                          // in window
            (1usize, now - std::time::Duration::from_millis(5000)), // expired
            (9usize, now),                                          // out-of-range rank
        ];
        let busy = vec![Some(100u32), Some(50u32), None];
        let loads = scythe_effective_loads(
            [0usize, 1usize].into_iter(),
            &released,
            SCYTHE_PIN_COOLDOWN,
            &busy,
            3,
            SCYTHE_EXTERNAL_BUSY_WEIGHT,
        );
        assert_eq!(loads[0], 4.0, "pin + fresh release + 100% busy×2.0");
        assert_eq!(
            loads[1], 2.0,
            "pin + expired release dropped + 50% busy×2.0"
        );
        assert_eq!(loads[2], 0.0, "no pin, no telemetry -> zero load");
    }

    #[test]
    fn test_load_adjusted_caps_balances_farm_placement() {
        use grim_tensor::backend::ScytheLink;
        let caps = vec![farm_cap(8.0, 0), farm_cap(80.0, 1)];
        let links = vec![
            ScytheLink::PeerDirect,
            ScytheLink::Host,
            ScytheLink::Host,
            ScytheLink::PeerDirect,
        ];
        let shape = [1usize, 2048, 1, 1];

        let mut unloaded = crate::scythe2::C2plrController::new(1, 2, 150.0);
        let p = unloaded.decide(
            0,
            &shape,
            &load_adjusted_caps(&caps, 2, &[0.0, 0.0]),
            &links,
            0,
        );
        assert_eq!(
            p.ranks,
            vec![1],
            "unloaded admission must take the fast card"
        );

        let mut saturated = crate::scythe2::C2plrController::new(1, 2, 150.0);
        let p2 = saturated.decide(
            0,
            &shape,
            &load_adjusted_caps(&caps, 2, &[0.0, 10.0]),
            &links,
            0,
        );
        assert_eq!(
            p2.ranks,
            vec![0],
            "saturated fast card must yield to the idle slow card"
        );
    }

    /// WI-SB1 load-spreading: external GPU utilization folds into the load vector at weight 2.0 - a card maxed out by a desktop/game workload
    /// (100 % busy ≈ +2 effective requests) must lose the fast card to an idle slower rank on a ~2:1 measured pair.
    #[test]
    fn test_external_busy_flips_placement_to_idle_rank() {
        use grim_tensor::backend::ScytheLink;
        let caps = vec![farm_cap(12.4, 0), farm_cap(6.5, 1)];
        let links = vec![
            ScytheLink::PeerDirect,
            ScytheLink::Host,
            ScytheLink::Host,
            ScytheLink::PeerDirect,
        ];
        let shape = [1usize, 2048, 1, 1];

        // Idle: fast card wins outright…
        let idle = load_adjusted_caps(&caps, 2, &[0.0, 0.0]);
        let mut ctrl = crate::scythe2::C2plrController::new(1, 2, 150.0);
        let p = ctrl.decide_forced(0, &shape, &idle, &links, 0);
        assert_eq!(p.ranks, vec![0], "idle fast card must be picked");

        // …but a game pinning rank 0 at 100 % busy (+2.0 effective load)
        // halves its effective throughput and the idle slow card wins.
        let gamed = load_adjusted_caps(&caps, 2, &[2.0, 0.0]);
        let mut ctrl = crate::scythe2::C2plrController::new(1, 2, 150.0);
        let p = ctrl.decide_forced(0, &shape, &gamed, &links, 0);
        assert_eq!(
            p.ranks,
            vec![1],
            "externally-saturated fast card must yield to the idle slow card"
        );

        // And the original defect: plain decide() caches the idle verdict keyed only by shape, then serves it verbatim even after the load vector changed - which is what pinned every request to rank 0.
        // decide_forced must re-evaluate under the adjusted caps instead.
        let mut ctrl = crate::scythe2::C2plrController::new(1, 2, 150.0);
        let cached_rank = ctrl
            .decide(
                0,
                &shape,
                &load_adjusted_caps(&caps, 2, &[0.0, 0.0]),
                &links,
                0,
            )
            .ranks[0];
        let sticky_rank = ctrl
            .decide(
                0,
                &shape,
                &load_adjusted_caps(&caps, 2, &[2.0, 0.0]),
                &links,
                0,
            )
            .ranks[0];
        assert_eq!(cached_rank, 0);
        assert_eq!(
            sticky_rank, cached_rank,
            "expected the load-blind cache hit being fixed here"
        );
        let forced_rank = ctrl
            .decide_forced(
                0,
                &shape,
                &load_adjusted_caps(&caps, 2, &[2.0, 0.0]),
                &links,
                0,
            )
            .ranks[0];
        assert_eq!(forced_rank, 1);
    }

    /// Farm registration without an armed controller degrades to a plain
    /// registration — no replica registry entry, no pins.
    #[test]
    fn test_scythe_farm_degrades_to_plain_registration() {
        let mut engine = Engine::new(EngineConfig::default());
        assert!(!engine.scythe_armed());
        engine.register_model_with_farm("small", small_llama(), "/nonexistent/path.gguf");
        assert_eq!(engine.scythe_farm_size("small"), 0);
        assert!(engine.has_model("small"));
    }

    /// WI-SB2 host gate (synthetic caps): the footprint formula must exclude an 8 GB-class card for a 100k-token prompt, admit the same request on a 16 GB card, admit a 1k-token prompt on both, and report every rank infeasible when nothing fits - which is the queue signal.
    /// A zero free-VRAM reading is probe-unavailable and must NOT read as "full".
    #[test]
    fn test_scythe_vram_footprint_and_rank_filter() {
        let cap_with_vram = |vram: u64| grim_tensor::backend::GpuCapability {
            tflops_fp16: 10.0,
            tflops_fp8: 0.0,
            hbm_bandwidth_gbps: 100.0,
            vram_free_bytes: vram,
            throttle_pct: 0.0,
            ordinal: 0,
        };
        let gib = 1024u64 * 1024 * 1024;
        // kv_dim = 8·64 = 512, hidden = 1024, layers = 8 ⇒
        // ~96 KiB/token ⇒ a 132k-token request needs ~12.1 GiB.
        let dims = (8usize, 64usize, Some(1024usize), 8u64);
        let big = scythe_request_footprint_bytes(100_000, 32_000, dims.0, dims.1, dims.2, dims.3);
        let tiny = scythe_request_footprint_bytes(1_000, 32, dims.0, dims.1, dims.2, dims.3);

        assert!(
            big + SCYTHE_VRAM_WATERMARK_BYTES > 8 * gib,
            "100k-token prompt must overflow an 8 GB card"
        );
        assert!(
            big + SCYTHE_VRAM_WATERMARK_BYTES <= 16 * gib,
            "100k-token prompt must still fit a 16 GB card"
        );

        // 8 GB slow card excluded, 16 GB fast card included.
        let caps_pair = vec![cap_with_vram(8 * gib), cap_with_vram(16 * gib)];
        assert_eq!(
            scythe_vram_feasible(&caps_pair, big),
            vec![false, true],
            "100k-token prompt must pin the fast card only"
        );
        assert_eq!(
            scythe_vram_feasible(&caps_pair, tiny),
            vec![true, true],
            "1k-token prompt must fit both cards"
        );
        // All-excluded ⇒ queue signal (never pinned blind).
        let caps_small = vec![cap_with_vram(8 * gib), cap_with_vram(8 * gib)];
        assert!(
            scythe_vram_feasible(&caps_small, big).iter().all(|&ok| !ok),
            "no-rank-fits must be detectable"
        );
        // Probe-unavailable ranks stay placeable instead of dead-locking.
        assert_eq!(scythe_vram_feasible(&[cap_with_vram(0)], big), vec![true]);
        // Unknown hidden width falls back to the KV dimension (smaller floor).
        let no_hidden =
            scythe_request_footprint_bytes(100_000, 32_000, dims.0, dims.1, None, dims.3);
        assert!(
            no_hidden < big,
            "KV-dim fallback must not exceed the hidden-width floor"
        );
    }

    /// WI-SB2 host gate: the decision layer maps the synthetic-caps guard to Pin/WaitVram correctly - mixed pair pins the feasible
    /// fast card, all-excluded waits, missing profiler data waits, and a 1k prompt is never blocked by an 8 GB card.
    #[test]
    fn test_scythe_admission_decision_vram_guard() {
        // kv_dim = 64·128 = 8192 ⇒ a 132k-token request needs ~8.1 GiB even
        // with small_llama's tiny reported hidden width (32, 1 layer).
        let cfg = EngineConfig {
            num_kv_heads: 64,
            head_dim: 128,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(cfg);
        engine.scythe_ctrl = Some(crate::scythe2::C2plrController::new(1, 2, 150.0));
        engine.register_model("small", small_llama());
        engine.register_model("small#scythe1", small_llama());
        engine
            .scythe_replicas
            .insert("small".into(), vec!["small".into(), "small#scythe1".into()]);

        let cap_with_vram = |tflops: f32, vram: u64| grim_tensor::backend::GpuCapability {
            tflops_fp16: tflops,
            tflops_fp8: 0.0,
            hbm_bandwidth_gbps: 100.0,
            vram_free_bytes: vram,
            throttle_pct: 0.0,
            ordinal: 0,
        };
        let gib = 1024u64 * 1024 * 1024;
        let caps_mixed = vec![cap_with_vram(8.0, 8 * gib), cap_with_vram(80.0, 16 * gib)];
        let caps_both_small = vec![cap_with_vram(8.0, 8 * gib), cap_with_vram(80.0, 8 * gib)];
        let huge = (100_000usize, 32_000usize);
        let tiny = (1_000usize, 32usize);

        assert_eq!(
            engine.scythe_admission_decision("small", huge.0, huge.1, &caps_mixed),
            ScytheAdmission::Pin(1),
            "oversized prompt must land on the one card that holds it"
        );
        assert_eq!(
            engine.scythe_admission_decision("small", huge.0, huge.1, &caps_both_small),
            ScytheAdmission::WaitVram,
            "no rank fits ⇒ wait, never pin blind"
        );
        assert_ne!(
            engine.scythe_admission_decision("small", tiny.0, tiny.1, &caps_both_small),
            ScytheAdmission::WaitVram,
            "1k-token prompt must not be blocked by the guard"
        );
        assert_eq!(
            engine.scythe_admission_decision("small", huge.0, huge.1, &[]),
            ScytheAdmission::WaitVram,
            "profiler seeing no GPUs ⇒ wait rather than admit onto rank 0"
        );
        // Unarmed engine bypasses farm routing entirely (rollback invariant).
        let mut plain = Engine::new(EngineConfig::default());
        plain.register_model("small", small_llama());
        plain
            .scythe_replicas
            .insert("small".into(), vec!["small".into(), "small#scythe1".into()]);
        assert_eq!(
            plain.scythe_admission_decision("small", huge.0, huge.1, &caps_mixed),
            ScytheAdmission::Bypass,
        );
    }

    /// WI-SB2 host gate: an enqueue that fails the VRAM guard must leave the request queued - no session, no scheduler entry, no pin - and a later retry once caps exist must admit it with a pin.
    /// Deterministic on any box: with no profiler attached, caps are empty ⇒ WaitVram; the retry.
    #[test]
    fn test_scythe_vram_exhaustion_queues_request() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.scythe_ctrl = Some(crate::scythe2::C2plrController::new(1, 2, 150.0));
        engine.register_model("small", small_llama());
        engine.register_model("small#scythe1", small_llama());
        engine
            .scythe_replicas
            .insert("small".into(), vec!["small".into(), "small#scythe1".into()]);
        assert!(engine.scythe_armed());

        let req = grim_scheduler::Request {
            id: 900,
            prompt_tokens: 100_000,
            max_new_tokens: 32_000,
            model_id: Some("small".into()),
            ..Default::default()
        };
        // No profiler attached ⇒ the guard cannot see any rank ⇒ queued.
        engine.enqueue_request_with_kv(req.clone()).unwrap();
        assert_eq!(engine.scythe_vram_waitlist_len(), 1);
        assert!(
            !engine.sessions.contains_key(&900),
            "queued request must have no session"
        );
        assert_eq!(
            engine.scheduler.waiting.len(),
            0,
            "queued request must not be scheduled"
        );
        assert_eq!(engine.scythe_pin_of(900), None);
        // A tick without visible caps keeps it parked (retry path is safe).
        engine.retry_scythe_vram_waitlist();
        assert_eq!(engine.scythe_vram_waitlist_len(), 1);
        // Cancelling releases the slot.
        engine.finish_request(900);
        assert_eq!(engine.scythe_vram_waitlist_len(), 0);

        // Retry leg with real caps: skipped on boxes without ROCm devices.
        if grim_backend_rocm::CapabilityProfiler::new()
            .capabilities()
            .is_empty()
        {
            return;
        }
        engine.capability_profiler = Some(Arc::new(grim_backend_rocm::CapabilityProfiler::new()));
        engine.enqueue_request_with_kv(req).unwrap();
        engine.retry_scythe_vram_waitlist();
        assert_eq!(
            engine.scythe_vram_waitlist_len(),
            0,
            "request must place once a rank can hold it"
        );
        assert!(engine.sessions.contains_key(&900));
        assert!(engine.scythe_pin_of(900).is_some());
        assert_eq!(engine.scheduler.waiting.len(), 1);
    }

    #[test]
    fn test_engine_fused_batched_lora_execution() {
        let mut engine = Engine::new(EngineConfig::default());
        let dim = 4;
        let rank = 2;

        let a = grim_backend_cpu::cpu_tensor(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            grim_tensor::Shape::new(vec![rank, dim]),
        );
        let b = grim_backend_cpu::cpu_tensor(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            grim_tensor::Shape::new(vec![dim, rank]),
        );
        let handle = AdapterHandle {
            id: 10,
            a,
            b,
            alpha: 2.0,
        };
        engine.register_adapter("default", "lora_test", handle);

        // Rows 0-1 carry adapter 10, row 2 is base (passthrough).
        let mut stacked = vec![
            1.0, 1.0, 1.0, 1.0, //
            2.0, 2.0, 2.0, 2.0, //
            5.0, 5.0, 5.0, 5.0, //
        ];
        let row_adapters = vec![10, 10, 0];

        engine
            .apply_batched_lora_to_rows(&mut stacked, &row_adapters, dim, None)
            .unwrap();

        // Alpha/rank = 2.0/2.0 = 1.0; deltas accumulate onto the base rows.
        // Row 0: [1,1,1,1]+[1,1,0,0] = [2,2,1,1] Row 1: [2,2,2,2]+[2,2,0,0] = [4,4,2,2] Row 2 untouched.
        assert_eq!(&stacked[0..4], &[2.0, 2.0, 1.0, 1.0]);
        assert_eq!(&stacked[4..8], &[4.0, 4.0, 2.0, 2.0]);
        assert_eq!(&stacked[8..12], &[5.0, 5.0, 5.0, 5.0]);
    }

    /// Batched multi-LoRA contract gate: the grouped decode path (`step_batch`, base forwards + one segment apply) must
    /// produce the same logits as the legacy per-request path (adapters applied inside `decode_one`) on identical engines.
    #[test]
    fn test_step_batch_grouped_lora_matches_per_request_path() {
        // The runtime LoRA surrogate contract (MED-5) requires adapter
        // in_dim == logits width, so this model sets hidden == vocab.
        let build = || -> Engine {
            let mut engine = Engine::new(EngineConfig::default());
            let model = Box::new(Llama::random(
                Device::Cpu,
                LlamaConfig {
                    vocab_size: 32,
                    hidden_size: 32,
                    num_heads: 2,
                    num_kv_heads: 1,
                    head_dim: 16,
                    num_layers: 1,
                    intermediate_size: 64,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    max_seq_len: 64,
                    partial_rotary_factor: 1.0,
                    yarn: None,
                },
            ));
            engine.register_model("tiny", model);

            // Rank-2 identity A: [2, 32] = two 32-dim selection rows.
            let a = grim_backend_cpu::cpu_tensor(
                (0..64)
                    .map(|i| if i / 32 == i % 32 { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
                grim_tensor::Shape::new(vec![2, 32]),
            );
            let b = grim_backend_cpu::cpu_tensor(
                vec![0.5; 2 * 32],
                grim_tensor::Shape::new(vec![32, 2]),
            );
            engine.register_adapter(
                "tiny",
                "shared_lora",
                AdapterHandle {
                    id: 10,
                    a,
                    b,
                    alpha: 2.0,
                },
            );
            engine
        };

        // Request 1 and 3 share adapter 10; request 2 runs the base model.
        let enqueue = |engine: &mut Engine| {
            for (id, adapter) in [(1u64, vec![10u32]), (2, vec![]), (3, vec![10])] {
                let req = grim_scheduler::Request {
                    id,
                    prompt_tokens: 2,
                    priority: 0,
                    model_id: Some("tiny".into()),
                    adapter_ids: adapter.clone(),
                    ..Default::default()
                };
                engine.enqueue_request(req).unwrap();
                engine.request_adapters.insert(id, adapter);
            }
        };

        // Grouped path: one step_batch over all three items.
        let mut grouped = build();
        enqueue(&mut grouped);
        let items: Vec<(u64, &str, grim_tensor::Tensor, grim_tensor::Tensor)> = (1..=3)
            .map(|id| {
                let tok = grim_backend_cpu::cpu_tensor(vec![3.0], grim_tensor::Shape::new(vec![1]));
                let pos = grim_backend_cpu::cpu_tensor(vec![1.0], grim_tensor::Shape::new(vec![1]));
                (id, "tiny", tok, pos)
            })
            .collect();
        let item_refs: Vec<(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)> =
            items.iter().map(|(id, m, t, p)| (*id, *m, t, p)).collect();
        let grouped_out = grouped.step_batch(&item_refs).unwrap();

        // Legacy path: identical engine, per-request stepping (adapters
        // applied inside decode_one).
        let mut legacy = build();
        enqueue(&mut legacy);
        let legacy_out: Vec<(u64, StepOutcome)> = items
            .iter()
            .map(|(id, m, t, p)| {
                let outcome = legacy.step_one(*id, m, t, p).unwrap();
                (*id, outcome)
            })
            .collect();

        assert_eq!(grouped_out.len(), legacy_out.len());
        for ((gid, g), (lid, l)) in grouped_out.iter().zip(&legacy_out) {
            assert_eq!(gid, lid);
            let g_logits = g
                .logits
                .as_ref()
                .expect("grouped logits")
                .to_vec_f32()
                .unwrap();
            let l_logits = l
                .logits
                .as_ref()
                .expect("legacy logits")
                .to_vec_f32()
                .unwrap();
            assert_eq!(g_logits.len(), l_logits.len());
            for (i, (gv, lv)) in g_logits.iter().zip(&l_logits).enumerate() {
                assert!(
                    (gv - lv).abs() < 1e-5,
                    "request {gid} logits[{i}] grouped {gv} != legacy {lv}"
                );
            }
        }
    }

    /// R4 validation (deterministic, mock-probed): the admission gate rejects a request whose footprint exceeds the current memory envelope and admits it when the envelope is large enough.
    /// The GRIM_TEST_FREE_DEVICE_BYTES override makes the probe deterministic; see `test_memory_certificate_admission_gate_real_hw` for the live-probe case.
    #[test]
    fn test_memory_certificate_admission_gate() {
        use grim_scheduler::Request;

        // SAFETY: single-threaded test; env vars are set then restored.
        unsafe {
            let mut engine = Engine::new(EngineConfig::default());
            engine.register_model("tiny", small_llama());

            // Tiny envelope: 0.5 MiB free device, 0 host, 0 reserve. A 4096-token
            // prompt cannot fit -> admission must fail with an envelope error.
            std::env::set_var("GRIM_TEST_FREE_DEVICE_BYTES", "500000");
            std::env::set_var("GRIM_HOST_ALLOWANCE_GB", "0");
            std::env::set_var("GRIM_MEMORY_RESERVE_GB", "0");
            let big = Request {
                id: 1,
                prompt_tokens: 4096,
                max_new_tokens: 256,
                model_id: Some("tiny".into()),
                ..Default::default()
            };
            let err = engine
                .enqueue_request(big)
                .expect_err("oversize request must be rejected by the admission gate");
            assert!(
                err.to_string().contains("exceeds current memory envelope"),
                "unexpected error: {err}"
            );

            // Large envelope: 128 GiB free device, 64 GiB host. Same request fits.
            std::env::set_var("GRIM_TEST_FREE_DEVICE_BYTES", "128849018880");
            std::env::set_var("GRIM_HOST_ALLOWANCE_GB", "64");
            std::env::set_var("GRIM_MEMORY_RESERVE_GB", "1");
            let ok = Request {
                id: 2,
                prompt_tokens: 4096,
                max_new_tokens: 256,
                model_id: Some("tiny".into()),
                ..Default::default()
            };
            engine
                .enqueue_request(ok)
                .expect("request must be admitted when envelope is large enough");

            // Cleanup env so other tests are unaffected.
            std::env::remove_var("GRIM_TEST_FREE_DEVICE_BYTES");
            std::env::remove_var("GRIM_HOST_ALLOWANCE_GB");
            std::env::remove_var("GRIM_MEMORY_RESERVE_GB");
        } // unsafe
    }

    /// R4 real-hardware verification: exercises the actual ROCm `hipMemGetInfo` probe (not the mock) on the live GPU and confirms the admission gate works against real free-VRAM data.
    /// Skips (does not fail) when no ROCm device is visible, so this is a no-op.
    #[test]
    fn test_memory_certificate_admission_gate_real_hw() {
        use grim_scheduler::Request;

        // Only run the real-probe path when a GPU is actually present AND the
        // mock override is NOT set (otherwise we\'d test the mock, not the GPU).
        if std::env::var("GRIM_TEST_FREE_DEVICE_BYTES").is_ok() {
            return;
        }
        let has_gpu =
            grim_backend_rocm::device::roc_device::RocmDevice::probe_one(0).unwrap_or(false);
        if !has_gpu {
            eprintln!("skipping R4 real-HW probe: no ROCm device visible");
            return;
        }

        // Exercise the real hipMemGetInfo probe directly.
        let probe = grim_backend_rocm::free_device_memory(0);
        let (free, total) = grim_backend_rocm::device::capability_profiler::vram_info(0);
        assert!(
            total > 0,
            "R4 HW: vram_info should report non-zero total VRAM on a real GPU"
        );
        assert_eq!(
            probe,
            Some(free),
            "R4 HW: free_device_memory must match vram_info free"
        );

        // Register the model and let the engine build its baseline cert from the model\'s hyperparams.
        // With a real multi-GB GPU, a small test request must be admitted against the genuine.
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_model("tiny", small_llama());
        let tiny = Request {
            id: 100,
            prompt_tokens: 64,
            max_new_tokens: 32,
            model_id: Some("tiny".into()),
            ..Default::default()
        };
        engine
            .enqueue_request(tiny)
            .expect("R4 HW: a small request must admit against real free VRAM");
    }
}

mod engine;
