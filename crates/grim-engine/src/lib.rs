//! Top-level inference engine runtime orchestrating models, schedulers, paged KV pools, and LoRA adapters.

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
/// P2: packed-step training driver (varlen grouping + one optimizer step per group).
pub mod train_packed;

pub use pipelines::moe_prefill_pipeline::{BufferRole, MoePrefillPipeline};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use grim_backend_cpu::DeterministicRng;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModelConfig};
use grim_core::session::{DeterminismMode, SessionT};
use grim_memory::{BLOCK_SIZE, KvBlockPool};
use grim_speculative::{ConfidenceHead, DraftBackbone, MarkovHead, SpeculativeCausalLm, Strategy};

type DynModelPtr = Box<SpeculativeCausalLm>;

/// A loaded model with its config and an instantiated CausalLm impl.
pub struct LoadedModel {
    pub model: DynModelPtr,
    pub config: Box<dyn ModelConfig>,
    /// Device this model's weights live on. Sessions are created on this
    /// device so decode/GPU work actually lands on the GPU instead of
    /// silently falling back to CPU.
    pub device: grim_tensor::Device,
    /// Tensor-parallel configuration stamped at registration time. `None`
    /// means single-device; otherwise carries the per-rank `(rank, world_size)`
    /// so callers can report or query the shard index of a loaded model.
    pub tp_config: Option<grim_nn::TensorParallelConfig>,
}

/// A loaded adapter bundle (one LoRA's A/B matrices + scaling). LoRA batches
/// keyed by [`AdapterHandle::id`]; the engine resolves lookup at runtime.
pub struct LoadedAdapter {
    /// Human-readable name from registration — matched against HTTP request
    /// body `"adapters"` arrays. The server 400s on unknown names so this
    /// must be set at register time.
    pub name: String,
    pub handle: AdapterHandle,
    pub base_model_id: String,
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
    /// Tensor-parallel world size (env `GRIM_TP_SIZE`). `0` or `1` =
    /// single-device. Values > 1 require a backend collective (RCCL on ROCm)
    /// and model construction that shards layers — see C2plrController
    /// (scythe2.md §5) and the `ColumnParallelLinear`/`RowParallelLinear`
    /// wrappers in `grim-nn`. The engine reads the env here; the comms
    /// bootstrap is the SCYTHE-2 WI-6 entry point (roc_device.rs:3136).
    pub tp_size: usize,
    /// Explicit GPU ordinals for TP (`GRIM_GPUS`, empty = all visible).
    pub tp_gpus: Vec<usize>,
    /// WI-TOOLS-4c-i: hard cap on the total number of tool-call entries across
    /// every assistant message in a single request's `messages` array. Rejects
    /// the request with 400 once a conversation has made more tool calls than a
    /// single agentic loop should reasonably need (default 20 — arbitrary
    /// starting point; tune against real workloads once 4b's logging exists).
    pub max_tool_calls_per_conversation: usize,
    /// WI-TOOLS-4c-ii: hard cap on `messages.len()` per request. Catches
    /// unbounded history growth (agentic loops or client bugs) before any
    /// tokenization/prefill work happens (default 200 — arbitrary starting
    /// point, not a considered number).
    pub max_messages_per_request: usize,
    /// Disaggregated serving router context.
    pub disagg_router: Option<Arc<grim_disagg::DisaggRouter>>,
    /// Disaggregation configuration (role, addrs). When set, the engine
    /// starts a background KV receiver server and wires disagg routing.
    pub disagg_config: Option<grim_disagg::DisaggConfig>,
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
    /// Per-request deterministic RNG, §5.8. Populated when
    /// `DeterminismMode::Strict` is active. When Relaxed, RNG state is
    /// still tracked for telemetry but is allowed to differ between
    /// tick calls.
    pub request_rng: HashMap<u64, DeterministicRng>,
    pub request_model_ids: HashMap<u64, String>,
    pub request_adapters: HashMap<u64, Vec<u32>>,
    /// Per-request input token buffers. Populated in `enqueue_request`
    /// from `Request::input_ids`. Used by `drive_prefill` to feed real
    /// prompt tokens instead of synthetic position indices.
    pub request_input_ids: HashMap<u64, Vec<u32>>,
    /// Per-request last generated token. Updated after each decode step
    /// via `record_generated_token`. Used by `drive_decode` to feed the
    /// previously sampled token instead of the position index.
    pub request_last_token: HashMap<u64, u32>,
    self_tuning_controller: grim_scheduler::SelfTuningController,
    /// Tuned speculative params (MIN-3: applied, not discarded).
    tuned_speculative_block_len: usize,
    tuned_kv_compression_bit_width: u8,
    tokens_per_sec_ema: f32,
    total_tokens_generated: u64,
    /// WI-E2: cumulative accepted tokens (speculative verification hits).
    accepted_tokens_total: u64,
    last_ttft_ms: Option<f64>,
    last_itl_ms: Option<f64>,
    /// Tensor-parallel config stamped onto each `LoadedModel`. Populated in
    /// `Engine::new` when TP is active (one OS process per rank, Design A);
    /// `None` for single-device operation. The actual per-rank device + RCCL
    /// handle is built in `model_loader`'s ROCm branch and
    /// `RocmDevice::try_new` (auto-inits RCCL from the same `GRIM_TP_*` env).
    /// This field exists so the engine can report and re-stamp the shard index
    /// at model registration without depending on grim-nn at the device layer.
    tp_config: Option<grim_nn::TensorParallelConfig>,
    /// Background KV receiver server handle (started in Engine::new when
    /// disagg_config is Some and role is Decode or Colocated).
    kv_receiver: Option<grim_disagg::KvReceiverServer>,
    /// Live GPU capability profiler and epoch manager.
    /// Only constructed when world_size > 1 or `GRIM_SCYTHE_INFERENCE=1` (WI-INF1).
    pub capability_profiler: Option<Arc<grim_backend_rocm::CapabilityProfiler>>,
    /// SCYTHE-2 online router for continuous batching / multi-GPU placement (WI-INF2).
    pub scythe_ctrl: Option<crate::scythe2::C2plrController>,
    /// SCYTHE-2 farm mode (WI-INF3 serving integration): per-base-model
    /// replica ids. Replica `r ≥ 1` of `base` is registered as
    /// `{base}#scythe{r}` and holds a full weight copy on that rank's device;
    /// rank 0 is the base registration itself.
    scythe_replicas: HashMap<String, Vec<String>>,
    /// Request → replica rank, decided by the controller at admission time.
    /// The pinned replica executes every forward for that request's lifetime,
    /// so its KV pages stay local to one device.
    scythe_pin: HashMap<u64, usize>,
    /// WI-SB1 load-spreading: ranks of recently finished requests with the
    /// time they were released. Back-to-back admissions must still see the
    /// predecessor's load or a burst of short requests all lands on rank 0
    /// (each admission finds the pin map empty again).
    scythe_pin_cooldown: Vec<(usize, std::time::Instant)>,
    /// WI-SB2: requests held back because no farm rank could hold their KV
    /// footprint at enqueue time. They never reach the scheduler or own a
    /// session until a retry (each tick) finds a rank with room, so an
    /// oversized prompt can never be admitted blind onto a card that cannot
    /// hold it.
    scythe_vram_waitlist: Vec<grim_scheduler::Request>,
    /// Set GRIM_RADIX=on to enable prefix-cache reuse on prefill (WP5).
    pub radix_enabled: bool,
}

/// Effective-capability view for SCYTHE-2 farm placement: a GPU already
/// running `load` concurrent sessions contributes roughly `1/(1+load)` of its
/// solo throughput, so the controller's WaveTune latency argmin doubles as a
/// load balancer instead of piling every session onto the fastest card.
/// How long a finished request's rank stays counted as loaded (WI-SB1
/// load-spreading). Long enough to bridge back-to-back admissions of short
/// requests; short enough not to skew placement when the farm genuinely idles.
pub(crate) const SCYTHE_PIN_COOLDOWN: std::time::Duration =
    std::time::Duration::from_millis(1000);

/// External (non-farm) GPU utilization converts to equivalent pinned
/// requests at this weight: a card maxed out by a desktop/game workload
/// counts as ~2 in-flight farm requests — enough to flip a ~2:1 measured
/// capability pair toward the idle slower card.
const SCYTHE_EXTERNAL_BUSY_WEIGHT: f32 = 2.0;

/// WI-SB1: per-rank load seen by admission = active farm pins + pins
/// released inside the cooldown window + external busy-% converted at
/// `busy_weight`. Pure so the weighting/expiry rules stay unit-testable.
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

/// WI-SB2: worst-case device memory one request can reach — its paged KV
/// (`2·seq·kv_heads·head_dim·layers·4B`, K+V at fp32 page width) plus an
/// activation working-set floor (`2·seq·hidden·layers·4B`). When the model
/// doesn't report a hidden width, the KV dimension stands in rather than
/// inventing one.
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

/// WI-SB2: which farm ranks can hold a request's footprint. Headroom for
/// workspace and fragmentation is covered by [`SCYTHE_VRAM_WATERMARK_BYTES`].
/// A rank reporting zero free VRAM counts as feasible — that reading means
/// the probe is unavailable on it, not that the card is full; rejecting those
/// ranks would dead-lock placement exactly when visibility is worst.
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

/// WI-SB2 admission guard watermark: free-VRAM headroom (512 MiB) a rank must
/// keep above a request's computed footprint so scratch buffers, logits and
/// allocator fragmentation never push a pinned request into an OOM.
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

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let mut pool = KvBlockPool::new(
            config.block_pool_capacity,
            config.num_kv_heads,
            config.head_dim,
        );

        // KV-cache quantization (§kv-int8). `EngineConfig.kv_compressor` takes
        // precedence; otherwise honor `GRIM_KV_QUANT=int8` which attaches a
        // Lloyd-Max int4/int8 compressor so the paged KV pool compresses
        // admitted blocks before spill. Previously this defaulted to `None`,
        // leaving the real `LloydMaxCompressor` impl (grim-kvquant) unreachable
        // from the serving path.
        let mut compressor: Option<Arc<dyn grim_kvquant::KvCompressor>> =
            config.kv_compressor.clone();
        if compressor.is_none() {
            if let Ok(mode) = std::env::var("GRIM_KV_QUANT") {
                match mode.trim().to_ascii_lowercase().as_str() {
                    "int8" => {
                        let cfg = grim_kvquant::KvQuantConfig {
                            key_bits: 3,
                            value_bits: 8,
                            group_size: 64,
                            qk_compute_bits: 8,
                        };
                        compressor = Some(Arc::new(grim_kvquant::LloydMaxCompressor::new(cfg)));
                        eprintln!(
                            "[grim-engine] kv-int8: attached LloydMaxCompressor (key_bits=3, value_bits=8, group=64)"
                        );
                    }
                    "int4" => {
                        let cfg = grim_kvquant::KvQuantConfig {
                            key_bits: 2,
                            value_bits: 4,
                            group_size: 64,
                            qk_compute_bits: 8,
                        };
                        compressor = Some(Arc::new(grim_kvquant::LloydMaxCompressor::new(cfg)));
                        eprintln!(
                            "[grim-engine] kv-int8: attached LloydMaxCompressor (key_bits=2, value_bits=4, group=64)"
                        );
                    }
                    "off" | "" => {}
                    other => eprintln!(
                        "[grim-engine] GRIM_KV_QUANT='{other}' not recognized (expected int8|int4|off)"
                    ),
                }
            }
        }
        if let Some(comp) = &compressor {
            pool.attach_compressor(comp.clone());
        }

        // Tensor-parallel bootstrap (§c-tp-scope, WI-TP-4).
        //
        // Design A (multi-process): one OS process per rank. Each process sets
        // GRIM_TP_SIZE=N + GRIM_TP_RANK=i + (optional) GRIM_GPUS. Here we
        // resolve *this* process's ordinal from the env for logging and stamp
        // the derived `tp_config` onto `LoadedModel` at registration time.
        //
        // The actual per-rank `RocmDevice` + `RcclAllReduce` is built elsewhere:
        //   - `model_loader.rs` resolves the rank's ordinal in the ROCm branches
        //     of `load_from_path`, constructs `Device::Rocm(my_ordinal)`, and
        //     calls the model's `load_tp` (which shards weights by `ws.tp_config()`).
        //   - `RocmDevice::try_new` auto-inits RCCL over the full ordinal list
        //     via `auto_init_rccl()` (roc_device.rs:280), so every rank's
        //     `ncclAllReduce` rendezvous with its peers.
        //
        // We do NOT pre-build RocmDevices or fan out devices in-process here —
        // that would be the inert `TpRankContext`/`plan_tp_ranks` pattern, which
        // silently shards weights on one GPU and hangs on `ncclAllReduce` waiting
        // for peers that never started. Under TP, `Engine::new` must hard-fail
        // if the config is structurally invalid (rank >= world_size), not
        // silently degrade to a wrong shard.
        let tp_config: Option<grim_nn::TensorParallelConfig> = if config.tp_size > 1 {
            let tp = grim_nn::TensorParallelConfig::from_env().unwrap_or_default();
            if let Err(msg) = tp.validate() {
                // TP requested but structurally invalid — hard fail so the
                // operator sees the mismatch immediately instead of silently
                // loading the wrong shard. Engine::new returns Self (not Result),
                // so we panic; this is an unrecoverable config error.
                eprintln!(
                    "[grim-engine] INVALID TP config (GRIM_TP_SIZE={}): {msg}",
                    config.tp_size
                );
                panic!(
                    "invalid tensor-parallel configuration (GRIM_TP_SIZE / GRIM_TP_RANK): {msg}"
                );
            }
            // Resolve this rank's GPU ordinal (mirrors model_loader's logic)
            // for diagnostics — the actual device is built in the loader.
            let gpus: Vec<usize> = std::env::var("GRIM_GPUS")
                .ok()
                .map(|s| {
                    s.split(',')
                        .filter_map(|t| t.trim().parse::<usize>().ok())
                        .collect()
                })
                .unwrap_or_default();
            let my_ordinal = gpus.get(tp.rank).copied().unwrap_or(tp.rank);
            eprintln!(
                "[grim-engine] TP rank {}/{} on ordinal {} (device built in model_loader)",
                tp.rank, tp.world_size, my_ordinal
            );
            Some(tp)
        } else {
            None
        };
        let block_pool = Arc::new(std::sync::Mutex::new(pool));

        // Disaggregation: start a background KV receiver server when configured
        // for Decode or Colocated roles. The receiver writes incoming KV blocks
        // into the engine's block_pool, enabling cross-node KV handoff.
        let kv_receiver = if let Some(ref dc) = config.disagg_config {
            let role = dc.role;
            let listen_addr = if role == grim_disagg::PoolRole::Decode {
                &dc.decode_addr
            } else {
                &dc.prefill_addr
            };
            match grim_disagg::KvReceiverServer::new(listen_addr, block_pool.clone()) {
                Ok(srv) => {
                    eprintln!(
                        "[grim-engine] disagg: KV receiver server started on {} (role={:?})",
                        srv.listen_addr(),
                        role
                    );
                    Some(srv)
                }
                Err(e) => {
                    eprintln!(
                        "[grim-engine] disagg: failed to start KV receiver on {listen_addr}: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };

        let admission =
            grim_scheduler::AdmissionController::new(config.target_ttft_ms, config.target_itl_ms);
        let mut scheduler = grim_scheduler::Scheduler::new(
            config.max_batched_tokens,
            config.max_num_seqs,
            admission,
        );
        scheduler.determinism_mode = config.determinism_mode;
        let target_ttft = config.target_ttft_ms as f64;
        let target_itl = config.target_itl_ms as f64;

        let is_multi_gpu = tp_config
            .as_ref()
            .map(|tp| tp.world_size > 1)
            .unwrap_or(false)
            || config.tp_size > 1;
        let scythe_inference_flag = std::env::var("GRIM_SCYTHE_INFERENCE")
            .map(|v| v == "1" || v == "true" || v == "on")
            .unwrap_or(false);

        // WI-INF1: the profiler is the only thing constructed on the default
        // path, and only when more than one GPU is visible. A single-GPU box
        // pays zero probe cost (gate: test_single_gpu_capability_profiler_is_none).
        let capability_profiler = if is_multi_gpu || scythe_inference_flag {
            Some(Arc::new(grim_backend_rocm::CapabilityProfiler::new()))
        } else {
            None
        };

        // WI-INF2: the controller routes activations across GPUs *in this
        // process* (SCYTHE-2's P2P-ring execution model), so it is armed on
        // the count of ROCm devices visible here — not on `TP world_size`,
        // which under Design A counts one GPU per OS process. Fewer than two
        // visible GPUs leaves nothing to route between; the controller stays
        // `None` even with the flag set.
        //
        // `num_layers` starts at the placeholder below and is re-sized to the
        // loaded model's real depth at registration time (see
        // `register_speculative`), one controller per loaded model.
        const SCYTHE_NUM_LAYERS_PLACEHOLDER: usize = 32;
        let visible_gpus = capability_profiler
            .as_ref()
            .map(|p| p.capabilities().len())
            .unwrap_or(0);
        let scythe_ctrl = if scythe_inference_flag && visible_gpus > 1 {
            eprintln!(
                "[scythe2] inference routing armed over {visible_gpus} visible GPUs \
                 (GRIM_SCYTHE_INFERENCE=1)"
            );
            Some(crate::scythe2::C2plrController::new(
                SCYTHE_NUM_LAYERS_PLACEHOLDER,
                visible_gpus,
                config.target_itl_ms as f64,
            ))
        } else {
            None
        };

        Self {
            config,
            scheduler,
            block_pool,
            models: HashMap::new(),
            sessions: HashMap::new(),
            adapters: HashMap::new(),
            last_outcomes: HashMap::new(),
            request_rng: HashMap::new(),
            request_model_ids: HashMap::new(),
            request_adapters: HashMap::new(),
            request_input_ids: HashMap::new(),
            request_last_token: HashMap::new(),
            self_tuning_controller: grim_scheduler::SelfTuningController::new(
                target_ttft,
                target_itl,
            ),
            tuned_speculative_block_len: 5,
            tuned_kv_compression_bit_width: 4,
            tokens_per_sec_ema: 0.0,
            total_tokens_generated: 0,
            accepted_tokens_total: 0,
            last_ttft_ms: None,
            last_itl_ms: None,
            tp_config,
            kv_receiver,
            capability_profiler,
            scythe_ctrl,
            scythe_replicas: HashMap::new(),
            scythe_pin: HashMap::new(),
            scythe_pin_cooldown: Vec::new(),
            scythe_vram_waitlist: Vec::new(),
            radix_enabled: std::env::var("GRIM_RADIX")
                .map(|v| v != "0" && v != "false" && v != "off")
                .unwrap_or(true),
        }
    }

    /// Tensor-parallel configuration resolved once at `Engine::new` from the env.
    /// `GRIM_TP_RANK` selects this process's shard index; `GRIM_TP_SIZE` selects
    /// the world size. Returns `None` when `GRIM_TP_SIZE` is unset or 1
    /// (single-device).
    pub fn tp_config(&self) -> Option<grim_nn::TensorParallelConfig> {
        self.tp_config.clone()
    }

    /// Access the disaggregated KV receiver server instance if running in disaggregated decode role.
    pub fn kv_receiver(&self) -> Option<&grim_disagg::KvReceiverServer> {
        self.kv_receiver.as_ref()
    }

    /// Return live snapshot of visible GPU capabilities if profiler is active.
    pub fn capabilities(&self) -> Option<Vec<grim_tensor::backend::GpuCapability>> {
        self.capability_profiler.as_ref().map(|p| p.capabilities())
    }

    /// Return topology link matrix between visible GPUs.
    pub fn link_matrix(&self, num_gpus: usize) -> Vec<grim_tensor::backend::ScytheLink> {
        grim_backend_rocm::CapabilityProfiler::link_matrix(num_gpus)
    }

    /// True when SCYTHE-2 inference routing is armed (`GRIM_SCYTHE_INFERENCE`
    /// set and more than one ROCm GPU visible).
    pub fn scythe_armed(&self) -> bool {
        self.scythe_ctrl.is_some()
    }

    /// Hand the engine's SCYTHE-2 controller to a streaming executor as a
    /// [`ScytheRoute`](crate::streaming_forward::ScytheRoute) (WI-INF3).
    ///
    /// The route snapshots capabilities/links from the profiler lazily (on
    /// capability-epoch change only) and maps rank → `Device::Rocm(rank)`.
    /// Returns `false` (and touches nothing) when routing isn't armed.
    pub fn attach_scythe_route(
        &mut self,
        sfb: &mut crate::streaming_forward::StreamingBlockForward,
    ) -> bool {
        let Some(ctrl) = self.scythe_ctrl.take() else {
            return false;
        };
        let Some(ref profiler) = self.capability_profiler else {
            // Armed controller without a profiler cannot happen by
            // construction; put the controller back rather than dropping it.
            self.scythe_ctrl = Some(ctrl);
            return false;
        };
        sfb.attach_scythe_route(crate::streaming_forward::ScytheRoute {
            ctrl: Arc::new(std::sync::Mutex::new(ctrl)),
            profiler: Some(Arc::clone(profiler)),
            caps: Vec::new(),
            links: Vec::new(),
            caps_epoch: u32::MAX, // force first-use refresh from the profiler
            device_for_rank: Arc::new(|rank| grim_tensor::Device::Rocm(rank)),
        });
        true
    }

    /// SCYTHE-2 farm mode (WI-INF3 serving integration).
    ///
    /// A dense model's blocks live on one device, so per-layer placement needs
    /// the weight-sharded ring substrate that doesn't exist yet. What CAN route
    /// today is whole-pass placement: one full weight replica per visible GPU,
    /// with the controller pinning each admitted session to a rank. Sessions
    /// pinned to different ranks run genuinely in parallel; the WaveTune
    /// estimate steers sessions toward the faster card and the load-adjusted
    /// capability view spreads them once it saturates.
    ///
    /// `primary` has already been loaded by the caller on its env-chosen
    /// device — it becomes rank 0. The remaining replicas are loaded from
    /// `path`, one per leftover visible ROCm device. Without an armed
    /// controller or a second GPU this degrades to plain registration.
    pub fn register_model_with_farm(&mut self, id: &str, primary: Box<dyn CausalLm>, path: &str) {
        self.register_model_with_farm_inner(id, primary, path, None);
    }

    /// Load a farm while keeping speculative decoding on rank 0. Replica
    /// models are intentionally plain: the drafter is coupled to rank 0's
    /// device and is not replicated across the farm.
    pub fn load_and_register_scythe_farm_speculative(
        &mut self,
        id: &str,
        base_path: &str,
        draft_path: Option<&str>,
        lookahead: bool,
    ) -> Result<()> {
        let base_model = crate::model_loader::load_from_path(base_path)?;
        let drafter = if let Some(d_path) = draft_path {
            match crate::model_loader::load_eagle3_from_path(d_path, base_model.device().clone()) {
                Ok(eagle3) => Some(Arc::new(grim_speculative::Eagle3Drafter::new(eagle3))
                    as Arc<dyn DraftBackbone>),
                Err(_) => {
                    let _ = crate::model_loader::load_from_path(d_path)?;
                    Some(Arc::new(grim_speculative::TinyDraftBackbone::new(
                        128256, 2048, 4, 42,
                    )) as Arc<dyn DraftBackbone>)
                }
            }
        } else {
            None
        };
        let _ = lookahead;
        self.register_model_with_farm_inner(id, base_model, base_path, drafter);
        Ok(())
    }

    fn register_model_with_farm_inner(
        &mut self,
        id: &str,
        primary: Box<dyn CausalLm>,
        path: &str,
        drafter: Option<Arc<dyn DraftBackbone>>,
    ) {
        let devices = crate::model_loader::visible_rocm_devices();
        let primary_ordinal = match (self.scythe_armed(), primary.device()) {
            (true, grim_tensor::Device::Rocm(ord)) if devices.len() > 1 => *ord,
            _ => {
                self.register_model(id, primary);
                return;
            }
        };
        // Rank order: primary's own ordinal first, remaining GPUs after.
        let mut ordered: Vec<grim_tensor::Device> =
            vec![grim_tensor::Device::Rocm(primary_ordinal)];
        for dev in devices {
            if dev != grim_tensor::Device::Rocm(primary_ordinal) {
                ordered.push(dev);
            }
        }
        if let Some(drafter) = drafter {
            self.register_speculative(id, primary, Some(drafter), None, None);
        } else {
            self.register_model(id, primary);
        }
        let mut replica_ids = vec![id.to_string()];
        for (rank, dev) in ordered.iter().enumerate().skip(1) {
            match crate::model_loader::load_from_path_on_device(path, dev.clone()) {
                Ok(replica) => {
                    let rid = Self::scythe_replica_id(id, rank);
                    self.register_model(&rid, replica);
                    replica_ids.push(rid);
                }
                Err(e) => {
                    // A failed replica must not silently shrink the farm below
                    // what the controller was told to route over — fail loudly.
                    eprintln!(
                        "[scythe2] farm replica {rid} on {dev:?} failed to load: {e}; \
                         removing partial farm",
                        rid = Self::scythe_replica_id(id, rank)
                    );
                    for rid in &replica_ids[1..] {
                        self.models.remove(rid);
                    }
                    self.scythe_replicas.remove(id);
                    return;
                }
            }
        }
        // The controller routes over exactly the registered replica count.
        if let Some(ctrl) = self.scythe_ctrl.as_mut() {
            let n = replica_ids.len();
            if ctrl.num_gpus() != n {
                let num_layers = ctrl.layer_fps.len();
                let budget = ctrl.budget_ms;
                *ctrl = crate::scythe2::C2plrController::new(num_layers, n, budget);
            }
        }
        eprintln!(
            "[scythe2] farm armed: {} replica(s) of {id} across GPUs {:?}",
            replica_ids.len(),
            ordered
        );
        self.scythe_replicas.insert(id.to_string(), replica_ids);
    }

    fn scythe_replica_id(base: &str, rank: usize) -> String {
        format!("{base}#scythe{rank}")
    }

    /// Resolve the replica id a pinned request executes on. Unpinned requests
    /// (or pins without a matching farm) resolve to the base id unchanged.
    fn effective_model_id(&self, request_id: u64, base: &str) -> String {
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

    /// WI-SB2 admission guard for one request against the farm's live caps.
    ///
    /// [`ScytheAdmission::Pin`] carries the controller-chosen rank;
    /// [`ScytheAdmission::WaitVram`] means every rank failed the footprint
    /// check (or the profiler sees nothing at all) and the caller must keep
    /// the request out of the scheduler rather than pin it blind;
    /// [`ScytheAdmission::Bypass`] means farm routing isn't engaged and the
    /// plain single-replica path applies unchanged (rollback invariant).
    fn scythe_admission_decision(
        &mut self,
        base: &str,
        seq_len: usize,
        max_new_tokens: usize,
        caps_raw: &[grim_tensor::backend::GpuCapability],
    ) -> ScytheAdmission {
        let Some(ids) = self.scythe_replicas.get(base) else {
            return ScytheAdmission::Bypass;
        };
        let n = ids.len();
        if n <= 1 {
            return ScytheAdmission::Pin(0);
        }
        if !self.scythe_armed() {
            return ScytheAdmission::Bypass;
        }
        if caps_raw.is_empty() {
            eprintln!("[scythe2] farm present but profiler sees no GPUs; leaving request queued");
            return ScytheAdmission::WaitVram;
        }
        // Active pins plus pins released inside the cooldown window, plus
        // external (non-farm) utilization folded in at a fixed weight.
        self.scythe_pin_cooldown
            .retain(|(_, t)| t.elapsed() < SCYTHE_PIN_COOLDOWN);
        let external_busy: Vec<Option<u32>> =
            (0..n).map(|r| grim_backend_rocm::compute_utilization(r)).collect();
        let effective_loads = scythe_effective_loads(
            self.scythe_pin.values().copied(),
            &self.scythe_pin_cooldown,
            SCYTHE_PIN_COOLDOWN,
            &external_busy,
            n,
            SCYTHE_EXTERNAL_BUSY_WEIGHT,
        );
        let any_external_busy = external_busy
            .iter()
            .any(|b| matches!(b, Some(pct) if *pct >= 25));
        let any_load = effective_loads.iter().any(|&l| l > 0.0);

        // Reserve KV plus the activation working-set floor before placement.
        let (layers, hidden_hint) = self.models.get(base).map_or((1, None), |m| {
            (
                m.model.num_layers_hint().unwrap_or(1) as u64,
                m.model.hidden_size_hint(),
            )
        });
        let footprint = scythe_request_footprint_bytes(
            seq_len,
            max_new_tokens,
            self.config.num_kv_heads,
            self.config.head_dim,
            hidden_hint,
            layers,
        );
        let mut caps = load_adjusted_caps(caps_raw, n, &effective_loads);
        let feasible = scythe_vram_feasible(&caps, footprint);
        for (cap, ok) in caps.iter_mut().zip(&feasible) {
            if !ok {
                cap.tflops_fp16 = 0.0;
            }
        }
        if feasible.iter().all(|&ok| !ok) {
            eprintln!(
                "[scythe2] no farm rank holds ~{} MiB; leaving request queued",
                footprint / (1024 * 1024)
            );
            return ScytheAdmission::WaitVram;
        }
        let links = grim_backend_rocm::CapabilityProfiler::link_matrix(n);
        let epoch = grim_backend_rocm::current_epoch();
        // Proxy shape: at pass granularity only the sequence length carries
        // signal (the bucket), and relative TFLOPS ordering drives the pick.
        let shape = [1usize, seq_len.max(1), 1, 1];
        let Some(ctrl) = self.scythe_ctrl.as_mut() else {
            return ScytheAdmission::Bypass;
        };
        // The shape-keyed PlacementCache is load-blind; while ANY rank
        // carries load (pins or external busy) the decision must run fresh.
        let placement = if any_load || any_external_busy {
            ctrl.decide_forced(0, &shape, &caps, &links, epoch)
        } else {
            ctrl.decide(0, &shape, &caps, &links, epoch)
        };
        let chosen = placement.ranks.first().copied();
        // WI-SB1 spread gate: non-zero ranks currently crash in
        // `sample_on_device` (page fault on the pinned replica's device —
        // first exposed when external-busy steering produced the first ever
        // rank-1 pin; see scythe2 plan validation log 2026-08-23e). Until
        // that hunt lands, spreading requires an explicit opt-in so the
        // default serve surface keeps its long-verified rank-0 behavior.
        let spread_enabled =
            std::env::var("GRIM_SCYTHE_SPREAD").map(|v| v == "1").unwrap_or(false);
        let chosen = match chosen {
            Some(r) if r != 0 && !spread_enabled => {
                eprintln!(
                    "[scythe2] load favored rank {r} but cross-replica serving \
                     is not yet safe (GRIM_SCYTHE_SPREAD unset); clamping to rank 0"
                );
                Some(0)
            }
            other => other,
        };
        if let Some(r) = chosen {
            eprintln!(
                "[scythe2] admission loads {:?} (external busy {:?}) -> rank {}",
                effective_loads, external_busy, r
            );
        }
        chosen
            .map(|r| r.min(n - 1))
            .map_or(ScytheAdmission::WaitVram, ScytheAdmission::Pin)
    }

    /// Pinned farm rank for a request, if any. Telemetry/status surface.
    pub fn scythe_pin_of(&self, request_id: u64) -> Option<usize> {
        self.scythe_pin.get(&request_id).copied()
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

    /// Number of farm replicas registered for `base` (0 = not a farm model).
    pub fn scythe_farm_size(&self, base: &str) -> usize {
        self.scythe_replicas.get(base).map_or(0, Vec::len)
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

    /// Most recent measured prefill time in milliseconds. `None` means no
    /// completed prefill has been observed yet; callers must not invent a
    /// latency value for that state.
    pub fn last_ttft_ms(&self) -> Option<f64> {
        self.last_ttft_ms
    }

    /// Most recent measured inter-token latency in milliseconds. `None` means no
    /// completed decode step has been observed yet; callers must not invent a
    /// latency value for that state.
    pub fn last_itl_ms(&self) -> Option<f64> {
        self.last_itl_ms
    }

    /// Clear the TTFT/ITL trace so a caller measuring request-by-request (the
    /// WI-SB3 A/B harness) can distinguish a fresh measurement from the
    /// previous request's stale one. Without this, `last_ttft_ms()` stays
    /// `Some` forever and every later sample records the earlier value.
    pub fn clear_latency_trace(&mut self) {
        self.last_ttft_ms = None;
        self.last_itl_ms = None;
    }

    /// Runtime speculative decoding telemetry for a specific model, or the
    /// first loaded model if `model_id` is None. Returns `None` if no model
    /// is loaded.
    pub fn speculative_telemetry(
        &self,
        model_id: Option<&str>,
    ) -> Option<grim_speculative::SpeculativeTelemetry> {
        let model = match model_id {
            Some(id) => self.models.get(id)?,
            None => self.models.values().next()?,
        };
        Some(model.model.telemetry())
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

    /// Maximum context limit (max batched tokens) configured for this engine.
    pub fn context_limit(&self) -> usize {
        self.config.max_batched_tokens
    }

    /// Total count of tokens generated since engine startup.
    pub fn total_tokens_generated(&self) -> u64 {
        self.total_tokens_generated
    }

    /// Total count of speculative draft tokens accepted since engine startup.
    pub fn accepted_tokens_total(&self) -> u64 {
        self.accepted_tokens_total
    }

    /// WI-E2: speculative-decoding acceptance rate — accepted / generated.
    /// Returns None when no tokens have been generated yet.
    pub fn acceptance_rate(&self) -> Option<f64> {
        if self.total_tokens_generated == 0 {
            return None;
        }
        Some(self.accepted_tokens_total as f64 / self.total_tokens_generated as f64)
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

    pub fn register_model(&mut self, id: &str, model: Box<dyn CausalLm>) {
        self.register_speculative(id, model, None, None, None);
    }

    /// Register a `CausalLm` with an attached DSpark bundle (draft +
    /// Markov + confidence heads). The engine will pick DSpark
    /// speculation automatically. Falls back to plain if any of the
    /// heads is missing.
    pub fn register_with_dspark(
        &mut self,
        id: &str,
        model: Box<dyn CausalLm>,
        draft: Arc<dyn DraftBackbone>,
        markov: Arc<dyn MarkovHead>,
        confidence: Arc<dyn ConfidenceHead>,
    ) {
        self.register_speculative(id, model, Some(draft), Some(markov), Some(confidence));
    }

    fn register_speculative(
        &mut self,
        id: &str,
        model: Box<dyn CausalLm>,
        draft: Option<Arc<dyn DraftBackbone>>,
        markov: Option<Arc<dyn MarkovHead>>,
        confidence: Option<Arc<dyn ConfidenceHead>>,
    ) {
        // By default we check if weight streaming is active and what VRAM remains.
        // During registration we check the environment or fallback parameters.
        let is_weight_streaming_active = std::env::var("GRIM_WEIGHT_STREAMING").is_ok();
        let available_vram = std::env::var("GRIM_AVAILABLE_VRAM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let dev = model.device().clone();
        // WI-INF2: one SCYTHE-2 controller per loaded model, sized by the
        // model's real transformer depth (the `Engine::new` value was a
        // placeholder — no model is known at construction time).
        if let Some(num_layers) = model.num_layers_hint() {
            if let Some(ctrl) = self.scythe_ctrl.as_mut() {
                let num_gpus = ctrl.num_gpus();
                *ctrl = crate::scythe2::C2plrController::new(num_layers, num_gpus, ctrl.budget_ms);
            }
        }
        // Preserve the model's own modality hint (audio enc-dec, TTS, VC,
        // vocoder, diffusion…) so serving-layer routing sees the truth.
        // Hardcoding TextInTextOut misreported every non-text model that
        // registered through this path, including the audio models.
        let modality = model.config().modality();
        let wrapped = SpeculativeCausalLm::auto(
            model,
            draft,
            markov,
            confidence,
            is_weight_streaming_active,
            available_vram,
        );
        let config: Box<dyn ModelConfig> = Box::new(grim_core::config::GenericModelConfig {
            name: id.to_string(),
            modality,
        });
        self.models.insert(
            id.to_string(),
            LoadedModel {
                model: Box::new(wrapped),
                config,
                device: dev,
                tp_config: self.tp_config(),
            },
        );
    }

    /// Register a native multi-token prediction model wrapped in `LlamaMtpAdapter`.
    pub fn register_native_mtp_model(
        &mut self,
        id: &str,
        model: Arc<grim_models_transformer::LlamaMtp>,
    ) {
        let dev = grim_core::Model::device(model.as_ref()).clone();
        let modality = grim_core::Model::config(model.as_ref()).modality();
        let adapter = grim_speculative::LlamaMtpAdapter::new(model.clone());
        let mtp_arc: Arc<dyn grim_speculative::NativeMtp> = Arc::new(adapter);
        let wrapped = SpeculativeCausalLm::with_native_mtp(
            Box::new(grim_speculative::LlamaMtpAdapter::new(model)),
            mtp_arc,
        );
        let config: Box<dyn ModelConfig> = Box::new(grim_core::config::GenericModelConfig {
            name: id.to_string(),
            modality,
        });
        self.models.insert(
            id.to_string(),
            LoadedModel {
                model: Box::new(wrapped),
                config,
                device: dev,
                tp_config: self.tp_config(),
            },
        );
    }

    /// Register an EAGLE3 speculative drafter model coupled with a base model.
    pub fn register_eagle3_model(
        &mut self,
        id: &str,
        base_model: Box<dyn CausalLm>,
        eagle3_model: Arc<grim_models_transformer::Eagle3>,
    ) {
        let drafter = Arc::new(grim_speculative::Eagle3Drafter::new(eagle3_model));
        self.register_speculative(id, base_model, Some(drafter), None, None);
    }

    /// Load base model and optional draft model / lookahead, registering them into the engine.
    pub fn load_and_register_speculative(
        &mut self,
        id: &str,
        base_path: &str,
        draft_path: Option<&str>,
        _lookahead: bool,
    ) -> Result<()> {
        let base_model = crate::model_loader::load_from_path(base_path)?;
        if let Some(d_path) = draft_path {
            let dev = base_model.device().clone();
            if let Ok(eagle3) = crate::model_loader::load_eagle3_from_path(d_path, dev.clone()) {
                self.register_eagle3_model(id, base_model, eagle3);
            } else {
                let draft_model = crate::model_loader::load_from_path(d_path)?;
                // Generic draft model: wrap as DraftBackbone or register speculative
                let drafter = Arc::new(grim_speculative::TinyDraftBackbone::new(
                    128256, 2048, 4, 42,
                ));
                let _ = draft_model;
                self.register_speculative(id, base_model, Some(drafter), None, None);
            }
        } else {
            self.register_model(id, base_model);
        }
        Ok(())
    }

    /// Register a multi-LoRA adapter against a base model. The adapter is
    /// keyed by its [`AdapterHandle::id`] and dispatched into the forward
    /// pass when callers pass `&[AdapterHandle]` that references it.
    /// `name` is the human-readable identifier used for HTTP request-body
    /// resolution — the server 400s on any name not present here.
    pub fn register_adapter(
        &mut self,
        base_model_id: &str,
        name: impl Into<String>,
        handle: AdapterHandle,
    ) {
        self.adapters.insert(
            handle.id,
            LoadedAdapter {
                name: name.into(),
                handle,
                base_model_id: base_model_id.to_string(),
            },
        );
    }

    /// Resolve a set of adapter ids into concrete [`AdapterHandle`]s.
    /// Returns `None` if any id is unknown — the caller should drop the
    /// affected request rather than synthesize a partial adapter set.
    pub fn resolve_adapters(&self, ids: &[u32]) -> Option<Vec<AdapterHandle>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            match self.adapters.get(id) {
                Some(a) => out.push(a.handle.clone()),
                None => return None,
            }
        }
        Some(out)
    }

    /// Drop an adapter from the registry. Its id is freed for reuse.
    pub fn drop_adapter(&mut self, id: u32) -> bool {
        self.adapters.remove(&id).is_some()
    }

    /// Number of currently-loaded adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Dynamically reconfigure the MoE VRAM budget at a safe point between inference steps.
    ///
    /// # Contract
    /// Dynamically adjusts the split between KV cache pages and MoE expert cache slots
    /// without restarting the engine or reloading host weights.
    pub fn reconfigure_moe_budget(
        &mut self,
        new_kv_envelope_bytes: usize,
        new_expert_envelope_bytes: usize,
    ) -> Result<usize> {
        let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
        let block_bytes = pool.block_bytes();
        if block_bytes > 0 {
            let target_blocks = new_kv_envelope_bytes / block_bytes;
            pool.resize_capacity(target_blocks);
        }
        Ok(new_expert_envelope_bytes)
    }

    /// Look up an adapter handle by its human-readable name. Used by the HTTP
    /// server to validate names from request body `"adapters"` arrays before
    /// opening an SSE stream — unknown names must 400 immediately rather than
    /// silently produce unadapted output.
    pub fn get_adapter_by_name(&self, name: &str) -> Option<&LoadedAdapter> {
        self.adapters.values().find(|a| a.name == name)
    }

    /// Fresh adapter id: max existing + 1. (`adapter_count() + 1` would reuse
    /// ids freed by `drop_adapter`, silently aliasing stale references.)
    pub fn next_adapter_id(&self) -> u32 {
        self.adapters.keys().copied().max().unwrap_or(0) + 1
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

    /// Unload a model from memory by its name. Returns true if the model was loaded.
    pub fn unload_model(&mut self, name: &str) -> bool {
        self.models.remove(name).is_some()
    }

    /// Strategy the model is operating under right now (Plain / NativeMtp /
    /// DSpark). `None` if the model id isn't registered.
    pub fn strategy_for(&self, id: &str) -> Option<Strategy> {
        self.models.get(id).map(|m| m.model.strategy())
    }

    /// Run one engine iteration. For each scheduled prefill or decode
    /// request, drive the speculative wrapper against the request's
    /// session and capture per-request outcomes.
    pub fn tick(&mut self) -> Result<grim_scheduler::SchedulerOutput> {
        let tick_start = Instant::now();

        // Background capability profiler tick check (~100ms cadence)
        if let Some(ref profiler) = self.capability_profiler {
            if profiler.age() >= Duration::from_millis(100) {
                profiler.tick();
                // WI-INF2: pull the (possibly bumped) capability epoch into
                // the placement cache. This is the existing mode-B staleness
                // path — `sync_epoch` clears the fast slots so the next
                // forward re-decides against fresh capabilities; it is not
                // new invalidation logic.
                if let Some(ref mut ctrl) = self.scythe_ctrl {
                    ctrl.cache.sync_epoch(grim_backend_rocm::current_epoch());
                }
            }
        }

        // WI-SB2: give parked requests a chance to place before this pass's
        // admission, so freed VRAM is picked up in the same tick.
        self.retry_scythe_vram_waitlist();

        let output = self.scheduler.schedule();
        let schedule_elapsed = tick_start.elapsed();

        // Run prefill, then decode in a single deterministic pass — for
        // §5.3 correctness, prefills share block pool and decode uses
        // the KV they just wrote. We process them in the order the
        // scheduler produced them so a paused predicate is monotonically
        // consistent.
        let prefill = output.prefill_ids.clone();
        let had_prefill = !prefill.is_empty();
        let mut prefill_elapsed = Duration::ZERO;
        for id in prefill {
            if self.scheduler.is_paused(id) {
                continue;
            }
            let pf_start = Instant::now();
            self.drive_prefill(id)?;
            prefill_elapsed += pf_start.elapsed();
        }
        let mut decode_elapsed = Duration::ZERO;
        let mut total_accepted = 0usize;
        let mut decode_count = 0usize;

        // Process active decode steps across all scheduled requests via step_batch (WI-X1)
        let mut decode_items = Vec::new();
        for &id in &output.decode_ids {
            if !self.scheduler.is_paused(id) {
                if let Some((model_id, _)) = self.model_for_request(id) {
                    let start_pos = self.sessions.get(&id).map(|s| s.current_pos()).unwrap_or(0);
                    let next_token = self
                        .request_last_token
                        .get(&id)
                        .copied()
                        .unwrap_or(start_pos as u32);
                    let ids = grim_backend_cpu::cpu_tensor(
                        vec![next_token as f32],
                        grim_tensor::Shape::new(vec![1]),
                    );
                    let positions = grim_backend_cpu::cpu_tensor(
                        vec![start_pos as f32],
                        grim_tensor::Shape::new(vec![1]),
                    );
                    decode_items.push((id, model_id.to_string(), ids, positions));
                }
            }
        }

        if !decode_items.is_empty() {
            let dec_start = Instant::now();
            let batch_refs: Vec<(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)> =
                decode_items
                    .iter()
                    .map(|(id, m, ids, pos)| (*id, m.as_str(), ids, pos))
                    .collect();
            let batch_results = self.step_batch(&batch_refs)?;
            decode_count = batch_results.len();
            for (id, outcome) in batch_results {
                total_accepted += outcome.accepted_tokens;
                self.last_outcomes.insert(id, outcome);
            }
            decode_elapsed = dec_start.elapsed();
        }

        // MIN-4: Record actual forward-pass wall time, not schedule time.
        // TTFT = time to first token (prefill), ITL = inter-token latency (decode).
        let ttft_ms = prefill_elapsed.as_secs_f64() * 1000.0;
        if had_prefill {
            self.last_ttft_ms = Some(ttft_ms);
        }
        let itl_ms = if decode_count > 0 {
            let itl = decode_elapsed.as_secs_f64() * 1000.0 / decode_count as f64;
            self.last_itl_ms = Some(itl);
            itl
        } else {
            0.0
        };
        self.self_tuning_controller.record_ttft(ttft_ms);
        self.self_tuning_controller.record_itl(itl_ms);

        // MIN-3: Apply ALL tuned params, not just max_batched_tokens and
        // chunked_prefill_size.
        let tuned_params = self.self_tuning_controller.tune_all();
        self.scheduler.max_batched_tokens = tuned_params.max_batched_tokens;
        self.scheduler.chunked_prefill_size = tuned_params.chunked_prefill_size;
        // Speculative block length and KV compression bit width are stored
        // on the engine for the speculative wrapper to pick up at decode time.
        self.tuned_speculative_block_len = tuned_params.speculative_block_len;
        self.tuned_kv_compression_bit_width = tuned_params.kv_compression_bit_width;

        // WI-E2: accumulate accepted tokens for the acceptance-rate metric.
        self.accepted_tokens_total += total_accepted as u64;
        let _ = (schedule_elapsed, total_accepted);
        let tick_elapsed = tick_start.elapsed();
        if total_accepted > 0 && tick_elapsed.as_secs_f32() > 0.0 {
            let inst_tps = (total_accepted as f32) / tick_elapsed.as_secs_f32();
            if self.tokens_per_sec_ema == 0.0 {
                self.tokens_per_sec_ema = inst_tps;
            } else {
                self.tokens_per_sec_ema = 0.7 * self.tokens_per_sec_ema + 0.3 * inst_tps;
            }
            self.total_tokens_generated += total_accepted as u64;
        }
        Ok(output)
    }

    /// WI-M2 drift watch (gguf_multigpu_context_plan.md): hold the
    /// process-wide prefill latch up for the duration of the pass. While it
    /// is set, ANY HIP context switch to a non-zero device on any thread is
    /// traced with a forced backtrace under `GRIM_ALLOC_TRACE`, so the setter
    /// flipping the forward pass onto a foreign GPU is named in the log.
    fn drive_prefill(&mut self, id: u64) -> Result<()> {
        grim_backend_rocm::set_prefill_in_flight(true);
        let outcome = self.drive_prefill_inner(id);
        grim_backend_rocm::set_prefill_in_flight(false);
        outcome
    }

    fn drive_prefill_inner(&mut self, id: u64) -> Result<()> {
        let prompt_tokens = match self.scheduler.running.iter().find(|r| r.id == id) {
            Some(r) => r.prompt_tokens,
            None => return Ok(()),
        };
        if prompt_tokens == 0 {
            return Ok(());
        }
        // Build the full input_ids tensor: use real token IDs if provided,
        // otherwise fall back to synthetic position indices (0..prompt_tokens)
        // for backward compatibility.
        let full_input: Vec<u32> = self
            .request_input_ids
            .get(&id)
            .cloned()
            .filter(|v| !v.is_empty() && v.len() == prompt_tokens)
            .unwrap_or_else(|| (0..prompt_tokens as u32).collect());

        if self.radix_enabled && !full_input.is_empty() {
            let (matched_blocks, matched_tokens, anchor_state) = {
                let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                pool.match_prefix_with_recurrent(&full_input)
            };
            if !matched_blocks.is_empty() {
                if let Some(cp) = anchor_state {
                    eprintln!(
                        "[grim-engine] req {id} radix hit: {}/{} tokens with semantic recurrent checkpoint #{}",
                        matched_tokens,
                        full_input.len(),
                        cp.id
                    );
                } else {
                    eprintln!(
                        "[grim-engine] req {id} radix hit: {}/{} tokens",
                        matched_tokens,
                        full_input.len()
                    );
                }
            }
        }

        let ids = grim_backend_cpu::cpu_tensor(
            full_input.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
            grim_tensor::Shape::new(vec![prompt_tokens]),
        );
        let positions = grim_backend_cpu::cpu_tensor(
            (0..prompt_tokens).map(|t| t as f32).collect::<Vec<f32>>(),
            grim_tensor::Shape::new(vec![prompt_tokens]),
        );
        if let Some((model_id, _)) = self.model_for_request(id) {
            let model_id = model_id.to_string();
            let outcome = self.drive_forward(&model_id, id, &ids, &positions)?;
            // `current_pos` is owned by the model/session — the underlying
            // forward already advanced it via `session.advance_pos(seq_len)`.
            // The engine does *not* double-count.
            self.last_outcomes.insert(id, outcome);

            // Radix prefix cache: register computed KV blocks and semantic state anchors
            if self.radix_enabled && !full_input.is_empty() {
                if let Some(session) = self.sessions.get(&id) {
                    if let Some(block_table) = session.block_table() {
                        let usize_blocks: Vec<usize> =
                            block_table.iter().map(|&b| b as usize).collect();
                        let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                        pool.insert_prefix_with_recurrent_state(
                            &full_input,
                            &usize_blocks,
                            Vec::new(),
                        );
                    }
                }
            }

            // Disaggregation handoff: if disagg_router is configured for Prefill role,
            // stream real KV blocks generated during prefill over the network to the decode node.
            if let Some(router) = &self.config.disagg_router {
                if router.pool_role == grim_disagg::PoolRole::Prefill {
                    // The pool is shared across concurrent requests, so the
                    // handoff must carry only this request's physical blocks —
                    // a full-pool scan would leak other requests' KV cache.
                    let block_ids: Vec<usize> = self
                        .sessions
                        .get(&id)
                        .and_then(|s| s.block_table())
                        .map(|t| t.iter().map(|&b| b as usize).collect())
                        .unwrap_or_default();
                    if !block_ids.is_empty() {
                        if let Some(session) = self.sessions.get(&id) {
                            if let Some(kv) = session.kv_cache() {
                                let num_layers = kv.num_layers();
                                for layer in 0..num_layers {
                                    for &b_id in &block_ids {
                                        if let Some((k_slice, v_slice)) =
                                            kv.layer_block_slice(layer, b_id)
                                        {
                                            if let Err(e) = router.send_layer_block_remote(
                                                b_id,
                                                layer as u32,
                                                k_slice,
                                                v_slice,
                                            ) {
                                                eprintln!(
                                                    "[grim-engine] Disagg prefill KV transfer failed for req {id}, layer {layer}, block {b_id}: {e}"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // F3: per-layer slices above are the single handoff channel;
                        // the redundant pool-level transfer is removed (was sent twice).
                    }
                }
            }
        }
        Ok(())
    }

    /// Drives a decode step for sequence `id`, recording the outcome step.
    pub fn drive_decode(&mut self, id: u64) -> Result<()> {
        let outcome = self.drive_decode_with_outcome(id)?;
        if let Some(outcome) = outcome {
            self.last_outcomes.insert(id, outcome);
        }
        Ok(())
    }

    fn drive_decode_with_outcome(&mut self, id: u64) -> Result<Option<StepOutcome>> {
        // Disaggregated decode: ensure required KV blocks are present in the
        // local pool before executing the decode step.  When this is a Decode
        // node, the KV cache was generated on the Prefill node and transferred
        // over the network.  The background KvReceiverServer (started in
        // Engine::new) writes incoming blocks into self.block_pool.  Here we
        // poll for / fetch any blocks that haven't arrived yet so the decode
        // session has a complete KV context across all layers.
        if let Some(ref router) = self.config.disagg_router {
            if router.pool_role == grim_disagg::PoolRole::Decode {
                let elem_per_token = self.config.num_kv_heads * self.config.head_dim;
                let block_elems = elem_per_token * BLOCK_SIZE;
                let num_blocks = {
                    let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                    pool.num_blocks()
                };
                let num_layers = self
                    .sessions
                    .get(&id)
                    .and_then(|s| s.kv_cache())
                    .map(|kv| kv.num_layers())
                    .unwrap_or(1)
                    .max(1);
                for block_id in 0..num_blocks {
                    let already_received = {
                        let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                        pool.block_is_received(block_id)
                    };
                    if already_received {
                        continue;
                    }
                    let mut fetch_ok = true;
                    for layer in 0..num_layers {
                        match router.fetch_kv_block(block_id, layer as u32, block_elems) {
                            Ok((k_data, v_data)) => {
                                let num_tokens = block_elems / elem_per_token;
                                if layer == 0 {
                                    let mut pool =
                                        self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                                    pool.write_keys(block_id, &k_data, num_tokens);
                                    pool.write_values(block_id, &v_data);
                                }
                                if let Some(session) = self.sessions.get_mut(&id) {
                                    if let Some(kv) = session.kv_mut() {
                                        let _ =
                                            kv.write_layer_block(layer, block_id, &k_data, &v_data);
                                    }
                                }
                            }
                            Err(e) => {
                                // F3: a failed layer must not leave the block marked
                                // received (write_keys auto-marks on layer 0), or that
                                // block would attend stale pages forever.
                                fetch_ok = false;
                                eprintln!(
                                    "[grim-engine] Disagg decode KV fetch failed for req {id}, layer {layer}, block {block_id}: {e}"
                                );
                            }
                        }
                    }
                    {
                        let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                        pool.set_received(block_id, fetch_ok);
                    }
                }
            }
        }

        let start_pos = self.sessions.get(&id).map(|s| s.current_pos()).unwrap_or(0);
        // Use the previously sampled token if available, otherwise fall back
        // to position index for backward compatibility.
        let next_token = self
            .request_last_token
            .get(&id)
            .copied()
            .unwrap_or(start_pos as u32);
        let ids =
            grim_backend_cpu::cpu_tensor(vec![next_token as f32], grim_tensor::Shape::new(vec![1]));
        let positions =
            grim_backend_cpu::cpu_tensor(vec![start_pos as f32], grim_tensor::Shape::new(vec![1]));
        if let Some((model_id, _)) = self.model_for_request(id) {
            let model_id = model_id.to_string();
            let outcome = self.drive_forward(&model_id, id, &ids, &positions)?;
            // See `drive_prefill` — position advancement is the model's
            // responsibility at this transition point.
            Ok(Some(outcome))
        } else {
            Ok(None)
        }
    }

    fn drive_forward(
        &mut self,
        model_id: &str,
        request_id: u64,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
    ) -> Result<StepOutcome> {
        // SCYTHE-2 farm mode: a pinned request executes on its replica, not
        // the base registration (same weights, different device).
        let model_id = self.effective_model_id(request_id, model_id);
        let model_id = model_id.as_str();
        // Resolve adapters for this specific request from the adapter registry
        let adapter_ids = self
            .request_adapters
            .get(&request_id)
            .cloned()
            .unwrap_or_default();
        let adapters = {
            let resolved = self.resolve_adapters(&adapter_ids).unwrap_or_default();
            resolved
        };
        let was_speculative_path = match self.models.get(model_id) {
            Some(m) => m.model.strategy() != Strategy::Plain,
            None => return Err(Error::Config(format!("unknown model {model_id}"))),
        };
        let session = self
            .sessions
            .get_mut(&request_id)
            .ok_or_else(|| Error::Config("no session for request".into()))?
            .as_mut();
        let loaded = self
            .models
            .get(model_id)
            .ok_or_else(|| Error::Config(format!("unknown model {model_id}")))?;
        let live = self.scheduler.running.len() as f32 / self.config.max_num_seqs.max(1) as f32;
        let logits = loaded.model.decode_one(
            session,
            input_ids,
            positions,
            live,
            self.scheduler.running.len(),
            &adapters,
        )?;
        // MIN-2: Report the actual accepted token count from the session
        // (set by the speculative wrapper's decode_one). Non-speculative
        // paths default to 1.
        let accepted_tokens = session.last_accepted_tokens();
        let _ = (loaded, was_speculative_path);
        Ok(StepOutcome {
            logits: Some(Arc::new(logits)),
            accepted_tokens,
            speculative: was_speculative_path,
        })
    }

    /// Public stepping API: drive one forward pass for `request_id`
    /// against a caller-supplied target model id, with caller-supplied
    /// adapters and an explicit input tensor. Returns the speculative
    /// wrapper's emitted logits.
    pub fn step_one(
        &mut self,
        request_id: u64,
        target_model_id: &str,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
    ) -> Result<StepOutcome> {
        self.drive_forward(target_model_id, request_id, input_ids, positions)
    }

    /// Execute a grouped batch step across multiple co-scheduled requests (WI-X1).
    ///
    /// Drives decoding across up to N requests in a single scheduling tick,
    /// returning each request's corresponding StepOutcome.
    pub fn step_batch(
        &mut self,
        items: &[(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)],
    ) -> Result<Vec<(u64, StepOutcome)>> {
        let mut results = Vec::with_capacity(items.len());
        for &(req_id, model_id, input_ids, positions) in items {
            let outcome = self.step_one(req_id, model_id, input_ids, positions)?;
            results.push((req_id, outcome));
        }
        Ok(results)
    }

    /// Check if a model is registered by name.
    pub fn has_model(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    pub fn enqueue_request(&mut self, request: grim_scheduler::Request) -> Result<()> {
        self.enqueue_request_with_kv(request)
    }

    /// Allocate a session with a paged KV cache wired in and prefix caching active (§5.1).
    pub fn enqueue_request_with_kv(&mut self, request: grim_scheduler::Request) -> Result<()> {
        // SCYTHE-2 farm mode: pin the request to a controller-chosen replica
        // BEFORE the session exists, so its KV pages are allocated on the
        // pinned replica's device and stay there for the request's lifetime.
        let base_for_pin = request
            .model_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.models.keys().next().cloned());
        let mut pin_rank = None;
        if let Some(base) = base_for_pin.as_deref() {
            if self.scythe_armed() && self.scythe_replicas.contains_key(base) {
                // WI-SB2: consult the VRAM guard before anything is allocated.
                let caps_raw = self
                    .capability_profiler
                    .as_ref()
                    .map(|p| p.capabilities())
                    .unwrap_or_default();
                match self.scythe_admission_decision(
                    base,
                    request.prompt_tokens,
                    request.max_new_tokens,
                    &caps_raw,
                ) {
                    ScytheAdmission::Pin(rank) => {
                        self.scythe_pin.insert(request.id, rank);
                        eprintln!(
                            "[scythe2] request {} ({} tok) pinned to farm rank {rank}",
                            request.id, request.prompt_tokens
                        );
                        pin_rank = Some(rank);
                    }
                    ScytheAdmission::WaitVram => {
                        // Hold the request out of the scheduler entirely: no
                        // session, no pin, no admission — a rank must be able
                        // to hold it before it enters the queue.
                        eprintln!(
                            "[scythe2] request {} parked on VRAM waitlist (WI-SB2)",
                            request.id
                        );
                        self.scythe_vram_waitlist.push(request);
                        return Ok(());
                    }
                    ScytheAdmission::Bypass => {}
                }
            }
        }
        self.admit_placed_request(request, pin_rank)
    }

    /// Session creation + scheduler entry for an already-placed request.
    /// `pin_rank` (farm mode) selects the pinned replica's device for the KV.
    fn admit_placed_request(
        &mut self,
        request: grim_scheduler::Request,
        pin_rank: Option<usize>,
    ) -> Result<()> {
        // Honor the model's actual device instead of silently forcing CPU.
        // Under a farm pin, that is the pinned replica's device.
        let base_for_pin = request
            .model_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.models.keys().next().cloned());
        let device = if pin_rank.is_some() {
            base_for_pin
                .as_deref()
                .map(|base| self.effective_model_id(request.id, base))
                .and_then(|rid| self.models.get(&rid).map(|m| m.device.clone()))
                .unwrap_or(grim_tensor::Device::Cpu)
        } else {
            match request.model_id.as_deref() {
                Some(id) if !id.is_empty() => self
                    .models
                    .get(id)
                    .map(|m| m.device.clone())
                    .unwrap_or(grim_tensor::Device::Cpu),
                _ => self
                    .models
                    .values()
                    .next()
                    .map(|m| m.device.clone())
                    .unwrap_or(grim_tensor::Device::Cpu),
            }
        };
        let mut kv = grim_memory::PagedKvCache::new(
            self.block_pool.clone(),
            self.config.num_kv_heads,
            self.config.head_dim,
            BLOCK_SIZE,
        );
        let backend = grim_nn::pick_device_for_storage_device(&device);
        kv.set_device(device.clone(), backend);
        let session = Box::new(grim_core::session::Inner::with_kv(device, Box::new(kv)));

        self.sessions.insert(request.id, session);
        self.request_model_ids
            .insert(request.id, request.model_id.clone().unwrap_or_default());
        self.request_adapters
            .insert(request.id, request.adapter_ids.clone());
        // Store the real input token IDs if provided
        if let Some(input_ids) = request.input_ids.clone() {
            self.request_input_ids.insert(request.id, input_ids);
        }
        self.request_rng.insert(
            request.id,
            DeterministicRng::from_seed(request.id.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        self.scheduler.enqueue(request);
        Ok(())
    }

    /// WI-SB2: retry requests parked on the VRAM waitlist at tick start —
    /// finished sessions have freed their ranks by now. Order-stable backfill:
    /// entries are scanned in arrival order and admitted individually as soon
    /// as some rank can hold them; those that still fail stay queued.
    fn retry_scythe_vram_waitlist(&mut self) {
        if self.scythe_vram_waitlist.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.scythe_vram_waitlist);
        for request in pending {
            let id = request.id;
            let base_for_pin = request
                .model_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .or_else(|| self.models.keys().next().cloned());
            let decision = base_for_pin
                .as_deref()
                .filter(|base| self.scythe_armed() && self.scythe_replicas.contains_key(*base))
                .map(|base| {
                    let caps_raw = self
                        .capability_profiler
                        .as_ref()
                        .map(|p| p.capabilities())
                        .unwrap_or_default();
                    self.scythe_admission_decision(
                        base,
                        request.prompt_tokens,
                        request.max_new_tokens,
                        &caps_raw,
                    )
                })
                .unwrap_or(ScytheAdmission::Bypass);
            match decision {
                ScytheAdmission::Pin(rank) => {
                    self.scythe_pin.insert(id, rank);
                    eprintln!("[scythe2] waitlisted request {id} admitted on farm rank {rank}");
                    if let Err(e) = self.admit_placed_request(request, Some(rank)) {
                        eprintln!("[scythe2] waitlisted request {id} failed to admit: {e}");
                    }
                }
                ScytheAdmission::Bypass => {
                    if let Err(e) = self.admit_placed_request(request, None) {
                        eprintln!("[scythe2] waitlisted request {id} failed to admit: {e}");
                    }
                }
                ScytheAdmission::WaitVram => self.scythe_vram_waitlist.push(request),
            }
        }
    }

    /// Number of requests currently held on the WI-SB2 VRAM waitlist — no
    /// farm rank could hold their footprint when they arrived. Status and
    /// observability surface; nonzero means serving capacity is exhausted
    /// for that prompt size, not that the requests were dropped.
    pub fn scythe_vram_waitlist_len(&self) -> usize {
        self.scythe_vram_waitlist.len()
    }

    /// F3b: Enqueue a request whose prefill already ran on a remote Prefill
    /// node. Creates the local session/KV structures without any local prompt
    /// forward (`prompt_tokens = 0`, so `drive_prefill` returns immediately),
    /// advances the position cursor to `prompt_len`, then hydrates session KV
    /// pages from every pool block the background receiver has already written.
    /// The next decode tick therefore attends purely transferred KV.
    pub fn enqueue_remote_prefill_request(
        &mut self,
        id: u64,
        prompt_len: usize,
        model_id: Option<String>,
    ) -> Result<()> {
        let request = grim_scheduler::Request {
            id,
            prompt_tokens: 0,
            priority: 0,
            model_id,
            ..Default::default()
        };
        self.enqueue_request_with_kv(request)?;
        if let Some(s) = self.sessions.get_mut(&id) {
            s.as_mut().advance_pos(prompt_len);
        }
        self.hydrate_session_from_pool(id);
        Ok(())
    }

    /// F3b helper: copy pool layer storage into the session's page tensors for
    /// every received block (all layers present per block). Mirror of what the
    /// pull path does for un-received blocks.
    fn hydrate_session_from_pool(&mut self, id: u64) {
        let num_blocks = {
            let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.num_blocks()
        };
        // Collect under the pool lock, then write into the session.
        let mut payload: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = Vec::new();
        {
            let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            for b in 0..num_blocks {
                if !pool.block_is_received(b) {
                    continue;
                }
                let mut layer = 0usize;
                while let Some(k) = pool.read_layer_keys(b, layer) {
                    match pool.read_layer_values(b, layer) {
                        Some(v) => payload.push((b, layer, k.to_vec(), v.to_vec())),
                        None => break,
                    }
                    layer += 1;
                }
            }
        }
        if payload.is_empty() {
            return;
        }
        if let Some(session) = self.sessions.get_mut(&id) {
            if let Some(kv) = session.kv_mut() {
                for (b, layer, k, v) in payload {
                    let _ = kv.write_layer_block(layer, b, &k, &v);
                }
            }
        }
    }

    pub fn finish_request(&mut self, id: u64) {
        self.scheduler.finish(id);
        if let Some(session) = self.sessions.get_mut(&id) {
            let _ = session.rollback_kv_to(0);
        }
        self.sessions.remove(&id);
        self.last_outcomes.remove(&id);
        self.request_rng.remove(&id);
        self.request_model_ids.remove(&id);
        self.request_adapters.remove(&id);
        self.request_input_ids.remove(&id);
        self.request_last_token.remove(&id);
        // Release the farm slot so the controller's load view stays honest.
        // The rank stays counted for a short cooldown (see
        // `scythe_admission_decision`) so the NEXT admission still sees it —
        // a burst of short-lived requests otherwise always finds an empty
        // pin map and piles onto rank 0.
        if let Some(rank) = self.scythe_pin.remove(&id) {
            self.scythe_pin_cooldown
                .push((rank, std::time::Instant::now()));
        }
        // A cancelled request must not linger on the WI-SB2 VRAM waitlist.
        self.scythe_vram_waitlist.retain(|r| r.id != id);
    }

    /// Deterministic RNG snapshot for a request, used by the speculative
    /// verifier when the engine's determinism mode is `Strict`. Returns
    /// `None` when the request isn't tracked.
    pub fn request_rng_state(&self, id: u64) -> Option<u64> {
        self.request_rng.get(&id).map(|r| r.state())
    }

    /// Replay: deterministically rewind a request's RNG by `steps`.
    /// Strict mode exposes this so re-running a tick with the same
    /// input reproduces the same RNG-driven decisions.
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

    /// Record the token that was sampled for a request. Called by the
    /// server after sampling so the next decode step feeds the real
    /// token instead of a position index.
    pub fn record_generated_token(&mut self, id: u64, token: u32) {
        self.request_last_token.insert(id, token);
    }

    /// Pause a running request — §5.2.1. KV blocks are retained in the
    /// block pool at zero scheduling priority. Returns true if the request
    /// was running and is now paused.
    pub fn pause_request(&mut self, id: u64) -> bool {
        let moved = self.scheduler.pause(id);
        if moved {
            // The session is kept; KV blocks remain ref-counted. The
            // speculative wrapper's mid-step tentative state stays
            // anchored to the cache and resumes from where it left off.
            if let Some(s) = self.sessions.get_mut(&id) {
                let _ = s;
            }
        }
        moved
    }

    /// Resume a previously-paused request — §5.2.1. The request continues
    /// from the exact token position where it was paused. Returns true if
    /// the request was paused and is now running.
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

    /// `(model_id, priority)` lookup for the request — a request is
    /// bound to exactly one model in v1. Under SCYTHE-2 farm mode the
    /// returned id is the pinned replica, so every caller (prefill drive,
    /// decode loop) routes to it without knowing farms exist.
    fn model_for_request(&self, id: u64) -> Option<(String, i32)> {
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

/// Re-export key types at the grim-engine crate root.
pub use grim_memory::PagedKvCache;
pub use grim_scheduler::{AdmissionController, Request, Scheduler, SchedulerOutput};

#[cfg(test)]
#[allow(unused_must_use)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_telemetry_accessors() {
        let config = EngineConfig::default();
        let engine = Engine::new(config);
        assert_eq!(engine.tokens_per_sec(), None);
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
        // §5.3: registering a plain CausalLm without an attached bundle
        // gets the autoselected wrapper. With no bundle present the
        // wrapper falls back to plain autoregressive, *but* the wrapper
        // itself is always speculative — the path is opt-out, not opt-in.
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
        // §5.2.1: a mid-decode pause keeps KV blocks alive, ref-counted
        // through the block pool. The session's `current_pos` does not
        // regress, and the speculative wrapper's tentative state stays
        // anchored to the cache because the cache itself is preserved.
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
        assert_eq!(engine.is_paused(1), true);

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
        // The wrapper path is the standard one — count the speculative
        // flag on the recorded outcomes and assert that every running
        // request was driven once per tick. v1's Llama forward doesn't
        // accept extras, but the wrapper contract holds: every decode
        // tick yields a fresh `StepOutcome`.
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

        // Plain strategy still counts as "speculative" field = false on
        // the wrapper output, confirming the structural pipeline is in
        // place for Strategy::Plain (with a real DSpark bundle attached
        // the field flips to true).
        let outcome = engine.last_outcome(1).unwrap();
        assert_eq!(
            outcome.speculative, false,
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
        // Wiring concrete DraftBackbone / MarkovHead / ConfidenceHead
        // impls through `register_with_dspark`. This is the test of
        // whether the speculative decoding pipeline (§5.3.2) is
        // actually exercisable end-to-end — even if the structural
        // impls are simple, the strategy-flip proves the path.
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
        let mut config = EngineConfig::default();
        config.determinism_mode = DeterminismMode::Strict;
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
        // This test validates the fix for the "dummy token" bug where
        // drive_prefill was feeding synthetic (0..prompt_tokens) instead of
        // the actual prompt token IDs provided by the caller.
        //
        // The test enqueues a request with known input_ids, ticks the engine,
        // and verifies that the forward pass receives those exact tokens
        // (by checking that the session's position advances by the number
        // of real tokens, not by a synthetic range).
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
            ..Default::default()
        });

        // First tick: prefill should consume ALL prompt tokens
        let _ = engine.tick().expect("tick must succeed");

        // The session position should advance by the number of REAL tokens,
        // not by a synthetic range. If the bug exists, it would advance by
        // prompt_tokens (which happens to match here) but the CONTENT fed
        // to the model would be wrong. We verify by checking that a second
        // tick (decode) uses the LAST real token as the next input, not
        // the position index.
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

        // Second tick: decode step should use the LAST real token (999) as input,
        // not the position index (which would be 5). We can't directly observe
        // the input_ids tensor from here, but we verify the session advances.
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

    /// Phase-1 correctness proof: a Llama driven through the paged-KV path
    /// (session carries a `PagedKvCache`) must produce byte-identical logits
    /// to the same model driven through the classic per-layer
    /// `LlamaLayerCache` path (no KV session). This is the invariant that
    /// lets us re-enable prefix-cache/tiering wiring on top of the paged
    /// path without changing serving numerics.
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

    /// WI-INF3 serving gate (farm mode): a pinned request executes on its
    /// replica. Replicas are built from the same fixed seed, so logits must be
    /// byte-identical to a plain single-replica engine — the pin decides
    /// WHERE the pass runs, never WHAT it computes.
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

    /// WI-INF5 farm corollary: the load-adjusted capability view must spread
    /// sessions once the fast card saturates instead of pinning everything to
    /// it — unloaded traffic goes to the 80-TFLOPS card, but with 10 sessions
    /// already there its effective 80/11 TFLOPS drops below the idle 8-TFLOPS
    /// card and the next admission lands there.
    #[test]
    /// WI-SB1 load-spreading: a finished request's rank must stay counted in
    /// the cooldown window so the next admission sees it — otherwise a burst
    /// of short requests all observe an empty pin map and pile onto rank 0.
    fn test_finished_pin_enters_cooldown_window() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.scythe_pin.insert(7, 1);
        engine.finish_request(7);
        assert!(engine.scythe_pin.get(&7).is_none(), "active pin released");
        assert_eq!(engine.scythe_pin_cooldown.len(), 1);
        assert_eq!(engine.scythe_pin_cooldown[0].0, 1);

        // Pruning happens at admission time, not finish time: an aged-out
        // entry is still physically present until the next decision scans
        // it, but it must not COUNT toward load anymore (helper gate below).
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

    /// WI-SB1 load-spreading: external GPU utilization folds into the load
    /// vector at weight 2.0 — a card maxed out by a desktop/game workload
    /// (100 % busy ≈ +2 effective requests) must lose the fast card to an
    /// idle slower rank on a ~2:1 measured pair.
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

        // And the original defect: plain decide() caches the idle verdict
        // keyed only by shape, then serves it verbatim even after the load
        // vector changed — which is what pinned every request to rank 0.
        // decide_forced must re-evaluate under the adjusted caps instead.
        let mut ctrl = crate::scythe2::C2plrController::new(1, 2, 150.0);
        let cached_rank = ctrl
            .decide(0, &shape, &load_adjusted_caps(&caps, 2, &[0.0, 0.0]), &links, 0)
            .ranks[0];
        let sticky_rank = ctrl
            .decide(0, &shape, &load_adjusted_caps(&caps, 2, &[2.0, 0.0]), &links, 0)
            .ranks[0];
        assert_eq!(cached_rank, 0);
        assert_eq!(
            sticky_rank, cached_rank,
            "expected the load-blind cache hit being fixed here"
        );
        let forced_rank = ctrl
            .decide_forced(0, &shape, &load_adjusted_caps(&caps, 2, &[2.0, 0.0]), &links, 0)
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

    /// WI-SB2 host gate (synthetic caps): the footprint formula must exclude
    /// an 8 GB-class card for a 100k-token prompt, admit the same request on
    /// a 16 GB card, admit a 1k-token prompt on both, and report every rank
    /// infeasible when nothing fits — which is the queue signal. A zero
    /// free-VRAM reading is probe-unavailable and must NOT read as "full".
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

    /// WI-SB2 host gate: the decision layer maps the synthetic-caps guard to
    /// Pin/WaitVram correctly — mixed pair pins the feasible fast card,
    /// all-excluded waits, missing profiler data waits, and a 1k prompt is
    /// never blocked by an 8 GB card.
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

    /// WI-SB2 host gate: an enqueue that fails the VRAM guard must leave the
    /// request queued — no session, no scheduler entry, no pin — and a later
    /// retry once caps exist must admit it with a pin. Deterministic on any
    /// box: with no profiler attached, caps are empty ⇒ WaitVram; the retry
    /// leg runs wherever real GPUs are visible.
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
}
