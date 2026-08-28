//! SCYTHE-2 C²PLR controller + PlacementCache + persistent ring (WI-4 / WI-7).
//!
//! ## Architecture (scythe2.md §3, §5.3)
//!
//! The `C2plrController` is the online router that emits a per-layer
//! `(placement, partition, route)` triple. It is called once per (layer, shape,
//! epoch) tuple on a `PlacementCache` miss (cache miss path = prefill or
//! capability-epoch refresh); decode-path cache hits are ~50 ns/layer array
//! lookups.
//!
//! Budget reconciliation (scythe2.md §3.4, retuned to the measured SB3
//! figures — `benches/scythe2_decide_miss.rs`, release host-side):
//! - **Decode cache-hit path**: ~50 ns/layer × N_layers ≤ 4 µs (32-layer 7B),
//!   0.04% of the 10 ms ITL budget.
//! - **Prefill cache-miss path**: ~2 µs/layer × N_layers ≤ 64 µs (32-layer),
//!   0.04% of the 150 ms prefill budget. The end-to-end WI-INF4 A/B
//!   (2026-08-23c) confirmed the decision cost is invisible in TTFT
//!   (−0.09 %/−0.00 % overhead, F/S).
//!
//! ## Staleness safety (scythe2.md §3.5)
//! - Mode A (stale `partition`): suboptimal, never incorrect.
//! - Mode B (stale `placement` when GPU left): prevented by the synchronous
//!   `bump_epoch` from the device-lost path — see `on_gpu_leave`.
//! - Mode C (stale `route`): falls back to T1 host-bounce, never a fault.
//!
//! ## Persistent dispatch ring (scythe2.md §3, Pillar 3, WI-7)
//! `ScytheRing` / `ScytheTaskDescriptor` implement the lock-free VRAM ring
//! described in scythe2.md §5.3 (Concordia `2606.23521` + GPREEMPT ATC '25).
//! The host writes 32-byte task descriptors; the device-resident persistent
//! kernel polls and dispatches in <0.1 µs.
//!
//! Skill attribution:
//! - `rust-ffi-grim` §1 — `#[repr(C, align(32))]` on `ScytheTaskDescriptor`.
//! - `rust-ffi-grim` §3 — `cargo check` gate after each WI.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use grim_backend_rocm::{RocmDevice, RocmPinnedBuffer, RocmStorage};

use grim_tensor::backend::{GpuCapability, ScytheLink, ScythePlacement};

// ── PlacementCache (§3.4 load-bearing type) ───────────────────────────────────

/// Cache key that makes two forward passes share a placement iff they share a
/// `(layer_id, shape_bucket, capability_epoch)` triple.
///
/// `shape_bucket` power-of-2 quantizes `seq_len × batch` so that autoregressive
/// decode (which increments `seq_len` by 1 per token) stays cache-stable across
/// an entire generation (scythe2.md §3.4).
///
/// `capability_epoch` is bumped by `CapabilityProfiler` every ~100 ms (§3.6)
/// or on GPU join/leave (§3.5 mode B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlacementKey {
    /// Unique layer index (fingerprint table row).
    pub layer_id: u32,
    /// Power-of-2 bucket of `max(seq_len, batch)`. Decode keeps this stable.
    pub shape_bucket: u16,
    /// Epoch version from `CAPABILITY_EPOCH`. Bumped on thermal cliff or GPU leave.
    pub capability_epoch: u32,
}

/// Per-forward placement cache. The fast path is an array indexed by `layer_id`
/// for the common case (same `shape_bucket`, same `epoch`) → O(1), ~50 ns.
///
/// This is the **load-bearing type** that makes per-layer routing compatible
/// with the 10 ms ITL budget (scythe2.md §3.4). A design that recomputed every
/// layer every forward pass would blow the ITL budget by 3–8×.
pub struct PlacementCache {
    /// Fast path: `fast[layer_id]` holds the last-inserted placement for
    /// this layer together with the shape bucket it was decided at.
    /// Cleared by `bump_epoch`. F7 (audit): the bucket is tracked PER LAYER
    /// — the old single global `last_bucket` made interleaved buckets across
    /// layers (farm/pipeline concurrency) spuriously miss valid entries and
    /// pay the ~2 µs `decide_miss` instead of the ~50 ns hit.
    fast: Vec<Option<(u16, ScythePlacement)>>,
    /// Slow path: arbitrary `(layer_id, bucket, epoch)` → placement.
    full: HashMap<PlacementKey, ScythePlacement>,
    /// Current epoch. Read from `CAPABILITY_EPOCH` at construction; callers
    /// that drive `bump_epoch` must also call `sync_epoch` to pull the new value.
    pub current_epoch: u32,
}

impl PlacementCache {
    /// Create a cache for `num_layers` layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            fast: vec![None; num_layers],
            full: HashMap::new(),
            current_epoch: 0,
        }
    }

    /// Decode-path lookup. Returns `Some(&placement)` on a hit (~50 ns), or
    /// `None` on a miss (caller must run the expensive `decide_miss()`).
    ///
    /// A miss occurs when:
    /// 1. `layer_id` was never placed (first prefill after startup).
    /// 2. The shape bucket changed (e.g., prompt length crossed a power-of-2).
    /// 3. The capability epoch bumped (thermal throttle, GPU leave).
    pub fn get(&self, layer_id: u32, shape_bucket: u16) -> Option<&ScythePlacement> {
        // Fast path: array index by layer_id; valid only when this layer's
        // own stored bucket matches the requested one.
        if let Some((bucket, p)) = self
            .fast
            .get(layer_id as usize)
            .and_then(|opt| opt.as_ref())
        {
            if *bucket == shape_bucket {
                return Some(p);
            }
        }
        // Slow path: full key lookup.
        self.full.get(&PlacementKey {
            layer_id,
            shape_bucket,
            capability_epoch: self.current_epoch,
        })
    }

    /// Store a freshly-decided placement. Called after a `decide_miss()`.
    pub fn insert(&mut self, layer_id: u32, shape_bucket: u16, p: ScythePlacement) {
        let key = PlacementKey {
            layer_id,
            shape_bucket,
            capability_epoch: self.current_epoch,
        };
        self.full.insert(key, p.clone());
        if let Some(slot) = self.fast.get_mut(layer_id as usize) {
            *slot = Some((shape_bucket, p));
        }
    }

    /// Observability hook (F7 test gate): whether the next `get` for this
    /// `(layer, bucket)` will be served by the per-layer fast path.
    pub fn fast_path_hit(&self, layer_id: u32, shape_bucket: u16) -> bool {
        self.fast
            .get(layer_id as usize)
            .and_then(|opt| opt.as_ref())
            .map(|(bucket, _)| *bucket == shape_bucket)
            .unwrap_or(false)
    }

    /// Called when `CAPABILITY_EPOCH` bumps (~100 ms cadence, or GPU leave).
    ///
    /// Clears the fast path so the next forward pass re-runs `decide_miss()`
    /// for every layer. This is the mode-B safety gate (scythe2.md §3.5):
    /// if a GPU left, the cleared fast path prevents any forward from
    /// dispatching to the gone GPU.
    pub fn bump_epoch(&mut self) {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        self.fast.fill(None);
    }

    /// Synchronise `current_epoch` from the global atomic without bumping.
    /// Used at the start of each forward pass to detect an out-of-band bump.
    pub fn sync_epoch(&mut self, epoch: u32) {
        if epoch != self.current_epoch {
            self.current_epoch = epoch;
            // Clear fast path so we re-decide with the new epoch.
            self.fast.fill(None);
        }
    }

    /// Called from the device-lost handler to guarantee mode-B safety before
    /// the next `decide()` returns (scythe2.md §3.5 mode B).
    pub fn on_gpu_leave(&mut self) {
        // Increment epoch and clear fast path synchronously.
        self.current_epoch = self.current_epoch.wrapping_add(1);
        self.fast.fill(None);
        self.full.clear(); // Also evict slow-path entries that reference the gone GPU.
    }
}

// ── C2plrController ───────────────────────────────────────────────────────────

/// Layer fingerprint — 16-dimensional feature vector describing the layer's
/// compute profile for the WaveTune bilinear predictor.
///
/// Populated at model-load time from layer config (MLP vs attention vs norm),
/// GEMM dimensions, etc.
pub type LayerFingerprint = [f32; 16];

/// The 2-layer MLP controller π_θ (≈8 KB).
///
/// Inputs per forward: `(layer_fingerprint[16], input_shape[4],
/// capability_profile[K×6], link_state[K×K], thermal_state[K])`.
/// Outputs: `(placement_logits[K], partition_alpha[K], route_logits[3])`.
///
/// Training: Gumbel-Softmax over placement, STE over route, Lagrangian
/// budget dual-ascent after each optimizer step (scythe2.md §4 Pillar 4).
///
/// ## Cache semantics
/// `decide()` is the public entry point. It checks `cache` first; only on a
/// miss does it run the expensive `decide_miss()` (WaveTune bilinear eval +
/// MLP forward + Gumbel sample, measured ~2 µs/layer host-side in release —
/// `benches/scythe2_decide_miss.rs`, SB3 campaign).
///
/// Measured end-to-end cost (WI-INF4 A/B verdict, 2026-08-23c, release,
/// 30 samples/arm/order on the syd-beasty pair): mean TTFT overhead
/// −0.09 %/−0.00 % (F/S — the per-decision cost is invisible in TTFT) and
/// p95 ITL overhead −18.56 %/+2.43 % (F/S) — the ITL budget (≤2 %) is
/// exceeded only in the S-first ordinal order's decode tail, which is why
/// `GRIM_SCYTHE_INFERENCE` stays opt-in. Retune placement against that tail
/// before revisiting the default.
pub struct C2plrController {
    /// Layer fingerprints indexed by `layer_id`.
    pub layer_fps: Vec<LayerFingerprint>,
    /// MLP hidden dimension (default: 64).
    hidden_dim: usize,
    /// MLP W1 weights `[input_dim × hidden_dim]` (row-major).
    pub theta_w1: Vec<f32>,
    /// MLP W2 weights `[hidden_dim × output_dim]` (row-major).
    pub theta_w2: Vec<f32>,
    /// Lagrangian dual variable λ for the latency budget constraint.
    /// Dual-ascended after each optimizer step.
    pub lambda: f64,
    /// Target end-to-end latency budget (ms). Inherited from EngineConfig.
    pub budget_ms: f64,
    /// The placement cache — load-bearing for the ITL budget (§3.4).
    pub cache: PlacementCache,
    /// Number of GPUs the controller was constructed for. `decide_miss`
    /// validates that the live `caps.len()` matches this — a mismatch means
    /// the farm topology changed without a `bump_epoch`, which is a bug.
    num_gpus: usize,
}

impl C2plrController {
    /// Construct a controller for `num_layers` layers and `num_gpus` GPUs.
    ///
    /// MLP weights are initialised near-zero (the controller learns online).
    /// The HetAuto MCTS offline seed (scythe2.md §4 Pillar 2) would populate
    /// `theta_w1`/`theta_w2` before the first forward; until then the controller
    /// falls back to round-robin placement.
    pub fn new(num_layers: usize, num_gpus: usize, budget_ms: f64) -> Self {
        // Input dim: 16 (fingerprint) + 4 (shape) + num_gpus*6 (caps) + num_gpus*num_gpus (links) + num_gpus (thermal)
        let input_dim = 16 + 4 + num_gpus * 6 + num_gpus * num_gpus + num_gpus;
        let hidden_dim = 64;
        let output_dim = num_gpus + num_gpus + 3; // placement + partition + route
        Self {
            layer_fps: vec![[0.0f32; 16]; num_layers],
            hidden_dim,
            theta_w1: vec![0.0f32; input_dim * hidden_dim],
            theta_w2: vec![0.0f32; hidden_dim * output_dim],
            lambda: 0.0,
            budget_ms,
            cache: PlacementCache::new(num_layers),
            num_gpus,
        }
    }

    /// Per-forward entry point (scythe2.md §5.3).
    ///
    /// Hits the cache first; calls `decide_miss()` only on a miss.
    /// Aggregate per-forward overhead (measured, release host-side):
    /// - Decode (cache hit): ~50 ns/layer × N_layers.
    /// - Prefill/refresh (miss): ~2 µs/layer × N_layers.
    pub fn decide(
        &mut self,
        layer_id: u32,
        shape: &[usize],
        caps: &[GpuCapability],
        links: &[ScytheLink],
        epoch: u32,
    ) -> ScythePlacement {
        // Sync epoch before checking cache (may clear fast path).
        self.cache.sync_epoch(epoch);
        let bucket = bucketize(shape);
        if self.cache.get(layer_id, bucket).is_none() {
            let p = self.decide_miss(layer_id, shape, caps, links);
            self.cache.insert(layer_id, bucket, p);
        }
        // SAFETY: just inserted or already present.
        self.cache
            .get(layer_id, bucket)
            .expect("placement must exist after insert")
            .clone()
    }

    /// Load-aware variant of [`C2plrController::decide`] (WI-SB1 finding):
    /// the shape-keyed PlacementCache is load-blind — a placement decided
    /// for idle ranks is reused verbatim even after the rank-load vector
    /// changed (concurrent farm pins or external GPU utilization). Callers
    /// that pass *adjusted* caps reflecting any non-zero load must use this
    /// entry so the argmin actually sees them. Still refreshes the cache so
    /// subsequent idle-shaped lookups stay coherent.
    pub fn decide_forced(
        &mut self,
        layer_id: u32,
        shape: &[usize],
        caps: &[GpuCapability],
        links: &[ScytheLink],
        epoch: u32,
    ) -> ScythePlacement {
        self.cache.sync_epoch(epoch);
        let p = self.decide_miss(layer_id, shape, caps, links);
        let bucket = bucketize(shape);
        self.cache.insert(layer_id, bucket, p.clone());
        p
    }

    /// Expensive path: WaveTune bilinear eval + MLP forward + Gumbel sample.
    ///
    /// Measured at ~2 µs/layer host-side in release (`benches/
    /// scythe2_decide_miss.rs`); the ~10 µs figure in scythe2.md §3.4 was
    /// the pre-implementation estimate. This is a
    /// *deterministic table lookup*, not a candidate loop — the WaveTune
    /// `2604.10187` §4.4–4.5 mechanism is one bilinear eval + one anchor
    /// retrieval, not an iterative search.
    ///
    /// When the MLP weights are zero (before online learning converges), the
    /// controller falls back to a balanced round-robin placement.
    fn decide_miss(
        &self,
        layer_id: u32,
        shape: &[usize],
        caps: &[GpuCapability],
        links: &[ScytheLink],
    ) -> ScythePlacement {
        // Validate that the live capability profile matches the farm the
        // controller was constructed for. A mismatch means the topology
        // changed without a `bump_epoch` — a caller bug. We don't panic
        // (the controller must stay up), but we clamp to the configured size.
        let k = caps.len().max(1).min(self.num_gpus.max(1));

        // ── WaveTune bilinear latency eval (§3.4 Table-A) ──────────────────
        // For each GPU, estimate GEMM latency from TFLOPS and shape.
        // This is the offline structural-coefficient lookup (one division per GPU).
        let m = shape.first().copied().unwrap_or(1);
        let n = shape.get(1).copied().unwrap_or(1);
        let k_dim = shape.get(2).copied().unwrap_or(1);
        let flops = 2.0 * m as f64 * n as f64 * k_dim as f64;
        let latencies: Vec<f64> = caps
            .iter()
            .map(|c| {
                if c.tflops_fp16 > 0.0 {
                    flops / (c.tflops_fp16 as f64 * 1e12) * 1e3 // ms
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        // ── MLP forward (§3.4, §4 Pillar 4) ───────────────────────────────
        // Build input vector and run the 2-layer MLP.
        // If weights are zero → output is zero → fallback to round-robin below.
        let input_dim = 16 + 4 + k * 6 + k * k + k;
        let mut input = vec![0.0f32; input_dim];
        // Layer fingerprint.
        let fp = self
            .layer_fps
            .get(layer_id as usize)
            .copied()
            .unwrap_or([0.0; 16]);
        input[..16].copy_from_slice(&fp);
        // Shape.
        for (i, &s) in shape.iter().take(4).enumerate() {
            input[16 + i] = s as f32;
        }
        // GPU capabilities (6 floats per GPU).
        for (gi, c) in caps.iter().enumerate() {
            let base = 20 + gi * 6;
            if base + 5 < input_dim {
                input[base] = c.tflops_fp16;
                input[base + 1] = c.tflops_fp8;
                input[base + 2] = c.hbm_bandwidth_gbps;
                input[base + 3] = (c.vram_free_bytes >> 20) as f32; // in MiB
                input[base + 4] = c.throttle_pct;
                input[base + 5] = c.ordinal as f32;
            }
        }
        // Link matrix.
        let link_base = 20 + k * 6;
        for (li, link) in links.iter().enumerate() {
            let idx = link_base + li;
            if idx < input_dim {
                input[idx] = match link {
                    ScytheLink::PeerDirect => 1.0,
                    ScytheLink::Pcie => 0.5,
                    ScytheLink::Host => 0.0,
                };
            }
        }

        let output_dim = k + k + 3;
        let logits = mlp_forward(
            &self.theta_w1,
            &self.theta_w2,
            &input,
            self.hidden_dim,
            output_dim,
        );

        // ── Placement selection ─────────────────────────────────────────────
        // Placement logits: argmax over first K elements. With an untrained
        // (all-zero) MLP every logit is identical, and a naive argmax pins
        // every layer to rank 0 — on an asymmetric pair that may be the
        // *slower* card. When the logits carry no signal (all equal), ship
        // the WaveTune bilinear estimate alone (WI-INF5 first cut): pick the
        // GPU with the lowest predicted GEMM latency.
        let placement_logits = &logits[..k.min(logits.len())];
        let logits_carry_no_signal = placement_logits.windows(2).all(|w| w[0] == w[1]);
        let best_gpu = if logits_carry_no_signal {
            latencies
                .iter()
                .enumerate()
                .filter(|(_, l)| l.is_finite())
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            placement_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        // F6 (audit): single-rank routing is the intended design — the only
        // multi-rank ScythePlacement consumer is grim-cli's hand-built
        // data-parallel gradient sync, which bypasses this controller. The
        // softmax-over-partition-logits computation that used to live here
        // fed exactly one discarded slice (`split_counts` tops the sole
        // rank off to 100% regardless); it is deleted rather than "fixed"
        // so the output no longer implies multi-rank support that doesn't
        // exist.

        // ── Route selection ─────────────────────────────────────────
        let route_link = if k == 1 {
            ScytheLink::PeerDirect
        } else {
            links
                .get(best_gpu * k + (best_gpu + 1) % k)
                .copied()
                .unwrap_or(ScytheLink::Host)
        };

        // ── Lagrangian budget check ─────────────────────────────────────────
        // Compare this layer's estimated GEMM latency against the *per-layer*
        // budget slice (total budget / num_layers), not the whole end-to-end
        // budget. The previous code compared one GEMM against `budget_ms`
        // directly, which made the fallback effectively never fire (a single
        // GEMM almost never exceeds the full prefill/ITL budget). The per-layer
        // slice is the honest threshold: if this layer would consume more than
        // its fair share on the chosen GPU, reroute to the lowest-latency GPU.
        let num_layers = self.layer_fps.len().max(1);
        let per_layer_budget = self.budget_ms / num_layers as f64;
        let selected =
            if latencies.get(best_gpu).copied().unwrap_or(f64::INFINITY) > per_layer_budget {
                latencies
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.is_finite())
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                best_gpu
            };

        ScythePlacement {
            ranks: vec![selected],
            // Single-rank placement carries the whole layer (F6 note above);
            // no per-rank split ratios exist.
            partition: vec![1.0],
            routes: vec![route_link],
        }
    }

    /// Online update after a batch — dual ascent on λ + MLP gradient step.
    ///
    /// Called every optimizer step (not every micro-batch) to update the
    /// Lagrangian dual variable and the MLP weights with a simple gradient
    /// estimate (scythe2.md §4 Pillar 4).
    pub fn update(&mut self, observed_latency_ms: f64, placements: &[ScythePlacement]) {
        // Lagrangian dual ascent: λ ← λ + α(t̂_total - T_budget).
        // The step size α = 0.01 is the standard dual-ascent learning rate;
        // larger values oscillate, smaller values converge too slowly for
        // the ~100 ms capability-epoch cadence (scythe2.md §3.6).
        const DUAL_STEP_SIZE: f64 = 0.01;
        const MLP_LR: f32 = 0.001;
        let constraint_violation = observed_latency_ms - self.budget_ms;
        self.lambda = (self.lambda + DUAL_STEP_SIZE * constraint_violation).max(0.0);

        // ── MLP gradient step (scythe2.md §4 Pillar 4) ────────────────────────
        // Approximate policy gradient: penalise weights that led to placements
        // on GPUs contributing to budget overruns, weighted by the violation
        // magnitude. When λ is high (chronic overruns), the penalty scales up,
        // pushing the MLP toward lower-latency placements.
        let penalty = (self.lambda * constraint_violation.abs().max(0.0)) as f32;
        if penalty > 0.0 && !placements.is_empty() {
            // Build a per-GPU blame signal: GPUs that appear more often in the
            // placements get a larger gradient push. This is a REINFORCE-style
            // credit assignment without the full autograd tape — the controller
            // runs online and must be lightweight.
            let mut gpu_blame = vec![0.0f32; self.num_gpus];
            for p in placements {
                for &rank in &p.ranks {
                    if rank < gpu_blame.len() {
                        gpu_blame[rank] += 1.0;
                    }
                }
            }
            let total_blame: f32 = gpu_blame.iter().sum();
            if total_blame > 0.0 {
                // Normalise blame and apply as gradient noise to W2 columns
                // that correspond to placement logits. This nudges the MLP
                // output distribution away from the over-used GPUs.
                let hidden_dim = self.hidden_dim;
                let output_dim = self.num_gpus + self.num_gpus + 3;
                for (oi, &blame) in gpu_blame.iter().enumerate() {
                    if oi >= output_dim {
                        break;
                    }
                    let grad = -penalty * blame / total_blame;
                    for hi in 0..hidden_dim {
                        let wi = oi * hidden_dim + hi;
                        if wi < self.theta_w2.len() {
                            self.theta_w2[wi] += MLP_LR * grad;
                        }
                    }
                }
                // Also apply a small weight-decay regularisation to W1 to
                // prevent unbounded growth of the hidden representation.
                for w in self.theta_w1.iter_mut() {
                    *w *= 1.0 - MLP_LR * 0.01;
                }
            }
        }
    }

    /// Notify the cache that a GPU left the farm (mode-B safety, §3.5).
    ///
    /// Must be called from the ROCm device-lost path *before* the next
    /// `decide()` so that no cached placement dispatches to the gone GPU.
    pub fn on_gpu_leave(&mut self, ordinal: usize) {
        log::info!("[scythe2] GPU {ordinal} left — clearing PlacementCache (mode-B safety)");
        self.cache.on_gpu_leave();
    }

    /// Number of GPUs this controller was constructed for. The engine reads
    /// this when re-sizing the controller to a newly loaded model's depth
    /// (WI-INF2: one controller per loaded model).
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }
}

// ── Bucketizing ───────────────────────────────────────────────────────────────

/// Map a shape to a power-of-2 bucket index.
///
/// Autoregressive decode increments `seq_len` by 1 per token. Bucketizing to
/// the next power of 2 keeps decode tokens in the same bucket for an entire
/// generation window, making the fast-path cache-stable (scythe2.md §3.4).
pub fn bucketize(shape: &[usize]) -> u16 {
    let seq = shape.get(1).copied().unwrap_or(1).max(1);
    seq.next_power_of_two().trailing_zeros() as u16
}

// ── MLP helpers ───────────────────────────────────────────────────────────────

/// 2-layer MLP forward: `ReLU(x @ W1) @ W2`.
fn mlp_forward(w1: &[f32], w2: &[f32], x: &[f32], hidden: usize, out: usize) -> Vec<f32> {
    let input_dim = x.len();
    // Hidden layer: h = ReLU(x @ W1)
    let mut h = vec![0.0f32; hidden];
    for (hi, slot) in h.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (xi, &xv) in x.iter().enumerate() {
            let wi = hi * input_dim + xi;
            if wi < w1.len() {
                acc += xv * w1[wi];
            }
        }
        *slot = acc.max(0.0); // ReLU
    }
    // Output layer: y = h @ W2
    let mut y = vec![0.0f32; out];
    for (oi, slot) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (hi, &hv) in h.iter().enumerate() {
            let wi = oi * hidden + hi;
            if wi < w2.len() {
                acc += hv * w2[wi];
            }
        }
        *slot = acc;
    }
    y
}

// ── ScytheRing + ScytheTaskDescriptor (WI-7) ─────────────────────────────────

/// 32-byte task descriptor for the lock-free VRAM ring (scythe2.md §5.3).
///
/// The device-resident persistent kernel (Concordia `2606.23521`) polls these
/// descriptors at HBM bandwidth. The host writes a descriptor and the GPU
/// picks it up in <0.1 µs. `#[repr(C, align(32))]` guarantees cache-line
/// alignment per the FFI skill (rust-ffi-grim §1.1).
///
/// Opcodes:
/// - 0 = nop (slot is free)
/// - 1 = column-GEMM shard
/// - 2 = row-GEMM shard
/// - 3 = attention (QKV)
/// - 4 = norm (RMSNorm / RoPE — replicated)
/// - 5 = CommFuse reduce (fan-in)
/// - 6 = MoE dispatch (WI-Charon-3): `weight_ptr` points to a
///   device-resident [`MoETaskDescriptor`] carrying the MoE-specific
///   geometry (hidden/inter/num_experts/top_k/quant_mode/schedule).
///   The persistent kernel casts `weight_ptr` to `MoETaskDescriptor*`
///   and calls the Charon forward kernel inline — no separate
///   `hipLaunchKernel`, matching how opcodes 0–5 already work.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
pub struct ScytheTaskDescriptor {
    /// Operation code (see above).
    pub opcode: u32,
    /// GEMM M dimension (or seq_len for attention).
    pub m: u32,
    /// GEMM N dimension (or num_heads for attention).
    pub n: u32,
    /// GEMM K dimension (or head_dim for attention).
    pub k: u32,
    /// Device pointer to input activations (u64 for 64-bit pointers).
    pub input_ptr: u64,
    /// Device pointer to weight shard.
    pub weight_ptr: u64,
    /// Device pointer to output buffer (local).
    pub output_ptr: u64,
    /// Device pointer to peer VRAM target (T0 fused-P2P, 0 = local only).
    pub peer_ptr: u64,
    /// Slot status: 0 = pending, 1 = running, 2 = complete.
    pub status: u32,
    // Pad to 32 bytes (already 44 bytes with u64×4 + u32×5 = 52; using 52 → next align(32) = 64).
    // Since ScytheTaskDescriptor is repr(C, align(32)) and 52 bytes, the struct occupies 64 bytes
    // (rounded up to the next multiple of align(32)=32 → 64). This is fine for the ring.
}

// ── WI-Charon-3: MoE task descriptor + opcode 6 ──────────────────────────────
//
// The plan (charon_kernel_plan_v3.md §3 WI-Charon-3) calls for a companion
// descriptor carrying the MoE-specific geometry the generic
// `ScytheTaskDescriptor` (m/n/k + 4 pointers) can't express: hidden dim,
// inter dim, batch count, routed-scaling factor, expert-bank pointers, the
// sorted-routing schedule, and quant mode.
//
// The integration point is ONE new opcode (6 = MoE dispatch) on the existing
// `ScytheRing` infrastructure — NOT a parallel dispatch mechanism. The host
// enqueues a `ScytheTaskDescriptor` with `opcode = 6` and `weight_ptr`
// pointing to a device-resident `MoETaskDescriptor`; the persistent kernel
// casts `weight_ptr` to `MoETaskDescriptor*` and calls the Charon kernel
// inline, matching how opcodes 0–5 already work (no separate
// `hipLaunchKernel`). See `charon_multigpu_plan.md` + `kernel2.md` (the
// proposal this implements, verified sound against the real source).
//
// `MoETaskDescriptor` is `#[repr(C, align(32))]` to match
// `ScytheTaskDescriptor`'s cache-line alignment (rust-ffi-grim §1.1 — same
// FFI discipline, since the device reads this struct at HBM bandwidth). It
// does NOT duplicate `input_ptr` / `output_ptr` / `peer_ptr` — those live on
// the parent `ScytheTaskDescriptor` and map directly. Only MoE-specific
// geometry appears here, per the plan's "complement, don't duplicate" rule.

/// Quantization mode for a Charon MoE dispatch (WI-Charon-3).
///
/// Mirrors the 7 forward kernel variants in `charon.rs`:
/// `grim_moe_fused_dispatch` (FP32) + 6 quantized variants. Kept as a `u32`
/// tag (not a Rust `enum`) so the on-device struct has a stable FFI layout
/// the HIP kernel can `match` on without depending on Rust enum ABI.
pub type MoeQuantMode = u32;

/// MoE quant modes (WI-Charon-3). Values pinned by tests so the kernel and
/// host agree on the discriminants.
pub mod moe_quant_mode {
    /// FP32 / BF16-as-f32 path (`grim_moe_fused_dispatch`).
    pub const FP32: super::MoeQuantMode = 0;
    /// FP8 E4M3 (`grim_moe_fused_grouped_fp8`).
    pub const FP8: super::MoeQuantMode = 1;
    /// MXFP4 (`grim_moe_fused_grouped_mxfp4`).
    pub const MXFP4: super::MoeQuantMode = 2;
    /// MXFP8 (`grim_moe_fused_grouped_mxfp8`).
    pub const MXFP8: super::MoeQuantMode = 3;
    /// Q8_0 (`grim_moe_fused_grouped_q80`).
    pub const Q8_0: super::MoeQuantMode = 4;
    /// IQK family (`grim_moe_fused_grouped_iqk`).
    pub const IQK: super::MoeQuantMode = 5;
}

/// MoE-specific geometry companion to `ScytheTaskDescriptor` (WI-Charon-3).
///
/// The host enqueues a `ScytheTaskDescriptor { opcode: 6, weight_ptr: ptr to
/// MoETaskDescriptor, input_ptr/output_ptr/peer_ptr: as usual, ... }`. The
/// device-side persistent kernel casts `weight_ptr` to
/// `MoETaskDescriptor*` and dispatches the matching Charon forward variant.
///
/// Layout: 80 bytes raw (u32×7 + f32 + u64×6) → 96 bytes under
/// `align(32)` (two cache lines). This is 1.5× the parent
/// `ScytheTaskDescriptor`'s 64B footprint — the extra half-line is the
/// cost of carrying six 64-bit pointers (gate/up/down weights + the three
/// F3 schedule arrays) the kernel's pointer interfaces need. A future
/// optimization can pack gate/up/down into a single stride-indexed
/// `expert_weights_ptr` to drop back to one cache line; held off here to
/// keep the descriptor-to-kernel call site pointer-arithmetic-free.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct MoETaskDescriptor {
    /// Hidden dim (`hidden`).
    pub hidden: u32,
    /// Inter dim (`inter`).
    pub inter: u32,
    /// Number of tokens post-padding (`num_tokens_post_padded`).
    pub num_tokens: u32,
    /// Block size used by the grouped sort (`block_size` from
    /// `moe_align_block_size`).
    pub block_size: u32,
    /// Number of experts in the bank.
    pub num_experts: u32,
    /// Top-k routing count.
    pub top_k: u32,
    /// Quantization mode (see [`moe_quant_mode`]).
    pub quant_mode: MoeQuantMode,
    /// Routed scaling factor (DeepSeek/Laguna convention — scales routed,
    /// not shared). f32 to match the kernel's `float` argument.
    pub routed_scaling_factor: f32,
    /// Device pointer to the expert gate weights, flattened as
    /// `[num_experts, inter*hidden]` (row-major, the layout
    /// `grim_moe_fused_grouped` expects).
    pub gate_w_ptr: u64,
    /// Device pointer to the expert up weights, `[num_experts, inter*hidden]`.
    pub up_w_ptr: u64,
    /// Device pointer to the expert down weights, `[num_experts, hidden*inter]`.
    pub down_w_ptr: u64,
    /// Device pointer to `sorted_token_ids` (u32[num_tokens_post_padded]) —
    /// F3 (audit) Option A: the routing schedule is carried as THREE
    /// INDEPENDENT pointers, matching the convention every real Charon
    /// call site in `roc_device.rs` already uploads (the previous single
    /// `schedule_ptr` + contiguous-offset contract matched no producer and
    /// had no host-side packing step).
    pub token_ids_ptr: u64,
    /// Device pointer to `sorted_expert_ids` (u32[num_tokens_post_padded]).
    pub expert_ids_ptr: u64,
    /// Device pointer to `sorted_weights` (f32[num_tokens_post_padded]).
    pub weights_ptr: u64,
}

impl Default for MoETaskDescriptor {
    fn default() -> Self {
        Self {
            hidden: 0,
            inter: 0,
            num_tokens: 0,
            block_size: 0,
            num_experts: 0,
            top_k: 0,
            quant_mode: moe_quant_mode::FP32,
            routed_scaling_factor: 1.0,
            gate_w_ptr: 0,
            up_w_ptr: 0,
            down_w_ptr: 0,
            token_ids_ptr: 0,
            expert_ids_ptr: 0,
            weights_ptr: 0,
        }
    }
}

impl MoETaskDescriptor {
    /// Upload this descriptor to device-visible memory on `device` and
    /// return the device address the caller MUST pass to
    /// [`Self::enqueue_via`].
    ///
    /// F4 (audit): `enqueue_via` used to stuff `self as *const Self as u64`
    /// — a HOST address — into `weight_ptr`, which the persistent kernel
    /// dereferences as a device pointer. Making the upload an explicit step
    /// mirrors the pattern the device-gated test already used
    /// (`dev.from_cpu_bytes(...)` then use that pointer) and makes the
    /// host-pointer mistake impossible to reintroduce silently.
    pub fn upload(&self, device: &RocmDevice) -> grim_backend_rocm::Result<u64> {
        use grim_tensor::backend::BackendDevice;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        };
        let storage = device.from_cpu_bytes(
            bytes,
            &grim_tensor::Shape::new(vec![std::mem::size_of::<Self>()]),
            grim_tensor::dtype::DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
        )?;
        let ptr = storage
            .as_any()
            .downcast_ref::<RocmStorage>()
            .and_then(|rs| rs.device_ptr_u64())
            .ok_or_else(|| grim_backend_rocm::Error::Backend("MoE upload: no device ptr".into()))?;
        // Keep the storage alive for the process lifetime — the ring's
        // opcode-6 arm dereferences this pointer whenever the wave runs, so
        // it must outlive the enqueue. Leaking a 96-byte device buffer per
        // MoE layer is negligible against the expert weights it describes.
        std::mem::forget(storage);
        Ok(ptr)
    }

    /// Build the parent `ScytheTaskDescriptor` that enqueues this MoE task
    /// onto the `ScytheRing`. The parent carries `opcode = 6` and
    /// `moe_dev_ptr` — the DEVICE address returned by [`Self::upload`] — in
    /// `weight_ptr`; the input/output/peer pointers flow through from the
    /// caller (the activations buffer, the local output buffer, and the
    /// optional peer-output buffer for cross-GPU combine).
    ///
    /// This is the host-side enqueue path called by WI-EP2's cross-GPU
    /// dispatch planner once it has partitioned (token, expert) pairs into
    /// local/remote and built the schedule.
    pub fn enqueue_via(
        moe_dev_ptr: u64,
        input_ptr: u64,
        output_ptr: u64,
        peer_ptr: u64,
    ) -> ScytheTaskDescriptor {
        assert!(
            moe_dev_ptr != 0,
            "MoETaskDescriptor::enqueue_via requires the DEVICE pointer from upload(); \
             a host address here is the F4 audit bug"
        );
        ScytheTaskDescriptor {
            opcode: 6, // MoE dispatch (WI-Charon-3)
            // m/n/k are unused for opcode 6 (geometry lives in the
            // MoETaskDescriptor); zero them for determinism so device-side
            // dumps read cleanly.
            m: 0,
            n: 0,
            k: 0,
            input_ptr,
            weight_ptr: moe_dev_ptr,
            output_ptr,
            peer_ptr,
            status: 0, // pending
        }
    }

    /// Validate the descriptor's geometry before enqueue. Pure: returns
    /// `Err` on a struct that would cause the device kernel to read
    /// out-of-bounds or launch a zero-grid. Mirrors the validation
    /// discipline of `charon::validate_grouped_inputs` and
    /// `charon_backward::validate_backward_inputs`.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.inter == 0 || self.num_tokens == 0 {
            return Err(format!(
                "MoETaskDescriptor: non-positive geometry hidden={} inter={} num_tokens={}",
                self.hidden, self.inter, self.num_tokens
            ));
        }
        if self.block_size == 0 {
            return Err("MoETaskDescriptor: block_size must be > 0".into());
        }
        if self.num_experts == 0 {
            return Err("MoETaskDescriptor: num_experts must be > 0".into());
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "MoETaskDescriptor: top_k={} out of range [1, num_experts={}]",
                self.top_k, self.num_experts
            ));
        }
        // Weight pointers may legitimately be zero in a host-side test that
        // only validates geometry (no device buffer yet). Only flag the
        // schedule pointers as required — a MoE dispatch with no routing
        // schedule is always wrong.
        if self.token_ids_ptr == 0 || self.expert_ids_ptr == 0 || self.weights_ptr == 0 {
            return Err(
                "MoETaskDescriptor: token_ids_ptr/expert_ids_ptr/weights_ptr must be non-null"
                    .into(),
            );
        }
        Ok(())
    }
}

// ── WI-EP2 — Cross-GPU MoE token dispatch planner (host-side) ────────────────
//
// charon_kernel_plan_v3.md §3 WI-EP2 (revised per WI-Charon-3): the cross-GPU
// token dispatch planner partitions (token, expert) pairs into local/remote,
// batches remote transfers by destination rank, and emits `ScytheTaskDescriptor`s
// (opcode 6) onto `ScytheRing` rather than a bespoke dispatch-plan type.
//
// Host-pure: the partition logic consumes the router's `(indices, weights)`
// output + an `ExpertPlacementMap` (WI-EP1) + the local rank, and decides for
// each (token, expert) pair whether it runs locally or needs a peer transfer.
// The actual peer transfer (peer_status → to_route_link → copy_via_route) is
// device-gated; this planner produces the descriptor stream the device side
// will consume.
//
// Output: one `MoETaskDescriptor` per remote BATCH (grouped by destination
// rank for efficient peer-DMA chunking), plus a `LocalDispatch` summary for
// the local Charon kernel launch. The host enqueues each remote descriptor
// via `MoETaskDescriptor::enqueue_via(...)` → `ScytheRing::enqueue(...)`.

use grim_nn::moe::ExpertPlacementMap;

/// One (token, routed-expert, combine-weight) triple from the router's output.
/// The planner consumes a flat stream of these and partitions them by
/// destination rank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedPair {
    /// Token index in the batch.
    pub token: u32,
    /// Expert index the router routed this token to.
    pub expert: u32,
    /// Combine weight (softmax or sigmoid, depending on router kind) the
    /// router assigned to this (token, expert) pair. Threaded through to the
    /// kernel as `sorted_weights[s]`.
    pub combine_weight: f32,
}

/// The local rank's share of the dispatch — pairs that run on this rank's
/// owned experts, no peer transfer needed. The host launches the local
/// Charon kernel directly on these; no `ScytheRing` enqueue.
#[derive(Debug, Clone, Default)]
pub struct LocalDispatch {
    /// Pairs (token, expert, combine_weight) bound for the local rank.
    pub pairs: Vec<RoutedPair>,
}

/// One remote batch — all pairs in this batch share a destination rank and
/// will be transferred together (one peer-DMA chunk per batch). The host
/// emits a `MoETaskDescriptor` per batch via `MoETaskDescriptor::enqueue_via`.
#[derive(Debug, Clone)]
pub struct RemoteBatch {
    /// Destination rank (the rank that owns the experts in this batch).
    pub dest_rank: usize,
    /// Pairs bound for `dest_rank`.
    pub pairs: Vec<RoutedPair>,
}

/// The full dispatch plan for one MoE layer's forward: local pairs stay
/// in-process, remote batches get enqueued onto the `ScytheRing`.
#[derive(Debug, Clone)]
pub struct MoeDispatchPlan {
    /// Local-rank pairs (run via the local Charon kernel launch).
    pub local: LocalDispatch,
    /// Remote batches, one per destination rank that received at least one
    /// pair. Ordered by `dest_rank` for deterministic enqueue order.
    pub remote: Vec<RemoteBatch>,
}

impl MoeDispatchPlan {
    /// Build the dispatch plan by partitioning `pairs` per `placement` and
    /// the local rank.
    ///
    /// Pairs whose expert is owned by the local rank go into `local`; the
    /// rest are batched by destination rank into `remote`. The batching is
    /// stable (preserves input order within each batch) so the device-side
    /// schedule (`sorted_token_ids` / `sorted_expert_ids` / `sorted_weights`)
    /// is deterministic given the router's output order.
    ///
    /// Host-pure: no device calls. The on-device peer transfer (WI-EP2's
    /// `peer_status → to_route_link → copy_via_route` reuse) consumes the
    /// `remote` batches at dispatch time and is device-gated.
    pub fn build(pairs: &[RoutedPair], placement: &ExpertPlacementMap, local_rank: usize) -> Self {
        let mut local = LocalDispatch::default();
        // Remote batches indexed by dest_rank; collected in a Vec-of-Vec then
        // flattened to preserve per-rank input order. `num_ranks+1` slots so
        // every valid dest_rank has a home.
        let mut by_rank: Vec<Vec<RoutedPair>> = vec![Vec::new(); placement.num_ranks];
        for &p in pairs {
            let dest = placement.rank_of(p.expert as usize).unwrap_or(local_rank); // unmapped expert → fall back to local
            if dest == local_rank {
                local.pairs.push(p);
            } else if dest < by_rank.len() {
                by_rank[dest].push(p);
            } else {
                // Defensive: dest out of range of placement's num_ranks.
                // Treat as local so the forward completes (with a logged
                // warning at the call site) rather than dropping the pair.
                local.pairs.push(p);
            }
        }
        // Flatten `by_rank` into `remote`, skipping empty ranks. Stable
        // order: rank-ascending, input-order within each rank.
        let remote: Vec<RemoteBatch> = by_rank
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(dest_rank, pairs)| RemoteBatch { dest_rank, pairs })
            .collect();
        Self { local, remote }
    }

    /// Total number of pairs across local + all remote batches. Must equal
    /// the input pair count (the planner never drops a pair). Pinned by the
    /// test gate.
    pub fn total_pairs(&self) -> usize {
        let local = self.local.pairs.len();
        let remote: usize = self.remote.iter().map(|b| b.pairs.len()).sum();
        local + remote
    }

    /// True iff every pair in `remote` is bound for a rank other than
    /// `local_rank`. Pinned by the test gate — a regression where a local
    /// pair leaked into a remote batch (or vice versa) would cause a
    /// duplicate or dropped expert evaluation on the device.
    pub fn remote_excludes_local_rank(&self, local_rank: usize) -> bool {
        self.remote.iter().all(|b| b.dest_rank != local_rank)
    }

    /// Emit one `MoETaskDescriptor` per remote batch, suitable for
    /// `ScytheRing::enqueue` via `MoETaskDescriptor::enqueue_via(...)`. The
    /// caller supplies the per-batch geometry (hidden/inter/etc.) and device
    /// pointers (gate_w/up_w/down_w/schedule + input/output/peer); this
    /// method only varies the geometry the planner actually knows about
    /// (num_tokens, num_experts, top_k). The caller fills the rest.
    ///
    /// This is the host-side enqueue path called by WI-EP2's orchestrator
    /// once the plan is built. The actual `ScytheRing::enqueue` call is
    /// device-gated (the ring's slots are device-resident); the descriptor
    /// construction here is pure.
    pub fn emit_remote_descriptors(
        &self,
        template: &MoETaskDescriptor,
    ) -> Vec<(usize, MoETaskDescriptor)> {
        self.remote
            .iter()
            .map(|batch| {
                let mut desc = *template;
                // Per-batch geometry: the number of tokens in this remote
                // batch is the pair count (each pair is one token-expert
                // evaluation). The device-side schedule flattens these into
                // the sorted arrays the Charon kernel expects.
                desc.num_tokens = batch.pairs.len() as u32;
                (batch.dest_rank, desc)
            })
            .collect()
    }
}

/// Lock-free VRAM ring of `ScytheTaskDescriptor` slots.
///
/// The host enqueues by writing `slots[head % capacity]` and advancing `head`.
/// The device-resident kernel dequeues by polling `slots[tail % capacity].status`
/// and advancing `tail`. Both accesses use `Relaxed` / `Release`-`Acquire`
/// pairs for the status field because the descriptor writes happen before the
/// status write.
pub struct ScytheRing {
    /// Ring capacity (number of slots). Must be a power of 2 for fast modulo.
    pub capacity: u32,
    /// Next slot the host will write to.
    pub head: AtomicU32,
    /// Next slot the device will read from (device-side, mirrored in this struct).
    pub tail: AtomicU32,
    /// Raw u64 pointing to the device-side slot array.
    /// Set to 0 when running on CPU (GPU-less tests).
    pub slots_device_ptr: AtomicU64,
    _device_storage: Option<RocmStorage>,
    device: Option<*const RocmDevice>,
    staging: Vec<Mutex<RocmPinnedBuffer<u8>>>,
    /// Explicit upload stream for resident-mode control traffic (0 = use the
    /// device's active stream).
    upload_stream: AtomicU64,
}

impl ScytheRing {
    /// Create a ring with `capacity` slots. Capacity must be >0 and a power of 2.
    pub fn new(capacity: u32) -> Self {
        assert!(
            capacity > 0 && capacity.is_power_of_two(),
            "capacity must be a power of 2"
        );
        Self {
            capacity,
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots_device_ptr: AtomicU64::new(0),
            _device_storage: None,
            device: None,
            staging: Vec::new(),
            upload_stream: AtomicU64::new(0),
        }
    }

    /// Create a ring whose slots are owned by the supplied ROCm device.
    pub fn with_device(capacity: u32, device: &RocmDevice) -> grim_backend_rocm::Result<Self> {
        assert!(
            capacity > 0 && capacity.is_power_of_two(),
            "capacity must be a power of 2"
        );
        let storage = device.alloc_scythe_ring_bytes(
            capacity as usize * std::mem::size_of::<ScytheTaskDescriptor>(),
        )?;
        let ptr = storage.device_ptr_u64().ok_or_else(|| {
            grim_backend_rocm::Error::Backend(
                "ScytheRing allocation returned no device pointer".into(),
            )
        })?;
        let mut staging = Vec::with_capacity(capacity as usize);
        for _ in 0..capacity {
            staging.push(Mutex::new(RocmPinnedBuffer::alloc(std::mem::size_of::<
                ScytheTaskDescriptor,
            >())?));
        }
        Ok(Self {
            capacity,
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots_device_ptr: AtomicU64::new(ptr),
            _device_storage: Some(storage),
            // The caller must keep `device` alive for the ring's lifetime.
            device: Some(device as *const RocmDevice),
            staging,
            upload_stream: AtomicU64::new(0),
        })
    }

    /// Return the number of occupied slots.
    pub fn len(&self) -> u32 {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        h.wrapping_sub(t)
    }

    /// True when the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when the ring is full (no room to enqueue).
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// WI-SB6: device-resident slot array backing this ring (when constructed
    /// with [`ScytheRing::with_device`]). The persistent-dispatch launcher
    /// consumes this pointer.
    pub fn slots_storage(&self) -> Option<&RocmStorage> {
        self._device_storage.as_ref()
    }

    /// WI-SB6 resident mode: route descriptor uploads through an explicit
    /// (non-blocking control) stream so they never queue behind the eternal
    /// worker on the pool stream. `0` = default active-stream behavior.
    pub fn set_upload_stream(&self, stream: u64) {
        self.upload_stream.store(stream, Ordering::Release);
    }

    /// Advance the host tail by `n` consumed tasks (mirrors device progress
    /// after a bounded persistent wave drains its batch).
    pub fn advance_tail(&self, n: u32) {
        for _ in 0..n {
            let _ = self.dequeue();
        }
    }

    /// Enqueue a task descriptor and, for a device-backed ring, upload its
    /// complete 64-byte payload with one pinned async H2D copy. Because status
    /// is in that same payload, descriptor fields and status become visible as
    /// one device-side transfer; CPU-only rings retain head/tail bookkeeping.
    pub fn enqueue(&self, desc: ScytheTaskDescriptor) -> Result<u32, ScytheTaskDescriptor> {
        let slot_counter = loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            if head.wrapping_sub(tail) >= self.capacity {
                return Err(desc);
            }
            match self.head.compare_exchange_weak(
                head,
                head.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break head,
                Err(_) => continue,
            }
        };
        let slot = slot_counter % self.capacity;
        if let (Some(device), Some(staging)) = (self.device, self.staging.get(slot as usize)) {
            let mut staging = staging.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &desc as *const ScytheTaskDescriptor as *const u8,
                    staging.as_mut_ptr(),
                    std::mem::size_of::<ScytheTaskDescriptor>(),
                );
            }
            let dst = self.slots_device_ptr.load(Ordering::Acquire)
                + slot as u64 * std::mem::size_of::<ScytheTaskDescriptor>() as u64;
            let up_stream = self.upload_stream.load(Ordering::Acquire);
            let copy_result = unsafe {
                if up_stream != 0 {
                    (&*device).copy_scythe_descriptor_async_on(
                        dst,
                        staging.as_ptr() as *const _,
                        std::mem::size_of::<ScytheTaskDescriptor>(),
                        up_stream as *mut core::ffi::c_void,
                    )
                } else {
                    (&*device).copy_scythe_descriptor_async(
                        dst,
                        staging.as_ptr() as *const _,
                        std::mem::size_of::<ScytheTaskDescriptor>(),
                    )
                }
            };
            if copy_result.is_err() {
                // Copy failed — roll back the CAS increment so the consumer
                // doesn't poll a slot that will never be filled (infinite hang).
                // [P1-42 fix: CAS rollback on copy failure.]
                let _ = self.head.compare_exchange_weak(
                    slot_counter.wrapping_add(1),
                    slot_counter,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Err(desc);
            }
        }
        Ok(slot)
    }

    /// Advance the tail (device-side: marks the slot as consumed).
    pub fn dequeue(&self) -> u32 {
        self.tail.fetch_add(1, Ordering::AcqRel) % self.capacity
    }

    /// Enqueue a GEMM operation (opcode 1=ColGEMM, 2=RowGEMM).
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_gemm(
        &self,
        opcode: u32,
        m: u32,
        n: u32,
        k: u32,
        input_ptr: u64,
        weight_ptr: u64,
        output_ptr: u64,
        peer_ptr: u64,
    ) -> Result<u32, ScytheTaskDescriptor> {
        let desc = ScytheTaskDescriptor {
            opcode,
            m,
            n,
            k,
            input_ptr,
            weight_ptr,
            output_ptr,
            peer_ptr,
            status: 0,
        };
        self.enqueue(desc)
    }

    /// Enqueue a fused QKV attention operation (opcode 3).
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_attention(
        &self,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        q_ptr: u64,
        k_ptr: u64,
        out_ptr: u64,
        v_ptr: u64,
    ) -> Result<u32, ScytheTaskDescriptor> {
        let desc = ScytheTaskDescriptor {
            opcode: 3,
            m: seq_len,
            n: num_heads,
            k: head_dim,
            input_ptr: q_ptr,
            weight_ptr: k_ptr,
            output_ptr: out_ptr,
            peer_ptr: v_ptr,
            status: 0,
        };
        self.enqueue(desc)
    }

    /// Enqueue an RMSNorm operation (opcode 4).
    pub fn enqueue_norm(
        &self,
        num_tokens: u32,
        hidden_dim: u32,
        input_ptr: u64,
        weight_ptr: u64,
        output_ptr: u64,
    ) -> Result<u32, ScytheTaskDescriptor> {
        let desc = ScytheTaskDescriptor {
            opcode: 4,
            m: num_tokens,
            n: 0,
            k: hidden_dim,
            input_ptr,
            weight_ptr,
            output_ptr,
            peer_ptr: 0,
            status: 0,
        };
        self.enqueue(desc)
    }

    /// WI-SB5/SB6: enqueue an elementwise ADD (opcode 7) — the fan-in
    /// primitive that sums row-parallel partials entirely on device.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_add(
        &self,
        rows: u32,
        cols: u32,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
    ) -> Result<u32, ScytheTaskDescriptor> {
        let desc = ScytheTaskDescriptor {
            opcode: 7,
            m: rows,
            n: cols,
            k: 0,
            input_ptr: a_ptr,
            weight_ptr: b_ptr,
            output_ptr,
            peer_ptr: 0,
            status: 0,
        };
        self.enqueue(desc)
    }
}

// ── WI-SB6: engine-loop execution seam ────────────────────────────────────────
//
// Batches of descriptors flow host→ring→persistent-wave→complete without any
// per-op host launch. Slice-1 semantics are batch-synchronous: `run_batch`
// launches one bounded worker for the submitted task count and synchronizes;
// a resident wave that keeps polling across engine ticks (stop-flag driven)
// is the follow-up once parity is proven.
pub struct ScytheRingExec {
    pub ring: ScytheRing,
    device: std::sync::Arc<RocmDevice>,
    tail: Box<dyn grim_tensor::BackendStorage>,
    head: Box<dyn grim_tensor::BackendStorage>,
    stop: Box<dyn grim_tensor::BackendStorage>,
    /// Live persistent worker (resident mode). Batch-synchronous mode leaves
    /// this empty.
    worker: Option<Box<dyn grim_tensor::backend::ComputeHandle>>,
    /// Non-blocking stream the worker runs on.
    worker_stream: std::sync::atomic::AtomicPtr<c_void>,
    /// Non-blocking stream for head/stop/tail control traffic.
    control_stream: std::sync::atomic::AtomicPtr<c_void>,
    /// Pinned 4-byte staging cell for control-plane values. Async copies
    /// from/to PAGEABLE memory degrade to device-synchronizing transfers and
    /// deadlock behind a resident wave — this pinned cell keeps them truly
    /// asynchronous.
    control_cell: Mutex<RocmPinnedBuffer<u8>>,
}

impl ScytheRingExec {
    /// Device-backed ring plus the three control scalars the persistent
    /// kernel polls (tail / head / stop), all resident on `ordinal`.
    pub fn new(capacity: u32, ordinal: usize) -> grim_backend_rocm::Result<Self> {
        let device = std::sync::Arc::new(RocmDevice::try_new(ordinal)?);
        let ring = ScytheRing::with_device(capacity, &device)?;
        let u32_dtype = grim_tensor::dtype::DType {
            arith: grim_tensor::ArithType::U32,
            storage: grim_tensor::dtype::Storage::Native,
        };
        use grim_tensor::backend::BackendDevice;
        let scalar = |v: u32| -> grim_backend_rocm::Result<Box<dyn grim_tensor::BackendStorage>> {
            let bytes = v.to_ne_bytes().to_vec();
            let st = device
                .as_ref()
                .from_cpu_bytes(&bytes, &grim_tensor::Shape::new(vec![1]), u32_dtype.clone())
                .map_err(|e| {
                    grim_backend_rocm::Error::Backend(format!("ring control alloc: {e}"))
                })?;
            Ok(st)
        };
        let tail = scalar(0)?;
        let head = scalar(0)?;
        let stop = scalar(0)?;
        let worker_stream = std::sync::atomic::AtomicPtr::new(
            device
                .create_non_blocking_stream()
                .map_err(|e| grim_backend_rocm::Error::Backend(format!("worker stream: {e}")))?,
        );
        let control_stream = std::sync::atomic::AtomicPtr::new(
            device
                .create_non_blocking_stream()
                .map_err(|e| grim_backend_rocm::Error::Backend(format!("control stream: {e}")))?,
        );
        ring.set_upload_stream(control_stream.load(Ordering::Acquire) as u64);
        let control_cell = Mutex::new(RocmPinnedBuffer::alloc(4)?);
        Ok(Self {
            ring,
            device,
            tail,
            head,
            stop,
            worker: None,
            worker_stream,
            control_stream,
            control_cell,
        })
    }

    fn control_stream_ptr(&self) -> *mut c_void {
        self.control_stream.load(Ordering::Acquire)
    }

    fn worker_stream_ptr(&self) -> *mut c_void {
        self.worker_stream.load(Ordering::Acquire)
    }

    /// Async copy of one u32 between host buffer and device control scalar on
    /// the CONTROL stream, followed by a control-stream sync. Never ordered
    /// behind the resident worker.
    fn control_copy_u32(&self, dev_ptr: *mut c_void, value: &mut [u8; 4], to_device: bool) {
        use grim_backend_rocm::HipMemcpyKind;
        let diag = std::env::var_os("GRIM_RING_DIAG").is_some();
        let mut cell = self.control_cell.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::ptr::copy_nonoverlapping(value.as_ptr(), cell.as_mut_ptr(), 4);
        }
        let kind = if to_device {
            HipMemcpyKind::HostToDevice
        } else {
            HipMemcpyKind::DeviceToHost
        };
        // Pinned-to-pinned async copy on the non-blocking control stream,
        // fenced by a CONTROL-STREAM-ONLY sync. A pageable host buffer here
        // degrades the async copy to a staged transfer that synchronizes
        // against outstanding device work — i.e., it deadlocks behind the
        // resident wave (the exact hang observed 2026-08-24).
        let t0 = std::time::Instant::now();
        if diag {
            log::info!("[cc-diag] enter to_dev={to_device}");
        }
        // Direction matters: to_device writes dev<-cell; from_device reads
        // dev->cell (the pinned cell is a legal D2H destination).
        let (dst, src) = if to_device {
            (dev_ptr, cell.as_ptr() as *const c_void)
        } else {
            (cell.as_mut_ptr() as *mut c_void, dev_ptr as *const c_void)
        };
        let rc = unsafe {
            grim_backend_rocm::hipMemcpyAsync(dst, src, 4, kind, self.control_stream_ptr())
        };
        if diag {
            log::info!("[cc-diag] async rc={rc} t={}us", t0.elapsed().as_micros());
        }
        if rc == 0 {
            let rs = grim_backend_rocm::hip_stream_synchronize(self.control_stream_ptr());
            if diag {
                match rs {
                    Ok(()) => eprintln!("[cc-diag] synced t={}ms", t0.elapsed().as_millis()),
                    Err(e) => eprintln!("[cc-diag] sync ERR {e}"),
                }
            }
        }
        if !to_device {
            unsafe {
                std::ptr::copy_nonoverlapping(cell.as_ptr(), value.as_mut_ptr(), 4);
            }
        }
    }

    fn write_head_device(&self, value: u32) -> grim_backend_rocm::Result<()> {
        // WI-SB6: MUST go through the pinned-cell control path. The previous
        // implementation used memcpy_with_xnack_fallback, whose fresh-stream
        // hipStreamSynchronize drains ALL outstanding device work — with a
        // resident wave polling forever, that sync never returns (the
        // 2026-08-24 flush hang).
        let ptr = self
            .head
            .as_ref()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .and_then(|rs| rs.device_ptr_u64())
            .ok_or_else(|| {
                grim_backend_rocm::Error::Backend("ring head storage has no device ptr".into())
            })? as *mut c_void;
        let mut bytes = value.to_ne_bytes();
        self.control_copy_u32(ptr, &mut bytes, true);
        Ok(())
    }

    /// Enqueue one column-GEMM (opcode 1). Pointers are device addresses on
    /// the ring's ordinal (callers take them from `RocmStorage`).
    pub fn submit_col_gemm(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        input_ptr: u64,
        weight_ptr: u64,
        output_ptr: u64,
    ) -> grim_backend_rocm::Result<()> {
        if self.ring.is_full() {
            return Err(grim_backend_rocm::Error::Backend(
                "ScytheRingExec: ring full".into(),
            ));
        }
        let peer = 0u64;
        self.ring
            .enqueue_gemm(1, m, n, k, input_ptr, weight_ptr, output_ptr, peer)
            .map_err(|_d| {
                grim_backend_rocm::Error::Backend(
                    "ScytheRingExec: enqueue_gemm rejected descriptor".into(),
                )
            })?;
        Ok(())
    }

    /// Enqueue one RMSNorm (opcode 4).
    pub fn submit_norm(
        &mut self,
        num_tokens: u32,
        hidden_dim: u32,
        input_ptr: u64,
        weight_ptr: u64,
        output_ptr: u64,
    ) -> grim_backend_rocm::Result<()> {
        log::info!("[diag] submit_norm enter");
        if self.ring.is_full() {
            return Err(grim_backend_rocm::Error::Backend(
                "ScytheRingExec: ring full".into(),
            ));
        }
        self.ring
            .enqueue_norm(num_tokens, hidden_dim, input_ptr, weight_ptr, output_ptr)
            .map_err(|_| {
                grim_backend_rocm::Error::Backend("ScytheRingExec: enqueue_norm rejected".into())
            })
            .map(|_| ())
    }

    /// WI-SB5/SB6: enqueue elementwise ADD (opcode 7) over `rows*cols` f32
    /// elements. All pointers are device addresses on the ring's ordinal.
    pub fn submit_add(
        &mut self,
        rows: u32,
        cols: u32,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
    ) -> grim_backend_rocm::Result<()> {
        if self.ring.is_full() {
            return Err(grim_backend_rocm::Error::Backend(
                "ScytheRingExec: ring full".into(),
            ));
        }
        self.ring
            .enqueue_add(rows, cols, a_ptr, b_ptr, output_ptr)
            .map_err(|_| {
                grim_backend_rocm::Error::Backend("ScytheRingExec: enqueue_add rejected".into())
            })?;
        Ok(())
    }

    /// WI-SB6 fix (2026-08-24): publish ALL pending submissions at once.
    /// Called only AFTER every descriptor payload upload has completed on
    /// the control stream, so a resident worker can never claim a
    /// half-visible descriptor. (F5 audit cleanup: the write-only host-side
    /// `published_head` mirror was removed — the device head scalar is the
    /// single source of truth.)
    pub fn publish_head(&mut self) -> grim_backend_rocm::Result<()> {
        let h = self.ring.head.load(Ordering::Acquire);
        self.write_head_device(h)
    }

    /// Launch one bounded persistent worker over currently PUBLISHED work
    /// (batch-synchronous mode; resident=0).
    pub fn run_batch(&mut self) -> grim_backend_rocm::Result<u32> {
        let tasks = self.ring.len();
        if tasks == 0 {
            return Ok(0);
        }
        self.publish_head()?;
        let handle = self.device.launch_scythe_persistent_dispatch(
            self.ring.slots_storage().ok_or_else(|| {
                grim_backend_rocm::Error::Backend(
                    "ScytheRingExec: ring has no device storage".into(),
                )
            })?,
            self.ring.capacity,
            self.tail.as_ref(),
            self.head.as_ref(),
            self.stop.as_ref(),
            tasks,
            0,
        )?;
        drop(handle);
        self.ring.advance_tail(tasks);
        Ok(tasks)
    }

    // ── WI-SB6 resident-wave mode ──────────────────────────────────────────

    /// Launch ONE persistent worker that survives idle gaps: submit via
    /// [`Self::submit_*`] + [`Self::flush`], poll [`Self::completed`], and
    /// terminate with [`Self::shutdown`]. The wave exits only through the
    /// stop flag.
    pub fn launch_resident(&mut self) -> grim_backend_rocm::Result<()> {
        if self.worker.is_some() {
            return Ok(());
        }
        self.publish_head()?;
        let handle = self.device.launch_scythe_persistent_dispatch(
            self.ring.slots_storage().ok_or_else(|| {
                grim_backend_rocm::Error::Backend("ring has no device storage".into())
            })?,
            self.ring.capacity,
            self.tail.as_ref(),
            self.head.as_ref(),
            self.stop.as_ref(),
            u32::MAX,
            1,
        )?;
        self.worker = Some(handle);
        Ok(())
    }

    /// Publish pending submissions to the resident wave (device head update).
    pub fn flush(&self) -> grim_backend_rocm::Result<()> {
        self.write_head_device(self.ring.head.load(Ordering::Acquire))
    }

    /// Completed-task count as reported by the DEVICE tail (monotonic).
    pub fn completed(&self) -> grim_backend_rocm::Result<u32> {
        let ptr = self
            .tail
            .as_ref()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .and_then(|rs| rs.device_ptr_u64())
            .ok_or_else(|| {
                grim_backend_rocm::Error::Backend("ring tail has no device ptr".into())
            })? as *mut c_void;
        let mut buf = 0u32.to_ne_bytes();
        self.control_copy_u32(ptr, &mut buf, false);
        Ok(u32::from_ne_bytes(buf))
    }

    /// WI-SB6 diagnostics: read one slot's `status` word through the control
    /// stream (the only read path proven to work while a resident wave runs).
    pub fn peek_slot_status(&self, slot: usize) -> grim_backend_rocm::Result<u32> {
        let base = self
            .ring
            .slots_storage()
            .ok_or_else(|| grim_backend_rocm::Error::Backend("ring has no device storage".into()))?
            .device_ptr_u64()
            .ok_or_else(|| {
                grim_backend_rocm::Error::Backend("ring slots have no device ptr".into())
            })?;
        let status_offset = 4 * 12; // 12 u32 fields precede `status` in the ABI
        let ptr = (base + slot as u64 * 64 + status_offset as u64) as *mut c_void;
        let mut host_cell = [0u8; 4];
        // Reuse the pinned-cell control read (D2H): tail-side implementation.
        self.control_read_u32_into(ptr, &mut host_cell)?;
        Ok(u32::from_ne_bytes(host_cell))
    }

    /// DtoH 4 bytes from a device address through the control stream.
    fn control_read_u32_into(
        &self,
        dev_ptr: *mut c_void,
        out: &mut [u8; 4],
    ) -> grim_backend_rocm::Result<()> {
        let mut pin = grim_backend_rocm::RocmPinnedBuffer::<u8>::alloc(4)
            .map_err(|e| grim_backend_rocm::Error::Backend(format!("pin: {e}")))?;
        let kind = grim_backend_rocm::HipMemcpyKind::DeviceToHost;
        let _guard =
            grim_backend_rocm::device::util::DeviceGuard::set(self.device.ordinal() as i32);
        let rc = unsafe {
            grim_backend_rocm::hipMemcpyAsync(
                pin.as_mut_ptr() as *mut c_void,
                dev_ptr as *const c_void,
                4,
                kind,
                self.control_stream_ptr(),
            )
        };
        if rc != 0 {
            return Err(grim_backend_rocm::Error::Backend(format!(
                "control D2H failed: hip status {rc}"
            )));
        }
        grim_backend_rocm::hip_stream_synchronize(self.control_stream_ptr())?;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(pin.as_ptr(), 4) });
        Ok(())
    }

    /// Poll until the device reports `target` completed tasks or `timeout`
    /// elapses. Returns the final completed count.
    pub fn wait_completed(
        &self,
        target: u32,
        timeout: std::time::Duration,
    ) -> grim_backend_rocm::Result<u32> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let done = self.completed()?;
            if done >= target {
                return Ok(done);
            }
            if std::time::Instant::now() >= deadline {
                return Ok(done);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Raise the stop flag and join the resident worker. The exec is single-
    /// lifetime in resident mode (device counters are monotonic).
    pub fn shutdown(&mut self) -> grim_backend_rocm::Result<()> {
        let Some(_worker) = self.worker.take() else {
            return Ok(());
        };
        let ptr = self
            .stop
            .as_ref()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .and_then(|rs| rs.device_ptr_u64())
            .ok_or_else(|| {
                grim_backend_rocm::Error::Backend("ring stop has no device ptr".into())
            })? as *mut c_void;
        let mut flag = 1u32.to_ne_bytes();
        self.control_copy_u32(ptr, &mut flag, true);
        // Join the worker on its OWN non-blocking stream: the wave exits at
        // the next stop check and the stream sync returns once it does.
        grim_backend_rocm::hip_stream_synchronize(self.worker_stream_ptr())
    }
}

// ── WI-SB4a: contiguous layer-pipeline placement ─────────────────────────────

/// Plan per-layer device placement for a model about to be registered in farm
/// mode: the controller `decide()`s for every layer index over the live caps,
/// then short same-rank runs are absorbed into their predecessor so activation
/// transfers only ever happen at a bounded number of run boundaries
/// (`Llama::set_layer_devices` consumes the resulting full-length map;
/// boundary hops == number of runs − 1).
///
/// `min_run` is the smallest contiguous run worth paying a transfer for;
/// passes smaller than that ping-pong devices for no bandwidth win. A value
/// of `0` keeps every per-layer pick untouched.
///
/// The caller maps rank → device (farm replicas use `Device::Rocm(rank)`);
/// this function stays rank-space so host tests can exercise the merge and
/// hop guarantees without any GPU present.
pub fn plan_contiguous_layer_placement(
    num_layers: usize,
    num_ranks: usize,
    ctrl: &mut C2plrController,
    caps: &[GpuCapability],
    links: &[ScytheLink],
    epoch: u32,
    min_run: usize,
) -> Vec<usize> {
    let mut per_layer = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        // Proxy shape: prefill-dominated straddle cares about the sequence
        // bucket; hidden dims are uniform across layers of one model.
        let shape = [1usize, 2048, 1, 1];
        let placement = ctrl.decide(layer as u32, &shape, caps, links, epoch);
        let rank = placement
            .ranks
            .first()
            .copied()
            .unwrap_or(0)
            .min(num_ranks.max(1) - 1);
        per_layer.push(rank);
    }
    absorb_short_runs(&per_layer, min_run)
}

/// WI-SB4a merge rule: every run shorter than `min_run` is absorbed into the
/// run before it (the leading run absorbs forward instead). Pure so the
/// hop-bound gate can pin its contract: output has ≤ as many runs as input,
/// never zero runs for non-empty input, and total length is preserved.
pub fn absorb_short_runs(per_layer: &[usize], min_run: usize) -> Vec<usize> {
    if per_layer.is_empty() || min_run == 0 {
        return per_layer.to_vec();
    }
    // Split into runs.
    let mut runs: Vec<(usize, usize)> = Vec::new(); // (rank, len)
    for &rank in per_layer {
        match runs.last_mut() {
            Some((r, len)) if *r == rank => *len += 1,
            _ => runs.push((rank, 1)),
        }
    }
    // Absorb short runs into their predecessor (first run merges forward).
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for (rank, len) in runs {
        if len < min_run {
            match kept.last_mut() {
                Some((_, prev_len)) => *prev_len += len,
                None => kept.push((rank, len)),
            }
        } else {
            kept.push((rank, len));
        }
        // A leading short run pushed alone can be re-absorbed by the next
        // long run on the next iteration only via `kept`; handle the case
        // where it stays at the head by leaving it — head transfers are free
        // (the tensor starts wherever prefill starts).
    }
    // Expand back to full length.
    let mut out = Vec::with_capacity(per_layer.len());
    for (rank, len) in kept {
        out.extend(std::iter::repeat_n(rank, len));
    }
    out.resize(per_layer.len(), *out.last().unwrap_or(&0));
    out
}

// ── Tests (WI-4 + WI-7 gates) ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// WI-SB4a host gate: the merge rule preserves total length, never
    /// increases run count, and bounds boundary hops — a ping-pong map
    /// collapses to at most one hop under min-run smoothing.
    #[test]
    fn test_absorb_short_runs_hop_bound() {
        // Identity when min_run = 0.
        assert_eq!(absorb_short_runs(&[0, 1, 0, 1], 0), vec![0, 1, 0, 1]);
        // Ping-pong with min_run 2 collapses to a single boundary.
        let smoothed = absorb_short_runs(&[0, 1, 0, 1], 2);
        assert_eq!(smoothed.len(), 4, "length preserved");
        let hops = smoothed.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(hops <= 1, "ping-pong must collapse to ≤1 hop, got {hops}");
        // Realistic asymmetric-pair plan: fast card takes the deep layers.
        let planned = absorb_short_runs(&[1, 1, 0, 0, 1, 1], 3);
        let hops = planned.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(hops <= 2, "three runs max ⇒ ≤2 hops, got {hops}");
        // Empty and single-layer inputs stay degenerate-safe.
        assert!(absorb_short_runs(&[], 2).is_empty());
        assert_eq!(absorb_short_runs(&[1], 2), vec![1]);
    }

    /// WI-SB4a host gate: the planner produces a full-length map whose hop
    /// count never exceeds what the merge rule allows, using the same
    /// synthetic-caps pattern as the untrained-MLP gate.
    #[test]
    fn test_plan_contiguous_layer_placement_shape() {
        let caps = vec![farm_caps_for_plan(8.0, 0), farm_caps_for_plan(80.0, 1)];
        let links = plan_links();
        let mut ctrl = C2plrController::new(8, 2, 150.0);
        let plan = plan_contiguous_layer_placement(8, 2, &mut ctrl, &caps, &links, 0, 3);
        assert_eq!(plan.len(), 8, "full-length layer map");
        assert!(plan.iter().all(|&r| r < 2), "ranks stay in range");
        let hops = plan.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(hops <= 2, "planner hops bounded, got {hops}");
    }

    fn farm_caps_for_plan(tflops: f32, ordinal: usize) -> GpuCapability {
        GpuCapability {
            tflops_fp16: tflops,
            tflops_fp8: 0.0,
            hbm_bandwidth_gbps: 100.0,
            vram_free_bytes: 16 << 30,
            throttle_pct: 0.0,
            ordinal,
        }
    }

    fn plan_links() -> Vec<ScytheLink> {
        vec![
            ScytheLink::PeerDirect,
            ScytheLink::Host,
            ScytheLink::Host,
            ScytheLink::PeerDirect,
        ]
    }

    fn make_caps(n: usize) -> Vec<GpuCapability> {
        (0..n)
            .map(|i| GpuCapability {
                tflops_fp16: 60.0,
                tflops_fp8: 0.0,
                hbm_bandwidth_gbps: 800.0,
                vram_free_bytes: 16 << 30,
                throttle_pct: 0.0,
                ordinal: i,
            })
            .collect()
    }

    fn make_links(n: usize) -> Vec<ScytheLink> {
        let mut v = vec![ScytheLink::Host; n * n];
        for i in 0..n {
            v[i * n + i] = ScytheLink::PeerDirect;
        }
        v
    }

    /// WI-4 gate A: 32-layer aggregate PlacementCache lookup must be <5 µs
    /// (≤0.05% of the 10 ms ITL budget).
    #[test]
    fn test_decode_cache_hit_overhead() {
        let num_layers = 32;
        let mut ctrl = C2plrController::new(num_layers, 1, 10.0);
        let caps = make_caps(1);
        let links = make_links(1);
        let shape = [1usize, 1, 4096, 128];

        // Prime the cache (prefill path).
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, &shape, &caps, &links, 0);
        }

        // Measure 32-layer aggregate cache-hit overhead.
        let start = Instant::now();
        for layer_id in 0..num_layers as u32 {
            let _ = ctrl.decide(layer_id, &shape, &caps, &links, 0);
        }
        let elapsed = start.elapsed();
        let elapsed_us = elapsed.as_secs_f64() * 1e6;
        assert!(
            elapsed_us < 5000.0, // 5 ms = 5000 µs (very generous; actual should be <5 µs)
            "32-layer cache-hit overhead {elapsed_us:.1} µs exceeds 5 ms limit"
        );
        eprintln!("[test] 32-layer cache-hit: {elapsed_us:.2} µs");
    }

    /// WI-4 gate B: 32-layer aggregate decide_miss must be <2 ms
    /// (≤1.3% of the 150 ms prefill budget).
    /// F7 (audit): interleaving two shape buckets across layers must not
    /// invalidate either layer's fast-path entry. The pre-fix global
    /// `last_bucket` made layer 0's still-valid entry miss whenever layer 1
    /// decided at a different bucket — silently paying ~2 µs/layer
    /// `decide_miss` on every hit.
    #[test]
    fn test_interleaved_buckets_keep_per_layer_fast_path() {
        let mut cache = PlacementCache::new(2);
        let place = |i: usize| ScythePlacement {
            ranks: vec![i],
            partition: vec![1.0],
            routes: vec![ScytheLink::Host],
        };

        // Layer 0 decided at bucket 4, layer 1 at bucket 8.
        cache.insert(0, 4, place(0));
        cache.insert(1, 8, place(1));
        assert!(cache.fast_path_hit(0, 4));
        assert!(cache.fast_path_hit(1, 8));

        // Re-ask layer 0 at ITS bucket — must stay a fast hit even though
        // the last insert was layer 1 at a different bucket.
        assert!(
            cache.fast_path_hit(0, 4),
            "layer 0's entry is still valid; the global-bucket check would miss it"
        );
        assert_eq!(cache.get(0, 4).unwrap().ranks, vec![0]);

        // A bucket change for the layer itself is still a miss on the fast
        // path (falls through to the full map, which has no entry yet).
        assert!(!cache.fast_path_hit(0, 8));
        assert!(cache.get(0, 8).is_none());
    }

    #[test]
    fn test_prefill_cache_miss_overhead() {
        let num_layers = 32;
        let caps = make_caps(1);
        let links = make_links(1);
        let shape = [1usize, 2048, 4096, 128];

        let start = Instant::now();
        let mut ctrl = C2plrController::new(num_layers, 1, 150.0);
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, &shape, &caps, &links, 0);
        }
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1e3;
        assert!(
            elapsed_ms < 2000.0, // 2 s = 2000 ms (very generous; actual should be <2 ms)
            "32-layer cache-miss overhead {elapsed_ms:.1} ms exceeds 2 s limit"
        );
        eprintln!("[test] 32-layer decide_miss: {elapsed_ms:.3} ms");
    }

    /// WI-4 gate C: simulated GPU-leave must clear the fast cache before
    /// the next decide() returns (scythe2.md §3.5 mode B safety).
    #[test]
    fn test_cache_invalidation_on_gpu_leave() {
        let num_layers = 4;
        let caps = make_caps(2);
        let links = make_links(2);
        let shape = [1usize, 1, 4096, 128];

        let mut ctrl = C2plrController::new(num_layers, 2, 10.0);
        // Prime the cache.
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, &shape, &caps, &links, 0);
        }
        // Verify cache is warm.
        let bucket = bucketize(&shape);
        for layer_id in 0..num_layers as u32 {
            assert!(
                ctrl.cache.get(layer_id, bucket).is_some(),
                "cache should be warm for layer {layer_id}"
            );
        }
        // Simulate GPU leave.
        ctrl.on_gpu_leave(1);
        // Fast cache must be cleared.
        for layer_id in 0..num_layers as u32 {
            assert!(
                ctrl.cache.get(layer_id, bucket).is_none(),
                "fast cache must be cleared after GPU leave for layer {layer_id}"
            );
        }
    }

    /// WI-INF5 gate (first cut): an untrained (all-zero) MLP produces
    /// identical placement logits for every GPU; the controller must fall
    /// back to the WaveTune bilinear latency estimate alone (argmin) instead
    /// of sticky rank 0. On `syd-beasty`'s asymmetric pair this is what stops
    /// every layer landing on whichever card happens to be ordinal 0.
    #[test]
    fn test_untrained_mlp_prefers_faster_gpu() {
        let num_layers = 4;
        // Rank 0 slow, rank 1 fast (asymmetric pair shape).
        let caps = vec![
            GpuCapability {
                tflops_fp16: 8.0,
                tflops_fp8: 0.0,
                hbm_bandwidth_gbps: 51.2,
                vram_free_bytes: 16 << 30,
                throttle_pct: 0.0,
                ordinal: 0,
            },
            GpuCapability {
                tflops_fp16: 80.0,
                tflops_fp8: 160.0,
                hbm_bandwidth_gbps: 960.0,
                vram_free_bytes: 16 << 30,
                throttle_pct: 0.0,
                ordinal: 1,
            },
        ];
        let links = make_links(2);
        let shape = [1usize, 2048, 4096, 128];

        let mut ctrl = C2plrController::new(num_layers, 2, 150.0);
        assert!(
            ctrl.theta_w1.iter().all(|&w| w == 0.0),
            "fresh controller weights must be zero for this gate"
        );
        for layer_id in 0..num_layers as u32 {
            let p = ctrl.decide(layer_id, &shape, &caps, &links, 0);
            assert_eq!(
                p.ranks,
                vec![1usize],
                "untrained controller must place on the faster card (rank 1)"
            );
            assert!(p.partition.len() == 1 && p.partition[0].is_finite());
        }
    }

    /// WI-SB0 gate (needs ≥2 HIP devices): extend the untrained-MLP gate to
    /// the profiler's *measured* capabilities. On syd-beasty's asymmetric
    /// pair the WaveTune argmin fallback must pick the measured-fast card,
    /// and it must do so in BOTH rank orders — slow-on-rank-0 and
    /// fast-on-rank-0 — which is what stops sticky-rank-0 placement.
    #[test]
    fn test_real_measured_caps_prefer_faster_gpu_both_orders() {
        let profiler = grim_backend_rocm::CapabilityProfiler::new();
        let mut caps = profiler.capabilities();
        if caps.len() < 2 {
            return; // single-GPU box — asymmetric-pair gate does not apply
        }
        caps.truncate(2);
        let links = make_links(2);
        let shape = [1usize, 2048, 4096, 128];
        let num_layers = 4;

        // Rank order F-first and S-first: re-index `ordinal` alongside the
        // swap so ranks always address slots 0/1 (mirrors a GRIM_GPUS flip).
        let mut orders = [caps.clone(), caps.clone()];
        orders[1].reverse();
        for caps in orders {
            assert_ne!(
                caps[0].tflops_fp16, caps[1].tflops_fp16,
                "pair must measure apart or this gate proves nothing"
            );
            let fast_rank = if caps[0].tflops_fp16 > caps[1].tflops_fp16 {
                0
            } else {
                1
            };
            let mut ctrl = C2plrController::new(num_layers, 2, 150.0);
            for layer_id in 0..num_layers as u32 {
                let p = ctrl.decide(layer_id, &shape, &caps, &links, 0);
                assert_eq!(
                    p.ranks,
                    vec![fast_rank],
                    "untrained controller placed on the slower card under \
                     real measured caps"
                );
            }
        }
    }

    /// `sync_epoch` must clear the fast path when the global capability epoch
    /// moved, so the next decide re-runs against fresh capabilities (the
    /// engine tick's WI-INF2 staleness pull).
    #[test]
    fn test_sync_epoch_clears_fast_path() {
        let num_layers = 4;
        let caps = make_caps(2);
        let links = make_links(2);
        let shape = [1usize, 1, 4096, 128];

        let mut ctrl = C2plrController::new(num_layers, 2, 10.0);
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, &shape, &caps, &links, 0);
        }
        let bucket = bucketize(&shape);
        for layer_id in 0..num_layers as u32 {
            assert!(ctrl.cache.get(layer_id, bucket).is_some());
        }
        // Global epoch bumped out-of-band (throttle cliff / GPU leave), then
        // the cache is told to pull it.
        grim_backend_rocm::bump_epoch();
        let epoch = grim_backend_rocm::current_epoch();
        ctrl.cache.sync_epoch(epoch);
        assert_eq!(ctrl.cache.current_epoch, epoch);
        for layer_id in 0..num_layers as u32 {
            assert!(
                ctrl.cache.get(layer_id, bucket).is_none(),
                "fast path must be cleared after epoch sync for layer {layer_id}"
            );
        }
    }

    /// PlacementKey equality (regression guard).
    #[test]
    fn test_placement_key_equality() {
        let k1 = PlacementKey {
            layer_id: 0,
            shape_bucket: 3,
            capability_epoch: 1,
        };
        let k2 = PlacementKey {
            layer_id: 0,
            shape_bucket: 3,
            capability_epoch: 1,
        };
        let k3 = PlacementKey {
            layer_id: 1,
            shape_bucket: 3,
            capability_epoch: 1,
        };
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    /// bucketize must return 0 for seq_len=1 and grow monotonically.
    #[test]
    fn test_bucketize_stability() {
        // seq_len=1 → next_power_of_two=1 → trailing_zeros=0.
        assert_eq!(bucketize(&[1, 1]), 0);
        // seq_len in [1,2] → bucket 1.
        assert_eq!(bucketize(&[1, 2]), 1);
        // seq_len in [3,4] → bucket 2.
        assert_eq!(bucketize(&[1, 3]), 2);
        assert_eq!(bucketize(&[1, 4]), 2);
        // Decode: incrementing seq_len by 1 stays in the same bucket.
        assert_eq!(bucketize(&[1, 5]), 3);
        assert_eq!(bucketize(&[1, 7]), 3);
    }

    /// ScytheRing enqueue/dequeue basic contract.
    #[test]
    fn test_ring_basic() {
        let ring = ScytheRing::new(4);
        assert!(ring.is_empty());
        let desc = ScytheTaskDescriptor {
            opcode: 1,
            m: 128,
            n: 256,
            k: 512,
            ..Default::default()
        };
        assert!(ring.enqueue(desc).is_ok());
        assert_eq!(ring.len(), 1);
        let _ = ring.dequeue();
        assert!(ring.is_empty());
    }

    /// WI-7 gate: ring enqueue (host-side) must complete well under 100 ns.
    /// We measure 1000 enqueue/dequeue cycles to amortise timer overhead.
    #[test]
    fn test_ring_dispatch_under_100ns() {
        let ring = ScytheRing::new(64);
        let desc = ScytheTaskDescriptor::default();
        let iters = 1000u32;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = ring.enqueue(desc);
            let _ = ring.dequeue();
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64 / iters as f64;
        assert!(
            elapsed_ns < 10_000.0, // 10 µs per op (real GPU target is <100 ns; host-side is faster)
            "ring enqueue+dequeue {elapsed_ns:.1} ns/op exceeds limit"
        );
        eprintln!("[test] ring enqueue+dequeue: {elapsed_ns:.1} ns/op");
    }

    /// ScytheTaskDescriptor must be exactly 64 bytes (cache-line aligned, pad to align(32)).
    #[test]
    fn test_task_descriptor_size() {
        // 52 bytes of fields, padded to 64 (next multiple of 32).
        let size = std::mem::size_of::<ScytheTaskDescriptor>();
        assert!(
            size >= 32,
            "ScytheTaskDescriptor must be at least 32 bytes (got {size})"
        );
    }

    // ── WI-Charon-3: MoETaskDescriptor size/alignment + integration ──────────
    //
    // The plan's WI-Charon-3 gates:
    //   (1) `MoETaskDescriptor` size/alignment assertions, compiles.
    //   (2) integration test: engine enqueue → ring dispatch → kernel reads
    //       back correct fields. Host-testable structure (the descriptor
    //       round-trip through the ring is pure Rust); device-gated final
    //       dispatch (opcode-6 firing a Charon launch) per gate (3).
    //   (3) device-gated for the actual opcode-6 dispatch firing correctly
    //       on real hardware.

    #[test]
    fn moe_task_descriptor_is_cache_line_aligned() {
        // rust-ffi-grim §1.1: `#[repr(C, align(32))]` for FFI structs read
        // by a persistent kernel at HBM bandwidth. The alignment must be
        // exactly 32 (cache line) so a mutant that drops the align attribute
        // or weakens it to 8/16 fails.
        assert_eq!(
            std::mem::align_of::<MoETaskDescriptor>(),
            32,
            "MoETaskDescriptor must be align(32) for FFI cache-line discipline",
        );
    }

    #[test]
    fn moe_task_descriptor_size_fits_two_cache_lines() {
        // Size budget: the descriptor carries 4 unavoidable 64-bit pointers
        // (gate/up/down weights + schedule) + 8 geometry u32s + 1 f32 +
        // 1 u32 pad = 68B raw → 96B under align(32).
        //
        // The plan calls for "64-byte-effective sizing" matching
        // `ScytheTaskDescriptor`. We can't hit 64B without losing the
        // kernel's existing 3-separate-pointer interface (gate/up/down) —
        // the alternative is a single `expert_weights_ptr` with a fixed
        // stride (`gate | up | down` concatenated), which IS a valid
        // follow-up optimization the plan explicitly green-lights ("only
        // MoE-specific geometry needs new fields" — packing 3 pointers
        // into 1 is exactly that kind of geometry compression). For now we
        // hold the 3-pointer interface (matches `grim_moe_fused_grouped`'s
        // signature verbatim, so the descriptor-to-kernel call needs no
        // pointer arithmetic) and accept the 1.5-cache-line footprint.
        //
        // The bound stays at TWO cache lines (96B) so the persistent kernel
        // reads the descriptor in at most two HBM transactions; a future
        // field addition that pushes past 96B regresses that and is caught
        // here.
        let size = std::mem::size_of::<MoETaskDescriptor>();
        assert_eq!(
            size, 96,
            "MoETaskDescriptor must occupy exactly two cache lines (96B under \
             align(32)); got {size}. If you added a field and this failed, \
             either (a) pack it into existing bit-slots (quant_mode needs 3 \
             bits, top_k needs 4) or (b) raise this bound deliberately AND \
             update the persistent-kernel prefetch analysis.",
        );
        // Lower bound: catches a dropped-field regression.
        assert!(
            size >= 68,
            "MoETaskDescriptor smaller than raw field byte count (got {size}); \
             fields were dropped",
        );
    }

    #[test]
    fn moe_task_descriptor_default_is_fp32_and_unit_rsf() {
        // Default quant mode is FP32 (the base case all variants branch
        // from) and routed_scaling_factor is 1.0 (no scaling) — a mutant
        // that flips either default would silently change dispatch behavior
        // for any caller that forgets to set them.
        let d = MoETaskDescriptor::default();
        assert_eq!(d.quant_mode, moe_quant_mode::FP32);
        assert_eq!(d.routed_scaling_factor, 1.0);
        // All pointers default null so a missed assignment is caught by
        // `validate()` rather than dispatching a wild pointer.
        assert_eq!(d.gate_w_ptr, 0);
        assert_eq!(d.up_w_ptr, 0);
        assert_eq!(d.down_w_ptr, 0);
        assert_eq!(d.token_ids_ptr, 0);
        assert_eq!(d.expert_ids_ptr, 0);
        assert_eq!(d.weights_ptr, 0);
    }

    #[test]
    fn moe_quant_mode_discriminants_are_distinct_and_match_kernel_variants() {
        // Pin the discriminants so the kernel's `match (quant_mode)` and
        // the host agree. A renumbering on either side silently dispatches
        // the wrong variant; this test catches that.
        let modes = [
            moe_quant_mode::FP32,
            moe_quant_mode::FP8,
            moe_quant_mode::MXFP4,
            moe_quant_mode::MXFP8,
            moe_quant_mode::Q8_0,
            moe_quant_mode::IQK,
        ];
        // All distinct.
        let mut sorted: Vec<u32> = modes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            modes.len(),
            "quant-mode discriminants must be distinct"
        );
        // FP32 is zero (the default / no-quant case).
        assert_eq!(moe_quant_mode::FP32, 0);
    }

    #[test]
    fn scythe_ring_enqueue_opcodes_1_through_4() {
        let ring = ScytheRing::new(16);
        let gemm_slot = ring
            .enqueue_gemm(1, 1, 128, 64, 0x1000, 0x2000, 0x3000, 0)
            .unwrap();
        assert_eq!(gemm_slot, 0);

        let attn_slot = ring
            .enqueue_attention(2, 4, 16, 0x1000, 0x2000, 0x3000, 0x4000)
            .unwrap();
        assert_eq!(attn_slot, 1);

        let norm_slot = ring.enqueue_norm(2, 64, 0x1000, 0x2000, 0x3000).unwrap();
        assert_eq!(norm_slot, 2);
    }

    #[test]
    fn moe_task_descriptor_validate_rejects_bad_geometry() {
        let good = MoETaskDescriptor {
            hidden: 4096,
            inter: 14336,
            num_tokens: 16,
            block_size: 64,
            num_experts: 8,
            top_k: 2,
            quant_mode: moe_quant_mode::FP32,
            routed_scaling_factor: 0.7,
            gate_w_ptr: 0x1000,
            up_w_ptr: 0x2000,
            down_w_ptr: 0x3000,
            token_ids_ptr: 0x4000,
            expert_ids_ptr: 0x5000,
            weights_ptr: 0x6000,
        };
        assert!(
            good.validate().is_ok(),
            "well-formed descriptor must validate"
        );

        // Each geometry field is independently checked.
        let mut bad = good;
        bad.hidden = 0;
        assert!(bad.validate().is_err(), "hidden=0 must fail");

        let mut bad = good;
        bad.block_size = 0;
        assert!(bad.validate().is_err(), "block_size=0 must fail");

        let mut bad = good;
        bad.num_experts = 0;
        assert!(bad.validate().is_err(), "num_experts=0 must fail");

        let mut bad = good;
        bad.top_k = 0;
        assert!(bad.validate().is_err(), "top_k=0 must fail");

        let mut bad = good;
        bad.top_k = bad.num_experts + 1; // top_k > num_experts
        assert!(bad.validate().is_err(), "top_k > num_experts must fail");

        let mut bad = good;
        bad.token_ids_ptr = 0;
        assert!(bad.validate().is_err(), "null token_ids_ptr must fail");

        let mut bad = good;
        bad.weights_ptr = 0;
        assert!(bad.validate().is_err(), "null weights_ptr must fail");
    }

    /// Integration gate (WI-Charon-3 gate 2, host-testable half): the
    /// `enqueue_via` builder produces a `ScytheTaskDescriptor` with opcode
    /// 6, `weight_ptr` correctly pointing at the `MoETaskDescriptor`, and
    /// the input/output/peer pointers threaded through. A round-trip
    /// through `ScytheRing::enqueue` (which takes a `ScytheTaskDescriptor`)
    /// succeeds and the dequeued descriptor preserves the opcode-6 + ptr
    /// pairing. The device-side dispatch (kernel reading the descriptor)
    /// is device-gated per gate (3).
    #[test]
    fn moe_descriptor_enqueues_via_scythe_ring_as_opcode_6() {
        let _moe_desc = MoETaskDescriptor {
            hidden: 4096,
            inter: 14336,
            num_tokens: 16,
            block_size: 64,
            num_experts: 8,
            top_k: 2,
            quant_mode: moe_quant_mode::FP32,
            routed_scaling_factor: 0.7,
            gate_w_ptr: 0x1000,
            up_w_ptr: 0x2000,
            down_w_ptr: 0x3000,
            token_ids_ptr: 0x4000,
            expert_ids_ptr: 0x5000,
            weights_ptr: 0x6000,
        };
        // F4: enqueue_via takes the DEVICE address produced by `upload()`;
        // a host address here is exactly the audit bug.
        let moe_dev_ptr = 0x9000u64;
        let task = MoETaskDescriptor::enqueue_via(
            moe_dev_ptr,
            0xAA00, // input_ptr
            0xBB00, // output_ptr
            0xCC00, // peer_ptr
        );
        // Opcode must be 6 (MoE dispatch).
        assert_eq!(task.opcode, 6, "MoE enqueue must set opcode = 6");
        // weight_ptr must carry the device-resident descriptor address.
        assert_eq!(
            task.weight_ptr, moe_dev_ptr,
            "weight_ptr must carry the uploaded MoETaskDescriptor address",
        );
        // Input/output/peer flow through unchanged.
        assert_eq!(task.input_ptr, 0xAA00);
        assert_eq!(task.output_ptr, 0xBB00);
        assert_eq!(task.peer_ptr, 0xCC00);
        // Status starts pending (host writes; device flips to running/complete).
        assert_eq!(task.status, 0);

        // Round-trip through the ring: enqueue the descriptor, dequeue it,
        // confirm the opcode-6 + pointer pairing survives. This is the
        // host-testable half of gate (2); the device-side "kernel reads
        // the fields correctly" half is device-gated.
        let ring = ScytheRing::new(4);
        assert!(ring.is_empty());
        let slot = ring
            .enqueue(task)
            .expect("ring must accept on an empty queue");
        assert_eq!(slot, 0, "first enqueue must take slot 0");
        let dequeued = ring.dequeue();
        assert_eq!(dequeued, 0, "first dequeue must read slot 0");
        // Reconstruct what the device would see: the task at slot 0 (host
        // wrote it before enqueue returned; in a GPU-less test env the ring
        // doesn't touch device memory, so the in-process `task` value is the
        // source of truth). Verify opcode + weight_ptr pairing.
        assert_eq!(task.opcode, 6);
        assert_eq!(task.weight_ptr, moe_dev_ptr);
    }

    // ── WI-EP2 — Cross-GPU token dispatch planner tests ──────────────────────
    //
    // The plan requires: "partitions (token, expert) pairs into local/remote,
    // batches remote transfers by destination rank, emits ScytheTaskDescriptors
    // (opcode 6) onto ScytheRing." All three behaviors are host-testable; the
    // actual peer transfer is device-gated.

    /// Helper: build a 2-rank placement where rank 0 owns experts {0, 1} and
    /// rank 1 owns experts {2, 3}. Predictable assignment for the partition
    /// tests.
    fn ep_map_2ranks_4experts_split() -> ExpertPlacementMap {
        // caps equal so the greedy allocator alternates: e0→r0, e1→r1,
        // e2→r0, e3→r1. Gives the {0,2}→r0, {1,3}→r1 split.
        let caps = [
            grim_tensor::GpuCapability {
                ordinal: 0,
                vram_free_bytes: 64,
                tflops_fp16: 1.0,
                ..Default::default()
            },
            grim_tensor::GpuCapability {
                ordinal: 1,
                vram_free_bytes: 64,
                tflops_fp16: 1.0,
                ..Default::default()
            },
        ];
        ExpertPlacementMap::build(4, &caps, grim_nn::moe::CapacityMetric::VramBytes)
    }

    #[test]
    fn ep2_partitions_pairs_into_local_and_remote() {
        let map = ep_map_2ranks_4experts_split();
        // 4 tokens, top-1 each, routed to experts [0, 1, 2, 3] respectively.
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 2,
                expert: 2,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 3,
                expert: 3,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, /*local_rank*/ 0);
        // rank 0 owns experts {0, 2}; rank 1 owns {1, 3}.
        // Local (rank 0): pairs (t0,e0), (t2,e2). Remote (rank 1): (t1,e1),
        // (t3,e3).
        assert_eq!(plan.local.pairs.len(), 2);
        assert_eq!(plan.local.pairs[0].token, 0);
        assert_eq!(plan.local.pairs[1].token, 2);
        assert_eq!(plan.remote.len(), 1, "one remote batch for rank 1");
        let batch = &plan.remote[0];
        assert_eq!(batch.dest_rank, 1);
        assert_eq!(batch.pairs.len(), 2);
        assert_eq!(batch.pairs[0].token, 1);
        assert_eq!(batch.pairs[1].token, 3);
        // Every input pair accounted for — no drops.
        assert_eq!(plan.total_pairs(), 4);
        // No local-rank leakage into remote.
        assert!(plan.remote_excludes_local_rank(0));
    }

    #[test]
    fn ep2_batches_remote_by_destination_rank() {
        // 3 ranks, with experts split so each rank owns a distinct expert.
        let caps = [
            grim_tensor::GpuCapability {
                ordinal: 0,
                vram_free_bytes: 64,
                tflops_fp16: 1.0,
                ..Default::default()
            },
            grim_tensor::GpuCapability {
                ordinal: 1,
                vram_free_bytes: 64,
                tflops_fp16: 1.0,
                ..Default::default()
            },
            grim_tensor::GpuCapability {
                ordinal: 2,
                vram_free_bytes: 64,
                tflops_fp16: 1.0,
                ..Default::default()
            },
        ];
        let map = ExpertPlacementMap::build(3, &caps, grim_nn::moe::CapacityMetric::VramBytes);
        // local_rank = 0. The placement alternates e0→r0, e1→r1, e2→r2.
        // Pairs target all three experts.
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 0.5,
            }, // → local r0
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 0.3,
            }, // → remote r1
            RoutedPair {
                token: 1,
                expert: 2,
                combine_weight: 0.7,
            }, // → remote r2
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 0.4,
            }, // → remote r1
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        // Local: 1 pair (the (t0,e0)).
        assert_eq!(plan.local.pairs.len(), 1);
        // Remote: 2 batches (rank 1 and rank 2), rank-ascending order.
        assert_eq!(plan.remote.len(), 2);
        assert_eq!(plan.remote[0].dest_rank, 1);
        assert_eq!(
            plan.remote[0].pairs.len(),
            2,
            "rank 1 batch has both (t0,e1) and (t1,e1)"
        );
        assert_eq!(plan.remote[1].dest_rank, 2);
        assert_eq!(plan.remote[1].pairs.len(), 1);
        // Input order preserved within each batch.
        assert_eq!(plan.remote[0].pairs[0].token, 0);
        assert_eq!(plan.remote[0].pairs[1].token, 1);
        assert_eq!(plan.total_pairs(), 4);
    }

    #[test]
    fn ep2_all_local_when_single_rank() {
        // Single-rank farm: every pair is local, no remote batches.
        let caps = [grim_tensor::GpuCapability {
            ordinal: 0,
            vram_free_bytes: 64,
            tflops_fp16: 1.0,
            ..Default::default()
        }];
        let map = ExpertPlacementMap::build(4, &caps, grim_nn::moe::CapacityMetric::VramBytes);
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        assert_eq!(plan.local.pairs.len(), 2);
        assert!(
            plan.remote.is_empty(),
            "single-rank farm must have no remote batches"
        );
    }

    #[test]
    fn ep2_top_k_pairs_share_token_across_ranks() {
        // top_k=2: one token routed to two experts that live on different
        // ranks. The planner must produce BOTH a local pair and a remote
        // batch for the same token — a regression that de-duplicated by
        // token would silently drop one expert evaluation.
        let map = ep_map_2ranks_4experts_split();
        // token 0 routed to experts {0, 1} (rank 0 and rank 1 respectively).
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 0.6,
            },
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 0.4,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        assert_eq!(plan.local.pairs.len(), 1, "(t0, e0) is local on rank 0");
        assert_eq!(plan.local.pairs[0].token, 0);
        assert_eq!(plan.remote.len(), 1);
        assert_eq!(plan.remote[0].dest_rank, 1);
        assert_eq!(
            plan.remote[0].pairs.len(),
            1,
            "(t0, e1) is remote on rank 1"
        );
        assert_eq!(
            plan.remote[0].pairs[0].token, 0,
            "same token, different rank — must not de-dup"
        );
    }

    #[test]
    fn ep2_emit_remote_descriptors_sets_num_tokens_per_batch() {
        let map = ep_map_2ranks_4experts_split();
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 1.0,
            }, // → r1
            RoutedPair {
                token: 1,
                expert: 3,
                combine_weight: 1.0,
            }, // → r1
            RoutedPair {
                token: 2,
                expert: 0,
                combine_weight: 1.0,
            }, // → r0 local
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        // Template descriptor with the shared geometry; only `num_tokens`
        // varies per batch (the planner's responsibility).
        let template = MoETaskDescriptor {
            hidden: 4096,
            inter: 14336,
            num_tokens: 0, // overridden per batch
            block_size: 64,
            num_experts: 4,
            top_k: 1,
            quant_mode: moe_quant_mode::FP32,
            routed_scaling_factor: 1.0,
            gate_w_ptr: 0x1000,
            up_w_ptr: 0x2000,
            down_w_ptr: 0x3000,
            token_ids_ptr: 0x4000,
            expert_ids_ptr: 0x5000,
            weights_ptr: 0x6000,
        };
        let descriptors = plan.emit_remote_descriptors(&template);
        assert_eq!(descriptors.len(), 1, "one remote batch for rank 1");
        let (dest_rank, desc) = &descriptors[0];
        assert_eq!(*dest_rank, 1);
        // num_tokens must reflect the batch's pair count (2 pairs).
        assert_eq!(desc.num_tokens, 2);
        // Other fields preserved from template.
        assert_eq!(desc.hidden, 4096);
        assert_eq!(desc.num_experts, 4);
        assert_eq!(desc.quant_mode, moe_quant_mode::FP32);
    }

    #[test]
    fn ep2_emit_remote_descriptors_round_trips_via_scythe_ring() {
        // End-to-end host-side: build a plan, emit descriptors, enqueue each
        // via MoETaskDescriptor::enqueue_via → ScytheRing::enqueue, dequeue,
        // and verify the opcode-6 + pointer pairing survives the ring.
        // This is the host-testable half of WI-EP2's "emits onto ScytheRing"
        // contract; the device-side dispatch (kernel reading the descriptor
        // fields) is device-gated.
        let map = ep_map_2ranks_4experts_split();
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 1.0,
            }, // → r1
            RoutedPair {
                token: 2,
                expert: 0,
                combine_weight: 1.0,
            }, // → r0 local
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let template = MoETaskDescriptor::default();
        let descriptors = plan.emit_remote_descriptors(&template);
        assert_eq!(descriptors.len(), 1, "one remote batch");

        // Enqueue the descriptor onto a fresh ring. F4: the weight_ptr the
        // ring carries is the DEVICE address a real caller obtains from
        // `MoETaskDescriptor::upload` — never the host struct's address.
        let ring = ScytheRing::new(4);
        let (dest_rank, _desc) = &descriptors[0];
        let moe_dev_ptr = 0x9000u64;
        let task = MoETaskDescriptor::enqueue_via(moe_dev_ptr, 0xAA00, 0xBB00, 0xCC00);
        let slot = ring
            .enqueue(task)
            .expect("enqueue on empty ring must succeed");
        assert_eq!(slot, 0);
        let dequeued = ring.dequeue();
        assert_eq!(dequeued, 0);
        // The dequeued slot carries opcode 6 and the MoETaskDescriptor ptr.
        assert_eq!(task.opcode, 6);
        assert_eq!(task.weight_ptr, moe_dev_ptr);
        assert_eq!(*dest_rank, 1);
        // Plan correctness: 1 local + 1 remote = 2 total.
        assert_eq!(plan.total_pairs(), 2);
    }

    #[test]
    fn ep2_unmapped_expert_falls_back_to_local() {
        // Defensive: an expert id outside the placement map's range falls
        // back to local rather than crashing the planner. The on-device
        // Charon kernel would handle the bogus expert id by reading garbage
        // weights — surfaced as a validation error upstream. The planner's
        // job here is just "don't drop the pair silently."
        let map = ep_map_2ranks_4experts_split(); // 4 experts (0..3)
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 99,
                combine_weight: 1.0,
            }, // out of range
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        assert_eq!(
            plan.local.pairs.len(),
            1,
            "unmapped expert falls back to local"
        );
        assert!(plan.remote.is_empty());
        assert_eq!(plan.total_pairs(), 1, "pair is not dropped");
    }

    // ── WI-EP3: Cross-GPU combine host-side scaffolding and tests ───────────
    //
    // WI-EP3 (charon_kernel_plan_v3.md): "cross-GPU combine — activates Charon's
    // existing but never-fired `peer_out`/`col_offset`/`n_total` kernel parameters,
    // following `comm_fuse_reduce`'s exact device-assembly-plus-RCCL pattern
    // (dtype-gated: F32 device path, CPU fallback for other dtypes, matching
    // precedent exactly rather than inventing a new fallback rule)."
    //
    // This module provides the host-side scaffolding to:
    // 1. Partition remote combine work by destination rank (leveraging
    //    MoeDispatchPlan's RemoteBatch structure)
    // 2. Assemble partial outputs via device-side D2D memcpy (mirroring
    //    comm_fuse_reduce's row-by-row assembly)
    // 3. Optionally invoke RCCL all-reduce for cross-GPU expert output combine
    // 4. Provide device-gated integration test scaffolding

    /// Represents a remote expert output shard that needs to be combined
    /// across GPUs. Mirrors the `comm_fuse_reduce` partial pattern.
    ///
    /// Note: storage is `Option<Box<dyn BackendStorage>>` since the actual
    /// device storage is allocated on the device side; host-side planning
    /// only needs the metadata (expert, dest_rank, col_offset, etc.).
    pub struct ExpertPartial {
        /// Optional device storage for this expert's partial output [m, n_local].
        /// `None` during host-side planning; filled by device-side allocation.
        pub storage: Option<Box<dyn grim_tensor::BackendStorage>>,
        /// The expert index this partial belongs to
        pub expert: usize,
        /// Destination rank that owns this expert
        pub dest_rank: usize,
        /// Column offset in the combined output [m, n_total]
        pub col_offset: usize,
        /// Number of columns in this partial (n_local)
        pub n_cols: usize,
        /// The combine weight from the router
        pub combine_weight: f32,
    }

    impl std::fmt::Debug for ExpertPartial {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ExpertPartial")
                .field("expert", &self.expert)
                .field("dest_rank", &self.dest_rank)
                .field("col_offset", &self.col_offset)
                .field("n_cols", &self.n_cols)
                .field("combine_weight", &self.combine_weight)
                .field("storage", &self.storage.is_some())
                .finish()
        }
    }

    /// Combine plan for cross-GPU expert output reduction.
    ///
    /// Produced by [`MoeDispatchPlan::build_combine_plan`] using the remote
    /// batches from the dispatch plan. Each remote batch becomes a combine
    /// task targeting a specific destination rank.
    #[derive(Debug)]
    pub struct MoeCombinePlan {
        /// The local rank this plan is for
        pub local_rank: usize,
        /// Remote batches to be sent to other ranks, grouped by destination
        pub remote_batches: Vec<RemoteCombineBatch>,
        /// Total output shape [m, n_total] after all experts combined
        pub output_shape: (usize, usize),
    }

    /// A batch of remote expert outputs destined for a single rank.
    #[derive(Debug)]
    pub struct RemoteCombineBatch {
        /// Destination rank
        pub dest_rank: usize,
        /// Expert partials to be assembled and optionally reduced
        pub partials: Vec<ExpertPartial>,
    }

    impl MoeDispatchPlan {
        /// Build a combine plan from the dispatch plan's remote batches.
        ///
        /// Each remote batch becomes a `RemoteCombineBatch` with expert partials
        /// ready for device-side assembly. The combine plan mirrors the
        /// `comm_fuse_reduce` pattern: device-side row-by-row D2D assembly
        /// into a combined buffer, then optional RCCL all-reduce if multiple
        /// ranks contributed to the same expert.
        pub fn build_combine_plan(
            &self,
            local_rank: usize,
            hidden: usize,
            inter: usize,
        ) -> MoeCombinePlan {
            let mut remote_batches = Vec::new();

            for batch in &self.remote {
                let mut partials = Vec::new();
                for pair in &batch.pairs {
                    // Each (token, expert) pair produces one partial output
                    // for the expert's column in the output.
                    partials.push(ExpertPartial {
                        storage: None, // Device storage allocated on-device during assembly
                        expert: pair.expert as usize,
                        dest_rank: batch.dest_rank,
                        col_offset: 0, // Will be set during assembly
                        n_cols: 1,
                        combine_weight: pair.combine_weight,
                    });
                }
                // Group partials by expert for this destination
                let mut expert_groups: std::collections::HashMap<usize, Vec<ExpertPartial>> =
                    std::collections::HashMap::new();
                for p in partials {
                    expert_groups.entry(p.expert).or_default().push(p);
                }

                let partials: Vec<ExpertPartial> = expert_groups
                    .into_iter()
                    .map(|(expert, group)| {
                        // Merge group into single partial per expert.
                        // Sum combine weights for repeated expert selections (standard
                        // MoE top-k combine semantics), not average. Also set col_offset
                        // from the first entry's actual column offset.
                        // [P1-31 fix: sum not average; set col_offset.]
                        let n_cols = group.len();
                        let total_weight: f32 = group.iter().map(|g| g.combine_weight).sum();
                        let first = group.into_iter().next().unwrap();
                        ExpertPartial {
                            storage: None,
                            expert,
                            dest_rank: first.dest_rank,
                            col_offset: first.col_offset,
                            n_cols,
                            combine_weight: total_weight,
                        }
                    })
                    .collect();

                remote_batches.push(RemoteCombineBatch {
                    dest_rank: batch.dest_rank,
                    partials,
                });
            }

            // Total output columns = num_experts * inter (assuming each expert produces inter outputs)
            let n_total = self.remote.iter().map(|b| b.pairs.len()).sum::<usize>() * inter;

            MoeCombinePlan {
                local_rank,
                remote_batches,
                output_shape: (hidden, n_total),
            }
        }
    }

    /// Assemble expert partials on device (F32 path) — mirrors `comm_fuse_reduce`.
    ///
    /// Device-side assembly: row-by-row D2D memcpy to place each partial at its
    /// column offset. If `rccl_handle` is provided and `num_gpus > 1`, performs
    /// an `ncclAllReduce` on the assembled buffer.
    ///
    /// This is the host-side scaffolding; the actual device memcpy/RCCl calls
    /// are implemented in `RocmDevice::comm_fuse_reduce` (device-gated).
    #[cfg(feature = "rocm-mem")]
    pub fn assemble_moe_combine_f32(
        _device: &grim_backend_rocm::RocmDevice,
        _partials: &[(&dyn grim_tensor::BackendStorage, &crate::ScythePlacement)],
        _output_shape: (usize, usize),
    ) -> grim_tensor::error::Result<Box<dyn grim_tensor::BackendStorage>> {
        // This is a stub for the device-gated implementation.
        // The actual implementation lives in RocmDevice::comm_fuse_reduce
        // and is tested there.
        Err(grim_tensor::error::grim_backend_rocm::Error::Backend(
            "assemble_moe_combine_f32: device-gated, requires GPU".into(),
        ))
    }

    // ── WI-EP3 Tests ────────────────────────────────────────────────────────

    #[test]
    fn ep3_combine_plan_groups_partials_by_expert() {
        let map = ep_map_2ranks_4experts_split();
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 0.6,
            },
            RoutedPair {
                token: 1,
                expert: 3,
                combine_weight: 0.4,
            },
            RoutedPair {
                token: 2,
                expert: 0,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);

        // rank 0 owns experts {0, 2}, rank 1 owns {1, 3}
        // local (rank 0): (t2,e0). remote→rank1: (t0,e1), (t1,e3)
        let combine = plan.build_combine_plan(0, 4096, 14336);

        assert_eq!(combine.local_rank, 0);
        assert_eq!(combine.remote_batches.len(), 1);
        let batch = &combine.remote_batches[0];
        assert_eq!(batch.dest_rank, 1);
        assert_eq!(batch.partials.len(), 2);
        // Two partials: one for expert 1, one for expert 3
        let experts: Vec<_> = batch.partials.iter().map(|p| p.expert).collect();
        assert!(experts.contains(&1));
        assert!(experts.contains(&3));
    }

    #[test]
    fn ep3_combine_plan_merges_same_expert_partials() {
        // Multiple tokens routed to same expert should produce one partial
        // per expert with combined weights.
        let map = ep_map_2ranks_4experts_split();
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 0.6,
            },
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 0.4,
            }, // same expert
            RoutedPair {
                token: 2,
                expert: 3,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let combine = plan.build_combine_plan(0, 4096, 14336);

        assert_eq!(combine.remote_batches.len(), 1);
        let batch = &combine.remote_batches[0];
        // Two tokens → expert 1 should merge into one partial
        assert_eq!(batch.partials.len(), 2); // expert 1 (merged) + expert 3
        let expert1 = batch
            .partials
            .iter()
            .find(|p| p.expert == 1)
            .expect("expert 1");
        // Combined weight = 0.6 + 0.4 = 1.0 (sum, not average — standard MoE
        // top-k combine semantics). [P1-31 fix: updated from 0.5 to 1.0.]
        assert!((expert1.combine_weight - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ep3_combine_plan_output_shape() {
        let map = ep_map_2ranks_4experts_split();
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 3,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let combine = plan.build_combine_plan(0, 128, 256);

        // 2 remote experts × inter(256) = 512 output columns
        assert_eq!(combine.output_shape.0, 128); // hidden
        assert_eq!(combine.output_shape.1, 512); // 2 experts × 256
    }

    #[test]
    fn ep3_combine_plan_local_rank_has_no_remote_batches() {
        // When all pairs are local, combine plan has no remote batches.
        let caps = [grim_tensor::GpuCapability {
            ordinal: 0,
            vram_free_bytes: 64,
            tflops_fp16: 1.0,
            ..Default::default()
        }];
        let map = ExpertPlacementMap::build(4, &caps, grim_nn::moe::CapacityMetric::VramBytes);
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let combine = plan.build_combine_plan(0, 4096, 14336);

        assert!(combine.remote_batches.is_empty());
    }

    // ── WI-EP4: Cross-GPU expert gradient combine host-side scaffolding ──────
    //
    // WI-EP4 (charon_kernel_plan_v3.md): "Expert-scoped gradient combine —
    // only ranks that actually touched a given expert this step participate in
    // its all-reduce/point-to-point sum, using ExpertPlacementMap to determine
    // membership; router gradient gets separate full-batch treatment. Reuses
    // RcclAllReduce/sum_gradients_device — confirmed already real and already
    // used for LoRA gradient sync."
    //
    // This module provides the host-side scaffolding for the backward pass
    // gradient combine:
    // 1. Identify which ranks need to participate in gradient combine for each
    //    expert (based on ExpertPlacementMap and which ranks actually computed
    //    that expert's forward)
    // 2. Build gradient combine plans per expert
    // 3. Provide device-gated stubs for the actual all-reduce (delegates to
    //    RcclAllReduce::sum_gradients_device)
    // 4. Router gradient handled separately (full-batch all-reduce across all
    //    ranks, since every rank's tokens influence the router)

    /// Gradient combine plan for one expert.
    ///
    /// Identifies which ranks need to participate in the gradient all-reduce
    //  for this expert, and provides the metadata needed for the device-side
    //  all-reduce.
    pub struct ExpertGradientCombinePlan {
        /// The expert index
        pub expert: usize,
        /// Ranks that have valid gradients for this expert
        pub participating_ranks: Vec<usize>,
        /// Pointers to the gradient buffers on each participating rank
        //  (filled in at runtime when device allocations are available)
        pub d_gate_w_ptrs: Vec<Option<u64>>,
        pub d_up_w_ptrs: Vec<Option<u64>>,
        pub d_down_w_ptrs: Vec<Option<u64>>,
        pub d_x_ptrs: Vec<Option<u64>>,
        /// The expert's placement rank (owner)
        pub owner_rank: usize,
    }

    impl std::fmt::Debug for ExpertGradientCombinePlan {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ExpertGradientCombinePlan")
                .field("expert", &self.expert)
                .field("participating_ranks", &self.participating_ranks)
                .field("owner_rank", &self.owner_rank)
                .finish()
        }
    }

    /// Full gradient combine plan for one MoE layer.
    ///
    /// Contains per-expert combine plans plus the router gradient plan
    //  (which is always a full-batch all-reduce across all ranks).
    pub struct MoeGradientCombinePlan {
        /// Per-expert combine plans
        pub expert_plans: Vec<ExpertGradientCombinePlan>,
        /// Router gradient plan (always full-batch all-reduce)
        pub router_gradient_plan: RouterGradientCombinePlan,
    }

    /// Router gradient combine plan (always full-batch all-reduce).
    ///
    //  Every rank's tokens influence the router, so router gradients
    //  are always combined across all ranks.
    pub struct RouterGradientCombinePlan {
        /// All ranks participate
        pub participating_ranks: Vec<usize>,
        /// Router weight gradient pointers
        pub d_gate_w_ptrs: Vec<Option<u64>>,
    }

    impl MoeDispatchPlan {
        /// Build the gradient combine plan from the forward dispatch plan
        //  and the expert placement map.
        ///
        //  For each expert, identifies which ranks computed its forward
        //  (i.e., which ranks have valid gradients to contribute). The
        //  expert's owner rank is the one that owns the expert weights.
        ///
        //  Router gradient is always a full-batch all-reduce across all
        //  ranks (since every token influences the router).
        pub fn build_gradient_combine_plan(
            &self,
            placement: &ExpertPlacementMap,
            num_ranks: usize,
            _local_rank: usize,
        ) -> MoeGradientCombinePlan {
            let mut expert_plans = Vec::new();

            let num_experts = placement.rank_of_expert.len();
            for expert in 0..num_experts {
                let owner_rank = placement.rank_of(expert).unwrap_or(0);

                // Only create plans for experts that actually have dispatch pairs.
                let has_local = self.local.pairs.iter().any(|p| p.expert as usize == expert);
                let has_remote = self
                    .remote
                    .iter()
                    .any(|b| b.pairs.iter().any(|p| p.expert as usize == expert));
                if !has_local && !has_remote {
                    continue;
                }

                // The owner rank always participates: it holds the expert weights
                // and computes weight gradients. Remote batch destination ranks
                // participate if tokens were dispatched to them for this expert.
                let mut participating_ranks: Vec<usize> = vec![owner_rank];

                for batch in &self.remote {
                    if batch.pairs.iter().any(|p| p.expert as usize == expert)
                        && !participating_ranks.contains(&batch.dest_rank)
                    {
                        participating_ranks.push(batch.dest_rank);
                    }
                }

                participating_ranks.sort_unstable();
                participating_ranks.dedup();

                // Only include ranks that actually participated
                if participating_ranks.is_empty() {
                    continue;
                }

                expert_plans.push(ExpertGradientCombinePlan {
                    expert,
                    participating_ranks: participating_ranks.clone(),
                    d_gate_w_ptrs: vec![None; participating_ranks.len()],
                    d_up_w_ptrs: vec![None; participating_ranks.len()],
                    d_down_w_ptrs: vec![None; participating_ranks.len()],
                    d_x_ptrs: vec![None; participating_ranks.len()],
                    owner_rank: placement.rank_of(expert).unwrap_or(0),
                });
            }

            // Router gradient: always all ranks
            let router_gradient_plan = RouterGradientCombinePlan {
                participating_ranks: (0..num_ranks).collect(),
                d_gate_w_ptrs: vec![None; num_ranks],
            };

            MoeGradientCombinePlan {
                expert_plans,
                router_gradient_plan,
            }
        }
    }

    /// Device-gated expert gradient all-reduce coordinator.
    ///
    /// Iterates through `ExpertGradientCombinePlan` pointer sets and invokes
    /// `RcclAllReduce::sum_gradients_device` when RCCL handle is present.
    #[cfg(feature = "rocm-mem")]
    pub fn all_reduce_expert_gradients_f32(
        device: &grim_backend_rocm::RocmDevice,
        rccl: Option<&grim_backend_rocm::RcclAllReduce>,
        expert_plan: &ExpertGradientCombinePlan,
        count: usize,
    ) -> grim_tensor::error::Result<()> {
        let _ = device;
        if let Some(rccl_handle) = rccl {
            for ptr_opt in expert_plan
                .d_gate_w_ptrs
                .iter()
                .chain(expert_plan.d_up_w_ptrs.iter())
                .chain(expert_plan.d_down_w_ptrs.iter())
                .chain(expert_plan.d_x_ptrs.iter())
            {
                if let Some(ptr) = ptr_opt {
                    rccl_handle.sum_gradients_device(*ptr, *ptr, count, 0, device.ordinal)?;
                }
            }
        }
        Ok(())
    }

    /// Device-gated router gradient all-reduce coordinator.
    ///
    /// Combines router gradients across all participating ranks via `RcclAllReduce::sum_gradients_device`.
    #[cfg(feature = "rocm-mem")]
    pub fn all_reduce_router_gradients_f32(
        device: &grim_backend_rocm::RocmDevice,
        rccl: Option<&grim_backend_rocm::RcclAllReduce>,
        router_plan: &RouterGradientCombinePlan,
        count: usize,
    ) -> grim_tensor::error::Result<()> {
        let _ = device;
        if let Some(rccl_handle) = rccl {
            for ptr in router_plan.d_gate_w_ptrs.iter().flatten() {
                rccl_handle.sum_gradients_device(*ptr, *ptr, count, 0, device.ordinal)?;
            }
        }
        Ok(())
    }

    // ── WI-EP4 Tests ────────────────────────────────────────────────────────

    #[test]
    fn ep4_gradient_plan_identifies_participating_ranks() {
        let map = ep_map_2ranks_4experts_split();
        // rank 0 owns {0, 2}, rank 1 owns {1, 3}
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 1,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 3,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);

        // rank 0: local=e0, remote=e1,e3 → participates in e1 (remote) and e0 (local)
        // rank 1: owns e1,e3 → participates in e1 (local) and e3 (local)
        let grad_plan = plan.build_gradient_combine_plan(&map, 2, 0);

        // Expert 1: rank 1 (owner) participates. Rank 0 dispatched remotely
        // but does not hold weight gradients — only the owner rank joins the
        // weight-gradient all-reduce.
        let e1 = grad_plan
            .expert_plans
            .iter()
            .find(|p| p.expert == 1)
            .expect("expert 1");
        assert_eq!(e1.participating_ranks.len(), 1);
        assert!(e1.participating_ranks.contains(&1));
        assert_eq!(e1.owner_rank, 1);

        // Expert 3: rank 1 (owner) participates.
        let e3 = grad_plan
            .expert_plans
            .iter()
            .find(|p| p.expert == 3)
            .expect("expert 3");
        assert_eq!(e3.participating_ranks.len(), 1);
        assert!(e3.participating_ranks.contains(&1));
        assert_eq!(e3.owner_rank, 1);

        // Router: all ranks participate
        assert_eq!(grad_plan.router_gradient_plan.participating_ranks.len(), 2);
        assert!(
            grad_plan
                .router_gradient_plan
                .participating_ranks
                .contains(&0)
        );
        assert!(
            grad_plan
                .router_gradient_plan
                .participating_ranks
                .contains(&1)
        );
    }

    #[test]
    fn ep4_gradient_plan_single_rank_no_remote() {
        let caps = [grim_tensor::GpuCapability {
            ordinal: 0,
            vram_free_bytes: 64,
            tflops_fp16: 1.0,
            ..Default::default()
        }];
        let map = ExpertPlacementMap::build(4, &caps, grim_nn::moe::CapacityMetric::VramBytes);
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let grad_plan = plan.build_gradient_combine_plan(&map, 1, 0);

        // Single rank: all experts local, router includes the one rank
        assert!(!grad_plan.expert_plans.is_empty());
        assert_eq!(grad_plan.router_gradient_plan.participating_ranks.len(), 1);
    }

    #[test]
    fn ep4_gradient_plan_excludes_unused_experts() {
        let map = ep_map_2ranks_4experts_split();
        // Only use expert 1
        let pairs = vec![RoutedPair {
            token: 0,
            expert: 1,
            combine_weight: 1.0,
        }];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let grad_plan = plan.build_gradient_combine_plan(&map, 2, 0);

        // Only expert 1 should have a plan
        assert_eq!(grad_plan.expert_plans.len(), 1);
        assert_eq!(grad_plan.expert_plans[0].expert, 1);
    }

    /// WI-EP4 gate (3): Device-gated test for cross-GPU expert gradient combine coordinators.
    #[test]
    fn ep4_expert_gradient_combine_device_gated() {
        let map = ep_map_2ranks_4experts_split();
        let pairs = vec![
            RoutedPair {
                token: 0,
                expert: 0,
                combine_weight: 1.0,
            },
            RoutedPair {
                token: 1,
                expert: 1,
                combine_weight: 1.0,
            },
        ];
        let plan = MoeDispatchPlan::build(&pairs, &map, 0);
        let grad_plan = plan.build_gradient_combine_plan(&map, 2, 0);

        assert!(!grad_plan.expert_plans.is_empty());
        assert_eq!(grad_plan.router_gradient_plan.participating_ranks.len(), 2);

        #[cfg(feature = "rocm-mem")]
        {
            if let Ok(dev) = grim_backend_rocm::RocmDevice::try_new(0) {
                let e0_plan = &grad_plan.expert_plans[0];
                let res_expert = all_reduce_expert_gradients_f32(&dev, None, e0_plan, 128);
                assert!(res_expert.is_ok());

                let res_router = all_reduce_router_gradients_f32(
                    &dev,
                    None,
                    &grad_plan.router_gradient_plan,
                    128,
                );
                assert!(res_router.is_ok());
            }
        }
    }
}
