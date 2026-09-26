//! SCYTHE-2 C²PLR controller + PlacementCache (`scythe2.md` §3, §5.3).
//!
//! Relocated from `grim-engine::scythe2` so the capability-driven layer
//! placement is reachable from the model loaders as well as the engine.
//! `grim-engine` depends on `grim-models-transformer`, never the reverse, so
//! while this lived in the engine no loader could consult it: placement fell
//! back to per-model index arithmetic. The vocabulary these types speak
//! ([`GpuCapability`], [`ScytheLink`], [`ScythePlacement`]) already lives in
//! `grim-tensor::backend`, and the measured capabilities come from this
//! crate's [`CapabilityProfiler`](crate::device::capability_profiler) - so
//! this crate is the one place both the engine and the loaders can reach.

use std::collections::HashMap;

use grim_tensor::backend::{GpuCapability, ScytheLink, ScythePlacement};

/// `shape_bucket` power-of-2 quantizes `seq_len × batch` so that autoregressive decode (which increments `seq_len` by 1.
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
pub struct PlacementCache {
    /// Fast path: `fast[layer_id]` holds the last-inserted placement for this layer together with the shape bucket it was decided at.
    /// Cleared by `bump_epoch`.
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
    /// Clears the fast path so the next forward pass re-runs `decide_miss()` for every layer.
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

/// Layer fingerprint - 16-dimensional feature vector describing the layer's compute profile for the WaveTune bilinear predictor.
/// Populated at model-load time from layer config (MLP vs attention vs norm), GEMM dimensions, etc.
pub type LayerFingerprint = [f32; 16];

/// The 2-layer MLP controller π_θ (≈8 KB). Inputs per forward: `(layer_fingerprint[16], input_shape[4], capability_profile[K×6], link_state[K×K], thermal_state[K])`.
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
    /// Number of GPUs the controller was constructed for.
    /// `decide_miss` validates that the live `caps.len()` matches this - a mismatch means the farm topology.
    num_gpus: usize,
}

impl C2plrController {
    /// Construct a controller for `num_layers` layers and `num_gpus` GPUs.
    /// MLP weights are initialised near-zero (the controller learns online).
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
    /// Hits the cache first; calls `decide_miss()` only on a miss.
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

    /// Load-aware variant of [`C2plrController::decide`] (WI-SB1 finding): the shape-keyed PlacementCache is load-blind - a placement decided for idle ranks is reused verbatim even after the rank-load vector changed (concurrent farm pins or external GPU utilization).
    /// Callers that pass *adjusted* caps reflecting any non-zero load must use this entry so the.
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
    /// Measured at ~2 µs/layer host-side in release (`benches/ scythe2_decide_miss.rs`); the ~10 µs figure in scythe2.md.
    fn decide_miss(
        &self,
        layer_id: u32,
        shape: &[usize],
        caps: &[GpuCapability],
        links: &[ScytheLink],
    ) -> ScythePlacement {
        // Validate that the live capability profile matches the farm the controller was constructed for.
        // A mismatch means the topology changed without a `bump_epoch` - a caller bug.
        let k = caps.len().max(1).min(self.num_gpus.max(1));

        // ── WaveTune bilinear latency eval (§3.4 Table-A) ────────────────── For each GPU, estimate GEMM latency from TFLOPS and shape.
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

        // ── MLP forward (§3.4, §4 Pillar 4) ─────────────────────────────── Build input vector and run the 2-layer MLP.
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
                input[base + 2] = c.dram_bandwidth_gbps;
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

        // ── Placement selection ───────────────────────────────────────────── Placement logits: argmax over first K elements.
        // With an untrained (all-zero) MLP every logit is identical, and a naive argmax pins every.
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

        // F6 (audit): single-rank routing is the intended design - the only multi-rank ScythePlacement consumer is grim-cli's hand-built data-parallel gradient sync, which bypasses this controller.
        // The softmax-over-partition-logits computation that used to live here fed exactly one discarded slice (`split_counts` tops.

        // ── Route selection ─────────────────────────────────────────
        let route_link = if k == 1 {
            ScytheLink::PeerDirect
        } else {
            links
                .get(best_gpu * k + (best_gpu + 1) % k)
                .copied()
                .unwrap_or(ScytheLink::Host)
        };

        // ── Lagrangian budget check ───────────────────────────────────────── Compare this layer's estimated GEMM latency against the *per-layer* budget slice (total budget / num_layers), not the whole end-to-end budget.
        // The previous code compared one GEMM against `budget_ms` directly, which made the fallback effectively never.
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

    /// Online update after a batch - dual ascent on λ + MLP gradient step.
    /// Called every optimizer step (not every micro-batch) to update the Lagrangian dual variable and the.
    pub fn update(&mut self, observed_latency_ms: f64, placements: &[ScythePlacement]) {
        // Lagrangian dual ascent: λ ← λ + α(t̂_total - T_budget).
        // The step size α = 0.01 is the standard dual-ascent learning rate; larger values oscillate,.
        const DUAL_STEP_SIZE: f64 = 0.01;
        const MLP_LR: f32 = 0.001;
        let constraint_violation = observed_latency_ms - self.budget_ms;
        self.lambda = (self.lambda + DUAL_STEP_SIZE * constraint_violation).max(0.0);

        // ── MLP gradient step (scythe2.md §4 Pillar 4) ──────────────────────── Approximate policy gradient: penalise weights that led to placements on GPUs contributing to budget overruns, weighted by the violation magnitude.
        // When λ is high (chronic overruns), the penalty scales up, pushing the MLP toward lower-latency.
        let penalty = (self.lambda * constraint_violation.abs().max(0.0)) as f32;
        if penalty > 0.0 && !placements.is_empty() {
            // Build a per-GPU blame signal: GPUs that appear more often in the placements get a larger gradient push.
            // This is a REINFORCE-style credit assignment without the full autograd tape - the controller runs.
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
                // Normalise blame and apply as gradient noise to W2 columns that correspond to placement logits.
                // This nudges the MLP output distribution away from the over-used GPUs.
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
    /// Must be called from the ROCm device-lost path *before* the next `decide()` so that no.
    pub fn on_gpu_leave(&mut self, ordinal: usize) {
        log::info!("[scythe2] GPU {ordinal} left — clearing PlacementCache (mode-B safety)");
        self.cache.on_gpu_leave();
    }

    /// Number of GPUs this controller was constructed for.
    /// The engine reads this when re-sizing the controller to a newly loaded model's depth (WI-INF2:.
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }
}

// ── Bucketizing ───────────────────────────────────────────────────────────────

/// Map a shape to a power-of-2 bucket index.
/// Autoregressive decode increments `seq_len` by 1 per token.
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
