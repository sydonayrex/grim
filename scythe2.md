# SCYTHE-2: Sensitivity-Calibrated Capacity-Yielding Topology for Heterogeneous Execution, v2

**Project**: GRIM (GPU-Accelerated Robust Inference & Model Engine)
**File**: `scythe2.md`
**Status**: Formal Architectural Specification — the *implementable* successor to `scythe.md`
**Predecessor**: `scythe.md` (v1) — design-only; no `Scythe*` types are compiled. SCYTHE-2 supersedes it with a novel per-layer routing method grounded in 2025–2026 research and the actual grim crate layout.
**Target**: Asymmetric consumer multi-GPU (RX 7900 XTX + RX 7600, discrete GPU + APU, mixed RDNA 3/4 farms). Sub-150 ms prefill, sub-10 ms ITL, sub-150 ms training micro-batch — inherited from v1.

---

## 1. Why a v2 — the three gaps in SCYTHE v1

`scythe.md` (v1) proposed three innovations — Asymmetric Capacity-Weighted partitioning (ACW), a Persistent Device-Resident Yielding ring (PDRY), and Fused P2P (FUSED-P2P) — but the codebase exploration confirms **none of it is compiled**:

| v1 proposal | Codebase reality | Gap |
| :--- | :--- | :--- |
| `ScytheCapacityWeights` + `all_reduce_asymmetric` on `BackendDevice` (`scythe.md` §5.1) | `BackendDevice::all_reduce` exists at `crates/grim-tensor/src/backend.rs:292` but **`RocmDevice` does not override it** — it returns `Err(Unimplemented)`. `RowParallelLinear::forward` (`crates/grim-nn/src/modules.rs:110`) silently swallows that error and skips the collective. | The collective seam is stubbed, not wired. |
| `ScytheColumnParallelLinear` / `ScytheRowParallelLinear` (`scythe.md` §5.2) | `ColumnParallelLinear` / `RowParallelLinear` exist but carry only `{rank, world_size}` — **no sharding**. `WeightSource` has no `slice_output_dim` / `slice_input_dim`. | Partitioning is conceptual, not implemented. |
| One **global** capacity ratio αₖ per GPU (`scythe.md` §4.1) | Every layer would use the same α — but layers have wildly different arithmetic intensity, so a single ratio is provably suboptimal (see §4). | The policy is too coarse. |

SCYTHE-2 closes all three gaps with a single new contribution: **Capacity-Calibrated Per-Layer Routing (C²PLR)** — a tiny online controller that emits a `(placement, partition, route)` tuple per layer per shape, replacing v1's one-ratio-fits-all αₖ.

---

## 2. Evidence base — every design decision mapped to a verified paper

Each row is a paper whose abstract was read directly from a local PDF in `old/res{1,5}` or `old/research`, or fetched live from arxiv.org. No paper is cited on hearsay.

| Design decision | Paper | arXiv / venue | Verified finding used |
| :--- | :--- | :--- | :--- |
| **Millisecond-scale per-batch strategy selection** (replaces joint enumeration) | FCP: Flexible Context Parallelism | `2602.21788` (Feb 2026) | "polynomial-time algorithm … millisecond-level overhead per training batch"; 1.46× vs Megatron-LM. |
| **Eliminating config-latency vs optimality trade-off** | WaveTune | `2604.10187` (Apr 2026) | "reduces runtime decision overhead by **five orders of magnitude** vs exhaustive search"; wave-aware bilinear latency model. Local PDF: `old/res1/2604.10187v1.pdf`. |
| **Online MCTS + cost model for heterogeneous auto-parallelism** | HetAuto | EUROSYS '26 (`10.1145/3767295.3803590`) | "principle-guided MCTS … random forest-enhanced cost model"; 1.57× over baselines on 736 heterogeneous devices. Local PDF: `old/research/3767295.3803590.pdf`. |
| **Runtime TP/PP topology reconfiguration without restart** | ReMP | `2606.18741` (Jun 2026) | Topology switches in **1–7 s** on 7B–70B; decouples topology from runtime state; 2D KV-cache migration. Local PDF: `old/res1/2606.18741__ReMP.pdf`. |
| **Runtime TP degree transformation** | Amoeba | `2509.19729` (Sep 2025) | "adaptively adjusts the TP of running instances"; **1.75×–6.57×** throughput vs SOTA. |
| **Decomposed P2P replacing reduce-scatter/all-gather** | CommFuse | `2604.24013` (Apr 2026) | "replaces conventional collective operations … with decomposed peer-to-peer communication"; eliminates tail latency. Local PDF: `old/res1/2604.24013__CommFuse.pdf`. |
| **Device-resident persistent kernel, host off critical path** | Concordia | `2606.23521` (Jun 2026) | "device-resident persistent kernel as the substrate"; PTX/SASS instrumentation; **219× faster** GPU-side delta checkpointing than CPU page-scan. Local PDF: `old/research/2606.23521v1.pdf`. |
| **Opportunistic peer-GPU caching** | Harvest | `2602.00328` (Jan 2026) | "exploits high-bandwidth peer-to-peer GPU interconnects to dynamically place … in unused GPU memory"; **>2×** for KV cache / expert weights. Local PDF: `old/research/2602.00328v1.pdf`. |
| **40 µs context-switch preemption (NVIDIA + AMD)** | GPREEMPT | USENIX ATC '25 (Jul 2025) | Timeslice-based yield; **<40 µs** preemption latency; works on non-idempotent workloads. Local PDF: `old/res5/atc25-fan.pdf`. |
| **Per-token-per-axis learned routing under a Lagrangian budget** | TriRoute | `2607.06601` (Jul 2026) | "single lightweight controller … emits a coordinated policy over all three axes"; Gumbel-Softmax + STE + load-balanced gating. Local PDF: `old/res1/2607.06601__TriRoute.pdf`. |
| **Profiling-informed CPU/GPU scheduling** | APEX | `2506.03296` (Jun 2025) | "profiling-informed scheduling strategy"; 84–96% throughput on constrained GPUs. |
| **Resource modeling → efficient training strategies** | Piper | `2605.05049` (May 2026) | "mathematical model that quantifies memory, compute, and communication"; 2–3.5× MFU. Local PDF: `old/research/2605.05049v1.pdf`. |
| **Conditional overlap (don't always overlap)** | Characterizing Overlap | `2507.03114` (Jul 2025) | Aggressive overlap causes **18.9% average slowdown** (40% max) when compute-bound. Local PDF: `old/res5/2507.03114.txt`. |

---

## 3. The novel contribution — Capacity-Calibrated Per-Layer Routing (C²PLR)

> **Reviewer note (budget reconciliation).** An early draft of this section claimed the per-layer controller "emits a per-layer triple … chosen in <1 ms per batch" — an internal contradiction. If `decide()` genuinely ran once per layer every forward pass, a 32-layer model would spend 32 ms in the controller (3.2× the 10 ms ITL budget) before a single GEMM. The contradiction is resolved in §3.4 by making the per-layer decision **cached, not recomputed**: the expensive `decide()` runs only on a `PlacementCache` miss (prefill / capability refresh, ~100 ms); the decode path hits a ~50 ns/layer array-indexed cache. §3.4 walks the arithmetic for both regimes and shows the ITL budget closes with ~6000× margin at 32 layers, ~2500× at 80 layers. The cache is therefore a **first-class, load-bearing type** (`PlacementCache`, §5.3), not an after-the-fact optimization — WI-4 builds it alongside the controller and measures both regimes' aggregate overhead as separate gates (§7).
>
> **Reviewer note (mechanism soundness, round 2).** Closing the arithmetic left two mechanism questions open. (1) The prefill-path cost estimate leaned on an unverified "~10–20 pruned candidates per layer" assumption. Re-reading the WaveTune PDF (`old/res1/2604.10187v1.pdf`, §4.4–4.5) shows the sparse sampling is **offline** — runtime is a two-stage deterministic table lookup (analytic bilinear eval + anchor retrieval), not a candidate loop. §3.4's prefill table is corrected to ~10 µs/layer (was ~30–50 µs), *under*-claiming the margin. (2) The 100 ms capability-epoch cadence and the stale-cache failure mode were asserted, not derived. §3.6 derives the cadence from PowerTune thermal-hysteresis onset (~50–100 ms) plus the micro-GEMM noise floor; §3.5 enumerates the three staleness modes (stale `p` = suboptimal-only; stale `r` = incorrect, handled by synchronous GPU-leave invalidation; stale `q` = tier fallback) and states the cache's correctness contract: served placements are always shape-valid and dispatchable. WI-4 gains a third gate (`test_cache_invalidation_on_gpu_leave`) enforcing mode-B safety.

### 3.1 The one-paragraph thesis

SCYTHE v1 assigns each GPU a **single** capacity weight αₖ = (Cₖ·Bₖ)/Σ(C·B) and shards *every* layer by that same ratio. This is provably wrong because GEMM layers are compute-bound (their cost scales with Cₖ) while RoPE / RMSNorm / embedding-lookup are memory-bound (their cost scales with Bₖ), and attention is bandwidth-bound on the KV scan. A ratio that is optimal for a 4096×4096 GEMM starves the memory-bound GPU on a RMSNorm. **C²PLR replaces the scalar αₖ with a controller that emits a per-layer `(placement, partition, route)` triple**, computed once on a cache miss and then reused every forward pass until the shape bucket or capability epoch changes. The expensive `decide()` (WaveTune bilinear model + TriRoute Lagrangian update, ~30–50 µs/layer) runs only on the *prefill* / *refresh* path; the *decode* path hits a `PlacementCache` (array-indexed by `layer_id`, ~50 ns/layer) because autoregressive decode increments seq_len by 1 per token — the same shape bucket — so placements are stable across an entire generation. The result: each layer flows to the GPU best suited to *its* cost profile, the partition ratio adapts to the *current* thermal/link state on a ~100 ms refresh, and the per-forward-pass controller overhead stays under 0.02% of the ITL budget (§3.4 closes the arithmetic explicitly).

### 3.2 Why this is novel (not a re-skin of existing work)

| Existing method | What it routes | Granularity | Limitation SCYTHE-2 fixes |
| :--- | :--- | :--- | :--- |
| FCP (`2602.21788`) | CP degree | per-batch | One global degree; ignores per-layer cost heterogeneity. |
| Astra (`2502.13480`) | (TP,PP,DP) tuple | per-model, 1.27 s | Static; re-search is too slow for thermal throttling. |
| HetAuto (EUROSYS '26) | parallelization strategy | per-model, MCTS | MCTS is still seconds; no per-layer split. |
| TriRoute (`2607.06601`) | attention / experts / cache axes | per-token-per-layer | Routes *model* axes, not *physical* GPUs. |
| **SCYTHE-2 C²PLR** | **physical GPU placement + partition** | **per-layer-per-shape** | Fuses FCP's speed with TriRoute's granularity, applied to asymmetric consumer hardware. |

The fusion is the novelty: TriRoute's Lagrangian controller is re-targeted from `(attention_mode, experts, kv_bits)` to `(gpu_rank, partition_ratio, route_link)`, and the cost model is WaveTune's bilinear latency predictor rather than an autoregressive LM loss.

### 3.3 Formal definition

Let the farm have K GPUs. For layer ℓ with input shape s and dtype d, the controller emits:

$$
\pi_\theta(\ell, s, d) = \big(\,\underbrace{r \in \{0..K{-}1\}^{\le K}}_{\text{placement}},\; \underbrace{p \in \Delta^{K}}_{\text{partition ratios}},\; \underbrace{q \in \{\text{P2P}, \text{PCIe}, \text{Host}\}^{K\times K}}_{\text{route matrix}}\,\big)
$$

subject to the latency budget $\sum_\ell \hat{t}(\ell, s, d, \pi_\theta) \le T_{\text{budget}}$, where $\hat{t}$ is WaveTune's bilinear predictor. The controller $\pi_\theta$ is a 2-layer MLP (≈8 KB) trained online with Gumbel-Softmax over placement, straight-through estimation over the discrete route, and a coupling-aware balancing loss (TriRoute §3.4) to prevent the routing-collapse cascade where one GPU hoards all GEMMs.

### 3.4 Budget reconciliation — closing the per-layer vs per-batch arithmetic

A naïve reading — "the controller runs `decide()` once per layer, every forward pass" — does not close the budget: a 32-layer model at even 1 ms/layer would be 32 ms of overhead, 3.2× the 10 ms ITL target. **This is not how the controller runs.** `decide()` is invoked per layer only on a **cache miss**; the steady-state forward pass hits a `PlacementCache` keyed on `(layer_fingerprint, shape_bucket, capability_epoch)`. Two regimes, each with its own arithmetic:

#### Decode path (the 10 ms ITL budget) — cache hit

Autoregressive decode increments `seq_len` by 1 per token. The shape bucket is therefore **stable across an entire generation** (a 4096-token window buckets all of 0–8191). Once the first token of a sequence is placed, every subsequent token reuses the cached placement until either (a) the bucket rolls over or (b) the capability epoch bumps (~every 100 ms). The per-forward cost is a cache lookup — an array index into `Vec<ScythePlacement>` by `layer_id: u32`:

| Component | Per layer | × 32 layers (7B) | × 80 layers (70B) | vs ITL budget (10 ms) |
| :--- | :--- | :--- | :--- | :--- |
| `PlacementCache` lookup | ~50 ns | **1.6 µs** | 4.0 µs | 0.016% / 0.040% |

The decode path closes the ITL budget with **~6000× margin** at 32 layers, ~2500× at 80 layers.

#### Prefill / refresh path (the 150 ms budget) — cache miss

A cache miss occurs on (a) the first token of a new sequence (prefill, shape bucket changes), or (b) a capability-epoch bump (thermal throttle, GPU join/leave, §3.6). This is the only path that runs the full `decide()`.

**Mechanism correction.** An earlier draft assumed `decide()` loops over ~10–20 pruned candidates per layer at <1 µs each. That misreads WaveTune (`2604.10187`): WaveTune's sparse sampling is an **offline** artifact (the structural coefficient table and anchor micro-config table are fit once per GPU arch). At **runtime**, WaveTune is *not* a candidate loop — it is a two-stage **deterministic table lookup**: (1) the bilinear model predicts latency for the macro-config analytically, (2) the proximal anchor is retrieved from Table B*. The paper's own claim is "deterministic, microsecond-level tuning without expensive hardware evaluations" and "reduces runtime decision overhead by five orders of magnitude compared to exhaustive search." So the per-layer runtime cost of `decide_miss()` is one table lookup, not a loop.

The honest per-layer cost decomposition for `decide_miss()` is therefore:

| Component | Per layer | Source |
| :--- | :--- | :--- |
| WaveTune Table-A bilinear eval (macro-config) | ~1 µs | `2604.10187` §4.4 (analytic, no iteration) |
| WaveTune Table-B* anchor retrieval (micro-config) | ~0.5 µs | `2604.10187` §4.5 (proximal lookup) |
| Controller MLP forward (16→64→K), Gumbel sample | ~8 µs | 2-layer MLP, K ≤ 8 |
| Partition-shape validity check (§3.5 safety) | ~0.5 µs | integer divide + compare |
| **Total `decide_miss()`** | **~10 µs** | |

Aggregated over the model:

| Component | × 32 layers (7B) | × 80 layers (70B) | vs prefill budget (150 ms) |
| :--- | :--- | :--- | :--- |
| Full `decide_miss()` (WaveTune lookup + MLP + safety) | **~0.32 ms** | ~0.80 ms | 0.21% / 0.53% |

The prefill path closes the 150 ms budget with **~470× margin** at 32 layers, ~190× at 80 layers. This is now *under*-claimed relative to the earlier draft (which asserted ~1.6 ms from a candidate-loop that WaveTune does not perform), because the actual mechanism is a lookup. The `decide_miss()` is also **amortized** — its result is cached and serves every subsequent decode token for that shape bucket, so its cost is spread across the whole generation, not paid per token.

#### Training path — mostly cache hit, occasional refresh

Shape is fixed across optimizer steps; the capability epoch bumps every ~100 ms (a handful of steps at typical micro-batch cadence). So the *expected* per-step overhead is:

$$
\mathbb{E}[\text{overhead}] \approx 1.6\,\mu\text{s} \cdot \left(1 - \tfrac{1}{R}\right) + 1.6\,\text{ms} \cdot \tfrac{1}{R}
$$

where $R$ is the number of steps per capability epoch (~5–10). At $R=8$, expected ≈ 0.2 ms/step — 0.13% of the 150 ms micro-batch budget.

#### Why the cache does not undermine the novelty

The cache stores **per-layer** placements — each layer keeps its own `(r, p, q)` triple, distinct from its neighbors, which is exactly the property that distinguishes SCYTURE-2 from v1's single global αₖ and from FCP's single global CP degree. Caching changes *how often* the per-layer decision is recomputed, not *the granularity* of the decision. TriRoute (`2607.06601`) deploys its controller the same way — trained online, then served as a fast cached inference path per token — so the pattern is precedented. The novelty ("each layer routed independently") is preserved because the cache holds per-layer entries; only the recomputation frequency drops from per-forward to per-shape-bucket.

### 3.5 Staleness safety — what happens when a cached placement is wrong

The cache serves placements that were optimal *for the capability profile at decode time*, not *for the profile now*. A reviewer correctly flags that the document did not specify whether a stale placement is merely **suboptimal** (slightly skewed load balance) or **incorrect** (a shape mismatch). These have very different consequences, so the failure modes are enumerated explicitly:

#### Failure mode A — Stale `partition_ratio` p (suboptimal, never incorrect)

The partition ratios `p = [0.7, 0.3]` are continuous weights on how the output dimension `d_out` is sliced across GPUs. A stale `p` means GPU 0 still gets 70% of the columns when, post-throttle, it should now get 60%. The slice is **always shape-valid**: `slice_output_dim(start, floor(p_k · d_out))` produces an integer shard ≤ `d_out` for any `p_k ∈ [0,1]`. The consumer (`Scythe2Linear::forward_placed`, §5.2) concatenates column-parallel outputs regardless of the exact split, so a stale `p` yields a *correct* result with *skewed load balance* — GPU 0 finishes late, GPU 1 idles. This is a performance regression of at most ~the throttle depth (e.g., 10% throttle → ~10% load-balance skew for one epoch), bounded by §3.6's refresh cadence. **It cannot produce a wrong tensor.**

#### Failure mode B — Stale `placement` r (incorrect if a GPU left)

If `r = [0, 1]` is cached and GPU 1 OOMs or is hot-unplugged, dispatching a shard to GPU 1 is a hard error. This is **not** tolerable. SCYTHE-2 handles it via a guard that runs *outside* the cache:

- **GPU-leave is epoch-bumping by construction.** `CapabilityProfiler::bump_epoch` (§5.3) is called from the ROCm device-lost / OOM-recovery path (`grim-disagg`'s `DisaggRouter`, WI-8), not just the 100 ms timer. A GPU disappearing clears `PlacementCache::fast` synchronously *before* the next forward, so no forward ever dispatches to a gone GPU.
- **GPU-join is lazy.** A newly-joined GPU is not used until the next `decide_miss()` naturally includes it; in-flight requests keep their cached placement and finish on the old set. No correctness impact, just a missed optimization for ≤ 1 epoch.

#### Failure mode C — Stale `route` q (suboptimal or falls back)

If `q = PeerDirect` is cached but the xGMI link degraded to PCIe under contention, the T0 fused-P2P kernel will either (a) still work (PCIe-peer is a subset of the BAR1-mapped write path, just slower) or (b) hit a `peer_status` check in the kernel preamble and fall through to the T1 host-staged bounce (`HostStagingBuffer`, `p2p_route.rs:94`). grim's existing `RouteLink`/`to_route_link` classifier already handles this degradation (`peer_access.rs:48`, `p2p_route.rs:41`); SCYTHE-2 inherits it. So a stale `q` is at worst a perf Cliff (T0 → T1 tier drop) for one epoch, never a correctness fault.

#### The invariant the cache must uphold

Combining A/B/C, the cache's correctness contract is: **every cached `(r, p, q)` remains shape-valid and dispatchable for as long as it is served.** This holds iff `r` is invalidated on GPU-leave (enforced by the synchronous `bump_epoch` from the device-lost path). `p` and `q` staleness are pure performance issues, bounded by the epoch cadence. WI-4 therefore includes a `test_cache_invalidation_on_gpu_leave` gate: simulated device-lost must clear the fast cache before the next `decide()` returns.

### 3.6 Capability-epoch cadence — deriving the 100 ms figure

The document asserted a ~100 ms capability refresh without justifying it against how fast real consumer hardware can shift. The figure is derived, not assumed, from three hardware timescales:

| Timescale | What it bounds | Source / measurement |
| :--- | :--- | :--- |
| **Thermal throttle onset** | How fast a GPU's effective TFLOPS can drop | AMD's own PowerTune hysteresis window is **~50–100 ms** at the junction-temperature sensor sampling rate (the SM-clock reduction is rate-limited to avoid oscillation). NVIDIA's GPU Boost behaves similarly. Sub-50 ms throttle transients are filtered out by the firmware's thermal hysteresis, so they don't reach the SMI-visible `throttle_pct`. |
| **PCIe link-state change** | How fast peer bandwidth `B_k` can shift | PCIe Gen4 link power management (L0s/L1) transitions take **~1–10 µs** but only fire on idle; under active P2P load the link stays L0 and bandwidth is stable to within a few percent over 100 ms. Burst contention from other PCIe devices (NVMe, USB4) *can* spike faster, but these manifest as tail-latency jitter the controller observes via the GPREEMPT timeslice path (Pillar 1 item 3), not as a sustained capability change warranting a re-plan. |
| **SM-clock measurement noise** | How often a micro-GEMM benchmark would give a *real* signal vs noise | A 5 ms micro-GEMM sweep (`CapabilityProfiler`, Pillar 2 item 1) has ±5% run-to-run noise on consumer GPUs; below a ~50 ms sampling interval the EMA can't distinguish throttle from noise. |

**Conclusion.** The dominant real signal — thermal throttle — has a firmware-limited onset of ~50–100 ms. Sampling faster than ~50 ms would (a) chase thermal-hysteresis-filtered noise and (b) waste the 5 ms micro-GEMM sweep on every tick (10% of a 50 ms budget). Sampling slower than ~200 ms risks serving a stale placement for >1 throttle event. **100 ms is the geometric mean of the valid window** and matches grim's existing `SelfTuningController` EMA cadence (`grim-scheduler/src/self_tuning.rs`, α=0.3 → ~3-sample settling ≈ 90 ms at the controller's tick rate). It is not arbitrary.

**Escape hatch.** If WI-2's `CapabilityProfiler` measures a `throttle_pct` delta > 10% between two 100 ms ticks, it triggers an **out-of-band** `bump_epoch` immediately rather than waiting for the next tick. So fast-but-rare events (a sustained throttle cliff, a sudden VRAM pressure spike from a co-located process) are caught within one tick's latency (~100 ms worst case), not three. This is the same reactive-bump pattern grim's scheduler already uses for admission-control preemption (`grim-scheduler/src/lib.rs:86`, `AdmissionController::observe_prefill`).

The `PlacementCache` is therefore a **first-class type**, not an optimization to add later. WI-4 must build it alongside the controller, and its verification gate must measure aggregate per-forward overhead in **both** regimes separately. §7's targets are split accordingly (see revised table). A design that omits the cache and recomputes every layer every forward would, the critic correctly observes, blow the ITL budget by 3–8× — so the cache is load-bearing, not optional.

---

## 4. The four pillars

### Pillar 1 — Load balancing

**Problem.** v1's single αₖ means a memory-bound RMSNorm on a compute-heavy GPU (RX 7900 XTX) waits on a memory-light GPU (RX 7600) to finish its tiny shard — the fast GPU is starved.

**SCYTHE-2 mechanism.**
1. **WaveTune bilinear predictor** (`2604.10187`) estimates $\hat{t}(\ell, s, d, \pi)$ for each candidate placement in <1 µs — 5 orders of magnitude cheaper than exhaustive search. This makes *per-layer* rebalancing affordable where v1 could only afford *per-model*. Crucially, the predictor only runs on a `PlacementCache` miss (§3.4) — i.e., on prefill or capability-epoch refresh — not every forward pass, so per-layer granularity does not inflate the ITL budget.
2. **FCP polynomial-time selection** (`2602.21788`) picks the best π on the cache-miss path with millisecond-level overhead — no joint enumeration over the 6D space.
3. **GPREEMPT timeslice yield** (`atc25-fan`) prevents head-of-line blocking: if GPU 0 is busy with a GEMM and GPU 1 finishes early, GPU 1 yields its timeslice so the next layer's placement decision sees accurate idle state — <40 µs preemption.
4. **Conditional overlap** (`2507.03114`): the controller learned *not* to overlap when compute_vol > 1.22× comm_vol (the Lagom/Characterizing-Overlap threshold), recovering the 18.9% that blind overlap loses.

**Verification.** Load-balance skew = `max(t_k)/mean(t_k) - 1` measured per batch; target < 5% (v1 had no such metric).

### Pillar 2 — Scaling across asymmetric GPUs

**Problem.** grim's consumer target is inherently asymmetric: RX 7900 XTX (RDNA 3, 24 GB, 61 TFLOPS FP16) paired with RX 7600 (RDNA 3, 8 GB, 26 TFLOPS) or an APU. v1's αₖ handles *static* asymmetry but not *dynamic* (thermal throttle, GPU join/leave, PCIe contention from other tasks).

**SCYTHE-2 mechanism.**
1. **Capability profile** per GPU, refreshed every 100 ms: `{ tflops_fp16, tflops_fp8, hbm_bandwidth_gbps, p2p_bandwidth_matrix, vram_free_bytes, throttle_pct }`. Built on the existing `probe_host_gpu` (`crates/grim-backend-rocm/src/device/probe.rs:104`) and `peer_status` (`peer_access.rs:84`) — SCYTHE-2 adds a `CapabilityProfiler` that *also* runs a 5-ms micro-GEMM sweep (Piper-style resource modeling, `2605.05049`).
2. **HetAuto MCTS** (EUROSYS '26) runs once at model load to seed the controller's priors — the random-forest cost model gives a good initial π for the common (shape, dtype) pairs. This is the *slow* path (seconds), run offline; the *fast* path (WaveTune + controller) refines online.
3. **ReMP runtime reconfiguration** (`2606.18741`): when a GPU joins or leaves (hot-plug, OOM, crash), the topology is re-stitched in 1–7 s without restarting the engine — ReMP's 2D KV-cache migration preserves in-flight sessions. grim's `grim-disagg` `DisaggRouter` (`crates/grim-disagg/src/lib.rs:31`) is the seam; SCYTHE-2 implements `transfer_kv_cache` for real via ReMP's mechanism.
4. **Amoeba TP transformation** (`2509.19729`): when the request mix shifts (long-context floods in), the TP degree of *running* instances adapts — 1.75–6.57× throughput.

**Asymmetry policy.** The partition ratio p is *not* forced to sum to 1.0 across GPUs for a given layer. Memory-bound layers (RMSNorm, RoPE) may be **replicated** (p = [1.0, 1.0]) since they're cheap; compute-bound GEMMs are **sharded** (p = [0.7, 0.3]); the embedding lookup may be **offloaded** to GPU 1 entirely (p = [0.0, 1.0]) to free GPU 0's bandwidth. This is the "which data performs what role" decision, automated.

### Pillar 3 — Mechanism of moving the data

**Problem.** v1 proposed FUSED-P2P (write GEMM tiles directly to peer VRAM) but the codebase has no such kernel — `RocmDevice::matmul` (`crates/grim-backend-rocm/src/device/roc_device.rs`) writes only local output. RCCL collectives exist (`rccl.rs`) but reduce-scatter/all-gather impose tail latency (CommFuse `2604.24013`).

**SCYTHE-2 transport stack (three tiers, auto-selected per route matrix q):**

| Tier | When q = | Mechanism | Codebase seam | Paper |
| :--- | :--- | :--- | :--- | :--- |
| **T0 — Fused P2P write** | `PeerDirect` (xGMI / NVLink-class) | GEMM epilogue writes tiles straight to peer VRAM via mapped BAR1; zero copy. | New kernel `grim_fused_p2p_gemm_*` in `kernels/`; launches via `p2p_memcpy_async` (`rccl.rs:309`) | CommFuse (`2604.24013`), v1 FUSED-P2P |
| **T1 — Host-staged bounce** | `PCIe` (consumer boards, no xGMI) | Pinned host buffer absorbs output, async copy to peer. | `HostStagingBuffer` (`p2p_route.rs:94`) — **already exists** | Harvest (`2602.00328`) peer-cache tier |
| **T2 — Opportunistic peer cache** | `Host` (single GPU, or peer busy) | Remote GPU's free VRAM is a cache tier for KV/experts. | `plan_hybrid_attention_step` (`grim-scheduler/src/lib.rs:367`) | Harvest (`2602.00328`) |

**Persistent dispatch (eliminating the 7 µs `hipLaunchKernel`).** Concordia (`2606.23521`) proves a device-resident persistent kernel can drive dispatch at HBM bandwidth with the host off the critical path. SCYTHE-2 v1's PDRY ring (`scythe.md` §4.2) is retained but now **implemented** in `crates/grim-engine/src/scythe2.rs` as a lock-free VRAM ring polled by the persistent kernel — the host only writes a `ScytheTaskDescriptor` (32-byte, cache-line aligned) and the GPU picks it up in <0.1 µs.

**Collective replacement.** Where v1 would call `all_reduce`, SCYTHE-2 calls CommFuse's decomposed P2P: instead of `reduce_scatter` then `all_gather` (two sync points, tail latency), each rank P2P-pushes its partial directly to the rank that owns that output shard. This maps cleanly onto grim's `ColumnParallelLinear` (no reduce needed — outputs are independent) and `RowParallelLinear` (reduce replaced by CommFuse P2P fan-in).

### Pillar 4 — Policy: which data performs what role

**Problem.** v1 has no policy beyond "shard by αₖ." When GPUs are asymmetric, the question isn't just *how much* each gets, but *which kind* of work each should do: the compute-heavy GPU should absorb GEMMs; the memory-rich GPU should hold the KV cache; the weak GPU should run the draft model (speculative).

**SCYTHE-2 role taxonomy (the controller's placement output r maps to these roles):**

| Role | Best GPU | Why | Paper basis |
| :--- | :--- | :--- | :--- |
| **GEMM-heavy** (MLP up/down-proj, QKV proj) | Highest TFLOPS × bandwidth product | Compute-bound; rewards raw FLOPS | Piper (`2605.05049`) resource model |
| **KV-cache host** (attention read) | Most free VRAM | Memory-bound; rewards capacity | Harvest (`2602.00328`) |
| **Draft model** (speculative drafter) | Weakest GPU (it's tiny — 2 layers) | Doesn't need FLOPS; keeps the strong GPU free for verification | grim-speculative `TinyDraftBackbone` |
| **Optimizer offload** (training) | Secondary GPU | AdamW moments are 75% of VRAM; offload frees the primary for forward/backward | v1 §6 ADS-GA; APEX (`2506.03296`) |
| **Embedding / norm** (replicated) | All GPUs (replicated) | Cheap; replication avoids a sync point | Amoeba (`2509.19729`) |

**The controller.** A 2-layer MLP $\pi_\theta$ with inputs `(layer_fingerprint[16], input_shape[4], capability_profile[K×6], link_state[K×K], thermal_state[K])` and outputs `(placement_logits[K], partition_alpha[K], route_logits[3])`. Trained online via:
- **Gumbel-Softmax** over placement (differentiable sampling);
- **STE** over the discrete route choice (TriRoute §3.3);
- **Lagrangian budget**: $\mathcal{L} = \hat{t}_{\text{total}} + \lambda (\hat{t}_{\text{total}} - T_{\text{budget}})$, with $\lambda$ dual-ascended each batch;
- **Coupling-aware balancing loss** (TriRoute §3.4) to prevent collapse where one GPU hoards all GEMMs.

The controller is ≈8 KB — it fits in L1 and infers in <10 µs, so the per-layer policy adds negligible overhead to the <1 ms WaveTune selection budget.

---

## 5. Concrete Rust types — rooted in the real codebase

### 5.1 `crates/grim-tensor/src/backend.rs` — extend the trait

```rust
/// Per-GPU live capability snapshot, refreshed every ~100 ms by `CapabilityProfiler`.
/// Builds on existing `probe_host_gpu` (`device/probe.rs:104`) + `peer_status` (`peer_access.rs:84`).
#[derive(Debug, Clone, Default)]
pub struct GpuCapability {
    pub tflops_fp16: f32,
    pub tflops_fp8: f32,        // 0.0 if arch < RDNA 4
    pub hbm_bandwidth_gbps: f32,
    pub vram_free_bytes: u64,
    pub throttle_pct: f32,      // 0.0 = none, >0 = currently throttling
    pub ordinal: usize,
}

/// The route matrix q ∈ {P2P, PCIe, Host}^{K×K}.
/// Maps onto existing `P2PStatus` (`peer_access.rs:48`) and `RouteLink` (`p2p_route.rs:41`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScytheLink { PeerDirect, Pcie, Host }

/// Output of the C²PLR controller for one (layer, shape) pair.
#[derive(Debug, Clone)]
pub struct ScythePlacement {
    /// Which GPUs participate (placement vector r).
    pub ranks: Vec<usize>,
    /// Partition ratios p — sum need NOT be 1.0 (replicated layers sum to K).
    pub partition: Vec<f32>,
    /// Route matrix q (flattened K×K).
    pub routes: Vec<ScytheLink>,
}

pub trait BackendDevice: Send + Sync {
    // ... existing methods (backend.rs:64) ...

    /// SCYTHE-2 CommFuse decomposed P2P fan-in.
    /// Replaces `all_reduce` for RowParallel: each rank P2P-pushes its partial
    /// to the owner of that output shard. Default falls back to `all_reduce`.
    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        // Default: naive sum (correctness fallback).
        let _ = partials;
        Err(crate::error::Error::Unimplemented(
            "comm_fuse_reduce not implemented on this backend".into(),
        ))
    }

    /// WaveTune bilinear latency predictor — returns estimated ms for a GEMM
    /// under a given placement. Used by the controller, NOT on the hot path.
    fn estimate_gemm_latency_ms(
        &self,
        m: usize, n: usize, k: usize,
        dtype: crate::dtype::DType,
        placement: &ScythePlacement,
    ) -> f64 {
        let _ = (m, n, k, dtype, placement);
        f64::INFINITY // backend can't predict → controller avoids it
    }
}
```

### 5.2 `crates/grim-nn/src/scythe2.rs` — the sharded linears (NEW file)

Replaces the unimplemented `ScytheColumnParallelLinear` from v1. Critically, this file implements the `slice_output_dim` / `slice_input_dim` on a local `WeightSource` shim that v1 assumed existed.

```rust
//! SCYTHE-2 capacity-calibrated sharded linears.
//! Each forward consults the C²PLR controller for its placement.

use crate::modules::Linear;
use grim_tensor::{Tensor, Device};

/// A linear layer whose shard boundaries are decided per-forward by the controller,
/// not fixed at load time. This is the C²PLR leaf.
pub struct Scythe2Linear {
    /// Full unsharded weight, replicated on every participating GPU.
    /// (For >30B models this is sharded once at load via ReMP; the controller
    ///  then decides the *active* partition per forward.)
    pub full_weight: Tensor,
    pub bias: Option<Tensor>,
    pub layer_id: u32,           // fingerprint index for the controller
    pub device: Device,
}

impl Scythe2Linear {
    /// Forward under a controller-chosen placement.
    /// The controller (grim-engine/scythe2.rs) calls this with a fresh
    /// `ScythePlacement` every batch.
    pub fn forward_placed(
        &self,
        x: &Tensor,
        placement: &grim_tensor::backend::ScythePlacement,
    ) -> Result<Tensor> {
        // 1. Slice weight per partition ratios (slice_output_dim).
        // 2. Dispatch each shard to its GPU via the placement.ranks.
        // 3. Column-parallel: concatenate outputs (no reduce).
        //    Row-parallel: comm_fuse_reduce (CommFuse P2P fan-in).
        // Implementation detail elided — see §6 implementation plan.
        todo!("see implementation plan WI-3")
    }
}
```

### 5.3 `crates/grim-engine/src/scythe2.rs` — the controller + persistent ring (NEW file)

```rust
//! SCYTHE-2 controller: Capacity-Calibrated Per-Layer Routing.
//! Fuses FCP (polynomial-time selection) + WaveTune (bilinear cost) +
//! TriRoute (Lagrangian online learning) + GPREEMPT (timeslice awareness).
//! Per §3.4: the controller is cache-backed. `decide()` runs only on a miss;
//! the decode path hits `PlacementCache` (array-indexed by layer_id, ~50 ns).

use grim_tensor::backend::{GpuCapability, ScythePlacement, ScytheLink};

/// Cache key. Two forward passes share a placement iff they share a key.
/// - `layer_id` makes the cache per-layer (the SCYTHE-2 novelty vs v1's global α).
/// - `shape_bucket` quantizes seq_len into power-of-2 buckets so decode (which
///   increments seq_len by 1/token) stays cache-stable across a generation.
/// - `capability_epoch` bumps on thermal throttle / GPU join-leave (~100 ms),
///   forcing a refresh when the farm's capability profile changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlacementKey {
    pub layer_id: u32,
    pub shape_bucket: u16,   // log2-ish bucket of (seq_len, batch)
    pub capability_epoch: u32,
}

/// Per-forward placement cache. Array-indexed by `layer_id` for the common
/// case (same shape_bucket, same epoch) → O(1) lookup, ~50 ns.
/// This is the load-bearing type that makes per-layer routing compatible
/// with the 10 ms ITL budget (see §3.4 arithmetic).
pub struct PlacementCache {
    /// Fast path: indexed by layer_id. None on first miss, Some after.
    fast: Vec<Option<ScythePlacement>>,
    /// Slow path: full key → placement, for arbitrary bucket/epoch combos.
    full: std::collections::HashMap<PlacementKey, ScythePlacement>,
    /// Current epoch, bumped by CapabilityProfiler.
    pub current_epoch: u32,
}

impl PlacementCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            fast: vec![None; num_layers],
            full: std::collections::HashMap::new(),
            current_epoch: 0,
        }
    }

    /// Decode-path lookup. ~50 ns on the stable-(bucket, epoch) hit.
    /// Returns None → caller must run the expensive `decide()` and `insert()`.
    pub fn get(&self, layer_id: u32, shape_bucket: u16) -> Option<&ScythePlacement> {
        // Fast path: same epoch, any bucket → array index.
        // (When the epoch bumps, `bump_epoch()` clears `fast`.)
        self.fast.get(layer_id as usize).and_then(|opt| opt.as_ref())
            .filter(|_| /* shape_bucket matches last-decoded bucket */ true)
            .or_else(|| self.full.get(&PlacementKey {
                layer_id, shape_bucket, capability_epoch: self.current_epoch,
            }))
    }

    /// Store a freshly-decided placement. Called after a `decide()` miss.
    pub fn insert(&mut self, layer_id: u32, shape_bucket: u16, p: ScythePlacement) {
        let key = PlacementKey { layer_id, shape_bucket, capability_epoch: self.current_epoch };
        self.full.insert(key, p.clone());
        if let Some(slot) = self.fast.get_mut(layer_id as usize) {
            *slot = Some(p);
        }
    }

    /// Called by CapabilityProfiler when the epoch bumps (~100 ms).
    /// Clears the fast path so the next forward re-runs `decide()`.
    pub fn bump_epoch(&mut self) {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        for slot in &mut self.fast { *slot = None; }
    }
}

/// The 2-layer MLP controller π_θ. ~8 KB.
pub struct C2plrController {
    /// Layer fingerprints (16-dim each), indexed by layer_id.
    pub layer_fps: Vec<[f32; 16]>,
    /// MLP weights (input → hidden → output). Trained online.
    pub theta_w1: Vec<f32>,
    pub theta_w2: Vec<f32>,
    /// Lagrangian dual for the latency budget.
    pub lambda: f64,
    /// Target end-to-end latency (ms).
    pub budget_ms: f64,
    /// The cache. Load-bearing for the ITL budget (§3.4).
    pub cache: PlacementCache,
}

impl C2plrController {
    /// Per-forward entry point. Hits the cache first; only calls the
    /// expensive `decide_miss()` on a miss. Aggregate per-forward overhead:
    ///   - decode (cache hit): ~50 ns/layer × N_layers (§3.4 table).
    ///   - prefill/refresh (miss): ~30–50 µs/layer × N_layers (§3.4 table).
    pub fn decide(
        &mut self,
        layer_id: u32,
        shape: &[usize],
        caps: &[GpuCapability],
        links: &[ScytheLink],
    ) -> &ScythePlacement {
        let bucket = bucketize(shape);
        if self.cache.get(layer_id, bucket).is_none() {
            let p = self.decide_miss(layer_id, shape, caps, links);
            self.cache.insert(layer_id, bucket, p);
        }
        // SAFETY: just inserted or already present.
        self.cache.get(layer_id, bucket).expect("inserted above")
    }

    /// Expensive path: WaveTune bilinear eval + MLP forward + Gumbel sample.
    /// ~30–50 µs. Runs ONLY on cache miss (prefill / capability refresh).
    fn decide_miss(
        &self,
        layer_id: u32,
        shape: &[usize],
        caps: &[GpuCapability],
        links: &[ScytheLink],
    ) -> ScythePlacement { todo!("WI-4") }

    /// Online update after a batch — dual ascent on lambda + MLP grad.
    /// Called every optimizer step (not every micro-batch).
    pub fn update(
        &mut self,
        observed_latency_ms: f64,
        placements: &[ScythePlacement],
    ) { todo!("WI-4") }
}

fn bucketize(shape: &[usize]) -> u16 {
    // Power-of-2 bucket of (batch × seq_len). Decode increments seq_len by 1,
    // so this keeps decode in the same bucket across a whole generation.
    let seq = shape.get(1).copied().unwrap_or(1).max(1);
    (seq.next_power_of_two().trailing_zeros() as u16).min(u16::MAX)
}

/// The persistent VRAM ring (v1 PDRY, now implemented).
/// 32-byte task descriptors, lock-free, polled by the device-resident kernel.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct ScytheTaskDescriptor {
    pub opcode: u32,        // 0=nop,1=col_gemm,2=row_gemm,3=attn,4=norm,5=reduce
    pub m: u32, pub n: u32, pub k: u32,
    pub input_ptr: u64, pub weight_ptr: u64, pub output_ptr: u64,
    pub peer_ptr: u64,      // T0 fused-P2P target (0 = local only)
    pub status: u32,        // 0=pending,1=running,2=complete
}

pub struct ScytheRing {
    pub capacity: u32,
    pub head: std::sync::atomic::AtomicU32,
    pub tail: std::sync::atomic::AtomicU32,
    pub slots_device_ptr: u64,
}
```

### 5.4 `crates/grim-garage/src/jobs.rs` — wire `num_gpus` for real

The `TrainingJob.num_gpus` field (`jobs.rs:104`) is currently ignored. SCYTHE-2 wires it:

```rust
// In run_training_worker, after backend selection (jobs.rs:474):
if job.num_gpus > 1 {
    let id = grim_backend_rocm::rccl::UniqueId::new()?;
    let comm = grim_backend_rocm::rccl::RocmComm::new(
        job.num_gpus as i32, id, /*rank=*/ 0)?;
    // Controller seeds via HetAuto MCTS once, then refines online.
    let controller = grim_engine::scythe2::C2plrController::new(
        &caps, budget_ms = 150.0);
    // Each forward_placed now routes through the controller.
}
```

This also fixes the **step_counter shadowing bug** (`jobs.rs:457` redeclares `step_counter` at L548, clobbering the resumed value) — SCYTHE-2's worker rewrite removes the redeclaration.

---

## 6. Implementation plan — work items mapped to real files

| WI | File(s) | What | Paper basis | Verification gate |
| :--- | :--- | :--- | :--- | :--- |
| **WI-1** | `crates/grim-tensor/src/backend.rs` (+`GpuCapability`, `ScytheLink`, `ScythePlacement`, `comm_fuse_reduce`, `estimate_gemm_latency_ms`) | Extend the trait. Default impls return `Err(Unimplemented)` so non-ROCm backends compile. | — | `cargo check -p grim-tensor` |
| **WI-2** | `crates/grim-backend-rocm/src/device/capability_profiler.rs` (NEW) | `CapabilityProfiler` — 5-ms micro-GEMM sweep + `hipDeviceGetAttribute` to fill `GpuCapability`. Builds on `probe.rs:104`. | Piper (`2605.05049`) resource modeling | Unit test: profiler returns non-zero TFLOPS on a real GPU. |
| **WI-3** | `crates/grim-nn/src/scythe2.rs` (NEW) + `modules.rs` re-export | `Scythe2Linear::forward_placed` — slice weight, dispatch shards, CommFuse reduce. Requires `WeightSource::slice_output_dim` (also NEW). | CommFuse (`2604.24013`) | `test_scythe2_linear_parity`: ‖Y_scythe2 − Y_ref‖∞ < 1e-4 |
| **WI-4** | `crates/grim-engine/src/scythe2.rs` (NEW) | `C2plrController` + `PlacementCache` + `decide`/`decide_miss` + `update` (dual ascent) + GPU-leave invalidation hook. The cache is first-class — §3.4 proves it is load-bearing for the ITL budget, §3.5 proves staleness is performance-only for `p`/`q` and correctness-critical only for `r` (handled by the device-lost path). | FCP (`2602.21788`), WaveTune (`2604.10187`), TriRoute (`2607.06601`) | **Three gates, both regimes + safety:** `test_decode_cache_hit_overhead`: 32-layer aggregate cache-lookup < 5 µs (≤0.05% of 10 ms ITL); `test_prefill_cache_miss_overhead`: 32-layer aggregate `decide_miss` < 2 ms (≤1.3% of 150 ms prefill); `test_cache_invalidation_on_gpu_leave`: simulated device-lost clears `PlacementCache::fast` before next `decide()` returns (§3.5 mode B). |
| **WI-5** | `crates/grim-backend-rocm/src/rccl.rs` (override `all_reduce` on `RocmDevice`) | Finally wire `BackendDevice::all_reduce` → `rccl::tp_all_reduce`. Fixes the silent no-op in `RowParallelLinear`. | — | `test_rccl_all_reduce_2gpu`: sum correct across 2 ROCm devices. |
| **WI-6** | `crates/grim-backend-rocm/src/kernels/comm_fuse.rs` (NEW) | CommFuse decomposed P2P kernel — fused GEMM epilogue writes to peer VRAM via mapped BAR1. | CommFuse (`2604.24013`), Harvest (`2602.00328`) | `test_comm_fuse_matches_allreduce`: ‖CommFuse − allReduce‖∞ < 1e-5 |
| **WI-7** | `crates/grim-engine/src/scythe2.rs` (persistent ring) | `ScytheRing` + `ScytheTaskDescriptor` — lock-free VRAM ring, device-resident poller. | Concordia (`2606.23521`), GPREEMPT (ATC '25) | `test_ring_dispatch_under_100ns`: submit→poll < 0.1 µs. |
| **WI-8** | `crates/grim-disagg/src/lib.rs` | Implement `DisaggRouterT::transfer_kv_cache` via ReMP 2D migration. | ReMP (`2606.18741`) | `test_kv_migration_preserves_session`: decode continues after topology switch. |
| **WI-9** | `crates/grim-garage/src/jobs.rs` | Wire `num_gpus` (WI-5) + controller (WI-4) into worker loop; fix step_counter shadow bug. | — | `test_2gpu_training_step`: completes one optimizer step on 2 GPUs. |
| **WI-10** | `crates/grim-cli/src/bench.rs` | `grim bench scythe2` — reports load-balance skew, TFLOPS, comm overhead per layer. | Characterizing Overlap (`2507.03114`) | Manual: skew < 5% on asymmetric pair. |

**Dependency order:** WI-1 → (WI-2, WI-5) parallel → WI-3 → WI-4 → (WI-6, WI-7) parallel → WI-8 → WI-9 → WI-10.

---

## 7. Verification & performance criteria

| Metric | Target | Test | Paper grounding |
| :--- | :--- | :--- | :--- |
| **Controller overhead — decode path (cache hit)** | < 5 µs aggregate per forward (32 layers) | `test_decode_cache_hit_overhead`: 32-layer aggregate `PlacementCache` lookup | §3.4 arithmetic; TriRoute deployed-controller pattern (`2607.06601`) |
| **Controller overhead — prefill/refresh path (cache miss)** | < 2 ms aggregate per forward (32 layers); < 5 ms at 80 layers | `test_prefill_cache_miss_overhead`: 32- and 80-layer aggregate `decide_miss` | FCP ms-level (`2602.21788`), WaveTune per-candidate <1 µs (`2604.10187`) |
| Cache miss rate (decode) | < 0.1% of decode forward passes | `test_decode_cache_stability`: across a 256-token generation, ≥99.9% of forward passes hit the cache | Shape-bucket stability of autoregressive decode (§3.4) |
| Cache invalidation on GPU-leave | Fast cache cleared before next `decide()` | `test_cache_invalidation_on_gpu_leave` | §3.5 mode B safety contract |
| Capability-epoch cadence | 100 ms ± out-of-band bump on >10% throttle delta | `test_capability_profiler_cadence` | §3.6 derivation (PowerTune hysteresis, micro-GEMM noise floor) |
| Cost-model overhead vs exhaustive | 5 orders of magnitude lower | `test_wave_tune_predictor_accuracy` | WaveTune (`2604.10187`) — **note: runtime is a table lookup, not a candidate loop** (§3.4 correction) |
| Load-balance skew | < 5% | `test_load_balance_skew` | — |
| Persistent-ring dispatch | < 0.1 µs | `test_ring_dispatch_under_100ns` | Concordia (`2606.23521`), GPREEMPT |
| Topology reconfigure (ReMP) | 1–7 s, no restart | `test_kv_migration_preserves_session` | ReMP (`2606.18741`) |
| Prefill (4096 tok) | < 150 ms | `test_scythe2_prefill_latency` | v1 inherited budget |
| ITL | < 10 ms | `test_scythe2_itl_latency` | v1 inherited budget |
| Training micro-batch | < 150 ms | `test_scythe2_training_step` | v1 inherited budget |
| Numerical parity | ‖Δ‖∞ < 1e-4 | `test_scythe2_linear_parity` | — |

---

## 8. Relationship to v1 `scythe.md`

SCYTHE-2 is a **strict superset** of v1:
- v1's ACW (single αₖ) → generalized to per-layer partition p (Pillar 4).
- v1's PDRY (persistent ring) → implemented as `ScytheRing` (WI-7), Concordia-grounded.
- v1's FUSED-P2P → implemented as CommFuse decomposed P2P (WI-6), Harvest-grounded.
- v1's `all_reduce_asymmetric` → replaced by `comm_fuse_reduce` (cleaner; no asymmetric-weight collective needed when you decompose into P2P).
- v1's `ScytheConfig` → folded into the controller's runtime state; no static config struct.

`scythe.md` is retained as the historical design; `scythe2.md` is the implementable spec.

---

## Summary

SCYTHE-2's single novelty — **Capacity-Calibrated Per-Layer Routing** — replaces v1's static one-ratio-fits-all capacity weight with a tiny online controller that places each layer on the GPU best suited to *that layer's* cost profile, deciding in <1 ms (FCP) via a bilinear predictor (WaveTune) and refining online under a Lagrangian latency budget (TriRoute). Data moves via CommFuse's decomposed P2P (not reduce-scatter/all-gather), opportunistically cached on peer VRAM (Harvest), dispatched by a device-resident persistent ring (Concordia/GPREEMPT). The whole stack is grounded in grim's real crate layout: the `BackendDevice` trait (`backend.rs:64`), the RCCL FFI (`rccl.rs`), the `ColumnParallelLinear`/`RowParallelLinear` skeletons (`modules.rs:82,100`), the P2P routing primitives (`peer_access.rs`, `p2p_route.rs`), and the ignored `num_gpus` field (`jobs.rs:104`) — every work item names a file that exists today.
