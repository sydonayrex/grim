# MoE Research Synthesis — 34 papers + Mixture-of-Kittens → grim

**Status:** Complete. All 34 extracted papers read across six themed batches plus the Cursor MoK
megakernel post (`mok.pdf`). Per-batch detail files (all read-only, none touch grim source):

| Batch | Theme | Papers | Detail |
|---|---|---|---|
| A | Routing / router quality | 5 | `/tmp/opencode/moeres1_routing_out.md` |
| C | Overlap / dispatch / multi-GPU | 6 | `/tmp/opencode/moeres1_analysis.md` |
| D | Quantization & compression | 6 | `old/moe-quantization-paper-analysis.md` (this repo) |
| E | Serving, SLO, parallelism | 10 | `/tmp/opencode/moeres1_out/` |
| F | Expert residency / offload | 7 | `/tmp/opencode/moeres1_residency_out.md` |
| — | MoK megakernel (NVL72) | 1 | `mok.pdf` → mapped in `exploding_kittens.md` |

---

## Top 5 implementable ideas for grim, ranked by ROI

### 1. Replicate-hot + quantize-cold expert rebalancing (R&Q) with an LIS diagnostic + super-expert guard
**Where it lands:** `MoESchedule` + `charon` grouped dispatch / `comm_fuse`; per-expert quant layouts.
**Numbers:** up to **1.4× LIS reduction at ≤±0.6% accuracy** (arXiv 2602.19938); degrades gracefully
under streaming input. Guard: **never-quantize/never-skip Super-Experts** — 3 wrongly-cold SEs cost
**PPL 8.70→59.86, Pass@1→0 on AIME** (ICLR'26 Super-Experts).
**Cost:** one calibration pass + a runtime LIS metric per layer + a scheduler policy. Reuses every
existing kernel (q4k/IQ/K-quant/MXFP4). **No training.** → details in batch A ideas #1.

### 2. Concurrency-first admission/tuning (not placement gymnastics)
**Where it lands:** engine gating/batching layer, config/self-tuning.
**Numbers:** concurrency explains **51.7%** of throughput variance vs Strategy **6.6%** / Model **3.8%**
(batch C); linear regime is concurrency **32–64**; naive hybrids collapse to **20–28% of TP**.
**Cost:** queue-depth + continuous-batching target tuning. **No architecture change.** → batch C idea #1.

### 3. Self-assisted speculative decoding for the weight-streaming flag
**Where it lands:** speculative wrapper (draft = pinned hot expert subset) + coalesced expert fetches.
**Numbers:** **4.30× throughput / −76.73% PCIe traffic** on NLLB-MoE, training-free; HotTemporal
zero-cost refresh + affinity-L2 remap (<200 KB table). → batch C idea #2.

### 4. EMA warm-tier expert predictor driving prefetch/eviction (TriMoE-pattern)
**Where it lands:** a per-expert EMA inside `MoESchedule` (`EMA_e = 0.3·F_e + 0.7·EMA_e`, 38 KB,
>78% accuracy, +1.16× e2e, migration <3.3%) — prefetch **exactly one** highest-EMA expert, evict
lowest-EMA resident on a w≈4-token window. Pure scheduler change, no HW/training. Pairs with
SMOE's token-wise cache (~**8 experts = ~3%/layer**) and OSDI SLP prefill (**1,200 tok/s**).
→ batch F ideas #1–2, batch A idea #3.

### 5. Fused gate+up SwiGLU GEMM + K-streamed down-proj (MoK / TritonMoE / exploding_kittens.md)
**Where it lands:** `charon_wmma.rs` / `charon.rs` fused dispatch.
**Numbers:** fused gate+up cuts **35% of global memory traffic** (TritonMoE, 89–131% of Megablocks on
A100 and MI300X zero-code-change); MoK ships **2.37× (FP8) / 1.92× (BF16) forward** on NVL72 and kills
the `[batch, inter]` HBM intermediate entirely; grim's `exploding_kittens.md` already spells the
WMMA / K-streaming / LDS-SiLU version. **This is the plan already in flight.**

---

## Read-across map (which papers inform which grim lever)

| grim lever | Papers | Key number |
|---|---|---|
| Load balance / stragglers | R&Q (2602.19938), GEM, MoRE | 1.4× LIS; gap 11.9→23.4% w/ N; 3–4× wall-clock @ M≥4096 |
| Quant-safety of per-expert layouts | Super-Experts, GEMQ, MC#, DiEP, MoE-APEX, SPECTRA | never-cold SEs; LP bit budgets; ~92% perf at ½ experts |
| Conv-VRAM expert residency | SMOE, OSDI SLP/DSLP, DALI, TriMoE, MELINOE, Harvest | 8.68×/2.98× prefill, +20.9% decode, 1.2K tok/s, 22.77>15.80 tok/s (residency > count) |
| Expert-parallel + disaggregation | OSDI SmallEP, FluxMoE, MoE-Hub, ParallelKittens | 1.22× EP; expert paging (3.0×/3.7× @ 256-batch 4k ctx); destination-agnostic |
| Router cost at large M | MoRE, MEAN | r=64 rank router, fused no-HBM top-k, 6–14% lower PPL |
| Quant traffic & decode bandwidth | TritonMoE, MoEBlaze, Cost-of-Expertise | −35% memory; 4×; decode memory-bound, 2 batch regimes |
| Topk threshold for EP | MegaScale-MoE | top-k>6 ⇒ prefer all-gather EP |
| Comm/compute overlap granularity | ParallelKittens, GC/TMA thesis, Producer/Consumer | copy engines need ≥256 MB; TMA @2 KB; wait–launch pairs; intra-SM RS beats inter 1.2× |
| Fixed-point safety | MEAN, CPU-GPU-SLO | 5.11 fixed-point 0.004% err w/ layernorm; FP8→BF16 immediate expand L1 0.0017 |

---

## Included batch detail (single source of truth per batch)

- **Routing (A):** MoRE low-rank fused router; Super-Experts (SE catastrophic sensitivity); R&Q
  replicate/quantize rebalancing; SMOE consumer-GPU token-wise cache; OSDI CPU–GPU hybrid SLOs.
  Top-3: R&Q+LIS+SE-guard, low-rank r=64 router, token-wise cache+stream-load.
- **Overlap/dispatch/multi-GPU (C):** Layout/Fusion tradeoffs (token-major decode vs in-flight permute
  prefill); TritonMoE fused gate+up; tile-level producer/consumer combine overlap; ParallelKittens
  primitive algebra + transfer-granularity; MoE-Hub destination-agnostic dispatch; CMU persistent
  megakernel (intro-only). Top-3: fused gate+up, producer/consumer segmented combine, primitive algebra
  + lazy destination resolution.
- **Quantization (D):** GEMQ LP bit-allocation; MC# PMQ/OTP; DiEP differentiable pruning; MoE-APEX
  prefetch + precision descent on cache miss; SPECTRA-MoE demand-paged packs (2.8T→18–22 GiB resident);
  Sieve PIM scheduling (EMA cost table beats roofline 1.8–4.2×). Top-3 for `grim-quant`: global LP
  bit allocation, dynamic cost-aware expert loading, bimodal-workload EMA scheduling.
- **Serving/SLO (E):** CPU–GPU hybrid SLO machinery (SLP/DSLP/SmallEP, 1,200/1,800 tok/s); decode
  memory-bound two-regime cost; concurrency-first variance (51.7%); SpecMoE self-assisted spec-decode
  (4.30×); expert streaming QoS-token buffering; MegaScale-MoE (top-k>6 ⇒ EP, 1.41M tok/s); MoEBlaze
  hashed dispatch (4×); MoE-DisCo staged pretraining; GC/TMA wait–launch overlap; MEAN fixed-point
  safety. Top-3: concurrency-first tuning, self-assisted spec-decode, QoS-buffering+slice-stream+flags.
- **Residency (F):** DALI greedy heterogeneous placement (≥92% optimal at ~4.5% cost); TriMoE EMA
  warm-tier prefetch; MELINOE/DALI "predictability beats residency" (22.77 vs 15.80 tok/s);
  LEAST-LOADED EP; Harvest peer-GPU caching; FluxMoE expert paging (paged expert residency, budget-aware
  planner); GEM heterogeneous
  placement. Top-3: greedy device assignment + overlap-window migration, EMA predictor, sequence-aware
  eviction.
- **MoK (NVL72 megakernel):** pull/push dispatch combine (29% higher NVLink util, 5.8× lower signal
  latency), schedule-once 2-column table (no sort, <3% of runtime), minibatch tuning heuristic
  (T ≥ 2C·128·256/min(2I,H)), inter-SM comp/comms partitioning (TMA saturates with <⅓ of SMs), ring
  token buffer (no CPU-GPU sync), reversed ring for backward replay, determinism, MXFP8 + shared-expert
  BF16, router-dgrad fused into SwiGLU backward. Full mapping into `exploding_kittens.md`.

---

## Bottom line

Three moves ship this quarter, in order:

1. **Rebalancing + LIS + SE-guard** (batch A #1) — training-free, sits exactly at the dispatch seam,
   reuses shipping kernels, 1.4× balance at ±0.6% accuracy.
2. **Concurrency-first tuning** (batch C #1) — pure config, where 7.8× the variance lives.
3. **Fused gate+up + K-streamed down-proj** (MoK/TritonMoE, `exploding_kittens.md` WI-A) — already
   planned; −35% memory traffic and no `[batch, inter]` HBM intermediate are the measurable gates.

Longer-horizon: EMA warm-tier prefetch (batch F), self-assisted speculative decode for the
weight-streaming flag (batch C), and global-LP per-expert bit allocation in `grim-quant` (batch D).

All figures are transcribed directly from the papers; artifacts from fused-text PDF conversion are
possible in exact table values, and none of the file-based analysis has been cross-verified against
the original PDFs beyond the extracted text.