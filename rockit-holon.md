# rockit-holon.md — grim-backend-rocm + old/tuna1/tuna2 holistic synthesis

Corpus: all PDFs + HTML in `old/tuna1/` (25 PDFs + 1 HTML) and `old/tuna2/` (12 PDFs),
plus the grim-backend-rocm kernel surface reviewed from source.
Two passes: (1) individual lever inventory; (2) gestalt holistic composition.
This is an audit + systems-synthesis document, not a patch.

---

## Pass 1 — individual lever inventory (what each paper offers, what grim has)

This pass is the "per-paper vs per-kernel" view. It is necessary but not sufficient:
the holistic value is in Pass 2.

### 1.1 Auto-tuning / search strategy papers

**CharTuner — CharTuner_Characteristic_Analysis_and_Design_Space_Reduction_for_Efficient_Tensor_Program_Tuning_on_ROCm_Platforms.pdf**
(2025 IEEE ISPA; Wu, Xu, Chen, Wang, Li, Cui; MI210, ROCm 6.0)

- Core method: decompose the full CK GEMM parameter space (Table 1: BS, M/N/KPB, A/BK1,
  A/BAOR, A/BVDM, M/NXDL, M/NXPW, NKPS, CM/NWV, A/B/CTCL, etc.) into 8 semantic
  subspaces (Ω1..Ω8); benchmark each subspace across M=N=K∈[8..256]; PCA on the five-number
  summary (max/Q3/median/Q1/min) of per-shape improvement → weighted importance score; retain
  top-k (Ω6 + Ω4, 55.2% space reduction); then run 8 optimizer algorithms (GA, SA, BYS, DT,
  GBRT, PSO, SOA, RS) inside the reduced space.
- Results: avg 1.98× over NAIVE_TUNER, 3.73× over vendor CK default; PSO best (3.21–4.14×
  for large M=N=K); RS in reduced space still 1.64× vs NAIVE_TUNER's 0.51×; ResNet-50 3.14×
  avg; BERT layers 3.59–4.44×; Llama-3.3-70B layers 3.71–4.46×. Convergence: PSO in reduced
  space converges in ~130/51 iterations vs 282/118 in full space (53.9%, 56.8% reduction).
- Grim relevance: grim's `mxfp4_gemm.rs` + `wmma_gemm.rs` + quant GEMM paths are all
  templated-in-spirit (HIP source with tile/loop params that could be surfaced as CK-style
  subspaces). CharTuner's decomposition + PCA-ranking + top-k retention is a concrete method
  for pruning grim's own candidate space before any measured search. The 8-optimizer suite is a
  ready-made search-strategy zoo to evaluate against grim's current empirical compile+time search
  (this connects to the MLSys optimizer-design paper's "tune the optimizer" idea, and to Kernel
  Tuner's HIP support).

**MLSys 2026 Automated Algorithm Design for Auto-Tuning Optimizers** — LLM generates the search
strategy (LLaMEA + Kernel Tuner, HIP-supported); 72.4% improvement over SOTA human optimizers.
Grim has: autotune.rs empirical search. Gap: no eval of alternative search strategies on grim's
own measured runtime; no meta-optimizer design.

### 1.2 Latency prediction / model-based autotuning papers

**WaveTune — 2604.10187v1** — wave-conditioned piecewise bilinear latency model; precomputed
coeffs + dual-table retrieval; up to 1.83× kernel, 1.33× TTFT; 3 kernels, 5 GPUs, 2 vendors.
Grim has: wave_size, num_waves computed at runtime in qkv_attention.rs:78; arch-gated WMMA/MFMA.
Gap: no predictor/retrieval table; the wave-count term ceil(G/N_SM) is computable from existing
launch geometry.

**TTX — ISPASS 2026** — XGBoost over (shape, tuning params, IR features); ~10% MAPE; top-1 80%,
top-50 95% of oracle; cheap training. Grim: no predictor; source-level static features (smem,
threads, wave estimate, tile params) are the pragmatic first step.

**SwizzlePerf — 2508.20258v1** — LLM-guided program-ID remap for XCD locality; hardware-aware
context (rocprofv3 bottleneck metrics + HIP device attrs + arch guide) + L2 hit rate as
bottleneck metric; 9/10 kernels, up to 2.06×, 70% L2 hit improvement, <5 min vs 2 weeks.
Grim: no swizzling/remap layer; the persistent dispatch + grid/block geometry is a natural place
to add a program-ID remap pass informed by SWIZZLE-style hardware-aware context.

### 1.3 Persistent kernel / fusion / dataflow papers

**FlashMoE — NeurIPS-2025-flashmoe-fast-distributed-moe-in-a-single-kernel-Paper-Conference.pdf**
— single fused MoE kernel (incl. dispatch+compute+comm); clang + hiprtc GenAI path; locality-aware
token grouping + dynamic warp merging; scales to 72 GPUs; ASUS+MOE + Mixtral-8×7B case study,
accuracy <1e-7.

**Helm — 2607.02521v1** — persistent kernel for *training*: full-graph materialized pipeline,
ownership-based live buffers, worker specialization, fused optimizer step; 1.68× ImageNet epoch,
1.38× pretraining step; sustained occupancy across steps.

**Kitsune — 2502.18403v1** — dataflow execution on GPUs: inter-CTA queue via L2 + global atomics +
modified grid scheduler; 1.3–2.4× across 5 apps; 16–98% off-chip traffic reduction; extracts
parallelism from hidden/reduction dims; training support.

**DeepFusionKernel — 2026.acl-short.15.pdf** — aggressive SwiGLU fusion (4 launches → 1 kernel);
row/col-major tiling trade-offs; profiler-driven scheduler; 9.7% A100, 13.2% H100 throughput;
robust to long-context generation.

**Ada-MK — 2605.11581v1** — MegaKernel portability to Ada/L20 (no TMA, 128KB SMEM): 3D shared-mem
constraint model + K-dim splitting (−50% peak SMEM), MLIR DAG offline search (eliminates runtime
branching), heterogeneous hybrid (TensorRT-LLM Prefill + MegaKernel Decode); 23.6% L20 throughput,
50.2% over vLLM at BS=1.

**ComFuse — 2606.02963v1** — fused communication with progressive precision and ARGUS dynamic
collective config; 2.23× kernel, 90th-pct TP latency −42% on H100; gate + epilogue fusion.

**G-3/4/5 persistent & fusion primitives in grim** — `scythe_persistent.rs` persistent dispatch
scaffold with quant_mode branch; `charon.rs` MoE fused dispatch family; `fused_dequant_gemm.rs`
training-side fused path + `grim_madam_update_f32`; `mxfp4_gemm.rs` fused RMSNorm+MXFP4+RoPE+KV;
`comm_fuse.rs` fused allreduce+rmsnorm + p2p epilogue; `qkv_attention.rs` wave-partitioned LDS
merge. Grim has the individual fusion/persistent primitives; the gestalt gap is composing them into
a single pipeline with the right persistent/task decomposition, fusion boundaries, and register/LDS
budget accounting.

### 1.4 Chiplet / disaggregated / multi-GPU papers

**Fleet — 2604.15379v1** — chiplet-aware persistent kernel on MI350: Chiplet-task abstraction,
M-major windowed traversal, cooperative L2 weight tiling (L2 hit 16%→61% at bs=64, HBM traffic
−37%), hierarchical two-level sync (L2-local counters + single GPU-scope fence per XCD);
1.3–1.5× decode latency vs vLLM at bs=1–16.

**SwizzlePerf — 2508.20258v1** — program-ID remap for XCD locality (see above). Same chiplet/L2
locality problem as Fleet, attacked from the remap angle rather than the task-decomposition angle.
Composable with Fleet.

**Swarm — 3730584.pdf** — distributed inter-kernel GPU queues on Grace-Hopper; RDMA directly from
kernel; in-network reduction + coalescing; 2–3.5× at scale. Grim's multi-GPU path (rccl.rs +
multi_gpu_launch.rs + CommFuse p2p epilogue) is the intra-node analog; Swarm's inter-kernel queue
idea is the cross-kernel dataflow layer that could sit on top of the RCCL/CommFuse transport for
persistent multi-operator pipelines.

**Parallel Kittens — MLSys-2026-parallelkittens...** — three factors: transfer mechanism × scheduling
× overheads; <50 lines/device; up to 4.08× seq-parallel, 2.33× data/tensor parallel; copy engine
saturates >256MB, register ops for in-network reduction. Grim has transport primitives; gap is the
policy + overlap schedule.

**Multi-GPU in grim** — `rccl.rs` (NCCL, in-place device reduce `sum_gradients_device` at 604-655,
tp all-reduce hook at 492-498), `p2p_route.rs` (PeerDirect/HostBounce, PCIe threshold), `peer_access.rs`
(P2PStatus/LinkType probe, HIP symbols re-declared at 33-45), `multi_gpu_launch.rs` (M-shard split +
per-device JIT + RCCL all-reduce, lines 25-99, RCCL launch at 86-96), `comm_fuse.rs` (p2p epilogue
atomicAdd at 20-34, fused rmsnorm+allreduce at 36-69). Grim's transport layer is correctly built;
what's missing is the chiplet-aware scheduling (Fleet/SwizzlePerf), the inter-kernel queue (Swarm),
and the transfer/schedule policy (Parallel Kittens).

### 1.5 LLMs-as-optimizers / kernel generation / benchmarking papers

**QiMeng-Tensify — QiMeng-Tensify_Scaling_Up_Tensor_Computation_Optimization_via_Architecture-Aware_LLM-Guided_MCTS.pdf**
— architecture-aware MCTS for tensor optimization; LLM proposes + evaluator rejects; 25%/−11.56%
across QPOS; structured DAG space.

**Kernel Foundry — 2606.31238v1** — diagnosis-driven evolutionary kernel optimization; expert-guided
init via distillation + retrieval; multi-island evolution; experience library by error type + bottleneck;
100% correctness L2; outperforms LLM-based + automated baselines.

**FlashInfer-Bench — MLSys-2026-flashinfer-bench...** — closed-loop virtuous cycle: FlashInfer Trace
schema (definition/workload/solution/evaluation), curated dataset from real serving traces (79 defs,
2474 workloads; GEMM 17/440, GQA 20/293, MLA 4/85, DSA 3/248, GDN 9/403, Norm 13/102, Sampling
12/884, MoE 1/19), robust benchmarking (runtime isolation, low-bit + non-deterministic sampling
support), dynamic apply() substitution into SGLang/vLLM, leaderboard. Key findings: 30/32 correctness
errors are compile failures; models underuse hardware intrinsics (mma/tcgen05); Triton > CUDA for agents
on correctness+speed; CUDA ceiling higher; agent wins: GEMM best (116× over PyTorch via cuBLAS dispatch),
RMSNorm median 1.7× (peak 3.7×), GQA paged decode 6.1× on extreme query-group imbalance; GQA ragged,
MLA paged, MoE all <0.4× baseline — paging/KV-indexing/expert-routing synthesis is hard.

**KernelBench (referenced by FlashInfer-Bench)** — generation-capability benchmark; FlashInfer-Bench
adds the production-facing path (apply() into real engines) that KernelBench lacks.

### 1.6 Kernel isolation / reproducibility / workflow papers

**Kerncap — 2605.03208v2** — HSA-level kernel extraction for both HIP and Triton: intercept dispatches
at HSA runtime (libkerncap.so + HSA tool interposition: hsa_amd_queue_intercept_create), VA-faithful
device-memory snapshot (hsa_memory_copy, chunked streaming at 64MiB, address-space closure preserves
embedded T** pointers without DWARF/pointer chasing), automated source discovery (compile_commands.json
+ DWARF via llvm-dwarfdump + nm disambiguation + grep fallback; Triton AST walk for @triton.jit),
self-contained reproducer (Clang VFS overlay for HIP source-level recompile with exact original flags;
tuning-pinned Triton reproducers that bind captured autotuner config to preserve numerical contract),
validation (smoke test 4.6s llama.cpp/2.7s LAMMPS; byte-exact 129.4s; tolerance-based numpy.allclose
for Triton with NaN detection). Results: 6 workloads across CDNA2/CDNA3/RDNA3 (gfx942/gfx90a/gfx1100);
152MB–30GB snapshots; vLLM fused_moe_kernel VA-faithful capture (185 regions, ~30GB, 24×702MB gate/up
+ 24×330MB down); 4–5× inner-loop speedup, 13.6× on llama.cpp case study; interception-only overhead
≤1.2× on CDNA2/CDNA3, ≤1.4× on RDNA3 for large workloads; capture cost bandwidth-bound at ~1.7GB/s
(gfx942), ~830MB/s on W7900 (PCIe topology); rocprofv3 degrades llama.cpp tg32 −38% vs Kerncap's
interception-only 251 vs baseline 254; case study: hoist runtime branch outside unrolled inner loop via
if constexpr templating → refactored mul_mat_vec_q.

**Proteus — Characterizing_the_Performance_and_Usability_of_GPU_JIT_Compilation_Interfaces_using_Proteus.pdf**
— characterization of GPU JIT compilation interfaces; usability + performance of JIT paths. Relevance:
grim uses hiprtc JIT (hiprtcCompileProgram, hiprtcRunCallback) for source-assembling; Proteus frames
the JIT-interface design space that grim's hiprtc path lives in.

### 1.7 Misc papers (partial relevance / tangential)

**2502.18403v1** — Kitsune (covered above, dataflow relevance).
**2601.16294v2** — SFC-based comm-avoiding GEMM (Intel) — SFC thread mapping, 2.5D/3D decomposition,
BRGEMM TPP. Tangential to LLM decode GEMM on AMD; the SFC idea for 1D index→2D tile mapping is a
concrete thread-mapping primitive but not the central lever for grim's persistent/chiplet path.
**2511.15503v1** — Macaron SPMD (fused bidirectional attention on MI350/B200) — relevant as a chiplet-aware
attention fusion example but not directly targeting grim's GEMM/MoE/decode path.
**2606.09080** — GaLore (memory-efficient LLM training) — tangential to kernel autotuning.
**BVSampler, RefinedQuantizationFramework, EsotericKernels, ARA, ParallelKITTENS_MLSys2026, DistributedMoE**,
**KernelFusion_KernelBench**, **GrammaticalErrorCorrectionOtter**, **SofiaOptimizer** — various tangential
or architecture/method papers with limited direct transfer to the grim kernel surface as currently scoped.
**Hawkeye (62_Hawkeye...pdf)**, **Chapter 7 (1887_4301430-Chapter 7.pdf)**, **j.issn.1000-565X.240498.pdf**,
**OptimizingStandardConvolutionforDiversePrecisiononDCU.pdf**, **3730584.pdf (Swarm, covered)**,
**3804601.3804607.pdf (MemSpiro, covered)**, **2606.11357v2 (Goldschmidt SVD sparse attention, covered)**,
**2508.20258v1 (SwizzlePerf, covered)**, **2604.15379v1 (Fleet, covered)**, **2605.03208v2 (Kerncap, covered)**,
**CharTuner (covered)**, **2502.18403v1 (Kitsune, covered)** — assigned above.

---

## Pass 2 — holistic gestalt composition (the whole > sum of parts)

The central thesis: every paper above is one lever. The performance ceiling isn't reached by pulling
one lever; it's reached by composing them into a single optimization surface where each lever's output
becomes a constraint or input for the next, producing multiplicative rather than additive effects.

The composition has three layers: **(A) the development loop** (how you iterate), **(B) the kernel
surface** (how one persistent kernel is structured), and **(C) the autotune/search surface** (how the
best configuration is found and retrieved). Grim already has fragments of all three. The gestalt
contribution is showing how the full corpus composes them.

### 2.1 Layer A — the development loop: Kerncap isolation × FlashInfer-Bench virtuous cycle × Helm/Fleet persistent validation

Purpose: make kernel iteration fast enough that aggressive composition is economically viable.

Composition:
1. **Kerncap** (HSA-level intercept + VA-faithful snapshot + VFS reproducer + tuning-pinned Triton
   reproducer) turns a full-application rebuild-and-rerun loop (128s llama.cpp; 64s LAMMPS) into an
   isolated edit-recompile-validate loop (18.3s build + 6.9s run on llama.cpp; 13.8s + 4.3s on LAMMPS;
   4–5× inner-loop speedup, 13.6× end-to-end on the case study). Critically, Kerncap's tuning-pinned
   Triton reproducers preserve the *numerical contract* of the autotuner config — without that, replaying
   a JIT kernel under the autotuner silently picks a different config and produces numerically different
   outputs (Flash Attn FP16 example: switching fastest→2nd-fastest config, only 7.7% slower, changes
   11.3% of output elements with max abs error 1.22e-4; the BLOCK_N tile reorders the softmax-denominator
   reduction across K). This is exactly the correctness hazard that makes aggressive autotuning dangerous.
2. **FlashInfer-Bench's apply() + Trace schema** turns validated kernel candidates into production impact
   without engine rewrites: the best validated kernel is dynamically substituted into SGLang/vLLM at runtime.
   For grim, the analog is: once an auto-tuned persistent kernel variant is validated (byte-exact or
   tolerance-based against the CPU oracle / reference), it is wired into the dispatch as the winning config
   for that (shape_class, arch, format) bucket — no engine rewrite, just a dispatch-table update.
3. **Helm's full-graph materialized persistent pipeline** + **Fleet's chiplet-task decomposition** give the
   *target structure* that the loop iterates on: not a single fused kernel, but a persistent kernel with
   task descriptors (RMSNorm, QKV, attention, gate+up+SiLU, down+res) scheduled within a persistent runtime,
   where edits to one task type don't require recompiling the whole decoder.

Grim's current position: `scythe_persistent.rs` has the persistent dispatch scaffold and a
`moe_task_descriptor_t`, but today it's FP32-only and the task graph is implicit. `qkv_attention.rs`,
`mxfp4_gemm.rs`, `fused_dequant_gemm.rs`, `comm_fuse.rs` carry the individual fused pieces. The missing
piece for Layer A is: (a) an HSA-level or hiprtc-level intercept/capture path (Kerncap analog) so that
iterating on one fused task in isolation is cheap; (b) a structured task graph (Fleet analog) that the
persistent runtime schedules; (c) an apply()-like dynamic substitution (FlashInfer-Bench analog) so the
validated winner is deployed without engine rewrite.

Concrete interaction: Kerncap's tuning-pinned reproducer concept directly solves the "don't tune a wrong
kernel" hazard that WaveTune/TTX/FlashInfer-Bench all warn about — the numerical contract of the autotuned
config is captured and preserved, so the predictor/retrieval layer (Layer C) can be evaluated against a
faithful reference, not a silently-different config.

### 2.2 Layer B — the kernel surface: Fleet chiplet-task decomposition × SwizzlePerf remap × Macaron SPMD × CharTuner subspace pruning × DeepFusionKernel fusion boundaries × ComFuse progressive comm

Purpose: structure one persistent kernel so that chiplet/L2 locality, fusion boundaries, and register/LDS
budget all compose into a single coherent micro-architecture decision.

Composition (single persistent decoder kernel on a chiplet GPU, e.g. MI350-class or RDNA3-class with
partitioned L2):
1. **Fleet's task model**: decompose the decoder layer into Chiplet-tasks (one per XCD for a GEMM),
   CU-tasks (wavefront-level), wavefront-tasks. For each GEMM (QKV, gate+up, down), Fleet uses 8
   Chiplet-tasks vs 96–256 CU-tasks in the naive decomposition (2.6× fewer tasks at bs=1; Table in
   Fleet Fig 4). This is the *task decomposition* that the persistent scheduler dispatches.
2. **Fleet's traversal order**: M-major windowed traversal so consecutive workers on the same XCD share
   weight tiles in L2. At bs=64, M-major raises L2 hit from 39% (no coop) to 61.4%, reducing HBM reads
   37% (6,203→3,925 GB). At bs=1–16 where m_tiles=1 and no coop reuse occurs, both M-tile and M-split
   perform identically (1.13–1.16× over Mirage), so the gain there is purely scheduling overhead reduction
   (fewer dispatches). The L2 hit rate model L2_hit_weight = 1 − 1/min(W, m_tiles) (Fleet Eq 1) gives the
   analytic prediction.
3. **SwizzlePerf's program-ID remap**: on top of Fleet's task decomposition, remap the program IDs so that
   cooperating tiles land on the same XCD. SwizzlePerf's LLM-guided remap (hardware-aware context: rocprofv3
   bottleneck metrics + HIP device attrs + arch guide; bottleneck metric = L2 hit rate; <5 min vs 2 weeks
   for expert) generates the exact remap formula for a given kernel/algorithm/grid shape. For GEMM, the remap
   co-locates tiles that reuse rows of A on the same XCD; for softmax, grouping all row chunks into the same
   XCD across the two-phase reduction; for layer norm, grouping column-chunks for the same row. SwizzlePerf
   generates correct patterns for 9/10 kernels (2.06× transpose, 1.54× softmax, 70% L2 hit improvement on
   stencil 2D). The key insight: SwizzlePerf's remap and Fleet's M-major traversal both target the same L2
   locality objective but from different angles — remap changes *which XCD a tile runs on*; traversal changes
   *the order tiles access weights*. They compose: remap first (so cooperating tiles are on the same XCD),
   then M-major traversal within each XCD (so those tiles share weight tiles in L2).
4. **Macaron SPMD** as a cross-check: Macaron's chiplet-aware fused bidirectional attention on MI350/B200
   (B200 2.30× prefill/3.44× decode) validates that chiplet-aware decomposition + fusion across chiplets is
   a real win on AMD hardware; its SPMD decomposition is the spatial analog of Fleet's task decomposition.
5. **CharTuner's subspace pruning** applied to the persistent kernel's GEMM tiles: each GEMM in the persistent
   kernel (QKV, gate+up, down) has a tile/loop parameter space. CharTuner's 8-subspace decomposition + PCA
   ranking + top-k retention (55.2% space reduction) prunes that space *before* any measured search. The
   reduced space then feeds Layer C's autotune. Crucially, CharTuner's finding that different shapes share
   optimal configs (e.g. PSO: M=N=K=88,120,160 identical config) vs need distinct configs means the autotune
   surface can be partially precomputed per shape-class — which is exactly the WaveTune/TTX retrieval model.
6. **DeepFusionKernel's fusion boundaries**: DeepFusionKernel's finding that true reductions (Softmax) are bad
   fusion targets (long-range dependencies limit cross-SM streaming) while GEMMs + pointwise gating are good
   (A2 = (XWUp) ⊗ SiLU(XWGate), Y = A2 WDown; fusing the first stage eliminates intermediates) directly
   informs *where* to cut the persistent kernel's task boundaries. The Fleet task graph already does this
   (QKV as one Chiplet-task, gate+up+SiLU fused into one Chiplet-task, down+res as another) — DeepFusionKernel
   provides the empirical justification for those boundaries on bandwidth-bound decode.
7. **ComFuse's progressive precision + ARGUS dynamic config** for the inter-task communication: within the
   persistent kernel, intermediate results pass between tasks via L2/register (Fleet's intra-chiplet) or via
   the CommFuse p2p epilogue (cross-chiplet/cross-device). ComFuse's progressive precision (don't materialize
   full FP32 intermediates when the consumer tolerates FP16/BF16) and ARGUS's dynamic collective config (choose
   the comm strategy per message size/link type) are the *communication* layer that sits between Fleet's tasks.

Grim's current position: `scythe_persistent.rs` has the persistent scaffold and task descriptor, but no
chiplet-task decomposition, no M-major traversal, no program-ID remap, no subspace-pruned GEMM tile autotune,
no fusion-boundary justification beyond "fuse what we fuse". The gestalt contribution is the ordered composition:
task decomposition (Fleet) → traversal + remap (Fleet + SwizzlePerf) → fusion boundaries (DeepFusionKernel) →
comm precision (ComFuse) → GEMM tile autotune within reduced subspace (CharTuner) → all scheduled by the
persistent runtime. Each step's output is a constraint for the next: the task decomposition determines which
GEMMs exist; their tile spaces are then pruned by CharTuner; the pruned tile configs then constrain the
register/LDS budget that determines whether M-major traversal + remap fit; the comm precision determines the
intermediate format between tasks.

### 2.3 Layer C — the autotune/search surface: CharTuner subspace + WaveTune bilinear + TTX XGBoost + SwizzlePerf LLM + FlashInfer-Bench apply() + Kernel Foundry diagnosis

Purpose: find and retrieve the best configuration for a given (shape, arch, format, system) without paying
full compile+time on the critical path.

Composition (four search modalities feeding one retrieval surface):
1. **Offline subspace pruning (CharTuner)**: decompose the per-kernel parameter space into semantic subspaces,
   PCA-rank them, retain top-k (55.2% reduction). This is the *first* filter: the full candidate set for each
   GEMM in the persistent kernel is never enumerated in full; only the reduced subspace is searched. This is
   done once per (kernel, arch) and cached — it's not on the critical path.
2. **Offline predictor training (TTX/WaveTune)**: from the measured data collected during the reduced-space
   search (shape, tile_config, format, arch, G, L, w, measured_latency), train a predictor. TTX's XGBoost
   (~10% MAPE, top-50 95% of oracle, cheap training) and WaveTune's bilinear model (wave-conditioned, with
   the wave-count term computable from existing launch geometry) are two candidate model families; the choice
   between them is itself an autotune decision (MLSys optimizer-design: tune the optimizer). Grim's existing
   compile+time loop is the data-collection path; Kerncap's tuning-pinned reproducer ensures the measured
   latency corresponds to a faithful config.
3. **Hardware-aware LLM remap (SwizzlePerf)**: for the chiplet/L2 locality objective, SwizzlePerf's LLM-guided
   remap generates the program-ID remapping formula from hardware-aware context + bottleneck metric feedback.
   This is a *different* search modality (LLM proposes, profiler validates, bottleneck metric guides) that sits
   alongside the GEMM tile autotune — it optimizes the *mapping* (which tile goes where), while CharTuner/TTX/
   WaveTune optimize the *tile config* (how big, how split). They are complementary and both feed the same
   persistent kernel's launch geometry.
4. **Runtime retrieval (WaveTune dual-table + FlashInfer-Bench apply())**: at runtime, for a hot/unseen shape,
   the bilinear model or XGBoost predictor retrieves the winning config in microsecond time (WaveTune's dual-table
   retrieval), and apply() deploys it into the dispatch without engine rewrite. Cold shapes fall back to the
   measured compile+time search in the reduced subspace (CharTuner's reduced space makes this cheaper than full
   space: PSO converges in ~130 vs 282 iterations).

Grim's current position: `autotune.rs` + `tile_picker.rs` + `gemm_tuning.rs` do an empirical compile+time search
over candidates + FCP fallback; `jit_cache.rs` caches hsacos by (entry, arch, spec, source-hash); `autotune.rs:247-256`
caches AutotuneConfig winners. What's missing: (a) subspace pruning (CharTuner) before the search; (b) a predictor
(WaveTune/TTX) on top of the cache for runtime retrieval; (c) a hardware-aware remap (SwizzlePerf) for chiplet/L2
locality; (d) the FlashInfer-Bench-style apply() deployment path; (e) Kernel Foundry's diagnosis-driven validation
(structured error-type + bottleneck library) to gate correctness before tuning.

Concrete interaction: CharTuner's finding that RS in the reduced space still achieves 1.64× (vs NAIVE_TUNER's 0.51×)
means that even a cheap random search in the pruned space beats a full-space search — this is the empirical
justification for doing subspace pruning *first* and then letting a cheap predictor/search run in the reduced space.
WaveTune's wave-count term (ceil(G/N_SM)) is computable from grim's existing grid/block/wavefront geometry with no
new probe — so the WaveTune model can be trained on grim's own measured data without adding new measurement burden.
SwizzlePerf's remap is generated from rocprofv3 bottleneck metrics + HIP device attrs + arch guide — grim already
has `peer_access.rs` (P2PStatus/LinkType probe) and could add rocprofv3-style bottleneck metrics to the autotune
measurement path, feeding both the predictor (Layer C) and the remap (Layer B).

### 2.4 Cross-cutting: sparsity / low-precision / correctness gate

**Goldschmidt (2606.11357v2)** — SVD-based sparse attention; 5.3–16.7× decode; structured sparsity avoids O(N²).
This is an *alternative* to dense attention GEMM on low-utilization ops; it composes with the persistent kernel by
replacing the attention task's GEMM with a sparse path when the operational intensity is below the ridge point
(Fleet's roofline model: AI_eff = B/(1−L2_hit_rate) vs ridge point 245 on MI350; Fig 7). The GEMM tile autotune
(Layer C) tells you when a GEMM is memory-bound enough to consider Goldschmidt's sparse alternative.

**MemSpiro (3804601.3804607)** — register-file KV spilling when L2/HBM bandwidth saturated; 7.3% end-to-end
latency reduction. Composes with the persistent kernel's attention task: when the KV cache no longer fits in the
persistent runtime's on-chip buffers (Fleet's L2-window model: active working set = one K-chunk tile per worker,
~1MB resident even when per-XCD partition exceeds L2 capacity by 6×), spill to the register file (MemSpiro) rather
than to HBM. This is the *memory hierarchy* decision that sits on top of Fleet's L2-window model.

**Correctness gate (cross-cutting)**: every layer above is gated by the same discipline that `q4k_dequant.rs` already
exhibits (host CPU mirror oracle) and that Kerncap's tuning-pinned reproducer enforces (numerical contract preserved):
(byte-exact for HIP variants; tolerance-based numpy.allclose for Triton with NaN detection). Kernel Foundry's
diagnosis-driven validation (structured error-type + bottleneck library) is the scalable version of this for the full
composition: each candidate kernel variant is diagnosed for correctness issues *and* dominant performance bottlenecks
before it's promoted. FlashInfer-Bench's finding that 30/32 correctness errors are compile failures (not runtime) is
a reminder that the gate must include compile-time validation, not just runtime.

---

## Pass 2 summary — the gestalt stack

The whole, composed:

- **Development loop** = Kerncap isolation (fast edit-recompile-validate, tuning-pinned numerical contract) ×
  FlashInfer-Bench apply() (deploy validated winner without engine rewrite) × Helm/Fleet persistent pipeline
  (the target structure being iterated on).
- **Kernel surface** = Fleet chiplet-task decomposition (8 Chiplet-tasks per GEMM vs 96–256 CU-tasks) ×
  M-major windowed traversal (L2 hit 16%→61%, HBM −37% at bs=64) × SwizzlePerf program-ID remap (co-locate
  cooperating tiles on same XCD; 2.06×, 70% L2 hit improvement) × Macaron SPMD (cross-chiplet attention fusion
  cross-check) × DeepFusionKernel fusion boundaries (fuse GEMMs+pointwise, don't fuse reductions) ×
  ComFuse progressive precision + ARGUS dynamic comm config (intermediate format + comm strategy between tasks) ×
  CharTuner subspace-pruned GEMM tile spaces (55.2% reduction before search).
- **Autotune surface** = CharTuner offline subspace pruning (first filter) × TTX/XGBoost or WaveTune/bilinear
  offline predictor trained on reduced-space measured data (runtime retrieval) × SwizzlePerf hardware-aware LLM
  remap (optimizes mapping, complementary to tile config) × WaveTune dual-table microsecond retrieval (hot shapes) ×
  measured compile+time in reduced space (cold shapes) × Kernel Foundry diagnosis-driven validation (correctness +
  bottleneck gate before promotion).
- **Alternative paths** = Goldschmidt sparse attention (when GEMM is memory-bound below ridge point) × MemSpiro
  register-file KV spill (when on-chip KV buffers saturated) — both triggered by the same roofline/L2-window
  models that Fleet's autotune surface produces.

The multiplicative claim: each layer's output constrains the next, so the composition isn't additive. Example:
Fleet's task decomposition determines which GEMMs exist; CharTuner prunes each GEMM's tile space; the pruned tile
configs determine the register/LDS budget that decides whether M-major traversal + SwizzlePerf remap fit; the
fusion boundaries (DeepFusionKernel) determine the intermediate format; ComFuse's progressive precision determines
the comm cost between tasks; the whole thing is validated by Kernel Foundry's diagnosis gate and deployed by
apply(). Pulling one lever in isolation (e.g. just SwizzlePerf remap, or just CharTuner pruning) leaves most of the
surface unoptimized; composing them yields the sustained-occupancy, chiplet-aware, autotuned, fused, correctness-gated
persistent kernel that the corpus's individual papers each point toward.

---

## What's new in the holistic synthesis that wasn't in the per-paper view

1. **Kerncap as the enabling infrastructure layer** — the per-paper view treats Kerncap as "kernel isolation tool";
   the holistic view places it as the *development loop* that makes the whole composition economically iterable
   (4–5× inner-loop speedup, 13.6× on the case study, tuning-pinned numerical contract that prevents silent
   autotuner config drift).
2. **Fleet + SwizzlePerf + Macaron as one chiplet-locality stack** — the per-paper view treats them as three
   separate chiplet papers; the holistic view composes them into task decomposition → traversal → remap, where each
   step's output constrains the next and they attack the same L2 locality objective from different angles.
3. **CharTuner subspace pruning as the first filter for the autotune surface** — the per-paper view treats CharTuner
   as "another autotuner"; the holistic view places it as the *space-reduction* step that makes the downstream
   WaveTune/TTX/SwizzlePerf search tractable (PSO converges in ~130 vs 282 iterations in reduced space; RS in
   reduced space still 1.64× vs full-space 0.51×).
4. **FlashInfer-Bench apply() as the deployment layer** — the per-paper view treats it as "benchmark for LLM agents";
   the holistic view places it as the *dynamic substitution* that deploys validated winners into the dispatch without
   engine rewrite, closing the loop from candidate → validated → deployed.
5. **Goldschmidt + MemSpiro as roofline-triggered alternative paths** — the per-paper view treats them as separate
   sparse/spilling papers; the holistic view triggers them from the same Fleet roofline/L2-window models that the
   autotune surface produces, making them conditional alternatives rather than independent replacements.
6. **Kernel Foundry diagnosis-driven validation as the scalable correctness gate** — replaces the ad-hoc "CPU oracle
   parity" with a structured error-type + bottleneck library that scales to the full composition.

---

## Concrete file+line patch sketches (reference, not applied)

These sketches show what fidelity looks like when the holistic composition is turned into edits against
`crates/grim-backend-rocm`. Line numbers are as reviewed and may drift.

### H.1 Persistent task graph + Fleet-style chiplet-task decomposition (Layer B, step 1–2)

**Grim anchors:**
- `src/kernels/scythe_persistent.rs:148-245` — persistent dispatch kernel; `moe_task_descriptor_t` +
  `quant_mode` branch at line ~215 (`if (moe->quant_mode == MOE_QUANT_FP32)`)
- `src/kernels/qkv_attention.rs:78` — `num_waves = blockDim.x / wave_size`
- `src/kernels/mxfp4_gemm.rs:327-359` — `grim_mxfp4_gemm_tiled` fixed 2D tile
- `src/device/roc_device.rs:7986-7989` — hardcoded `block_dim = HipDim3::new(16,16,1)`

**What to add:** extend `moe_task_descriptor_t` (or add a `fleet_task_descriptor_t`) to carry task type
(RMSNorm, QKV, attention, gate+up+SiLU, down+res, moe-dispatch) + chiplet-id + tile traversal order (M-major
windowed vs N-major) + program-ID remap formula (Fleet step 2 + SwizzlePerf). The persistent runtime scheduler
(one workgroup per chiplet, rest workers) dispatches these. This is the Fleet task model ported to grim's
persistent scaffold. Sketch:
```rust
// New task descriptor, companion to moe_task_descriptor_t
enum FleetTaskType { RMSNorm, QKV, Attention, GateUpSiLU, DownRes, MoEDispatch }
struct FleetTaskDescriptor {
    task_type: FleetTaskType,
    chiplet_id: u32,          // XCD this task is scoped to (Fleet Chiplet-task)
    traversal: TraversalOrder, // M_major_windowed | N_major | M_split
    remap_formula: RemapFormula, // program-ID remap for XCD locality (SwizzlePerf)
    tile: TileConfig,          // pruned by CharTuner subspace (Layer C)
}
```
**Files:** `src/kernels/scythe_persistent.rs` (extend the descriptor + dispatch loop near line 215),
`src/device/roc_device.rs` (launch geometry near 7948/7986 reads the descriptor's tile/remap instead of
hardcoding 16×16).

### H.2 CharTuner-style subspace pruning for grim's GEMM tile space (Layer C, step 1)

**Grim anchors:** `src/autotune.rs:118-150` (LaunchConfig), `src/autotune.rs:247-256` (Autotuner cache),
`src/kernels/mxfp4_gemm.rs:327-359`, `src/kernels/wmma_gemm.rs` (WMMA arm + FP8/MXFP4/MXFP8 Jay/Magpie).

**What to add:** decompose each GEMM kernel's tile parameter space (block_m, block_n, block_k, split_k,
lds_double_buffer, use_wmma/mfma, threads, per-format flags) into semantic subspaces (prefetching, tiling,
vectorization, thread clustering, C storage — CharTuner's Ω1..Ω8 map), benchmark each subspace across a
representative shape set, PCA-rank by five-number summary of per-shape improvement, retain top-k (CharTuner
keeps Ω6+Ω4, 55.2% reduction). The reduced subspace then feeds the autotune search. This is done once per
(kernel, arch) and cached — not on the critical path. Sketch:
```rust
// Subspace decomposition, applied to grim's LaunchConfig parameter space
enum GEMMSubspace { Prefetch, Tiling, Vectorization, ThreadCluster, CStorage, ... }
struct SubspaceRank { subspace: GEMMSubspace, pca_score: f64, top_k: bool }
// Computed offline per (kernel, arch), cached in Autotuner alongside AutotuneConfig wins
```
**Files:** `src/autotune.rs` (add subspace decomposition + PCA ranking near the Autotuner cache at line 251),
`src/kernels/mxfp4_gemm.rs` / `src/kernels/wmma_gemm.rs` (surface tile params as subspace-participating
parameters). Note CharTuner's MI210/ROCm-6.0 setup is the direct analog to grim's ROCm target; the 8-subspace
decomposition + 8-optimizer suite (GA/SA/BYS/DT/GBRT/PSO/SOA/RS) is a ready-made search-strategy zoo to evaluate
against grim's current empirical search.

### H.3 WaveTune/TTX predictor + SwizzlePerf remap on top of the reduced-space measured data (Layer C, steps 2–3)

**Grim anchors:** `src/autotune.rs:247-256` (cache), the empirical FCP fallback compile+time path, `peer_access.rs`
(P2PStatus/LinkType probe), the measurement records from H.2.

**What to add:**
1. Persist reduced-space measured samples `(shape, tile_config, format, arch, G, L, w, measured_latency)` as grim
   runs autotune, building the TTX/WaveTune training set. Extend `Autotuner`'s cache_dir shadow (line 255) or add a
   parallel measurement log.
2. Train TTX-style XGBoost or WaveTune-style bilinear model offline; for ROCm start with source-level static features
   (smem bytes, thread count, wave estimate, tile params) + the wave-count term `ceil(G/N_SM)` computable from existing
   launch geometry. Runtime side loads the model and pre-filters the candidate set before any compile.
3. Add SwizzlePerf-style hardware-aware remap generation: from rocprofv3-style bottleneck metrics (L2 hit rate,
   HBM traffic — grim could add these to the autotune measurement path via `peer_access.rs`-style probes + rocprofv3
   counters) + HIP device attrs + arch guide, an LLM-guided (or rule-based) remap generates the program-ID remapping
   formula for the persistent kernel's grid, optimizing L2 hit rate as the bottleneck metric. The remap formula is
   stored alongside the tile config in the Autotuner cache.

**Files:** `src/autotune.rs` (extend cache/measurement + add predictor load + remap formula storage near line 251),
`src/device/roc_device.rs` (dispatch path near 7948 reads tile config + remap formula from cache; wave-count term
computed from existing grid/block/wavefront geometry). A new offline training/remap-generation script (Python: XGBoost
for TTX; DSPy-style LLM loop for SwizzlePerf) consumes the persisted samples — not in the Rust crate.

### H.4 FlashInfer-Bench-style apply() dynamic substitution (Layer A, step 2 + Layer C, step 4)

**Grim anchors:** `src/autotune.rs:247-256` (cache wins), `src/device/roc_device.rs` GEMM dispatch path (search for
`matmul_op`/`matmul_with_solution`), `jit_cache.rs` (hsaco cache by entry/arch/spec/source-hash).

**What to add:** once a kernel variant is validated (byte-exact or tolerance-based against CPU oracle / reference —
Kerncap's tuning-pinned contract + q4k_dequant.rs's host mirror discipline), wire it into the dispatch as the winning
config for that (shape_class, arch, format) bucket via an apply()-like dynamic substitution: the dispatch looks up the
validated winner in a trace-indexed table (FlashInfer Trace schema analog: definition/workload/solution/evaluation) and
redirects to it at runtime, without engine rewrite. Cold shapes fall back to measured compile+time in the CharTuner-reduced
space; hot shapes retrieve the winner from the predictor/retrieval table (WaveTune dual-table). Sketch:
```rust
// FlashInfer Trace analog: a validated kernel contract
struct KernelTrace {
    definition: KernelDef,      // I/O tensors, dtypes, axes
    workload: Workload,         // concrete (M,N,K, shape_class, arch, format)
    solution: KernelSolution,   // source/hsaco + tile config + remap formula
    evaluation: Evaluation,     // correctness + measured latency + L2 hit rate
}
// apply()-like dispatch: look up validated winner, fall back to compile+time for cold shapes
fn dispatch_traced(shape: &Shape, arch: Arch, format: QuantFormat) -> LaunchConfig {
    if let Some(winner) = trace_table.lookup(shape.class(), arch, format) {
        if winner.evaluation.valid { return winner.solution.launch_config.clone(); }
    }
    // cold shape: measured search in CharTuner-reduced subspace
    autotune_in_reduced_subspace(shape, arch, format)
}
```
**Files:** `src/device/roc_device.rs` (dispatch path, near `matmul_op`/`matmul_with_solution` search hit), new
`src/trace.rs` (Trace schema + lookup table + apply() dispatch), `src/autotune.rs` (cold-shape fallback into reduced
subspace).

### H.5 Kerncap-style isolation path for grim's persistent kernel (Layer A, step 1)

**Grim anchors:** `src/kernels/scythe_persistent.rs:148-245`, `src/device/roc_device.rs` hiprtc path
(`hiprtcCompileProgram`, `hiprtcRunCallback`, source-assembly), `jit_cache.rs`.

**What to add:** an HSA-level or hiprtc-level intercept/capture path for grim's persistent kernel, so that iterating
on one fused task in isolation is cheap. The analog: intercept dispatches at the HSA runtime (Kerncap's
`hsa_amd_queue_intercept_create` + libkerncap.so LD_PRELOAD), snapshot the device-memory state VA-faithfully
(`hsa_memory_copy`, chunked streaming, address-space closure preserves embedded T** pointers without DWARF), discover
the persistent kernel's source + tile/remap params (compile_commands.json + DWARF + nm, or hiprtc source-hash from
`jit_cache.rs`), emit a self-contained reproducer with a VFS overlay for source-level recompile with the exact original
flags, and validate (byte-exact for HIP; tolerance-based for Triton with NaN detection). For grim's hiprtc JIT path,
the tuning-pinned reproducer binds the captured autotuner config (tile + remap) to preserve the numerical contract.
Sketch: a CLI `grimcap extract <task_type> --cmd "... launch persistent kernel ..." --source-dir ./src` that produces
an isolated reproducer directory with dispatch metadata, hsaco, device-memory snapshot, source + VFS overlay, and a
Makefile with recompile/replay/validate targets. The inner-loop build+run cost on grim's persistent kernel should drop
from full-application rebuild (128s llama.cpp analog) to isolated (18s build + 7s run analog), a 4–5× speedup.

**Files:** new `src/grimcap/` module (HSA intercept + VA-faithful snapshot + source discovery + reproducer generation +
validation), or a standalone tool consuming grim's `jit_cache.rs` source-hash + `scythe_persistent.rs` descriptor.
This is the enabling infrastructure that makes the whole Layer B/C composition iterable.

### H.6 Goldschmidt/MemSpiro alternative paths triggered by roofline (Layer C cross-cutting)

**Grim anchors:** `src/kernels/qkv_attention.rs` (attention), `src/kernels/mxfp4_gemm.rs` (GEMM), Fleet roofline
model (AI_eff = B/(1−L2_hit_rate) vs ridge point; L2 hit rate model L2_hit_weight = 1 − 1/min(W, m_tiles)).

**What to add:** in the autotune/predictor surface, compute the operational intensity and L2 hit rate for each GEMM/attention
task in the persistent kernel; when a GEMM is memory-bound below the ridge point, consider Goldschmidt-style sparse
alternative (structured sparsity that avoids O(N²) for attention when utilization is low); when the KV cache no longer
fits in the persistent runtime's on-chip buffers (active working set = one K-chunk tile per worker, ~1MB resident even
when per-XCD partition exceeds L2 by 6×), spill to the register file (MemSpiro) rather than HBM. These are conditional
alternatives triggered by the same roofline/L2-window models that the autotune surface produces. Sketch:
```rust
// Roofline-triggered alternative path selection
fn select_task_impl(task: &FleetTaskDescriptor, roofline: &RooflineModel) -> TaskImpl {
    if task.is_attention() && roofline.is_memory_bound_below_ridge(task) {
        TaskImpl::SparseAttention(GoldschmidtConfig)   // structured sparsity, O(N^2) avoided
    } else if task.is_kv_cache() && !roofline.fits_on_chip(task.kv_working_set()) {
        TaskImpl::SpilledKV(MemSpiroConfig)            // register-file spill, not HBM
    } else {
        TaskImpl::DenseGEMM(task.tile_config)
    }
}
```
**Files:** `src/kernels/qkv_attention.rs` (add sparse attention path gated by roofline), new `src/roofline.rs`
(Fleet-style roofline + L2 hit rate model + alternative-path selection), `src/kernels/mxfp4_gemm.rs` / attention
paths (dense GEMM vs sparse alternative dispatch).

---

## Truthfulness / scope caveats

- This is an audit + research synthesis, not a correctness proof, not a benchmark. Items needing on-device
  verification are stated as such; items backed by grim's existing parity tests (q4k_dequant.rs host mirror,
  charon.rs G-A2 routing parity) are noted.
- The composition claims are framed as "this is how the corpus's methods stack" rather than "this will definitely
  win on grim's hardware/shapes". The papers' results are on their own kernels/hardware; the transfer claim is that
  the *methods* compose onto grim's existing abstractions.
- Kerncap, CharTuner, SwizzlePerf, Fleet, Helm, FlashMoE, FlashInfer-Bench, Kitsune, DeepFusionKernel, ComFuse,
  Ada-MK, Macaron, Swarm, MemSpiro, Goldschmidt are all external research; the sketches above are reference patches
  a later implementer can re-derive from the cited lines and paper anchors, not applied edits.
- Golden-path dependency: the whole composition depends on Layer A (Kerncap-style isolation + apply() deployment) being
  in place before Layer B/C can be iterated economically. Without fast isolated iteration, the persistent task graph +
  remap + subspace pruning + predictor composition is too expensive to pursue. This is the single highest-leverage
  infrastructure piece to add first.
