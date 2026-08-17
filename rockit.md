# rockit.md — grim-backend-rocm kernel audit + old/tuna/ auto-tuning research synthesis

Scope: grim-backend-rocm kernel implementations (reviewed from source) cross-referenced
against research papers in `old/tuna/`, focused on means to auto-tune kernels per model + user
system. This is an audit/context-synthesis document, not a patch.

---

## Part A — grim-backend-rocm kernel audit (what exists, what's strong, what's missing)

### A.1 Kernel surface reviewed

From `crates/grim-backend-rocm/src/kernels/`:

- **Quantized GEMM (fused dequant + GEMM)** — per-format kernels:
  - `q4k_gemm.rs`, `q5k_gemm.rs`, `q2k_gemm.rs`, `q3k_gemm.rs` (not all read in full),
    `iq_gemm.rs` (IQ2_XS/XXS/S, IQ3_XXS/S, IQ4_NL/XS, plus standalone Q4_K/Q5_K/Q2_K/Q3_K/Q8_0).
  - Each is a per-(format, M,N,K) dot-product loop over dequantized weights; backward variants
    exist for most.
  - `q4k_dequant.rs` — standalone Q4_K dequant with a host CPU mirror used as oracle for parity
    tests (good: dequant correctness is gated by CPU oracle, not just "compiles").
- **Dense / arch-specialized GEMM**:
  - `wmma_gemm.rs` — HIP source with two arms: WMMA path (rocwmma, gfx11xx/gfx12xx) and a
    scalar fallback for GFX10/RDNA2. Also bundles FP8, MXFP4, MXFP8 fused-dequant GEMM kernels
    (Jay/Magpie) and gfx1200 MFMA stubs.
  - `decode_gemm.rs` — small F16 decode-shaped GEMM (M ≤ 8), simple dot-product, used as the
    opt-in `grim_decode_gemm_f16` path.
- **Attention**:
  - `qkv_attention.rs` — `grim_qkv_attention` (online-softmax, wave-partitioned LDS merge,
    GQA), `grim_qkv_attention_paged` (paged KV), `grim_tree_attention` (tree/draft attention).
    Wave count is computed from `blockDim.x / warpSize` at runtime, which matches the
    wave-aware framing in the research (WaveTune).
- **Fused / specialized**:
  - `fused_dequant_gemm.rs` — `grim_fused_dequant_gemm_f16` + backward (STE) + `grim_madam_update_f32`
    (M+Adam optimizer step). This is the training-side fused path.
  - `mxfp4_gemm.rs` — tiled MXFP4 GEMM (`grim_mxfp4_gemm_tiled`), split-K variant
    (`grim_mxfp4_gemm_splitk` + reduce), fused RMSNorm+MXFP4+RoPE+KV (`grim_fused_rmsnorm_mxfp4_gemm_rope_kv`),
    fused RMSNorm+MXFP4 MLP, and `grim_qk_norm_rope`. The MXFP4 path is JITed as a full HIP source.
  - `charon.rs` — MoE fused-dispatch family (`grim_moe_fused_dispatch`, `grim_moe_fused_grouped`).
  - `charon_wmma.rs` — WMMA variant of the grouped MoE dispatch (`grim_moe_fused_grouped_wmma`).
  - `charon_backward.rs` — backward/auxiliary helpers.
  - `scythe_persistent.rs` — persistent-dispatch scaffold with a `moe_task_descriptor_t` and a
    dispatch loop that branches on `quant_mode` (FP32 today; quantized variants noted as "would
    branch to grim_moe_fused_grouped_fp8/mxfp4/mxfp8/q80/iqk").
- **Host-side dispatch / tuning**:
  - `roc_device.rs` — `matmul_op` (op-tagged GEMM: TLOLog vs by-M), `matmul_with_solution`,
    `launch_mxfp4_gemm_tiled`, `launch_mxfp4_backward_gemm`, `launch_decode_gemm_f16`,
    `launch_wmma_gemm`, split-K path (`rocblas_gemm_strided_batched_ex` + reduction kernel),
    RCCL-backed `all_reduce`/`comm_fuse_reduce`, graph-capture path.
  - `autotune.rs`, `tile_picker.rs`, `gemm_tuning.rs` — tile candidate search + lookup tables +
    empirical FCP fallback (`fcp_fallback_tile_search`) that compiles + times candidates on-GPU and
    keeps the winner.
  - `jit_cache.rs` — hsaco cache keyed by (entry, arch, spec, source-hash).

### A.2 What grim already does well (against the research framing)

1. **Wave-aware execution is structurally present.** `qkv_attention.rs` computes `num_waves =
   blockDim.x / warpSize` at runtime and partitions KV work across waves, merging partials in LDS
   on wave 0. That is exactly the "wave is a first-class discretization dimension" idea WaveTune
   argues for. The MXFP4/`wmma` paths also carry arch-gated compile-time wave/hardware branching
   (gfx11xx/gfx12xx WMMA; gfx1200 MFMA stubs). Grim encodes the *hardware-conditioned* branching
   WaveTune says is the right prior — it just doesn't yet have a *predictive model* sitting on top
   of it.

2. **Correctness gates are in place for quant dequant.** `q4k_dequant.rs` runs a host CPU mirror
   against the `grim_quant::dequant_q4k` oracle in tests. That's the right discipline: before tuning
   a quant kernel you need a trusted reference. Same pattern should be required for any new
   auto-tuned variant.

3. **ROCm transport layer is correctly built.** `rccl.rs` (NCCL wrappers, in-place device all-reduce
   via `sum_gradients_device`), `p2p_route.rs` (PeerDirect vs HostBounce, PCIe threshold), `peer_access.rs`
   (P2PStatus probe, LinkType topology). This is the *transport-mechanism* substrate that Parallel Kittens
   says must be chosen deliberately; grim has the primitives, not the tuned policy yet.

4. **MoE dispatch kernel family is real and persistent-style.** `charon.rs` has the grouped/sortless
   fused dispatch; `scythe_persistent.rs` wraps it in a persistent descriptor dispatch with a
   `quant_mode` branch ready for quantized variants. The CommFuse epilogue (`comm_fuse.rs`) does
   atomicAdd to peer buffers, which is the register-level in-network reduction Parallel Kittens
   identifies as the high-performance choice.

### A.3 Where grim is missing the auto-tuning layer the research targets

1. **No latency predictor — only measured search.** Grim's autotune does `jit_compile_or_cache` +
   `hipModuleLaunchKernel` + `hipEvent` timing on-GPU and keeps the winner (empirical FCP search in
   `tile_picker.rs`). That's offline/cache-fill caliber, not the microsecond online retrieval WaveTune
   and TTX target for serving. There is no bilinear model, no XGBoost/GBT model, no table-based
   predictor generalizing to unseen (M,N,K) without a fresh compile+time.

2. **No IR/static-feature extraction for a model.** TTX uses PTX/LLVM-IR features; grim compiles HIP
   source to hsaco and times it. The lighter-weight analog grim could do cheaply: extract static
   features from the source *before* compile (shared memory bytes from `__shared__` declarations,
   thread count from launch bounds / grid-block geometry, estimated wave count from grid/block/wavefront),
   plus the already-available tile config. That's a concrete missing feature class.

3. **No meta-optimizer design for the search itself.** Grim's current search is empirical compile+time
   over a candidate set with roofline pre-filtering. The MLSys optimizer-design paper's point: the
   search strategy itself can be auto-designed and evaluated against measured runtime in a framework
   that already supports HIP (Kernel Tuner). Grim's tile/format/arch search space is irregular and
   constrained — whether the current search is the best way to navigate it is an open question.

4. **MXFP4 tiled path has a single fixed launch geometry.** `mxfp4_gemm_tiled` uses a fixed 2D grid
   (one thread per (row,col)) and a K/32 loop. The WMMA path has autotuned tile picking; the MXFP4
   path doesn't appear to carry an equivalent per-shape tile autotune, despite being JITed and thus
   capable of parametrizing the micro-kernel at generation time (the RVjit lesson).

5. **Multi-GPU path is transport-complete but policy-incomplete.** `multi_gpu_launch.rs` does a simple
   M-shard split + per-device JIT + RCCL all-reduce. It doesn't select the transfer mechanism or overlap
   granularity by message size / link type the way Parallel Kittens prescribes, and it doesn't overlap
   the RCCL collective with the per-device compute the way PK's intra-SM schedule does.

6. **Quantized MoE fused path is scaffolded but not wired.** `scythe_persistent.rs` branches on
   `quant_mode` and names the quantized variants (`grim_moe_fused_grouped_fp8/mxfp4/mxfp8/q80/iqk`)
   but today only the FP32 arm is implemented. The MXFP4-MoE fused kernel does not exist as a dedicated
   entry; MXFP4-quantized MoE today would route through the standalone/tiled MXFP4 GEMM kernels per expert,
   not through a fused MoE-MXFP4 kernel.

---

## Part B — research synthesis from old/tuna/ (auto-tuning-relevant papers)

Papers reviewed (PDF → text extraction via pdftotext; full inventory in `old/tuna/` has 31 PDFs).
Only the auto-tuning / GEMM / multi-GPU / JIT-compilation papers are synthesized below; the
vision-convolution auto-tuners, CNN-specific papers, correctness/attention/training-system papers,
and the optimizer-design paper are noted as tangential or partially relevant.

### B.1 WaveTune — 2604.10187v1 (primary relevance)

- Core claim: tiled kernel latency is *wave-conditioned piecewise bilinear* — discrete wave effects
  (step at `ceil(G / N_SM)`) plus approximately linear intra-wave scaling. Modeling that structure
  beats black-box search and ML cost models on accuracy + runtime overhead.
- Runtime mechanism: precomputed model coefficients + lightweight dual-table retrieval, microsecond
  overhead, no fresh compile+time on the critical path.
- Results: up to 1.83× kernel-level speedup, up to 1.33× TTFT reduction, across 3 kernels (dense
  GEMM, grouped GEMM/MoE, FlashAttention) and 5 GPU architectures from 2 vendors.
- Direct applicability to grim: grim already knows `wave_size`, `blockDim.x`, grid dims, LDS bytes,
  tile config. The WaveTune model term `ceil(G / N_SM)` is computable from grim's existing grid+block+
  wavefront_size with no new probe. The missing piece is the predictor + retrieval table, not the
  underlying hardware abstraction.

### B.2 TTX — ISPASS 2026, "Towards Autotuning Triton Kernels via Latency Prediction with XGBoost"

- XGBoost predictor over (input shape, tuning params, IR-level features) → latency. ~10% MAPE on
  compute-heavy ops (GEMM, BMM, conv, FlashAttention, quantized GEMM, MoE) across V100/A100/H100/MI250.
- Top-1 ≈ 80% of oracle, top-50 ≈ 95%. Training is cheap (3-4s on EPYC for 20k points).
- Key lesson: IR-level features matter; reusing a best config across shapes can cause large degradation
  (Figure 1) — shapes must be part of the model, not just params.
- Applicability to grim: grim's compile+time loop could be replaced/augmented by a predictor trained
  offline from grim's own measured latency data. The pragmatic first step is source-level static features
  (shared memory, thread count, wave estimate, tile params) because grim on ROCm doesn't expose PTX-style
  IR trivially; full hsaco/ISA feature extraction is a later step.

### B.3 MLSys 2026 — "Automated Algorithm Design for Auto-Tuning Optimizers"

- LLMs generate the *optimizer* (search strategy), not the kernel. LLaMEA + Kernel Tuner: LLM proposes a
  metaheuristic; an EA loop evaluates it against measured kernel runtime in Kernel Tuner (which already
  supports HIP). Best LLM-generated optimizers: 72.4% improvement over SOTA human-designed optimizers on
  auto-tuning tasks.
- Relevance: grim's autotune search space (tile_m, tile_n, block_k, split_k, lds_double_buffer,
  use_wmma, use_mfma, threads, plus per-format flags) is exactly the kind of irregular constrained integer
  space Kernel Tuner targets. The actionable takeaway: treat the search strategy as a tunable artifact and
  benchmark it against grim's current empirical search on grim's own measured runtime — don't assume the
  current search is optimal.

### B.4 Parallel Kittens — MLSys 2026 (multi-GPU kernel design)

- Decomposes multi-GPU kernel performance into three factors: (1) transfer mechanism (copy engine vs TMA
  vs register ops — different saturation points by message size), (2) scheduling (inter-SM vs intra-SM
  overlap), (3) design overheads (sync/buffering choices in NCCL-like libs can cost 1.7×–4.5×).
- Results: up to 2.33× data/tensor parallel, 4.08× sequence parallel, 1.22× expert parallel; <50 lines
  of device code per kernel beyond the single-GPU base.
- Applicability to grim: grim has the transport primitives (rccl, p2p_route, peer_access) and the
  CommFuse register-level epilogue; what's missing is a *policy* that picks transfer mechanism + overlap
  granularity by message size and link type, and a schedule that overlaps the RCCL collective with the
  per-device compute. PK's microbenchmark numbers (copy engine saturates >256MB, register ops needed for
  in-network reduction) give the calibration grim currently lacks.

### B.5 RVjit — IPDPS 2026 (JIT for vector-length-agnostic SIMD)

- JIT assembler framework that dynamically picks register grouping, loop ordering, kernel shape at code-gen
  time based on probed hardware. 1.43× over hand-tuned GEMM on RISC-V.
- Relevance (transferred): grim already JITs full HIP source per kernel variant via hiprtc. The RVjit lesson
  is to parametrize the micro-kernel at JIT time and specialize by probed hardware — grim's WMMA path does
  this partially (compile-time arch gating + runtime tile picker), but the MXFP4 tiled path and the
  standalone quant GEMM paths are mostly one fixed geometry and could add a tile-parameterized JIT variant
  and auto-tune it the way RVjit auto-tunes register grouping.

### B.6 Papers reviewed but judged tangential / not the auto-tuning lever

- Vision-convolution/CNN auto-tuners on different hardware (`s11227-026-08327-6`, `s44443-026-00937-7_reference`,
  `3818618`, `CSE 26-122 YW TL` x2) — convolution on non-GPU or non-llm surfaces; limited transfer.
- System/attention/training/correctness papers (`osdi26-qiang`, `2603.21331`, `2605.26118`, `2606.09080`,
  `2606.14598`, `2606.25453`, `2606.28372`, `2607.23099`, `2607.27231`, `2512.12949`, `2512.23236`,
  `2601.14910`, `2601.15727`, `2603.09511`, `2603.10085`, `2605.03208`, `2605.28213`, `ssrn-6873159`,
  `1887_4301430` full text + chapter 6) — relevant to broader architecture/IO-aware attention/training
  design but do not propose kernel auto-tuning methods for the grim kernel surface.
- "Just-in-Time Convolution and GEMM Code Generation for SIMD Architectures" (IPDPS 2026, RVjit) — covered
  above; SIMD CPU focus, the JIT-parameterize-by-probed-hardware lesson transfers.

---

## Part C — synthesized recommendations (means to auto-tune grim kernels)

These are concrete levers, ordered by expected yield and feasibility given grim's existing code.

### C.1 Add a lightweight latency predictor on top of the existing cache (WaveTune-style)

What grim has: hsaco_cache + autotune wins per (entry, arch, M,N,K) + grid/block/wavefront/LDS/tile info.
What to add: a bilinear (or small GBT) latency model keyed on (tile.grid_stride_m, tile.grid_stride_n,
split_k, num_waves, lds_bytes, arch, format) that can predict the winner for an unseen (M,N,K) at runtime
without a fresh compile+time. The wave-count term `ceil(grid_m*grid_n / NSMs)` is already computable from
grIM's launch geometry. Retrieval is then table-based and microsecond-scale. This fits into autotune.rs /
jit_cache.rs without changing the FFI surface.

### C.2 Train a predictor from grim's own measured data (TTX-style)

Collect (shape, tile_config, format, arch, measured_latency) from grim's existing compile+time loop as it
runs in autotune / FCP fallback. Use those data points to train a predictor (XGBoost or small GBT, TTX's
choice) offline. At runtime, use the predictor to pre-filter the candidate set before any compile. For ROCm,
start with source-level static features (shared memory bytes, thread count, wave estimate, tile params) as
TTX does with IR features; upgrade to hsaco/ISA features later if the static features plateau.

### C.3 Treat the search strategy itself as a tunable (MLSys optimizer-design + Kernel Tuner)

Grim's current empirical candidate search is a baseline, not necessarily optimal for grim's irregular space.
Use Kernel Tuner's HIP support to evaluate alternative search strategies (greedy, stratified sampling,
Model-Guided Top-K like TTX, or an LLM-generated metaheuristic) against grim's own measured runtime. The
actionable first step: wrap grim's matmul/attention/quant GEMM dispatch as a Kernel-Tuner-evaluable target
and benchmark a couple of search strategies on real measured latency, rather than assuming the current
compile+time-over-candidates is the best navigator.

### C.4 Parametrize and auto-tune the MXFP4 tiled path (RVjit-style)

`mxfp4_gemm_tiled` currently uses one fixed launch geometry. Since it's JITed, parametrize the micro-kernel
(tile_m, tile_n, K-split/loop tiling, vector load grouping) inside the HIP source and auto-tune those per
arch + shape + (M,N,K), the way RVjit parametrizez register grouping per SoC. Pair with the WaveTune-style
predictor so the tuned geometry for a given (M,N,K) can be retrieved at runtime. Same idea applies to the
standalone quant GEMM paths (q4k/q2k/q5k/iq) where a parametrized tile over the dequant loop could beat one
fixed geometry for some (M,N,K) regimes.

### C.5 Add a multi-GPU transfer/schedule policy (Parallel Kittens-style)

Grim has the transport layer and the CommFuse register-level epilogue; add a policy function that, given
(message_size_bytes, P2PStatus, num_gpus, link_type), selects:
- transfer mechanism (peer-direct copy-engine vs direct register P2P vs host-bounce) by message size and
  link saturation points,
- overlap granularity (inter-SM vs intra-SM) for compute vs collective,
- whether to use the CommFuse atomicAdd peer-buffer path vs RCCL all-reduce vs host-bounce fan-in.

Calibrate the thresholds using PK-style microbenchmarks on the target hardware (copy engine saturates >256MB,
register ops needed for in-network reduction, TMA-like bulk transfers for mid-range). This is the gap between
"we have RCCL" and "we have a tuned multi-GPU kernel."

### C.6 Wire the quantized MoE fused path and add a fused MoE-MXFP4 entry

`scythe_persistent.rs` already has the `quant_mode` branch and names the quantized variants. The next step is
to implement the MXFP4 (and fp8/mxfp8/q80/iq) arms and add a dedicated fused MoE-MXFP4 kernel rather than
routing MXFP4-quantized MoE through per-expert standalone MXFP4 GEMM. This is gated by correctness: every new
quant variant needs a CPU oracle parity gate (mirroring `q4k_dequant.rs`'s host mirror) before it's tuned.

### C.7 Cross-cutting guardrails

- Keep correctness gates before tuning: quant dequant parity against CPU oracle; WMMA/MFMA numeric parity
  tests; attention online-softmax parity against a CPU/rocBLAS reference.
- Don't tune a kernel whose correctness isn't gated — tuning a wrong kernel produces a fast wrong answer.
- Prefer offline/cache-fill measured search for cold shapes and model-based retrieval for hot/unseen shapes;
  that's the WaveTune trade-off resolution (accuracy of search + microsecond overhead of heuristics).

---

## Part D — side-by-side: research vs grim-backend-rocm (concrete mapping)

| Research lever | grim component that already exists | grim gap / next step |
|---|---|---|
| Wave-conditioned latency model (WaveTune) | wave_size, num_waves runtime calc in qkv_attention; arch-gated WMMA/MFMA branching; tile config | no latency predictor/retrieval table; add bilinear model over (tiles,waves,LDS,arch,format) on top of jit_cache + autotune wins |
| XGBoost/IR-feature predictor (TTX) | compile+time empirical search + hsaco cache | no predictor; collect grim's own measured data, train offline, use source-level static features first (smem, threads, wave estimate, tile params) |
| Auto-designed optimizer (MLSys + Kernel Tuner) | autotune.rs tile candidate search, FCP fallback, empirical winner | current search is a baseline; evaluate alternative search strategies via Kernel Tuner's HIP backend on grim's measured runtime |
| JIT parametrize-by-probed-hardware (RVjit) | hiprtc JIT of full HIP source; WMMA compile-time arch gating + runtime tile picker | MXFP4 tiled path and standalone quant GEMM paths are mostly one fixed geometry; parametrize micro-kernel at JIT time and auto-tune per arch/shape |
| Multi-GPU transfer/schedule policy (Parallel Kittens) | rccl.rs (NCCL, in-place device reduce), p2p_route.rs (PeerDirect/HostBounce, PCIe threshold), peer_access.rs (P2PStatus/LinkType), comm_fuse.rs register-level atomicAdd epilogue, multi_gpu_launch.rs M-shard split + RCCL all-reduce | no policy selecting transfer mechanism + overlap granularity by message size/link type; no compute/collective overlap schedule; add PK-calibrated thresholds + CommFuse-vs-RCCL-vs-host policy |
| Quantized MoE fused path | scythe_persistent.rs quant_mode branch scaffold naming fp8/mxfp4/mxfp8/q80/iq variants; charon MoE dispatch family | MXFP4-MoE fused kernel doesn't exist; implement quantized arms with CPU-oracle parity gates; add dedicated fused MoE-MXFP4 entry instead of per-expert standalone MXFP4 GEMM |
| Correctness-before-tuning guardrail | q4k_dequant.rs host mirror vs grim_quant oracle; parity tests across kernels | extend oracle-gated parity to every new quant/format variant before tuning; don't tune un-gated kernels |

---

## Part E — what this audit does and doesn't claim

- This is a source-reading audit + research synthesis, not a correctness proof by hand-calc, and not a
  benchmark. Items that would need on-device verification are stated as "would need GPU verification" where
  relevant; items already backed by grim's existing parity tests are noted as such.
- The research recommendations are framed as levers to evaluate, not as proven wins on grim's specific
  hardware/shapes — the papers' results are on their own kernels/hardware; the transfer claim is that the
  *methods* map onto grim's existing abstractions.

---

## Part F — action items for the team owning the WIP tree

1. **Near-term, low-risk:** add source-level static feature extraction (smem bytes, thread count, wave estimate,
   tile params) to the autotune data collection path so a future predictor has features to train on. No kernel
   change; just richer measurement records.
2. **Near-term, low-risk:** implement the WaveTune-style `ceil(grid_m*grid_n / NSMs)` wave-count term as a
   first predictive feature computed from existing launch geometry — it's free and directly mirrors the paper's
   primary discretization.
3. **Medium-term:** train a lightweight latency predictor from grim's own measured autotune/FCP data and use it
   to pre-filter candidates before compile for hot/unseen shapes; keep measured search for cold shapes.
4. **Medium-term:** parametrize the MXFP4 tiled micro-kernel (tile_m, tile_n, K-loop tiling) at JIT time and
   auto-tune it per arch/shape, gated by CPU-oracle parity where applicable.
5. **Medium-term:** add a multi-GPU transfer/schedule policy function (message_size, P2PStatus, link_type →
   transfer mechanism + overlap) calibrated by PK-style microbenchmarks on the target system; wire CommFuse
   register-level path as the in-network-reduction option.
6. **Higher-effort / gated:** implement the quantized MoE fused arms (mxpf4 first) under scythe_persistent's
   quant_mode branch with CPU-oracle parity gates; add a dedicated fused MoE-MXFP4 kernel.
7. **Evaluation harness:** wrap grim's matmul/attention/quant-GEMM dispatch as a Kernel-Tuner-evaluable target
   and benchmark alternative search strategies against grim's current empirical search on real measured runtime.

End of rockit.md.
