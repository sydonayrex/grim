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

## Part G — concrete patch examples (file + line references)

These are illustrative edit sketches to show what fidelity looks like when the synthesis is turned into
edits. They are **not** applied; they are reference patches a later implementer can re-derive from the
cited lines. Line numbers are as of the reviewed source tree and may drift.

### G.1 WaveTune-style bilinear latency predictor on top of the existing autotune cache

**Paper anchor:** WaveTune (2604.10187v1) §3.3, Eq. (1): `T(G, L) ≈ α·G·L + β·G + γ·L + δ`; §4.3
fits a compact bilinear model per `⟨cmacro, w⟩` bucket; §4.4 uses two-stage runtime selection
(macro via coefficient table, micro via anchor table).

**Grim anchor:** grim already computes the wave count that WaveTune's model needs:
- `src/kernels/qkv_attention.rs:78` — `const int num_waves = blockDim.x / wave_size;`
- `src/kernels/qkv_attention.rs:325-329` — paged KV variant, same `num_waves = blockDim.x / WARP_SIZE` with wave partitioning
- `src/kernels/mxfp4_gemm.rs:327-359` — `grim_mxfp4_gemm_tiled` (fixed 2D tile, K/32 loop)
- `src/device/roc_device.rs:7986-7989` — hardcoded launch geometry in `launch_mxfp4_gemm_tiled`
  (`block_dim = HipDim3::new(16, 16, 1)`, `grid_x = ((n+15)/16)`, `grid_y = ((m+15)/16)`)
- `src/device/roc_device.rs:8035` — fixed `num_splits` heuristic in `launch_mxfp4_gemm_splitk`
- `src/autotune.rs:118-150` — `LaunchConfig` struct (block_m, block_n, block_k, split_k, threads)
- `src/autotune.rs:247-256` — `Autotuner` cache (in-memory + optional on-disk shadow)

**What to add:**

1. Extend `AutotuneConfig` (or add a sibling struct) to carry WaveTune model coefficients.
   **File:** `src/autotune.rs`, near `AutotuneConfig` (line ~186).
   **Sketch:**
   ```rust
   // New struct, companion to AutotuneConfig
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct WaveTuneCoeffs {
       pub alpha: f64,  // G*L interaction term
       pub beta:  f64,  // G marginal
       pub gamma: f64,  // L marginal
       pub delta: f64,  // constant offset
       pub wave_count: u32, // w = ceil(G / N_SM) this coeffs are valid for
       pub n_sm: u32,       // GPU SM count (cached at probe time)
   }
   ```
   Then add a `HashMap<MacroConfigKey, HashMap<u32, WaveTuneCoeffs>>` (macro-config → wave-count → coeffs)
   to `Autotuner` alongside the existing `cache: HashMap<KernelKey, AutotuneConfig>` (line 251).

2. Add the wave-count + grid-size computation to the measurement path so each autotune sample records
   `(G, L, w, c_macro, c_micro, measured_latency)` matching WaveTune's sample tuple.
   **File:** `src/autotune.rs` or the empirical FCP fallback near `fcp_fallback_tile_search`.
   **Sketch:** compute `G = grid_m * grid_n`, `L = ceil(K / T_K)`, `w = ceil(G / n_sm)` from the launch
   geometry (already available from `LaunchConfig` + the GEMM dims) and store alongside the latency.

3. Add the bilinear prediction + dual-table retrieval at the top of the launch-dispatch path so the
   predicted winner can be retrieved at microsecond cost for hot/unseen shapes.
   **File:** `src/device/roc_device.rs` near `launch_mxfp4_gemm_tiled` (line ~7948) or in a new
   `autotune_dispatch` helper called from `matmul_op` / `matmul_with_solution`.
   **Sketch:**
   ```rust
   // Stage I (WaveTune §4.4): for each candidate macro-config c_macro,
   // compute G, L, w = ceil(G / n_sm), look up coeffs theta{c_macro, w},
   // predict T = alpha*G*L + beta*G + gamma*L + delta, pick min.
   // Stage II: retrieve micro-config from anchor table keyed on (c_macro, w, L).
   ```
   The wave-count term `ceil(G / N_SM)` is directly computable from existing data (grid_m*grid_n and
   the SM count from `HardwareSpec`), so no new probe is required — this mirrors WaveTune §3.2's
   derivation of `w = ceil(G / N_SM)`.

4. Keep measured compile+time search as the cold-shape path and use the model for hot/unseen shapes —
   that's the WaveTune trade-off resolution (accuracy of search + microsecond overhead of heuristics).
   **File:** the dispatch path that currently calls `jit_compile_or_cache` + `hipModuleLaunchKernel` +
   `hipEvent` timing; add a model-lookup early-exit before the compile when a cached coeffs entry exists
   for the (shape_class, arch, w) bucket.

### G.2 MXFP4 tiled path parametrization (RVjit-style) + WaveTune auto-tune

**Paper anchor:** RVjit (IPDPS 2026) — JIT assembler parametrizes register grouping, loop ordering, kernel
shape at code-gen time based on probed hardware; 1.43× over hand-tuned GEMM on RISC-V. Grim already JITs
full HIP source via hiprtc, so the lesson transfers: parametrize the micro-kernel at JIT time and
auto-tune per arch + shape.

**Grim anchor:**
- `src/kernels/mxfp4_gemm.rs:327-359` — `grim_mxfp4_gemm_tiled`: one thread per (row, col), K/32 loop,
  hardcoded `exps_per_row = K / 32`
- `src/device/roc_device.rs:7986-7989` — hardcoded `block_dim = HipDim3::new(16, 16, 1)` launch
- `src/device/roc_device.rs:7968-7977` — fixed skinny-M heuristic routing to split-K
- `src/device/roc_device.rs:8035` — fixed `num_splits` selection
- `src/kernels/mxfp4_gemm.rs:368-406` — `grim_mxfp4_gemm_splitk`: split-K variant
- `src/kernels/scythe_persistent.rs:148-245` — persistent dispatch kernel; `quant_mode` branch at
  line ~215 (`if (moe->quant_mode == MOE_QUANT_FP32)`)

**What to add:**

1. Parametrize the MXFP4 tiled micro-kernel at JIT time. Replace the hardcoded tile (16×16) and K/32
   loop with a templated tile in the HIP source string, generated per (arch, shape) by
   `compute_kernel_source_with_spec` (called at `src/multi_gpu_launch.rs:64` and in `roc_device.rs`).
   **File:** `src/kernels/mxfp4_gemm.rs` (the kernel source string near line 327), generated from
   `src/kernels/source_asm.rs` style machinery.
   **Sketch:** the HIP source should take tile_m, tile_n, k_loop_tile as `#define`-generated constants
   (or template args) so the same JIT mechanism that already compiles the source can produce variants
   with different tile geometries; the launch geometry in `launch_mxfp4_gemm_tiled` (line 7986) then
   reads those constants instead of hardcoding `HipDim3::new(16, 16, 1)`.

2. Add the auto-tune loop for these tile params per (arch, M, N, K), gated by CPU-oracle parity.
   **File:** `src/autotune.rs` (extend the candidate generator / FCP fallback) or a new mxpf4-specific
   tuner paired with the existing `Autotuner` cache.
   **Sketch:** candidate set = {(tile_m ∈ {8,16,32}, tile_n ∈ {8,16,32}, k_loop_tile ∈ {8,16,32})}
   filtered by LDS budget (`LaunchConfig::smem_cost` pattern at `src/autotune.rs:130-132`) and the
   MXFP4 K%32 constraint already enforced at `src/device/roc_device.rs:7960`; measure each on-GPU and
   cache the winner per (arch, M, N, K).

3. Pair with the WaveTune predictor (G.1) so the tuned geometry for a given (M,N,K) is retrievable at
   runtime without a fresh compile+time — same architecture as RVjit's "probed hardware → parametrize →
   select" loop, but with a microsecond retrieval layer on top.

4. Extend the persistent dispatch `quant_mode` arm at `src/kernels/scythe_persistent.rs:215` to call a
   fused MoE-MXFP4 kernel (see G.4) instead of routing through per-expert standalone MXFP4 GEMM.

### G.3 TTX-style predictor from grim's own measured data (offline training)

**Paper anchor:** TTX (ISPASS 2026) — XGBoost predictor over (input shape, tuning params, IR-level features),
~10% MAPE on compute-heavy ops, top-50 reaches 95% of oracle, training ~3-4s on EPYC for 20k points.

**Grim anchor:** `src/autotune.rs:247-256` (cache), the empirical FCP fallback compile+time path, and the
measurement records captured in G.1 step 2.

**What to add:**

1. Persist the measured samples `(shape, tile_config, format, arch, G, L, w, measured_latency)` to disk
   as grim runs autotune / FCP fallback, building the training set TTX needs.
   **File:** extend `Autotuner`'s `cache_dir` shadow (line 255, `src/autotune.rs`) or add a parallel
   measurement log.

2. Train an XGBoost (or small GBT) model offline over those samples; for ROCm, start with source-level
   static features (shared memory bytes from `__shared__` declarations, thread count from launch bounds /
   grid-block geometry, estimated wave count from grid/block/wavefront, tile params) because grim on ROCm
   doesn't expose PTX-style IR trivially — full hsaco/ISA feature extraction is a later step.
   **File:** a new offline training script (Python/XGBoost) consuming the persisted samples; not in the
   Rust crate. The runtime side (Rust) only needs to load the trained model and call it to pre-filter the
   candidate set before any compile.

3. Integrate the predictor as an early filter in the dispatch path so hot/unseen shapes get a predicted
   winner without compile+time.
   **File:** `src/device/roc_device.rs` near the GEMM dispatch (e.g., `matmul_op` / `matmul_with_solution` —
   search for those symbols in `src/device/roc_device.rs` to place the call), or in the autotune lookup
   path at `src/autotune.rs:291-293` (`lookup`).

### G.4 Multi-GPU transfer/schedule policy (Parallel Kittens-style)

**Paper anchor:** Parallel Kittens (MLSys 2026) — three factors: transfer mechanism (copy engine vs TMA vs
register ops, different saturation by message size), scheduling (inter-SM vs intra-SM overlap), design
overheads (NCCL-like sync/buffering choices can cost 1.7-4.5×). Copy engine saturates >256MB; register
ops needed for in-network reduction.

**Grim anchor:**
+- `src/rccl.rs:56-80` — RCCL bindings (NCCL), `ncclAllReduce`, in-place device reduce
+- `src/rccl.rs:604-655` — `RcclAllReduce::sum_gradients_device` (in-place device all-reduce via `ncclAllReduce`)
+- `src/rccl.rs:449-490` — `p2p_memcpy_async` (hipMemcpyPeerAsync wrapper) + tensor-parallel all-reduce hook (P2-WI-2 / WI-R3 canonical call site)
+- `src/rccl.rs:492-498` — single canonical call site for TP all-reduce; designed as interception point for a future `CommComputeOverlapConfig` stream-overlap
- `src/p2p_route.rs:40-49` — `RouteLink` enum (PeerDirect, HostBounce)
- `src/peer_access.rs:47-58` — `P2PStatus` enum (P2P, Pcie, Host), peer probe before any peer memcpy
- `src/peer_access.rs:33-45` — HIP symbols re-declared locally (hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess)
- `src/multi_gpu_launch.rs:25-99` — `launch_multi_gpu_kernel`, M-shard split + per-device JIT + RCCL all-reduce,
  lines 86-96 do the RCCL launch
- `src/kernels/comm_fuse.rs:20-34` — `grim_comm_fuse_p2p_epilogue` atomicAdd to peer buffer (register-level
  in-network reduction)
- `src/kernels/comm_fuse.rs:36-69` — `grim_fused_allreduce_rms_norm` fused RMSNorm+allreduce

**What to add:**

1. Add a policy function `select_multi_gpu_path(message_size_bytes, p2p_status, num_gpus, link_type) ->
   MultiGpuPolicy { transfer_mechanism, overlap_mode, use_commfuse: bool, use_rccl: bool, use_host_bounce: bool }`.
   **File:** new module or extend `src/p2p_route.rs` / `src/multi_gpu_launch.rs`.
   **Sketch:**
   ```rust
   pub enum TransferMechanism { PeerDirectCopyEngine, RegisterP2P, HostBounce, RcclCollective }
   pub enum OverlapMode { None, InterSm, IntraSm }
   pub struct MultiGpuPolicy {
       pub transfer: TransferMechanism,
       pub overlap: OverlapMode,
       pub use_commfuse_epilogue: bool,  // register-level atomicAdd to peer buffer
   }
   // Calibrate thresholds using PK-style microbenchmarks on the target hardware:
   // copy engine saturates >256MB, register ops needed for in-network reduction,
   // TMA-like bulk transfers for mid-range. PK's numbers (copy engine 81-82% of
   // theoretical max on H100/B200, TMA 74-78%, register ops 70-76%) give the
   // starting calibration; re-measure on the deployment box.
   ```

2. Wire the policy into `launch_multi_gpu_kernel` (line 25, `src/multi_gpu_launch.rs`) so the per-device
   launch + RCCL collective selection is driven by the policy rather than always doing M-shard split + RCCL
   all-reduce. In particular, for small per-shard message sizes where peer-direct copy-engine or register P2P
   wins, prefer the CommFuse atomicAdd path (`grim_comm_fuse_p2p_epilogue`) over RCCL all-reduce; for large
   messages where copy engine saturates, use the copy-engine route; for mid-range, use TMA-like bulk transfers.

3. Add the compute/collective overlap schedule (intra-SM where compute and communication granularities align,
   inter-SM where it reduces transfer size) — PK's intra-SM schedule overlaps the per-device compute with the
   RCCL launch; grim currently launches compute first (lines 56-83) then RCCL (lines 86-96) with no overlap.

### G.5 Static feature extraction for the predictor (TTX/IR features, lighter-weight first step)

**Paper anchor:** TTX uses PTX/LLVM-IR features; grim on ROCm doesn't expose those trivially, so start with
source-level static features.

**Grim anchor:**
- `src/kernels/mxfp4_gemm.rs:327-359` — `__launch_bounds__(256)`, `__shared__` absent in the tiled kernel
  (LDS accessed via the default shared memory window; SMEM bytes can be derived from the LDS declarations in
  the full source string)
- `src/autotune.rs:130-132` — `LaunchConfig::smem_cost(bytes_per_elem)` already computes LDS bytes from
  block_m * block_k + block_k * block_n — reuse/extending this pattern
- `src/kernels/qkv_attention.rs:78` — `num_waves` computed from `blockDim.x / warpSize`;
  `src/kernels/qkv_attention.rs:63-78` — wave_size resolved from warpSize (32 on RDNA2, 64 on CDNA)

**What to add:**

1. Add a `StaticKernelFeatures` struct captured at JIT-compile time (before or alongside `hiprtcCompileProgram`),
   extracted from the HIP source string and the launch configuration:
   **File:** new, near `src/kernels/source_asm.rs` or in `src/autotune.rs`.
   **Sketch:**
   ```rust
   #[derive(Debug, Clone)]
   pub struct StaticKernelFeatures {
       pub shared_memory_bytes: u32,    // from __shared__ declarations in source, or smem_cost
       pub thread_count: u32,           // blockDim.x * blockDim.y * blockDim.z
       pub wave_count_estimate: u32,    // ceil(grid_blocks / n_sm) using grid/block/wavefront
       pub tile_m: u32,                 // from launch config
       pub tile_n: u32,
       pub tile_k: u32,
       pub k_loop_tiles: u32,           // ceil(K / tile_k)
       pub uses_wmma: bool,
       pub uses_mfma: bool,
       pub arch: String,
   }
   ```
   Extract `__shared__` bytes by scanning the source string for `__shared__ float s_mem[N]` / `__shared__ ...`
   declarations (regex or simple parser), or reuse `LaunchConfig::smem_cost` pattern where the LDS is derived
   from tile geometry. The wave count estimate `ceil(grid_blocks / n_sm)` mirrors WaveTune's `w = ceil(G / N_SM)`.

2. Store these features alongside each autotune measurement record (G.1 step 2, G.3 step 1) so the offline
   predictor has features to train on. This is the lighter-weight first step before any hsaco/ISA feature
   extraction.

### G.6 Quantized MoE fused path + dedicated fused MoE-MXFP4 kernel

**Paper anchor:** WaveTune covers grouped GEMM (MoE) as one of its three representative kernels (Table 2:
Grouped GEMM (MoE) row), with physical coordinates `G = sum_i ceil(M_i/T_M) * ceil(N/T_N)`, `L = ceil(K/T_K)`.
Grim's persistent dispatch already routes MoE through `grim_moe_fused_grouped_device`; adding a fused
MoE-MXFP4 entry means the MXFP4-quantized MoE runs through a single fused kernel instead of per-expert
standalone MXFP4 GEMM.

**Grim anchor:**
- `src/kernels/scythe_persistent.rs:148-245` — persistent dispatch kernel; `quant_mode` branch at line 215;
  the else-arm (line 224) currently sets `ST_ERROR` for non-FP32 quant modes
- `src/kernels/scythe_persistent.rs:193-225` — the FP32 MoE arm calls `grim_moe_fused_grouped_device`
- `src/kernels/scythe_persistent.rs:215` — `if (moe->quant_mode == MOE_QUANT_FP32)` is the only wired arm
- `src/kernels/charon.rs` — `grim_moe_fused_grouped` family (the FP32 dispatch kernel)
- `src/kernels/charon_wmma.rs` — WMMA variant `grim_moe_fused_grouped_wmma`
- `src/device/roc_device.rs:8378` — `launch_mxfp4_gemm_tiled` called for the MXFP4 QKV path
- `src/kernels/q4k_dequant.rs` (not read in full, but referenced) — host CPU mirror oracle pattern for Q4_K

**What to add:**

1. Implement the quantized arms of the persistent dispatch `quant_mode` branch at `src/kernels/scythe_persistent.rs:215`.
   **File:** `src/kernels/scythe_persistent.rs`, the `if (moe->quant_mode == MOE_QUANT_FP32)` arm (line 215) and
   its else (line 224).
   **Sketch:** add arms for `MOE_QUANT_MXFP4` (and fp8/mxfp8/q80/iq) that call the corresponding fused grouped
   dispatch kernel (e.g., a new `grim_moe_fused_grouped_mxfp4_device` for MXFP4) instead of falling through to
   `ST_ERROR`. Each new arm needs a CPU-oracle parity gate (see G.7) before it is tuned.

2. Add a dedicated fused MoE-MXFP4 kernel so MXFP4-quantized MoE runs through one fused kernel rather than
   per-expert standalone MXFP4 GEMM.
   **File:** new kernel source in `src/kernels/` (e.g., `moe_mxfp4_grouped.rs`), wired into
   `src/kernels/source_asm.rs` and the persistent dispatch arm.
   **Sketch:** the fused kernel should dequantize MXFP4 expert weights on-the-fly (like `mxfp4_gemm_tiled`'s
   `mxfp4_dot_block` at `src/kernels/mxfp4_gemm.rs:349-356`) and accumulate per-expert, matching the WaveTune
   grouped GEMM physical coordinate decomposition (sum over experts of the per-expert GEMM grid).

3. Wire the new kernel into `src/kernels/scythe_persistent.rs:215` arm and add the launch path in
   `src/device/roc_device.rs` (next to `launch_mxfp4_gemm_tiled` at line 7948).

### G.7 Correctness-before-tuning guardrail — extend oracle parity to every new quant variant

**Paper anchor:** not from a single paper — cross-cutting discipline. Grim's `q4k_dequant.rs` already runs a
host CPU mirror against `grim_quant::dequant_q4k` as an oracle; every new auto-tuned variant should have the
same gate.

**Grim anchor:**
- `src/memory/storage.rs:560-563` — host dequant dispatch for Q2K/Q3K/Q4K/Q5K/Q6K via `grim_quant::dequant_*`
- `src/device/roc_device.rs:6778-6806` — `launch_dequant_q4k` (device-side dequant)
- `src/device/roc_device.rs:6808-6836` — `launch_dequant_fp8`
- `src/device/roc_device.rs:6838-6850` — `launch_dequant_mxfp4` (start of MXFP4 dequant launcher)
- `src/kernels/charon.rs:2100-2102` — G-A2 parity test: routing assignment from synthetic SoftmaxTopK vs CPU
  reference (`grim_nn::moe::MoeRouter::route`)

**What to add:**

1. For every new quant variant (MXFP4, FP8, MXFP8, Q8_0, IQK) added in G.6, add a host CPU mirror dequant
   that produces the reference F32 weights using the same quantization convention as the device kernel, then
   parity-test the device output against the CPU reference at a few (M, N, K) points before the variant is
   auto-tuned.
   **File:** extend the host dequant dispatch in `src/memory/storage.rs` (near line 560, add MXFP4/FP8/MXFP8/Q8_0/
   IQK arms) and add parity tests in the kernel's `mod tests` (cf. `src/kernels/charon.rs:2100` G-A2 pattern).

2. Don't auto-tune a kernel whose correctness isn't gated — tuning a wrong kernel produces a fast wrong answer.
   Add an assertion or test-gate that the parity test passes before the auto-tuned winner is cached in
   `Autotuner` (line 251, `src/autotune.rs`).

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
