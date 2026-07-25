# GRUM: Grim ROCm Unified Method

**Status:** Draft Specification v0.1
**Author:** Specification derived from synthesis of 27 papers (old/res3/) + grim codebase analysis
**Architecture review:** software-factory/CTO
**Quality gates:** software-factory/CQO
**Execution plan:** software-factory/COO + writing-plans
**Date:** July 2026

---

## 0. Executive Summary

GRUM is a 4-level hierarchical method that unifies the optimizer, kernel, quantization, and fusion research from `old/res3/` into a single coherent system for grim — a pure-Rust LLM inference+training engine targeting AMD ROCm GPUs. Each level solves one bottleneck, and together they compound.

**Why a hierarchy, not a flat list.** An optimizer that halves memory doesn't help if the kernel tiles are 4x too coarse. A fused kernel suite doesn't help if quantization resets destroy all optimization progress. GRUM's four levels are ordered by dependency: lower levels must land first because upper levels assume their invariants.

---

## 1. Context: Where grim Is Today

Grim is a 30-crate Rust workspace. The critical paths for this spec:

| Crate | What's there now | What GRUM changes |
|-------|-----------------|-------------------|
| `grim-backend-rocm/` | 9 kernel modules, JIT-compiled via hipRTC, rocBLAS FFI for GEMM, hipBLASLt for quantized GEMM | Level 1 (tile decomp) + Level 2 (fusion) + Level 3 (quant) |
| `grim-backend-rocm/src/kernels/wmma_gemm.rs` | Single 16x16x16 f16 WMMA tile, scalar fallback on RDNA2 | Level 1: hierarchical 32x32x16 tile decomposition |
| `grim-backend-rocm/src/kernels/decode_gemm.rs` | Single-buffer naive GEMM (one thread per output element) | Level 2: fused into the fused-dequant-gemm path |
| `grim-backend-rocm/src/kernels/qkv_attention.rs` | Phase-1 f32 online-softmax, Wave64, 4-way KV parallel | Level 2: incorporate MFMA tiles when gfx arch allows |
| `grim-autograd/` | AdamW optimizer (m + v moment buffers per param, CPU-side Vec<f32>) | Level 4: GRUM-Optimizer (fused GPU-side, 3 buffers instead of 2, Nesterov + spectral correction) |
| `grim-backend-rocm/src/kernels/fused_dequant_gemm.rs` | Fused dequant + GEMM for GPTQ | Level 3: SPARKLING-style asymmetric quantization reset |

ROCm hardware reality (per rocminfo):
- Primary target: gfx1036 (RDNA2 / Radeon 610M) — **no WMMA, no FP8**
- Future targets: gfx110x (RDNA3), gfx1200 (RDNA4) — **WMMA native**
- All kernels compile via hipRTC; no C++ build step

---

## 2. Level 1: Hierarchical WMMA Tile Decomposition

### 2.1 Problem

The current WMMA GEMM uses a single 16×16×16 f16 tile. On RDNA3+, rocWMMA's `fragment<matrix_a, 16, 16, 16>` executes one matrix-core operation per 64-thread wavefront, but the rocBLAS reference uses a 128×128 macro-tile internally. The gap means peak utilization is never reached because the K-dimension stride (16 elements per load) is too short to amortise the load-compute pipeline.

### 2.2 Synthesis from Papers

- **COSMOS** (ICLR 2026): hierarchical decomposition of the optimizer — SOAP for the leading subspace, MUON for the remainder. The same principle applies to GEMM: one tile size cannot be optimal for all M/N/K shapes. Use a tiling hierarchy keyed by shape.
- **RMNP** (arXiv, May 2026): row-wise ℓ2 normalization replaces Newton-Schulz iteration, showing that simple operations at the right granularity outperform complex ones. For WMMA, hierarchical decomposition with a small number of tile types beats a single general tile.

### 2.3 Design

Introduce three tile configurations, selected at dispatch time by the host launcher:

| Configuration | Block Tile | WMMA Fragment | Target Shape | Source |
|--------------|-----------|---------------|--------------|--------|
| `SMALL` | 16×16×16 | 16×16×16 | M≤32, N≤64, any K | Current implementation, unchanged |
| `MEDIUM` | 32×32×16 | 16×16×16 (2×2 WMMA calls per block) | M≥64, N≥128, K≥256 | COSMOS hierarchical decomposition |
| `LARGE` | 64×64×16 | 16×16×16 (4×4 WMMA calls per block) | M≥128, N≥256, K≥1024 | RMNP row-wise normalization applied per tile row |

The key insight from COSMOS: hierarchical tiling isn't just for throughput. By structuring the K loop as independent sub-tile rows, each row can apply a per-row scale factor (RMNP's ℓ2 normalization) with zero cross-row synchronization — exactly the pattern RMNP exploits for 43× faster preconditioning.

### 2.4 Implementation Plan

```
Phase 1a: Add MEDIUM and LARGE tile launcher variants
  - Introduce TileConfig enum in grim-backend-rocm/src/wmma.rs
  - Implement hipRTC kernel source for 32x32 and 64x64 macro-tiles
  - Add host-side tile selector (shape → tile config mapping)
  - Tests: microbenchmark each tile config against rocBLAS reference
  - Gate: each tile config achieves ≥80% of rocBLAS peak for its target shapes

Phase 1b: RMNP row-scale integration
  - After MEDIUM/LARGE tile K loop, apply per-row ℓ2 normalization to accumulator
  - f32 scale factor computed per row of the output tile (no LDS needed for 32 rows)
  - Tests: numerical parity with full-precision reference at 1e-5 absolute tolerance
```

### 2.5 Quality Gates (CQO)

- `cargo test --package grim-backend-rocm` must pass on gfx1036 (scalar fallback path)
- On gfx1100+ hardware: `MEDIUM` tile baseline ≥ 1.6× throughput vs `SMALL` for shapes in its target domain
- **Strong-test constraint** (from strong-tests skill): property test over M ∈ {1, 8, 16, 32, 64, 128}, N ∈ {64, 128, 256, 512, 1024}, K ∈ {64, 256, 1024, 4096} — numeric parity with CPU f32 reference at 1e-3 tolerance (f16 accumulation error)

---

## 3. Level 2: Multi-Kernel Fusion

### 3.1 Problem

Grim currently launches separate kernels for attention, GEMM, dequantization, RoPE, and elementwise operations. Each launch incurs hipModuleLaunchKernel overhead (~3–10μs on RDNA). For short sequence lengths (decode: M=1, N=4096, K=4096), kernel launch latency can be 30–50% of end-to-end time.

### 3.2 Synthesis from Papers

- **Nemotron-Flash** (NeurIPS 2025): systematically evolved attention operators for latency-optimal hybrid SLMs. The key result: operator fusion is architecture-specific — the optimal fused operator set for one GPU generation differs from another. For RDNA, the high launch cost relative to compute means *anything that touches the same K-dimension memory should fuse*.
- **Power Lines** (NeurIPS 2025): scaling laws for weight decay and batch size in LLM pre-training. For inference engine serving, batch size is typically 1–4, which amplifies the launch overhead problem.

### 3.3 Design

Two fusion targets, ordered by expected benefit:

**Fusion 1: GQA + RoPE + QKV-attention** (QKV-attention already in a single kernel — fuse RoPE into the same kernel so the input Q/K matrices are rotated before the attention loop, avoiding a separate hipModuleLaunch and a kernel readback of 2× head_dim floats.)

**Fusion 2: Fused-dequant-GEMM-RMSNorm.** The current `fused_dequant_gemm` kernel loads a quantized weight block, dequantizes it in registers, and computes a partial GEMM into a float accumulator. After this, a separate RMSNorm kernel reads the GEMM output. Fuse RMSNorm into the same kernel as the last reduction step, eliminating the intermediate write-to-HBM and read-back.

### 3.4 Implementation Plan

```
Phase 2a: RoPE-attention fusion
  - Embed RoPE rotation into qkv_attention kernel source, gated by #ifdef FUSE_ROPE
  - Accept sin/cos lookup table pointers as kernel arguments (or compute on-the-fly)
  - The rotation happens on the Q and K tiles before the K_V loop starts
  - Tests: output parity with separate RoPE kernel + existing qkv_attention kernel

Phase 2b: Fused-dequant-GEMM-RMSNorm
  - The fused_dequant_gemm kernel already loads weights in blocks and accumulates
    per-output-row partial sums. Add a final pass that loads the row's accumulated
    output, computes rms = sqrt(mean(sq(row)) + eps), and writes row / rms
  - Maintain the same dequant-tile-reuse pattern; no additional HBM round-trip
  - Tests: numerical parity with fused_dequant_gemm + separate rms_norm kernel
```

### 3.5 Quality Gates (CQO)

- Phase 2a: end-to-end latency reduction ≥ 15% for decode (M=1) at context=4096
- Phase 2b: end-to-end latency reduction ≥ 10% for prefill (M=seq_len) at seq_len=2048
- All fused paths must pass property-based numerical parity tests (same inputs → same outputs within f32 tolerance, 1e-5 absolute)

---

## 4. Level 3: Per-Layer Quantization with Asymmetric Reset

### 4.1 Problem

Grim supports GPTQ quantization (fused_dequant_gemm). But the quantization is static — once calibrated, it stays at that precision forever. During LoRA/QLoRA training, different layers converge at different rates. Over-quantized layers cause training quality loss; under-quantized layers waste memory bandwidth.

### 4.2 Synthesis from Papers

- **SPARKLING** (ICML 2026): mid-stage width expansion with asymmetric optimizer state reset — after expansion, the widened layers get a fresh optimizer state while unchanged layers keep theirs. The "asymmetric reset" principle generalises: after a change to a layer's numerical format (e.g. from int4 to fp16), the optimizer state for that layer should be reset asymmetrically while unmodified layers keep theirs.
- **GradLite** (arXiv, Oct 2025): low-rank Jacobian approximation with error-feedback correction. For quantized layers, the dequantization error can be fed back into the next step's gradient, correcting the quantization drift without full-precision retraining.

### 4.3 Design

Introduce a quantization configuration per layer block (not per individual layer, to keep config size manageable):

```
QuantConfig {
  precision: Int4 | Int8 | Fp16,
  block_size: 32 | 64 | 128,  // for groupwise quantization
  reset_optimizer_on_change: bool,  // SPARKLING asymmetric reset
  error_feedback: bool,  // GradLite correction on this layer
}
```

The configuration is determined by a profiling pass that runs for 10–50 steps at the start of training and measures each layer's activation variance and gradient magnitude. Layers with high variance or large gradients stay at fp16; layers with low variance drop to int8 or int4.

The asymmetric reset flag means: when a layer's precision changes (either up or down), its optimizer state (m, v, or GRUM buffers) is zeroed. All other layers keep their state. This prevents stale momentum from corrupting the new numerical regime.

### 4.4 Implementation Plan

```
Phase 3a: Per-block quant config
  - Extend fused_dequant_gemm to accept a block-level quantization descriptor
  - Add quant_config profiling pass (10–50 warmup steps, measure per-layer variance)
  - Serialize config to .grim.train sidecar alongside TrainState

Phase 3b: Asymmetric reset + error feedback
  - When a layer's quant config changes between training runs, zero optimizer state
    for that layer's param IDs (in grim-autograd)
  - GradLite feedback: during fused_dequant_gemm, the dequantization residual
    (x - dequant(quant(x))) is stored per group; next forward pass adds it to the
    activation before the GEMM, compensating for the quantization error
  - This residual is only kept for the current microbatch, not accumulated
```

### 4.5 Quality Gates (CQO)

- Training with per-layer quant + asymmetric reset must achieve ≥ 95% of full-fp16 validation perplexity at 40% memory reduction (measured as peak VRAM during forward-backward)
- GradLite error feedback must not regress convergence stability (monitor gradient norm variance)
- Property test: switching a layer from int4 to fp16 mid-training must recover to within 1e-3 of the fp16-only loss within 100 steps

---

## 5. Level 4: GRUM-Optimizer

### 5.1 Problem

The current AdamW in `grim-autograd/src/adamw.rs` stores two moment buffers (m, v) per parameter, runs on CPU as `Vec<f32>`, and requires a round-trip through `to_vec_f32()` / `from_vec_f32()` every step. For a 7B LoRA with rank=16, the adapter parameters are ~30M floats — 120MB of momentum on CPU, transferred across PCIe every step.

### 5.2 Synthesis from Papers

GRUM-Optimizer combines four research lines into a single fused GPU kernel:

| Component | Source Paper | What It Contributes |
|-----------|-------------|-------------------|
| **COSMOS hybrid decomposition** | COSMOS (ICLR 2026) | Split update into leading-eigensubspace (SOAP-like) and residual (MUON-like). Apply nested momentum to the leading subspace, lightweight update to the residual. |
| **RMNP row-norm preconditioning** | RMNP (arXiv May 2026) | Replace the Newton-Schulz iteration in the MUON path with row-wise ℓ2 normalization. 43× cheaper per step. |
| **HTMuon spectral correction** | HTMuon (ACL 2026) | Raise the momentum matrix's singular values to p=0.125 before the orthogonalization step. This preserves heavy-tailed spectra. |
| **AdamN nested momentum** | AdamN (MDPI Feb 2026) | Compound EMA: `n = β₂·n + (1-β₂)·m`, then use `m · n` as the update direction. Does not require a separate normalization step. 2.25× faster time-to-quality on Llama 3.1-8B. |
| **Schedule-Free stabilisation** | Through the River (NeurIPS 2025) | No LR schedule. The AdamN nested momentum structure naturally provides the weight-averaging effect that Schedule-Free achieves implicitly. |

### 5.3 Design

GRUM-Optimizer keeps **three** buffers per parameter (growing from 2 for AdamW), all GPU-side, updated within a single fused kernel:

```
Buffer representation:
  m:  Nesterov momentum (one EMA)
  n:  nested momentum (EMA of m)
  s:  spectral correction factor (p=0.125 on running svd estimate)

Per-step update (all fused in GPU registers, one kernel per param):
  1. g = load_gradient(param_id)
  2. g = g + error_feedback(g_dequant_residual)    // Level 3 cross-talk
  3. m = β₁·m + (1-β₁)·g                            // Nesterov momentum
  4. n = β₂·n + (1-β₂)·m                            // AdamN nested momentum
  5. cosmo_split(m, leading_rank=k) where k = min(64, param_dim)
      5a. leading subspace: apply nested momentum update (m ⊙ n)
      5b. residual: apply RMNP row-norm (row / ||row||₂)
  6. spectral_correct: s = s * p + (1-p) * svd_estimate(m)   // HTMuon p=0.125
  7. update = (leading_update + residual_update) / (1 + s)
  8. param = param - lr * update - weight_decay * param
```

k in step 5 is fixed at 64 (per COSMOS: the leading subspace saturates at ~64 dimensions for transformer parameters up to 1.5B). This means the SVD implicit in step 5a is a 64×64 matrix — a trivial computation on GPU, done entirely in registers.

The fusion into a single kernel avoids:
- Three separate Buffer reads/writes (m, n, s) that would be required if decomposed into separate HIP launches
- The CPU round-trip that the current AdamW makes
- Redundant memory loads: gradient, param, m, n, and s are all loaded once

### 5.4 Implementation Plan

```
Phase 4a: GPU-side optimizer kernel
  - New kernel module: grim-backend-rocm/src/kernels/grum_optimizer.rs
  - Kernel signature: grim_grum_optimizer(param, grad, m, n, s, lr, beta1, beta2, p, k, weight_decay, numel)
  - Compiled via hipRTC alongside existing kernels
  - Host launcher in grim-autograd dispatches to RocmDevice via BackendDevice trait
  - Tests: numerical parity with CPU AdamW for the same gradient sequence

Phase 4b: COSMOS decomposition
  - Add leading-subspace extraction (64-dim SVD via hipSOLVER or hand-rolled power iteration)
  - The SVD runs once every T steps (default T=10) — the leading subspace changes slowly
  - Tests: verify that the k=64 leading subspace captures ≥ 90% of the gradient covariance
    (measured via frobenius norm ratio)

Phase 4c: HTMuon spectral correction
  - Maintain running estimate of largest singular value of m (power iteration, 5 steps per checkpoint)
  - Raise momentum to p=0.125 per HTMuon: s_hat = log(s) * p; s = exp(s_hat)
  - Apply as update /= (1 + s) — COSMOS + HTMuon combined scale

Phase 4d: Schedule-Free integration
  - Drop LR schedule entirely (per Schedule-Free findings)
  - AdamN's nested momentum already provides the weight-averaging effect.
    No additional logic needed — this phase is deleting code from the host launcher.
  - Tests: verify convergence is monotonic without LR schedule
```

### 5.5 Quality Gates (CQO)

- Per-step wall time of GRUM-Optimizer kernel ≤ 1.5× the current CPU AdamW + transfer time (measured on same GPU for same param count)
- Memory: 3 buffers × f32 vs 2 buffers × f32 (50% increase in optimizer state), but the current 120MB CPU-side allocation moves entirely to GPU. Peak VRAM increase is bounded at 3/2 × old GPU alloc + freed CPU alloc.
- Convergence: on a LoRA fine-tuning task (rank=16 on Llama 3.2-1B or equivalent), GRUM-Optimizer must reach the AdamW validation loss target in ≤ 60% of steps (bounding: AdamN's 2.25× claim, de-risked to 1.67×)
- **Strong-test constraint** (from strong-tests skill): property test over random gradient sequences — GRUM-Optimizer must not diverge for any sequence of L2-bounded gradients (test over 100 random seeds, 1000 steps each, gradient norm ≤ 10.0)

---

## 6. Dependency Order and Execution Plan

```
Level 1 (Tile Decomp) ──→ Level 2 (Fusion) ──→ Level 3 (Quant) ──→ Level 4 (Optimizer)
       |                       |                     |                    |
       v                       v                     v                    v
  Phase 1a (2d)           Phase 2a (3d)          Phase 3a (5d)        Phase 4a (5d)
  Phase 1b (2d)           Phase 2b (3d)          Phase 3b (5d)        Phase 4b (5d)
                                                                       Phase 4c (3d)
                                                                       Phase 4d (1d)
```

Total estimated engineering time: **34 person-days** for all phases (ponytail estimate — one senior Rust/ROCm engineer, no external dependencies beyond what grim already uses).

### Ponytail cuts (if time-constrained)

| If you have only | Ship these phases | Defer |
|-----------------|-------------------|-------|
| 10 days | Level 1 + Level 4a | Level 2, 3, 4bcd — the GPU-side optimizer kernel alone is the highest-impact single change |
| 20 days | Level 1 + Level 4 (full) | Level 2, 3 — fused optimizer achieves most of the training speedup; tile decomposition covers GEMM |
| 34 days (full) | All 4 levels | — |

---

## 7. Testing Strategy (per tdd + strong-tests)

Every phase has three test layers, run in CI:

**Layer 1 — Unit tests (Phase n before merge)**
- Parsing/validation tests for config structs
- Kernel source string contains the correct entry points (pattern established in `decode_gemm.rs`)

**Layer 2 — Numerical parity tests (Phase n must pass before next phase starts)**
- Each kernel variant produces outputs within tolerance of a CPU f32 reference
- Property tests over the shape space (M, N, K) using proptest/quickcheck
- The GRUM optimizer is tested against a Python reference implementation of the same algorithm (shipped as `tests/grum_reference.py`)

**Layer 3 — Integration benchmarks (Phase n must pass before merging)**
- Microbenchmark each kernel vs the pre-GRUM baseline
- Record to `docs/benchmarks/grum-{phase}-{date}.csv` for audit trail
- Any regression > 5% in throughput or > 2× in P99 latency blocks the PR

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RDNA2 (gfx1036) has no WMMA, so Level 1 only benefits gfx1100+ | High | Medium | Level 4 (optimizer) is hardware-agnostic and delivers the main training speedup. Level 1's fallback path preserves existing perf. |
| hipRTC compiles large kernels slowly for Level 2 fusion | Medium | Low | JIT cache already exists (`jit_cache.rs`). Pre-compile fused variants in `build.rs` using offline `hipcc`. |
| GRUM-Optimizer's 3 buffers increase total VRAM vs AdamW's 2 | Low | Medium | But the CPU→GPU migration frees 120MB of pinned host memory. Net system memory decreases. |
| SPARKLING asymmetric reset interacts badly with GradLite error feedback | Low | High | Phase 3b includes a property test: reset must not amplify error feedback. If interaction is negative, ship quantization without error feedback. |
| Level 2 fusion regresses on long sequences where separate kernels allow better overlap | Medium | Medium | Nemotron-Flash found that fusion wins on decode but can lose on prefill > 4096. Gate fusion on `seq_len < threshold`. |

---

## 9. What We Are Not Doing

- **Level 3 / SPARKLING width-progressive training.** Width expansion changes model architecture mid-training. Grim's focus is inference + LoRA fine-tuning, not pre-training. Only the asymmetric reset + error feedback mechanisms are adopted.
- **Cubecl port.** The existing `docs/spec-hip-kernels-cubecl-port-07-12-2026.md` covers a separate effort to port kernels to cubecl. GRUM kernels are written as hipRTC source strings (existing pattern). If cubecl lands first, the GRUM kernels can be ported later with no algorithmic change.
- **Distributed training.** Grim is single-node. This spec assumes one GPU. Multi-GPU (RCCL) is a separate effort.
- **FP8.** Not available on gfx1036. Gated on gfx1200+ per existing convention.

---

## 10. Acceptance Criteria

The spec is done when:
1. Each phase passes its quality gates (Section 2.5 / 3.5 / 4.5 / 5.5)
2. A LoRA fine-tuning run using Level 4 (GRUM-Optimizer) converges to target perplexity in ≤ 60% of the AdamW steps
3. Throughput on gfx1100+ hardware for decode and prefill improves by ≥ 10% from Level 1 + Level 2 combined
4. Peak VRAM for a 7B-scale LoRA training run does not exceed current baseline by more than 10% (Level 4's 3rd buffer is offset by quant savings from Level 3)
