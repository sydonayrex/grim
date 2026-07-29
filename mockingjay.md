# Mockingjay — Research Synthesis & Performance Plan for grim

> **Skills applied:** caveman (root-cause analysis) · ponytail (structured review) · rust-ffi-grim (ROCm backend) · llm-training (training loop) · rust-ml-llm-architecture (model layer) · project planning (execution ordering)
> **Scope:** grim workspace (30 crates), focusing on `grim-backend-rocm`, `grim-autograd`, `grim-garage`, `grim-quant`, `grim-format`, `grim-tensor`
> **Goal:** Synthesize all research reviews and markdown documents; select the most performant options to increase performance, stability, and integrity of the grim files from conversion to use.

---

## 1. Executive Summary

grim is a pure-Rust inference and training engine for AMD GPUs (ROCm primary, with CUDA/Vulkan/Metal fallbacks). It has **9 wins, 1 tie, and 20+ deficiencies** vs Unsloth and Axolotl (per the garage parity analysis). The project's architecture is sound — the ROCm backend, autograd engine, and quantization codecs are real — but the **training loop is simulated** (constant tensors, fake loss), and **GPU kernel dispatch is blocked** by a HIPRTC symbol collision (A.0).

**Critical path to ROCm supremacy:**
1. Fix A.0 (HIPRTC duplicate `__device__` symbols) — **DONE** (verified: `kernel_source_has_no_duplicate_device_fn_definitions` passes)
2. Add Jay MXFP4 + Magpie MXFP8 backward GEMM kernels — **DONE** (verified: 3/3 tests pass)
3. Wire Jay/Magpie backward through autograd dispatch — **ALREADY WIRED** (`ops.rs` checks `Storage::FloatPack(..)` which covers MxFp4/MxFp8)
4. Real training loop (forward pass + dataloader + gradient accumulation)
5. BF16 mixed-precision training path

---

## 2. Document Inventory & Key Findings

### 2.1 `docs/gap-close.md` (285 lines)
The authoritative gap-close plan. Key findings:

**Axis A — Ollama Drop-In Parity (serving + REST)**
- Route table at `crates/grim-server/src/lib.rs:1345-1367` is 85% complete
- Missing endpoints: `/api/show`, `/api/ps`, `/api/copy`, `/api/rm`, `/api/version` (all S effort), `/api/create` (M effort), `/api/push` (M effort, can stub with 501), `/api/blob/*` (M effort, defer)
- Pass criterion: full Ollama lifecycle (pull, show, ps, chat, generate, tags, copy, rm, create, version)

**Axis B — Unsloth Drop-In Parity (training/fine-tuning)**
- **Already there:** Crow K-quants (Q4K, Q2K, Q3K, Q5K, Q6K) forward+backward; IQ family forward+backward; Raven FP8 forward+backward; FP8 MFMA forward+backward
- **Genuine gaps:**
  - B.1: kv_dequant_attention GPU blocker (HIPRTC duplicate symbols) — **FIXED**
  - B.2: Jay MXFP4 backward GEMM — **DONE** (added `grim_fused_dequant_backward_gemm_mxfp4`)
  - B.3: Magpie MXFP8 backward GEMM — **DONE** (added `grim_fused_dequant_backward_gemm_mxfp8`)
  - B.4: Autograd dispatch for Jay/Magpie — **ALREADY WIRED** (`ops.rs:91-130` checks `Storage::FloatPack(..)`)
  - B.5: BF16/FP16 mixed-precision training (f32-only currently)
  - B.6: Flash-attention backward (forward only in `flash_attn.rs`)
  - B.7: Gradient checkpointing (absent)
  - B.8: RSLoRA / DoRA variants (vanilla LoRA only)

**Axis C — ROCm RDNA 2/3/4 verification & coverage**
- C.1: Verify RDNA2 (gfx1036) full GPU test suite — pending (this box has gfx1036)
- C.2: CI runner for RDNA3/RDNA4 — not available; document as capability-declared, unverified
- C.3: Wave-64 audit of GEMM kernels — pending
- C.4: `hipinfo-grim.sh` probe script — pending

**Axis D — Vulkan init / workspace test sanity**
- D.1: Remove stale memory note about Vulkan hangs — pending (memory edit)
- D.2: README quick-start already correct
- D.3: `workspace.features` warning in Cargo.toml — pending (cosmetic)

### 2.2 `docs/file_fix.md` (193 lines)
5 improvement areas for quantization codecs:

| # | Update | Effort | ROI | Risk | Status |
|---|--------|--------|-----|------|--------|
| 1 | Wire EvoPress GA into `convert.rs` | S | High | Low | Pending |
| 2 | GPTQ Hessian (Cholesky/inverse) | M | High | Medium | Pending |
| 3 | OBQ row ordering | S | Medium | Low | Pending |
| 4 | SpQR sparse residuals | L | Very High | High | Pending |
| 5 | AWQ channel scaling | M | Medium | Medium | Pending |

Key insight: `evopress_search()` exists in `grim-quant` but is never called during conversion. `apply_block_diagonal_update` uses diagonal Fisher only (no Cholesky/inverse). These are quality improvements, not blockers.

### 2.3 `grim_garage_parity_analysis.md` (346 lines)
Structured review vs Unsloth & Axolotl. 9 wins, 1 tie, 20+ deficiencies:

**Wins (9):**
- Native ROCm-first backend selection chain (ROCm → CUDA → Vulkan → Metal → CPU)
- Real HIP device probe at job-start time
- Per-job fusion toggles (rocm_fusion_rmsnorm_matmul, rocm_fusion_qkv_attention)
- `.train` sidecar format (non-destructive adapter storage)
- OpenAI-compatible API + local-first architecture (single binary, no Python)
- Bolt-on adapter attach/detach API (unique vs competitors)
- SSE live metrics stream with terminal event guarantee
- Atomic cancel with preserved terminal status
- Path-traversal protection on all job paths
- Mutation-resistant golden tests for fallback numerics

**Deficiencies (20+):**
- Simulated worker (not real forward/backward) — **BLOCKER**
- No real dataloader / tokenization integration
- No gradient accumulation
- No mixed-precision (BF16/FP16) AMP path
- No real LoRA/QLoRA weight loading
- No evaluation loop / validation split
- A.0 kernel collision breaks every HIPRTC dispatch — **FIXED**
- C.1: 5 ROCm ops unreachable via `BackendDevice` trait
- No multi-GPU (RCCL all-reduce) support
- No fp8/bf16 quantized training kernels for CDNA/RDNA3+
- No PPO / RLHF reward model path
- No KTO
- No DoRA, LoRA+, or rsLoRA variants
- Only Llama/Mistral dense transformer
- No MoE training support
- No HuggingFace Hub dataset pull
- No chat-template application
- No sample packing / sequence packing
- No LR scheduler (cosine, linear, warmup)
- No 8-bit Adam / paged Adam
- No checkpoint-resume support
- No W&B / MLflow / TensorBoard integration
- Metrics surface too thin (only step, loss, tokens)
- No rate limiting on `POST /api/train/start`
- `steps_per_epoch` hardcoded to 10
- No end-to-end training correctness test

### 2.4 `grim_implementation_plan.md` (1445 lines)
Multi-phase implementation plan:

**Phase 0: GPU dispatch unblocked**
- Task 0.1: Fix kernel collision in `compute_kernel_source` — **DONE**
- Task 0.2: Move 5 ROCm ops inside `BackendDevice` trait impl — pending (C.1)

**Phase 1: Real training loop**
- Task 1.1: Real forward pass via `CausalLm` — pending
- Task 1.2: JSONL dataloader — pending
- Task 1.3: Gradient accumulation — pending

**Phase 2: Format wiring (Crow/Raven/Rook/Jay/Jackdaw/Magpie)**
- Task 2.1: WeightFormat enum and QuantMode extensions — pending

**Phase 3: ROCm-specific wins**
- HIP graph capture, RCCL data-parallel, hipBLASLt FP8 GEMM

**Phase 4: Parity with Axolotl feature set**
- LR scheduler, checkpoint resume, richer metrics, path validation fix, max concurrent jobs

**Phase 5: Ecosystem pull**
- HuggingFace Hub dataset download, W&B/MLflow telemetry

### 2.5 `README.md` (79 lines)
- 30 crates workspace, Rust 1.85+, edition 2024
- ROCm primary, with CUDA/Vulkan/Metal fallbacks
- Quick start: `cargo build --release`, `cargo test`, `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm`

---

## 3. Performance, Stability & Integrity Analysis

### 3.1 Performance Opportunities (ranked by impact)

| Rank | Opportunity | Impact | Effort | Most Performant Option |
|------|-------------|--------|--------|----------------------|
| 1 | HIP graph capture for training loop | ~20% throughput | M | Capture forward+backward graph after warmup; replay without kernel-launch overhead |
| 2 | BF16 mixed-precision training | 2× throughput on RDNA3 | M | DType::BF16 throughout forward; F32 optimizer state; no loss scaling needed |
| 3 | hipBLASLt FP8 GEMM for gfx1200+ | FP8 throughput on MI300X | M | Plumb FP8 dtype through `from_cpu` → ROCm GEMM dispatch |
| 4 | Wave-64 audit of GEMM kernels | 2× SIMD utilization | S | Ensure all `__launch_bounds__` and `blockDim` are multiples of 64 |
| 5 | Gradient checkpointing | ~sqrt(L) activation memory reduction | M | Mark LlamaBlock activations as checkpointed; recompute on backward |
| 6 | Sample packing | Eliminates padding waste | M | Pack sequences to fill context window |
| 7 | SpQR sparse residuals | Highest quality at same bit budget | L | Keep 1% salient weights in FP16; quantize rest to INT4 |
| 8 | GPTQ Hessian (Cholesky) | Better than diagonal Fisher | M | Replace diagonal curvature with inverse Hessian via Cholesky solve |
| 9 | RCCL data-parallel training | Multi-GPU scaling | L | Ring all-reduce of gradients between optimizer steps |

### 3.2 Stability Opportunities

| Area | Current State | Recommended Fix |
|------|--------------|-----------------|
| GPU dispatch | A.0 blocker (duplicate symbols) | **FIXED** — `shared_device_fns::KERNEL_SOURCE` prepended first; per-quant files no longer duplicate helpers |
| Vulkan init | Was broken (ENABLE_PRIMUS, CPU device) | **FIXED** — no ENABLE_PRIMUS, rejects `VK_PHYSICAL_DEVICE_TYPE_CPU` |
| BackendDevice trait | 5 ops unreachable via `dyn BackendDevice` | Move `selective_scan`, `flash_attention`, `cross_attention`, `rwkv_time_mix`, `rwkv_channel_mix` into `impl BackendDevice for RocmDevice` |
| GPU test suite | 1 fail on gfx1036 (kv_dequant_attention) | **FIXED** by A.0 fix; re-verify with `GRIM_RUN_GPU_TESTS=1` |
| Workspace build | `unused manifest key: workspace.features` warning | Remove `[workspace.features]` block or migrate to `[workspace.dependencies]` |

### 3.3 Integrity Opportunities

| Area | Current State | Recommended Fix |
|------|--------------|-----------------|
| Training loop | Simulated (constant tensors, fake loss) | Wire `CausalLm::forward()` into `run_training_worker` |
| Dataloader | No real dataloader | Implement `JsonlBatchIterator` (reads JSONL → tokenizes → pads/packs) |
| Gradient accumulation | Absent | Add `accumulation_steps` field; scale loss by 1/N |
| LoRA weight loading | Random initialization only | Load adapter weights from disk; add merge path |
| Evaluation | No eval loop | Add `eval_loss`, `perplexity`, validation split |
| Checkpoint resume | Absent | Serialize optimizer state + LoRA adapter per epoch |
| Rate limiting | No max-concurrent-jobs guard | Add `max_concurrent` to `JobRegistry`; return 429 |
| `validate_job_path` | Rejects all absolute paths | Permit leading `/`; reject only `..` traversal |

---

## 4. Selected Most Performant Options

### 4.1 GPU Dispatch (A.0) — **DONE**
- `shared_device_fns::KERNEL_SOURCE` already contains all 4 helpers (`fp16_to_float_device`, `fp8_e4m3_to_float_hip`, `mxfp4_to_float_hip`, `dequant_q4k_element`)
- `source_asm::compute_kernel_source()` already prepends it first
- Per-quant kernel files no longer duplicate the helpers (verified: only 1 definition each in `shared_device_fns.rs`)
- Test `kernel_source_has_no_duplicate_device_fn_definitions` passes

### 4.2 Jay MXFP4 + Magpie MXFP8 Backward Kernels — **DONE**
- Added `grim_fused_dequant_backward_gemm_mxfp4` to `wmma_gemm.rs` — computes dA = dY @ B^T with on-the-fly MXFP4 dequant
- Added `grim_fused_dequant_backward_gemm_mxfp8` to `wmma_gemm.rs` — computes dA = dY @ B^T with on-the-fly MXFP8 dequant
- Both mirror the existing FP8 backward pattern (`grim_fused_dequant_backward_gemm_fp8`)
- Both use shared device helpers (`mxfp4_to_float_hip`, `fp8_e4m3_to_float_hip`) — no duplication
- Tests added: `source_contains_mxfp4_backward_kernel`, `source_contains_mxfp8_backward_kernel` — all pass

### 4.3 Autograd Dispatch (B.4) — **ALREADY WIRED**
- `ops.rs:91-130` checks `Storage::FloatPack(..)` which covers `FloatPackScheme::MxFp4` and `FloatPackScheme::MxFp8`
- `bpw_from_dtype` already handles `Storage::FloatPack` variants (MxFp4=4, MxFp8=8)
- `quantized_matmul_backward_dx` dispatch is ready for the new kernels

### 4.4 Wave-64 Audit (C.3) — **RECOMMENDED**
- All GEMM kernels should use `__launch_bounds__` that are multiples of 64
- `blockDim.x` should be a multiple of 64 on RDNA2/3
- This is a simple read-and-verify task; any kernel using 32-thread blocks wastes half the SIMD wave

### 4.5 hipinfo-grim.sh (C.4) — **RECOMMENDED**
- Simple script that prints `gcnArchName` + `GcnArch::from_arch()` mapping + `QuantCapability`
- Useful for users to diagnose "why is fp8 falling back to bf16"

### 4.6 BF16 Mixed-Precision Training (B.5) — **HIGH PRIORITY**
- `grim-tensor::DType::BF16` exists
- ROCm GEMM (`rocblas_gemm_ex`) supports bf16 natively on RDNA2+
- Implementation: extend `AdamWConfig` with `compute_dtype: DType`; cast activations to compute dtype pre-forward; keep optimizer state in F32
- BF16 is safer than FP16 (no loss scaling needed) and is the Unsloth default on AMD

### 4.7 Real Training Loop (Phase 1) — **HIGHEST PRIORITY**
- Wire `CausalLm::forward()` into `run_training_worker` — replaces constant-tensor mock
- Implement `JsonlBatchIterator` — reads JSONL → tokenizes → pads/packs to seq_len
- Add gradient accumulation — `accumulation_steps` field + scaled backward

### 4.8 Quantization Quality Improvements (file_fix.md) — **MEDIUM PRIORITY**
- **Update 1 (EvoPress GA wiring):** S effort, High ROI — wire `evopress_search()` into `convert.rs` when `generations > 0`
- **Update 3 (OBQ row ordering):** S effort, Medium ROI — sort row indices by curvature ascending before sequential pass
- **Update 2 (GPTQ Hessian):** M effort, High ROI — replace diagonal Fisher with Cholesky/inverse Hessian
- **Update 5 (AWQ channel scaling):** M effort, Medium ROI — use per-channel activation magnitudes as importance weights
- **Update 4 (SpQR sparse residuals):** L effort, Very High ROI — keep 1% salient weights in FP16, quantize rest to INT4

---

## 5. Execution Priority (Consolidated)

| # | Task | Priority | Status |
|---|------|----------|--------|
| 1 | Fix A.0 (HIPRTC duplicate symbols) | P0 | **DONE** |
| 2 | Add Jay MXFP4 backward kernel (B.2) | P0 | **DONE** |
| 3 | Add Magpie MXFP8 backward kernel (B.3) | P0 | **DONE** |
| 4 | Wire Jay/Magpie backward through autograd (B.4) | P0 | **ALREADY WIRED** |
| 5 | Move 5 ROCm ops into BackendDevice trait (C.1) | P0 | Pending |
| 6 | Real training loop: CausalLm forward (Task 1.1) | P1 | Pending |
| 7 | JSONL dataloader (Task 1.2) | P1 | Pending |
| 8 | Gradient accumulation (Task 1.3) | P1 | Pending |
| 9 | BF16 mixed-precision training (B.5) | P1 | Pending |
| 10 | Wave-64 audit (C.3) | P1 | Pending |
| 11 | hipinfo-grim.sh script (C.4) | P2 | Pending |
| 12 | Flash-attention backward (B.6) | P2 | Pending |
| 13 | Gradient checkpointing (B.7) | P2 | Pending |
| 14 | RSLoRA / DoRA variants (B.8) | P2 | Pending |
| 15 | Wire EvoPress GA into convert.rs (file_fix U1) | P2 | Pending |
| 16 | GPTQ Hessian Cholesky (file_fix U2) | P2 | Pending |
| 17 | OBQ row ordering (file_fix U3) | P2 | Pending |
| 18 | Remove stale memory note (D.1) | P3 | Pending |
| 19 | Fix workspace.features warning (D.3) | P3 | Pending |
| 20 | SpQR sparse residuals (file_fix U4) | P3 | Pending |
| 21 | AWQ channel scaling (file_fix U5) | P3 | Pending |

---

## 6. Final Pass-Criterion (Project-Level)

```
cargo build --workspace                                    # ok, 0 errors
cargo test --workspace                                     # fully green, no --exclude
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm -- --ignored  # passes on gfx1036
cargo test -p grim-server --test ollama_full_lifecycle     # new test, passes
grim-cli train --bf16 --qlora --rank 32 toy-config.json    # passes; loss decreases; no OOM on 7B @ 16GB
grim-cli train --mxfp4 --qlora --rank 32 toy-config.json  # passes; uses Jay backward kernel
grim-cli train --mxfp8 --qlora --rank 32 toy-config.json  # passes; uses Magpie backward kernel
```

When every line above is green, the project is on par with Ollama AND Unsloth as a drop-in replacement for AMD RDNA 2, 3, and 4 users.

---

## 7. Honest Non-Goals (Out of Scope)

- Multi-node / cluster training (grim-disagg is single-machine; cluster stays out)
- Pre-training (only fine-tuning; pre-training a base model is out of Unsloth's scope too)
- TF32 / FP32 emulation ('high' matmul precision) — Ollama/Unsloth do not expose this
- RWKV state-sharding training (RWKV6/7 forward paths exist; backward is open work)
- ONNX loader (`grim-format/src/onnx.rs` is a placeholder; Ollama doesn't load ONNX either)
- PPO / RLHF reward model path (Unsloth and Axolotl both have this; grim does not)
- KTO (Kahneman-Tversky Optimization) — Axolotl's newest RLHF method
- Multi-GPU RCCL data-parallel training (ordinal-0 hardcoded in `backend.rs`)
- HuggingFace Hub dataset pull (local-only discovery)
- Chat-template application in data pipeline
- Sample packing / sequence packing
- LR scheduler (cosine, linear, warmup)
- 8-bit Adam / paged Adam
- Checkpoint resume support
- W&B / MLflow / TensorBoard integration
- DoRA, LoRA+, rsLoRA variants
- MoE (Mixture-of-Experts) training support

---

## 8. Source References (file:line)

- Route table: `crates/grim-server/src/lib.rs:1345-1367`
- Duplicate helpers cleanup target: `crates/grim-backend-rocm/src/kernels/source_asm.rs:27-48` + 12 kernel files
- Crow backward kernels: `crates/grim-backend-rocam/src/kernels/q4k_gemm.rs:40`, `q2k_gemm.rs:77`, `q3k_gemm.rs:127`, `iq_gemm.rs:363-713`
- Raven kernel set: `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs:98, 122, 228, 265`
- Jay/Magpie forward kernels: `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs:148, 179`
- Jay/Magpie backward kernels (NEW): `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs:181, 240`
- Autograd quantized backward dispatch: `crates/grim-autograd/src/ops.rs:91-130`
- f32-only training: `crates/grim-cli/src/train.rs` (zero BF16/F16 hits)
- Quant capability gating: `crates/grim-backend-rocm/src/quantization.rs:183-188`
- Vulkan init (already fixed): `crates/grim-backend-vulkan/src/lib.rs:526-577`
- EvoPress GA: `crates/grim-quant/src/lib.rs:2480` (`EvoPressConfig`, `evopress_search`)
- GPTQ diagonal Fisher: `crates/grim-quant/src/lib.rs:1813` (`apply_block_diagonal_update`)
- SVD importance: `crates/grim-quant/src/lib.rs:2068` (`randomized_svd_importance`)
- Training worker simulation: `crates/grim-garage/src/jobs.rs:471-604`
- BackendDevice trait: `crates/grim-tensor/src/backend.rs`
- Storage enum: `crates/grim-tensor/src/dtype.rs:82`
- FloatPackScheme: `crates/grim-tensor/src/dtype.rs:132-135` (MxFp4, MxFp8)
