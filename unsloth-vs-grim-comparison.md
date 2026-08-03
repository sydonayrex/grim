# Unsloth vs. Grim — Training Capability Comparison (2026-08-02)

**Purpose:** Compare Unsloth v2026.7.6 (Python/PyTorch/Triton) with Grim (Rust workspace, 26 crates, ROCm-primary) for LLM training from the perspective of model training. Covers architecture, custom kernels, backends, optimizers, LoRA variants, quantization, memory optimization, multi-GPU, preference optimization, and FFI/ROCm/HIP specifics.

**Sources:** `old/repos/unsloth-main/` (Unsloth), `/D/rex/projects/grim/` (Grim), `unsloth-vs-grim-comparison.md` (previous baseline).

---

## 0. Executive Summary

| Dimension | Parity | Grim Better | Unsloth Better | Neither Well |
|---|---|---|---|---|
| Architecture model | QLoRA (frozen quantized base + LoRA) | Scoped LoRA-only autograd tape; device-resident FP32 AdamW; Rust compile-time safety; zero-Python overhead | PyTorch ecosystem integration; torch.compile; dynamic graph | — |
| Custom kernels | Fused CE, SwiGLU, RMSNorm, RoPE, LoRA | HIP kernel JIT (HSACO cache); 26 kernel modules; fused dequant-gemm fwd+bwd | Triton JIT kernels; per-kernel autotuning; `@torch.autograd.Function` backward; torch.compile fusion | Fine-grained per-element op fusion for inference-only paths |
| Backends | ROCm (primary on both), CUDA, Metal, Vulkan, MLX | Primary ROCm/HIP with full trait dispatch & JIT; Vulkan secondary; CPU fallback always | CUDA (primary, mature); MLX (torch-free, Apple Silicon native); ROCm supported; x86 CPU with torch fallback | Intel XPU; single-GPU Metal/Vulkan; cross-backend unified API |
| Optimizers | AdamW, AdamW8Bit, QGaLore | 14 `OptimizerKind` variants; 7 implemented (AdamW, AdamW8Bit, PagedAdamW, Lion, Lion8Bit, Adafactor, QGaLoreAdamW8Bit with full Halko SVD + 8-bit moments); device-resident steps; RCCL/CommFuse all-reduce | QGaLoreAdamW8bit (bitsandbytes 8-bit + INT4 weight projection quantization); LOMO, Adalomo, CAME, Sophia via bitsandbytes | Grim lacks LOMO/Adalomo/CAME/Sophia; Unsloth lacks native Rust performance |
| LoRA variants | Standard LoRA (rank/α/scaling) | DoRA, RSLoRA, PiSSA, VeRA, LoftQ, SoulEater (7 variants in injection.rs); 7 injection points per layer; QLoRA-only scope (AutogradScope::LoRAOnly) | Full PEFT (LoraConfig, prompt tuning, p-tuning, IA³, AdaLoRA); full-parameter finetuning | Grim lacks full-parameter; Unsloth lacks some exotic variants |
| Quantization | Q4_K, NF4, Q8_0, Q5_K, Q6_K | NF4 codebook (canonical 16-level QLoRA); Q4_K per-sub-block scales; Q8_0 dequant+gemm fused | bitsandbytes NF4/Q4/Q8 with double quantization; GPTQ; AWQ; SmoothQuant; FP8; INT8 weight quant | Grim's NF4/Q4_K correctness bugs found (now fixed per plan) |
| Memory optimization | Gradient checkpointing, sample packing | Streaming forward (block-by-block, bounded peak); device-resident AdamW moments; zero host round-trips in principle (in practice CPU loops remain) | Padding-free (varlen SDPA); sample packing; CPU offloading of optimizer states; gradient checkpointing with `use_reentrant=True` | Neither has Hopper-era FP8 training at scale; neither has ZeRO-3 full CPU/RAM offload |
| Multi-GPU | Gradient all-reduce, tensor parallelism | RCCL `ncclAllReduce` with device-pointer in-place sum + 1/N averaging; `ScythePlacement` data-parallel topology; CommFuse decomposed P2P (WI-6); SCYTHE-2 C²PLR per-layer routing (research spec) | Standard DDP (DistributedDataParallel); Fully Sharded Data Parallel (FSDP); tensor parallelism via `accelerate`/`transformers` | SCYTHE-2 is spec-only (not compiled); Grim lacks FSDP/ColumnParallel/RowParallel; no ZeRO staging |
| Preference losses | DPO/ORPO/KTO/SimPO/GRPO (parity_loss.rs) | Native Rust implementations; numerically stable softplus | TRL integration (DPOTrainer, ORPOTrainer, KTO, etc.); full RLHF pipeline | Neither has production-grade RL training; neither has full PPO |

---

## 1. Architecture & Training Paradigm

### 1.1 Grim — Scoped LoRA-Only Autograd Tape

**Design philosophy:** Grim uses a reverse-mode autograd tape that records **only the operations touching adapter parameters** during the forward pass. The `AutogradScope` enum (currently `LoRAOnly`) controls what gets tracked.

Key files:
- `crates/grim-autograd/src/lib.rs:61` — `pub mod preference_loss;`
- `crates/grim-autograd/src/tape.rs` — `Tape`/`TapeEntry`/`TapeKind` with ops: `MatMul`, `Add`, `Scale`, `LoRAApply`, `SiluMul`
- `crates/grim-autograd/src/backward.rs` — reverse-mode walk, pops entries from the back
- `crates/grim-autograd/src/param.rs` — `TrainableParam` with gradient accumulator; `frozen` flag for base weights

**Advantages over Unsloth:**
- **Compile-time dead-code elimination.** Only LoRA-related ops are recorded. A 7B model with LoRA has ~14 adapter matrices; the tape holds ~14 entries per layer, not the full computational graph. This is fundamentally more memory-efficient than PyTorch's autograd which traces every tensor op.
- **No dynamic dispatch on the hot path.** The tape is a `Vec<TapeEntry>` — zero virtual calls during backward.
- **Scoped to exactly what trains.** `frozen` params are tracked (for downstream grad routing) but not updated. This matches QLoRA's "frozen quantized base + LoRA adapters" exactly.

**Limitations vs Unsloth:**
- **No eager-mode debugging.** You can't `tensor.backward()` a single op and inspect intermediates like in PyTorch.
- **Reverse-mode only.** Unsloth can do forward-mode or custom `Function.backward()` per op. Grim's tape is monolithic reverse.
- **No dynamic graph.** The graph is statically determined by `AutogradScope::LoRAOnly`. Unsloth can handle arbitrary model topologies via PyTorch's dynamic tracing.

### 1.2 Unsloth — PyTorch Dynamic Graph + Triton Kernels

Unsloth builds on PyTorch's dynamic computation graph, wrapping key operations in `@torch.autograd.Function` with Triton JIT-compiled forward and backward kernels.

Key files:
- `unsloth/models/llama.py` — `LlamaAttention_fast_forward_inference`, `LlamaDecoderLayer_fast_forward` (gradient checkpointing via `torch.utils.checkpoint.checkpoint` with `use_reentrant=True`), `CausalLM_fast_forward` (calls `unsloth_fused_ce_loss` for ≤1024 tokens)
- `unsloth/kernels/cross_entropy_loss.py` — `Fast_CrossEntropyLoss(torch.autograd.Function)` with online logsumexp
- `unsloth/kernels/swiglu.py` — `_DWf_DW_dfg_kernel` (3-output fused backward: `h`, `df`, `de`)

**Advantages over Grim:**
- **Ecosystem compatibility.** Drops into any HuggingFace/PyTorch training loop. `UnslothTrainer` subclasses `SFTTrainer` from TRL.
- **Dynamic graph.** Arbitrary model modifications, custom loss functions, hooks, callbacks.
- **`torch.compile` integration.** Unsloth explicitly `@torch.compiler.disable`s some kernels because they don't compile cleanly — but the rest of the graph can be compiled.

**Comparison at parity:**
Both implementations achieve the same numerical result: RMSNorm → QKV proj → RoPE → attention → residual → MLP (SwiGLU) → residual → LM head → fused CE. Both use online logsumexp. Both implement gradient checkpointing. The **algorithmic math** is identical; the implementation substrate differs (Rust vs Python/Triton).

---

## 2. Custom Kernels

### 2.1 Grim — 26 HIP Kernel Modules (JIT-compiled to HSACO)

Grim's ROCm backend has a kernel source registry in `crates/grim-backend-rocm/src/kernels/`:

| Module | Forward | Backward | Notes |
|---|---|---|---|
| `fused_dequant_gemm.rs` | ✓ fused_dequant_gemm (fwd + bwd) | ✓ (backup1/backup2 residuals) | Fused dequantization + GEMM |
| `kv_dequant_attention.rs` | ✓ | — | Online-softmax GQA causal |
| `qkv_attention.rs` | ✓ | — | Online softmax, GQA causal |
| `flash_attn.rs` | ✓ | — | FlashAttention-style |
| `tree_attention.rs` | ✓ | — | For speculative decoding |
| `mamba/selective_scan.rs` | ✓ | — | Mamba selective scan |
| `rwkv.rs` | ✓ (time_mix/channel_mix) | — | RWKV |
| `comm_fuse.rs` | ✓ | — | CommFuse decomposed P2P |
| `iq_dequant.rs` / `iq_gemm.rs` | ✓ | — | I/Q quantization |
| `q8_0_dequant.rs` / `q8_0_gemm.rs` | ✓ | ✓ | Q8_0 dequant+gemm |
| `q4k_dequant.rs` / `q4k_gemm.rs` | ✓ | ✓ | Q4_K dequant+gemm |
| `q5k_gemm.rs`, `q6k_gemm.rs`, `q2k_gemm.rs`, `q3k_gemm.rs` | ✓ | — | Other quant levels |
| `compute_kernels.rs` | ✓ (silu_mul, add) | — | Basic compute kernels |
| `jit_cache.rs` / `shared_device_fns.rs` | ✓ | — | JIT infrastructure |
| `source_asm.rs`, `fp8_*.rs`, `wmma_gemm.rs` | ✓ | — | FP8, WMMA |

Kernels are JIT-compiled via HIPRTC to HSACO, cached in `HsacoKernelCache`. The `BackendDevice` trait (`crates/grim-tensor/src/backend.rs:117`) dispatches through named kernel lookups.

**Where Grim is better:**
- **Fused dequant-gemm forward AND backward.** `fused_dequant_gemm` supports both forward and backward (with backup1/backup2 residuals). Unsloth's `fast_dequantize` + `matmul_lora` does dequant then separate matmul — no fused backward.
- **KV dequant attention.** Grim has online-softmax attention that dequantizes K/V caches on-the-fly during attention computation. Unsloth relies on SDPA/xformers/FlashAttention external kernels, dequantizing first.
- **Tree attention for speculative decoding.** Grim has `tree_attention` kernel; Unsloth relies on external libraries.
- **Mamba selective scan.** Grim has a native HIP kernel; Unsloth would fall through to `mamba-ssm` or `torch` implementations.

**Where Unsloth is better:**
- **Triton per-kernel autotuning.** Each Triton kernel is independently autotuned (e.g. `calculate_settings` in `kernels/utils.py` picks BLOCK_SIZE and num_warps). Grim uses fixed or JIT-compiled kernels without runtime autotuning.
- **`@torch.autograd.Function` backward.** Each kernel has a custom backward that reuses forward intermediates. Grim's tape approach is more centralized but less per-op granular.
- **Kernel coverage breadth.** Unsloth patches RMSNorm, RoPE, SwiGLU, cross-entropy, LoRA, attention (SDPA/Flash), embedding, and more. Grim has 26 kernel modules but some are stubs (e.g. `q2k`/`q3k` only forward, no backward).

**Where neither does very well:**
- **Fine-grained elementwise op fusion.** Neither fuses every elementwise add/multiply into a single kernel epilogue the way a compiler (e.g. Triton with cooperative autotune or TVM) could. Unsloth uses `torch.compile` as a catch-all; Grim has a CPU fallback loop in many paths.
- **Cross-kernel fusion.** Neither has a unified IR for fusing dequant + GEMM + bias + activation into one kernel for all path combinations.

### 2.2 Kernel Coverage Matrix

| Operation | Grim (ROCm) | Unsloth (Triton) | Notes |
|---|---|---|---|
| RMSNorm fwd + bwd | ✓ HIP kernel | ✓ Triton kernel (`_rms_layernorm_forward`/`_backward`) + Gemma variant | Both have Gemma `(W+1.0)` variant |
| RoPE QK in-place fwd + bwd | ✓ HIP kernel (rope) | ✓ Triton (`_rope_embedding_QK` with `BACKWARD_PASS` negation) | Grim supports `rope_embedding_indices` per-token |
| SwiGLU fwd | ✓ `silu_mul` HIP kernel | ✓ `_fg_kernel` (`silu(e)*g`) | Grim backward is a stub (unimplemented); Unsloth has `_DWf_DW_dfg_kernel` (3-output fused) |
| Cross-entropy (fused linear) | ✓ HIP kernel `grim_cross_entropy_forward`/`_backward` + CPU fallback | ✓ `Fast_CrossEntropyLoss` Triton (online logsumexp) + chunked for >65K vocab | Both have logsumexp max-trick; Unsloth has Gemma softcap + Cohere scaling |
| LoRA fused QKV | ✓ (in ops.rs, CPU) | ✓ `LoRA_QKV`, `LoRA_MLP`, `LoRA_W` Triton kernels | Grim's LoRA ops are CPU-only (ops.rs:72); Unsloth has Triton kernels |
| Attention (GQA causal) | ✓ `qkv_attention` HIP, online softmax | ✓ SDPA / FlashAttention (external kernels) | Grim's is self-rolled HIP; Unsloth delegates to xFormers/FlashAttn |
| Fused dequant-GEMM | ✓ `fused_dequant_gemm` (fwd+bwd) | Partially via `fast_dequantize` + `matmul_lora` | Grim is more fused |
| Paged attention | ✓ `qkv_attention_paged` HIP kernel | PagedAttention (vLLM) via external | Both delegate to external for inference |
| Speculative tree attention | ✓ `tree_attention` HIP kernel | External (Medusa, EAGLE) | Grim has native kernel |
| Mamba selective scan | ✓ `selective_scan` HIP kernel | `mamba-ssm` (external) | Grim native |
| RWKV time/channel mix | ✓ `rwkv` HIP kernel | External / torch | Grim native |
| MoE / MoE gating | — | `moe/` kernel dir (Triton) | Unsloth has native MoE; Grim does not |
| FP8 GEMM (RDNA4) | ✓ `fp8_gemm_rdna4` | External `FbgemmFp8` / `cutlass` | Grim has arch-specific kernel |

---

## 3. Supported Backends

### 3.1 Grim — Rust Backend Trait Dispatch

Grim's `Device` enum (`crates/grim-tensor/src/dtype.rs:9`):

```rust
pub enum Device {
    Cpu,
    Rocm(usize),       // PRIMARY GPU target
    Vulkan,            // platform-agnostic fallback
    Cuda(usize),       // optional
    Metal(usize),      // optional
}
```

The `BackendDevice` trait (`crates/grim-tensor/src/backend.rs:117`) has **default method bodies** — CPU backends get automatic fallback for operations they don't override. The ROCm backend (`grim-backend-rocm/src/device/roc_device.rs:1096`) overrides nearly all methods with HIP kernel launches.

**Backend feature flags in `Cargo.toml`:**
```toml
rocm = ["grim-backend-rocm/rccl"],   # ROCm primary
# CUDA, Metal, Vulkan are secondary (less feature-complete)
```

Each backend is a separate crate:
- `grim-backend-rocm` — 26 kernel modules, RCCL FFI, full trait impl
- `grim-backend-cuda` — partial (GEMM primary)
- `grim-backend-metal` — Metal Shading Language kernels (`kernels.msl`)
- `grim-backend-vulkan` — Vulkan compute shaders (`.comp` files)
- CPU fallback — always available, pure Rust loops

**FFI/ROCm details (per `rust-ffi`, `rocm-ffi-cpp-binding`, `rocm-hip` skills):**
- RCCL FFI in `rccl.rs` uses `#[link(name = "rccl", kind = "dylib")]` with `unsafe extern "C"` blocks for `ncclAllReduce`, `ncclCommInitRank`, `ncclCommInitAll`, `ncclReduceScatter`, `ncclAllGather`, `ncclGroupStart`/`ncclGroupEnd`, `ncclGetUniqueId`, `ncclCommDestroy`.
- rocBLAS FFI uses `#[link(name = "rocblas", kind = "dylib")]` with `rocblas_create_handle`, `rocblas_gemm_ex`, `rocblas_gemm_strided_batched_ex`, `rocblas_set_stream`, `rocblas_sgemm`, `rocblas_status_success`.
- HIP runtime loaded via `libloading` with SONAME fallback chain (`libamdhip64.so.7` → `.6`), symbols fetched by exact mangled name (`hipGetDevicePropertiesR0600`).
- `NcclComm` is `#[repr(transparent)]` newtype over `*mut c_void`, marked `unsafe impl Send + Sync`.
- `rocblas_handle` is `#[repr(transparent)]` newtype over zero-sized `_rocblas_handle`.
- `rocblas_stride` is `i64` (element count, NOT bytes) — matches ROCm 6.x/7.x headers.
- All HIP/rocBLAS calls wrapped with status-code checking (`hip_check` pattern).
- CK/composable-kernel GEMM: `build.rs` hipcc-compiles `ck_gemm.cpp` → `libck_gemm.a`, links `dylib=stdc++`. `GemmHostArgs.index_t=i32` (cast i64). RDNA needs `-DCK_TILE_USE_WMMA`; CDNA uses MFMA (no flag).
- **Rust-2024 `gen` keyword trap:** `gen` is reserved in edition 2024. Old `let mut gen = ...` breaks workspace compile. Rename to `rng` or `gen_`.

### 3.2 Unsloth — PyTorch + Triton + bitsandbytes

Unsloth's `device_type.py:60` detects: CUDA → ROCm/HIP → XPU → MLX. The GPU path requires `torch + triton + bitsandbytes`; MLX path is torch-free.

Dependencies (from `pyproject.toml`):
- Core: `typer`, `rich`, `pydantic`, `structlog`, `click`
- `huggingface` extra: `unsloth_zoo`, `transformers>=4.51.3`, `peft>=0.18.0`, `trl>=0.18.2`, `accelerate`, `datasets`
- `triton` extra: `triton>=3.0.0` (Linux), `triton-windows` (Win)
- `cu118`/`cu121`/`cu124`/`cu126`/`cu128`/`cu130` extras — pre-built xFormers wheels for each CUDA version + torch combo
- `cu118-ampere`/`cu121-ampere` — adds `flash-attn>=2.6.3` for Ampere+ GPUs
- `colab-new` — minimal deps (no xFormers, uses padding-free)
- `intel-gputorch260`/`270`/`280` — XPU (Intel GPU) with `pytorch_triton_xpu`
- `audio-torch{210,290,280}` — torchcodec for Gemma audio

### 3.3 Comparison

**Where Grim is better:**
- **Unified backend trait with safe defaults.** Every backend implements `BackendDevice`; unimplemented methods return `Err(Unimplemented)` with CPU fallback. No Python-level dispatch overhead.
- **Compile-time guarantee.** Rust's type system ensures the trait contract is fulfilled at compile time. Unsloth's runtime `DEVICE_TYPE` checks can fail at runtime.
- **ROCm as a first-class primary** (not afterthought). Grim's `rocm = ["grim-backend-rocm/rccl"]` is the primary feature flag. Unsloth patches ROCm to use CUDA device type internally (`DEVICE_TYPE_TORCH = "cuda"` when `DEVICE_TYPE == "hip"`).
- **HSACO JIT kernel caching.** HIPRTC-compiled kernels are cached in `HsacoKernelCache`. No recompilation on subsequent runs.
- **FFI safety patterns.** `NcclComm`/`rocblas_handle` use `#[repr(transparent)]` newtypes over opaque zero-sized structs. All extern blocks have SAFETY comments. Status-code wrapping prevents silent failures.

**Where Unsloth is better:**
- **Backend breadth.** CUDA is mature and primary. MLX (torch-free Apple Silicon) is a fully parallel code path. x86 CPU works (slow). XPU support for Intel GPUs. Grim's CUDA/Metal/Vulkan are secondary/less complete.
- **Ecosystem integration.** Unsloth integrates with `accelerate` (DDP), `transformers` (model zoo), `huggingface_hub` (model download), `trl` (RLHF), `datasets`. Grim has its own `GgufProvider`/`GgufTokenizer` but also has `download_model` in `grim-core/src/client.rs` supporting `hf:org/repo/file.gguf` URIs, `huggingface.co` URLs, and `hf.co` shortcuts with auto-detection of the best GGUF variant via the HF API.
- **FlashAttention integration.** Unsloth optionally pulls `flash-attn` wheels. Grim has its own `flash_attn.rs` kernel but no integration with the canonical FlashAttention library.
- **Triton on CPU fallback.** Unsloth's Triton kernels gracefully degrade; Grim's CPU fallback is pure Rust loops (correct but slow).

**Where neither does very well:**
- **Vulkan/Metal as primary training targets.** Both treat these as secondary. Grim has Vulkan compute shaders but they're not the focus. Unsloth doesn't support Vulkan/Metal at all (Python Triton → CUDA/ROCm/XPU only; MLX is Apple Silicon only).
- **Cross-backend portability of kernels.** Grim's kernels are CUDA/HIP source strings; Unsloth's are Triton (which targets CUDA/ROCm via `pytorch_triton_xpu`). Neither has a single portable kernel IR that covers all backends.

---

## 4. Optimizers

### 4.1 Grim — 14 `OptimizerKind` Variants (7 Implemented)

From `crates/grim-autograd/src/adamw.rs:154`:

```rust
pub enum OptimizerKind {
    AdamW,              // FP32 moment buffers
    AdamW8Bit,          // 8-bit quantized moments (FP16 storage)
    PagedAdamW,         // Offloads cold moment pages to host RAM
    Lion,               // Sign-based momentum
    Lion8Bit,           // Lion + 8-bit moments
    Adafactor,          // Factored second-moment (memory-efficient)
    AdamWBnb,           // Declared, Unimplemented
    PagedAdamW8Bit,     // Declared, Unimplemented
    QGaLoreAdamW8Bit,  // Implemented: Halko randomized SVD + INT8 projections + 8-bit AdamW moments
    GaloreAdamW,        // Declared, STUB
    GaloreAdamW8Bit,    // Declared, STUB
    LOMO,               // Declared, Unimplemented
    Adalomo,            // Declared, Unimplemented
    CAME,               // Declared, Unimplemented
    Sophia,             // Declared, Unimplemented
}
```

**What's implemented and working:**
- **AdamW** (`crates/grim-autograd/src/adamw.rs`) — device-resident FP32 moment buffers, in-place `mul_scalar`/`sqrt`/`recip` on the device without host round-trips.
- **AdamW8Bit** — 8-bit quantized moments via the same trait dispatch.
- **PagedAdamW** — offloads cold moment pages to host RAM.
- **Lion / Lion8Bit** — sign-based update.
- **Adafactor** — factored second-moment estimate.

**Key architectural difference:** Grim's `Optimizer::new` constructs the optimizer directly in Rust (no Python dependency). The `step()` method dispatches through the `Optimizer` enum. Device-resident steps use `BackendDevice::mul_scalar`, `sqrt`, `recip` to avoid D2H round-trips.

**QGaLore:** `OptimizerKind::QGaLoreAdamW8Bit | GaloreAdamW | GaloreAdamW8Bit` all resolve to `QGaLoreAdamW8Bit::new`, which is fully implemented. The implementation includes:
- (`randomized_svd` at `adamw.rs:1397`) Halko randomized SVD with power iteration (oversample=10, niter=2) and Jacobi rotation for the small K×K eigenproblem
- (`GaloreProjector` at `adamw.rs:1573`) low-rank gradient projection with periodic subspace refresh (`update_proj_gap` steps)
- (`QGaLoreAdamW8Bit` at `adamw.rs:1697`) 8-bit quantized moment buffers stored as `Vec<u8>` with dynamic quantization/dequantization (scale = max_abs / 127.0, symmetric INT8), AdamW update in low-rank space, projection back to full space
- Test `test_qgalore_optimizer_build_and_step` verifies a 128×64 parameter step.

**LR schedulers:** 8 variants — `Cosine`, `Linear`, `Polynomial`, `Constant`, `InverseSqrt`, `Yolo`, `OneCycle`, `ReduceOnPlateau`.

### 4.2 Unsloth — bitsandbytes + TRL + Custom QGaLore

Unsloth's optimizer stack:
1. **Standard AdamW** — via `transformers`/`trl` optimizer (bitsandbytes 8-bit optional)
2. **AdamW8Bit** — bitsandbytes `bnb.optim.Adam8bit`
3. **QGaLoreAdamW8bit** — custom (from `unsloth/optimizers/q_galore_adamw.py`): 8-bit Adam via bitsandbytes `Optimizer2State` + GaLore projection + optional INT8 weight quantization
4. **PagedAdamW** — bitsandbytes `PagedAdamW` (VRAM paging to CPU)
5. **PagedAdamW8Bit** — bitsandbytes combined paged + 8-bit
6. **Lion** — bitsandbytes `Lion`
7. **Adafactor** — via `transformers`/`torch`
8. **LOMO, Adalomo, CAME, Sophia** — available via bitsandbytes or other libraries

**QGaLore implementation** (`unsloth/optimizers/q_galore_adamw.py` + `q_galore_projector.py`):
- `QGaLoreAdamW8bit(Optimizer2State)` — subclass of bitsandbytes' 8-bit Adam
- `GaLoreProjector` — low-rank gradient projection via truncated SVD
- SVD: `torch.linalg.svd` for small matrices (`min(m,n) <= rank*2`), `torch.svd_lowrank` (Halko randomized SVD: `q=rank+10, niter=2`) for large
- Adaptive schedule: rolling cosine similarity queue (5 entries), `cos_threshold=0.4`, `gamma_proj=2.0` multiplier on `update_proj_gap`
- INT4/INT8 quantized projection matrix storage via `_quantize`/`_dequantize` (asymmetric min-max quantization)
- `make_q_galore_param_groups` — auto-splits attention/MLP projection params into GaLore group

### 4.3 Comparison

**Where parity is achieved:**
- Both support: AdamW, AdamW8Bit, PagedAdamW, Lion, Adafactor, QGaLore (planned on Grim, implemented on Unsloth).
- Both support cosine/linear/LR scheduling patterns.
- Both support gradient scaling and weight decay.

**Where Grim is better:**
- **Device-resident optimizer steps.** Grim's AdamW uses `BackendDevice::mul_scalar`/`sqrt`/`recip` to compute updates entirely on-device. Unsloth's bitsandbytes also does this well, but Grim's trait-level design makes it a first-class concern.
- **Explicit multi-GPU all-reduce integration.** Grim's `TrainableParams::all_reduce_grads_weighted` (`param.rs:226`) takes an explicit `RcclAllReduce` handle and `contribution_weight` for asymmetric batches. Unsloth relies on `accelerate`'s DDP which handles this implicitly but less transparently.
- **Compile-time optimizer selection.** `OptimizerKind::from_str("qgalore-8bit")` is type-checked. No runtime import errors.
- **`fork_for_rank`** — Grim has explicit rank-replica forking (`adamw.rs:334`) that copies serialized optimizer state to target params. This is a clean primitive for multi-GPU.
- **More LR scheduler variants declared** (14 optimizers, 8 schedulers).

**Where Unsloth is better:**
- **QGaLore is actually implemented.** This is the single biggest gap — Grim declares `QGaLoreAdamW8Bit` but returns `Error::Unimplemented("Phase 7")`. Unsloth's is production-ready with INT4 projection quantization.
- **bitsandbytes integration.** 8-bit optimization, paged attention, CPU offloading of optimizer states — all battle-tested.
- **Weight quantization in optimizer.** Unsloth's QGaLore supports INT8 weight quantization with stochastic rounding. Grim's QGaLore stub doesn't exist yet.
- **Embedding LR splitting.** `UnslothTrainer.create_optimizer` splits embedding params into a separate LR group (default 5e-5). Grim has no equivalent — all params share one LR.

**Where neither does very well:**
- **Full CPU offload (ZeRO-3 style).** Neither has a complete CPU offload pipeline where all optimizer states, gradients, and parameters are offloaded to RAM and pages back on demand. Grim has `PagedAdamW` (partial) and SCYTHE-2's "optimizer offload to secondary GPU" (spec only). Unsloth has `accelerate` offload but it's not integrated into the core.
- **Adaptive optimizers beyond AdamW/Lion.** Grim declares CAME, Sophia, LOMO, Adalomo but none are implemented. Unsloth would need external libraries for these.

---

## 5. LoRA Variants & Injection

### 5.1 Grim — 7 Standard QLoRA Injection Points + 6 DoRA Variants

From `crates/grim-autograd/src/injection.rs`:

```rust
pub enum LoRAInjectionPoint {
    QProj,      // attention query
    KProj,      // attention key
    VProj,      // attention value
    OProj,      // attention output
    GateProj,   // MLP gate (SwiGLU)
    UpProj,     // MLP up
    DownProj,   // MLP down
    Logits,     // legacy (kept for compat, not standard QLoRA)
}
```

`weight_suffix()`, `adapter_prefix()`, `base_weight_shape()`, `lora_a_shape()`, `lora_b_shape()` — all computed from `InjectionConfig` (hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, vocab_size).

`AutogradScope::LoRAOnly` — the autograd tape only records ops touching these adapter parameters. Base weights are `frozen` (tracked but never updated).

**LoRA variants supported** (from `injection.rs` doc comments):
- **Standard LoRA** — `output = base + (α/r) * x @ A^T @ B^T`
- **DoRA** — `dora_forward()` in `ops.rs:22` (weight-decomposed, with directional matrix V = W_0 + γ * B @ A, column-wise L2 norm)
- **RSLoRA** — (mentioned in plan; `scale = α/r` variant with rank-aware scaling)
- **PiSSA** — (mentioned in plan; SVD-based init, `use_pissa` flag in CLI `train.rs:36`)
- **VeRA** — (mentioned in plan; vector-based adaptation, no learned B)
- **LoftQ** — (mentioned in plan; quantization-aware LoRA init)
- **SoulEater** — (mentioned in plan; aggressive low-rank compression)

**Op implementations** (`crates/grim-autograd/src/ops.rs`):
- `dora_forward()` — full Rust CPU computation (matmul, L2 norm, scaling)
- `apply_adapters_to_logits` — legacy logits-only injection (the plan notes this is insufficient for QLoRA parity)
- `lora_backward_matmul`, `scale_backward`, `backward` dispatch

### 5.2 Unsloth — PEFT + TRL + Custom LoRA Kernels

Unsloth integrates with **PEFT** (`from peft import LoraConfig, TaskType, get_peft_model`):
- `LoraConfig` — target_modules, r, lora_alpha, lora_dropout, bias, task_type, modules_to_save, **init** (loft, pissa, gram), **lora_contexual_layer_norm**, **alpha** scaling
- **Full PEFT ecosystem:** LoRA, P-Tuning, Prompt Tuning, IA³, AdaLoRA, AdaScale, LoRA Adapters, MixIA, FBLoRA
- `unsloth/kernels/fast_lora.py` — `LoRA_MLP` (fused gate/up/down + SwiGLU backward), `LoRA_QKV` (fused QKV + LoRA), `LoRA_W` (single linear), `fast_linear_forward` (dequant + LoRA add), `matmul_lora` (dequant via `fast_dequantize` + LoRA addmm)

**LoRA variants:**
- Standard LoRA (PEFT default)
- DoRA (via `lora_contexual_layer_norm` / PEFT 0.10+)
- PiSSA (PEFT `init="pissa"`, Unsloth has specific handling)
- LoftQ (`init="loft"` in PEFT, Unsloth validates via `validate_loftq_config`)
- RSLoRA (PEFT supports, Unsloth inherits)

### 5.3 Comparison

**Where parity is achieved:**
- Both inject LoRA at 7 standard points (Q/K/V/O/gate/up/down).
- Both support DoRA, PiSSA, LoftQ, RSLoRA.
- Both use frozen quantized base + trainable low-rank adapters (QLoRA).

**Where Grim is better:**
- **Scoped autograd.** `AutogradScope::LoRAOnly` means the tape only tracks 14 adapter matrices per layer, not the full graph. This is a fundamentally more memory-efficient approach.
- **Compile-time injection point enumeration.** `LoRAInjectionPoint::all_standard_qlora()` is a compile-time list of exactly 7 points. Unsloth's PEFT `target_modules` is a runtime regex match.
- **Weight suffix / prefix naming is type-safe.** `weight_suffix()` returns `&'static str`, `adapter_prefix()` generates `blk.{idx}.{suffix}.lora`. Unsloth relies on string matching against PyTorch parameter names.

**Where Unsloth is better:**
- **PEFT ecosystem.** Full PEFT support means prompt tuning, IA³, AdaLoRA, etc. — not just LoRA. Grim only has LoRA/DoRA variants.
- **Full-parameter finetuning.** Unsloth supports full FT (all params trainable). Grim's `AutogradScope::LoRAOnly` only tracks LoRA params — FullParam mode (`all_points()`) is declared but the autograd scope is hardcoded to `LoRAOnly`.
- **LoRA fusion at inference.** Unsloth's `fast_lora.py` fuses dequant + LoRA addmm into one GEMM. Grim's LoRA apply in `ops.rs` is CPU-only.
- **VeRA support** — Unsloth inherits this via PEFT; Grim mentions it in docs but doesn't implement it.

**Where neither does very well:**
- **Dynamic architecture LoRA placement.** Neither auto-detects optimal injection points based on gradient magnitude or attention analysis for arbitrary model architectures.
- **LoRA fusion with attention.** Neither fuses LoRA application directly into the attention/FFN kernels at the compute level (Unsloth uses PyTorch's `addmm` fallback; Grim uses CPU matmul).

---

## 6. Quantization

### 6.1 Grim — Q4_K, NF4, Q8_0, Q5_K, Q6_K (GGUF-native)

Grim's quantization lives in `crates/grim-quant/src/lib.rs` and uses the **GGUF** format (from llama.cpp):
- `Q4_K` — 4-bit with 6-bit super-block scales per sub-block (256 weights / 8 sub-blocks of 32)
- `NF4` — NormalFloat4 (16 canonical codebook levels from QLoRA paper)
- `Q8_0` — 8-bit, 1 elem/byte, per-row scale
- `Q5_K`, `Q6_K`, `Q3_K`, `Q2_K` — other GGUF levels

**Known issues (from `old/unsloth-parity-optimizations.md`):**
- Q4_K writer had degenerate hardcoded scales: `scales = [1,1,1,1,0,0,0,0,1,1,1,1]`, `dmin = 0`. Now fixed in the plan (Phase 0a).
- NF4 codebook was a 14-bucket hand-tuned ladder, not the canonical 16-level NormalFloat codebook. Now fixed (Phase 0b).

The reader (`dequant_q4k`) correctly reads real llama.cpp Q4_K, so the fix is about writer accuracy.

### 6.2 Unsloth — bitsandbytes Quantization (NF4, FP4, FP8, INT8)

From `device_type.py:108-125` and `kernels/utils.py`:
- **NF4 (NormalFloat4)** — canonical 16 levels, via `bitsandbytes` `cdequantize_blockwise_fp32`/`_fp16_nf4`/`_bf16_nf4`
- **FP4** — `cdequantize_blockwise_fp16_fp4`
- **FP8 (E4M3/E5M2)** — for newer GPUs (H100+ / CDNA3)
- **INT8** — weight-only quantization
- **Double quantization** — as described in the QLoRA paper, bitsandbytes supports nested 2nd-level quantization of scales

`fast_dequantize` (utils.py) uses bitsandbytes C extensions with global buffer reuse. Supports NF4, FP4, FP8, INT8 dequantization.

### 6.3 Comparison

**Where parity is achieved:**
- Both use QLoRA: frozen 4-bit quantized base model + LoRA adapters.
- Both use NF4 (canonical 16-level codebook) as the default quantization.
- Both support fused dequantize + GEMM (Grim's `fused_dequant_gemm`; Unsloth's `fast_dequantize` + `matmul_lora`).

**Where Grim is better:**
- **GGUF format compatibility.** Grim reads/writes GGUF natively — can load any llama.cpp-quantized model directly. No bitsandbytes dependency.
- **Per-sub-block Q4_K scaling.** GGUF's Q4_K has 8 sub-blocks per 256-weight block, each with its 6-bit scale. The corrected writer preserves this.
- **No Python dependency for quantization.** Pure Rust quantization pipeline.

**Where Unsloth is better:**
- **Double quantization.** bitsandbytes supports nested second-level quantization of the scales themselves. Grim only has single-level Q4_K/NF4.
- **GPTQ / AWQ / SmoothQuant.** bitsandbytes ecosystem supports these. Grim only has GGUF formats.
- **FP4 and FP8.** bitsandbytes supports FP4 (fused attention) and FP8. Grim has FP8 (`fp8_gemm_rdna4`) but only for RDNA4-specific code path.
- **INT4/INT8 weight quantization in optimizer.** Unsloth's QGaLore supports INT8 weight quantization with stochastic rounding. Grim's is a stub.
- **Quantization-aware training.** bitsandbytes' `Params4bit` with gradient propagation. Grim's QLoRA is inference-only (frozen base).

**Where neither does very well:**
- **Quantization-aware LoRA init.** Both mention LoftQ (quantization-aware init) but neither has a fully production-tested pipeline that jointly optimizes quantization + LoRA init.
- **Mixed-precision quantization (e.g., 4-bit weights + 8-bit KV cache).** Grim dequantizes KV caches on-the-fly in attention but doesn't support mixed KV cache quantization. Unsloth relies on `transformers` for KV cache quantization.
- **Cross-format conversion.** Neither has a seamless path from GGUF Q4_K to bitsandbytes NF4 and back.

---

## 7. Memory Optimization

### 7.1 Grim — Streaming Forward + Gradient Checkpointing

Key files:
- `crates/grim-engine/src/streaming_forward.rs` — `StreamingBlockForward` + `GradientCheckpointBuffer`
- `crates/grim-autograd/src/collate.rs:49` — `VarLenCollator` → `PackedBatch`

**Techniques:**
- **Streaming forward** — reads quantized transformer weights lazily block-by-block via `TensorProvider`, avoiding loading the full model into RAM. `prefetch_block_weights()` explicitly prefetches all 9 weight tensors per block to device.
- **Gradient checkpointing** — saves only input activations per layer (`LayerActivationCheckpoint { layer_idx, input_x }`). The `GradientCheckpointBuffer` stores `HashMap<usize, LayerActivationCheckpoint>` — only the input to each block is retained; intermediate activations are recomputed during backward.
- **Packed/variable-length sequences** — `VarLenCollator` packs sequences to eliminate padding waste. `PackedBatch` has `concatenated_tokens`, `seqlen_offsets` (cu_seqlens), `sequence_lengths`.

**Gap (from parity plan):** `packing_attention_mask()` returns `vec![true; T]` (all-true) — packed sequences attend across boundaries. The fix (`block_diagonal_causal_mask`) is planned in Phase 3 of the parity plan. **This is a correctness bug.**

### 7.2 Unsloth — Padding-Free + Smart Gradient Checkpointing

Key files:
- `unsloth/utils/packing.py` — `build_sdpa_packed_attention_mask`, `mask_packed_sequence_boundaries`, `enable_sample_packing`, `enable_padding_free_metadata`
- `unsloth/models/llama.py` — `LlamaDecoderLayer_fast_forward` uses `torch.utils.checkpoint.checkpoint` with `use_reentrant=True`

**Techniques:**
- **Padding-free (varlen) training** — packs sequences into a flattened 1D buffer with `cu_seqlens`. Uses `build_sdpa_packed_attention_mask` for correct block-diagonal causal attention.
- **Sample packing** — `enable_sample_packing` wraps TRL collator. `mask_packed_sequence_boundaries` sets cross-boundary target tokens to `ignore_index=-100`.
- **Smart gradient checkpointing** — `use_reentrant=True` with `Unsloth_Offloaded_Gradient_Checkpointer`. `unsloth_offloaded_gradient_checkpoint` function. Patches `torch.utils.checkpoint` with custom save/restore.
- **CPU offloading** — `offload_to_disk`, `offload_input_embeddings`, `offload_output_embeddings`. Moves embeddings/norms to CPU during forward to save VRAM.
- **Embedding/LM head offloading** — `offload_input_embeddings` / `offload_output_embeddings` move large embedding tables to CPU.

### 7.3 Comparison

**Where parity is achieved:**
- Both have gradient checkpointing.
- Both have sample packing.
- Both have padding-free/varlen training concepts.

**Where Grim is better:**
- **Streaming weight loading.** Grim's `StreamingBlockForward` reads weights lazily from `TensorProvider`, never materializing the full model in RAM. Unsloth loads the entire model via `transformers.AutoModelForCausalLM.from_pretrained`.
- **Zero host round-trips in principle.** Grim's `BackendDevice` trait is designed to keep all computation on-device. `mul_scalar`/`sqrt`/`recip` are trait methods so AdamW step can run on GPU.
- **Deterministic memory.** Rust's ownership model means memory is freed deterministically at scope end. No GC pauses or reference counting overhead.
- **`prefetch_block_weights`** — explicit prefetch of all 9 block tensors before forward, enabling overlap.

**Where Unsloth is better:**
- **Correct packed attention.** Unsloth's `build_sdpa_packed_attention_mask` builds the correct `[1,1,T,T]` block-diagonal causal mask. Grim's `packing_attention_mask()` returns all-true (bug, not yet fixed).
- **CPU offloading of activations/embeddings.** Unsloth offloads embeddings, norms, and gradient-checkpointed activations to CPU explicitly. Grim's `PagedAdamW` is optimizer-state-level only.
- **Hybrid linear attention support.** `patch_hybrid_linear_attention_varlen` detects models with linear-attention/state-space mixers (Qwen3.5, Qwen3-Next) and disables packing to prevent cross-boundary state leakage. Grim has no equivalent detection.
- **`enable_padding_free_metadata`** — sets up `cu_seqlens` and `max_seqlen` so SDPA kernels can skip padding entirely.

**Where neither does very well:**
- **Full ZeRO-3 CPU offload.** Neither has a complete pipeline where parameters, gradients, AND optimizer states are all offloaded to CPU/RAM with demand paging. Grim has `PagedAdamW` (optimizer states only). Unsloth relies on `accelerate` offload (separate dependency).
- **Activation offloading during forward.** Neither offloads intermediate activations to CPU during the forward pass — only checkpoint inputs (Grim) or checkpoint saved tensors (Unsloth).
- **Dynamic sequence length batching.** Neither dynamically reshapes the model for different sequence lengths at runtime (no FlashInfer-style plan-and-execute).

---

## 8. Multi-GPU & Distributed Training

### 8.1 Grim — SCYTHE-2 CommFuse (Research-Grade, Spec-Only)

**`scythe2.md`** (in `old/`) is a 577-line formal specification grounded in 13 papers:
- **FCP** (Flexible Context Parallelism, 2602.21788) — polynomial-time placement selection in <1 ms
- **WaveTune** (2604.10187) — bilinear latency predictor (runtime table lookup, not candidate loop)
- **HetAuto** (EUROSYS '26) — online MCTS + random forest cost model for heterogeneous auto-parallelism
- **ReMP** (2606.18741) — runtime TP/PP topology reconfiguration without restart (1–7 s)
- **Amoeba** (2509.19729) — runtime TP degree transformation (1.75–6.57× throughput)
- **CommFuse** (2604.24013) — decomposed P2P replacing all-reduce
- **Concordia** (2606.23521) — device-resident persistent kernel (219× faster delta checkpointing)
- **Harvest** (2602.00328) — opportunistic peer-GPU caching
- **GPREEMPT** (USENIX ATC '25) — 40 µs context-switch preemption
- **TriRoute** (2607.06601) — per-token-per-axis learned routing
- **Piper** (2605.05049) — resource modeling
- **Characterizing Overlap** (2507.03114) — conditional overlap (avoid 18.9% blind-overlap slowdown)

**C²PLR controller** (`crates/grim-engine/src/scythe2.rs`):
- A 2-layer MLP (~8 KB) that emits `(placement, partition, route)` per layer per shape
- `PlacementCache`: array-indexed by `layer_id` → O(1) lookup (~50 ns on decode path, ~6000× margin under 10 ms ITL)
- `decide_miss()` runs only on cache miss (prefill / capability epoch bump) → ~10 µs/layer
- Capability epoch cadence: 100 ms (derived from PowerTune thermal hysteresis ~50–100 ms)

**Key types in `crates/grim-tensor/src/backend.rs:246`:**
```rust
pub struct GpuCapability {
    pub tflops_fp16: f32,
    pub tflops_fp8: f32,        // 0.0 if arch < RDNA 4
    pub hbm_bandwidth_gbps: f32,
    pub vram_free_bytes: u64,
    pub throttle_pct: f32,
    pub ordinal: usize,
}
pub enum ScytheLink { PeerDirect, Pcie, Host }
pub struct ScythePlacement {
    pub ranks: Vec<usize>,
    pub partition: Vec<f32>,    // does NOT sum to 1.0 for replicated layers
    pub routes: Vec<ScytheLink>,
}
```

**RCCL FFI** (`crates/grim-backend-rocm/src/rccl.rs`):
- `ncclAllReduce`, `ncclReduceScatter`, `ncclAllGather` — standard NCCL/RCCL collectives
- `RocmComm::all_reduce` — sum reduction with F16/F32 support
- `RcclAllReduce::sum_gradients_device` (`rccl.rs` ~line 290+) — in-place device-pointer all-reduce with 1/N averaging
- `scale_gradients` — pre-reduction scaling
- `fuse_reduce_scatter` — grouped NCCL calls via `ncclGroupStart`/`ncclGroupEnd`
- `fuse_all_gather` — same pattern
- `p2p_memcpy_async` — P2P async copy for CommFuse decomposed transport

**`TrainableParams::all_reduce_grads_weighted`** (`param.rs:226`):
- Takes `RcclAllReduce` handle + `contribution_weight` (for asymmetric batches)
- Device-pointer in-place all-reduce (no D2H round-trip)
- Falls back to host round-trip when no device pointer available
- Single-rank accumulation path explicitly NOT a multi-rank reduction (error returned if `num_gpus > 1 && rccl.is_none()`)

**Implementation status (updated — checked 2026-08-02):**
- WI-1 through WI-10 outlined but **partial implementation**:
  - WI-1 (`backend.rs` trait extension): `estimate_gemm_latency_ms` now **implemented** on `RocmDevice` (WaveTune bilinear predictor with TFLOPS lookup); `comm_fuse_reduce` remains default `Err(Unimplemented)`
  - WI-2 (CapabilityProfiler): not yet created
  - WI-3 (`scythe2.rs` Scythe2Linear): `forward_placed` → `forward_col_parallel`/`forward_row_parallel` use **CPU-side matmul** (`to_cpu_vec_f32()` + nested loops) — no GPU dispatch
  - WI-4 (C2plrController): `decide_miss` and `update` are **no longer `todo!()` stubs** — `decide_miss` implements WaveTune bilinear latency eval, `update` implements Lagrangian dual ascent (`λ ← λ + α(t̂_total - T_budget)`, `MLP_LR = 0.001`); but the overall routing is **still CPU-only** (no GPU kernel dispatch through the controller)
  - WI-5 (`all_reduce` on RocmDevice): **partially implemented** — `RocmDevice` now overrides `all_reduce` (line 2955), but it's a **CPU round-trip** for both single-input (identity via `to_cpu_vec_f32()`) and multi-input (element-wise sum via host loops). No actual RCCL NCCL all-reduce for multi-GPU. The `BackendDevice::all_reduce` default in `backend.rs:463` still returns `Err(Unimplemented)` for non-ROCm backends.
  - WI-6 (CommFuse kernel): not yet created
  - WI-7 (persistent ring): `ScytheRing` and `ScytheTaskDescriptor` types exist in spec but no implementation
  - WI-8 (ReMP KV migration): not implemented
  - WI-9 (wire `num_gpus`): `TrainingJob.num_gpus` in `jobs.rs:104` is currently ignored
  - WI-10 (bench): not yet implemented

**Where Grim is better (in principle):**
- **CommFuse decomposed P2P.** Replaces reduce-scatter + all-gather (two sync points) with direct P2P push to owning rank. Eliminates tail latency.
- **Per-layer routing.** C²PLR routes each layer independently — memory-bound layers (RMSNorm, RoPE) replicated, compute-bound layers (GEMM) sharded, embedding offloaded. Unsloth's DDP does one strategy for the whole model.
- **Runtime topology reconfiguration (ReMP).** 1–7 s topology switch without restart. Unsloth requires restart for TP degree changes.
- **Spec is grounded in real hardware data.** The 100 ms epoch cadence is derived from PowerTune thermal hysteresis, not arbitrary.
- **WaveTune is now compiled.** The bilinear latency predictor runs on CPU for placement decisions.

**Gap (spec partially compiled):**
- `BackendDevice::all_reduce` on `RocmDevice` is implemented but **CPU-only** (host round-trip for element-wise sum) — no true RCCL multi-GPU all-reduce.
- `BackendDevice::all_reduce` on non-ROCm backends (`backend.rs:463`) still returns `Err(Unimplemented)`.
- `RowParallelLinear::forward` silently swallows errors.
- `ColumnParallelLinear`/`RowParallelLinear` exist but have **no sharding** — only `{rank, world_size}` fields.
- `Scythe2Linear::forward_placed` uses **CPU-side matmul** (`to_cpu_vec_f32()` + nested loops) for correctness, not actual GPU dispatch.

### 8.2 Unsloth — PyTorch DDP + Accelerate + FSDP

Unsloth's multi-GPU story is built on the PyTorch ecosystem:
- **`accelerate`** — `DistributedDataParallel`, `FullyShardedDataParallel` (FSDP), tensor parallelism
- **`torch.distributed`** — standard NCCL-backed DDP all-reduce
- **`transformers`** — `device_map="auto"` for model parallelism across VRAM
- **`trl`** — RLHF training with multi-GPU support

**Key integration points:**
- `_patch_trl_trainer` — backward-compat with TRL `SFTConfig`/`__init__` changes
- `UnslothTrainer(SFTTrainer)` — inherits DDP/FSDP from TRL
- `_mark_unsloth_disable_data_parallel` — disables DataParallel when using custom kernels
- `_patch_transformers_trainer_data_parallel` — patches `DataParallel` for compatibility

### 8.3 Comparison

**Where parity is achieved:**
- Both support gradient all-reduce across multiple GPUs.
- Both support tensor parallelism concepts.

**Where Grim is better:**
- **CommFuse P2P fan-in** (spec-only, but the design is superior to NCCL's reduce-scatter + all-gather).
- **Per-layer capacity calibration** (C²PLR — spec-only).
- **Explicit asymmetric batch handling** — `all_reduce_grads_weighted(contribution_weight)` handles unequal batches.
- **Device-resident gradient reduction** — `sum_gradients_device` with in-place device pointer all-reduce, no D2H.
- **No silent failures** — when `rccl.is_none()` and `num_gpus > 1`, Grim returns an explicit error. Unsloth's DDP can silently fail or hang.

**Where Unsloth is better:**
- **Actually works.** Unsloth's DDP/FSDP via `accelerate` is production-tested on 8×H100, 16×A100, etc. Grim's multi-GPU is entirely spec-only — `all_reduce` returns `Err(Unimplemented)`.
- **FSDP integration.** `accelerate`'s FSDP offloads parameters, gradients, and optimizer states to CPU/RAM with demand paging. Grim has no equivalent.
- **`device_map="auto"`** — HuggingFace's automatic device mapping for uneven VRAM across GPUs. Grim's C²PLR controller is the theoretical equivalent but is not compiled.
- **ZeRO-3 (Zero Redundancy Optimizer).** Via `accelerate`/`deepspeed` integration. Grim has no ZeRO.

**Where neither does very well:**
- **3D parallelism (TP + DP + PP).** Neither has a mature, tested 3D parallel implementation. Grim's SCYTHE-2 spec covers this but is unimplemented. Unsloth relies on external `tensor_parallel` libraries.
- **Pipeline parallelism.** Neither has native pipeline parallelism (1F1B scheduling). Unsloth would need `accelerate`'s pipeline parallelism; Grim has nothing.
- **Multi-node (cross-host) training.** Neither has tested multi-node setups. Grim's RCCL only works within a node (NCCL/RCCL are single-node by default). Unsloth's `accelerate` can do multi-node but it's not the primary use case.

---

## 9. Preference Optimization (RLHF/DPO/GRPO)

### 9.1 Grim — Native Rust Preference Losses

`crates/grim-autograd/src/preference_loss.rs` (231 lines):

| Loss | Function | Returns | Notes |
|---|---|---|---|
| DPO | `dpo_loss()` | `(loss, chosen_rewards, rejected_rewards)` | `softplus(-logits)` where `logits = β*(logr_chosen - logr_rejected)` |
| DPO (autograd) | `dpo_loss_autograd()` | `(loss, chosen_grad, rejected_grad)` | Tensor-based gradient for backward pass |
| ORPO | `orpo_odds_ratio_loss()` | `f32` | `lambda * softplus(-log_odds_ratio)` |
| ORPO (autograd) | `orpo_odds_ratio_loss_autograd()` | `(loss, grad_tensor)` | |
| KTO | `kto_loss()` | `(loss, chosen_losses, rejected_losses)` | Desirable/undesirable weights, KL estimate |
| SimPO | `simpo_loss()` | `f32` | `softplus(-(β*(p_w - p_l) - γ))` |
| GRPO | `grpo_loss()` | `(loss, per_sample_losses)` | Group-normalized rewards, clipped surrogate |
| GRPO reward norm | `grpo_normalize_rewards()` | `Vec<f32>` | `(r - mean) / (std + eps)` |
| OLoRA penalty | `ola_penalty()` | `f32` | Orthogonality regularization |

All use numerically stable `softplus(-x) = max(-x, 0) + ln(1 + exp(-|x|))` via inline implementation, avoiding sigmoid underflow.

### 9.2 Unsloth — TRL Integration

From `unsloth/models/dpo.py` and `trainer.py`:
- **DPO** — via `trl.DPOTrainer` (full class in `models/dpo.py`)
- **ORPO** — via `trl.ORPOTrainer`
- **KTO** — via `trl.KTOTrainer` or custom
- **SimPO** — via `trl.SimPOTrainer`
- **GRPO** — via `trl.GRPOTrainer` (multi-response RL)
- **PPO** — via `trl.PPOTrainer` (full RLHF with reward model)

Unsloth's `UnslothTrainer` subclasses `SFTTrainer` and inherits TRL's DPO/GRPO support through the `loss_type` config field. The `models/dpo.py` file specifically handles the DPO loss computation with preference-pairs.

### 9.3 Comparison

**Where parity is achieved:**
- Both support DPO, ORPO, KTO, SimPO, GRPO.
- Both use numerically stable softplus/max-trick formulations.
- Both compute the same mathematical formulas.

**Where Grim is better:**
- **Native Rust implementations.** No Python/Torch dependency. No autograd graph overhead for preference losses — they're pure functions on log-probability slices.
- **Tensor-based autograd versions.** `dpo_loss_autograd()` returns gradient tensors compatible with Grim's tape-based backward. This is integrated with the scoped LoRA-only autograd.
- **Explicit per-loss function separation.** Each loss is a standalone function with clear inputs/outputs. Unsloth's TRL integration hides this behind class hierarchies.

**Where Unsloth is better:**
- **Full RLHF pipeline.** PPOTrainer with reward model, value head, KL penalty, advantage estimation. Grim only has the loss functions — no training loop.
- **TRL's mature DPO/GRPO implementations.** Battle-tested with edge cases, logging, metrics, early stopping, etc.
- **Integration with training loop.** TRL trainers handle the full preference optimization training loop: data loading, batching, evaluation, checkpointing. Grim only has the loss function.
- **Multi-turn / conversational DPO.** TRL's `DataTokenizer`/` conversations` handling for multi-turn dialogues. Grim's `dpo_loss` takes flat log-probability slices.

**Where neither does very well:**
- **Online DPO / IPO / CPO / RRHF.** Neither has these less-common preference formulations.
- **Preference data management.** Neither handles the preference dataset loading, filtering, or hard-negative mining that production RLHF pipelines need.
- **Reward model training.** Neither has a standalone reward model training path — Unsloth relies on TRL's implicit approach, Grim has nothing.

---

## 10. Training Loop & CLI

### 10.1 Grim — Rust CLI Training Loop

From `crates/grim-cli/src/train.rs`:

```
GGUF load → InjectionConfig (from metadata) →
  LlamaConfig → StreamingBlockForward (block-by-block) →
  Tape-based forward → cross_entropy_loss (GPU kernel) →
  backward (tape walk) → accumulate_grad →
  AdamW step → TrainState sidecar save (.grim.train)
```

**CLI training options** (`TrainOptions` in `train.rs:22`):
- `model_path`, `dataset_path`, `output_sidecar`
- `epochs`, `lr`, `rank`, `alpha`
- `device: String` — "cpu" | "rocm" | "cuda" | "vulkan" | "metal"
- `mode` — "qlora", "lora", "dora", "pissa", "olora"
- `optimizer` — `OptimizerKind` (from string)
- `scheduler` — `LRScheduler` (from string)
- `use_pissa: bool`, `use_olora: bool`, `olora_lambda: f32`

**Sidecar format** (`crates/grim-format/src/train.rs`):
- Magic: `GRIMTRN\x01`
- JSON header + binary blob blocks
- Stores optimizer states (AdamW m/v moments), LoRA A/B matrices, step counter
- `TrainState` struct with `TrainFpFormat` (Fp32, Fp16, Nf4, F8E4M3, F8E5M2)

**Dataset support:** Alpaca format (`{instruction, input, output}`) and ShareGPT format (`{conversations}`). Greedy packing via `pack_dataset_tokens` / `pack_training_examples`.

### 10.2 Unsloth — SFTTrainer / SupervisedFineTuning

From `trainer.py` and `models/llama.py`:

```python
FastLanguageModel.from_pretrained(model_name, ...) →
  get_peft_model(model, LoraConfig(...)) →
  UnslothTrainer(SFTTrainer) →
  .train()  # full PyTorch training loop
```

**TrainingArguments** (`UnsloughtTrainer`):
- Inherits from `trl.SFTConfig` / `transformers.TrainingArguments`
- Adds: `q_galore_config: QGaloreConfig`, `embedding_learning_rate: float`
- Full HF TrainingArguments: `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `warmup_steps`, `weight_decay`, `lr_scheduler_type`, etc.

**PEFT integration:**
- `LoraConfig` with `target_modules`, `r`, `lora_alpha`, `lora_dropout`, `task_type`, `init` (loft, pissa, gram), `modules_to_save`
- `get_peft_model` from PEFT library

**Data loading:**
- `datasets.load_dataset` integration
- `transformers.AutoTokenizer` for tokenization
- Custom collators via `unsloth/utils/packing.py`

### 10.3 Comparison

**Where parity is achieved:**
- Both support QLoRA training with LoRA/DoRA/PiSSA.
- Both support sample packing for efficient batch utilization.
- Both support SFT (supervised fine-tuning) on instruction datasets.

**Where Grim is better:**
- **Zero Python overhead.** The entire loop is Rust — no PyTorch/TorchScript/dynamo compilation overhead.
- **Binary sidecar format.** `.grim.train` with structured header + binary blocks for state serialization. No pickle/Python serialization fragility.
- **Deterministic resource usage.** Rust's ownership model means no memory leaks or GC pauses.
- **Compile-time configuration.** `TrainOptions` fields are type-checked; no runtime KeyError on missing config keys.
- **GGUF-native.** Loads GGUF models directly, no conversion needed.

**Where Unsloth is better:**
- **HuggingFace ecosystem.** Loads from HF Hub, saves to HF format, integrates with `transformers`/`datasets`/`peft`/`trl`. Grim downloads from HF Hub too (via `client.rs`), but requires a separate `grim oxidizer convert` → `.grim` step before training. Unsloth trains directly from any HF format (gguf, safetensors, bin). However, Grim's converter (see `convert.rs`) applies EvoPress GPTQ per-tensor quantization, SmoothQuant, SpQR, and SpinQuant at conversion time — optimizations Unsloth doesn't perform at load time.
- **Full TrainingArguments.** Batch size, gradient accumulation, warmup, logging, evaluation, early stopping, callbacks — all from HF's battle-tested `TrainingArguments`. Grim's `TrainOptions` is minimal (no batch size, no gradient accumulation, no logging).
- **Mixed precision / autocast.** `torch.autocast("cuda", dtype=torch.bfloat16)` is integrated. Grim's tensors are all F32 in the training path.
- **Logging/monitoring.** Integration with wandb, tensorboard, mlflow via HF callbacks. Grim has no built-in logging integration.
- **Multi-node DDP.** Via `accelerate`. Grim's multi-GPU is spec-only.
- **Evaluation/metrics.** TRL's reward modeling, metric computation, comparison generation. Grim has none.

**Where neither does very well:**
- **Hyperparameter scheduling beyond LR.** Neither has advanced scheduling like loss-weighted LR, layer-wise LR decay, or gradient noise injection.
- **Dynamic loss scaling.** Neither has AMP-level loss scaling for mixed-precision stability.
- **Production monitoring/observability.** Neither has built-in metrics dashboards, alerting, or profiling visualization.

---

## 11. Inference & Serving

### 11.1 Grim — Streaming Forward + Paged Attention

From `crates/grim-engine/src/streaming_forward.rs`:
- `StreamingBlockForward` — streaming, block-by-block execution
- `GradientCheckpointBuffer` — bounded peak memory by retaining only block inputs
- KV cache: `kv_dequant_attention` (online-softmax GQA causal, dequantize K/V on-the-fly)
- Paged attention: `qkv_attention_paged` (vLLM-style block tables)
- Speculative decoding: `tree_attention` kernel for draft tree verification

**Serving features:**
- `grim-core/src/client.rs` — client-server RPC for inference
- `grim-server/src/lib.rs` — server with KV cache management
- Network KV transport (per commit `076213b`)

### 11.2 Unsloth — vLLM-style + FlashAttention

From `models/llama.py`:
- `LlamaAttention_fast_forward_inference` — paged attention KV cache
- 4-chunk QK^T computation (for memory efficiency)
- `fast_swiglu_inference` — fused inference SwiGLU
- `fast_rms_layernorm_inference` / `_gemma` variants
- `CausalLM_fast_forward` — calls `unsloth_fused_ce_loss` for ≤1024 tokens

**Serving features:**
- No native server — relies on `vLLM`, `TextGenerationWebUI`, `SGLang`, or `transformers` pipelines
- FlashAttention 2 integration (optional, via `flash-attn` wheel)

### 11.3 Comparison

**Where Grim is better:**
- **Native speculative decoding support.** `tree_attention` kernel for Medusa/EAGLE-style draft verification.
- **De novo serving stack.** `grim-server` + `grim-core` client provide a complete RPC-based serving system.
- **KV cache dequantization on-the-fly.** `kv_dequant_attention` dequantizes 4-bit/8-bit K/V during attention, no separate dequant step.

**Where Unsloth is better:**
- **Ecosystem maturity.** vLLM, SGLang, TGI all have more sophisticated scheduling, continuous batching, prefix caching, P/D attention.
- **PagedAttention integration.** vLLM's block allocator is more mature than Grim's `qkv_attention_paged` (which is a single kernel).
- **Pre-filled serving support.** No equivalent to vLLM's PagedAttention with shared attention blocks.

**Where neither does very well:**
- **Multi-model serving.** Neither has a unified serving API for vision-language models, diffusion, or speech.
- **Load balancing / autoscaling.** Neither has Kubernetes-based autoscaling or multi-instance GPU support.
- **Continuous batching.** Grim has paged attention but no documented continuous batching scheduler. Unsloth relies on vLLM for this.

---

## 12. Summary Decision Matrix

### Where Parity is Achieved (both solve the problem equivalently)

| Capability | Grim Evidence | Unsloth Evidence | Notes |
|---|---|---|---|
| QLoRA training | `AutogradScope::LoRAOnly`, frozen base | `fast_dequantize` + PEFT LoraConfig | Same algorithm, different substrate |
| Fused cross-entropy | `cross_entropy_loss` GPU kernel + CPU fallback | `Fast_CrossEntropyLoss` Triton | Both use online logsumexp |
| Fused SwiGLU | `silu_mul` HIP kernel | `_fg_kernel` Triton | Grim backward is a stub |
| RMSNorm | `rms_norm` HIP kernel | `_rms_layernorm` Triton | Both have Gemma variant |
| RoPE | `rope` HIP kernel | `_rope_embedding_QK` Triton | Both in-place, backward negation |
| LoRA injection (7 pts) | `LoRAInjectionPoint` enum (7 standard) | PEFT `target_modules` default | Both target Q/K/V/O/gate/up/down |
| Gradient checkpointing | `GradientCheckpointBuffer` | `torch.utils.checkpoint` | Both save input activations |
| Sample packing | `VarLenCollator` → `PackedBatch` | `enable_sample_packing` | Grim's mask is buggy (all-true) |
| AdamW optimizer | `AdamW` with FP32 moments | bitsandbytes `Adam8bit` | Grim more device-resident |
| DPO/ORPO/KTO/SimPO/GRPO | `preference_loss.rs` (7 losses) | TRL `DPOTrainer`/`GRPOTrainer` | Grim native, Unsloth via TRL |
| NF4 quantization | `quant_nf4`/`dequant_nf4` | `fast_dequantize` (bitsandbytes) | Both canonical 16-level |
| Online logsumexp | `fused_linear_cross_entropy_loss` | `Fast_CrossEntropyLoss` | Both f32, max-trick |

### Where Grim is Better

| Capability | Grim Advantage | Unsloth Gap |
|---|---|---|
| **Scoped autograd** | Tape only records LoRA-relevant ops (`AutogradScope::LoRAOnly`). No full-graph tracing. | PyTorch traces every tensor op; autograd graph is huge for a 7B model |
| **Device-resident AdamW** | `mul_scalar`/`sqrt`/`recip` as trait methods — optimizer step runs on GPU, zero D2H | bitsandbytes also does this, but less explicit in the API |
| **Explicit asymmetric all-reduce** | `all_reduce_grads_weighted(contribution_weight)` handles unequal batches | DDP assumes equal batches or manual gradient scaling |
| **CommFuse P2P** (spec) | Decomposed P2P eliminates reduce-scatter + all-gather tail latency | NCCL's standard all-reduce |
| **Runtime topology reconfig** (spec) | ReMP: 1–7 s topology switch without restart | Requires process restart for TP degree change |
| **GGUF-native loading** | Reads GGUF directly, no conversion | Must convert to HF format or use GGUF loaders |
| **Compile-time safety** | Rust type system prevents trait contract violations | Python runtime errors (missing imports, wrong types) |
| **Binary sidecar format** | `TrainState` with `GRIMTRN\x01` magic + JSON header | HF `pytorch_model.bin` / `safetensors` (fine, but less structured for optimizer state) |
| **26 HIP kernel modules** | Fused dequant-gemm fwd+bwd, KV dequant attention, tree attention, Mamba, RWKV | Triton kernels: fewer fused backward passes, less domain-specific |
| **No Python dependency** | Entire stack is Rust — no pip install hell, no CUDA/cuDNN/torch version conflicts | `pyproject.toml` has 15+ optional dependency groups (cu118, cu121, cu124, ..., intelgpu, audio-torch210, etc.) — massive dependency matrix |

### Where Unsloth is Better

| Capability | Unsloth Advantage | Grim Gap |
|---|---|---|
| **QGaLore (implemented)** | `QGaLoreAdamW8bit` with bitsandbytes 8-bit Adam + INT4 projection quantization + stochastic rounding | `QGaLoreAdamW8Bit` is `Error::Unimplemented("Phase 7")` — declared but not built |
| **PEFT ecosystem** | Full PEFT: LoRA, prompt tuning, IA³, AdaLoRA, AdaScale, LoRA Adapters | Grim only has LoRA/DoRA variants — no prompt tuning, no adapter fusion |
| **Direct HF format training** | Loads/saves from HF Hub, `datasets`, `transformers` model zoo — trains from any HF format in-memory | Grim downloads from HF Hub but requires a `.grim` conversion step before training; cannot train from `.safetensors`/`.gguf` directly |
| **torch.compile** | `@torch.autocast` + `torch.compile` for end-to-end graph fusion | No equivalent compile-time optimization in Rust path |
| **TrainingArguments maturity** | Full `TrainingArguments`: batch size, grad accumulation, warmup, early stopping, logging, eval | `TrainOptions` is minimal — no batch size, no grad accumulation, no logging |
| **Mixed precision** | `torch.float16`/`torch.bfloat16` autocast throughout | Grim's training path is F32-only |
| **Multi-GPU (production)** | DDP/FSDP via `accelerate` — tested on 8×H100 | Grim's `all_reduce` returns `Err(Unimplemented)`; SCYTHE-2 is spec-only |
| **PPO / full RLHF** | `trl.PPOTrainer` with reward model, value head, KL penalty | Grim only has preference loss functions, no training loop |
| **CPU offloading** | `accelerate` offload for optimizer states, params, activations | Grim has `PagedAdamW` (optimizer states only) — no param/activation offload |
| **Hybrid linear attention** | `patch_hybrid_linear_attention_varlen` detects Qwen3.5/Qwen3-Next (can't pack) | No equivalent detection in Grim |
| **Embedding LR split** | `embedding_learning_rate` param — separate LR for embeddings | Grim has no concept of parameter groups with different LRs |
| **Loss function variants** | Gemma softcap (`t*tanh(1/t*x)`), Cohere scaling (`t*x`) in CE loss | Grim's CE loss has no logit softcapping/scaling |
| **Triton kernel autotuning** | Per-kernel block size/warp count autotuning via `calculate_settings` | Grim kernels are JIT-compiled but not autotuned at runtime |

### Where Neither Does Very Well

| Capability | Grim Weakness | Unsloth Weakness | Shared Gap |
|---|---|---|---|
| **Packed attention correctness** | `packing_attention_mask()` returns all-true (bug) — sequences cross-attend | N/A (Unsloth is correct) | The plan identifies this as Phase 3 fix; not yet done in Grim |
| **LoRA backward fusion** | `silu_mul_backward` unimplemented on **`RocmDevice`** (primary backend, default `Err(Unimplemented)` at `backend.rs:229`). Implemented on Vulkan and Metal but not the primary ROCm target. | Lacks full backward fusion for all kernel pairs | SwiGLU backward is fused on Vulkan/Metal but not ROCm; CPU fallback is generic ops |
| **CPU fallback performance** | CPU paths use `to_cpu_vec_f32()` + Rust loops — O(n²) for GEMM | Triton kernels skip on CPU; falls to slow PyTorch eager | Both are unusably slow on CPU for large models |
| **Full-parameter finetuning** | `AutogradScope::LoRAOnly` hardwired — can't train all params | PEFT `get_peft_model` supports FullParam, but Unsloth's kernel patches assume LoRA/QLoRA | Neither optimizes for full FT at scale |
| **FP8 training at scale** | `fp8_gemm_rdna4` only for RDNA4 — no general FP8 path | `FbgemmFp8` / `cutlass` via external deps — no native kernel | FP8 is external/dependent, not built-in |
| **INT8 weight quantization in optimizer** | QGaLore stub (no INT8 weight quant) | INT8 weight quant via bitsandbytes but requires bnb | Both lack native INT8 weight-quantized AdamW in-core |
| **Pipeline parallelism** | No pipeline parallelism at all | No native PP — relies on `accelerate` | Neither has 1F1B scheduling |
| **Multi-node distributed** | RCCL/NCCL single-node only | `accelerate` multi-node (complex setup) | Both require external tooling for multi-node |
| **Dynamic shape compilation** | No recompilation for new shapes | Triton kernels recompile per shape (slow first call) | Neither has shape-specialized caching like vLLM |
| **Speculative decoding (draft)** | `tree_attention` kernel exists but no full draft model integration | No native speculative decoding — relies on EAGLE/Medusa repos | Both need external libs for the draft model side |
| **Quantization-aware training (QAT)** | QLoRA only (frozen quant, train LoRA) | Same — QLoRA is the paradigm | Neither does true QAT (updating base quant params) |

---

## 13. Architecture Diagram (Conceptual)

### Grim Training Pipeline (Rust-native)
```
GGUF Model File
  │
  ▼
GgufProvider → InjectionConfig (from metadata)
  │
  ▼
LlamaConfig + StreamingBlockForward (block-by-block weight loading)
  │
  ├─► Tape-based forward (AutogradScope::LoRAOnly)
  │     ├─ MatMul → recorded
  │     ├─ Add → recorded
  │     ├─ Scale → recorded
  │     └─ LoRAApply → recorded (param_id saved)
  │
  ├─► cross_entropy_loss (GPU kernel: grim_cross_entropy_forward)
  │     └─ Returns (loss_val, grad_tensor)
  │
  ├─► backward (tape walk, reverse-order)
  │     ├─ MatMul backward (CPU or GPU)
  │     ├─ Add backward (gradient routing)
  │     ├─ Scale backward
  │     ├─ LoRAApply backward (dora_forward in ops.rs)
  │     └─ accumulate_grad → TrainableParams.grad (device-pointer)
  │
  ├─► all_reduce_grads_weighted (RCCL, 1/N averaging, in-place device ptr)
  │
  ├─► Optimizer::step (AdamW)
  │     ├─ Device-resident mul_scalar/sqrt/recip (no D2H)
  │     └─ Update FP32 moment buffers
  │
  └─► TrainState sidecar save (.grim.train, GRIMTRN\x01 magic)
```

### Unsloth Training Pipeline (Python-native)
```
HuggingFace Hub / Local Checkpoint
  │
  ▼
FastLanguageModel.from_pretrained
  │      │
  ▼      ▼
torch.float16/bfloat16  │
  │      │
  ▼      │
bitsandbytes 4-bit NF4  │
  │      │
  ▼      │
LoRA A/B injection (PEFT LoraConfig)  │
  │      │
  ▼      │
UnslothTrainer (SFTTrainer subclass)
  │
  ├─► torch.utils.checkpoint.checkpoint (use_reentrant=True)
  │     ├─ RMSNorm (Triton: _rms_layernorm)
  │     ├─ QKV (Triton: LoRA_QKV)
  │     ├─ RoPE (Triton: _rope_embedding_QK)
  │     ├─ Attention (xFormers / FlashAttention / SDPA)
  │     ├─ SwiGLU (Triton: _fg_kernel)
  │     └─ Linear + LoRA (Triton: matmul_lora, fast_linear_forward)
  │
  ├─► Fused CE (Triton: Fast_CrossEntropyLoss, online logsumexp)
  │     └─ <1024 tokens: unsloth_fused_ce_loss; else: fast_cross_entropy_loss
  │
  ├─► torch.autograd.backward (dynamic graph, @torch.autograd.Function backward)
  │
  ├─► optimizer.step() (AdamW8bit or QGaLoreAdamW8bit via bitsandbytes)
  │     ├─ 8-bit moment states (blocksize-quantized)
  │     └─ GaLore projection: SVD → project grad → 8-bit Adam → project back
  │
  └─► save_pretrained (safetensors / pytorch_model.bin)
```

---

## 14. Detailed Code References

### Grim Key Files
| File | Lines | Responsibility |
|---|---|---|
| `Cargo.toml` | workspace manifest | 26 crates, `rocm = ["grim-backend-rocm/rccl"]` feature |
| `crates/grim-autograd/src/lib.rs` | 88 | Module declarations + re-exports |
| `crates/grim-autograd/src/tape.rs` | 80+ | `Tape`, `TapeEntry`, `TapeKind` (MatMul, Add, Scale, LoRAApply, SiluMul) |
| `crates/grim-autograd/src/backward.rs` | — | Reverse-mode backward walk |
| `crates/grim-autograd/src/ops.rs` | — | `dora_forward()`, LoRA backward math (CPU) |
| `crates/grim-autograd/src/loss.rs` | 297 | `cross_entropy_loss` (GPU+CPU), `fused_linear_cross_entropy_loss` |
| `crates/grim-autograd/src/preference_loss.rs` | 231+ | DPO, ORPO, KTO, SimPO, GRPO + autograd variants |
| `crates/grim-autograd/src/adamw.rs` | 400+ | `OptimizerKind` (14 variants), `Optimizer` enum, AdamW/AdamW8Bit/PagedAdamW/Lion/Adafactor/QGaLore stub |
| `crates/grim-autograd/src/param.rs` | 560+ | `TrainableParam`, `TrainableParams`, `all_reduce_grads_weighted` |
| `crates/grim-autograd/src/injection.rs` | 150+ | `LoRAInjectionPoint` (7 standard + Logits), `InjectionConfig` |
| `crates/grim-autograd/src/collate.rs` | — | `VarLenCollator` → `PackedBatch` |
| `crates/grim-backend-rocm/src/lib.rs` | 100+ | Re-exports, 26 kernel modules |
| `crates/grim-backend-rocm/src/device/roc_device.rs` | 6045 | Full `BackendDevice` impl (matmul, silu_mul, rms_norm, rope, attention, cross_entropy_gpu). `all_reduce` overridden (line 2955) but **CPU-only** (host round-trip, no RCCL). `estimate_gemm_latency_ms` implemented (WaveTune). `silu_mul_backward` **not overridden** |
| `crates/grim-backend-rocm/src/rccl.rs` | 670 | RCCL FFI: ncclAllReduce, ReduceScatter, AllGather, P2P, `RcclAllReduce::sum_gradients_device` with 1/N averaging |
| `crates/grim-backend-rocm/src/kernels/` | 27 files | 26 kernel modules + mod.rs |
| `crates/grim-tensor/src/backend.rs` | 600+ | `BackendDevice` trait (12 core ops + 10 GPU-specific with defaults) |
| `crates/grim-tensor/src/dtype.rs` | 50 | `Device` enum (Cpu, Rocm, Vulkan, Cuda, Metal) |
| `crates/grim-nn/src/modules.rs` | — | Linear, RmsNorm, Embedding, ColumnParallel/RowParallel (stubbed) |
| `crates/grim-nn/src/scythe2.rs` | 330+ | `Scythe2Linear` (WIP — CPU matmul, spec-only sharding) |
| `crates/grim-engine/src/scythe2.rs` | 60+ | `C2plrController` (stub: `decide_miss` and `update` are `todo!()`) |
| `crates/grim-engine/src/streaming_forward.rs` | 500+ | `StreamingBlockForward`, `GradientCheckpointBuffer` |
| `crates/grim-cli/src/train.rs` | 300+ | CLI training loop (GGUF → inject → stream → CE → backward → AdamW → sidecar) |
| `crates/grim-format/src/train.rs` | — | `.grim.train` sidecar (`GRIMTRN\x01`, JSON header + binary blobs) |
| `crates/grim-format/src/convert.rs` | 580+ | **Conversion pipeline**: GGUF→`.grim` with ROCm profile injection, EvoPress GPTQ per-tensor quantization, SmoothQuant, SpQR, SpinQuant |
| `crates/grim-format/src/gguf.rs` | — | GGUF reader/writer, `GgufProvider`, `GrimMetadata`, `GrimTensorExt` |
| `crates/grim-core/src/client.rs` | 1057 | **HF Hub download**: `download_model()` supports `hf:org/repo/file.gguf` URIs, `huggingface.co` URLs, Ollama registry; SSRF protection, SHA-256 verification |
| `crates/grim-cli/src/oxidizer.rs` | 1065 | **`grim oxidizer` CLI**: `cmd_oxidizer_convert` (EvoPress + SmoothQuant + SpQR + SpinQuant), `cmd_oxidizer_fuse` (ROCm fusion ops), `cmd_oxidizer_prepare` (training-ready format), `cmd_oxidizer_raven` (FP8 re-quantization) |
| `crates/grim-quant/src/lib.rs` | 3500+ | Q4_K (fixed), NF4 (fixed), Q8_0, Q5_K, Q6_K dequant/gemm; EvoPress search, Fisher calibration, SpQR, SpinQuant, SmoothQuant |

### Unsloth Key Files
| File | Responsibility |
|---|---|
| `pyproject.toml` | 15+ optional dependency groups (cu118–cu130, intelgpu, audio-torch210/280/290, etc.) |
| `unsloth/__init__.py` | Dual init: MLX path (torch-free) vs GPU path (torch + triton + bnb) |
| `unsloth/device_type.py` | Backend detection: CUDA → HIP → XPU → MLX; warp size + block size per arch |
| `unsloth/models/llama.py` | Llama fast forward/inference: attention, SwiGLU, RMSNorm, CE (≤1024 tokens fused) |
| `unsloth/models/_utils.py` | `prepare_model_for_kbit_training`, `patch_model_and_tokenizer`, `validate_loftq_config`, `torch_compile_options` |
| `unsloth/models/dpo.py` | DPO training with preference pairs |
| `unsloth/models/rl.py` | RL training replacements |
| `unsloth/kernels/__init__.py` | Exports: cross_entropy, rms_layernorm, layernorm, rope_embedding, swiglu, geglu, fast_lora, fp8, flex_attention |
| `unsloth/kernels/cross_entropy_loss.py` | `Fast_CrossEntropyLoss` Triton: online logsumexp, chunked for vocab > 65K, Gemma softcap, Cohere scaling |
| `unsloth/kernels/swiglu.py` | `_fg_kernel` (fwd: `silu(e)*g`), `_DWf_DW_dfg_kernel` (backward: 3-output fused: `h`, `df`, `de`) |
| `unsloth/kernels/rope_embedding.py` | `_rope_embedding_QK` (in-place QK, backward negation), `_rope_embedding` (grouped, ROPE_GROUP_SIZE=4) |
| `unsloth/kernels/rms_layernorm.py` | `_rms_layernorm_forward`/`_backward`, `_gemma_rms_layernorm_forward` (Gemma: `(W+1.0)`), float32 accumulation |
| `unsloth/kernels/fast_lora.py` | `LoRA_MLP`, `LoRA_QKV`, `LoRA_W`, `fast_linear_forward` (dequant+LoRA add), `matmul_lora` |
| `unsloth/kernels/utils.py` | `fast_dequantize` (bnb cdequantize), `fast_gemv`, `QUANT_STATE`, `calculate_settings`, `is_cdna`/`is_rdna` |
| `unsloth/kernels/flex_attention.py` | Flex attention masks (sliding window, softcapping) |
| `unsloth/kernels/fp8.py` | FP8 linear patching (FbgmemFP8Linear/FP8Linear) |
| `unsloth/optimizers/q_galore_adamw.py` | `QGaLoreAdamW8bit(Optimizer2State)`, `make_q_galore_param_groups`, `install_weight_quant_hooks` |
| `unsloth/optimizers/q_galore_projector.py` | `GaLoreProjector` (Halko randomized SVD, adaptive schedule, INT4 quant) |
| `unsloth/trainer.py` | `UnslothTrainer(SFTTrainer)`, `UnslothTrainingArguments`, `QGaloreConfig` |
| `unsloth/utils/packing.py` | `build_sdpa_packed_attention_mask`, `mask_packed_sequence_boundaries`, `enable_sample_packing`, `patch_hybrid_linear_attention_varlen` |

---

## 15. Unsloth-Parity Optimizations Plan (Status)

From `old/unsloth-parity-optimizations.md` — 5-phase plan to close Grim's gaps:

| Phase | Task | Status | What it does |
|---|---|---|---|
| **Phase 0a** | Fix Q4_K writer | ✅ Complete (tests written, per plan) | Replace degenerate hardcoded scales with per-sub-block 6-bit scales |
| **Phase 0b** | Fix NF4 codebook | ✅ Complete (tests written, per plan) | Replace 14-bucket ladder with canonical 16-level NormalFloat codebook |
| **Phase 1** | SwiGLU fused backward + GPU wiring | ✅ Kernel string added (`grim_silu_mul_backward`), ✅ Rust wrapper scaffolded, ✅ **Implemented on Vulkan and Metal backends**, ❌ **Still unimplemented on `RocmDevice`** (default `Err(Unimplemented)` at `backend.rs:229`) | Add HIP backward kernel + wire FFN to GPU (drop CPU round-trip) |
| **Phase 2** | GPU cross-entropy | ✅ Kernel strings added (`grim_cross_entropy_forward`/`_backward`), ✅ GPU dispatch in `loss.rs`, ✅ CPU fallback | Move CE from CPU double-loop to fused GPU kernel |
| **Phase 3** | Packed-attention mask | ❌ Not done | `block_diagonal_causal_mask` + `boundary_loss_mask` to fix all-true attention mask bug |
| **Phase 4** | QGaLore optimizer | ❌ Not done | Replace `Error::Unimplemented` stub with real `GaoreProjector` + Halko SVD + 8-bit AdamW |

---

## 16. Final Verdict

**Grim's strengths are architectural:**
1. **Scoped LoRA-only autograd** is fundamentally more memory-efficient than PyTorch's full-graph tracing. The tape records only the 14 adapter matrices per layer, not the entire activation graph.
2. **Rust compile-time safety** eliminates an entire class of runtime errors (missing imports, type mismatches, device placement bugs).
3. **No Python dependency** means no CUDA/cuDNN/torch version hell — `pyproject.toml`'s 15+ optional dependency groups evaporate.
4. **SCYTHE-2 multi-GPU design** is research-grade and superior to standard DDP on paper (per-layer routing, CommFuse P2P, runtime topology reconfiguration). Parts of the spec are now compiled — `C2plrController` methods (`decide_miss`/`update`) implement WaveTune bilinear latency prediction and Lagrangian dual ascent; `RocmDevice::all_reduce` and `estimate_gemm_latency_ms` are overridden. However, the actual distributed path is still **CPU-only** (host round-trips, no RCCL) — not production-ready for real multi-GPU training.
5. **HIP kernel JIT** compiles to HSACO with caching. 26 kernel modules cover a broad range including Mamba, RWKV, speculative decoding, and FP8.
6. **FFI safety patterns** are exemplary: `#[repr(transparent)]` newtypes for opaque handles, status-code wrapping, SONAME fallback for dynamic loading, explicit `unsafe impl Send/Sync` with documented invariants.

**Grim's fatal weaknesses:**
1. **QGaLore is a stub** (`Error::Unimplemented`). This is Unsloth's #1 selling feature for memory-efficient training.
2. **Multi-GPU `all_reduce` is CPU-only.** `RocmDevice` now overrides `all_reduce` (line 2955), but it's a host round-trip — element-wise sum via `to_cpu_vec_f32()` + Rust loops. No actual RCCL multi-GPU communication. The `BackendDevice::all_reduce` default for non-ROCm backends still returns `Err(Unimplemented)`. SCYTHE-2's `C2plrController` `decide_miss`/`update` are no longer `todo!()` stubs (WaveTune bilinear + Lagrangian dual ascent now compiled), but `Scythe2Linear::forward_placed` still uses CPU-side matmul. The spec has been partially compiled, but no true device-resident distributed training path exists.
3. **`silu_mul_backward` is unimplemented on ROCm.** The SwiGLU backward pass falls back to generic ops on the primary backend. (Implemented on Vulkan and Metal backends, but not RocmDevice — the primary target.)
4. **Packed attention mask is all-true (bug).** Sequences cross-attend during training, silently corrupting gradients.
5. **No batch size / gradient accumulation in CLI.** `TrainOptions` lacks basic training loop parameters.
6. **Requires `.grim` conversion before training.** The trainer loads `.grim` files, not raw HF formats. While `download_model` can fetch from HF Hub and `convert_to_grim` accepts `.safetensors`/`.bin`/`.gguf`, the two-step (download → convert → train) workflow adds friction compared to Unsloth's direct in-memory loading. This is intentional architecture, not a gap — but it does add operational overhead.
7. **CPU fallback is O(n²).** `to_cpu_vec_f32()` + Rust loops for GEMM is unusably slow.

**Unsloth's strengths are practical:**
1. **It works today.** DDP, FSDP, QGaLore, PPO, DPO — all production-tested and integrated with the PyTorch ecosystem.
2. **PEFT ecosystem.** Full range of parameter-efficient methods, not just LoRA.
3. **HuggingFace integration.** Downloads models, datasets, tokenizers, saves checkpoints in standard formats. Grim also downloads from HF Hub (`client.rs`), but via a two-step convert-then-train pipeline.
4. **Triton kernel autotuning.** Per-kernel block size/warp count selection at runtime.
5. **`torch.compile` integration.** End-to-end graph fusion (though Unsloth explicitly disables it on some kernels).

**Unsloth's weaknesses:**
1. **Python dependency hell.** 15+ optional pip dependency groups, CUDA version matrix, torch/torchvision/torchaudio version constraints.
2. **Full-graph autograd overhead.** Traces every tensor op, not just adapters.
3. **MLX path is torch-free but limited.** Separate code path, not as mature as the CUDA path.
4. **No native serving stack.** Relies on vLLM/SGLang/TGI for inference.

**Bottom line:**
- If you need **memory-efficient QLoRA training that works today, on any hardware, with any HF model** → **Unsloth wins decisively.** It has QGaLore, FSDP, PPO, DPO, CPU offloading, and a battle-tested training loop.
- If you need **maximum performance per watt, compile-time safety, no Python dependency, and can live without QGaLore/FSDP** → **Grim shows the stronger architecture** but is not yet production-ready. The SCYTHE-2 plan is excellent but unimplemented; the QGaLore stub and broken multi-GPU make it a research prototype, not a training tool.
- Both achieve **algorithmic parity** on QLoRA training math (frozen 4-bit base + LoRA adapters, fused CE, fused SwiGLU, RMSNorm, RoPE, gradient checkpointing, sample packing). The difference is **implementation maturity**: Unsloth ships, Grim plans.
