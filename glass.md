# Grim-Backend-Metal Parity Plan (mirrors ROCm)

> Audit of `grim-core` + `grim-backend-metal` vs `grim-backend-rocm`, with a phased implementation plan to bring Metal to feature parity.
> Written 2026-08-30. Author: [REDACTED].

---

## 1. Executive summary

`grim-backend-metal` (6,013 lines of Rust + 63 MSL compute kernels) covers the bulk of the `grim-backend-rocm` feature surface, but is **missing 5 backward-pass kernels, 2 fused GEMM wrappers, 1 attention decode path, 2 quantized-GEMM wrappers, and multiple dequantizer / GEMM family wrappers** that ROCm exposes through `roc_device.rs`. The `grim-core` crate itself is backend-agnostic and healthy; the gap is purely in the Metal backend dispatch surface and its `.msl` shader catalog.

**Top-line numbers:**

| Surface | ROCm (`grim-backend-rocm`) | Metal (`grim-backend-metal`) | Delta |
|---|---|---|---|
| `src/kernels/*.rs` modules | 53 | n/a (uses `src/kernels.msl`, 63 `kernel void` functions) | different representation |
| `BackendDevice` trait methods | ~70 | ~55 | ~15 missing |
|| Backward passes (`*_backward`) | 6 | 6 (`silu_mul_backward`, `quantized_matmul_backward_dx`, `rmsnorm_backward`, `rope_backward`, `softmax_backward`, `embedding_backward`) | **0 — parity** |
|| Fused GEMM wrappers | `fused_mxfp4_gemm_qk_norm_rope_kv`, `fused_rmsnorm_mxfp4_gemm*`, `fused_linear_cross_entropy_*` | `fused_linear_cross_entropy_forward` + `_backward` dispatchers wired (5 of 5: kernel + pipeline + dispatch all present) | **4 remaining (mxfp4, rmsnorm_mxfp4)** |
|| Attention decode | `flash_decode` (stage1+stage2) | `flash_decode` dispatcher in `lib.rs` wrapping `grim_flash_decode_split_k` + `grim_softmax_merge` (complete) | **parity** |
| Quantized GEMM families | `q2k/q3k/q4k/q5k/q6k/q8_0/iq_*` + AWQ/GPTQ + `bf16`/`compressed` | `q4k/q5k/q6k/q8_0` + `iq2xxs/iq2xs/iq2s/iq3xxs/iq3s/iq4nl/iq4xs` dequant only | **−10 missing** |
| Dequant wrappers (host) | full + `dequant_w4a16`, `dequant_wna16` | partial | partial |

---

## 2. grim-core audit

`grim-core` is the backend-neutral crate. It defines the architecture enum, model loaders, hyperparams, KV-cache, sampler, session, and the `BackendDevice`/`CoreTensorOps`/`BackendStorage` traits in `grim-tensor`. No GPU-specific logic lives here — the parity work is entirely in the backend crates.

### 2.1 Module map (`crates/grim-core/src`)

| File | Lines | Exports / role |
|---|---|---|
| `lib.rs` | ~40 | Re-exports all public modules; `Backend`, `RuntimeEnv`, `Error`, `Result`, `Sampler`, `Session`, `KvCache`, `Model*`, `TensorRole`, `TensorNamingRegistry`, `ModelArchitecture`, `ModelEntry` |
| `architecture.rs` | 1,774 | `ModelArchitecture` enum (~90 variants incl. `SmolLm2`, `Llama4`, `Qwen3Moe`, `Gemma4`, `Mistral3/4`, `ExaoneMoe`, `PhiMoe`, `JinaBertV2/3`, `NomicBertMoe`, etc.), `TensorNamingRegistry`, `TensorRole`, `from_str`/`as_str` |
| `hyperparams.rs` | 445 | `ArchHyperparameters`, `HyperparameterExtractor`, `MetadataLookup` |
| `model.rs` | 197 | `Model`, `ModelConfig`, `CausalLm`, `Encoder`, `EncoderDecoderLm`, `DiffusionModel`, `StatefulSequence`, `ModalityHint`, `AudioVocoder`, `VoiceConversionModel`, `NoiseScheduler`, `SsmState` |
| `session.rs` | 351 | `Session`, `DeterminismMode`, `GraphBuilder` |
| `kv_cache.rs` | 260 | `KvCache` |
| `sampler.rs` | 493 | `Sampler` |
| `catalog.rs` | ~470 | `ModelEntry`, `GgufEnrichment`, `list_local_models`, `resolve_model_path`, `save`, `load_for`, `enrich_from_gguf`, `self_heal_sidecar` |
| `client.rs` | ~1,100 | `download_model`, `download_model_with_progress`, `load_login_token`, `save_login_token`, `delete_model`, `set_default_model`, `check_model_cache`, HF token handling, model search paths |
| `disagg_placement.rs` | — | `GpuCapability`, `PlacementAdvice`, `advise_placement`, `bandwidth_split`, `validate` |
| `memory_certificate.rs` | — | `MemoryCertificate`, `ModelInventory`, `AuthorityGrade`, `BoundaryVector`, `ExactnessContract` |
| `env_config.rs` | — | `Backend`, `RuntimeEnv` |
| `error.rs` | — | `Error`, `Result`, `TensorError` |
| `paths.rs` | — | config/log/model/plugin dirs |
| `rng.rs` | — | PRNG |
| `config.rs` | — | config helpers |

### 2.2 Core traits (in `grim-tensor/src/backend.rs`)

`CoreTensorOps` — every backend MUST implement: `zeros`, `matmul`, `matmul_with_solution`, `transpose_2d`, `add`, `mul`, `silu_mul`, `rms_norm`, `rms_norm_inplace`, `softmax`, `embedding`, `from_cpu`, `advise`.

`BackendDevice` — adds: `mul_scalar`, `add_scalar`, `sub_scalar`, `div_scalar`, `sub`, `reduce_sum`, `reduce_max`, `argmax`, `sqrt`, `recip`, `sample_on_device`, `sage_attention`, `kv_dequant_attention`, `rope`, `rerope`, `mla_q_kv_norm_split`, `mla_absorbed_decode`, `qkv_attention`, `qkv_attention_alibi`, `qkv_attention_paged`, `tree_attention`, `flash_attention`, `cross_attention`, `silu_mul_quantize`, `fused_add_rms_norm`, `fused_mxfp4_gemm_qk_norm_rope_kv`, `broadcast_bias`, `scale_bias_epilogue`, `silu_mul_backward`, `rmsnorm_backward`, `rope_backward`, `softmax_backward`, `embedding_backward`, `lora_accumulate`, `fused_adamw_step`, `fused_lion_step`, `fused_madam_step`, `quantized_matmul`, `quantized_matmul_backward_dx`, `quantize`, `fused_quant_gemm`, `short_conv1d_causal_step`, `kda_gated_delta_rule_step`, `selective_scan`, `rwkv_time_mix`, `rwkv_channel_mix`, `all_reduce`, `comm_fuse_reduce`, `estimate_gemm_latency_ms`, `from_cpu_bytes`, `alloc_storage`, `copy_slice_into`, graph capture (`begin_graph_capture`/`end_graph_capture`/`replay_graph`/`has_captured_graph`).

Default impls return `Err(Unimplemented)` — so a backend can ship incrementally.

### 2.3 grim-core findings

- `ModelArchitecture` enum is comprehensive and up to date (includes `SmolLm2`, `Llama4`, `Qwen3Moe`, `Gemma4`, `Mistral3/4`, `ExaoneMoe`, `PhiMoe`, `JinaBertV2/3`, `NomicBertMoe`).
- `SmolLm2::load_tp` at `grim-models/transformer/src/smollm2.rs:60` wraps `Llama::load_tp` with `partial_rotary_factor: 1.0` and `yarn: None`, and explicitly promotes GGUFs whose architecture string is `llama` but tensor signature shows `output_norm` with no `output.weight`.
- `model_loader.rs` dispatch covers `SmolLm2` at line 2561 — loading works.
- No backend-specific code in `grim-core` — parity is purely a backend-crate concern.

---

## 3. grim-backend-metal audit

### 3.1 Source layout

```
crates/grim-backend-metal/
  Cargo.toml
  src/
    lib.rs                  (~6,013 lines) — main BackendDevice impl + dispatchers
    caps.rs                 (4,238 lines)  — MetalCaps probe + feature flags
    autotune.rs             (3,484 lines)  — MetalTileConfig, GemmOp (Attention/Ffn/LmHead/Other), shape class, tile search
    kernels/mod.rs          (764 lines)    — MSL source aggregator
    kernels.msl             (6,913+ lines) — 63 Metal compute kernels (kernel void grim_*)
    kernels/quantization.msl
    kernels/attention.msl
    kernels/gemm.msl
    kernels/moe.msl
    kernels/optimizer.msl
    kernels/speculative.msl
    kernels/math.msl
```

### 3.2 BackendDevice impl structure

`impl grim_tensor::BackendDevice for MetalDevice {}` at line 4811 is **empty** — all trait methods live inside the free-standing `impl MetalDevice` block (line 549 + 4814). This is unusual but functional.

Key dispatchers in `lib.rs`:
- `matmul_with_op` (1524) — generic GEMM with autotune via `MetalTileConfig`
- `qkv_attention` (2724) — standard attention
- `qkv_attention_paged` (2753) — paged attention
- `mla_absorbed_decode` (3013) — MLA decode
- `mla_q_kv_norm_split` (3085) — MLA QK norm
- `rope` (3158) — RoPE (with YaRN via `grim_rope_yarn`)
- `sage_attention` (2990) — delegates to `qkv_attention`
- `flash_attention` (3409), `cross_attention` (3439), `tree_attention` (2880)
- `kv_dequant_attention` (2540)
- `silu_mul` (2005), `rms_norm` (2033), `softmax` (2127), `embedding` (2224), `add` (1949), `mul` (1977)
- `transpose_2d` (1798), `zeros` (1883), `from_cpu` (2333), `from_cpu_bytes` (4744)
- `quantized_matmul` (3672), `quantize` (3662), `quantized_matmul_backward_dx` (4140)
- `silu_mul_backward` (3466)
- `all_reduce` (4480), `comm_fuse_reduce` (4597)
- `selective_scan` (4325), `rwkv_time_mix` (4386), `rwkv_channel_mix` (4440)
- `fused_adamw_step` (3578), `fused_lion_step` (3620), `fused_add_rms_norm` (1040 — **pub fn**, not trait)
- `moe_fused_dispatch` (1350 — pub fn, WI-M5)
- `estimate_gemm_latency_ms` (4723), `non_multiple_of_eight_falls_back` (5910), `eligible_shapes_select_variant` (5928)

### 3.3 MSL kernel catalog (`kernels.msl`, 63 kernels)

Elementwise: `grim_add`, `grim_mul`, `grim_silu_mul`, `grim_mul_scalar`, `grim_sqrt`, `grim_recip`, `grim_sub`, `grim_reduce_sum`, `grim_reduce_max`, `grim_argmax`.

Norm/activation: `grim_rms_norm`, `grim_add_rms_norm`, `grim_silu_mul_backward`.

Quantize/dequant: `grim_quant_q8_0`, `grim_quant_fp8`, `grim_quant_mxfp4`, `grim_quant_mxfp8`, `grim_quant_q4k`, `grim_dequant_q8_0`, `grim_dequant_q4k`, `grim_dequant_fp8`, `grim_dequant_mxfp4`, `grim_dequant_mxfp8`, `grim_dequant_iq2xxs`, `grim_dequant_iq2xs`, `grim_dequant_iq2s`, `grim_dequant_iq3xxs`, `grim_dequant_iq3s`, `grim_dequant_iq4nl`, `grim_dequant_iq4xs`.

GEMM: `grim_matmul`, `grim_matmul_split_k`, `grim_reduce_split_k`, `grim_quantized_matmul_q8_0`, `grim_quantized_matmul_residualpacked`, `grim_quantized_matmul_backward_q8_0`, `grim_all_reduce`, `grim_comm_fuse_reduce`, `grim_matmul_simdgroup_f32`, `grim_matmul_simdgroup_f32_16`.

Attention: `grim_qkv_attention`, `grim_qkv_attention_paged`, `grim_tree_attention`, `grim_kv_dequant_attention`, `grim_qkv_attention_paged_dequant`, `grim_mla_decode`, `grim_sage_attention`, `grim_mrope`, `grim_rope`, `grim_rope_yarn`.

MoE/fused: `grim_moe_fused_dispatch`, `grim_fused_dequant_gemm_q4k`, `grim_fused_dequant_gemm_fp8`, `grim_fused_dequant_gemm_q8_0`, `grim_fused_adamw`, `grim_fused_lion`, `grim_fused_linear_ce`.

Other: `grim_embedding`, `grim_quantized_matmul_q8_0` (duplicate name in catalog), `grim_flash_decode_split_k`, `grim_softmax_merge`, `grim_speculative_acceptor`, `grim_marlin_gemm`.

---

## 4. Parity gap analysis (Metal vs ROCm)

### 4.1 Missing backward passes (critical — breaks training/finetune)

|| Missing on Metal | ROCm (`roc_device.rs` line) | MSL kernel needed |
||---|---|---|
|| `rmsnorm_backward` | 4139 | **`grim_rmsnorm_backward` — implemented in `.msl` + dispatch in `lib.rs`** |
|| `rope_backward` | 4197 | **`grim_rope_backward` — implemented in `.msl` + dispatch in `lib.rs`** |
|| `softmax_backward` | 4246 | **`grim_softmax_backward` — implemented in `.msl` + dispatch in `lib.rs`** |
|| `embedding_backward` | 4293 | **`grim_embedding_scatter_add` + `grim_zeros_f32` — MSL + dispatch in `lib.rs`** |
|| `quantized_matmul_backward_dx` (full) | 5052 | Partially in `.msl` (Q8_0 only); widen to non-Q8_0 formats |

Note: Metal has `silu_mul_backward` (3466) and a partial `quantized_matmul_backward_dx` (4140, Q8_0 only). ROCm has `silu_mul_backward`, `rmsnorm_backward`, `rope_backward`, `softmax_backward`, `embedding_backward`, `quantized_matmul_backward_dx` (with multi-format support).

### 4.2 Missing fused GEMM wrappers

| Missing on Metal | ROCm line | Notes |
|---|---|---|
|| `fused_linear_cross_entropy_forward` | 14645 | **`grim_fused_linear_ce` kernel in `.msl` + dispatcher in `lib.rs`** (Phase 2 complete) |
|| `fused_linear_cross_entropy_backward` | 14729 | **`grim_fused_linear_ce_backward` kernel in `.msl` + dispatcher in `lib.rs`** — fully wired: kernel at `kernels.msl:1987`, pipeline slot `fused_linear_ce_backward` registered in `MetalPipelines` + `get_pipeline` call in `MetalContext::get()`, dispatch fn `fused_linear_cross_entropy_backward` in `lib.rs` |
| `fused_mxfp4_gemm_qk_norm_rope_kv` | — | MXFP4 GEMM + QK-norm + RoPE |
| `fused_rmsnorm_mxfp4_gemm` | — | |
| `fused_rmsnorm_mxfp4_gemm_rope_kv` | — | |

### 4.3 Missing attention / decode paths

| Missing on Metal | ROCm line | Notes |
|---|---|---|
| `flash_decode` (stage1 + stage2) | 11454, 11635 | `grim_flash_decode_split_k` exists in `.msl` but has **no dispatch wrapper** in `lib.rs` |
| `extend_attention` | — | prefill/extend path |
| `prefill_compact` | — | compact prefill |
| `preshuffled_attention` | — | preshuffled KV layout |

### 4.4 Missing quantized GEMM families

ROCm splits quantized GEMM into per-format kernels (`q2k`, `q3k`, `q4k`, `q5k`, `q6k`, `q8_0`, `iq2xxs`, `iq2xs`, `iq2s`, `iq3xxs`, `iq3s`, `iq4nl`, `iq4xs`, `fp8`, `mxfp4`, `mxfp8`, `w4a16_marlin`, `awq`, `gptq`, `wna16`, `bitnet`). Metal currently only dispatches `q4k`, `q5k`, `q6k`, `q8_0` with dequant-only support for the IQ family.

Missing device-side wrappers:
- `q2k_gemm`, `q3k_gemm` (2-bit / 3-bit quant)
- `q4k_gemm`, `q5k_gemm`, `q6k_gemm` (4/5/6-bit — exist in `.msl` but no `fn` wrapper)
- `fp8_standalone`, `mxfp4_gemm`, `mxfp_standalone`
- `awq_gemm`, `gptq_gemm`, `compressed_gemm`, `bitnet_gemm`
- `iq_gemm`, `iq_dequant` device paths
- `wmma_gemm` (RDNA WMMA path)
- `matmul_lm_head` (ROCm has explicit `matmul_lm_head` at 13879; Metal has `matmul_with_op` which is a different API)

### 4.5 Missing utility / dequant wrappers

| Missing on Metal | Notes |
|---|---|
| `dequant_w4a16_blob_to_f32`, `dequant_wna16_to_f32`, `dequant_wna16_int_to_f32` | Marlin / WNA16 dequant |
| `awq_segment_offsets`, `gptq_segment_offsets` | AWQ/GPTQ segment helpers |
| `estimate_gemm_latency_ms` | Metal has at 4723 ✓ |
| `allocator_stats`, `alloc_scythe_ring_bytes`, `copy_cross_device_bounce` | Multi-GPU / NCCL-side (less relevant for single-device Metal) |

### 4.6 Structural / API differences

1. **`BackendDevice` impl is empty** (`impl grim_tensor::BackendDevice for MetalDevice {}` at 4811). All methods live in `impl MetalDevice`. ROCm puts them directly in `impl BackendDevice for RocmDevice`. Both work, but the empty-impl pattern is confusing for new contributors.

2. **`fused_add_rms_norm` is `pub fn` not trait method** (line 1040). ROCm has it as both `pub fn` and delegates from trait. Metal's is only reachable via `MetalDevice::fused_add_rms_norm`, not through the `BackendDevice` trait — breaks polymorphism for callers that use `&dyn BackendDevice`.

3. **`matmul_with_op` vs `matmul_lm_head`**. Metal uses `matmul_with_op(op: GemmOp)`, ROCm uses `matmul_lm_head`. The `GemmOp::LmHead` -> `TLOLog` shape class is present in Metal's autotune. The naming mismatch should be reconciled (add `matmul_lm_head` as an alias or rewire callers).

4. **Autotune parity**. Metal's `MetalTileConfig` / `MetalTileConfig::search_tile_config_measured` is feature-equivalent to ROCm's `rocm_device::autotune_attention_block_dim` + `wmma_route_decision`. Metal already does on-GPU measurement (good). Missing: `wmma_gemm` variant path (`should_use_wmma_path`, `non_f16_output_skips_wmma`, `disabled_config_skips_wmma`) — Metal uses `simdgroup_matrix` instead, which is fine, but the `wmma_gemm` kernel family (and its autotune) is missing.

5. **Graph capture**. `BackendDevice` trait has `begin_graph_capture`/`end_graph_capture`/`replay_graph`/`has_captured_graph`. Metal likely falls back to the default (no-op) — check Metal's impl. ROCm wires these to HIP graph capture.

6. **`from_cpu_managed`**. ROCm has `pub fn from_cpu_managed` (1706) for ROCm-managed-memory uploads. Metal uses `new_buffer_with_bytes`/`from_cpu` instead — equivalent via `MTLResourceOptions::StorageModeShared`.

---

## 5. Implementation plan — phases

### Phase 1: Critical backward passes (unblocks training/finetune)

**Goal:** Add the 4 missing backward kernels to `kernels.msl` and wire dispatchers in `lib.rs`.

1. **`rmsnorm_backward`**
   - Add `grim_rmsnorm_backward` kernel to `kernels.msl` (mirrors ROCm `src/kernels/rmsnorm_backward.rs` — warp-per-row 32-lane reduce).
   - Register `rmsnorm_backward` pipeline in `MetalDeviceInner` (line ~290).
   - Add `fn rmsnorm_backward(...)` to `impl MetalDevice` (mirrors `silu_mul_backward` pattern at 3466).
   - Wire through `BackendDevice` trait (currently returns `Unimplemented`).

2. **`rope_backward`**
   - Add `grim_rope_backward` kernel to `kernels.msl`.
   - Register + dispatch in `lib.rs`.

3. **`softmax_backward`**
   - Add `grim_softmax_backward` kernel to `kernels.msl`.
   - Register + dispatch.

4. **`embedding_backward`**
   - Add `grim_embedding_backward` kernel to `kernels.msl` (scatter-add: `dweight[token_ids[t], :] += out_grad[t, :]`).
   - Register + dispatch.

5. **Widen `quantized_matmul_backward_dx`**
   - Extend beyond Q8_0 to support `q4k`, `q5k`, `q6k`, and IQ formats (matches ROCm's multi-format backward).
   - Add `grim_quantized_matmul_backward_q4k`, `_q5k`, `_q6k` kernels to `.msl`.

**Verification:** Run `cargo test --package grim-backend-tests --test parity_cpu_vulkan_metal` and add unit tests in `grim-backend-tests/tests/kernel_numerical_parity.rs` for each backward pass.

### Phase 2: Fused GEMM wrappers (unblocks lm_head / cross-entropy training)

**Goal:** Add `fused_linear_cross_entropy_*`, `fused_mxfp4_gemm_qk_norm_rope_kv`, `fused_rmsnorm_mxfp4_gemm*`.

1. `fused_linear_cross_entropy_forward` / `_backward` — wraps `grim_fused_linear_ce` kernel (already in `.msl`). Add `fn` dispatchers in `lib.rs`.
2. `fused_mxfp4_gemm_qk_norm_rope_kv` — wraps the MXFP4 QKV GEMM + QK-norm + RoPE path. Add to `.msl` if not already present; wire dispatcher.
3. `fused_rmsnorm_mxfp4_gemm` / `_rope_kv` — fused RMSNorm + MXFP4 GEMM (+ optional RoPE). Add kernels + dispatchers.

**Verification:** End-to-end with a LoRA/GQA model on Metal; compare numeric output vs ROCm reference.

### Phase 3: Attention decode & prefill paths

**Goal:** Wire `flash_decode`, `extend_attention`, `prefill_compact`, `preshuffled_attention`.

1. `flash_decode` — `flash_decode` dispatcher in `lib.rs` wrapping `grim_flash_decode_split_k` + `grim_softmax_merge` (complete — shipped with CPU fallback).
2. `extend_attention` — add kernel + dispatcher (compact attention for long-context extend).
3. `prefill_compact` — compact prefill for long sequences.
4. `preshuffled_attention` — preshuffled KV layout for decode efficiency.

### Phase 4: Quantized GEMM family expansion

**Goal:** Bring Q2K/Q3K/fp8/mxfp4/mxfp8/AWQ/GPTQ/WNA16/BitNet device-side paths up to ROCm parity.

1. Add `q2k_gemm`, `q3k_gemm` kernels to `.msl`.
2. Add `fp8_standalone`, `mxfp4_gemm`, `mxfp_standalone` wrappers.
3. Add `awq_gemm`, `gptq_gemm`, `compressed_gemm`, `bitnet_gemm` wrappers.
4. Add `iq_gemm` / `iq_dequant` device paths.
5. Add `wmma_gemm` (RDNA WMMA) — note: Metal uses `simdgroup_matrix` instead; either alias or implement a parallel simdgroup path.
6. `matmul_lm_head` — **complete**: `pub fn matmul_lm_head` added at `lib.rs:2040` (alias for `matmul_with_op(GemmOp::LmHead)`, mirrors ROCm `roc_device.rs:13879`). |

### Phase 5: Utility & architectural cleanup

1. Wire `BackendDevice` trait methods through for: `rms_norm_inplace`, `lora_accumulate`, `fused_adamw_step`, `fused_lion_step`, `fused_madam_step`, `fused_quant_gemm`, `short_conv1d_causal_step`, `kda_gated_delta_rule_step`, `selective_scan`, `rwkv_*`, `sample_on_device`, `blend_kv_rope`, `scale_bias_epilogue`, `broadcast_bias`.
2. Reconcile `fused_add_rms_norm` to be accessible via `&dyn BackendDevice` (or document the `pub fn` path).
3. Reconcile `matmul_with_op` vs `matmul_lm_head` naming.
4. Add graph-capture wiring (HIP-graph equivalent via Metal command buffers).
5. Port `estimate_gemm_latency_ms`, `allocator_stats`, `copy_slice_into` wrappers.

### Phase 6: Tests & benchmarks

1. Expand `kernel_numerical_parity.rs` to cover all new kernels (backward passes, fused GEMM, flash_decode).
2. Add `parity_cpu_metal.rs`-style tests for each new kernel.
3. Port ROCm autotune benchmark harness (`tests/autotune.rs`) to Metal (`tests/autotune.rs`).
4. Add a `cargo test --package grim-backend-metal` smoke suite covering all dispatched kernels.

---

## 6. Workstream priorities (what to tackle first)

1. **Phase 1 (backward passes)** — highest ROI. Without these, training/finetune is impossible on Metal. ~1 week.
2. **Phase 2 (fused lm_head / cross-entropy)** — unblocks next-token prediction training. ~1 week.
3. **Phase 4 (quantized GEMM family)** — unblocks GGUF Q4_K/Q5_K/Q6_K/MXFP4/MXFP8 inference on Metal. ~2 weeks.
4. **Phase 3 (flash_decode / extend / prefill)** — unblocks long-context serving. ~1 week.
5. **Phase 5 (utilities + cleanup)** — polish. ~1 week.
6. **Phase 6 (tests)** — ongoing.

---

## 7. Risks & mitigations

- **MSL syntax vs CUDA/HIP**: Metal's `.msl` is a different shader language. Kernels must be rewritten (not ported line-for-line). Use the existing `.msl` kernels as templates — the elementwise/norm/GEMM patterns are already well-established in `kernels.msl`.
- **Autotune on Metal**: Metal's `MTLComputePipelineState` compilation is expensive; the `search_tile_config_measured` path already measures on-GPU latency. Keep using it; don't reimplement.
- **IQ format bugs**: The existing memory notes call out `q2k` (BLOCK_BYTES=82≠84) and `iq4nl` fabricated. Fix the `q2k` layout before expanding IQ support in Metal.
- **`grim` `gen` reserved keyword**: Rust 2024 reserves `gen`; rename any variables named `gen` to `rng` in new kernels.
- **`primus` layer init**: Vulkan tests hang due to primus; exclude with `cargo test --workspace --exclude grim-backend-vulkan` when running the full suite on Metal.
- **Rust 2024 `gen` reserved**: Rename any `gen` vars to `rng` in new kernels.

---

## 8. Verification checklist

- [ ] `cargo test --package grim-backend-metal` passes
- [ ] `cargo test --package grim-backend-tests --test parity_cpu_vulkan_metal` passes (Metal-specific tests)
- [ ] `cargo test --workspace --exclude grim-backend-vulkan` passes
- [ ] Each new backward kernel has a golden numerical-parity test vs CPU reference
- [ ] Each new fused GEMM kernel has a numeric-parity test vs ROCm reference output
- [ ] `cargo clippy --package grim-backend-metal` clean
- [ ] `cargo fmt --package grim-backend-metal` applied

---

## 9. File manifest (for reference)

- `/D/rex/projects/grim/Cargo.toml` — workspace root
- `/D/rex/projects/grim/crates/grim-core/Cargo.toml`, `src/lib.rs`, `src/architecture.rs`, `src/hyperparams.rs`, `src/model.rs`, `src/session.rs`, `src/kv_cache.rs`, `src/sampler.rs`
- `/D/rex/projects/grim/crates/grim-backend-metal/Cargo.toml`, `src/lib.rs`, `src/caps.rs`, `src/autotune.rs`, `src/kernels.msl`, `src/kernels/mod.rs`, `src/kernels/quantization.msl`, `src/kernels/attention.msl`, `src/kernels/gemm.msl`, `src/kernels/moe.msl`, `src/kernels/optimizer.msl`, `src/kernels/speculative.msl`, `src/kernels/math.msl`
- `/D/rex/projects/grim/crates/grim-backend-rocm/Cargo.toml`, `src/lib.rs`, `src/device/roc_device.rs`, `src/kernels/mod.rs`, `src/kernels/*.rs` (53 modules)
- `/D/rex/projects/grim/crates/grim-tensor/src/backend.rs` — `BackendDevice` / `CoreTensorOps` trait definitions
- `/D/rex/projects/grim/crates/grim-tensor/src/dtype.rs` — `QuantFormat`, `FloatPackScheme`, `Storage`, `DType`
- `/D/rex/projects/grim/crates/grim-backend-tests/tests/parity_cpu_vulkan_metal.rs` — existing Metal parity test harness
- `/D/rex/projects/grim/crates/grim-backend-tests/tests/kernel_numerical_parity.rs` — kernel numeric parity tests
- `/D/rex/projects/grim/crates/grim-engine/src/model_loader.rs` — `SmolLm2` dispatch at 2561
- `/D/rex/projects/grim/crates/grim-models/transformer/src/smollm2.rs` — `SmolLm2::load_tp` (wraps `Llama::load_tp`)
- `/D/rex/projects/grim/crates/grim-nn/src/moe.rs` — `forward_metal` at 2133, `forward_vulkan` at 1909
