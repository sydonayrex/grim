# Grim Models Audit Report: Mamba, Vision, Audio

**Status**: Substantial structural skeletons with working CPU/F32 implementations — **NOT stubs**, but missing critical production pieces.

---

## Executive Summary

All three crates (`grim-models-mamba`, `grim-models-vision`, `grim-models-audio`) implement their respective `grim-core` capability traits (`StatefulSequence`/`CausalLm`, `Encoder`, `EncoderDecoderLm`) and compile. Tests pass. They are **structural v1 implementations** suitable for unit-testing the trait wiring and serving-layer integration, but **not shippable** for production inference without the items below.

---

## Per-Crate Assessment

### grim-models-mamba (370 lines lib.rs + 116 configs + 182 rwkv)

| Component | Status | Gaps |
|-----------|--------|------|
| `Mamba` (Mamba1-style) | **Working CPU impl** | Selective scan is a vanilla placeholder (§5.1); no chunked/parallel scan; no kernel; no weight loading for conv1d/A/dt_bias |
| `MambaState` + `SsmState` trait | Complete | Snapshot/restore works; pool integration is mocked (`request_id = 999`) |
| `Rwkv` (v6-style time/channel mix) | Loadable + step | No kernel; attention reduction simulated via `add_tensors`; `state_xy` unused |
| Config variants (Mamba2, Jamba, Nemotron-H, Granite-Hybrid) | **Stubs only** | Config structs exist; **no model impls** |
| Weight loading (`Mamba::load`, `MambaBlock::load`) | Partial | Missing conv1d, A_log, D, dt_bias paths in some variants; no GGUF/safetensors format handling |
| Quantization | **None** | F32 only; no INT8/FP8/NF4 paths |
| ROCm kernels | **None** | All CPU; selective scan needs custom HIP kernel (Rule 0: not GEMM) |

**Verdict**: Mamba1 + RWKV v6 are *runnable on CPU* for correctness tests. Mamba2/Jamba/hybrids are config-only. Production needs: selective scan kernel, weight format support, quantization, GPU backend.

---

### grim-models-vision (352 vit.rs + 219 bert.rs + 59 configs)

| Component | Status | Gaps |
|-----------|--------|------|
| `Vit` (ViT/CLIP encoder) | **Working CPU impl** | `encode_image` does manual patch extract + matmul loops (no im2col/conv); no positional-interpolation for variable resolution; no kernel |
| `Bert` (bidirectional encoder) | Loadable + forward | Uses `add_tensors` via backend (odd indirection); LayerNorm re-implemented locally instead of `grim_nn::RmsNorm`; no `load` for full model (only blocks) |
| Config variants (ModernBERT, NomicBERT, T5Encoder) | **Stubs only** | Config structs only; no model impls |
| Weight loading | Partial | `BertBlock::load` works; `Vit` has no `load` at all |
| Quantization | **None** | F32 only |
| ROCm kernels | **None** | Attention, patch projection, LayerNorm all CPU |

**Verdict**: ViT and BERT *encode correctly on CPU* for tests. Missing: `Vit::load`, variable-resolution pos-embed, kernel fusion, quantization, config-variant implementations.

---

### grim-models-audio (375 whisper.rs)

| Component | Status | Gaps |
|-----------|--------|------|
| `Whisper` encoder-decoder | **Working CPU impl** | Encoder blocks skip self-attention (only MLP residual); decoder blocks skip self-attn + cross-attn (only MLP residual); `_self_o`, `_cross_q`, `_cross_v`, `_cross_o`, `_ffn_norm` fields stored but **unused** |
| Weight loading | **Missing** | No `Whisper::load` or block `load` methods |
| Config variants | N/A | Single `WhisperConfig` only |
| Quantization | **None** | F32 only |
| ROCm kernels | **None** | Cross-attention, mel-projection, encoder/decoder blocks all CPU |
| Audio frontend | **Missing** | No mel-spectrogram / log-mel / VAD / chunking — `encode` expects pre-computed mel `(n_mels, T)` |

**Verdict**: Structural skeleton only. The attention paths are **stubbed out** (fields exist but forward passes skip them). Not usable for real ASR until attention is wired and weights load.

---

## Cross-Cutting Gaps (All Three Crates)

### 1. Weight Loading & Format Support
- **No GGUF loader** — grim uses `WeightSource` trait but no GGUF/safetensors/HF `model.safetensors` implementation exists in these crates
- **No tied-weight handling** — output head vs token embedding sharing (common in Llama/Mamba/Whisper)
- **No sharded/checkpoint loading** — large models need `safetensors` multi-file or GGUF single-file streaming
- **Mention**: The `grim-nn::Linear::load` and `Embedding::load` handle layout normalization, but model-level `load` ctors are incomplete (especially audio/vision).

### 2. Quantization Pipeline
| Need | Status |
|------|--------|
| INT8 symmetric/asymmetric (weight-only) | ❌ |
| FP8 (E4M3/E5M2) + block scaling | ❌ |
| NF4/QLoRA 4-bit | ❌ |
| Activation quantization (KV/SSM state) | ❌ |
| `.grim` model export with quant metadata | ❌ |

**Reference**: `grim-quant` crate exists but unused; `rocm-quantization-inference` skill has FP8 MFMA patterns for gfx1200.

### 3. GPU / ROCm Backend Integration
- All three crates hardcode `Device::Cpu` and `grim_backend_cpu::cpu_tensor`
- No device-agnostic construction: `Model::load` should accept `Device` and dispatch to `grim-backend-rocm`/`cuda`/`metal`
- **Selective scan (Mamba)** and **cross-attention (Whisper)** are the two non-GEMM kernels requiring custom HIP (Rule 0: don't rewrite GEMM)
- FlashAttention / paged-attention kernels needed for vision/audio transformers
- Wave64 block sizing, MFMA tile selection, LDS double-buffering per `rocm-hip-kernels` skill

### 4. Speculative Decoding / MTP Integration
- `MambaState::pos` cursor exists for rollback (§5.3) but no `MtpDepthProvider` impl
- No draft-model wiring for Mamba/RWKV/Whisper
- `grim-speculative` crate exists — models need to implement its provider traits

### 5. Diffusion Trait (`DiffusionModel`) — **Missing Entirely**
- `grim-models/diffusion` crate exists (Cargo.toml only)
- No UNet, DiT, or scheduler implementations
- `NoiseScheduler` trait in `grim-core` has zero impls

### 6. Testing & Correctness
| Gap | Detail |
|-----|--------|
| Numerical parity tests | No reference vs. CPU/GPU comparison |
| Mutation testing | No `cargo-mutants` or similar |
| Property-based tests | Only shape tests; no `proptest` for numerical invariants |
| Benchmark harness | No `criterion` benches for throughput/latency |
| Integration tests | No end-to-end generate/encode/decode with real weights |

### 7. Architecture / Code Quality
- **Duplicate RNG**: Each crate has identical `SimpleRng` — centralize to `grim-core` or `grim-nn`
- **BERT LayerNorm** re-implements RMSNorm logic instead of using `grim_nn::RmsNorm`
- **Whisper** stores unused attention weight fields (`_self_o`, `_cross_q`, etc.) — dead code
- **Mamba** `step_block` has nested loops over `d_inner × d_state` — O(N²) CPU fallback; needs kernel
- **ViT** manual patch extraction loops — should be `im2col` + GEMM or conv2d
- No `#[cfg(feature = "rocm")]` gating — CPU-only builds forced

---

## Required Work to Ship (Prioritized)

### Phase 0 — Foundation (1–2 weeks)
1. **Centralize RNG** → `grim-core` or `grim-nn`
2. **Add `Device` parameter** to all `::random` and `::load` ctors; remove hardcoded `Device::Cpu`
3. **Implement `Vit::load`** and `Whisper::load` + block loaders
4. **Wire Whisper attention** — replace MLP-only stubs with real self-attn + cross-attn (CPU first, kernel later)
5. **Delete dead fields** in Whisper decoder block
6. **Unify LayerNorm** — BERT should use `grim_nn::RmsNorm` or a shared `LayerNorm` module

### Phase 1 — Weight Formats & Quantization (2–3 weeks)
7. **GGUF loader** in `grim-nn` (extends `WeightSource`) — supports all three model families
8. **Safetensors loader** (multi-file, memory-mapped)
9. **Quantization pipeline**: INT8 weight-only → FP8 (gfx1200) → NF4
10. **`.grim` export** with quant metadata per `rocm-quantization-inference`

### Phase 2 — Kernels & GPU (3–4 weeks)
11. **Mamba selective scan HIP kernel** (Wave64, LDS tiling, persistent for decode-step)
12. **FlashAttention / paged-attention** for ViT/BERT/Whisper (reuse `grim-backend-rocm` infra)
13. **Cross-attention kernel** for Whisper decoder (encoder-out projected once, reused)
14. **RWKV time-mix kernel** (recurrent, not parallel — different pattern)
15. **FP8 MFMA gates** on `gcnArchName >= gfx1200` per `rocm-hip-kernels` Rule 0

### Phase 3 — Config Variants & Hybrids (2 weeks)
16. **Implement Mamba2** (chunked SSM, multi-head)
17. **Implement Jamba** (SSM + attention + MoE layers)
18. **Implement Nemotron-H / Granite-Hybrid** (attention/SSM interleaving)
19. **ModernBERT / NomicBERT / T5Encoder** in vision crate
20. **Diffusion crate**: UNet + DiT + DDPM/DDIM/Euler schedulers

### Phase 4 — Serving Integration (1–2 weeks)
21. **Speculative decoding hooks** (`MtpDepthProvider`, draft model wiring)
22. **LoRA/adapter fusion** via `AdapterHandle` (already in trait signatures)
23. **Batch/continuous batching** support in `StatefulSequence::step`
24. **Metrics / tracing** integration (`grim-observability`)

### Phase 5 — Hardening (ongoing)
25. **Numerical parity tests** vs. reference (HF transformers, llama.cpp, whisper.cpp)
26. **Property-based tests** for scan/attention invariants
27. **Criterion benches** with ROCm profiling (`rocprof-compute` occupancy + stalls)
28. **Mutation testing** on critical paths
29. **Fuzz input shapes** (variable batch/seq/resolution)

---

## Skill-Mapped Action Items

| Skill | Applied To |
|-------|------------|
| `ponytail` | Delete dead Whisper fields; unify RNG; don't build config variants until base works |
| `rocm-hip-kernels` | Selective scan, FlashAttn, cross-attn kernel specs (Wave64, MFMA, LDS) |
| `rocm-kernel-design` | Evidence-driven kernel spec → plan → validate loop |
| `rust-gpu-discipline` | No silent CPU fallback; `Device` threading; hipRTC JIT caching |
| `rust-ml-llm-architecture` | Model trait wiring, quantization metadata, speculative decode hooks |
| `ai-plan` | SEA learn loop for kernel autotuning; model routing |
| `architecture-blueprint` | Vertical slices per modality; domain-first modules |
| `requirements-clarity` | Each phase has YAGNI gate — "does this need to exist?" before building |
| `humanizer` | This report — plain, direct, no AI fluff |

---

## File Tree After Implementation (Target)

```
crates/grim-models/
├── mamba/
│   ├── src/
│   │   ├── lib.rs           # Mamba, RWKV, MambaState ✓
│   │   ├── mamba2.rs        # NEW: chunked SSM
│   │   ├── jamba.rs         # NEW: hybrid blocks
│   │   ├── nemotron_h.rs    # NEW
│   │   ├── granite_hybrid.rs# NEW
│   │   ├── configs.rs       # ✓ (all configs)
│   │   ├── rwkv.rs          # ✓
│   │   ├── selective_scan.rs# NEW: HIP kernel + CPU fallback
│   │   ├── load.rs          # NEW: GGUF/safetensors load
│   │   └── quant.rs         # NEW: INT8/FP8/NF4
├── vision/
│   ├── src/
│   │   ├── lib.rs           # ✓
│   │   ├── vit.rs           # ✓ + load + variable pos-embed
│   │   ├── bert.rs          # ✓ + unified LayerNorm
│   │   ├── modern_bert.rs   # NEW
│   │   ├── nomic_bert.rs    # NEW
│   │   ├── t5_encoder.rs    # NEW
│   │   ├── flash_attn.rs    # NEW: HIP kernel
│   │   ├── load.rs          # NEW
│   │   └── quant.rs         # NEW
├── audio/
│   ├── src/
│   │   ├── lib.rs           # ✓
│   │   ├── whisper.rs       # ✓ + real attention + load
│   │   ├── cross_attn.rs    # NEW: HIP kernel
│   │   ├── mel.rs           # NEW: log-mel frontend
│   │   ├── load.rs          # NEW
│   │   └── quant.rs         # NEW
├── diffusion/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── unet.rs
│   │   ├── dit.rs
│   │   ├── schedulers.rs
│   │   └── load.rs
└── transformer/             # (existing, dense LM)
```

---

## Sign-Off Criteria for "Shippable"

- [ ] All three crates load real GGUF/safetensors checkpoints (HF hub IDs documented)
- [ ] `cargo test --all-features` passes numerical parity vs. reference (±1e-3)
- [ ] ROCm kernels run on MI300X (gfx942) and Navi 31 (gfx1103) with ≥50% occupancy
- [ ] INT8 weight-only quantization < 1% perplexity degradation
- [ ] FP8 (gfx1200) kernel path compiles and runs
- [ ] Speculative decode works with `grim-speculative` draft model
- [ ] End-to-end generate (text), encode (image), transcribe (audio) demos in `grim-cli`
- [ ] Benchmarks published: tok/s, TTFT, memory, power

---

*Report generated from source audit — no stubs left unexamined.*

---

## TDD Test Specifications (Golden Standard)

Each Phase 0 item below specifies mutation-resistant tests following `grim-quant/tests/golden_*.rs` pattern:

1. **Hand-constructed inputs** — raw bytes, deterministic values, never reuse library code to build expected
2. **Independent expected values** — derived from format spec, documented per-bit in comments
3. **Tight tolerances** — `close()` helper with `abs == 0.0 || (abs/denom) < 1e-5` for numerics
4. **Silent-corruption gate** — truncated / zeroed inputs must error
5. **`golden_*` prefix** in each crate's `tests/` dir

### Phase 0.1 — Centralize RNG → `golden_rng_consistency.rs`

**File**: `crates/grim-models/mamba/tests/golden_rng_consistency.rs`
**Skills**: `ponytail` (delete duplicated code), `refactoring-patterns` (extract to shared crate), `clean-code` (DRY)

**Test**: `fn golden_rng_deterministic_seed_identity()`

| Element | Spec |
|---------|------|
| Input | Fixed seed `0xDEAD_BEEF`, request 8 f32 values |
| Procedure | Call centralized `grim_core::rng::SimpleRng::new(seed)` in mamba/vision/audio each produce same 8 floats |
| Expected | Pre-computed reference: `[0.1234, 0.5678, …]` — hardcoded literal array derived by running the reference impl once and pinning |
| Tolerance | `assert_eq!` exact bit-identical (deterministic PRNG) |
| Mutation target | Mutant that seeds differently per crate or replaces algorithm |

**Test**: `fn golden_rng_remove_duplicate_impls()`

| Element | Spec |
|---------|------|
| Input | Compile-time assertion |
| Procedure | Run `grep -r "struct SimpleRng" crates/grim-models/` in a build-script test; fail if > 1 copy exists |
| Expected | Exactly 1 `SimpleRng` definition |
| Mutation target | Mutant that forgets to delete a copy |

---

### Phase 0.2 — `Device` Parameter → `golden_device_param.rs`

**File**: `crates/grim-models/mamba/tests/golden_device_param.rs` (and parallel files in vision/, audio/)
**Skills**: `rust-gpu-discipline` (no CPU hardcoding, device threading), `architecture-blueprint` (clean ctor abstraction, device-agnostic)

**Test**: `fn golden_device_random_ctor_takes_device()`

| Element | Spec |
|---------|------|
| Input | `Device::Cpu` literal |
| Procedure | Call `Mamba::random(Device::Cpu, &config)` on small config; assert `Ok` |
| Expected | Returns `Ok(model)` where `model.device() == Device::Cpu` |
| Tolerance | Compile-time + runtime assertion |
| Mutation target | Mutant that hardcodes `Device::Cpu` internally and ignores parameter |

**Test**: `fn golden_device_load_ctor_takes_device()`

| Element | Spec |
|---------|------|
| Input | `Device::Cpu`, path to non-existent file |
| Procedure | Call `Vit::load(Device::Cpu, "/nonexistent")` |
| Expected | Returns `Err` (load error, not panic) |
| Tolerance | N/A — error-path structural |
| Mutation target | Mutant that panics instead of returning `Err` |

**Test**: `fn golden_device_compile_time_feature_gate()`

| Element | Spec |
|---------|------|
| Input | `#[cfg(feature = "rocm")]` block |
| Procedure | Build with `--features rocm`; call `::random(Device::Rocm(0), …)` |
| Expected | Returns `Err(UnsupportedBackend)` gracefully (not compile error) |
| Mutation target | Mutant that removes cfg gate and forces ROCm import unconditionally |

---

### Phase 0.3 — `Vit::load` + `Whisper::load` → `golden_load_shapes.rs`

**File**: `crates/grim-models/vision/tests/golden_load_shapes.rs`
**Skills**: `rust-ml-llm-architecture` (model loading traits, weight format patterns), `rust-ffi-grim` (binary weight layout, GGUF/safetensors FFI boundaries)

**Test**: `fn golden_vit_load_hand_constructed_weight_buffer()`

| Element | Spec |
|---------|------|
| Input | Hand-constructed in-memory weight buffer (simulating GGUF header + tensors) containing exactly 1 `patch_embed.proj.weight` of shape `[768, 3, 16, 16]` with all bytes = `0x3F800000` (f32 1.0) |
| Procedure | `Vit::load(Device::Cpu, &mock_reader)` |
| Expected | Returns `Ok(model)`; `model.patch_embed.proj.weight` has shape `[768, 3, 16, 16]`, dtype F32, all values == 1.0 |
| Tolerance | `assert_eq!` exact |
| Mutation target | Mutant that misreads dims, transposes axes, or drops a weight |

**Test**: `fn golden_vit_load_rejects_truncated_buffer()`

| Element | Spec |
|---------|------|
| Input | Buffer truncated mid-tensor (header says 768×3×16×16 but only 1000 bytes provided) |
| Procedure | `Vit::load(Device::Cpu, &truncated_reader)` |
| Expected | Returns `Err` |
| Mutation target | Mutant that silently zero-fills missing bytes |

**File**: `crates/grim-models/audio/tests/golden_load_shapes.rs`

**Test**: `fn golden_whisper_load_hand_constructed_weight_buffer()`

| Element | Spec |
|---------|------|
| Input | Buffer with 1 `encoder.conv1.weight` of shape `[1280, 80, 3]` filled with pattern: `row_i = (i % 256) as f32 / 256.0` |
| Procedure | `Whisper::load(Device::Cpu, &mock_reader)` |
| Expected | `Ok(model)`; conv1 weight dtype F32, values match pattern exactly |
| Mutation target | Mutant that reverses dim order or applies wrong stride |

---

### Phase 0.4 — Wire Whisper Attention → `golden_whisper_attention.rs`

**File**: `crates/grim-models/audio/tests/golden_whisper_attention.rs`
**Skills**: `rust-ml-llm-architecture` (attention layer wiring), `rocm-hip-kernels` (design CPU path for future GPU dispatch), `ponytail` (don't overbuild unused attention variants)

**Test**: `fn golden_whisper_self_attn_hand_constructed_weights()`

| Element | Spec |
|---------|------|
| Input | Single encoder block with hand-constructed Q/K/V/O weight buffers: set `W_q = I` (identity), `W_k = I`, `W_v = I`, `W_o = I`, biases zero. Input hidden `h = [1.0, 2.0, …, d_model]`. This forces attention to attend equally (softmax over identical cosine-sim keys). |
| Procedure | `block.forward(h)` (CPU) |
| Expected | Output = `h + attention(h)` where attention with identity weights = `softmax(QQ^T/√d) V`. For d_model=4 with h=[1,2,-1,-2], compute expected step-by-step: Q=K=V=h; scores = h·h/√4 = (1+4+1+4)/2 = 5.0; softmax(5.0) = 1.0; V=→ output = 1.0·h. So output = h + h = 2h = [2,4,-2,-4]. |
| Tolerance | `close()` with rel < 1e-5 |
| Mutation target | Mutant that skips residual add, uses wrong scale, or computes attn as identity |

**Test**: `fn golden_whisper_cross_attn_encoder_decoder_interaction()`

| Element | Spec |
|---------|------|
| Input | Decoder block with identity Q weight, cross-attn K/V identity. Encoder output = all-ones `[1,1,…]` (n_frames × d_model). Decoder hidden = `[1,2,3,4]` (single token). Forces cross-attn to attend uniformly over encoder frames. |
| Procedure | `decoder_block.forward(h, encoder_out)` |
| Expected | Compute by hand: Q=h; K=encoder_avg; V=encoder_avg; attention = softmax(h·encoder_avg/√d). For uniform K, attention distributes equally → output = encoder_avg = 1.0 per dim. |
| Tolerance | `close()` with rel < 1e-5 |
| Mutation target | Mutant that ignores encoder state (current stub behavior) |

**Test**: `fn golden_whisper_ffn_still_works_after_attn_wiring()`

| Element | Spec |
|---------|------|
| Input | After wiring attention, FFN path must not regress. Feed known input through full encoder block, assert FFN gate computes correctly. Hand-construct FFN weights: W_fc0 = 2×I, W_fc1 = 0.5×I. Input = [1.0, -1.0, 2.0, -2.0]. |
| Procedure | FFN = gelu(h·W_fc0)·W_fc1 |
| Expected | Pre-gelu = 2×h = [2,-2,4,-4]; gelu([2,-2,4,-4]) = [1.9545, -0.0455, 3.9999, -0.0001]; output = ×0.5 = [0.97725, -0.02275, 1.99995, -0.00005] |
| Tolerance | `close()` with rel < 1e-4 (GELU approximation variance) |
| Mutation target | Mutant that drops gelu or applies wrong activation |

---

### Phase 0.5 — Delete Dead Fields → `golden_whisper_layout.rs`

**File**: `crates/grim-models/audio/tests/golden_whisper_layout.rs`
**Skills**: `ponytail` (delete dead code, YAGNI), `clean-code` (no dead fields in prod structs)

**Test**: `fn golden_whisper_decoder_block_no_dead_fields()`

| Element | Spec |
|---------|------|
| Input | Compile-time `assert_eq!` on struct size via `mem::offset_of!` and `mem::size_of!` |
| Procedure | Assert `size_of::<DecoderBlock>()` equals expected size after dead field removal (compute from active fields: self_attn_q, self_attn_k, self_attn_v, self_attn_o, cross_attn_q, cross_attn_k, cross_attn_v, cross_attn_o, ffn_norm, self_attn_norm, cross_attn_norm, mlp). Reference size is `<pre-computed bytes>`. |
| Expected | Size matches reference (computed offline by compiling clean version) |
| Mutation target | Mutant that leaves a dead field (`_self_o`, `_cross_q`, `_cross_v`, `_cross_o`, `_ffn_norm`) in the struct — size would differ |

**Test**: `fn golden_whisper_no_unused_field_warnings()`

| Element | Spec |
|---------|------|
| Input | Run `cargo build 2>&1` in audio crate, filter for `warning: field `_self_o` is never read` |
| Procedure | Assert grep returns no matches |
| Expected | Zero warnings for unused fields prefixed with `_` |
| Mutation target | Mutant that re-introduces an unused `_`-prefixed field |

---

### Phase 0.6 — Unify BERT LayerNorm → `golden_layernorm_equivalence.rs`

**File**: `crates/grim-models/vision/tests/golden_layernorm_equivalence.rs`
**Skills**: `clean-code` (DRY, remove duplicate impl), `refactoring-patterns` (extract without changing behavior), `ponytail` (don't build new LayerNorm when grim_nn::RmsNorm exists)

**Test**: `fn golden_bert_layernorm_vs_grim_nn_rmsnorm_exact_values()`

| Element | Spec |
|---------|------|
| Input | Hand-construct input tensor `x = [0.5, -1.0, 2.0, -0.5]` (N=4). Hand-construct weight `w = [1.0, 2.0, 0.5, 1.5]` |
| Procedure | Run both `bert_layer_norm(x, w)` (the redundant impl) and `grim_nn::RmsNorm::forward(x, w)` |
| Expected | Both compute: `rms = sqrt(mean(x_i²) + 1e-5) = sqrt((0.25+1+4+0.25)/4 + 1e-5) = sqrt(5.5/4) = sqrt(1.375) = 1.172603…`; output_i = x_i / rms * w_i = [0.5/1.1726*1.0, -1.0/1.1726*2.0, 2.0/1.1726*0.5, -0.5/1.1726*1.5] = [0.4264, -1.7056, 0.8528, -0.6396] |
| Tolerance | `close()` with rel < 1e-5 (both are same formula; should match to f32 precision) |
| Mutation target | Mutant that uses a different epsilon, wrong normalization axis, or forgets weight scaling |

**Test**: `fn golden_bert_layernorm_rejects_wrong_shape()`

| Element | Spec |
|---------|------|
| Input | Input shape `[4]`, weight shape `[8]` (mismatch) |
| Procedure | Call the unified `LayerNorm::new(…).forward(x)` |
| Expected | Returns `Err` |
| Mutation target | Mutant that silently broadcasts mismatched weight |

---

### Implementation Order (TDD Red-Green)

For each item, write the golden test first (RED), then implement the fix (GREEN), then verify the test passes:

1. `golden_whisper_layout.rs` — dead fields (structural, quickest to verify)
2. `golden_rng_consistency.rs` — RNG centralization (structural + deterministic)
3. `golden_device_param.rs` — Device param (structural)
4. `golden_layernorm_equivalence.rs` — LayerNorm unify (numerical tracing path)
5. `golden_load_shapes.rs` — Vit/Whisper load (semi-numerical)
6. `golden_whisper_attention.rs` — attention wiring (full numerical tracing path)

Each test file follows this skeleton:

```rust
//! Mutation-resistant golden test for …
//!
//! Hand-constructs inputs so expected values are derived independently
//! from the format spec, not by calling the code under test.

use grim_core::device::Device;
// … per-item imports …

/// Relative + absolute tolerance helper (mirrored from grim-quant golden idiom).
fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

// … test functions …
```

---

## TDD Test Specs for Phase 1+ (Sketch)

These follow the same golden-standard pattern; full details deferred until Phase 0 lands.

### Quantization Pipeline (Phase 1)

| Item | Test Pattern |
|------|-------------|
| GGUF loader | `golden_gguf_header.rs` — hand-construct GGUF v3 binary with 1 tensor of known shape/dtype; assert `WeightSource::open` reads correct metadata |
| INT8 quant | `golden_int8_dequant.rs` — hand-construct block-scaled INT8 buffer per format spec; assert dequant yields exact expected f32 values |
| FP8 (E4M3) | `golden_fp8_e4m3_dequant.rs` — hand-construct 1 super-block of packed E4M3 bytes; assert every weight matches `fp8_e4m3_to_f32` independently |
| NF4 4-bit | `golden_nf4_dequant.rs` — hand-construct NF4 quantized super-block (2-bit scale + 4-bit codebook indices); assert dequant matches spec formula |
| `.grim` export | `golden_grim_export_roundtrip.rs` — export known tensor to `.grim` format, reload, assert bit-identical |

### Selective Scan Kernel (Phase 2)

| Item | Test Pattern |
|------|-------------|
| CPU fallback | `golden_selective_scan_identity.rs` — hand-construct A=1, B=1, C=1, dt=1; assert scan output = cumulative sum of input (closed-form) |
| HIP kernel | `golden_selective_scan_hip_parity.rs` — same input as CPU fallback; assert ROCm kernel output matches CPU fallback with rel < 1e-5 |

### Attention Wiring (Phase 2)

| Item | Test Pattern |
|------|-------------|
| FlashAttn stub | `golden_flash_attn_vs_naive.rs` — hand-construct Q/K/V with identity; assert flash attention matches naive O(N²) attention with `close()` |

### Truncated-Buffer / Error-Path Gate (All Phases)

Every `load`, `quant`, and `dequant` function must have:

```rust
#[test]
fn golden_rejects_truncated_buffer_<feature>() {
    let buf = vec![0u8; <minimal_valid_size - 1>];
    assert!(dequant_xxx(&buf, n).is_err());
}
```