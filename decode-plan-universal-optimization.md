# UNIVERSAL DECODE OPTIMIZATION PLAN: Democratize LFM2 gains across all 151 models

**Goal:** Extend the decode optimizations proven on LFM2.5 (fused Q8_0 QKV GEMV, device-base
RoPE, decode graph) to every model that shares the same compute topology, fuse the MoE expert
pipeline (Charon), consolidate per-quant GEMV kernels, and remove superseded kernels.

**Method:** red-green-refactor per phase. Every phase lands with parity tests + performance
benchmarks + all existing suites green.

**Current state after decode-graph-plan-refined.md:**
- LFM2.5-350M-Q8_0 on gfx1201: 1.6 ms/tok = 625 tok/s (target was ≥250, stretch 400)
- Fused Q8_0 QKV GEMV: 3→1 GEMV launches per layer, byte-identical parity
- Device-base RoPE: per-token H2D Vec→1 device write, max_diff = 0.0
- Decode graph orchestration with CAPTURE_POISON eager fallback (kernels + device path verified)

---

## ARCHITECTURE MAP

### Model topology
- **151 model implementations** in `crates/grim-models/transformer/src/`
- **~20 models** call `shared_attention::fused_or_scalar_attention` directly (gemma, falcon_h1,
  chameleon, commandr, gpt2, gptj, exaone4_5, bloom, hy_v4, dots3_note, …)
- **~6 models** route through `block.rs::LlamaBlock` (qwen2, solar_open2, laguna, minicpm,
  muse_glimmer, …)
- **8 models have MoE** (deepseek2, deepseek32, deepseek4, bailingmoe2, bailingmoe3, kimi_k3,
  lfm2, mellum)
- Many models (gemma, falcon_h1, exaone4_5, …) have their OWN forward() with the same
  3-GEMV attention projection pattern

### Kernel landscape (`grim-backend-rocm/src/kernels/` — 59 files, 19,656 LOC)
- **Per-quant GEMV/GEMM**: q2k, q3k, q4k, q5k, q6k, q8_0 (GEMV + dequant) — overlapping logic
- **WMMA variants**: wmma_gemm, wmma_fp8_gemm, wmma_iq_gemm, wmma_quantized_gemm — fused
  dequant+GEMM (SUPERSEDE the per-quant GEMV kernels)
- **MoE**: charon.rs (2375 LOC, grouped dispatch), charon_wmma.rs, charon_backward.rs,
  moe_mega_kernel.rs — persistent-SM comm-compute
- **Attention**: qkv_attention (671 LOC kernel source), flash_decode, sage_attention,
  mla_decode, preshuffled_attention, cross_attention, extend_attention
- **Fused ops (already exist)**: rmsnorm_quant (norm→q8_1), silu_mul_quant (SwiGLU→q8_1),
  blend_kv_rope, fused_linear_ce, comm_fuse, fused_dequant_gemm

### What LFM2 has that others don't
| Optimization | LFM2 | block.rs models | individual models |
|---|---|---|---|
| Fused Q8_0 QKV GEMV | ✅ | ❌ (3 GEMVs) | ❌ (3 GEMVs) |
| Device-base RoPE | ✅ | ❌ (per-token Vec) | ❌ (per-token Vec) |
| Decode graph (past_dev counter) | ✅ (device path) | ❌ | ❌ |
| rmsnorm_quant (fused norm→q8_1) | ✅ (OPFUSE path) | ❌ (separate norm+quant) | ❌ |

---

## PHASE 1: Democratize to `block.rs::LlamaBlock` (highest leverage)

**Goal:** Wire fused Q8_0 QKV GEMV + device-base RoPE into the shared `LlamaBlock` forward
path. Benefits qwen2, solar_open2, laguna, minicpm, muse_glimmer immediately with zero
per-model changes.

**Why this is highest leverage:** `block.rs` is the shared block definition — one change
benefits all models that instantiate it. The 3-GEMV pattern and per-token positions Vec
are the same waste as LFM2's pre-optimization state.

### Sub-step 1a: Fused Q8_0 QKV projection in `LlamaBlock`
- At block construction: if weights are Q8_0 and device is ROCm, build
  `wqkv_q80_fused: Option<FusedQkvWeights>` from `wq`/`wk`/`wv` (reuse `build_fused_qkv_q80`)
- In `forward_with_kv_paged`: when fused blob exists, replace the 3 separate `wq/wk/wv.forward`
  calls with ONE `launch_fused_qkv_dot4` + `RocmStorageView` slicing
- Fall back to 3-GEMV when blob absent (non-Q8_0, non-ROCm)
- **Escape hatch:** `GRIM_FUSED_QKV=0`

### Sub-step 1b: Device-base RoPE in `apply_rope_multi_head`
- Replace the per-call `ext_positions` Vec build + `dev.rope()` with `rope_dev_base_into`
  writing into a per-layer cached stable output buffer
- Seed the `pos_base_dev` buffer once per generation (before capture bracket)
- **Escape hatch:** `GRIM_ROPE_DEV_BASE=0`

### Sub-step 1c: Attention + KV-append graph capture
- Wrap `grim_kv_append` + `grim_qkv_attention_dev` + `grim_bump_i32` in
  `begin_graph_capture("llama_decode_attn_{layer}")` / `end_graph_capture`
- Pre-allocate stable output buffers (q_rot_dev, k_rot_dev, attn_out_dev) in
  `LlamaLayerCache` before the first capture bracket
- On replay: device pointers stable → one hipGraphLaunch per layer attention
- **Escape hatch:** `GRIM_DECODE_GRAPH=0`

### Verification
- All existing `block.rs` tests pass
- qwen2 generation matches stock output byte-for-byte
- rocprofv3 --hip-trace: kernel launches per decode token drop measurably (3→1 GEMV × layers)

---

## PHASE 2: Democratize to `shared_attention` dispatch (benefits ~20 models)

**Goal:** Add a fused-Q8_0-QKV path to `shared_attention` so models that call
`fused_or_scalar_attention` can opt into the fused projection without changing their
forward() structure.

### Sub-step 2a: `shared_attention::fused_qkv_project`
- New function: takes `norm_x` (device f32), `wqkv_q80_fused` (fused blob), and
  model topology → returns `(q_rot, k_rot, v)` all device-resident
- Internally: quantize q8_1 → fused GEMV → zero-copy slicing → RoPE → returns
- Models call this instead of `wq/wk/wv.forward` + separate rope

### Sub-step 2b: Wire into high-traffic models
- **gemma** (6 wq/wk/wv calls), **falcon_h1** (9 calls), **exaone4_5** (3 calls):
  replace the 3-GEMV block with `fused_qkv_project` when fused blob is available
- Add `build_fused_qkv_q80` call in each model's weight loading (gated on ROCm + Q8_0)
- **Escape hatch:** `GRIM_FUSED_QKV=0` (same as LFM2)

### Verification
- Per-model parity: stock vs fused+rope produce identical tokens
- All existing model-specific tests pass

---

## PHASE 3: MoE kernel fusion (Charon integration)

**Goal:** Fuse the MoE expert pipeline — route + expert GEMM + SwiGLU + weighted sum —
into fewer launches, leveraging the existing Charon grouped-dispatch infrastructure.

**Current MoE pattern (all 8 MoE models):**
1. Host: compute gate logits, sort by expert, dispatch tokens
2. Per expert: Linear (expert_gate_w), Linear (expert_up_w), silu_mul_on_device, Linear (expert_down_w)
3. Host: weighted sum of expert outputs
This is ~4 launches × num_experts × num_layers per token. With 8 experts × 16 layers = 512
launches per token for attention + FFN.

### Sub-step 3a: Wire Charon grouped dispatch into MoE models that don't use it
- deepseek2/32/4, bailingmoe2/3, kimi_k3, mellum each implement `forward_moe_device`
  individually. Consolidate into a shared `shared_moe::fused_moe_dispatch` that calls
  Charon's `grouped_dispatch` internally
- The shared function takes: routing table, expert weights, activation → returns fused output
- Models call it instead of their per-expert loops

### Sub-step 3b: Fuse SwiGLU into Charon expert compute
- Charon already does grouped GEMM. Fuse `silu_mul_quant` into the epilogue:
  after gate×up projection, apply SwiGLU + quantize to q8_1 in the same kernel
  (matching the existing `rmsnorm_quant` pattern: compute → quantize → write q8_1)
- The down-projection then consumes pre-quantized q8_1 via `dot4_q80_q81_gemv`
- This eliminates the separate `silu_mul_on_device` launch per expert

### Sub-step 3c: Fuse rmsnorm_quant into the MoE gate path
- The gate Linear (routing) takes `x_norm` as input. Fuse the RMSNorm + gate GEMV into
  one kernel (rmsnorm_quant → dot4_q80_q81_gemv with gate weights)
- Reduces the gate path from 2 launches (norm + gate) to 1

### Verification
- Per-MoE-model parity: forward_moe_device vs forward_moe_host produce same output
- Charon grouped dispatch vs per-expert loop: same routing, same expert selection
- rocprofv3: MoE FFN launch count drops measurably (num_experts×4 → num_experts×2 + routing)

---

## PHASE 4: Fused ops kernel development

**Goal:** Create fused kernels that combine operations already computed separately, reducing
total launch count and improving data locality.

### Sub-step 4a: rmsnorm_rope — fused RMSNorm + RoPE
- Currently: RMSNorm (kernel) → RoPE (kernel) = 2 launches
- Fused: one kernel reads raw activation, normalizes, applies RoPE, writes output
- Applies to: Q and K paths in ALL models (before attention)
- Saves 2 launches/layer/token × num_layers

### Sub-step 4b: attention_rope_out — fused attention + output projection
- Currently: attention kernel → Linear (wo) = 2 launches
- Fused: attention kernel epilogue applies wo projection (same as existing
  `fuse_o` epilogue in `grim_qkv_attention` — just enable it by default for decode)
- Saves 1 launch/layer/token

### Sub-step 4c: FFN gate+up fused GEMV
- Currently: ffn_gate (GEMV) + ffn_up (GEMV) = 2 launches (same input!)
- Fused: one GEMV with concatenated weights `[2*inter, hidden]`, writes gate and up
  into separate output regions
- Same pattern as the Item 1 fused QKV GEMV — reuse `build_fused_qkv_q80` with
  ffn_gate+ffn_up weights
- Saves 1 launch/layer/token

### Sub-step 4d: FFN silu_mul + down fused
- Currently: silu_mul_on_device (kernel) + ffn_down (GEMV) = 2 launches
- Fused: silu_mul_quant (already exists!) writes q8_1, then ffn_down consumes q8_1
  via dot4_q80_q81_gemv — this is already 2 launches but with q8_1 intermediate
  (halves the intermediate data volume and enables the quantized down-proj)
- Alternatively: fuse SwiGLU epilogue directly into the down-projection GEMV

### Verification
- Each fused kernel: parity vs unfused (max_diff ≤ 1e-5)
- End-to-end: model forward with all fused ops vs stock — identical tokens
- rocprofv3: kernel launches per layer drop from ~12 to ~7

---

## PHASE 5: Kernel consolidation (remove superseded kernels)

**Goal:** Remove kernels that are superseded by fused-ops variants, reducing the kernel
JIT compilation footprint and maintenance burden.

### Sub-step 5a: Remove per-quant GEMV kernels superseded by WMMA fused dequant
- The `quantized_matmul` dispatch already prefers WMMA fused dequant for Q4K/Q5K/Q6K/
  Q2K/Q3K/IQ* — the standalone per-quant GEMV kernels are dead code:
  - `q4k_gemm.rs` (349 LOC) — superseded by `launch_wmma_fused_dequant_q4k`
  - `q5k_gemm.rs` (119 LOC) — superseded by WMMA
  - `q6k_gemm.rs` (114 LOC) — superseded by WMMA
  - `q2k_gemm.rs` (107 LOC) — superseded by WMMA
  - `q3k_gemm.rs` (142 LOC) — superseded by WMMA
- Only remove after verifying no model routes to these directly (check dispatch order)
- Keep `q8_0_dequant.rs` (41 LOC) — used by `dequantize_q8_0_host`

### Sub-step 5b: Remove redundant attention kernels
- `flash_decode.rs` vs `qkv_attention.rs` vs `sage_attention.rs` — audit which models
  route to each. If flash_decode is never dispatched (dispatcher prefers qkv_attention
  or sage_attention), remove it
- `extend_attention.rs` vs `cross_attention.rs` — audit model usage

### Sub-step 5c: Consolidate dequant kernels
- `q4k_dequant.rs` (199 LOC) and `q8_0_dequant.rs` (41 LOC) — both are small host-side
  dequant helpers. Consolidate into `iq_dequant.rs` or a shared dequant module

### Sub-step 5d: Remove old MoE host-only kernels
- `charon_backward.rs` (245 LOC) — only needed for training. If the inference binary
  doesn't reference it, gate behind a `training` feature flag

### Verification
- `cargo build` succeeds — no dangling references
- All test suites green
- Binary size decreases (fewer JIT kernels compiled)
- `rocprofv3` confirms no performance regression (same kernels launched for the same models)

---

## PHASE 6: Democratize decode graph to all models

**Goal:** Once Phases 1-4 are in place, the decode graph (past_dev counter + device-driven
attention) applies uniformly. The graph bracket wraps the layer attention section for all
models that route through the shared infrastructure.

### Sub-step 6a: Thread past_dev through the session
- Add `decode_graph_state: Option<DecodeGraphBuffers>` to the session model_state
- `DecodeGraphBuffers` holds: past_dev (device u32 counter), pos_base_dev (aliased),
  attention output buffers per layer
- Seeded once at prefill completion; bump kernel increments per decode step

### Sub-step 6b: Enable in block.rs + shared models
- `LlamaBlock::forward_with_kv_paged` reads past_dev from the session state
- Attention section wraps in graph capture (kv_append + attention_dev + bump)
- Same escape hatch: `GRIM_DECODE_GRAPH=0`

### Verification
- rocprofv3 --hip-trace: hipGraphLaunch count = num_layers per decode token
  (vs num_layers × num_kernels eager launches)
- hipModuleLaunchKernel count drops ≥80% vs stock for the decode loop

---

## SEQUENCE & DEPENDENCIES

```
Phase 1 (block.rs)          ← highest leverage, benefits 6+ models immediately
  ↓
Phase 2 (shared_attention)  ← benefits ~20 models
  ↓
Phase 3 (MoE/Charon)        ← independent, benefits 8 MoE models
  ↓
Phase 4 (fused ops)         ← benefits ALL models, builds on Phases 1-2
  ↓
Phase 5 (cleanup)           ← after Phase 4 confirms no regressions
  ↓
Phase 6 (decode graph)      ← after Phases 1-4 stabilize the compute topology
```

Phases 1+2 can run in parallel with Phase 3 (different model subsets).
Phase 4 requires Phases 1-2 landed (so the fused ops feed into the right dispatch paths).
Phase 5 is safe only after Phase 4 confirms all models work with the fused kernels.
Phase 6 is the final capstone — requires all prior phases.

---

## RISKS

| Risk | Mitigation |
|---|---|
| Per-quant GEMV removal breaks a model with a direct dependency | Audit dispatch order before removal; grep for direct kernel references |
| Fused QKV GEMV slower than WMMA for some quant formats | Only enable for Q8_0 weights (dot4 GEMV is proven for Q8_0); keep WMMA for other formats |
| MoE fusion changes expert routing behavior | Parity test: fused dispatch vs per-expert loop with identical routing tables |
| Graph capture hits host syncs in non-attention ops | Scope the graph bracket to ONLY the attention section (kv_append + attn + bump); the rest runs eagerly |
| Stable buffers increase VRAM usage | Buffers are fixed-size for decode (steps=1); total added VRAM is O(num_layers × hidden) — negligible vs KV cache |
| 151 models can't all be updated individually | Focus on shared infrastructure (block.rs, shared_attention, shared_moe); models that route through these get the optimization for free |
| rocBLAS handle not bound to capture stream during graph capture | `begin_graph_capture` already binds rocblas; verify for all kernel types |

---

## SUCCESS CRITERIA

| Metric | Baseline | Target |
|---|---|---|
| Models with fused QKV GEMV | 1 (LFM2) | ≥ 20 (block.rs + shared_attention models) |
| Models with device-base RoPE | 1 (LFM2) | ≥ 20 |
| MoE FFN launches per token | ~num_experts × 4 | ~num_experts × 2 |
| Kernel files | 59 | ≤ 50 (after cleanup) |
| Kernel LOC | 19,656 | ≤ 17,000 (after cleanup) |
| Decode tok/s (LFM2.5-350M-Q8_0) | 625 (already achieved) | ≥ 625 (no regression) |
| Decode tok/s (qwen2-7B-Q8_0, if testable) | TBD | measurable improvement |
| Parity | — | All models: identical tokens stock vs optimized |
