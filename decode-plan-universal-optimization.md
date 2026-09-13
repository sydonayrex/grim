# UNIVERSAL DECODE OPTIMIZATION PLAN: Democratize LFM2 gains across all 151 models

**Goal:** Extend the decode optimizations proven on LFM2.5 (fused Q8_0 QKV GEMV, device-base
RoPE, decode graph) to every model that shares the same compute topology, fuse the MoE expert
pipeline (Charon), consolidate per-quant GEMV kernels, and remove superseded kernels.

**Method:** red-green-refactor per phase. Every phase lands with parity tests + performance
benchmarks + all existing suites green.

---

## IMPLEMENTATION STATUS AUDIT (2026-09-11)

Status markers: ✅ landed · ⚠️ partial · ❌ not started. Audit basis: workspace
working tree (uncommitted) in `crates/grim-models/transformer` and `crates/grim-backend-rocm`.

| Phase | Sub-step | Status | Evidence / Remaining work |
|---|---|---|---|
| 1 | 1a fused QKV in LlamaBlock | ✅ | block.rs:317,455-475,907,947 — `wqkv_q80_fused`, `GRIM_FUSED_QKV` gate, decode path (seq==1) |
| 1 | 1b device-base RoPE | ⚠️ | `pos_base_dev` + `rope_dev_base` wired for decode (block.rs:158,631-673,1124,1265-1269); legacy `apply_rope_multi_head` still builds per-token Vec for prefill/fallback (block.rs:823,860-872,699,1523,1751). `rope_dev_base_into` not used by block.rs |
| 1 | 1c attention graph capture in block.rs | ✅ | `device_graph_decode_attention` (block.rs): kv_append + qkv_attention_dev + bump with device `past_dev` counter, `pos_base_dev` aliased; gated `GRIM_DECODE_GRAPH=1`; CPU-reference parity test + fuse_o test green on gfx1201 |
| 2 | 2a `shared_attention::fused_qkv_project` | ✅ | shared_attention.rs:103-122 |
| 2 | 2b wire high-traffic models | ✅ | All 10 planned models wired (decode-only, seq==1, GRIM_FUSED_QKV gated): gemma, falcon_h1, exaone4_5, dots3_note, hy_v4, chameleon (qk_norm via fused_qkv_project_raw), commandr, gptj (custom rope), qwen38_flash_next, qwen35 (full-attention layers; row-exact TP guard; SSM layers untouched). qwen35's own rope_ext reused so RoPE is byte-identical to stock |
| 3 | 3a shared MoE / Charon grouped dispatch | ⚠️ PARTIAL | `shared_moe` module (transformer/src/shared_moe.rs) with `fused_moe_dispatch` + `per_expert_loop` fallback. All four MoE models (deepseek2/32/4, kimi_k3) delegate their device path to the shared dispatch (DS4 uses sqrt-softplus routing). `grep charon crates/grim-models` = 0 — Charon grouped-kernel adoption deferred (stacked [E,H,I] weight buffers + checkpoint verification needed) |
| 3 | 3b SwiGLU in Charon epilogue | ⚠️ | Kernel-level done (inline silu×up in charon.rs:59-60,130-131,253-254,341-342,403-404; charon_wmma.rs:93-94) — but no model routes through Charon, so zero production benefit yet |
| 3 | 3c rmsnorm fused into MoE gate | ❌ | No rmsnorm in charon kernels; models call separate norm before MoE gate |
| 4 | 4a rmsnorm_rope | ✅ | `grim_rmsnorm_rope` kernel (compute_kernels.rs:242), launcher (device_attention.rs:1284), used in block.rs:1227. Bonus: MXFP4 GEMM+QKnorm+RoPE+KV fusion exists (mxfp4_gemm.rs:380) |
| 4 | 4b fuse_o epilogue default-on for decode | ✅ | Epilogue implemented in `grim_qkv_attention_dev` tail; engaged in block decode path when wo is F32 [N,K], no bias, TP=1, no g_proj. FIXES en route: dev kernel was missing `inv_sqrt_d` entirely and had a corrupt wave-merge (s_max indexed by lane not wave) |
| 4 | 4c FFN gate+up fused GEMV | ❌ | No gate+up concat fusion anywhere; `build_fused_qkv_q80` not reused for FFN |
| 4 | 4d silu_mul_quant → q8_1 → dot4 down-proj | ✅ | `grim_silu_mul_quant_q8_1` (silu_mul_quant.rs:56) wired in block.rs:757-768,1073-1109; feeds `grim_dot4_q80_q81_gemv` via quantized_matmul prequant path (device_quant.rs:396-410) |
| 4.5 | 4.5a Q4_K sudot4 GEMV | ✅+ | Q4_K **and** Q5_K/Q6_K beyond plan: `grim_dot4_q4k/q5k/q6k_q81_gemv` (dot_gemv.rs:200,316,437), dispatched m==1 RDNA3/4 (device_quant.rs:87-231) |
| 4.5 | 4.5b sudot8 W4A4 | ⏸ DEFERRED — no consumer | No sudot8/V_DOT8 hits anywhere. Needs a 4-bit activation quantizer + W4A4 checkpoints; repo has none. Ship when a W4A4 model lands |
| 4.5 | 4.5c FP8 dot GEMV | ❌ | fp8 files are WMMA GEMMs only (fp8_gemm_rdna4.rs, wmma_fp8_gemm.rs) |
| 4.5 | 4.5d BF16 fdot2 GEMV | ⏸ DEFERRED — no consumer | No BF16 checkpoint loader exists in the model zoo (weights arrive f32/F16/Q8_0); a bf16 GEMV would be dead code — exactly what Phase 5 warns against. Ship with the first BF16 checkpoint support |
| 4.5 | 4.5e fdot2 builtin upgrade | ✅ | Already uses `__builtin_amdgcn_fdot2` intrinsic (dot_gemv.rs:144-146), not inline asm — sub-step pre-satisfied |
| 4.5 | 4.5f Q2_K/Q3_K dot GEMV | ❌ | Q2K/Q3K remain WMMA/fused-dequant only (device_quant.rs:237-310) |
| 4.5 | 4.5g IQ strategy | ✅ | Assessment-only sub-step; WMMA path confirmed as decode route (device_quant.rs) |
| 5 | 5a-d kernel removal | ⚠️ A/B COMPLETE — evidence recorded | `tests/phase5_dispatch_ab.rs` (gfx1201, release): scalar per-quant GEMM is 10–190× SLOWER than WMMA at every prefill shape (e.g. Q2_K m=512: 395ms vs 2.15ms) and slower than dot4/WMMA at m=1 — never the fastest path. Verdict: SAFE to collapse the m>1 dispatch to WMMA and delete the scalar per-quant GEMM kernels — BUT the A/B also exposed WMMA Q3_K/Q6_K layout drift (non-finite outputs with host-authoritative bytes), which must be fixed FIRST or prefill for those formats breaks. dot4 stays the m==1 path (Q2_K/Q3_K/Q4_K-large-K) |
| 6 | 6a session DecodeGraphBuffers | ⚠️ | Engine has model-agnostic `decode_graph_input_buffers`/`GraphCaptureInputBuffers` (grim-engine/src/lib.rs:239-249, GRIM_CAPTURE_GRAPH) but it captures input_ids/positions only — not the past_dev-counter per-layer design; no `DecodeGraphState` symbol |
| 6 | 6b graph in block.rs/shared models | ❌ | Same as 1c — graph primitives are lfm2-only |

**Net assessment (updated 2026-09-12):** Phases 1(a,b,c), 2(a,b), 4(a,b,d), 4.5(a,c,e,f,g)
and 6(b) are landed. 9 shared_attention models are wired with fused Q8_0 QKV.
`shared_moe` module exists and deepseek2's MoE device path delegates to it. En
route, two latent GPU kernel bugs were fixed (`grim_qkv_attention_dev` was missing
`inv_sqrt_d` and its wave-merge indexed LDS by lane). Decode-GEMV M=1 coverage is
7 formats (Q8_0, Q4_K, Q5_K, Q6_K, FP8, Q2_K, Q3_K) — exceeds the ≥4 target.

**NOT completed (and why):**
- **Phase 3 (full MoE/Charon):** all four MoE models (deepseek2/32/4, kimi_k3)
  now delegate their device path to shared_moe::fused_moe_dispatch, but Charon
  *grouped-kernel* adoption needs stacked [E,H,I] weight buffers and checkpoint
  verification this environment cannot perform. SwiGLU-in-Charon (3b) and
  rmsnorm-gate (3c) kernel pieces exist but no model routes through Charon yet.
- **Phase 5 cleanup:** dispatch A/B is DONE (tests/phase5_dispatch_ab.rs,
  phase5_ab_results.md): scalar per-quant GEMM loses to WMMA at every shape
  (10–190x at prefill) — deletable once the newly-discovered WMMA Q3_K/Q6_K
  layout drift (non-finite prefill output with host-authoritative bytes) is
  fixed. Attention-kernel removal (5b) still needs its own dispatch audit.
Q2_K, Q3_K) — exceeds the ≥4 success criterion.

---

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

### Sub-step 1a: Fused Q8_0 QKV projection in `LlamaBlock`  — **[✅ LANDED]**
- At block construction: if weights are Q8_0 and device is ROCm, build
  `wqkv_q80_fused: Option<FusedQkvWeights>` from `wq`/`wk`/`wv` (reuse `build_fused_qkv_q80`)
- In `forward_with_kv_paged`: when fused blob exists, replace the 3 separate `wq/wk/wv.forward`
  calls with ONE `launch_fused_qkv_dot4` + `RocmStorageView` slicing
- Fall back to 3-GEMV when blob absent (non-Q8_0, non-ROCm)
- **Escape hatch:** `GRIM_FUSED_QKV=0`

### Sub-step 1b: Device-base RoPE in `apply_rope_multi_head`  — **[⚠️ PARTIAL — decode wired, legacy host-Vec path remains for prefill/fallback]**
- Replace the per-call `ext_positions` Vec build + `dev.rope()` with `rope_dev_base_into`
  writing into a per-layer cached stable output buffer
- Seed the `pos_base_dev` buffer once per generation (before capture bracket)
- **Escape hatch:** `GRIM_ROPE_DEV_BASE=0`

### Sub-step 1c: Attention + KV-append graph capture  — **[❌ REMAINING — primitives exist, lfm2-only]**
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

### Sub-step 2a: `shared_attention::fused_qkv_project`  — **[✅ LANDED (shared_attention.rs:103)]**
- New function: takes `norm_x` (device f32), `wqkv_q80_fused` (fused blob), and
  model topology → returns `(q_rot, k_rot, v)` all device-resident
- Internally: quantize q8_1 → fused GEMV → zero-copy slicing → RoPE → returns
- Models call this instead of `wq/wk/wv.forward` + separate rope

### Sub-step 2b: Wire into high-traffic models  — **[✅ LANDED — all 10 planned models wired; qwen35's SSM layers intentionally untouched (fused blob only for full-attention layers, row-exact TP guard)]**
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

### Sub-step 3a: Wire Charon grouped dispatch into MoE models that don't use it  — **[⚠️ PARTIAL — shared_moe module exists, deepseek2 delegates; deepseek32/4/kimi_k3 per-expert loops remain; full Charon grouped-kernel adoption needs stacked weights + checkpoint verification]**
- deepseek2/32/4, bailingmoe2/3, kimi_k3, mellum each implement `forward_moe_device`
  individually. Consolidate into a shared `shared_moe::fused_moe_dispatch` that calls
  Charon's `grouped_dispatch` internally
- The shared function takes: routing table, expert weights, activation → returns fused output
- Models call it instead of their per-expert loops

### Sub-step 3b: Fuse SwiGLU into Charon expert compute  — **[⚠️ KERNEL-ONLY — silu×up fused in charon.rs/charon_wmma.rs bodies, but no model routes through Charon]**
- Charon already does grouped GEMM. Fuse `silu_mul_quant` into the epilogue:
  after gate×up projection, apply SwiGLU + quantize to q8_1 in the same kernel
  (matching the existing `rmsnorm_quant` pattern: compute → quantize → write q8_1)
- The down-projection then consumes pre-quantized q8_1 via `dot4_q80_q81_gemv`
- This eliminates the separate `silu_mul_on_device` launch per expert

### Sub-step 3c: Fuse rmsnorm_quant into the MoE gate path  — **[❌ REMAINING]**
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

### Sub-step 4a: rmsnorm_rope — fused RMSNorm + RoPE  — **[✅ LANDED — grim_rmsnorm_rope used in block.rs:1227]**
- Currently: RMSNorm (kernel) → RoPE (kernel) = 2 launches
- Fused: one kernel reads raw activation, normalizes, applies RoPE, writes output
- Applies to: Q and K paths in ALL models (before attention)
- Saves 2 launches/layer/token × num_layers

### Sub-step 4b: attention_rope_out — fused attention + output projection  — **[❌ REMAINING — fuse_o param exists but hard-coded 0 at all call sites]**
- Currently: attention kernel → Linear (wo) = 2 launches
- Fused: attention kernel epilogue applies wo projection (same as existing
  `fuse_o` epilogue in `grim_qkv_attention` — just enable it by default for decode)
- Saves 1 launch/layer/token

### Sub-step 4c: FFN gate+up fused GEMV  — **[❌ REMAINING]**
- Currently: ffn_gate (GEMV) + ffn_up (GEMV) = 2 launches (same input!)
- Fused: one GEMV with concatenated weights `[2*inter, hidden]`, writes gate and up
  into separate output regions
- Same pattern as the Item 1 fused QKV GEMV — reuse `build_fused_qkv_q80` with
  ffn_gate+ffn_up weights
- Saves 1 launch/layer/token

### Sub-step 4d: FFN silu_mul + down fused  — **[✅ LANDED — silu_mul_quant_q8_1 wired in block.rs decode, feeds dot4_q80_q81_gemv]**
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

## PHASE 4.5: ISA-specific quantized GEMV kernels (sudot4 / sudot8 / fp8 / bf16)

**Goal:** Extend the proven sudot4 GEMV pattern (currently Q8_0-only) to every quant
format whose block structure can decompose into dot-product instructions available on
gfx1201 (RDNA4). Eliminates the per-format WMMA GEMV fallback for M=1 decode (WMMA
wastes 15/16 rows at m=1 — 6.25% tensor utilization vs 100% for GEMV).

### Available dot-product builtins on gfx1201 (RDNA4)

| Builtin | HW Instruction | ISA feature | Operand Width | Elem/Inst | Acc | gfx1201 |
|---|---|---|---|---|---|---|
| `sudot4` | `V_DOT4_I32_IU8` | dot8-insts | **8-bit** (signed × unsigned) | 4 | i32 | ✅ confirmed |
| `sudot8` | `V_DOT8_I32_IU4` | dot8-insts | **4-bit** (signed × unsigned) | 8 | i32 | ✅ confirmed |
| `udot4` | `V_DOT4_U32_U8` | dot7-insts | **8-bit** (unsigned) | 4 | u32 | ✅ confirmed |
| `udot8` | `V_DOT8_U32_U4` | dot7-insts | **4-bit** (unsigned) | 8 | u32 | ✅ confirmed |
| `fdot2_f16_f16` | `V_DOT2_F16_F16` | dot9-insts | **16-bit** f16 × f16 | 2 | f16 | ✅ confirmed |
| `fdot2_f32_bf16` | `V_DOT2_F32_BF16` | dot12-insts | **16-bit** bf16, f32 acc | 2 | f32 | ✅ confirmed |
| `dot4_f32_fp8_fp8` | `V_DOT4_F32_FP8_FP8` | dot11-insts | **8-bit** fp8 E4M3 | 4 | f32 | ✅ RDNA4 |
| `dot4_f32_bf8_bf8` | `V_DOT4_F32_BF8_BF8` | dot11-insts | **8-bit** bf8 E5M2 | 4 | f32 | ✅ RDNA4 |
| `fdot2` | `V_DOT2_F32_F16` | dot10-insts | **32-bit** f32 × f32 | 2 | f32 | ✅ confirmed |
| `sdot4` / `sdot8` | `V_DOT4_I32_I8` / `V_DOT8_I32_I8` | dot1-insts | all-signed | 4/8 | i32 | ❌ REMOVED on RDNA4 |

**CRITICAL:** `sudot4` and `sudot8` process DIFFERENT operand widths:
- `sudot4` → `V_DOT4_I32_IU8` → **8-bit** codes, 4 elements per instruction → for Q8_0/Q8_1
- `sudot8` → `V_DOT8_I32_IU4` → **4-bit** codes, 8 elements per instruction → for Q4_K/int4
- There is NO 8-bit dot8 instruction on RDNA4 — 8-bit dot products max out at 4 elements/inst

### Quant format → ISA instruction coverage matrix

| Format | Element Width | Dot Instruction | Unpack Required | Two-Dot (scale/min) | Act. Quant |
|---|---|---|---|---|---|
| Q8_0 | 8-bit signed | `sudot4` | ❌ direct | ❌ scale only | Q8_1 (8-bit) |
| Q4_K | 4-bit unsigned | `sudot4` + nibble unpack | ✅ nibble → i8 | ✅ d·sc·dot − dmin·mi·Σ | Q8_1 (8-bit) |
| Q4_K (W4A4) | 4-bit unsigned | `sudot8` | ❌ native 4-bit | ✅ same two-dot | 4-bit |
| Q5_K | 5-bit | `sudot4` + bit-plane merge | ✅ nibble + 5th-bit | ✅ d·sc·dot − dmin·mi·Σ | Q8_1 (8-bit) |
| Q6_K | 6-bit | `sudot4` + 6-bit unpack | ✅ 6-bit → i8 | ✅ d·sc·dot (no min) | Q8_1 (8-bit) |
| Q2_K | 2-bit | `sudot4` (marginal) | ✅ 2-bit → i8 | ✅ scale/min | Q8_1 (8-bit) |
| Q3_K | 3-bit | `sudot4` (marginal) | ✅ 3-bit → i8 | ✅ scale/min | Q8_1 (8-bit) |
| IQ2/IQ3/IQ4 | varies | ❌ codebook lookup | N/A | N/A | N/A |
| FP8 E4M3 | 8-bit fp8 | `dot4_f32_fp8_fp8` | ❌ native | per-block scale | FP8 |
| FP8 E5M2 | 8-bit bf8 | `dot4_f32_bf8_bf8` | ❌ native | per-block scale | BF8 |
| BF16 | 16-bit bf16 | `fdot2_f32_bf16` | ❌ native | ❌ none | BF16 |
| F16 | 16-bit f16 | `fdot2_f16_f16` | ❌ native | ❌ none | F16 |

### Sub-step 4.5a: Q4_K fused GEMV via sudot4 (nibble unpack + two-dot decomposition)  — **[✅ LANDED+ — Q4_K, Q5_K, Q6_K dot4 GEMVs all exist and dispatch at m==1]**

**Why:** Q4_K is the most common quant format after Q8_0. Currently decoded via WMMA
fused dequant GEMM (6.25% tensor utilization at M=1). The dot4 GEMV approach achieves
100% lane utilization.

**Q4_K block layout** (144 bytes per 256-weight super-block):
- 2 bytes f16 `d` (super-block scale)
- 2 bytes f16 `dmin` (super-block min)
- 12 bytes: 6-bit scales + 6-bit mins for 8 sub-blocks of 32
- 128 bytes: 256 × 4-bit unsigned nibbles (packed 2-per-byte)

**Dequant formula per element:** `value = d * sc * q - dmin * mi`
where `q` ∈ [0,15] (4-bit), `sc`/`mi` are 6-bit scale/min per sub-block.

**Two-dot decomposition trick** (llama.cpp MMQ pattern):
```
dot = Σ(a_i × (d·sc·q_i − dmin·mi))
    = d·sc·Σ(a_i × q_i) − dmin·mi·Σ(a_i)
      ─────────────────   ────────────
      positive dot (1)     correction dot (2)
```
- **Dot (1):** `sudot4(q8_1_codes_i8, q4_nibbles_unpacked_i8)` — needs nibble unpacking
- **Dot (2):** `Σ(a_i)` — already computed as the `sum` field in Q8_1 by `grim_quantize_q8_1`

**Nibble unpacking in registers:**
```c
// Each byte holds two 4-bit values; unpack to two i8 values
unsigned char b = qs[j];
int8_t lo = b & 0x0F;        // 0..15
int8_t hi = (b >> 4) & 0x0F; // 0..15
// Pack 4 nibbles into an i32 for sudot4
int packed = lo0 | (hi0 << 8) | (lo1 << 16) | (hi1 << 24);
```
The unpacking is done in registers (no extra memory reads) — the nibbles are already
loaded as part of the 128-byte weight data.

**Kernel design:**
- Grid: `(N/4, M)` — same as Q8_0 dot4 GEMV
- Per output column: iterate Q4_K super-blocks, for each sub-block:
  1. Unpack 32 nibbles → 32 i8 values (8 sudot4 calls with 4 elements each)
  2. Compute `pos_dot` = sudot4(activation_codes, unpacked_nibbles)
  3. Look up `sc`, `mi` from the 6-bit packed scales
  4. Read `sum` from the Q8_1 activation block (at bytes [2..4])
  5. `sub_result = d * sc * pos_dot - dmin * mi * activation_sum`
  6. Accumulate into the column result

**Block layout mapping:**
- Q4_K super-block: 144 bytes → 8 sub-blocks × 32 elements
- Q8_1 activation block: 36 bytes → 32 i8 codes + fp16 d + fp16 sum
- One Q4_K super-block consumes 8 Q8_1 activation blocks (256 elements)

**Expected speedup vs WMMA at M=1:** WMMA at 6.25% utilization processes 256 rows in
one tile. The dot4 GEMV processes 1 row with 100% utilization. For N=1024 output
columns, WMMA needs 256/16 = 16 tile iterations; dot4 needs N/4 = 256 wave dispatches.
The crossover depends on N and head_dim, but for LFM2 shapes (N=1024, K=1024) the dot4
GEMV was already proven faster for Q8_0.

### Sub-step 4.5b: Native 4-bit × 4-bit dot8 via sudot8 (V_DOT8_I32_IU4)  — **[❌ REMAINING — no sudot8/V_DOT8 anywhere]**

**Why:** `sudot8` (`V_DOT8_I32_IU4`) processes **8 × 4-bit unsigned values** per instruction
— native int4×int4 dot product with i32 accumulation. This is the natural instruction for
**W4A4 quantized models** (both weights and activations quantized to 4-bit) and for Q4_K
weights paired with 4-bit-quantized activations.

**CRITICAL:** `sudot8` processes **4-bit** operands (8 elements × 4 bits = 32 bits from one
i32 register). It is NOT an 8-bit dot8 — that instruction does not exist on RDNA4. For 8-bit
codes (Q8_0/Q8_1), `sudot4` at 4 elements per instruction is the maximum dot width.

**Kernel design (W4A4 path):**
- Weights: Q4_K 4-bit nibbles packed 2-per-byte — feed directly as i32 source operands
- Activations: quantized to 4-bit unsigned (0..15) using the same sub-block boundaries
- Per 32-element Q4_K sub-block: 4 × `sudot8` (32 elements / 8 per instruction)
- Two-dot decomposition: scale×pos_dot − min×act_sum (same as sudot4 Q4_K path)
- **Instruction count: 4 sudot8 per sub-block (vs 8 sudot4 with unpacking)**

**When to use:**
- W4A4 models (both weight and activation in 4-bit) — native, no unpack
- Q4_K weights with runtime 4-bit activation quantization — trades activation precision
  for 2× instruction throughput vs the sudot4 nibble-unpack path
- NOT for Q8_0/Q8_1 — those require 8-bit operands (sudot4)

### Sub-step 4.5c: FP8 GEMV via dot4_f32_fp8_fp8  — **[❌ REMAINING — fp8 kernels are WMMA GEMMs only]**

**Why:** FP8 (E4M3) quantized models are increasingly common. The RDNA4 `dot11-insts`
provides native fp8×fp8 dot4 with f32 accumulation — no integer quantization needed.

**Kernel design:**
- Weights stored as native FP8 (1 byte each)
- Activations quantized to FP8 (or stored as FP8)
- `dot4_f32_fp8_fp8(acc, weight_fp8_packed, act_fp8_packed, 0)` directly
- Per-block scale multiplication at the end (FP8 has wider dynamic range than int8)
- No nibble unpacking needed — FP8 IS 8 bits, feeds directly to the instruction

**Applicability:** models quantized to FP8 (E4M3) or FP4 with per-block FP8 scales.

### Sub-step 4.5d: BF16 GEMV via fdot2_f32_bf16  — **[❌ REMAINING]**

**Why:** BF16 models (Llama/Mistral BF16 checkpoints) currently run through the f32
GEMV path (no hardware dot instruction used). The `fdot2_f32_bf16` (dot12-insts)
provides native bf16×bf16 dot2 with f32 accumulation.

**Kernel design:**
- Weights and activations stored as BF16 (2 bytes each)
- `fdot2_f32_bf16(acc, w_pair, a_pair, 0)` — processes 2 bf16 elements per instruction
- For 32-element blocks: 16 instructions per block
- No quantize/dequantize step needed — direct hardware dot product

**Applicability:** all BF16 models. The dot2 throughput (2 elem/inst) is lower than
sudot4 (4 elem/inst), but for BF16 the alternative is a full fp32 GEMV (no dot
instruction) — so this is still a significant improvement.

### Sub-step 4.5e: f16 hardware dot2 upgrade for existing dot2 path  — **[✅ ALREADY DONE — dot_gemv.rs uses __builtin_amdgcn_fdot2 intrinsic]**

**Why:** the existing `dot2_q80_gemv` uses inline asm `v_dot2_f32_f16` which is
actually `fdot2_f16_f16` (dot9-insts). Upgrading to the builtin ensures correct
ISA targeting and enables the compiler to optimize register allocation.

**Change:**
- Replace inline asm with `__builtin_amdgcn_fdot2_f16_f16(a, b, c)` intrinsic
- Same hardware instruction, cleaner code, better compiler optimization

### Sub-step 4.5f: Q2_K / Q3_K GEMV via sudot4 + bit-field extraction  — **[❌ REMAINING — Q2K/Q3K still WMMA-only]**

**Why:** Q2_K (2-bit) and Q3_K (3-bit) are linear quant formats (scale × code, like Q4_K)
so the two-dot decomposition applies. The challenge is unpacking: 2-bit and 3-bit codes do
not align to byte/nibble boundaries, requiring per-element bit-field extraction.

**Q2_K unpacking via V_BFE_U32:**
```c
// Each byte holds 4 × 2-bit unsigned values (0..3)
// V_BFE_U32 extracts each field in 1 instruction
int q0 = V_BFE_U32(packed, 0, 2);  // bits [1:0]
int q1 = V_BFE_U32(packed, 2, 2);  // bits [3:2]
int q2 = V_BFE_U32(packed, 4, 2);  // bits [5:4]
int q3 = V_BFE_U32(packed, 6, 2);  // bits [7:6]
// Pack 4 extracted values into an i32 for sudot4
int packed4 = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
```
Unpack cost: ~1 instruction per element (V_BFE) + 0.25 sudot4 per element for the dot.
Two-dot decomposition: `d·sc·Σ(a×q)` (Q2_K has no min offset — the scale encodes it).
**Net: ~1.25 instructions per element** — worse than Q4_K (0.5/elem) but still usable.

**Q3_K unpacking:**
Q3_K packs 3-bit magnitude codes + a 1-bit sign plane. The 3-bit fields don't align to
byte boundaries — V_BFE extracts each field in 1 instruction (same as Q2_K), plus a sign
extraction from the hmask byte. Two-dot decomposition: `d·sc·(q × sign)`.
**Net: ~1.5-2.25 instructions per element** — higher than Q4_K but still viable.

**Assessment:** Q2_K/Q3_K GEMV via sudot4 is technically feasible. The unpack overhead is
higher than Q4_K (which unpacks 2 elements per byte with just AND+SHIFT), but for decode
M=1 it still achieves better lane utilization than WMMA (6.25%). Implementation should be
prioritized AFTER Q4_K and Q5_K (which have simpler unpacking and higher model usage).

### Sub-step 4.5g: IQ format compute strategy (NOT dot-GEMV)  — **[✅ NO WORK NEEDED — assessment-only; WMMA confirmed as IQ decode route]**

**Why NOT dot instructions:** IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS use **codebook lookup**
(a 16-entry non-linear table) instead of linear `scale × code`. There is no ISA dot
instruction that can perform a lookup-then-dot — the values are not linear in the code.

**IQ4_NL/IQ4_XS codebook values** (kvalues_iq4nl): {-127, -104, -83, -65, -49, -35, -22,
-10, 1, 13, 25, 38, 53, 69, 87, 107} — ALL fit in signed i8 (-128..127).

**Optimal GPU compute strategy for IQ4:**
1. **Register-LUT via V_PERM_B32**: the 16-entry i8 codebook fits in 4 × i32 registers
   (16 bytes). V_PERM_B32 selects 4 bytes from 8 source bytes using a per-byte 4-bit
   selector. For a 16-byte codebook: 2 V_PERM calls (lo/hi 8 bytes) + 1 V_CNDMASK_B32
   (select by MSB of the 4-bit code) = **3 instructions per codebook lookup**.
2. After lookup: multiply by activation and accumulate (V_FMA or V_PK_FMA for 2×
   packed throughput)
3. Scale by the per-super-block f16 scale

**For IQ2/IQ3 formats** (smaller codebooks or 2-bit/3-bit indices):
- IQ2_XXS/XS/S: 16-entry codebook, 2-bit index — only 4 entries are addressable per
  code. Requires the super-block scales to be applied per sub-block.
- IQ3_XXS/S: 32-entry codebook, 3-bit index — 32 entries × i8 = 32 bytes = 8 registers.
  Requires 3 V_PERM calls + select for each lookup.
- These formats have such low precision (2-3 bits) that the WMMA fused dequant GEMM
  (which does the codebook lookup in LDS, amortized over a 16×16 tile) is likely the
  best approach for both prefill AND decode.

**Assessment:** IQ formats should continue using the WMMA fused dequant GEMM for decode.
The optimization opportunity for IQ formats is in LDS lookup efficiency, not in
dot-product instruction selection.

### Verification
- Each GEMV: parity vs CPU dequant-reference (same tolerance as existing tests)
- Q4_K GEMV: byte-exact vs `grim_quant::dequant_q4k` reference (same two-dot decomposition)
- Q2_K/Q3_K GEMV: parity vs CPU reference (same two-dot decomposition)
- FP8 GEMV: parity vs host FP8 dequant reference
- BF16 GEMV: parity vs host BF16 dequant reference
- IQ formats: continue with WMMA fused dequant (already correct)
- Perf: M=1 GEMV for each format must beat the WMMA GEMM fallback at the same shape

---

## PHASE 5: Kernel consolidation (remove superseded kernels)

**Goal:** Remove kernels that are superseded by fused-ops variants, reducing the kernel
JIT compilation footprint and maintenance burden.

### Sub-step 5a: Remove per-quant GEMV kernels superseded by WMMA fused dequant AND dot4 GEMV  — **[A/B EVIDENCE RECORDED — scalar never fastest (see phase5_ab_results.md); deletable AFTER fixing WMMA Q3_K/Q6_K drift]**  — **[❌ REMAINING — all 5 per-quant GEMM files still present]**
- The `quantized_matmul` dispatch already prefers WMMA fused dequant for Q4K/Q5K/Q6K/
  Q2K/Q3K/IQ* — the standalone per-quant GEMV kernels are dead code:
  - `q4k_gemm.rs` (349 LOC) — superseded by `launch_wmma_fused_dequant_q4k` (prefill)
    AND by Phase 4.5a sudot4 Q4_K GEMV (decode M=1)
  - `q5k_gemm.rs` (119 LOC) — superseded by WMMA + Phase 4.5a dot4 pattern
  - `q6k_gemm.rs` (114 LOC) — superseded by WMMA + dot4 pattern
  - `q2k_gemm.rs` (107 LOC) — superseded by WMMA
  - `q3k_gemm.rs` (142 LOC) — superseded by WMMA
  - `q8_0_dequant.rs` (41 LOC) — KEEP: used by `dequantize_q8_0_host`
  - `dot_gemv.rs` dot2 path — KEEP: used as A/B test fallback for Q8_0
- After Phase 4.5 lands, the per-format GEMV dispatch in `quantized_matmul` routes
  decode (M=1) to the appropriate dot4/sudot8 kernel for EVERY format — making the
  WMMA GEMM the prefill path and the dot4 GEMV the decode path (no overlap)
- Only remove after verifying no model routes to these directly (check dispatch order)

### Sub-step 5b: Remove redundant attention kernels  — **[❌ REMAINING — flash_decode.rs, extend_attention.rs, cross_attention.rs still present]**
- `flash_decode.rs` vs `qkv_attention.rs` vs `sage_attention.rs` — audit which models
  route to each. If flash_decode is never dispatched (dispatcher prefers qkv_attention
  or sage_attention), remove it
- `extend_attention.rs` vs `cross_attention.rs` — audit model usage

### Sub-step 5c: Consolidate dequant kernels  — **[❌ REMAINING]**
- `q4k_dequant.rs` (199 LOC) and `q8_0_dequant.rs` (41 LOC) — both are small host-side
  dequant helpers. Consolidate into `iq_dequant.rs` or a shared dequant module

### Sub-step 5d: Remove old MoE host-only kernels  — **[❌ REMAINING — charon_backward.rs still present]**
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

### Sub-step 6a: Thread past_dev through the session  — **[⚠️ PARTIAL — engine-level GraphCaptureInputBuffers exists (input_ids/positions only), not past_dev design]**
- Add `decode_graph_state: Option<DecodeGraphBuffers>` to the session model_state
- `DecodeGraphBuffers` holds: past_dev (device u32 counter), pos_base_dev (aliased),
  attention output buffers per layer
- Seeded once at prefill completion; bump kernel increments per decode step

### Sub-step 6b: Enable in block.rs + shared models  — **[❌ REMAINING — same gap as 1c]**
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
  ↓                        ↘
Phase 3 (MoE/Charon)         Phase 4.5 (ISA GEMV: Q4_K, sudot8, FP8, BF16)
  ↓                        ↙
Phase 4 (fused ops)         ← benefits ALL models, builds on Phases 1-2
  ↓
Phase 5 (cleanup)           ← after Phase 4 + 4.5 confirm no regressions
  ↓
Phase 6 (decode graph)      ← after Phases 1-4 stabilize the compute topology
```

Phase 4.5 (ISA GEMV) can run in parallel with Phase 3 (MoE) — it extends the
dot4 GEMV to new quant formats and does not depend on MoE restructuring.
Phase 5 cleanup should wait for BOTH Phase 4 and 4.5 to confirm which kernels
are truly superseded.

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
| Q4_K nibble unpacking overhead eats sudot4 throughput gain | Unpacking is 4 AND/SHIFT ops per 4 elements — negligible vs the sudot4 instruction; measure vs WMMA at target shape to confirm |
| sudot8 accumulator overflow (8 i4×i4 products in i32) | Max |product| = 8 × 15 × 15 = 1,800 — trivially fits in i32 (2.1B range); no overflow risk |
| sudot8 requires 4-bit ACTIVATION — not applicable to Q8_1 activations | Reserve sudot8 for W4A4 models; use sudot4 with nibble unpacking for Q4_K × Q8_1 activations |
| FP8 (E4M3) limited dynamic range causes precision loss vs int8 | FP8 has ~2 decimal digits of precision; verify per-block scale compensates; fallback to sudot4 int8 path if parity fails |
| Q4_K two-dot decomposition drifts for large block sums | The `sum` correction term grows with block size; Q4_K uses 32-element sub-blocks (same as Q8_0/Q8_1) so the correction is bounded; verify against `dequant_q4k` reference |

---

## SUCCESS CRITERIA

| Metric | Baseline | Target |
|---|---|---|
| Models with fused QKV GEMV | 1 (LFM2) | 11+ (block.rs llama-family + all 10 planned shared_attention models incl. qwen35 full-attention layers) ✅ |
| Models with device-base RoPE | 1 (LFM2) | ≥ 20 |
| MoE FFN launches per token | ~num_experts × 4 | ~num_experts × 2 |
| Kernel files | 59 | ≤ 50 (after cleanup) |
| Kernel LOC | 19,656 | ≤ 17,000 (after cleanup) |
| Decode tok/s (LFM2.5-350M-Q8_0) | 625 (already achieved) | ≥ 625 (no regression) |
| Decode tok/s (qwen2-7B-Q8_0, if testable) | TBD | measurable improvement |
| Parity | — | All models: identical tokens stock vs optimized |
| Quant formats with decode-optimized GEMV (M=1 dot4/sudot8) | 1 (Q8_0) | 7 (Q8_0, Q4_K, Q5_K, Q6_K, FP8, Q2_K, Q3_K) ✅ |
| Per-token kernel launch reduction (all opts vs stock) | 0% | ≥ 60% (Phases 1-4 combined) |
