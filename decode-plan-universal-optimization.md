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

## PHASE 4.5: ISA-specific quantized GEMV kernels (sudot4 / sudot8 / fp8 / bf16)

**Goal:** Extend the proven sudot4 GEMV pattern (currently Q8_0-only) to every quant
format whose block structure can decompose into dot-product instructions available on
gfx1201 (RDNA4). Eliminates the per-format WMMA GEMV fallback for M=1 decode (WMMA
wastes 15/16 rows at m=1 — 6.25% tensor utilization vs 100% for GEMV).

### Available dot-product builtins on gfx1201 (RDNA4)

| Builtin | ISA feature | Types | Throughput | gfx1201 |
|---|---|---|---|---|
| `sudot4` | dot8-insts | i8 × i8 (signed/unsigned mix), i32 acc | 4 elem/inst | ✅ confirmed |
| `sudot8` | dot8-insts | i8 × i8 (signed/unsigned mix), i32 acc | 8 elem/inst | ✅ confirmed |
| `udot4` | dot7-insts | u8 × u8, u32 acc | 4 elem/inst | ✅ confirmed |
| `udot8` | dot7-insts | u8 × u8, u32 acc | 8 elem/inst | ✅ confirmed |
| `fdot2_f16_f16` | dot9-insts | f16 × f16, f16 acc | 2 elem/inst | ✅ confirmed |
| `fdot2_f32_bf16` | dot12-insts | f32 × bf16, f32 acc | 2 elem/inst | ✅ confirmed |
| `dot4_f32_fp8_fp8` | dot11-insts | fp8 × fp8, f32 acc | 4 elem/inst | ✅ RDNA4 |
| `dot4_f32_bf8_bf8` | dot11-insts | bf8 × bf8, f32 acc | 4 elem/inst | ✅ RDNA4 |
| `sdot4` / `sdot8` | dot1-insts | i8 × i8 (all signed) | 4/8 elem/inst | ❌ REMOVED on RDNA4 |
| `fdot2` | dot10-insts | f32 × f32, f32 acc | 2 elem/inst | ✅ confirmed |

### Sub-step 4.5a: Q4_K fused GEMV via sudot4 (nibble unpack + two-dot decomposition)

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

### Sub-step 4.5b: Q8_0 throughput upgrade via sudot8

**Why:** `sudot8` processes 8 i8 elements per instruction (vs sudot4's 4) — 2× throughput
for the same register pressure. The Q8_0 × Q8_1 GEMV already feeds i8×i8 to sudot4;
switching to sudot8 halves the inner-loop instruction count.

**Change:**
- In `grim_dot4_q80_q81_gemv`: replace 2 × `sudot4` with 1 × `sudot8`
- Each sudot8 takes two i32 operands (8 bytes total) and an i32 accumulator
- The Q8_1 activation block has 32 i8 codes = 8 bytes = 2 × i32 → exactly 1 sudot8
- The Q8_0 weight block has 32 i8 codes = 32 bytes → 4 × sudot8 per weight block
  (vs 8 × sudot4 currently)
- **Instruction count: 8 sudot4 → 4 sudot8 per 32-element block = 2× fewer**

**Compatibility:** same ISA feature (`dot8-insts`), same accumulator type. Drop-in
replacement — just change the builtin call and the packing.

### Sub-step 4.5c: FP8 GEMV via dot4_f32_fp8_fp8

**Why:** FP8 (E4M3) quantized models are increasingly common. The RDNA4 `dot11-insts`
provides native fp8×fp8 dot4 with f32 accumulation — no integer quantization needed.

**Kernel design:**
- Weights stored as native FP8 (1 byte each)
- Activations quantized to FP8 (or stored as FP8)
- `dot4_f32_fp8_fp8(acc, weight_fp8_packed, act_fp8_packed, 0)` directly
- Per-block scale multiplication at the end (FP8 has wider dynamic range than int8)
- No nibble unpacking needed — FP8 IS 8 bits, feeds directly to the instruction

**Applicability:** models quantized to FP8 (E4M3) or FP4 with per-block FP8 scales.

### Sub-step 4.5d: BF16 GEMV via fdot2_f32_bf16

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

### Sub-step 4.5e: f16 hardware dot2 upgrade for existing dot2 path

**Why:** the existing `dot2_q80_gemv` uses inline asm `v_dot2_f32_f16` which is
actually `fdot2_f16_f16` (dot9-insts). Upgrading to the builtin ensures correct
ISA targeting and enables the compiler to optimize register allocation.

**Change:**
- Replace inline asm with `__builtin_amdgcn_fdot2_f16_f16(a, b, c)` intrinsic
- Same hardware instruction, cleaner code, better compiler optimization

### Verification
- Each GEMV: parity vs CPU dequant-reference (same tolerance as existing tests)
- Q4_K GEMV: byte-exact vs `grim_quant::dequant_q4k` reference (same two-dot decomposition)
- sudot8 upgrade: bit-identical to sudot4 path (same math, different instruction width)
- FP8 GEMV: parity vs host FP8 dequant reference
- BF16 GEMV: parity vs host BF16 dequant reference
- Perf: M=1 GEMV for each format must beat the WMMA GEMM fallback at the same shape

---

## PHASE 5: Kernel consolidation (remove superseded kernels)

**Goal:** Remove kernels that are superseded by fused-ops variants, reducing the kernel
JIT compilation footprint and maintenance burden.

### Sub-step 5a: Remove per-quant GEMV kernels superseded by WMMA fused dequant AND dot4 GEMV
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
| sudot8 accumulator overflow (8 i8×i8 products in i32) | Max |product| = 8 × 127 × 127 = 129,032 — fits easily in i32 (2.1B range); safe for any block size ≤ 4096 |
| FP8 (E4M3) limited dynamic range causes precision loss vs int8 | FP8 has ~2 decimal digits of precision; verify per-block scale compensates; fallback to sudot4 int8 path if parity fails |
| Q4_K two-dot decomposition drifts for large block sums | The `sum` correction term grows with block size; Q4_K uses 32-element sub-blocks (same as Q8_0/Q8_1) so the correction is bounded; verify against `dequant_q4k` reference |

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
| Quant formats with decode-optimized GEMV (M=1 dot4/sudot8) | 1 (Q8_0) | ≥ 4 (Q8_0, Q4_K, Q8_0-sudot8, FP8) |
| Per-token kernel launch reduction (all opts vs stock) | 0% | ≥ 60% (Phases 1-4 combined) |
