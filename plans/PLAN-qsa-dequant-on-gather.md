# PLAN: dequant-on-gather for grim's sparse (QSA) attention — strata transfer

Source concept: `old/repo/strata/include/strata/kernels/kv_q8.hpp` +
`src/kernels/kv_q8_parity.cpp` (strata's "plan v0.3 P7: 8-bit KV storage for
the QSA layers"). This plan scopes the same architecture for grim.

## Strata's design (the reference)

- Storage: int8 codes + **one fp16 scale per 64 values** (per token-cell, per
  KV head), scales in a **separate array**. `scale = fp16(max|x|/127)`,
  `code = clamp(rint(x/scale), -127, 127)`, `x' = fp16(code*scale)`.
  1.03 B/value (vs 2.0 fp16).
- **Dequant-on-gather**: `kv_gather_q8_step` dequantizes the *selected* cells
  into the same fp16 scratch the existing attention kernels read, so
  `qsa_attend` is untouched. Quantization is a pure storage concern; the
  attention math stays fp16 (never quantized).
- Cheap because QSA attention is **sparse**: ~2,051 indexer-selected cells of
  32K are gathered per layer, not the whole context.
- Gate: `kv_q8_parity.cpp` — codes+scales bitwise vs host; gather bitwise vs
  host dequant; ≤ 0.624 quantization steps vs fp16.

## Grim's QSA path today

`crates/grim-models/transformer/src/qwen4exp_flash_next.rs` (+`qwen4exp_qsa.rs`,
`qwen4exp_qsa_blocks.rs`), transcribed from upstream `qwen4exp.cpp`:

- The indexer and the masked softmax are **host reference code**
  (`GRIM_QWEN38_QSA=1`, default on). Per token per sparse layer the K/V
  history is read back across PCIe for the indexer — a documented real cost.
- The selection becomes an **additive mask**; the attention reads the KV
  **arena** (Q8_1 default since 2226f520) over ALL rows with the mask.
- So grim today: host indexer (full history read) + dense quantized attention
  (all rows read). Strata: device indexer + gather of selected cells only.

## Proposed grim design (scoped)

1. **Device gather+dequant kernel** `grim_kv_gather_q81_dequant` (or reuse the
   arena-agnostic form): given per-layer selected row ids (the QSA top-K
   cells), read the Q8_1 arena rows and write **fp16 scratch**
   `[id][kv_head][head_dim]` — exactly strata's `kv_gather_q8_step` contract.
   For int8 arenas the same kernel with `(code-128)/127*scale`.
2. **Device indexer** for QSA: the indexer math (pool → norm → rope →
   per-head ReLU(q·k) → sum → top-K) as a HIP kernel. Grim already has the
   pattern for MLA (`grim_sparse_index_score`, `launch_sparse_index_score` in
   `crates/grim-backend-rocm/src/kernels/mla_decode.rs` +
   `device/attention/flash_decode.rs`) — the QSA indexer is a second instance,
   not a new idea.
3. **Dense fp16 attention over the gathered rows**: the gathered scratch is
   read by the existing fp16 attention path (`grim_qkv_attention_dev*` with an
   f16 arena), eliminating both the per-token PCIe history read and the
   all-rows masked read.
4. **Storage**: keep the Q8_1 arena (default); the gather is the only new
   buffer (fp16 scratch, sized by top-K × kv_heads × head_dim per layer).

## Gates (strata methodology, already adopted for arenas)

- Three-part quant gate on the gathered values: codes/scales bitwise vs host
  (pattern: `tests/q81_kv_arena_parity.rs`, `tests/i8_kv_arena_parity.rs`),
  gathered values within half a step + fp16 rounding of the original.
- Parity vs the current host-QSA masked path: exact-token parity on a fixed
  prompt, plus the KL/top-1 harness (`tests/kv_quant_quality.rs`) against the
  fp16-KV reference.
- Perf: gather traffic (1.03 B read + 2 B write per gathered cell) vs the
  saved full-history PCIe read + full masked arena read; measure with
  `GRIM_QWEN38_QSA=0/1` A/B.

## Scope / effort

- Gather kernel + launcher + gate: S (the kernel is ~50 lines; the arena row
  addressing already exists in `grim_kv_append_q81`).
- Device QSA indexer: M (second instance of `grim_sparse_index_score`).
- Model-path wiring (qwen4exp_flash_next using the gather instead of the host
  mask): M, parity-gated.
- NOT in scope: applying the gather to dense layers (4–5× traffic, see the
  kv_quant_quality/arena notes) — sparse selection is the whole point.

## Standing convention (adopted)

**Quantize KV storage, never the attention math.** Grim's int8 arena kernel
keeps Q in f32; the q8_1 kernels were fixed to do the same (13f70e91) after
quantizing Q flipped Gate D token 3. Any new quantized-KV path (including the
QSA gather) must keep Q/softmax in fp32/fp16 and dequantize K per element.
