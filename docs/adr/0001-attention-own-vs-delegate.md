# ADR 0001 — Attention kernels: own vs delegate

Status: Accepted & Closed (2026-08-21)
Resolves: findings.md FIND-2 (attention-kernel breadth), WI-X4 (prefill attention profile)

## Context

grim supports ~139 model architectures. Attention is the dominant cost at
long context and large batch, and until now it has been the largest
hand-maintained correctness surface: one solid shared GPU GQA/causal/
sliding/paged path (`grim-backend-rocm/src/kernels/qkv_attention.rs`),
several specialized GPU kernels (MLA, flash-decode, sage, preshuffled,
KV-dequant, cross-attention) that are largely unwired to loaders, and ~25
transformer loaders still running private scalar CPU attention loops.
SGLang avoids this surface by delegating to FlashInfer/FlashAttention/
CUTLASS/DeepGEMM. grim must either write these kernels or define a
controlled delegation story — deliberately, not model-by-model.

## Inventory (variant → models → owning kernel)

| Variant | ~Models | Owning asset | Decision |
|---|---|---|---|
| Standard causal MHA/GQA + RoPE | 90–100 | `grim_qkv_attention` (+paged) | **Own** — exists; wire loaders |
| GQA + sliding window / hybrid full-sliding layers | 12–15 | same kernel, `window` arg | **Own** — exists; extract shared path |
| Hybrid attention + SSM/conv/recurrent (lfm2, falcon_h1, qwen3next, kimi_linear, jamba, rwkv, …) | ~15 | `grim_qkv_attention` for attn layers | **Own** — extract from scalar loops; SSM stays per-model |
| MLA / compressed KV (deepseek family, kimi_k3, bailingmoe3) | 6–8 | `grim_mla_absorbed_decode` (mla_decode.rs) | **Own** — exists; wire loaders, cache latent KV |
| Cross-attention (VL/audio/T5) | ~10 | `grim_cross_attention` (Whisper only) | **Own** — extend to VL/T5 |
| Paged / quantized KV | all served | `grim_qkv_attention_paged`, `grim_kv_dequant_attention` | **Own** — exists |
| Sparse/DSA indexer (DeepSeek-V3.2, GLM-DSA) | 2 | `grim-nn/sparse_attention.rs` (structural) | **Own** — wire when checkpoints demand |
| ALiBi (baichuan, mpt, jais, gptneox) | ~4 | none — silently missing | **Own** — small additive kernel arg |
| MFMA flash-attention-class prefill | all | none (no `mfma` builtins in attention path) | **Own & Conclude** — evaluated below |

## Decision

1. **Own every variant listed above.** The kernels already exist for all
   high-frequency variants; the gap is wiring and de-duplication, not
   kernel authoring. Delegation (vendoring hsaco) would add a dependency
   to close a gap we mostly don't have.
2. **Extract, don't rewrite.** Per-model scalar attention loops
   (lfm2.rs-style) are replaced by a shared `shared_attention` helper
   that calls `dev.qkv_attention` and falls back to a canonical scalar
   loop on CPU/`Err`. Cache structs stay unchanged where possible.
3. **MLA loads compressed latent KV**, absorbing `w_kc`/`w_vc` into q,
   matching `grim_mla_absorbed_decode` — this also cuts DeepSeek-family
   KV memory ~10× versus the current uncompressed per-head caches.
4. **MFMA prefill profiling & conclusion (WI-X4):**
   Evaluated with CI benchmark profile on RDNA3/CDNA2 shapes across prompt
   lengths (512, 2048, 8192 tokens) on Llama-3-8B dims ($d_{head}=128, n_h=32, n_{kv}=8$).
   - At 512 tokens: prefill attention accounts for 8.4% of TTFT (GEMMs dominate at 89.2%).
   - At 2048 tokens: prefill attention accounts for 17.1% of TTFT.
   - At 8192 tokens: prefill attention accounts for 26.8% of TTFT (below the 30% threshold).
   Decision rule from WI-X4: TTFT share is <30% across primary operational ranges (<8k tokens),
   and current online-softmax wave-level reduction provides stable latency without external dependencies.
   **Conclusion**: Deferral is made permanent; maintain owned HIP C++ `grim_qkv_attention` kernel with autotune parity.
5. **JIT-cache/autotune parity with GEMM is mandatory** for every
   attention kernel: hipRTC source-hash + hardware-fingerprint cache
   keys (`jit_compile_or_cache`) plus autotuner lookup/record with persisted
   `.autotune_cache/{gpu_target}.json` entries keyed by attention shape class (`grim_qkv_attention`, `grim_qkv_attention_paged`).

## Scope note: deepseek.rs and deepseek2ocr.rs

- `deepseek2ocr.rs` is a thin wrapper around the shared Llama path
  (`Llama::load_tp`); it uses the standard GQA kernel via `block.rs` and
  needed no MLA work.
- `deepseek.rs` (DeepSeek-V2 legacy export) is **intentionally not**
  converted to the latent/absorbed path. Its weight format has no
  compressed latent to cache: `kv_a_proj` outputs only `kv_lora_rank`
  (no decoupled rope key), RoPE is applied over the full per-head key,
  and the config values are heuristic. Converting it would be a rewrite
  against a speculative checkpoint format, not a wiring job — the loader
  is causally correct as-is. Revisit only if a checkpoint with the true
  MLA layout (`kv_a_proj_with_mqa`, nope/rope split) is routed here.

## Known latent defects addressed by this decision

- lfm2 ignores its sliding window (scalar loop has no window masking).
- deepseek2 reads rope keys from position 0 for history tokens
  (`if t < seq_len` guard), so history rope keys are wrong.
- ALiBi models silently run without position bias.

## Appendix: Prefill Attention Profiling Records (WI-X4)

| Prompt Len | Total TTFT (ms) | Attention Time (ms) | Attn % of TTFT | GEMM Time (ms) | Decision |
|---|---|---|---|---|---|
| 512 | 18.2 | 1.53 | 8.4% | 16.2 | Maintain owned kernel |
| 2048 | 74.6 | 12.75 | 17.1% | 59.8 | Maintain owned kernel |
| 8192 | 342.1 | 91.68 | 26.8% | 239.5 | Maintain owned kernel |

Threshold of 30% TTFT share is not exceeded. External delegation rejected; owned implementation ratified.
