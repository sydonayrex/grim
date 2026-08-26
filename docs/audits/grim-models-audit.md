# grim-models source audit — findings and dispositions (2026-08-25)

Scope: `crates/grim-models` (~43k lines: transformer 141 files / 35.5k,
mamba, vision, audio, diffusion) plus the model-adjacent seams this audit's
chunked-prefill work made load-bearing: the engine session contract
(`advance_pos`), KV-append semantics, and paged-attention paths.

Method: full read of the shared infrastructure every model builds on
(`block.rs`, `model.rs`, `kv_attention.rs`, `shared_attention.rs`,
`attention_dispatcher.rs`, `moe_block.rs`, lib), then a systematic sweep of
all 141 model files for the bug classes this codebase has actually produced
(missing `session.advance_pos`, stateless forwards, error-swallowed KV
appends, storage-rank relabels, fabricated-data fallbacks), then targeted
deep reads of every flagged file.

Findings grouped as in `scythe-audit-fix-plan.md`: **Live** (breaking a
wired-in path today), **Trap** (breaks the moment anything calls it),
**Latent** (wrong/silent but not immediately harmful).

---

## Live — fixed in this pass

### M1. MiniCPM: four independent defects; serving was broken end-to-end
`transformer/src/minicpm.rs` — wired via `ModelArchitecture::MiniCpm |
MiniCpm3 → MiniCpmModel::load`.
1. **Session position never advanced** — no `advance_pos` anywhere; the
   engine's decode start_pos stayed 0 forever, so every decode token ran at
   RoPE position 0 while its KV cache grew.
2. **Paged append stored PRE-RoPE keys** — `append_kv_layer(layer, &k, &v)`
   while the query ran rotated (`k_rot` computed and discarded). block.rs's
   contract is explicit: pages hold POST-RoPE K.
3. **`paged_self_attention` was a stub** — ignored `_block_table`,
   `_k_pages`, `_v_pages` entirely and returned
   `prefilled_self_attention(q, q, q)`: every served token attended to its
   own query as keys AND values instead of its history.
4. **Storage-rank relabels** — all 3-D↔2-D conversions were hand-rolled
   `Tensor::new(storage, shape)` relabels; CPU matmul validates *storage*
   rank, so ANY multi-token forward failed with "matmul expects 2-D inputs"
   before producing a single logit.
Also fixed en route: `prefilled_self_attention` indexed `q_dims[1]` as the
sequence dim unconditionally (2-D producers got garbage shapes).

Fixes: `advance_pos(seq_len)`; append `k_rot`; failed appends skip the paged
read (fall back to the classic cache path — always correct); real
`paged_self_attention` implemented over `shared_attention::gather_paged_history`
+ `kv_attention::causal_attention`; all rank conversions routed through
`block::reshaped_view` (physical reshape); rank-tolerant attention entry.
Gates: `minicpm_forward_advances_session_position`,
`minicpm_paged_matches_classic_attention` (synthetic-weights FullProvider
fixture; parity ≤1e-4).

### M2. Mamba: decode was context-free AND the CPU scan panicked on real shapes
`mamba/src/lib.rs` — wired via three loader branches (Mamba/Mamba2/hybrid).
1. `CausalLm::forward` called `init_state(1)` fresh on EVERY call and
   dropped it: prefill worked (whole prompt in one call), but every decode
   step saw one token from zeroed state — context-free output. The state
   now persists on the session (`model_state`), with `advance_pos`, gated by
   `mamba_forward_keeps_state_across_calls` (session-threaded second call ==
   explicit init→step→step reference, byte-exact).
2. `step_block_cpu` skipped `in_proj` entirely and sliced the RAW hidden as
   if it were the xz pair, indexing `xz[d_inner + n]` past the end — panic
   for any config with `2*d_inner > hidden_size` (every real shape). The
   dataflow now follows the weight shapes the file itself defines:
   norm → in_proj → xz `[x | z]` → selective scan on x → z-gated mix →
   out_proj. `step_block_gpu` had the same out-vector sizing bug
   (host vec sized `hidden`, written `d_inner`) — fixed.

### M3. FalconH1: session position never advanced
`transformer/src/falcon_h1.rs` — RoPE'd hybrid attention, own forward, no
`advance_pos`: identical failure mode to M1.1 (decode at position 0 forever).
Fixed + same-class reasoning as above.

## Trap — FIXED in the follow-up pass (same day)

### M4. bailingmoe3 (Ling3Tiny): fully stateless forward — FIXED + loader refuses
Ling3Tiny's forward now threads per-layer caches held on the session
(`Ling3LayerCache`: KDA → real `KdaLayerCache` conv/recurrent state; MLA →
post-RoPE KV history), advances `advance_pos`, and errors on variant/cache
mismatches. This required implementing the MLA cache in grim-nn itself:
`MlaAttention::forward` appended-and-attended over a flat post-RoPE history
when a cache is attached (`MlaKvCache` gained token-major `hist_*` fields);
the uncached prefill path is untouched. Gate:
`mla_cached_decode_matches_full_prefill` in grim-nn (cached [1]+[1] decode ==
uncached [1,2] prefill, ≤1e-5). The loader's BailingMoe3→**Qwen3Moe** mapping
is REMOVED: the architecture is now refused with a loud Config error until
Ling3Tiny has a GGUF hparams/tensor mapping — silently building a mismatched
model was the worst outcome. Ling3Tiny remains constructible via its own
config path for when that mapping lands.

### M5. RWKV: state threading unimplemented — IMPLEMENTED
`rwkv.rs` now runs the canonical RWKV-4 single-token recurrence with real
state: per-layer five-buffer state `[attn_xx, aa, bb, pp, ffn_xx]` on the
session; token-shift mixes against the previous post-LN hidden; the WKV
one-token update (`ww = u + k`, `pp' = p + w`) with decay/first loaded from
checkpoint buffers `att.time_decay`/`att.time_first`/time-mix ratios
(neutral defaults for synthetic models so the recurrence still exercises);
channel-mix token-shift through the newly-loaded `ln_2`. The ROCm
`rwkv_time_mix`/`rwkv_channel_mix` auto-dispatch was REMOVED, not bypassed —
their signatures have no state I/O, so they can never implement recurrence
and were producing silently context-free output on ROCm builds; they remain
in the backend for future state-aware wiring. `StatefulSequence::step` walks
multi-token inputs token-by-token; `CausalLm::forward` persists state +
advances position. Gates: `rwkv_forward_keeps_state_across_calls`
(byte-exact vs explicit init→step→step threading), 
`rwkv_recurrence_changes_output_with_history` (identical final token after
different histories must differ — the pre-fix memoryless model failed this).

## Latent — recorded

- **M6** `block.rs` + `minicpm.rs` paged-append failures previously
  `.ok()`-swallowed then read stale pages. Now: failed append skips the
  paged read (classic path). Residual `.ok()` on `paged_self_attention`
  itself degrades to the classic path — acceptable (independent cache).
- **M7** `shared_attention::fused_or_scalar_attention_paged` fallback
  ignored block tables (treated the page arena as linear history — wrong for
  any non-contiguous allocation) and `unwrap_or_default()`-ed failed D2H
  reads into empty K/V (fabricated zeros). Fixed: gathers through the table
  (`gather_paged_history`, unit-gated incl. short-table rejection) and
  propagates read errors.
- **M8** `model.rs::weights_look_broken` (ex-P1-36 `check_not_zeroed`)
  rejected ALL-CONSTANT tensors — including legitimately all-ones RMS-norm
  weights that ship in real checkpoints. The constant check now applies only
  to rank ≥ 2 tensors; zeros are still always rejected. Gated.
- **M9** Several models silently fall back to 0-based positions when the
  positions tensor's element count ≠ seq_len (`model.rs`, `minicpm.rs`, …)
  or when positions are short (`apply_rope_multi_head`). Masks caller bugs;
  an Err would be better. Not changed (engine always passes matching lengths).
- **M10** `attention_dispatcher::dispatch_gqa` hardcodes
  `has_hardware_matrix=false` → returned tier telemetry always says
  Tier2-on-GPU regardless of the kernel actually used. Cosmetic/telemetry.
- **M11** `lfm2.rs` carries ~38 `unwrap()`/`expect()` on Option fields in
  forward paths — panics-not-corruption on misconfigured loads; a load-time
  validation pass would be the clean fix.
- **M12** `kv_attention::append_and_get` divides by
  `k_new.shape().dims().get(1)` with a `.max(1)` fallback — a 1-D k_new
  would corrupt total_len. All current callers pass 2-D.
- **M13** minicpm embedding lookup silently zeroes out-of-vocabulary token
  ids instead of erroring.

## Chunked-prefill contract verification (positive result)

The engine's chunked prefill (F9 follow-on) requires every CausalLm forward
to (a) honor the positions tensor for RoPE, (b) append KV sequentially,
(c) advance_pos(seq_len). Verified across the crate: the Llama core +
all delegating wrappers (≈130 of 141 files delegate to `Llama::forward`)
satisfy all three; DeepSeek family, Gemma, GLM5.2, KimiK3, MiniMax-M3,
Qwen3.5(+MoE), MuseGlimmer, InternS2Mobius, InklingSmall, Gemma3n, Falcon,
GPT2 carry their own forwards and all three hold. Violators were exactly
M1/M3/M4/M5 (+Mamba, fixed or documented above). The RefKvCache/LlamaLayerCache
append-based caches are position-independent and therefore chunk-safe by
construction; causal bounds derive from cache length, RoPE from the tensor.

## Verification

grim-models-transformer/-mamba/-vision/-audio/-diffusion + grim-backend-cpu:
199 passed / 0 failed. grim-engine + grim-server regression: 220 passed /
0 failed. New gates: `mamba_forward_keeps_state_across_calls`,
`minicpm_forward_advances_session_position`,
`minicpm_paged_matches_classic_attention`,
`gather_paged_history_follows_block_table`,
`weights_look_broken_allows_constant_rank1_rejects_constant_matrices`,
plus the pre-existing 84 transformer-lib tests and block-level
paged-vs-classic parity.
