# grim-models + grim-plugin audit verification & remediation record

Scope: crates/grim-models (transformer, mamba, diffusion, audio, vision
subcrates) and crates/grim-plugin, as of this verification.

## Claim-by-claim verification — grim-models

| # | Claim | Verdict | Action |
|---|---|---|---|
| 1 | DeltaNet context-free across calls (local `states` vec) | **CONFIRMED** | **FIXED** — delta states now persist in `session.model_state` (`Vec<Option<Vec<f32>>>`, resized to layer count), mirroring the Mamba session-state fix. Gated by `deltanet_session_state_makes_decode_context_aware`: sequential single-token decode through one session must equal a batched 2-token prefill's last-token logits (would fail on the pre-fix code). |
| 2 | DeltaNet uses `Session` whose lack of `model_state` prevents persistence | **INCORRECT** | `Session::new` deliberately returns `Inner`, which implements `model_state`/`set_model_state`. No architectural gap; no action. |
| 3 | Mamba b_param inconsistent: GPU rejects empty, CPU silently zero-fills | **CONFIRMED** | **FIXED** — `step_block_cpu` now returns `Err(Unimplemented)` on empty `b_param`, matching the GPU MOD-1 refusal (a zero-B matrix silently degenerates the SSM). |
| 4 | SolarOpen2 `expect` on session model_state type (panic vs Llama's error) | **CONFIRMED** | **FIXED** — replaced with `ok_or_else(\|\| Error::Session(...))?`. |
| 5 | SolarOpen2Block unwraps construction-invariant Options | CONFIRMED, acceptable | Same class as Llama's load-guarded unwraps (audit's own assessment). Documented here rather than adding a validate() pass. |
| 6 | Whisper `from_hf` silently defaults partial configs to whisper-tiny shape | **CONFIRMED** (design choice) | **MITIGATED** — `from_hf` now prints a loud warning listing which fields fell back (mirrors the sage_attention loud-fallback precedent). Behavior (partial configs still load) unchanged. |
| 7 | DeltaNet silently skips OOB tokens (zero hidden row, no signal) | **CONFIRMED** | **FIXED** — out-of-vocab token ids now return `Err(Session)` naming the id, position, and vocab size. Gated by `deltanet_rejects_out_of_vocab_tokens`. |
| 8 | flux2 `denoise_step` silently uses t=0.0 for malformed timestep tensors | **CONFIRMED** | **FIXED** — empty timestep tensor is now `Err(Shape)`. |

## Claim-by-claim verification — grim-plugin

| # | Claim | Verdict | Action |
|---|---|---|---|
| 1 | `validate_abi` never called from the dylib load path | **CONFIRMED** | **FIXED** — `load_with_manifest` now runs `validate_abi(manifest, ENGINE_ABI_VERSION)` BEFORE the library is opened (a foreign vtable layout is UB, so the gate must precede the load). New const `ENGINE_ABI_VERSION = 1`. Gated by `dylib_load_path_enforces_abi_version`. |
| 2 | WASM sampler double-copies logits | CONFIRMED, by design | Inherent to the sandbox boundary; documented in the loader. No action. |
| 3 | Dylib (element count) vs WASM (byte count) sampler unit mismatch | CONFIRMED, documented | Both call sites document the unit. A cross-backend runtime validation would need an ABI change; recorded as a roadmap item. |
| 4 | dylib name extraction substitutes "invalid-name" for non-UTF-8 | CONFIRMED | Graceful degradation by design; the registered name is distinct and inspectable. No action. |
| 5 | `scan_plugin_directory` uses `println!` | **CONFIRMED** | **FIXED** — replaced with `tracing::info!`. |
| 6 | `register_manifest` requires BOTH stage and priority; a processor with only one is invisible to `processor_chain` | **CONFIRMED** | **FIXED** — a processor declaring either field now enters the chain with the missing field defaulted (stage `"default"`, priority `0`); duplicate (stage, priority) detection runs on the effective values. Gated by `partial_processor_manifests_still_enter_chain` (includes a defaulted-collision rejection case). |

**Discovered while verifying (not in the audit):** the `dylib-loading`
feature did not compile — `_path` vs `path`, a missing
`Self::compute_sha256_file` qualification, and test bindings
(`_manifest`) that the feature-gated tests read. Fixed so the feature
builds and its tests run (`cargo test -p grim-plugin
--features dylib-loading`: 26 passed).

**NEW BUG FOUND by the new RWKV numeric reference test:** `RwkvBlock::step`'s
`flat` closure documented its argument as a ROW of the `[3, dim]` mixed
tensor but sliced by ELEMENT offset — the value projection read elements
1..dim+1 and the receptance projection 2..dim+2, blending channels across
the k/v/r boundaries. Every RWKV forward produced silently wrong attention
output (shapes/boundedness tests could not see it). **FIXED** to slice by
row (`row * dim..(row+1)*dim`) and pinned by
`rwkv_wkv_two_steps_match_f64_reference`.

Also found and fixed while wiring the reference tests:
`Mamba2Block::step_block` never advanced `state.pos` (MambaState documents
pos as the per-step token cursor and Mamba-1 advances it) — speculative
snapshots would read a stale position.

**Latent bug recorded, not fixed (backend scope):** `RwkvBlock::step`
accepts a 1-D input tensor by its length check, but the residual
`add_tensors` then panics in the CPU broadcast path (rank-1 vs rank-2).
All real callers pass 2-D; the rank-1 add broadcast in
grim-backend-cpu::device should be hardened separately.

## Numeric reference tests added (the audit's "not rigorously tested" list)

| Pathway | Test | Method |
|---|---|---|
| Mamba-1 selective scan | `mamba1_scan_two_steps_match_f64_reference` | two steps vs independent f64 recomputation of h' = A·h + B·x, out = Σh' + z·D, through norm→in_proj→scan→out_proj |
| Mamba-2 SSD recurrence | `mamba2_ssd_two_steps_match_f64_reference` | f64 reference of decay = −exp(A_log)·softplus(dt), SiLU z-gate, D-skip, group-shared B/C |
| RWKV WKV + channel mix | `rwkv_wkv_two_steps_match_f64_reference` | two tokens vs f64 recomputation of token-shift, WKV numerator/denominator/max triples, sigmoid gates |
| DeltaNet delta rule | `delta_rule_matches_f64_reference` | S' = S + β(v − S·k)kᵀ, o = q·S'ᵀ vs f64, state carried across tokens |
| DeltaNet cross-call state | `deltanet_session_state_makes_decode_context_aware` | sequential decode ≡ batched prefill last token |
| LFM2 shortconv | `shortconv_causal_conv_matches_f64_reference` | full block (norm → in_proj → causal conv with ring state → out_proj → residual → SwiGLU FFN) vs f64 |
| ALiBi slopes | `alibi_slopes_match_press_reference` | powers of two vs the Press et al. closed form (incl. the n=8 worked example); non-powers pinned to the documented interleave convention (noted: differs from the official repo's append-style interleave — convention, not bug) |
| RoPE | already covered | grim-tensor `tests/golden_rope.rs` pins the kernel against hand-computed values; `apply_rope_multi_head` is a shape-relabel wrapper around it |

## Missing features — verified but deliberately NOT built here

The audit's missing-feature list (tokenizer runtime, processor/tokenizer
plugin runtimes, capability-based dispatch, plugin unload API, tied
embeddings wiring, max_seq_len forward guard, mel/image frontends) is
confirmed accurate, but these are feature developments, not bug fixes —
each is a subsystem in its own right and none is stubbed in. They remain
the documented next steps. The one enforcement-adjacent gap (dylib ABI at
load) was fixed as above.

## Verification

- `cargo check --workspace --tests` clean.
- grim-models (5 subcrates) + grim-plugin default features: **188 passed,
  0 failed**.
- grim-plugin with `dylib-loading`: **26 passed, 0 failed** (feature now
  compiles for the first time in its current form).
- New tests: 3 reference suites in mamba (scan, SSD, WKV), 4 gates in
  transformer (delta rule, session state, OOB rejection, ALiBi), 1 in
  LFM2 (shortconv), 2 in grim-plugin (ABI-at-load, chain registration).
