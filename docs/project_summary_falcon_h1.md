# Project Summary: Falcon-H1 (NeVe-Cascade-S-90M) port to grim

## Objective
Implement falcon-h1 end-to-end in grim and verify it loads + runs against `./models/Neve-Cascade-S-90M-Q8_0.gguf`.

## Results
- **Falcon-H1 model implemented** in `crates/grim-models/transformer/src/falcon_h1.rs`:
  `FalconH1Config` + `FalconH1Block` (conv-state/scan cache) + `FalconH1Model: Model + CausalLm`, with a faithful Mamba-2 selective-scan forward on CPU (causal 1-D conv over the `z|xBC|dt` projection, per-head `A`/`D`/`dt` scalars, SwiGLU FFN, GQA attention, NEOX RoPE, tied LM head) — modeled on `lfm2.rs`.
- **Loader wired**: `ModelArchitecture::FalconH1` GGUF dispatch arm added in `crates/grim-engine/src/model_loader.rs` (builds `FalconH1Config` from GGUF hparams and calls `FalconH1Model::load_tp`).
- **Re-exports**: `mod falcon_h1;` + pub re-exports in `crates/grim-models/transformer/src/lib.rs`.
- **Verified end-to-end**:
  - `cargo check -p grim-models-transformer -p grim-engine -p grim-cli` — 0 errors, no warnings from new code.
  - `.GGUF` model loads: `arch=falcon-h1, layers=24, hidden=512, vocab=32768`.
  - `grim run --max-tokens 4 ./models/Neve-Cascade-S-90M-Q8_0.gguf "Q: What is 2+2?\nA:"` exits 0, generates 4 finite tokens, prints `[grim] Done. Generated 4 tokens.` (finite logits + finite sampled tokens confirmed).

### Issues found & fixed during verification
1. **Tensor name mismatch**: GGUF uses `attn_q/k/v/output`, `ffn_gate/up/down`, bare `ffn_norm` (no `.weight`). `load_block` initially used wrong Linear sub-prefixes (`wq`/`wk`/`w_up`...); corrected to `attn_q`/`attn_k`/`attn_v`/`attn_output`/`ffn_gate`/`ffn_up`/`ffn_down` and `ws.get(...,"ffn_norm")`.
2. **`ssm_in_dim` formula** produced 920 vs actual 1688. Correct: `ssm_in_dim = ssm_d_inner + ssm_conv_dim + ssm_dt_rank` (768 + 896 + 24 = 1688), where `ssm_conv_dim = ssm_d_inner + 2*n_group*d_state` (768 + 128 = 896).
3. **`FalconH1LayerCache.conv_state` width** was `(d_conv-1)*ssm_d_inner` (2304) but conv buffer rows are `ssm_conv_dim` wide (896) → misaligned indexing (index-out-of-bounds at buf.len 9472). Fixed to `(d_conv-1)*ssm_conv_dim`.
4. **Session bootstrap**: `run`/`bench` construct sessions via `Inner::new`/`SessionInner` (no caches), so `forward`'s downcast panicked. Made `forward` lazily initialize `Vec<FalconH1LayerCache>` via `set_model_state` when `model_state` is `None`.

### Caveat (not a correctness bug)
`run.rs` has a TEMP-DIAG full-sequence-recompute loop (feeds all `tokens` each step, O(N²)). On CPU this is slow for long contexts; `--max-tokens 4` completes fast, longer runs hit the 90s timeout on the bench harness (a `run.rs` perf issue, not Falcon-H1 correctness).

### Pre-existing (unrelated) breakage
`crates/grim-garage/src/backend.rs` (and `grim-backend-rocm`) is broken on the current working tree via `RocmDevice::shared(0)` (no such method) — `cargo check --workspace` fails there. Confirmed pre-existing: `cargo check --workspace` builds clean on a clean tree (`git stash`); fails only with the pre-existing `backend.rs`/`roc_device.rs` modifications. **Not caused by or related to the Falcon-H1 work** (those crates are untouched by it).

## Status: Complete
All success criteria met: loads, finite logits, finite first token. Pre-existing grim-garage ROCm break excluded from scope and verified independent.

## Relevant Files
- NEW: `crates/grim-models/transformer/src/falcon_h1.rs`
- EDIT: `crates/grim-models/transformer/src/lib.rs` (mod + re-exports)
- EDIT: `crates/grim-engine/src/model_loader.rs` (import + GGUF dispatch arm)
- EDIT: `crates/grim-cli/src/bench.rs` (use `model.new_session()`)
- Reference dump: `crates/grim-format/examples/inspect_falcon.rs` (full GGUF meta + blk.0 tensor shapes)
