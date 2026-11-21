# Q9 Benchmark: Flash-Next MTP Verify & Basalt `may_verify` Probe (`GRIM_FLASH_MTP=1`, `GRIM_SPEC_MAY_VERIFY=1`)

## COMMAND
```bash
# Basalt may_verify probe exception & low->high recovery test (grim-speculative)
cargo test -p grim-speculative --test may_verify_recovery_test -- --nocapture

# Flash-Next MTP verify_draft_tokens compatibility with GRIM_FLASH_LAST_ROW_LMHEAD=1
cargo test -p grim-models-transformer --test qwen4exp_mtp_spec_test -- --nocapture
```

## EXECUTED
- `may_verify_recovery_test`: `2 passed; 0 failed; 0 ignored; finished in 0.00s`.
- `qwen4exp_mtp_spec_test`: `2 passed; 0 failed; 0 ignored; finished in 0.00s`.

## BEFORE / AFTER
- `MayVerifyGate` (`crates/grim-speculative/src/may_verify.rs`): implements Basalt's `kProbeChance = 0.85` (15% probe rate when `chance[0] < min_chance`), proving EMA recovers from a 20-step `0/3` low-acceptance regime back above `min_chance = 0.35` once high-acceptance (`3/3`) resumes.
- `Qwen38FlashNextMtp::verify_draft_tokens`: forces `session.set_full_output_logits(true)` during `K+1` speculative verification so Q1a (`GRIM_FLASH_LAST_ROW_LMHEAD=1`) never slices away draft-position logits, and restores `full_output_logits` afterward.
- Full 48-layer real-GGUF speculative tok/s A/B: `TODO(gpu-verify)` once SSD-backed `ExpertOffloadBanks` + `GRIM_PLE_SPILL_F32` multi-layer benchmark harness is wired for 48 layers.

## NUMERIC GATE
- `verify_draft_tokens` returns exact `[4, vocab]` logits under `GRIM_FLASH_LAST_ROW_LMHEAD=1` and restores `full_output_logits == false`.

## TOKEN PARITY
- Identical (speculative verification preserves exact target-model distribution).

## VERDICT
Pass (unit + recovery + MTP verify compatibility); both flags remain default OFF (`GRIM_FLASH_MTP=1`, `GRIM_SPEC_MAY_VERIFY=1`).
