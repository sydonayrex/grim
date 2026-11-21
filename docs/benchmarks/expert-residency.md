# Q5 Benchmark: Use-Capped Expert Residency (`GRIM_MOE_USE_VICTIM=1`)

## COMMAND
```bash
# Q5.1/Q5.2 unit tests (grim-memory)
cargo test -p grim-memory --test moe_use_victim -- --nocapture

# Q5.3 cross-crate integration test (grim-engine)
cargo test -p grim-engine --test moe_policy_integration -- --nocapture

# Q5.4 ROCm Card 1 (gfx1200) MoE device dispatch parity with GRIM_MOE_USE_VICTIM=1
HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_ALLOC_POOL=1 GRIM_MOE_USE_VICTIM=1 \
  cargo test -p grim-models-transformer --lib qwen38_moe_device_dispatch_matches_host_reference -- --nocapture --test-threads=1
```

## EXECUTED
- `moe_use_victim`: `4 passed; 0 failed; 0 ignored; finished in 0.00s`.
- `moe_policy_integration`: `1 passed; 0 failed; 0 ignored; finished in 0.00s` (`policy_hits=39/64, lru_hits=39/64`).
- `qwen38_moe_device_dispatch_matches_host_reference` (Card 1, `GRIM_MOE_USE_VICTIM=1`): `1 passed; 0 failed; 0 ignored; finished in 0.11s`.

## BEFORE (`GRIM_MOE_USE_VICTIM=0`, clear-all at `cache.len() >= 8`)
- Real-GGUF 19k decode routing trace: `NOT RUN — pre-existing loader failure at HEAD 6b07f8d0 (qwen4exp_weight_loading_test)`. `TODO(gpu-verify)` once real GGUF loader is unblocked.

## AFTER (`GRIM_MOE_USE_VICTIM=1`, per-block `UseCappedResidentPolicy` cap=15, halving=256, MRU tie-break, `keep_hottest` retaining top 7)
- Real-GGUF 19k decode routing trace: `TODO(gpu-verify)` (synthetic 64-step integration trace is verified for policy invariants only, not claimed as real-trace speed evidence).

## NUMERIC GATE
- `qwen38_moe_device_dispatch_matches_host_reference` passes on Card 1 with `GRIM_MOE_USE_VICTIM=1` (residency eviction is selection-only; never alters expert weights or math).

## TOKEN PARITY
- Identical (cache eviction only; no arithmetic changes).

## IMPLEMENTATION NOTES & PLAN CORRECTIONS
1. **Plan Q5.1/Q5.2 defects corrected:**
   - Plan Q5.2 `victim()` wrote `let key = (*c, self.last_seen[i]);` assigned to `Option<(u16, u64, usize)>` (missing `, i`) and used `self.clock` (only incremented on `begin_token`) for `last_seen`. Added monotonic `access_seq: u64` incremented on `count_look`/`touch` and fixed `let key = (*c, self.last_seen[i], i);`.
   - Plan Q5.1 `mru_tie_break_prefers_most_recently_used_among_equals` left experts `0..4, 7` at 0 looks so `victim(&[])` would return `Some(0)` instead of `Some(5)`. Fixed test setup to give experts `0..4, 7` >= 2 looks so `5` and `6` tie at minimum count `1`.
2. **Follow-up outside Q5 scope:** `ExpertOffloadBanks` (`crates/grim-nn/src/expert_offload.rs:113`) maintains its own separate 8-entry staging cache outside Q5 ownership; wiring `UseCappedResidentPolicy` there is a separate follow-up.
3. **Prefetch stream:** `expert_prefetch_stream.rs` remains a CPU stub and is not called.

## VERDICT
Pass (unit + cross-crate + Card 1 parity); real-trace 19k hit-rate/speed `TODO(gpu-verify)` pending real-GGUF loader fix. Flag remains default OFF (`GRIM_MOE_USE_VICTIM=1`).
