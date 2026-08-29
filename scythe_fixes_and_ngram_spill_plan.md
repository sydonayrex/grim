# SCYTHE host-materialization fix + n-gram embedding spill support

## Purpose

Two issues remain open against `crates/grim-autograd/src/scythe.rs`, both confirmed against current source, not the stale ones already fixed (Sigma clamp floor is already `1e-4`; OASIS reconstruction bias is already resolved by computing gradients against raw `x`):

1. `fused_step_with_oasis` still fully materializes `out_grad`, `x`, `u`, `v`, and `sigma` via `to_vec_f32()` before the bounded 64-row tile loop runs. The tile loop itself is genuinely bounded; the surrounding data movement is not, and nothing tests for it.
2. Neither `LocalSpillManager`/`SharedSpillManager` (KV-cache spill, live-wired into `grim-scheduler`) nor `NvmeWeightStreamer` (unwired, `LAYER_ELEMS = 1024`-fixed weight cache) can hold an n-gram embedding table. The former is structurally locked to `(Vec<f32>, Vec<f32>)` key/value pairs; the latter is a fixed-size, disconnected type with no caller anywhere.

This plan treats these as three separable pieces of work, ordered by dependency and risk, per the project's stated gate order (correctness → compile → architecture-cleanliness → performance).

## Scope

In scope:

- Add a regression test that catches unnecessary host materialization in `fused_step_with_oasis` when the underlying tensors are already CPU-resident, and document (not eliminate) the host round-trip when tensors are GPU-resident, since eliminating that round-trip is GPU-kernel work explicitly out of scope here (matches the original `SCYTHE_FORGE_SCALE_PLAN.md`'s own scoping).
- Generalize `NvmeWeightStreamer` from its fixed `LAYER_ELEMS = 1024` assumption to accept arbitrary tensor sizes, and wire it into wherever the embedding table is loaded so it becomes a real, callable spill path rather than dead code.
- Add a new `EmbeddingSpillManager` type (or extend `NvmeWeightStreamer`) that supports a single large flat tensor rather than the KV-cache's k/v-pair shape, with `Gpu → HostRam → NvMe` tiering consistent with the existing `CacheTier` enum so `grim-scheduler` can reason about embedding placement the same way it reasons about KV blocks.

Out of scope:

- The ROCm/HIP fused kernel for SCYTHE's tile loop itself — this was already correctly deferred as a separate, later plan in the prior session, and nothing here changes that call.
- Changing `SharedSpillManager`'s existing KV-cache API shape — n-gram embedding spill gets its own type/methods rather than forcing a k/v pair onto data that isn't naturally key/value shaped.
- Actual n-gram model/tokenizer implementation — this plan only addresses whether an existing embedding table can be spilled under VRAM pressure, not building n-gram scoring itself.

## What exists today

- `crates/grim-autograd/src/scythe.rs` — `ScytheOptimizer::fused_step_with_oasis` calls `to_vec_f32()` on `out_grad`, `x`, `adapter.u`, `adapter.v`, `adapter.sigma` (lines confirmed via source read) before the tile loop. The tile loop itself allocates only `tile_rows * r`-sized buffers, matching the FORGE claim; test `test_scythe_never_allocates_full_gradient_tensor` mechanically enforces only the tile-buffer bound, not the surrounding full-tensor copies.
- `crates/grim-tensor/src/tensor.rs` — `Tensor::device()` and `Tensor::to_vec_f32()` exist; `BackendStorage::to_cpu_vec_f32`'s own doc comment states "production code paths should keep data on-device and avoid this when possible," confirming this is a known, documented anti-pattern the codebase already flags, not a new standard being invented here.
- `crates/grim-kvtransport/src/lib.rs`:
  - `LocalSpillManager` / `SharedSpillManager` — `demote_to_host(block_id: BlockId, k: Vec<f32>, v: Vec<f32>)`, `demote_to_nvme(block_id: BlockId)`, `retrieve(block_id: BlockId) -> Option<(Vec<f32>, Vec<f32>)>`. `BlockId = usize`. Wired live into `grim-scheduler::plan_hybrid_attention_step` (WI 3.4.1).
  - `NvmeWeightStreamer` — `prefetch_layer_async(layer_id: usize)`, `const LAYER_ELEMS: usize = 1024` hardcoded inside the function body. No callers outside `grim-kvtransport` itself (confirmed via `grep -rln`). Recently fixed from a mock-data bug (`vec![0.5f32; 1024]`) to real `pread`-based disk reads.
  - `CacheTier` enum: `Gpu`, `HostRam`, `NvMe`, `NvMeWeightStream` — the last variant already exists for exactly this use case but nothing populates it today.

## Issue 1 — bound the host-materialization test, document the GPU-resident gap

Duration: half a day.

The tile loop's memory bound is real and already tested. What's missing is a test that catches regressions in the *surrounding* copies, and an honest doc comment about what this fix does and doesn't buy.

Modified files:

- `crates/grim-autograd/src/scythe.rs`:
  - Add a doc comment above `fused_step_with_oasis` stating plainly: "FORGE bounds the U/V gradient tile buffer to `tile_rows * r` elements. It does not currently avoid the host round-trip for `out_grad`/`x`/`u`/`v`/`sigma` — those are copied to `Vec<f32>` via `to_vec_f32()` regardless of the tensor's underlying device. On a CPU-resident tensor this copy is a no-op in spirit but still an extra allocation; on a GPU-resident tensor this is a real PCIe transfer per call, and eliminating it requires a device-resident fused kernel (out of scope for this crate — see the ROCm/HIP follow-up plan)." This directly addresses the gap between the FORGE claim and what's actually eliminated, without overclaiming a fix that isn't being made yet.
  - Add a new test, `test_fused_step_no_redundant_cpu_copy_when_already_cpu_resident`: construct a `ScytheAdapter`/`ScytheOptimizer` with CPU-backed tensors, call `fused_step`, and assert (via an allocation-counting harness or a `#[cfg(test)]` copy-counter wrapped around `to_vec_f32`) that the number of full-tensor copies per call is bounded to a known constant (currently 5: `out_grad`, `x`, `u`, `v`, `sigma`) rather than growing — this doesn't reduce the copies yet, but it locks the current count so a future change that silently adds more copies (e.g. a careless refactor re-fetching `u_slice` twice) gets caught immediately.
- `crates/grim-autograd/tests/` — add `scythe_host_materialization.rs` integration test asserting the same copy count end-to-end through `ScytheOptimizer::step_param` and `fused_step`, matching the project's existing pattern of separate `tests/` files per subsystem (mirrors `scythe1_integration.rs`).

Test criteria:

- New test explicitly documents "5 copies" as the current baseline with a comment explaining each one's purpose, so any PR that adds a 6th copy without justification fails CI, and any PR that removes one down to 4 (a genuine improvement) requires updating the constant with a one-line explanation — this makes future host-copy creep visible instead of silent.
- No production code path changes in this phase — this phase is test coverage and honest documentation only, since actually reducing the copy count on CPU tensors (e.g. mutating `u`/`v`/`sigma` in place rather than round-tripping through `Vec<f32>`) is a genuine follow-on optimization but touches `grim-tensor`'s in-place-mutation API surface, which is riskier and belongs in its own reviewed change, not folded into this fix.

## Issue 2 — generalize `NvmeWeightStreamer`, wire it into embedding loading

Duration: 2 days.

`NvmeWeightStreamer` is the closer of the two existing types to what an n-gram embedding table needs (flat tensor, not k/v pair), but it's hardcoded to 1024-float units and has zero callers. Rather than inventing a new type from scratch, generalize this one and give it a real caller.

Modified files:

- `crates/grim-kvtransport/src/lib.rs`:
  - Replace `const LAYER_ELEMS: usize = 1024` with a field on `NvmeWeightStreamer`: `pub unit_elems: usize`, set in `NvmeWeightStreamer::new(weights_path: PathBuf, lru_capacity_layers: usize, unit_elems: usize)`. This is a breaking signature change to `new()`; since there are no external callers today (confirmed), this is a safe, zero-blast-radius change.
  - Rename the "layer" framing to something format-neutral where it doesn't change behavior — `prefetch_layer_async(unit_id: usize)` keeps its name if callers conceptually still think in layer-like units (an embedding table sharded into row-blocks fits this framing fine: `unit_id` becomes "which row-block", `unit_elems` becomes "how many floats per row-block").
  - Add `CacheTier::NvMeWeightStream` population: today the enum variant exists but nothing sets it. When `NvmeWeightStreamer` evicts a unit from its host RAM LRU cache to disk, record `CacheTier::NvMeWeightStream` in a shared tier-tracking map (reuse `LocalSpillManager`'s `block_tiers: HashMap<BlockId, CacheTier>` pattern, or add an equivalent map local to `NvmeWeightStreamer`) so `grim-scheduler` can query embedding-table placement the same way it queries KV-block placement via `get_tier`.

New files:

- `crates/grim-nn/src/embedding_spill.rs` (or nearest existing home for the embedding table type — confirm exact module by reading `crates/grim-nn/src/` structure before implementation, since this plan doesn't yet know the exact current home of the embedding lookup table): wraps an `NvmeWeightStreamer` instance around the model's embedding weight tensor, sharded into `unit_elems`-sized row-blocks (e.g., one "unit" per N vocabulary rows, sized so a unit is a convenient granularity for LRU eviction — a reasonable starting point is one unit per 4096 vocab rows at the model's `hidden_size`, tuned once real numbers are available rather than guessed here). Exposes `lookup(token_id: u32) -> Result<Vec<f32>>` that resolves which unit a token's row lives in, prefetches if not cached, and returns the row.

Test criteria:

- Unit test: construct an `NvmeWeightStreamer` with `unit_elems` set to a small embedding-table-realistic size (e.g. 4096 rows × 128 dims = 524,288 floats per unit, or a smaller synthetic size for fast test runs), write known synthetic embedding data to the scratch file, and assert `prefetch_layer_async` + retrieval round-trips the exact values — this is the same pattern the fixed `prefetch_layer_async` doc comment already describes ("surface an explicit `KvCache` error rather than substituting mock data"), extended to the new configurable size.
- Integration test: exercise eviction under a small `lru_capacity_layers`, confirm a unit evicted to NVMe and re-requested returns identical data (round-trip correctness, mirroring `LocalSpillManager`'s existing `demote_to_nvme`/`retrieve` test pattern already present in `grim-scheduler`'s `hybrid_tests` module).
- Confirm `grim-scheduler`'s `get_tier`-style query works against the new tier-tracking map with a test analogous to `plan_hybrid_attention_step`'s existing coverage, so embedding placement is inspectable the same way KV-block placement already is.

## Issue 3 — wire the generalized streamer into a live embedding-lookup path

Duration: 1 day, depends on Issue 2.

Modified files:

- Whatever module currently owns the token embedding lookup (needs to be located precisely before implementation — likely `crates/grim-models/transformer/src/` given where `lora.rs`/`eagle3.rs` live, but confirm the exact embedding-layer file by reading that directory rather than assuming) gets an optional code path: when the embedding table exceeds a configurable VRAM budget, construct the `embedding_spill.rs` wrapper from Issue 2 instead of holding the full table resident, and route lookups through it.
- This should be feature-gated or config-gated (e.g. only activates when `embedding_table_bytes > some_threshold` or an explicit CLI/config flag is set), since forcing every model through a disk-backed lookup path by default would regress latency for models whose embedding table already fits comfortably in VRAM.

Test criteria:

- Integration test: a small synthetic vocab/hidden-size embedding table forced under the spill threshold, confirm end-to-end forward-pass output matches the same table held fully resident (bit-identical, since this is just a storage-location change, not a numerical one).
- Confirm the spill path is inert (zero behavior change, zero performance cost) when the config/feature flag is off, matching the project's convention of additive, opt-in changes over default-path modifications.

## Dependency ordering

Issue 1 is fully independent and can land first — it's test-and-documentation-only, no dependency on the other two.

Issue 2 depends on nothing but touches a currently-dead-code type; low risk since there are no existing callers to break.

Issue 3 depends on Issue 2 (needs the generalized streamer to exist first) and requires locating the actual embedding-lookup module before scoping precisely — flagged above as needing confirmation rather than guessed.

## Ponytail-review checklist

- [ ] `fused_step_with_oasis`'s new doc comment lists exactly which 5 tensors are copied, not a vague "some data movement."
- [ ] `NvmeWeightStreamer::new`'s new `unit_elems` parameter type and default (if any) stated explicitly.
- [ ] Confirm zero external callers of `NvmeWeightStreamer::new` before changing its signature (already confirmed via `grep -rln` in this plan's research — worth re-confirming at implementation time in case something landed since).
- [ ] The exact file path for "current embedding lookup module" is filled in with a real path before Issue 3 work starts, not left as this plan's placeholder guess.
- [ ] Every new test asserts literal expected values (round-tripped bytes, exact tier enum values) — not "loss decreases" or other soft checks, per the project's standing test-quality bar.

## Notes

- This plan does not touch `grim-backend-rocm`. All three issues are addressed at the `grim-autograd`/`grim-kvtransport`/`grim-nn` (host/CPU) level, consistent with the decision from the prior planning session to keep SCYTHE as a CPU reference implementation until its correctness story is solid, rather than jumping to a HIP kernel.
- `NvmeWeightStreamer`'s `unit_elems` generalization is a minimal, additive change specifically because it currently has zero callers — if that changes before this plan is implemented, re-check for new dependents and adjust the "breaking signature change is safe" assumption in Issue 2 accordingly.
