# WI-X2-PREFILL-ARENA — device-resident prefill KV arenas (PLAN-reduce-d2h-h2d A5)

Status: PHASE 2 IMPLEMENTED 2026-09-16 (branch `a5-prefill-arena`, gfx1201).
Phase 1 needed NO code: verified the default ROCm path already writes
RoPE(Q/K/V) into device arenas (`decode_attention_device`: `launch_kv_append`
D2D + causal `launch_qkv_attention_dev`, any `steps`), so the plan's "no
device-arena equivalent" premise was stale for the current tree.

## Why (measured problem)

Non-fused prefill routes RoPE(Q/K/V) through host `Vec<f32>` into
`fused_or_scalar_attention`/`_arena`: ~150–300MB D2H+H2D for a 100-token /
32-layer prefill (plan §A5). Decode already has device arenas
(`decode_attention_device`, `lfm2.rs:977`); prefill has no equivalent.

## What exists today

- `decode_graph_buffers.rs: seed_kv_arena_from_eager(dev, per_layer)` — D2D
  copy of eager per-layer K/V into the graph `k_arena`/`v_arena` + sets
  `current_pos` / device pos scalar. Zero callers (as of 2026-09-16).
- `Lfm2LayerCache::Attention { k_dev, v_dev, .. }` — device KV populated on
  the decode path (`lfm2.rs:1017-1041`) and the MXFP4 fused prefill path
  (`LFM2_FUSED_KV_CACHE_LEN` arena). The NON-fused prefill path leaves
  `k_dev`/`v_dev` empty (host `k`/`v` vecs only) — so there is currently
  nothing to seed FROM for the common prefill case. Seeding alone does not
  fix A5; the prefill device-arena write path must come first.

## Spec (two phases, in order)

### Phase 1 — prefill writes RoPE(Q/K/V) into device arenas

Mirror `decode_attention_device` for `steps > 1`:

1. Reuse `Lfm2LayerCache::Attention` — NO parallel cache type (plan gate).
   Prefill appends rows `[cache_offset .. cache_offset+steps)` into the
   existing `k_dev`/`v_dev` growth buffers via D2D (`copy_slice_range`), then
   runs attention over the device arena (existing `qkv_attention` kernel
   already takes `kv_len` + offset — no kernel change needed for the
   append-then-attend shape).
2. Gate: same unified `decode_graph_active(device)` + ROCm + dense layer.
   Non-ROCm /grammars unchanged (host path stays).
3. Correctness gate: byte-identical KV cache contents after prefill vs
   current eager prefill (compare `k`/`v` host mirrors element-wise), then
   token-for-token decode parity over ≥64 tokens on LFM2.5-350M-Q8_0
   (`models/` has it on-box).

### Phase 2 — seed the graph arena from the prefill arenas, then capture

1. New `Lfm2` accessor (model crate, Rockm-only, returns `Err(Unimplemented)`
   elsewhere):
   ```rust
   pub fn export_eager_kv_for_seed(&self, caches: &[Option<Lfm2LayerCache>])
       -> Result<Vec<Option<EagerKvSource<'_>>>>
   ```
   mapping each `Attention` layer with `k_dev`/`v_dev` present to
   `EagerKvSource { k_dev: <raw ptr>, v_dev: <raw ptr>, prefill_len,
   kv_stride, _anchor }`. Non-attention layers → `None`. Raw pointers valid
   only for the call duration (documented; caller copies synchronously).
2. Call sites (both, same order):
   - CLI `run.rs::try_graph_decode_step` — needs session-cache access it does
     not currently have: change signature to take the export vec (caller in
     the generation loop owns `session`; add a `Lfm2`-specific downcast
     helper to extract caches — session-internals plumbing is the bulk of
     this WI).
   - Engine `lib.rs:~2267` capture site — same, via engine sessions.
   Order inside capture setup: allocate graph → `seed_kv_arena_from_eager`
   → `begin_capture` → `forward_capture` → `end_capture`. Seed runs OUTSIDE
   the capture bracket (it does H2D pos write + D2D copies + `synchronize` —
   all capture-poison; the existing code already treats first-capture abort
   as expected, but seeded+synced setup must precede `begin_capture`).
3. `g.buffers.current_pos` init changes from hardcoded `1` to the seeded
   `prefill_len` (seed fn already sets it — remove the call-site overwrite or
   assert equality).

## Acceptance gates

1. Correctness: post-prefill KV byte-parity (Phase 1) + seeded-graph decode
   == eager decode token-for-token incl. prompt attention (the divergence the
   seed fn's docstring describes must NOT reproduce) → compile →
   architecture-cleanliness (no parallel cache type; no new env vars — reuse
   `decode_graph_active`) → perf (`TODO(gpu-verify)` on `syd-beasty`,
   prefill GB/s + TTFT before/after).
2. Fallback contract: any seed/capture failure → existing eager fallback +
   B3 warn line (no new silent paths).

## Non-goals

- Changing `repeat_penalty` / `mxfp4` defaults (see B1/A4 — explicitly NOT to do).
- Persisting anything across processes (see plan §5).

## Implementation notes (2026-09-16, branch `a5-prefill-arena`)

- `Lfm2::eager_kv_seed_sources` (`lfm2_graph.rs`): fail-closed export
  (dense layer w/o arenas + `valid_rows>0` → `Err`); `valid_rows` from caller
  loop counters, never from stale host mirrors. `Lfm2LayerCache` re-exported
  at crate root for CLI/engine use.
- CLI `try_graph_decode_step`: seed after alloc, before `begin_capture`;
  `current_pos = valid_rows` comes from the seed (removed hardcoded `= 1`).
- Engine capture site: same, `valid_rows` from `session.current_pos()`.
- `tests/lfm2_seed_parity.rs`: seeded arenas bit-exact vs eager arenas +
  fail-closed cases, green on gfx1201.
- E2E (LFM2.5-350M-Q8_0, greedy): seed succeeds on hybrid models, then
  capture still (correctly) falls back eager on ShortConv with the B3 line.
- A1 CORRECTION (same session): unifying the ShortConv gate to default-on
  regressed greedy decode (`"Hello! How can I assist you today"` →
  `"Hello!<|im_end|>"`); bisected to `shortconv_step_device`, RoPE-seed and
  attention gates proven innocent. ShortConv gate reverted to opt-in
  (`GRIM_DECODE_GRAPH=1`) with a code comment; re-unify only with the
  ShortConv device rework + parity cover.
- KNOWN GAP (pre-existing, not A5): device-attention eager path output
  differs from the stock host path on this model (`GRIM_DECODE_GRAPH=0`
  gives sensible output; device path degenerates). Seed is bit-exact vs its
  (device) source, so this blocks end-to-end graph-vs-eager parity validation
  until the device numerics gap is closed. Flagged, not fixed here.
