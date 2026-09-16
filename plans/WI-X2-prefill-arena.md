# WI-X2-PREFILL-ARENA — device-resident prefill KV arenas (PLAN-reduce-d2h-h2d A5)

Status: SCOPED. Seed primitive exists (`DecodeGraphBuffers::seed_kv_arena_from_eager`,
now compiling + re-exported as `grim_backend_rocm::EagerKvSource`); end-to-end
wiring is genuinely new control flow, not a gating fix — tracked here.

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
