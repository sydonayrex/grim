# Session-continuity design (P2-1, PLAN-improve-grim-perf)

Status: DESIGN — implementation deferred per plan §P2-1 recommendation
("scope this as its own design document before implementation").

## Problem

Every `/api/generate` request mints a fresh `session_request_id`
(`grim-server/src/lib.rs:1564`, `REQUEST_ID_COUNTER.fetch_add`). The engine's
decode-graph gate (`grim-engine/src/lib.rs:2267`,
`decode_graphs.contains_key(&request_id)`) therefore always misses: each
request pays alloc + KV-arena seed + graph capture (~170 ms) before the first
replay. `finish_request` (`lib.rs:3193`) additionally removes the graph at
request end, so even a repeated id would not help today. Net effect: the
"amortized once per session" cost is paid on every HTTP call, and Grim loses
TTFT to Ollama (178 ms vs ~55 ms warm) despite a faster decode (2.57 vs
2.79 ms/tok).

## Goal

Warm-request TTFT in the tens-of-ms range by making decode-graph + KV-arena
lifetime per-model-instance rather than per-request, with optional
conversation-level KV-prefix reuse on top.

## Design

### Layer 1 — resource lifetime decoupling (required, closes the gap)

1. Introduce `GraphSlot` ownership in the engine: a per-model-instance pool of
   `FullDecodeGraph` + KV arenas (`decode_graphs` keyed by `model_id` instead
   of `request_id`; concurrency = N slots per model, LRU reused).
2. `grim_generate` acquires a slot at request start, releases (not destroys)
   at stream end. `finish_request` stops removing the graph; it releases the
   slot back to the pool.
3. KV arena contents between requests: for a NEW conversation the arena is
   logically stale — reset `current_pos`/`pos_dev` to 0 instead of re-seeding
   (no copy at all). Seeding from eager caches (`seed_kv_arena_from_eager`)
   is only needed when prefill ran on the eager path (A5 prefill-arena makes
   this unnecessary too: prefill writes the same device arenas).
4. Safety: stale-KV correctness is guaranteed by the position reset — a fresh
   conversation never reads another conversation's rows because attention is
   masked to `[0, pos)` and `pos` restarts at 0.

Cost: no new transport API. Server routing change + engine slot pool.

### Layer 2 — session identity (optional, prefix-cache win)

1. Transport: optional `session` field (or `x-grim-session` header) on
   `/api/generate` and `/v1/chat/completions`; server maps
   `hash(model, session)` → slot affinity + retained `current_pos`.
2. Multi-turn: skip prefill of the shared prompt prefix by validating cached
   logits against the new prompt tokens (llama.cpp `cache_prompt` semantics —
   longest-common-prefix check, truncate the rest).
3. Without a session token, behavior is identical to today (fresh conversation
   per request) — never guess reuse: reusing KV across unrelated requests is a
   correctness hazard, not an optimization.

### Sequencing

- Layer 1 is self-contained in `grim-server` request routing +
  `grim-engine` slot keying; no backend/kernel changes.
- Layer 2 builds on it and additionally needs a prefix-compare primitive in
  the engine (token-id compare against cached prompt ids per slot).

## Acceptance

- Warm repeat request on the same model: no `get_or_create_decode_graph`, no
  seed, no capture — TTFT < 50 ms on GPU 1 for the benchmark prompt.
- Interleaved different-prompt requests produce outputs identical to fresh
  sessions (position-reset correctness test).

## Implementation status (2026-09-16)

Layer 1 is implemented: `decode_graphs` is keyed by `model_id`
(`grim-engine/src/lib.rs`), the graph survives `finish_request`, and the
slot-hit path re-binds the arenas to the current request via the existing
`seed_kv_arena_from_eager` (fail-open to eager on seed error).

Measured: correctness verified end-to-end on GPU 1 (sequential requests
generate cleanly, no cross-request KV bleed, no faults). Caveat discovered:
the SERVER stepping loop itself (engine `drive_*` per-step round-trip +
sampler) costs ~0.4 s/token in server mode — it dominates everything and
masks the slot-pool TTFT win. Next work item: profile the server stepping
loop (likely sampler-thread handoff + per-step host sync), after which the
slot pool's TTFT benefit becomes measurable. The CLI one-shot path already
demonstrates the underlying decode speed (2.57 ms/tok, 389 tok/s).
