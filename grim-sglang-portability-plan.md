# Implementation Plan: Portable SGLang Concepts for Grim

## Scope

This plan covers items verified against Grim's actual source (not architecture docs or
memory) as of this investigation. Each phase references the specific code checked.

**Confirmed real gaps, in dependency order:**
1. RadixAttention (trie prefix cache) — `KvBlockPool::find_or_share_prefix` in
   `grim-memory/src/lib.rs` is a flat `HashMap<u64, BlockId>`, exact-match only.
2. Wiring the existing tiering primitives into the live serving path —
   `KvBlockPool::free_with_tier` / `promote_to_gpu` are real, working spill-to-host/NVMe
   mechanics, but are called from nowhere except `grim-memory`'s own unit tests.
3. Scheduler/GPU overlap — `Engine::tick()` in `grim-engine/src/lib.rs` calls
   `scheduler.schedule()` then runs prefill/decode strictly sequentially, no
   double-buffering anywhere in the loop.
4. HiCache-style layer-pipelined prefetch — genuinely new work, depends on #1 and #2.

**Explicitly excluded:** cache-aware load balancing (SGL-Router equivalent). No
multi-instance concept exists in `grim-server` (single `Mutex<Engine>` behind one Axum
router) or `grim-garage` (explicitly single-node by design per its own README). Nothing
to route on until #1 exists, and nothing to route *to* under the current architecture.
Not included in this plan.

---

## Phase 1 — Radix-tree prefix cache

**Replaces:** `KvBlockPool`'s flat `prefix_cache: HashMap<u64, BlockId>`
(`grim-memory/src/lib.rs:85`)

**Goal:** partial/branching prefix sharing across requests, not just exact
whole-prefix matches.

| Step | Work | Notes |
|---|---|---|
| 1.1 | Add `RadixNode` struct: `{ token_span: Range, block_id: BlockId, children: HashMap<TokenKey, NodeId>, ref_count: u32, last_access: Instant }` | Keyed at block granularity (one node per `BlockId`, matching existing block size), not per-token — avoids rewriting the block allocator. |
| 1.2 | Add `RadixTree` wrapping a `SlotMap<NodeId, RadixNode>` or `Vec<RadixNode>` + root | Own module inside `grim-memory`, not a new crate — tightly coupled to `KvBlockPool`'s `BlockId`/ref-count model. |
| 1.3 | `RadixTree::match_prefix(tokens: &[u32]) -> (Vec<BlockId>, usize)` — walk from root, return matched blocks + count of matched tokens | The lookup the scheduler calls before deciding how much of a request needs prefill. |
| 1.4 | `RadixTree::insert(tokens: &[u32], blocks: &[BlockId])` — extend tree with newly computed blocks after a prefill completes | Splits an existing node if the new sequence diverges mid-block — the one genuinely new piece of logic; everything else is bookkeeping. |
| 1.5 | Wire `KvBlockPool::find_or_share_prefix` callers to `RadixTree::match_prefix` instead | `find_or_share_prefix` deprecated/removed once callers migrate. |
| 1.6 | Eviction: replace ad hoc block LRU with trie-leaf LRU (evict childless nodes with lowest `last_access` first, walking up as parents become childless) | Needed so eviction doesn't reclaim a block another request's partial prefix still depends on. |
| 1.7 | Tests: exact-match parity with old behavior (regression), plus new partial-prefix cases (two requests sharing first N tokens, diverging after) | Parity test proves this isn't a silent behavior change for existing workloads. |

**Effort:** medium. Self-contained within `grim-memory`; no changes to
`grim-engine`/`grim-scheduler` call signatures beyond swapping the lookup call.

**Risk:** the node-splitting logic in 1.4 is the only intricate part — same class of
bug as any trie implementation (off-by-one on split boundaries). Budget real test time
here, not just the happy path.

---

## Phase 2 — Wire tiering primitives into the live serving path

**Uses:** `KvBlockPool::free_with_tier` / `promote_to_gpu` (`grim-memory`), currently
exercised only in unit tests.

**Goal:** memory pressure actually triggers demotion; a request whose prefix was
demoted actually gets promoted back, instead of recomputing from scratch.

| Step | Work | Notes |
|---|---|---|
| 2.1 | Add a pressure check in `Engine::tick()` (or `Scheduler::schedule()`, wherever `pressure_active` is already computed) that calls `free_with_tier(id, force_tier=true)` on the coldest trie leaves when GPU block pool utilization crosses a threshold | `pressure_active` already exists in `grim-scheduler/src/lib.rs:240` — hooks into an existing signal, not a new one. |
| 2.2 | On `RadixTree::match_prefix`, if a matched block is host/NVMe-resident, call `promote_to_gpu` before handing blocks to the request | This is the actual "cache hit but demoted" path SGLang exploits; without it, tiering only helps memory footprint, not latency. |
| 2.3 | Track block location (`Gpu`/`Host`/`Nvme`) explicitly rather than inferring it, since `promote_to_gpu` currently just asks the spill manager and gets `None` if nothing was demoted | Small addition: a location field on `KvBlock` or a parallel side table. |
| 2.4 | Tests: force a demotion under simulated pressure, issue a request that hits the demoted prefix, assert promotion happens and output is correct (not recomputed) | The test that actually proves the primitive is live, not just present. |

**Effort:** small-to-medium. Depends on Phase 1 only for the "which blocks are cold"
signal (trie leaf LRU) — could technically be done against the old flat cache, but
doing it after Phase 1 avoids rework.

**Risk:** low — the hard part (spill/demote/retrieve mechanics) is already built and
tested; this phase is plumbing plus a location-tracking side table.

---

## Phase 3 — Scheduler/GPU overlap

**Replaces:** `Engine::tick()`'s strictly sequential
`schedule() → drive_prefill → drive_decode` (`grim-engine/src/lib.rs:543` onward)

**Goal:** CPU-side scheduling for batch N+1 happens while GPU executes batch N.

| Step | Work | Notes |
|---|---|---|
| 3.1 | Double-buffer `SchedulerOutput`: compute output for the *next* tick while the *current* tick's GPU work is in flight | Requires `Scheduler::schedule()` to not depend on state that only exists after the current batch's GPU work completes — needs a careful audit of what `schedule()` reads. |
| 3.2 | Move GPU dispatch (prefill/decode drive calls) onto a background thread or async task; hand off pre-computed batch N+1 the instant batch N's dispatch is issued | The actual overlap — CPU prepares while GPU runs, not two full ticks pipelined. |
| 3.3 | Audit `AdmissionController::admit` / `predict_ttft` for any hidden dependency on synchronous ordering (e.g., relies on `observe_prefill` having been called for the just-finished batch before scheduling the next) | The real risk in this phase — SGLang's overlap works because their scheduler needs the previous batch's *shape*, not its *results*, to plan the next one. This property needs to be verified for Grim's admission logic, not assumed. |
| 3.4 | Tests: verify `DeterminismMode::Strict` (already in `schedule()`) still produces identical output under overlap | `Scheduler` already has a strict-determinism path for reproducibility; overlap must not break it. |

**Effort:** medium-to-large — the most invasive phase, touching the engine's core loop
rather than adding alongside it.

**Risk:** highest of the three. Real chance of subtle correctness bugs (races, stale
batch-shape assumptions) if 3.3 isn't done carefully. Recommend doing this last, after
Phases 1–2 are stable, so concurrency issues and cache correctness aren't being
debugged simultaneously.

---

## Phase 4 (stretch, not scoped in detail) — Layer-pipelined KV prefetch

Depends on Phase 1 (trie tells you what to prefetch) and Phase 2 (promotion mechanism
to prefetch with). Genuinely new work — prefetching layer N+1's KV while layer N
computes requires async I/O interleaved with the per-layer forward pass in
`grim-models`, which has not been audited at that granularity. Treat as a follow-up
investigation once Phases 1–3 land, not something to size now.

---

## Suggested sequencing

**1 → 2 → 3**, each independently shippable and testable.

- Phases 1 and 2 are additive: new code paths, old behavior preserved as fallback
  until cutover.
- Phase 3 is the only phase that changes existing control flow under load — doing it
  last means any regression is easier to attribute to a single cause.
