# grim-disagg Audit Report

Scope: /D/rex/projects/grim/crates/grim-disagg (lib.rs, disagg_integration.rs)
plus the handoff pathways it drives in grim-kvtransport (wire client +
receiver) and grim-memory (KvBlockPool). Adjacent grim-engine call sites
noted where they change the verdict.

Classes: L = logic fault (silent wrong result), B = latent bug (hits
only under specific conditions), G = gap (missing capability / untested
pathway), P = perf/correctness footgun.

Test suite status at audit time: 19/19 pass (12 unit + 7 integration).
Every finding below survives the current suite.

---

## L1. EVERY handoff path inflates destination num_tokens to BLOCK_SIZE — EMPIRICALLY PROVEN
   src/lib.rs:432-433 (p2p), grim-kvtransport/src/lib.rs:1077-1081 +
   grim-memory/src/lib.rs:678-698 (wire), grim-engine/src/lib.rs ~1703 (pull)

   The V2 wire header carries no valid-token-count field.grim-disagg sends
   the FULL block buffers (key_data/value_data are always BLOCK_SIZE ×
   elem_per_token elements, zero-padded past the valid prefix), and the
   receiver derives `num_tokens = num_elems / elem_per_token` = BLOCK_SIZE
   for every transfer. The P2P path is worse in a different way:
   `transfer_kv_p2p_direct` passes `k_data.len()` (an ELEMENT count, 128
   for the standard geometry) where `write_layer_keys` expects a TOKEN
   count; `num_tokens.min(BLOCK_SIZE)` = 16 again.

   Throwaway probe (run then deleted) confirmed:
     wire push: source num_tokens=5 → destination num_tokens=16
     p2p:       source num_tokens=7 → destination num_tokens=16

   Impact: `BlockTable::num_tokens()` (grim-memory:855-861) sums per-block
   `num_tokens` as sequence length — scheduler/admission/eviction math
   reads counts inflated up to 16× per block. Today's decode attention
   masks by position so single-request colocated output looks fine, which
   is why no test catches it: no test asserts num_tokens fidelity across
   ANY handoff.

   Fix direction: carry the source block's valid token count in the wire
   header (spare header field or the prompt control channel), have the
   p2p path read the source block's true count (needs a pool accessor),
   and add a round-trip num_tokens assertion to all three handoff tests.

## L2. dispatch_prefill / dispatch_decode transfer ONLY layer 0 of multi-layer blocks
   src/lib.rs:493-505 (prefill), 531-544 (decode)

   `extract_and_send_prefill`/`extract_and_send_decode` read
   `pool.read_keys(block_id)` — the layer-0 `key_data` mirror — and send
   with `layer_idx=0`. `transfer_kv_cache_real` by contrast iterates all
   layers via `read_layer_keys`. The two push APIs have contradictory
   wire shapes for the same logical operation; on any model with >1
   layer, the dispatch paths silently deliver layer 0 only (receiver
   happily stores layer 0, sender reports Ok). Second-hop handoffs of
   network-received blocks (which the receiver stores per-layer) lose
   layers 1..N. The trait doc claims "real KV blocks (its logical→physical
   block table)" — it is layer-0-only in fact.

## B1. Pool mutex held across network I/O with retry sleeps; push path has no socket timeout
   src/lib.rs:489-515, 528-545, 609-613; grim-kvtransport/src/lib.rs:593-602

   `transfer_kv_cache` (trait impl), `extract_and_send_prefill`, and
   `extract_and_send_decode` lock the engine's shared KvBlockPool and hold
   the guard across `retrying()` sends: per attempt a 500 ms connect
   timeout, plus backoff sleeps 50→800 ms. Worse, `send_block_remote`
   sets NO read/write timeout on the push socket — a receiver that accepts
   then wedges stalls `write_all` indefinitely under TCP backpressure,
   and the pool mutex is held the whole time. Every other request on the
   node that touches the pool stalls with it. No whole-operation deadline
   or circuit breaker exists; a 200-block × 80-layer handoff to a dead
   peer retries per block-layer.

## B2. Fire-and-forget push: receiver-side rejection is indistinguishable from success
   grim-kvtransport/src/lib.rs:599-602 (sender), 1040-1088 (receiver)

   `send_block_remote` returns Ok once bytes leave the kernel buffer —
   there is no ACK. Receiver-side checksum mismatches, out-of-range
   block ids, and oversized payloads are eprintln'd and dropped; the
   sender's retry machinery only ever sees connect/write errors. A
   handoff in which the receiver discarded EVERY block reports Ok(())
   and the decode node attends zeros. grim-disagg's transient-vs-final
   error classification provides false confidence on the push paths.

## B3. dispatch paths feed caller-supplied block_ids into panicking accessors
   src/lib.rs:493-495, 531-533; grim-memory/src/lib.rs:752-758, 656-667

   `read_keys`/`read_values`/`write_keys` index `self.blocks[id]`
   directly — out-of-range ids panic — while `read_layer_keys`/
   `write_layer_keys` bounds-check and return None/no-op. The dispatch
   paths use the panicking pair with block tables taken from callers;
   a malformed table panics the node instead of erroring the handoff.

## B4. Retry classification is Display-string matching and misses EPIPE
   src/lib.rs:41-53

   `is_transient_transfer_error` matches substrings like "Connection
   refused"/"reset by peer". "Broken pipe (os error 32)" — the sibling
   EPIPE failure that the fire-and-forget push is most likely to see on
   a receiver-side close — matches NO pattern, so it fails fast while
   its semantic twin "reset by peer" retries. Any future rewording of
   kvtransport error strings silently changes retry semantics. typed
   error kinds would remove both problems.

## B5. evaluate_failover never fails over before the first heartbeat
   src/lib.rs:250-266

   The `last_prefill_heartbeat_ms > 0` guard means a Decode node whose
   prefill peer NEVER heartbeats (e.g. all pushes silently dropped per
   B2, so no successful transfer ever marks a heartbeat) trusts it
   forever — the timeout path is unreachable until the first successful
   interaction. Defensible as a startup guard; there is no startup grace
   deadline backing it up.

## B6. record_heartbeat(Colocated) refreshes flags but not timestamps
   src/lib.rs:242-246

   The Prefill/Decode arms update both `*_healthy` and the heartbeat
   timestamp; the Colocated arm sets only the flags. A colocated
   heartbeat claims health without freshness — asymmetric, likely
   unintended.

## P1. ReMPMigrationBatch::migrate is O((L·C)²)
   src/lib.rs:101-116

   `blocks.iter().find(...)` per (layer, chunk) pair. 80 layers × 256
   chunks = 20,480 blocks scanned 20,480 times ≈ 4×10⁸ comparisons per
   migration. Sort once by (layer_idx, seq_chunk) or index in a HashMap.

## P2. ReMP batch validation ignores per-block data-length uniformity
   src/lib.rs:78-93

   `validate` checks only block count. Mixed-length blocks drain into a
   flat buffer whose implied uniform stride is wrong for any consumer
   slicing `flat[(l·C+c)·stride..]`.

## P3. "Zero-copy" claims are false
   src/lib.rs:409-413 (doc), 629-655

   `transfer_kv_p2p_direct` copies through per-layer host Vecs;
   `transfer_kv_colocated` allocates a fresh flat Vec. Both are ordinary
   host copies; neither touches VRAM-to-VRAM.

## P4. One TCP connection per (block, layer); "asynchronous" streamer is blocking
   grim-kvtransport/src/lib.rs:593; src/lib.rs:156-207

   A 200-block × 80-layer handoff opens 16,000 connections.
   `LayerPipelinedKvStreamer::stream_layer_block` doc says "asynchronously"
   but sleeps inline on the caller thread; it also duplicates the retry
   loop instead of sharing `DisaggRouter::retrying` (already drifted: no
   retry logging), and nothing in engine/CLI constructs it.

## P5. evaluate_failover's result never updates handles_prefill()/handles_decode()
   src/lib.rs:250-278

   Failover returns Colocated but `config.role` is untouched, so the
   `handles_*` predicates keep reporting the pre-failover role. The
   engine works around this by caching the effective role separately
   (grim-engine:667-679); direct users of the orchestrator get stale
   answers from the predicates.

## P6. KvReceiverServer cannot be shut down; PromptChannel is unbounded
   src/lib.rs:665-707; grim-kvtransport/src/lib.rs:795-828

   The accept thread runs forever after drop (handle is dead
   #[allow(dead_code)]); no shutdown API. Prompts for request ids that
   are never `take()`n accumulate for the process lifetime.

## G1. No num_tokens fidelity test on any handoff (the L1 hole)
## G2. transfer_paged_cache_real has ZERO test coverage
   src/lib.rs:442-470 — the PagedKvCache::layer_block_slice layout never
   round-trips through the wire in any test.
## G3. dispatch_prefill/dispatch_decode wire behavior untested
   Only the no-pool error paths are tested; the layer-0 under-transfer
   (L2) is invisible to the suite.
## G4. Float round-trip tests never exercise NaN/Inf/-0.0/denormals
   tidy arithmetic sequences only. (kvtransport's raw-byte FNV checksum
   + LE encode would handle them — disagg never checks.)
## G5. evaluate_failover edge cases untested in-crate
   Prefill-role branch, the ts==0 guard (B5), timeout boundary equality,
   and in-crate recovery after failover.
## G6. Integration tests sleep fixed 150-300 ms for async receiver commits
   Poll-with-deadline helpers would de-flake under CI load.

## C1. Strengths (notable good bits)
   - ZERO unwraps/expects in src/lib.rs proper; mutex poisoning is
     handled with contextual map_err everywhere — genuinely clean.
   - Retry design distinguishes transient vs final failures (rare
     discipline), and the fail-fast "not available" behavior is actually
     timing-asserted in a test.
   - Pull path never fabricates data: empty payload → explicit error.
   - Wire protocol is bit-exact: explicit LE encoding, checksum over raw
     received bytes (NaN-payload safe), verified before parse; KVT-1 cap
     and F8/F10 fixes show real hardening history.
   - Empty-block-list rejection is consistent across every transfer
     entry point.
   - Receiver-side num_tokens derivation is defensive (checked_div, min
     cap, elem mismatch warning) — L1 is a protocol gap, not sloppy code.

---

## Priority

CRITICAL: L1 — silent valid-length corruption on every handoff (proven).
HIGH:     L2, B1, B2 — under-transfer, pool-wide stall, undetectable
          handoff loss.
MEDIUM:   B3, B4, B5, G1-G3 — panic pathway, fragile retry classes,
          failover hole, test gaps hiding the criticals.
LOW:      B6, P1-P6, G4-G6 — asymmetries, perf, docs, de-flaking.

---

## Fixes applied (post-audit)

Wire protocol (grim-kvtransport, protocol V2 → V3):
- L1: `KvBlockHeader` gains `num_tokens: u32` (SIZE 28 → 32). Sender
  carries the block's valid token count end-to-end; the receiver stores
  it capped by the payload-derived count (V2 senders fall back to the
  old derived count).
- B2: every push (block and prompt) is now ACKed with a 1-header reply;
  `checksum == ACK_OK(1)` = committed, `0` = rejected. Rejections
  surface as FINAL sender errors ("KV receiver rejected block …");
  ACK transport failures are transient and retried. `fetch` replies
  carry the store's real fill state via the new
  `KvBlockStore::block_num_tokens` (KvBlockPool implements it).
- B1 (half): push sockets get 30 s read/write deadlines (were none);
  receiver-side accepted sockets too, so a stalled peer can no longer
  wedge the accept loop.
- P4 (half): new `send_blocks_batch_remote` pipelines many messages
  through one connection (write-all, then read all ACKs); the receiver
  drain loop handles both batched and legacy one-message connections.
  `send_block_remote` keeps one-shot semantics.
- Receiver hardening: `layer_idx` cap (4 096) on push messages before
  the per-layer allocation, NAK on every rejection path, accept-loop
  backoff on persistent errors, stoppable variant
  (`start_kv_receiver_server_stoppable`).
- P6: `PromptChannel` bounded at `MAX_PENDING_PROMPTS` (10 000) with
  eviction-on-full.

grim-memory:
- L1: `KvBlockPool::block_num_tokens(id)` (inherent + trait);
  `PagedKvCache::block_num_tokens(id)` computes full-page/tail fill
  from the block table and committed count.
- grim-core: `KvCache::block_num_tokens` provided method so the engine
  can read fill state through the trait object.

grim-disagg:
- L1: `transfer_kv_p2p_direct` now writes the source block's real token
  count (was the element count).
- L2: `dispatch_prefill`/`dispatch_decode` snapshot and send ALL layers
  (was layer-0 only), matching `transfer_kv_cache_real`.
- B1: every pool-touching path snapshots under the lock and releases it
  before any network I/O; sends go through chunked batch transfers
  (256 messages/connection) with retry per chunk.
- B3: block ids are bounds-checked against `num_blocks()` with an
  error, and unpopulated blocks (`!block_is_received`) error instead of
  silently transferring (or panicking on) zero data.
- B4: transient classifier also matches `Broken pipe` and
  `connection aborted`; ACK-rejection strings deliberately don't match
  (final).
- B5: failover baseline is the last heartbeat, or the first
  `evaluate_failover` call for peers that never heartbeated — one
  timeout window of startup grace, then colocated. No more trusting a
  silent peer forever.
- B6: a Colocated heartbeat advances both freshness timestamps.
- P1: `ReMPMigrationBatch::migrate` indexes blocks once (linear);
  duplicate and out-of-bounds (layer, chunk) entries are rejected.
- P2: `validate` requires uniform non-empty block data lengths.
- P3: "zero-copy" doc claims replaced with honest host-memcpy wording.
- P4: `LayerPipelinedKvStreamer` documented as synchronous, takes
  `num_tokens`, and shares the router's `retry_with_policy` helper.
- P5: orchestrator retains `effective_role`;
  `handles_prefill`/`handles_decode` report the post-failover truth.
- P6: `KvReceiverServer` signals its stop flag on Drop.
- Doc: `fetch_kv_block`/`send_layer_block_remote`/
  `stream_layer_block` signature changes documented.

grim-engine:
- Decode pull path stores the fetched block's wire-provided token
  count (was `block_elems / elem_per_token`, i.e. always full).
- Prefill streaming path sends `kv.block_num_tokens(b_id)` per block.

New tests: wire-push and P2P num_tokens fidelity (the L1 probes, now
permanent), receiver-rejection-surfaces-as-error (B2), dispatch_prefill
all-layers round-trip (L2), failover-without-heartbeat and recovery and
Colocated-heartbeat freshness (B5/B6), inconsistent/duplicate ReMP
blocks (P1/P2), PagedKvCache wire round-trip with a partial tail block
(G2), token-count assertions added to existing round-trip tests. Push
ACKs make most receiver-commit sleeps unnecessary; remaining sleeps are
poll loops.

Verification: grim-kvtransport 33, grim-memory 34, grim-disagg 29,
grim-core 24, grim-engine 126 (incl. disagg loopback + orchestrator
suites) — all passing. grim-server/grim-cli consume no changed
signatures (verified by grep); their cache-blocked recompile is noted
in the session log.

