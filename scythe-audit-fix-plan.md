# Fix plan — cross-crate audit findings (grim-backend-rocm, grim-engine, grim-nn, grim-kvtransport, grim-scheduler)

Scope: everything found across the multi-GPU `matmul` investigation, the resident-wave
wedge investigation, and the subsequent source audits of `scythe2.rs`,
`scythe_persistent.rs`, `grim-kvtransport`, `grim-scheduler`, `grim-disagg`, and
`grim-memory`. Crates not yet audited or only spot-checked
(`grim-core`, `grim-format`, `grim-quant`, `grim-constrain`, `grim-speculative`,
`grim-server`, `grim-garage`, `grim-models`, `grim-backend-cpu`, `grim-autograd`,
`grim-kvquant`) are out of scope for this plan beyond the specific note under F8.

Findings are grouped by urgency: **Live** (actively breaking something today),
**Trap** (will fault/hang/corrupt the instant something starts calling the code
path — currently safe only because nothing calls it yet), and **Latent** (wrong
behavior under real use, but silent — no crash, just bad output or bad perf).

---

## Tier 0 — Live, blocking production paths

### F0. Resident-wave busy-poll spin (the SB6 wedge)
- **Where:** `grim-backend-rocm/src/kernels/scythe_persistent.rs`,
  `grim_scythe_persistent_dispatch`, empty-queue branch.
- **Symptom:** resident worker wedges after an idle gap; batch A drains fine,
  batch B (after idle) hangs. Matches rocprofv3 seeing the kernel "active."
- **Fix:** bounded exponential backoff (`__builtin_amdgcn_s_sleep`) in the
  empty-queue branch, reset to minimum the instant work is claimed. Patch
  already drafted in this thread — not yet applied.
- **Validation:** `ring_resident_wave_two_batches`
  (`GRIM_GPU_TEST=1 GRIM_SCYTHE_RING_RESIDENT=1`) must pass batch B. Also
  re-run `rocm_persistent_dispatch_opcode_6_device_gated` to confirm the added
  shared var doesn't regress the non-resident bounded path.
- **If backoff doesn't fix it:** don't re-reach for the "aggregate JIT module"
  theory — nothing in this path recompiles between batches. Escalate to
  rocprof-compute / RGP capture on the *post-fix* build to find what else is
  parking the wave.
- **Owner note:** this was misdiagnosed for a while as a JIT-freshness flake;
  the source shows no module reload between batches, so that theory should be
  retired regardless of backoff outcome.

### F1. Missing `DeviceGuard` on 11 rocBLAS/HIP call sites (multi-GPU zeros/page-fault)
- **Where:** `grim-backend-rocm/src/device/roc_device.rs` —
  `matmul_op`, `matmul_with_solution`, `matmul_batched`, `time_kernel_ms`,
  `copy_scythe_descriptor_async`, `copy_via_route` (HostBounce legs),
  `begin_graph_capture`, `upload_from_pinned`, `read_to_host_async`,
  `read_into_pinned`, `copy_slice_into`.
- **Status:** confirmed fixed per your last update (commit `2f2b179` context).
  Listed here only so this document is a complete record — no further action
  unless a regression test surfaces.
- **Follow-up still open:** autotuner calibration data collected for ordinal 1
  before this fix is suspect and should be re-run (per original diagnosis).
  `copy_via_route`'s two `HostBounce` legs need per-leg guards pinned to
  `dst_device`/`src_device` respectively, not a single `self.ordinal` guard —
  confirm the applied patch actually did this per-leg, not device-wide, since
  a naive copy-paste of the `DeviceGuard::set(self.ordinal)` pattern would be
  wrong for a cross-device bounce.

### F10. `fetch_block_remote` has a real, live caller — `grim-disagg`'s pull-mode decode→prefill path
- **Where:** `grim-disagg/src/lib.rs`, `DisaggRouter::fetch_kv_block` wraps
  `kv_client.fetch_block_remote(...)` directly; doc comment labels it
  "decode → prefill pull," i.e. a real disaggregated-serving code path, not
  test-only scaffolding.
- **This escalates F8 from Tier 1 (trap) to Tier 0 (live).** Any real
  deployment using pull-mode KV transfer between prefill/decode nodes will
  deadlock on the first fetch, exactly as described in F8 below — this just
  confirms there's a real caller today, not a hypothetical future one.
- **No test anywhere calls `fetch_kv_block`.** `grim-disagg`'s own test file
  exercises the push path only (`extract_and_send_decode` /
  `send_block_remote`, backed by a real `KvBlockPool` test fixture). The pull
  path has zero coverage, live or in CI — see F8 for the combined fix.
- **Additional scope found for F8's fix (see F8 below for the full writeup):**
  `KvBlockStore` — the trait `start_kv_receiver_server` is generic over — is
  **write-only by design** (`write_keys`, `write_values`, `write_layer_keys`,
  `write_layer_values`, `block_is_received`; no read methods). `KvBlockPool`
  already has the needed inherent read methods (`read_keys`, `read_values`,
  `read_layer_keys` — confirmed present in `grim-memory/src/lib.rs`), they're
  just never exposed through the trait. F8's fix needs to add read methods to
  `KvBlockStore` itself before the server can answer a fetch request.
- **Also found:** `KvBlockPool::set_received`'s doc comment
  (`grim-memory/src/lib.rs` ~line 724) explicitly describes a **pull-path
  retry design that was never built**: *"the disagg pull path clears \[the
  received flag\] when any layer of a fetch fails, so a partial transfer is
  retried next tick instead of attending stale pages."* Nothing in
  `grim-disagg` implements per-layer fetch, retry-on-partial-failure, or a
  "tick" scheduler — `fetch_kv_block` is a single blocking whole-block call
  with no retry logic at all. This is a second, independent "designed but
  never implemented" gap layered on top of the missing server responder —
  budget for both when scoping F8's fix, not just the network protocol half.

### F9. Chunked-prefill `consumed_tokens` resets on the 3rd+ scheduling pass
- **Where:** `grim-scheduler/src/lib.rs`, `Scheduler::schedule()`, lines
  ~307/317/323.
- **Bug:** `running_req.consumed_tokens = chunk_size` and both
  `remainder_req.consumed_tokens = chunk_size` assignments use the *current
  chunk's* size instead of `r.consumed_tokens + chunk_size`. Correct only for
  a request's first chunk (where `r.consumed_tokens` starts at 0).
- **Impact:** any prompt requiring a 3rd+ scheduling pass under sustained
  `pressure_active` has its consumed-token count silently reset backward.
  `compute_token_backlog()` then overestimates remaining work, and if
  `consumed_tokens` is used anywhere as a KV/position offset for the next
  prefill chunk (needs confirming against the caller — not in this crate),
  this reprocesses prompt tokens from the wrong offset, corrupting that
  request's KV state.
- **Fix:**
  ```rust
  running_req.consumed_tokens = r.consumed_tokens + chunk_size;
  // and both remainder_req assignments, same change
  remainder_req.consumed_tokens = r.consumed_tokens + chunk_size;
  ```
- **Validation:** extend `test_chunked_prefill_draining` to call `schedule()`
  a second and third time on the same request (120 tokens, chunk size 50 →
  chunks of 50/50/20) and assert `consumed_tokens` is 50, then 100, then 120
  — not 50, 50, 20. This test gap is exactly why the bug shipped; close it
  as part of this fix, not as a follow-up.
- **Before merging:** grep whoever consumes `Request.consumed_tokens` /
  `SchedulerOutput.prefill_ids` downstream (likely `grim-engine` or
  `grim-server`, not yet audited) to confirm whether this field is used as a
  literal position offset. If it is, this is a correctness bug in production
  today for any long-prompt/high-pressure workload, not just an accounting
  wart — escalate priority accordingly once confirmed.

---

## Tier 1 — Traps: will break immediately if wired up, currently dormant

These are not on fire because nothing calls them yet. They become Tier-0
emergencies the moment someone connects the missing half. Fix before starting
any of: MoE-via-ring integration, attention-via-ring integration, or
multi-rank tensor-parallel placement.

(Cross-node KV fetch — originally listed here — moved to Tier 0 as F10/F8
after confirming `grim-disagg` has a real live caller. See above.)

### F4. `enqueue_via` stores a host pointer in a device-pointer field
- **Where:** `grim-engine/src/scythe2.rs`, `MoETaskDescriptor::enqueue_via`.
- **Bug:** `weight_ptr: self as *const Self as u64` — the address of the
  host-resident `MoETaskDescriptor` struct, fed to a kernel
  (`scythe_persistent.rs` opcode 6) that dereferences it as a device pointer.
- **Fix:** `MoETaskDescriptor` must be allocated in device-visible memory and
  H2D-copied before `weight_ptr` is set — mirror the pattern the *test*
  `rocm_persistent_dispatch_opcode_6_device_gated` already uses correctly
  (`dev.from_cpu_bytes(&moe_bytes, ...)`, then use *that* device pointer).
  `enqueue_via` needs either an `&RocmDevice` parameter to do this upload
  itself, or a separate `MoETaskDescriptor::upload(&self, dev) -> u64`
  step the caller runs before `enqueue_via`.
- **Depends on:** F3 below (schedule buffer layout) must be fixed in the same
  change — they're both part of the same "opcode 6 has no working host→device
  path" gap.

### F3. `schedule_ptr` contiguous-layout contract doesn't match the rest of the codebase
- **Where:** kernel side reads `sorted_token_ids`/`sorted_expert_ids`/
  `sorted_weights` as one contiguous buffer via pointer-offset arithmetic on
  `moe->schedule_ptr`; every real Charon call site in `roc_device.rs` (8+
  places) uploads these as three **independent** buffers with three
  independent pointers.
- **Fix — pick one, don't leave both conventions alive:**
  - **Option A (minimal kernel change):** change the opcode-6 kernel arm to
    take three separate pointers on `MoETaskDescriptor` (`token_ids_ptr`,
    `expert_ids_ptr`, `weights_ptr`) instead of one `schedule_ptr`, matching
    the convention every other call site already uses. Smaller diff, no new
    packing code needed on the host side.
  - **Option B (keep contiguous contract):** add a host-side packing step
    that concatenates the three arrays into one buffer before upload, and
    update `roc_device.rs`'s 8+ call sites to use it too, for consistency.
    Larger diff, but matches the doc comment on `schedule_ptr` as originally
    written.
  - **Recommendation:** Option A. The three-separate-pointer convention is
    already proven correct and exercised on real hardware in 8+ places;
    changing 8 working call sites to match one broken one is backwards.
- **Validation:** a new device-gated test that drives opcode 6 through
  `MoETaskDescriptor::enqueue_via` → `ScytheRing::enqueue` →
  `launch_scythe_persistent_dispatch` end-to-end (not the current
  hand-packed-bytes test, which bypasses the Rust API entirely) and checks
  the MoE output against a host reference. This is the test gap that let F3
  and F4 both ship silently — the existing device-gated test proves the
  kernel arm works with correctly-shaped input, not that anything in this
  crate produces correctly-shaped input.

### F8. `fetch_block_remote` has no server-side implementation — moved to Tier 0, see F10
- **Where:** `grim-kvtransport/src/lib.rs` — client in `fetch_block_remote`,
  server in `start_kv_receiver_server`. Real caller: `grim-disagg`'s
  `DisaggRouter::fetch_kv_block` (F10, Tier 0 — this is live code, not a
  dormant trap; kept the write-up here since the fix is entirely within
  `grim-kvtransport`/`grim-memory`).
- **Bug:** client sends a request with `FETCH_REQUEST_FLAG` set and blocks
  reading a response payload. Server never checks the flag, always behaves
  as a push-receiver, blocks reading a payload the client never sends. Both
  sides deadlock.
- **Fix, revised after auditing `grim-memory`/`grim-disagg` (three parts,
  not one):**
  1. **Add read methods to the `KvBlockStore` trait** in
     `grim-kvtransport/src/lib.rs`: `read_keys`, `read_values`, and
     `read_layer_keys`/`read_layer_values` (matching the write methods
     already there). Implement them on `KvBlockPool` in
     `grim-memory/src/lib.rs` by delegating to the inherent methods that
     already exist there (`read_keys`/`read_values`/`read_layer_keys` are
     already present, just not trait-exposed — confirmed in source).
  2. **Branch on `FETCH_REQUEST_FLAG` in `start_kv_receiver_server`'s accept
     loop:** after deserializing the header, if the flag is set, look up the
     block via the new trait read methods, serialize it the same way
     `fetch_block_remote`'s client expects (header + k-bytes + v-bytes,
     checksummed), and write it back on `stream` instead of trying to read a
     payload. If not set, existing push-receive logic, unchanged.
  3. **Decide whether to build the retry/tick machinery `set_received`'s doc
     comment describes, or fix the comment.** `KvBlockPool::set_received`
     documents a "disagg pull path" that clears the received flag on partial
     multi-layer fetch failure so a caller retries "next tick" — this
     doesn't exist in `grim-disagg` today (`fetch_kv_block` is one blocking
     call, no retry, no per-layer loop, no tick scheduler). Either scope a
     real per-layer fetch-with-retry loop into `DisaggRouter` as part of this
     fix, or — if that's genuinely future work — correct the doc comment so
     it doesn't describe a mechanism that doesn't exist, since as written it
     will mislead the next person auditing this path into assuming retry
     logic is already handled upstream.
- **Validation:** an integration test in `grim-disagg` (not just
  `grim-kvtransport`) that starts a real `start_kv_receiver_server` backed by
  a real `KvBlockPool`, populates a block via the existing push path, then
  calls `DisaggRouter::fetch_kv_block` against it and asserts the
  round-tripped data matches. This closes the actual coverage gap — the
  current test file only exercises push, and unit-testing `fetch_block_remote`
  in isolation from `grim-kvtransport` wouldn't have caught the missing
  `KvBlockStore` read-trait methods, since that gap is entirely in how
  `grim-memory` implements the trait, not in `grim-kvtransport` itself. Use
  a short connect/read timeout in the test so a regression hangs the test
  suite loudly instead of hanging CI silently.
- **Also worth doing while in this code:** add a connect/read timeout to the
  *client* side too (`fetch_block_remote` currently only times out the
  initial `connect_timeout`; the subsequent `read_exact` calls for header and
  payload have no deadline) so a future protocol mismatch fails fast instead
  of hanging a caller forever.

### F2. OP_ATTN kernel silently truncates for head_dim > 256
- **Where:** `grim-backend-rocm/src/kernels/scythe_persistent.rs`, OP_ATTN
  arm, `float acc[256]`.
- **Fix:** either (a) reject the task at claim time — check `head_dim <= 256`
  before dispatch and set `ST_ERROR` if not, so it fails loudly instead of
  silently truncating, or (b) if head_dim > 256 needs to actually work,
  redesign the accumulator (shared memory instead of a per-thread register
  array, or loop-tile over head_dim in chunks of 256). Given opcode 3 has no
  current caller, (a) is the safe minimum fix — restores the "fail loudly"
  property this codebase's other opcodes have, without committing to a
  bigger kernel rewrite for a path nothing uses yet.
- **Validation:** device-gated test with `head_dim=128` (typical) to confirm
  the common case still works, plus a `head_dim=512` case asserting `ST_ERROR`
  once (a) is in place.

---

## Tier 2 — Latent: wrong or wasteful, but silent

### F6. `C2plrController` computes a multi-rank partition softmax then discards all but one entry
- **Where:** `grim-engine/src/scythe2.rs`, `C2plrController::decide_miss`.
- **Bug:** `partition = softmax(logits[k..2k])` is a real per-GPU
  distribution; `ScythePlacement { ranks: vec![selected], partition:
  vec![partition.get(selected)...], ... }` throws away everything except one
  rank's raw (non-renormalized) weight. Currently masked because
  `split_counts`'s backfill logic tops off the sole rank to 100% regardless
  of the value passed in.
- **Fix — do NOT fix in isolation; this is a design decision, not a bug fix:**
  Before touching this, decide whether `C2plrController` is ever meant to
  produce genuine multi-rank (tensor-parallel, work-split) placements, or
  whether single-rank routing (pick the one best GPU per layer) is the
  intended final design and the multi-rank softmax computation is just dead
  weight that should be deleted instead.
  - If multi-rank is intended: extend `decide_miss` to select the *top-N*
    GPUs by logit (not just argmax), renormalize `partition` over just those
    N entries so they sum to 1.0, and return `ranks: top_n_indices,
    partition: renormalized`. This is real design work, not a one-line fix —
    scope it as its own item, not bundled into this cleanup pass.
  - If single-rank is intended: delete the now-misleading `partition`
    softmax computation (lines ~404-413) and the discarded slice, and return
    `partition: vec![1.0]` directly — cheaper, and doesn't imply
    multi-rank support that doesn't exist.
- **Recommendation:** given `grim-cli/src/train.rs` already has a working,
  separate, hand-built multi-rank `ScythePlacement` for data-parallel
  gradient sync (confirmed in this audit — does not go through
  `C2plrController` at all), and nothing else in the ~13 crates reviewed
  calls the multi-rank output path, lean toward the "delete the dead
  computation" option unless there's a near-term roadmap item that needs
  real tensor-parallel per-layer splitting.

### F7. `PlacementCache` fast-path gate uses one global `last_bucket` instead of per-layer
- **Where:** `grim-engine/src/scythe2.rs`, `PlacementCache::get`.
- **Bug:** `if self.last_bucket == shape_bucket` checks one shared field
  against a per-layer (`fast[layer_id]`) cache entry. Any interleaving of two
  different shape buckets across layers in the same forward (plausible under
  farm/pipeline concurrency) causes spurious fast-path misses for layers
  whose entry is actually still valid — falls through to the correct
  slow-path answer, but pays the ~10us/layer `decide_miss` cost instead of
  ~50ns, silently blowing the ITL budget this cache exists to protect.
- **Fix:** track the last-seen bucket **per layer**, not globally —
  e.g. change `fast: Vec<Option<ScythePlacement>>` to
  `fast: Vec<Option<(u16, ScythePlacement)>>` storing `(shape_bucket,
  placement)` per slot, and compare against that layer's own stored bucket
  in `get()`. Remove the standalone `last_bucket` field entirely once this
  lands — it becomes redundant.
- **Validation:** a test that calls `decide()` for layer 0 at bucket A, layer
  1 at bucket B, then layer 0 at bucket A again, and asserts the third call
  is a fast-path hit (currently it would incorrectly miss). Add a cheap
  instrumentation hook or just check via timing/call-count on `decide_miss`
  if there isn't already an easy way to observe fast vs. slow path from a
  test.

### F5. `ScytheRing::published_head` is write-only, never read
- **Where:** `grim-engine/src/scythe2.rs`, `ScytheRing::published_head`.
- **Not a bug** — cosmetic. Either wire it into something that actually
  benefits from an explicit "has this been published" signal (e.g. `enqueue`'s
  ring-full check could use it instead of `head` to avoid a subtle
  overwrite-of-unpublished-slot edge case — worth a five-minute look while
  in this file, not urgent), or remove the field and the store in
  `publish_head()` to reduce confusion for the next person reading this code
  and assuming it does something.

---

## Suggested execution order

1. **F0** (resident-wave backoff) — apply the drafted patch, run the two
   device-gated tests, close out SB6's active investigation either way.
2. **F8/F10** (kvtransport fetch deadlock) — promoted to first-tier urgency:
   this has a real, live, currently-uncovered caller in `grim-disagg`, not a
   dormant trap. Any pull-mode disaggregated-serving deployment deadlocks on
   first use today. Scope includes the `KvBlockStore` trait extension, not
   just the accept-loop branch — see F8 for the three-part fix.
3. **F9** (scheduler consumed_tokens) — small, isolated, but confirm the
   downstream consumer first since severity depends on whether
   `consumed_tokens` is used as a position offset.
4. **F2** (attention head_dim guard) — cheap, restores fail-loud behavior,
   do before opcode 3 gets a real caller.
5. **F3 + F4 together** (MoE-via-ring host↔device path) — larger, do as one
   unit right before MoE-via-ring integration work starts, not speculatively
   now. Needs the new end-to-end device-gated test either way.
6. **F6** — needs a design decision first (keep multi-rank or delete it),
   not just a patch. Flag to whoever owns the SCYTHE-2 roadmap.
7. **F7** — low urgency (perf only, self-heals via slow path), but cheap
   fix once someone's in this file for other reasons; bundle with F6's
   cleanup pass if convenient.
8. **F5** — no urgency, five-minute cleanup whenever convenient.

## Not yet audited / spot-checked only

- **`grim-tensor-graph`** — reviewed in full (327 lines, pure IR/data logic,
  no concurrency or device surface). Nothing found; consistent with your
  project notes on `rocm_fusion_ops` metadata being a known-unconsumed
  producer, not a new bug.
- **`grim-kvquant`** — function inventory only, not deep-audited. Owns KV
  compression math (Tucker decomposition, RotateKV, cross-modal salience);
  orthogonal to the transport-layer bug in F8 (compression acts on block
  content regardless of push/pull), but worth a real pass given it's on the
  same KV-cache critical path.
- **`grim-garage`, `grim-models`, `grim-backend-cpu`, `grim-autograd`,
  `grim-memory` (beyond the `KvBlockStore`/`KvBlockPool` surface covered
  above — radix tree, spill tiers, MoE resident budget, semantic anchor
  modules not reviewed), `grim-core`, `grim-format`, `grim-quant`,
  `grim-constrain`, `grim-speculative`, `grim-server`** — not reviewed this
  pass. `grim-models` (44k lines) and `grim-autograd` (18k lines) are the
  largest unaudited surfaces and worth prioritizing next given their size
  and centrality.

Everything in Tier 1 (F2–F4, F8) should also get a "why did the existing
tests not catch this" pass — every one of them has a test that verifies the
*type-level* contract (fields round-trip, structs are the right size,
opcode dispatch reads the right field) without ever exercising the *real*
host-to-device path end to end. Worth a standing rule for new ring-descriptor
work: no opcode ships without at least one device-gated test that goes
through the actual public Rust API (`enqueue_via`, `submit_*`), not a
hand-packed byte buffer, the way `rocm_persistent_dispatch_opcode_6_device_gated`
currently does it.
