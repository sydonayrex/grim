# Grim Performance Improvement Plan — Load / Prefill / Decode / Inference

Source: session tracing `Grim v Ollama.md` (2026-09-16, commit `7157204d`) against
current workspace source. Every item below is cited to a specific file/line traced
in this session, not inferred from the benchmark doc or prior audits. Two items
were re-scoped mid-session after a second AI's independent review corrected the
failure-class diagnosis on item P0-1 — that correction is verified against source
below and incorporated.

Priority is ordered by (a) whether it blocks re-enabling a real perf win that's
currently disabled for correctness, (b) crash > wrong-numbers > slow, (c) blast
radius across model families.

---

## P0 — Correctness bugs currently blocking a shipped-off perf win

### P0-1. Allocator use-after-free under pool eviction (the actual §5 crash)

**File:** `grim-backend-rocm/src/memory/allocator.rs:81-110`

**What's confirmed:** `RocmCachingAllocator::free()` pools buffers normally, but
once cached bytes exceed `cap_bytes`, it calls `hipFreeAsync(ptr, null stream)`.
That release is ordered only against other work already submitted to the **null
stream** — it gives no ordering guarantee against a kernel still in flight on
`active_stream()` (the compute or capture stream). If the fused Q8_0 QKV
consume path (`launch_quantize_q8_1` → `launch_dot4_q80_q81_gemv`) is still
executing on the compute stream when one of its input/output buffers gets
evicted from the pool and `hipFreeAsync`'d on the null stream, the page can be
unmapped before the kernel dereferences it. This produces exactly the observed
`Page not present` fault — a use-after-free/UAF, not a stale-data visibility
race.

**Why this explains the bisect evidence better than a visibility-race theory:**
- `HIP_LAUNCH_BLOCKING=1` masking the fault: forces synchronous launches, so
  the host thread blocks until the kernel completes before any subsequent
  free can run — eliminates the UAF window entirely, not just a memory-order
  hazard.
- Fault reproduces on **both** gfx1200 and gfx1201 (both RDNA4): the allocator
  bug is architecture-independent, unlike a hardware-conditioned visibility
  quirk.
- Individual fusion-plan commits pass, the merge combination faults: each
  commit's added memory footprint (A5 arenas, M2 staging, wider JIT
  translation units per the other AI's review) pushes cumulative pool usage
  over `cap_bytes` only in combination — explaining "why now."
- `GRIM_ALLOC_NO_POOL=1` (every free synchronized + real, bypassing the pool
  and the null-stream `hipFreeAsync` path entirely) was reported to clear the
  fault on the exact failing default config. This is the allocator's own
  existing diagnostic escape hatch (`allocator.rs:82-84`, dated to a prior
  "GGUF fault hunt") — a controlled experiment isolating the pool-eviction
  path as the crash mechanism, not a new discovery requiring separate
  verification.

**Fix:** stream-track frees. Record the owning stream at allocation time;
either (a) free via `hipFreeAsync` on *that* stream rather than the null
stream, so ordering against in-flight consumers on the same stream is
preserved by construction, or (b) fence the eviction with an explicit
`hipEventRecord`/`hipStreamWaitEvent` pair against the last stream the buffer
was used on. This is a root-cause fix at the allocator level — it protects
every kernel path that can trigger pool eviction under memory pressure, not
just fused QKV. Template this the same way for any other `hipFreeAsync`/pool
eviction call sites in the allocator.

**Scope guard:** do not conflate this with the `is_rdna34` sync-skip or
`upload_in_flight` items below (P1). Those are real but are a different
failure class (wrong numbers, not a crash) and must not gate this fix or be
used to justify delaying it.

### P0-2. `kv_append` hard-downcast rejects `RocmStorageView`, no fallback

**File:** `grim-backend-rocm/src/kernels/qkv_attention.rs:1677` (and the
`_batch` variant at 1732)

**What's confirmed:** `launch_kv_append`'s `k_rot` parameter is downcast via
`.as_any().downcast_ref::<RocmStorage>()`, which returns `None` — and thus a
fatal `Err("kv_append: k_rot must be RocmStorage")` — for the distinct
`RocmStorageView` type (the non-owning sub-view type used for zero-copy
slicing, per existing project notes on why `RocmStorageView` has no `Drop`
impl). `fused_qkv_dot4_decode` (`lfm2.rs:1627-1664`) builds its q/k/v outputs
as `RocmStorageView::from_offset(...)` slices of one fused GEMV output buffer.

**Not fully re-traced this session:** whether the *specific* `k_rot_storage`
value that reaches `launch_kv_append` at `lfm2.rs:1840` is actually the
`RocmStorageView` produced by the fused path, versus a different
`Box<dyn BackendStorage>` constructed earlier in the RoPE-device-base branch
(`lfm2.rs:948-970`). The two code paths are structurally close but weren't
conclusively unified in this session's trace. **Action item 0:** confirm this
call-graph edge before starting the fix — if the fused-QKV view never actually
reaches `kv_append` as a raw `RocmStorageView`, this item is moot and the
double-failure claim needs revising.

**Fix (once confirmed):**
1. Make `kv_append`'s downcast accept anything that exposes a device pointer +
   length — either a small trait (`DevicePtrView` or similar) implemented by
   both `RocmStorage` and `RocmStorageView`, or an explicit `RocmStorageView`
   branch alongside the existing `RocmStorage` branch.
2. Separately: wherever this `Err` currently propagates as fatal in the
   decode step, route it to the existing eager-fallback pattern already used
   elsewhere in this file (`return self.drive_forward(...)`-style degrade)
   instead of terminating the request. A type-mismatch on a view should never
   be a harder failure than "fall back to the slower, working path."

**Sequencing:** P0-2 depends on P0-1 being fixed first to even get a clean
enough run to observe it (per the other AI's report: suppressing the UAF is
what exposed this as the next failure). Do not attempt to validate P0-2 fixes
against a build that still has the allocator bug — a UAF-corrupted run will
produce noisy, non-reproducible secondary symptoms.

**Net effect once both land:** the fused Q8_0 QKV blob path (3 GEMV launches
→ 1 per decode step, the ~9% per-token win already visible in eager/non-fused
form) becomes safe to re-enable as the *default*, not just something reachable
via bisection. This directly improves default decode latency, not just the
disabled-by-workaround numbers already reported.

---

## P1 — Real bugs, wrong-numbers class, not release-blocking but should not regress further

### P1-1. `launch_quantize_q8_1` RDNA3/4 sync-skip is unverified for capture mode

**File:** `grim-backend-rocm/src/device/device_compute.rs:2240-2278`

**What's confirmed:** `if !self.is_rdna34 { self.synchronize(); }` — the
barrier between `quantize_q8_1` and the consuming dot4 GEMV is skipped
whenever `is_rdna34` is true (gfx1200/gfx1201 both qualify, confirmed via
`quantization.rs` arch classification). The skip's justification (comment,
same file) is scoped to eager same-stream ordering; it was never validated
under `hipStreamBeginCapture`/replay, where the same-stream assumption may not
hold the same way.

**Correction from this session's review process:** this is **not** the
mechanism behind the §5 crash (P0-1 is). This is a correctness risk in its own
right — worth closing — but should not be used to gate the P0-1/P0-2 fix or
the release. Re-scoped to P1 accordingly.

**Fix:** replace the `is_rdna34` boolean skip with an explicit
`hipEventRecord` (after quantize) / `hipStreamWaitEvent` (before dot4 GEMV)
pair. Removes the architecture-conditioned guess entirely in favor of a real
dependency edge that's correct under both eager and capture execution.

### P1-2. `active_stream()`'s `upload_in_flight` flag is a shared, non-per-pair race

**File:** `grim-backend-rocm/src/device/roc_device.rs:1065-1088`

**What's confirmed:** `upload_in_flight` is a single global `AtomicBool` +
`Mutex<Option<event>>`, consumed and cleared by *any* call to
`active_stream()`. Two kernel launches in the same decode step (e.g. quantize
then dot4 GEMV) can race on which one "claims" a pending upload's wait-event,
leaving the other silently unguarded if a third concurrent transfer sets the
flag back to `true` in between.

**Fix:** replace the single global flag with per-buffer or per-launch-pair
event tracking (mirrors the P1-1 fix pattern) rather than a shared flag any
caller can consume. Same root cause class as P1-1 — same fix shape — worth
doing as one pass over the file rather than two.

### P1-3. `min_tokens` is parsed but never enforced (early-EOS)

**File:** `grim-server/src/lib.rs:1204` (parsed into `SamplerParams`), never
read again anywhere in the workspace (confirmed via grep across
`grim-server`, `grim-constrain`, `grim-engine`).

**What's confirmed:** the streaming loop's EOS check
(`grim-server/src/lib.rs:1724`) is `hit_eos = eos_token_id_clone ==
Some(token_id)` — unconditional, no step-count gate. `--min-tokens`, used in
the original benchmark to get clean sustained-rate numbers, has no working
enforcement path; whatever suppressed EOS in that run was greedy-decoding
behavior, not this flag.

**Fix:** thread `min_tokens` from `SamplerParams` into the streaming loop;
gate `hit_eos` on `step >= min_tokens`. Small, contained change — one new
condition on one existing boolean.

**Note:** this doesn't improve ms/token or launch counts, but it's the fix
that makes the benchmark's own quality-of-output caveat (§4) go away, and
without it any future sustained-rate benchmark on small/low-temp models needs
a real guardrail, not an unenforced flag.

---

## P2 — Structural gaps: architecture doesn't support the win yet

### P2-1. TTFT: no session-continuity layer between HTTP requests

**Files:** `grim-server/src/lib.rs:1564` (`session_request_id` freshly
allocated per call to `grim_generate`, confirmed as the handler bound to
`/api/generate`); `grim-engine/src/lib.rs:2296` (`seed_kv_arena_from_eager`,
correctly gated by `!self.decode_graphs.contains_key(&request_id)` — already
skips re-seeding *within* one streaming session).

**What's confirmed:** the seed-skip logic already does the right thing given
a stable `request_id`. The problem is `request_id` is never stable across
separate HTTP calls — every `/api/generate` request is, by construction, a
brand-new session, so the ~170ms one-time seed+capture cost is paid on every
request, not just "the first request of a session" in the way the benchmark's
original fix suggestion implied.

**This is not a one-line fix.** It requires:
1. A session/conversation identity mechanism at the transport layer (a
   session token, or model+prior-state matching) so `grim_generate` can
   choose to reuse an existing `request_id`/decode-graph/KV-arena instead of
   allocating fresh.
2. Only once (1) exists does the already-correct seed-skip in
   `grim-engine` start paying off across requests instead of just across
   tokens within one request.

**Recommendation:** scope this as its own design document before
implementation — it touches request routing, not just the ROCm backend, and
has correctness implications (stale KV reuse across genuinely unrelated
requests if session matching is done wrong). This is the single highest-value
item for TTFT (178ms → likely tens-of-ms, putting Grim at or below Ollama on
the metric it currently loses worst on) but it's also the biggest single
scope item in this plan.

### P2-2. Qwen3.5 (dense) has no graph-capture path; multiple per-step D2H/H2D round-trips

**File:** `grim-models/transformer/src/qwen35.rs`

**What's confirmed:** no `GRIM_DECODE_GRAPH` gate, no `begin_capture`/
`forward_capture` usage anywhere in the file — Qwen3.5 dense never takes the
LFM2-specific fast path in `grim-engine/src/lib.rs` (gated on
`downcast_ref::<Lfm2>()`, always `None` here). It falls through to the
generic `rocm.begin_graph_capture()` → `decode_one()` →
`rocm.end_graph_capture()` bracket instead, which is real
`hipStreamBeginCapture`/`hipStreamEndCapture` (relaxed mode).

Inside that bracket, `qwen35.rs`'s forward path calls `to_vec_f32()` at least
five times per decode step (attention output, QKV, conv result, conv state,
gate tensor), each followed by a `from_cpu()` reupload. Synchronous
host-blocking ops are illegal mid-capture under HIP's stream-capture
semantics.

**Consequence, not fully resolved this session:** either (a) capture fails
outright and propagates as a hard `Err` with no eager-fallback at that
specific call site (`rocm.end_graph_capture(capture_key)?` is a bare `?`,
confirmed at `grim-engine/src/lib.rs:2402`), meaning Qwen3.5 decode may be
broken under graph-capture-enabled ROCm configs rather than merely slow, or
(b) something upstream already routes Qwen3.5 to plain eager `drive_forward`
before ever reaching this function, avoiding the crash but forgoing capture
entirely. **Action item:** trace the dispatch router (`grim-cli`/
`grim-server`/`grim-engine` call sites choosing between
`drive_forward_graph_capture` and `drive_forward`) to determine which of
these is actually happening today before prioritizing a fix.

**Fix, once the above is resolved:** move the D2H/H2D-crossing ops onto
device-resident equivalents (mirroring the pattern already used for LFM2's
`fused_or_scalar_attention_arena_device` and `block.rs`'s device-side arena
append) so the whole decode step becomes capture-eligible. This is a bigger
lift than any single item above — five separate host round-trips to
eliminate, not one — but unlocks the same 4.59x-class launch-count win your
own LFM2 microbench already demonstrated, for a second model family.

### P2-3. Qwen3.5-MoE: already correctly wired to Charon — verify only, no fix needed

**Files:** `grim-models/transformer/src/qwen35moe.rs`,
`grim-nn/src/moe.rs`

**What's confirmed:** contrary to an earlier hypothesis raised mid-session,
`Qwen35MoeLayer` already constructs `ExpertBank::from_linears(gates, ups,
downs)` from individually-owned 2-D `Linear` weights — precisely the shape
`MoeFfn::forward_rocm`'s Charon dispatch (`RocmResidentWeights::build`,
`moe_route_topk_on_device`, `moe_fused_dispatch_resident_routing`) expects.
`rocm-mem` is a default-on feature in `grim-nn`. The doc comment on
`Qwen35MoeLayer` claiming Charon dispatch is accurate, not aspirational — this
is unlike LFM2's MoE, which genuinely needs a new construction path (see
P2-4) because it stores stacked 3-D tensors rather than per-expert 2-D
`Linear`s.

**Not fully closed this session:** whether `move_to_device(&mlp_out,
x.device())` at `qwen35moe.rs:297` (called after `self.moe.forward(...)`) is
a true no-op when `forward_rocm` already returns a `Device::Rocm(ordinal)`
tensor, or whether it's doing real work. If it's a no-op, this item is fully
closed. If not, it's a small, well-scoped fix (short-circuit the device-move
when source and destination devices already match) — much smaller than the
D2H surface in P2-2.

**Action item:** trace `move_to_device`'s implementation before closing this
item out formally.

### P2-4. LFM2 MoE loader fixed; Charon bridging for 3-D stacked tensors still open

**File:** `grim-engine/src/model_loader.rs`

**What's confirmed:** both construction sites (~line 1312, ~line 3092) now
correctly read `num_local_experts`/`expert_count` from checkpoint metadata via
the shared `lfm2_moe_fields` helper — the previously-hardcoded `n_expert: 0`
bug is fixed. This doesn't move today's benchmark numbers (LFM2.5-350M is
dense), but blocks any LFM2 MoE variant from reaching the device path at all
until the remaining item lands: `shared_moe.rs`'s Charon path expects
`Vec<MoeExpert>` of individually-owned 2-D weights, but LFM2 stores stacked
3-D tensors `[n_expert, n_ff, hidden]`. Unlike Qwen3.5-MoE (P2-3), this
genuinely needs a new construction path — no existing type in the codebase
today converts LFM2's native layout into `ExpertBank`'s expected shape without
either a per-expert host-side slice-and-copy (defeats the purpose) or a new
device-side unstacking kernel.

**Fix:** as scoped in existing `PLAN-kernel-fusion.md` — write the new loader
path once, informed by how straightforward Qwen3.5-MoE's construction turned
out to be as a reference for what "done" looks like for a per-expert layout.

---

## Suggested execution order

1. **P0-1** (allocator stream-tracked frees) — unblocks everything else in
   the fused-QKV family; fixes an actual crash, not a slowdown.
2. **P0-2 action item 0** (confirm call graph), then **P0-2** fix — un-breaks
   the fused-QKV eager decode path once P0-1 makes it observable.
3. **P1-3** (`min_tokens` enforcement) — small, independent, fixes the
   benchmark's own quality caveat. Can be done in parallel with 1-2.
4. **P1-1 / P1-2** (event-based stream ordering, replacing the arch-boolean
   skip and the shared flag) — do as one pass, same fix shape, same file
   family.
5. **P2-2 action item** (trace the dispatch router for Qwen3.5) — determines
   whether this is a live crash risk (reprioritize to P0) or dormant
   (stays P2).
6. **P2-3 action item** (`move_to_device` no-op check) — cheap to resolve,
   likely closes the item outright.
7. **P2-1** (session continuity) — biggest scope item, highest TTFT payoff;
   start design work in parallel with 1-4, don't block on them.
8. **P2-4** (LFM2 MoE Charon bridging) — lowest urgency; no current benchmark
   exercises it.

---

## Implementation status (2026-09-16, commit e932b566+)

| Item | Status |
|---|---|
| P0-1 allocator eviction sync | **Hardened** (over-cap free now `hipDeviceSynchronize` + `hipFree`, never null-stream `hipFreeAsync`). **Fault persists** → eviction is not the (only) unmap source. Managed-memory `Drop` (hipFree while queued) and pooled-reuse remain suspects; requires rocprof page-fault attribution. Fault still confined to fused Q8_0 QKV blob path; workaround `GRIM_FUSED_QKV=0` unchanged. |
| P0-2 kv_append view acceptance | **Done** — `kv_rot_device_ptr` accepts `RocmStorage` + `RocmStorageView` in `launch_kv_append` + `_batch`. Call graph confirmed (action item 0): fused decode views reached `kv_append`. |
| P0-2 fallback | **Done** — `decode_attention_device` failure now degrades to eager attention paths (never fatal). |
| P1-1 quantize→dot4 ordering | **Done** — event pair (`hipEventRecord`/`hipStreamWaitEvent`, cached per-device `q81_event`) replaces the `is_rdna34` sync-skip. |
| P1-2 upload fence | **Done** — non-consuming fence: launches always wait on the latest upload event; no clear-once race. |
| P1-3 min_tokens | **Done** — `SamplerParams.min_tokens` threaded into the streaming loop; `hit_eos` gated on `step >= min_tokens`. |
| P2-1 session continuity | **Design doc**: `plans/session-continuity-design.md` (layer 1 slot pool = required; layer 2 session identity = optional prefix-cache). |
| P2-2 Qwen3.5 router trace + safety net | **Done** — dispatch traced (`drive_forward_graph_capture` generic bracket); begin/end/replay failures now degrade to eager instead of hard `Err`. Full device-path rewrite remains the bigger lift (documented). |
| P2-3 move_to_device | **Closed** — short-circuits on `x.device() == target` (`grim-nn/src/modules.rs:242`); no-op for ROCm-resident output. |
| P2-4 LFM2 MoE Charon bridging | **Closed** — `moe_expert_at` slices expert rows D2D via `copy_slice_range` (M1-fix, b6528c72); no host roundtrip. |
