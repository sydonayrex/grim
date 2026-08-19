# Grim — Security & Robustness Fix Plan

Filtered from `anaconda.md` (verified revision). Only items assessed as worth
implementing are included — see rationale under each item for why. Ranked in
implementation order, not by the source document's severity labels, since
that ranking conflated "real vulnerability" with "structurally unsound but
currently unexercised" with "correctness nice-to-have." Each item states the
threat/risk it addresses, the exact fix, and how to verify it.

Excluded entirely: N1 (a lint-policy suggestion, not a code fix), N2
(explicitly assessed as not currently exploitable — bounded by user input
length, no live risk), N3 (already fixed and already has a description of
what test to add — nothing new to plan here beyond what N3 itself says).

---

## 1. Plugin `.so`/`.dylib` loaded via `dlopen` with no integrity check

**Category:** Real vulnerability, present-tense. Not contingent on any usage
pattern or threading model — arbitrary code execution the moment a tampered
or malicious plugin binary is loaded.

**Where:** `crates/grim-plugin/src/dylib_loader.rs`, `DylibPluginLoader::load`
(currently ~line 111), which calls `libloading::Library::new(path.as_ref())`
(~line 121) directly against a config-supplied path with no verification
step beforehand.

**Fix:**
1. Add a `expected_sha256: Option<String>` field to whatever manifest struct
   already carries the dylib's path (check `PluginManifest` in
   `crates/grim-plugin/src/lib.rs` for the existing capability-grant fields
   this should sit alongside).
2. In `DylibPluginLoader::load`, before calling `Library::new`:
   - Read the file at `path` into memory (or hash it via a streaming
     `sha2::Sha256` reader to avoid loading the whole binary into RAM for
     large plugins).
   - Compute its SHA-256 digest.
   - If `expected_sha256` is present in the manifest, compare and return a
     clear `Err` (not a panic) on mismatch, naming the plugin and both
     digests in the error message so a legitimate mismatch (e.g. plugin
     rebuilt without updating the manifest) is easy to diagnose.
   - If `expected_sha256` is absent, the current behavior (no check) is
     conservative to preserve as a documented opt-in gap — do not silently
     require a hash for every plugin without a migration path. Instead, log
     a loud warning (`tracing::warn!` or equivalent) once per load:
     `"plugin {name} loaded with no pinned hash — integrity unverified"`.
3. Add a config-level flag (e.g. `require_pinned_hash: bool`, default
   `false` at first, with a note in the changelog that a future release
   should default it to `true`) that turns the warning into a hard `Err`
   when set — this lets operators opt into strict mode without breaking
   existing unpinned plugin setups immediately.

**Test:**
- Unit: a manifest with a correct `expected_sha256` loads successfully; one
  with an incorrect digest returns `Err` and never reaches `Library::new`
  (assert via a test double or by checking the returned error variant, not
  by inspecting process state).
- Unit: a manifest with no `expected_sha256` still loads (preserves current
  behavior) but the warning path is exercised — capture logs in the test if
  the logging framework supports it, or refactor the warn into a return
  value/side-channel the test can observe.
- Manual: flip `require_pinned_hash: true` in a local config, confirm an
  unpinned plugin now fails to load with a clear message.

---

## 2. `unsafe impl Send + Sync` cluster with prose-only safety contracts

**Category:** Structurally unsound, not currently exploitable. Verified that
every live call path into these types passes through `AppState.engine:
Mutex<Engine>` in `grim-server`, so no concurrent access is possible through
the server's actual API today. This is worth fixing as defense-in-depth and
to prevent a future refactor (worker pool, background prefetch thread,
multi-engine sharding) from silently reintroducing a real race with no
compiler-enforced guardrail stopping it — not because it's exploitable now.

Grouping these five together since they're the same pattern in five
locations and should be fixed with one consistent approach, not five
independent ad-hoc patches:

- `crates/grim-tensor/src/backend.rs:1760-1763` —
  `QuantizedMatmulBackwardResiduals`
- `crates/grim-backend-rocm/src/device/roc_device.rs:214-215` — `RocmDevice`
- `crates/grim-backend-rocm/src/rccl.rs:10-11` — `NcclComm`
- `crates/grim-backend-rocm/src/device/rocblas.rs:35` — `RocblasHandle`
  (**note:** already `Send`-only, not `Sync` — this one was already
  partially hardened in an earlier pass with a documented rationale comment;
  leave as-is, it does not need the guard treatment below, only confirm the
  existing comment stays accurate)
- `crates/grim-backend-rocm/src/device/handles.rs:35` — `RocmHandle`
  (`Send`-only already, same note as above)
- `crates/grim-backend-rocm/src/p2p_route.rs:100-101, 244-245` —
  `HostStagingBuffer`, `StagingCache`

**Fix — for the genuinely `Send + Sync` types (`QuantizedMatmulBackward
Residuals`, `RocmDevice`, `NcclComm`, `HostStagingBuffer`, `StagingCache`):**

Do not remove the `unsafe impl` blocks — the current application-level
locking makes that unnecessary churn and risks breaking legitimate call
sites that do rely on `Send` (moving a value into a worker thread once,
which is safe) rather than true concurrent `Sync` access. Instead, close the
actual gap, which is that nothing enforces the "exactly one caller at a
time" invariant these types depend on:

1. For each type, add a `SAFETY:` comment directly above the `unsafe impl`
   block (not just a doc comment above the struct) stating explicitly:
   - What the actual invariant is (e.g. "safe to `Send` because HIP handles
     are process-global; **not** safe for concurrent access — caller must
     serialize via an external lock").
   - Where that serialization currently lives in practice (e.g. "currently
     enforced by `AppState.engine: Mutex<Engine>` in grim-server — do not
     remove that lock or add a second concurrent access path without adding
     an internal `Mutex` here first").
2. Where a type is only ever constructed inside code that already holds the
   `Engine` mutex (true for all five here, per the verification above), add
   a debug-only assertion or a newtype wrapper that makes the "must be
   called under the engine lock" requirement structural rather than purely
   documented, if it can be done without a large refactor. If a lightweight
   structural guard isn't feasible without disproportionate churn, the
   `SAFETY:` comment from step 1 is the acceptable fallback — the goal is
   that the next person to add a second access path finds a comment that
   tells them exactly what will break, not silence.
3. Do **not** silently gate all device ops behind a new internal mutex as a
   first step — that would double-lock under the existing `Mutex<Engine>`
   and add contention with no correctness benefit given current usage. Only
   add internal locking if/when a real second concurrent access path is
   introduced (e.g. a worker-pool refactor); until then, the documented
   invariant plus the existing external lock is sufficient and correctly
   scoped to actual risk.

**Test:**
- No new runtime test needed for the comment-only fix — this is a
  documentation/guardrail change, not a behavior change, and adding a
  concurrency test against types with no current concurrent call path would
  test something that can't happen rather than something that does.
- Do add one regression test at the `AppState.engine` level: assert (via a
  code comment or, if the test harness supports it, a static check) that
  `AppState.engine` remains `Mutex<Engine>` and not e.g. `RwLock<Engine>` or
  unwrapped — since an `RwLock` would allow concurrent *readers*, which is
  exactly the access pattern these `unsafe impl` blocks aren't proven safe
  against. This is a cheap trip-wire against the most likely way this
  invariant gets silently broken later.

---

## 3. GGUF nested-array metadata parsing has no recursion depth cap

**Category:** Correctness/DoS robustness gap. Severity is contingent on
whether GGUF files are ever loaded from an untrusted source (model
marketplace, user upload, URL fetch) — worth confirming that with whoever
owns the model-loading trust model before treating this as urgent, but worth
fixing regardless since the fix is cheap and the current behavior (crash the
whole process on a crafted file) is a bad failure mode even for
trusted-but-corrupted input.

**Where:** `crates/grim-format/src/gguf.rs`, `read_gguf_value_with_tag`
(currently at line 1635), specifically the array branch (tag `9`, ~line
1675-1691) which recurses via `read_gguf_value_with_tag(r, elem_tag)` with
no depth tracking. The existing `count > 10_000_000` check bounds the
*width* of any single array but not nesting *depth* — a file with a long
chain of singly-nested one-element arrays defeats it entirely.

**Fix:**
1. Add a `depth: u32` parameter to `read_gguf_value_with_tag`, threaded
   through from `read_gguf_value` (which becomes the only call site that
   starts at `depth: 0`).
2. Add a named constant near the top of the file:
   `const MAX_GGUF_ARRAY_NESTING_DEPTH: u32 = 64;` — 64 is generous for any
   legitimate metadata structure GGUF actually uses (real-world GGUF
   metadata arrays are effectively always depth 1, i.e. a flat list of
   scalars; even unusually structured metadata is very unlikely to need
   more than a handful of nesting levels) while still bounding worst-case
   stack usage to something the default thread stack size comfortably
   survives.
3. At the top of the array branch, before recursing: if
   `depth >= MAX_GGUF_ARRAY_NESTING_DEPTH`, return
   `Err(Error::Backend(format!("GGUF array nesting exceeds max depth of
   {MAX_GGUF_ARRAY_NESTING_DEPTH}")))` instead of recursing further.
4. Pass `depth + 1` into the recursive call.

**Test:**
- Unit: construct a byte buffer encoding a legitimately nested array 2-3
  levels deep; confirm it still parses correctly (regression guard that the
  depth threading didn't break normal nested-array metadata).
- Unit: construct a byte buffer encoding arrays nested past
  `MAX_GGUF_ARRAY_NESTING_DEPTH`; confirm it returns the new `Err` rather
  than recursing — this test does not need to nest anywhere near deep enough
  to actually risk a stack overflow in the test process itself; it only
  needs to exceed the constant to prove the guard fires before that point.
- No change needed to the existing `count > 10_000_000` width check — that
  guard is correct and orthogonal to this fix; both should remain.

---

## 4. Document the existing WASM plugin fuel/memory defaults, and close the opt-out gap

**Category:** Mostly a documentation gap, not a live issue — verified
`PluginLimits::default()` already sets `fuel_per_invocation: Some(50_000)`
and `max_memory_mb: Some(64)`, both bounded. The one real (small) gap: both
fields are `Option`, and the enforcement sites in
`crates/grim-plugin/src/wasm_loader.rs` (`if let Some(fuel) = ...` /
`if let Some(max_mem) = ...`, ~lines 106-228) skip enforcement entirely when
a field is `None` — so a manifest can explicitly opt out of both limits,
silently, by omitting or nulling them.

**Where:** `crates/grim-plugin/src/lib.rs:133-145` (`PluginLimits` struct and
its `Default` impl); enforcement sites in
`crates/grim-plugin/src/wasm_loader.rs` (~lines 106, 120, 214, 228, 312).

**Fix:**
1. Add a doc comment on `PluginLimits` stating the default values explicitly
   (`fuel_per_invocation` defaults to 50,000, `max_memory_mb` defaults to 64
   MB) and stating plainly that setting either field to `None` in a manifest
   disables that limit entirely — this is the fix N4 itself asks for
   ("confirm a bounded default exists"), now made permanent as an in-code
   fact rather than something that has to be re-verified by reading the
   enforcement call sites each time.
2. Decide, with whoever owns the plugin trust model, whether an explicit
   unbounded opt-out should be allowed at all for untrusted/third-party
   plugins. If not: change `fuel_per_invocation` and `max_memory_mb` from
   `Option<u64>`/`Option<u32>` to plain `u64`/`u32` with the same default
   values, removing the ability to opt out of enforcement via the type
   system rather than via convention. This is a larger, more disruptive
   change (touches every construction site, several of which are visible in
   the existing test module setting explicit `Some(...)` values already) —
   scope it as a follow-up decision, not bundled into the documentation fix
   in step 1, since it changes behavior for any manifest currently relying
   on the opt-out.

**Test:**
- No new test required for step 1 (doc-only).
- If step 2 is taken: update every existing test construction site
  (`fuel_per_invocation: Some(100)` etc., several already visible in
  `wasm_loader.rs`'s test module) to the new non-`Option` type; add one test
  confirming a manifest that previously would have set `None` now fails to
  parse or is clamped to the default, whichever behavior is chosen — this
  needs an explicit decision before writing the test, not just before
  writing the code.
