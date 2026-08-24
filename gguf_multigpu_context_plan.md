# Plan: multi-GPU HIP context discipline — fixing the ctx_dev=2 page fault

Extends the fault hunt recorded in `scythe2_syd_beasty_plan.md` (validation
log, 2026-08-23). Every GGUF model page-faults on first prefill when more
than one HIP device is visible; single-device (`ROCR_VISIBLE_DEVICES=0`) runs
green end-to-end. The mechanism is caught on tape:

```
[launch-trace] self_dev=0 ctx_dev=0 grim_transpose_2d_f32   # correct
[launch-trace] self_dev=0 ctx_dev=2 grim_embedding          # APU context!
[launch-trace] self_dev=0 ctx_dev=2 grim_rms_norm           # APU context!
Memory access fault ... Page not present
```

A kernel launched while the calling thread's HIP context is parked on device
2 (the Ryzen iGPU) either executes there against device-0 pointers or writes
through foreign mappings — hence the observed 50/50 utilization split
(9070 XT idle at ~7%, APU 24–26%, one 100% spike on GPU[1]) and the
"Page not present" fault. All work items below are `[host]`-verifiable on
syd-beasty itself; sizes are S unless noted.

Status vocabulary: `[host]` verifiable off-box / on this box without serving ·
`[sb]` produces a number on syd-beasty. A WI is done when its gate row is
checked here AND the harness runs clean in the affected configuration.

---

## What is already true (commit 7031cb1)

- Fast path of `launch_compute_kernel_with_solution` pins
  `DeviceGuard::set(self.ordinal)` around `hipModuleLaunchKernel`
  (P1-3 discipline, mirroring the rocBLAS-handle fix).
- `DeviceGuard::set` emits `[ctx-trace]` + forced backtrace when targeting
  ordinal 2 under `GRIM_ALLOC_TRACE`; launches stamp `self_dev`/`ctx_dev`.
- Backtrace evidence so far: every *traced* `set(2)` comes from
  `CapabilityProfiler::new → measure_capability → probe_host_gpu`, which
  restores correctly (`prev=0`). The setter that flips the main thread to
  ctx=2 **between transpose completion and the embedding launch** has never
  been caught by that hook — meaning it either does not go through
  `DeviceGuard` at all, fires on a path the gate missed, or is a raw
  `hipSetDevice` inside a helper. That is WI-M1/M2's target.

## WI-M1 — Pin every raw HIP seam · gates everything

**Problem:** `storage.rs`'s three upload/download seams
(`copy_from_host`, `copy_from_host_raw_bytes`, the DtoH paths behind
`to_cpu_vec_f32`) call `hipMemcpy*` with **no DeviceGuard**, and the slow
(first-JIT) branch of `launch_compute_kernel_with_solution` loads modules
and calls `hipModuleLaunchKernel` unpinned. An unpinned H2D copy whose
thread context drifted allocates/writes on the wrong device — tensors end
up resident on the APU while every later kernel launches on ordinal 0.
This is the most plausible producer of the exact observed split.

**Changes**
- Wrap each seam body in `DeviceGuard::set(self_ordinal)` (allocator knows
  its ordinal; thread the device through or store it on the storage being
  written).
- Pin the slow launch branch identically to the fast path.
- Audit remaining bare `hipSetDevice` sites (`peer_access.rs` manages its
  own prev/save pair — verify balanced; `RocmDevice::try_new` init path).

**Gates**
- `[host]` `grep` contract test: no `hipSetDevice(` outside `util.rs`/
  `handles.rs`/`peer_access.rs` save-restore pairs (source-level assert,
  pattern already used by the attention structural gate).
- `[host]` GRIM_ALLOC_TRACE rerun of LFM2.5-230M multi-device: zero lines
  with `ctx_dev != self_dev`.

## WI-M2 — Catch the drift source red-handed

**Problem:** the traced `set(2)` hook only fires for guarded calls. The
mid-forward flip was not caught ⇒ either an unguarded `hipSetDevice`, a
guard dropped out of order across threads, or a constructor
(`RocmDevice::shared`, stream pool init) that sets the device permanently
on first use from a worker thread whose "prev" was another device's.

**Changes**
- Extend `[ctx-trace]` to fire for ANY `DeviceGuard::set(n≠0)` while a
  process-wide "prefill started" latch is set (not just ordinal 2), and add
  the same trace to a new `raw_set_device()` helper that WI-M1 introduces
  for the two legitimate unguarded callers.
- Record `std::thread::ThreadId` next to prev in the trace line, so
  cross-thread flips become obvious.

**Gates**
- `[host]` Multi-device harness run names the flipping frame (backtrace in
  the log). The flip must be attributable to a named function before WI-M3
  starts coding around it.

## WI-M3 — Context-correctness unit gates

**Changes**
- New device-gated test (gfx1201 box): spawn a worker thread, park its
  context on device 1 via `DeviceGuard::set(1)` held alive, then from the
  MAIN thread upload a tensor and launch `grim_rms_norm` through the public
  API; assert output correctness AND `ctx_dev == 0` at the launch seam via
  the trace hook compiled under `#[cfg(test)]`. Repeat with roles swapped.
- Same harness for `copy_from_host_raw_bytes`: allocate under drifted
  context, assert the storage lands on the intended ordinal
  (`device_ordinal()` check) rather than the context's.

**Gates**
- `[host]` Both tests green single-threaded; they must FAIL when the WI-M1
  pins are reverted (mutation check, run once manually).

## WI-M4 — Acceptance matrix and the road back to WI-INF4

**Gates**
- `[sb]` Differential matrix on syd-beasty, LFM2.5-VL-3B-Q4_K_M, caches
  purged, iters ≥ 2:
  `{ROCR=0}`, `{ROCR=0,1}`, `{all visible}` × `{arm=off}` — zero
  "Page not present", all samples carry `ttft_ms`.
- `[sb]` Then arm=on legs over the discrete pair (farm mode) — this is the
  step blocked since 2026-08-22; verdict rule as defined in
  `scythe2_syd_beasty_plan.md` WI-INF4.
- Any residual fault after M1–M3 ⇒ the flip is driver-side; file with
  AMD alongside the rocprofiler trace captured during M2 instead of
  patching further.

## Sequencing

```
M1 ──► M4 (matrix)
 │
M2 ──► M3 ─┘   (M2 names the frame; M3 locks it as a regression)
```

M1 is mechanical and removes the failure class even without naming the
exact setter; M2/M3 exist so the class cannot silently return.

## Checkbox ledger

| WI | host | sb | done |
|----|------|----|------|
| M1 pin all seams + slow launch | ☑ grep-contract test (`tests/hip_context_contract.rs`, 3/3) | ☑ GRIM_ALLOC_TRACE rerun: 0 `ctx_dev≠self_dev` lines across `{ROCR=0}`, `{ROCR=0,1}`, all-visible legs (289+ traced launches; see also the event-seam extension below) | ☑ |
| M2 name the drift frame | ☑ named-frame log: every latch-active flip attributed to `CapabilityProfiler::new → measure_capability` (probe/vram/clock/calibrate), balanced save/restore — no unattributed setter remains | — | ☑ |
| M3 context-drift unit gates | ☑ 4/4 green single-threaded in BOTH `GRIM_GPUS=0,1` and `GRIM_GPUS=1,0`, incl. mutation check: guard no-op'd → raw_bytes residency + roles_swapped FAILED; pins restored → green. Two test defects fixed en route (see validation log 2026-08-23b) | — | ☑ |
| M4 acceptance matrix | — | ☑ off-legs clean: `{ROCR=0}` exit 0, `{ROCR=0,1}` exit 0, `{all visible}` exit 0; 72 honest per-request samples, zero "Page not present", zero panics | ☑ |
| WI-INF4 verdict | — | ☑ recorded: **stays opt-in** — mean TTFT overhead ≈0% both orders, p95 ITL −18.6% (F) / +2.43% (S, fails ≤2% budget); cost model retune pending (see scythe2 plan §validation log 2026-08-23c) | ☑ |

## Implementation status (2026-08-23)

M1/M2/M3 code is landed in `grim-backend-rocm` + `grim-engine`:

- **M1 pins** — `DeviceGuard::set(owning ordinal)` now wraps: the three
  `storage.rs` upload/download seams (`copy_from_host`,
  `copy_from_host_managed`, `copy_from_host_raw_bytes` full body; the two
  direct `hipMallocManaged` branches in `alloc_gpu_with_bytes`; all DtoH
  paths in `to_cpu_vec_f32`), `prefetch_to_device`, `RocmStorage::drop`;
  allocator `alloc`/`free` real-release paths (`empty_cache` was already
  pinned); `upload_to_scratch` H2D and the comm_fuse D2D assembly;
  `memcpy_with_xnack_fallback`; `upload_device_buffer` gained an
  `ordinal` parameter (all ~30 call sites pass `self.ordinal`). The slow
  JIT branch of `launch_compute_kernel_with_solution` was already pinned
  (e3b13a538); verified no gap remains between module load and launch.
- **raw_set_device** — new traced setter in `device/util.rs`.
  `RocmDevice::try_new` routes through it and is now *context-neutral*
  (saves/restores the caller's device instead of parking it on `ordinal`
  forever — a constructor flip mid-forward was a prime drift suspect).
  `peer_access::enable_peer_access`'s save/restore pair also routes
  through it (balance asserted by contract test).
- **M2 tracing** — `[ctx-trace]` centralized: fires for any
  `DeviceGuard`/`raw_set_device` switch to n≠0 while the engine's prefill
  latch is up (`set_prefill_in_flight`, wired around `Engine::drive_prefill`),
  plus the legacy ordinal-2 TEMP-DIAG; every line carries
  `prev=` + `tid=` (ThreadId) + forced backtrace. Launch seams stamp
  `self_dev`/`ctx_dev` on every launch.
- **M3 gates** — `src/context_drift_tests.rs`: worker thread parks its
  context on a foreign ordinal while main uploads + launches `grim_rms_norm`
  through the public API (asserting output correctness AND
  `last_launch_context() == (self, self)`), roles swapped, plus a
  `copy_from_host_raw_bytes` residency gate using per-device free-VRAM
  deltas. Device-gated (`GRIM_GPU_TEST=1`, ≥2 HIP devices).

Verified on this box (single gfx1036 iGPU): `cargo check -p
grim-backend-rocm -p grim-engine --tests` clean; grep-contract 3/3;
latch smoke + drift tests green (self-skipped, need ≥2 devices);
live GPU runs green with `GRIM_ALLOC_TRACE=1` — caching-allocator reuse
test and `fused_add_rms_norm_tests` show `self_dev=0 ctx_dev=0` for every
launch, zero `[ctx-trace]` events.

**Remaining (needs syd-beasty / ≥2 devices):** the M1 trace rerun
(zero `ctx_dev != self_dev` under LFM2.5 multi-device), M2's named-frame
capture, M3's manual mutation check (revert pins → tests must fail), and
the whole M4 matrix incl. WI-INF4.

## Known pre-existing GPU flake (discovered 2026-08-23, NOT caused by M1–M3)

`tests/mxfp4_gemm_tests.rs::test_fused_mxfp4_gemm_qk_norm_rope_kv_parity`
intermittently produces **all-zero** Q/K/V ("Q mismatch at 0: actual=0,
expected=1.2004437") on gfx1036. Evidence it predates the context-discipline
work:

- A pristine `git worktree` at cf4ea0d2 fails with the **byte-identical**
  panic (`Q mismatch at 0: actual=0, expected=1.2004437`) on its first GPU
  run after a fresh JIT compile.
- Pure-HEAD main-tree runs passed 10/10 when reusing warm hsaco cache
  entries, but fail after cache purge at varying rates — the flake
  correlates with freshly compiled aggregate modules + immediate single-shot
  execution, i.e. host/GPU timing, not input data (uploads are synchronous
  `hipMemcpy`; both pipeline kernels launch on stream-pool slot 0, so the
  launches themselves are ordered).
- `test_rocm_quantize_fp8_roundtrip` showed one identical-class one-off
  failure and then passed 5/5; no FP8 path is touched by M1–M3.

The WI-M1 guard additions add a handful of host-side FFI calls per op,
which shifts this race's hit rate (~50% on the modified tree vs occasional
on HEAD under the same load); that sensitivity is why the Drop-path guard
was deliberately left out of `RocmStorage::drop` (see storage.rs note) and
why `emit_ctx_trace` early-returns on an atomic check before any env
lookup. Root-causing the underlying first-launch zeroing belongs to the
same fault hunt as the ctx_dev=2 page fault — recommended next step:
capture `rocprofv3` traces of a failing run per the M4 toolchain.

## Validation log (continued)

- 2026-08-23b (syd-beasty, session 2): M1–M4 gates closed. En-route fixes,
  all verified on-box:
  - **Event seams** — `upload_from_host_stream_ordered` /
    `copy_from_host_async` ran unpinned; the cached `upload_event` created
    under a drifted context poisoned every later record →
    `hipEventRecord failed with code 400` when farm replica `#scythe1`
    loaded in release. Both bodies now DeviceGuard-pinned; farm arms 2/2.
  - **Launch stamp order** — `(self_dev, ctx_dev)` was stamped at function
    entry, *before* the P1-3 pin, so WI-M3's gates measured ambient context
    rather than execution context (`roles_swapped` false-failed with
    ctx=0). Stamp now taken inside each launch path under the held guard.
  - **Drift-test helper bug** — `upload_and_launch_rms_norm` hardcoded its
    own `(i%17)` input while `roles_swapped` compared against an `(i%13)`
    reference: that gate could never pass on any hardware. The "mismatch"
    values were the helper's data correctly normalized. Helper now takes
    the caller's data. (The raw_bytes VRAM-delta gate additionally
    requires single-threaded test execution — parallel fixtures race the
    free-VRAM baseline.)
  - **HIP attribute constants** (root cause found while implementing
    scythe2 SB0): handles.rs/probe.rs used CUDA numbering, so WARP_SIZE=24
    read ManagedMemory (=1), MULTIPROCESSOR_COUNT=16 read Integrated,
    MAX_THREADS_PER_BLOCK=1 read AccessPolicyMaxWindowSize, LDS=3 read
    CanMapHostMemory, PAGEABLE=231 did not exist (always errored), and the
    profiler's "throttle" attr 74 actually selected
    MaxSharedMemoryPerBlock (~64 KB) which clamped to throttle_pct=1.0 and
    **zeroed every GPU's effective TFLOPS** whenever the profiler ran.
    Corrected to ROCm 7.x enum values verified live: warp 87→32(wave32),
    clock 13→5 (2570/2620 MHz), CUs 16→63 (WGP count), pageable
    231→65, LDS 3→74 (64 KB). Honest 0.0 replaces the fake throttle.
  - **LFM2 fused-KV growth** — `LFM2_FUSED_KV_CACHE_LEN = 4096` was a hard
    cap against a 128k-context model; decoding past position 4096 panicked
    `lfm2.rs:724` (range end 2097664 > len 2097152). The K/V scratch now
    grows (doubling) with `cache_offset + steps`; the M4 matrix runs clean.

## Risks

- Per-thread semantics mean fixes can look green single-threaded and break
  under the engine's rayon workers — M3's tests must cover a second thread
  holding a foreign guard concurrently, not just sequential drift.
- Blanket pinning adds guard churn per op; keep `DeviceGuard` (two hipCalls)
  and avoid `hipDeviceSynchronize` in the hot seams.
- The APU (gfx1036, 2 GB, unified memory) is a legitimate HIP device for
  probes; do not hide it globally — pin call sites instead, or laptop/
  iGPU-only boxes lose their backend.
