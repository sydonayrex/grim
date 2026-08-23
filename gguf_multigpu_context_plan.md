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
| M1 pin all seams + slow launch | ☐ grep-contract test | — | ☐ |
| M2 name the drift frame | ☐ named-frame log | — | ☐ |
| M3 context-drift unit gates | ☐ incl. mutation check | — | ☐ |
| M4 acceptance matrix | — | ☐ off-legs clean | ☐ |
| WI-INF4 verdict | — | ☐ farm on/off both orders | ☐ |

## Risks

- Per-thread semantics mean fixes can look green single-threaded and break
  under the engine's rayon workers — M3's tests must cover a second thread
  holding a foreign guard concurrently, not just sequential drift.
- Blanket pinning adds guard churn per op; keep `DeviceGuard` (two hipCalls)
  and avoid `hipDeviceSynchronize` in the hot seams.
- The APU (gfx1036, 2 GB, unified memory) is a legitimate HIP device for
  probes; do not hide it globally — pin call sites instead, or laptop/
  iGPU-only boxes lose their backend.
