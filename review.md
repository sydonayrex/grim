# review.md — Validation of `exploded.md` against the grim codebase

Scope: every claim, type, file, and work item in `exploded.md` checked against
HEAD of `/D/rex/projects/grim`. Verdict per item: CONFIRMED / INCONSISTENT /
ALREADY EXISTS / NEEDS MODIFICATION.

Evidence is cited as `path:line`. No code was changed.

---

## 0. Verdict summary

| # | Finding | Severity |
|---|---|---|
| F1 | `charon.rs` already ships the whole "still missing" fused-MoE WMMA kernel family (7 `__global__` entries incl. fp8/mxfp4/mxfp8/q8_0/iqk). Plan proposes creating them from scratch. | **Blocker** |
| F2 | `arch.rs` duplicates `probe::HostGpuCapabilities` (gcn/wave/LDS), `quantization::GcnArch`, and `accel_features::{wmma_supported, mfma_supported}`. | **Blocker** |
| F3 | `fp8_wmma.rs` / `mxfp8_wmma_dequant.rs` / `mxfp_weights.rs` duplicate `fp8_gemm_rdna4.rs`, `wmma_gemm.rs`, `fp8_standalone.rs`, `mxfp_standalone.rs`, and `grim-quant` fp8/mxfp codecs. | **Blocker** |
| F4 | `KernelArch::has_fp8_wmma = gfx1200+` contradicts shipped `wmma_supported()` (RDNA3 **and** RDNA4) and `wmma_gemm.rs` MFMA-on-gfx1200 comments. | High |
| F5 | `fp8_is_fnuz() -> true` for "all AMD ROCm targets" contradicts the repo: `grim-quant` implements **OCP e4m3** (bias 7, `lib.rs:1646`), and `tests/quantization.rs:7` notes no native fp8 on the target. Plan's `fnuz` CI gate would fail. | High |
| F6 | `MoETaskDescriptor` doc says "64 bytes" but declares `#[repr(C, align(32))]`; field list sums to 60 B → 64 B actual. Stated gate ("≥ 32 bytes and align(32)") is vacuous and doesn't check the 64-byte ABI the persistent kernel needs. | Medium |
| F7 | `ScytheTaskDescriptor` is 52 B rounded to 64 B (`scythe2.rs:582-584`) but the module doc still says "32-byte descriptors" (`scythe2.rs:26`, `:547`). exploded.md propagates the wrong 32-byte figure. | Medium |
| F8 | `ScytheRing::enqueue` does **not** write the descriptor to device memory — it only bumps `head` and drops `desc` (`scythe2.rs:645-655`). exploded.md's §3.2.1 and §3.6.1 assume a working ring protocol. | **Blocker** |
| F9 | Crate-dependency inversion: `charon_wmma.rs` (in `grim-backend-rocm`) is written to `use` `ScytheRing` from `grim_engine`, but `grim-engine` depends on `grim-backend-rocm` (`crates/grim-engine/Cargo.toml:14`). Cyclic. | **Blocker** |
| F10 | Peer/P2P "already covered" claim is only half true: `ScytheLink` lives in `grim-tensor`, but the real staging machinery is `p2p_route.rs` (`RouteLink`, `HostStagingBuffer`, `copy_route`) which the plan never mentions or reconciles. Two parallel link enums. | High |
| F11 | Opcode 5 = "CommFuse reduce" is a doc-only opcode. `comm_fuse.rs` exposes `comm_fuse_fan_in()` as a **host** function over CPU `&[f32]` slices — not a ring-dispatched device path. | High |
| F12 | No persistent kernel exists. Grep for `persistent` finds only doc comments + `selective_scan.rs` header. §3.6.1's "add opcode 6 to the dispatch loop" edits a file that does not exist. | **Blocker** |
| F13 | `GpuCapability` has no `vram_free`/`tflops_fp8` naming as written in §1 table (`vram_free_bytes`, `hbm_bandwidth_gbps`). Cosmetic but the plan's field list is not copy-pasteable. | Low |
| F14 | `Error::Backend(...)` used in sample code is correct (`grim-tensor/src/error.rs:22`); `use grim_tensor::error::{Error, Result}` is correct. | CONFIRMED |
| F15 | `minibatch_group_size` formula is misimplemented vs. its own doc comment: doc says `min(2*inter, hidden)`, code writes `(2*inter).min(hidden)` — same thing — but uses `comm_cu` as `C`, while MoK's `C` is the *compute* CU count. Off by ~2×. | Medium |
| F16 | `charon.rs` already has an adaptive variant selector (`CharonSelector`, `WaveCostModel`, `default_variant_table`, `routing_skew`) that overlaps the plan's WI-I occupancy tuning. Plan doesn't mention it. | High |
| F17 | Backward pass (WI-D) is genuinely missing for MoE, but `tests/quant_backward_gpu.rs` and `golden_silu_backward.rs` exist — plan should extend, not greenfield. | Medium |
| F18 | `shared_expert` already modelled end-to-end (`grim-nn/src/moe.rs:252`, `moe_block.rs:34/95`, `laguna.rs:97`). The plan's `shared_expert_mask_ptr` needs to be *derived from* that, not invented. | Medium |
| F19 | `routed_scaling_factor` already exists in `grim-nn/src/moe.rs:253` with defined semantics (applied to routed sum only). The descriptor field must document the same semantics or results diverge. | Medium |
| F20 | Timeline "15–23 days, no duplicated engine work" is not supportable: ≥ 60 % of Phase 1–2 is already implemented in `charon.rs` + quant kernels; the real remaining work is ring/persistent-kernel plumbing, which the plan under-budgets (F8/F12). | High |

---

## 1. §1 "What `scythe2.rs` Already Ships" — mostly CONFIRMED

| Claim | Status | Evidence |
|---|---|---|
| `C2plrController` exists | CONFIRMED | `crates/grim-engine/src/scythe2.rs` (861 lines), `decide()` used in tests `:703` |
| `PlacementCache` fast/slow path, epoch invalidation | CONFIRMED | `scythe2.rs:65-140` |
| `ScytheRing` + `ScytheTaskDescriptor` | **PARTIAL** | Types exist `:561-661`, but `enqueue` is a counter bump only — see F8 |
| `ScytheLink` P2P classification | CONFIRMED but misattributed | Defined in `crates/grim-tensor/src/backend.rs:27`, *not* in `scythe2.rs`. Also see F10. |
| `ScythePlacement` | CONFIRMED, misattributed | `grim-tensor/src/backend.rs` |
| `GpuCapability` fields | CONFIRMED w/ wrong names | `grim-tensor/src/backend.rs:9-23` — see F13 |

**Correction to the plan's framing:** the load-bearing types are in
`grim-tensor::backend`, consumed by `grim-engine`. Anything in
`grim-backend-rocm` can already `use grim_tensor::backend::{ScytheLink,
ScythePlacement}` (as `comm_fuse.rs:3` does) with no cycle. Only
`ScytheRing`/`ScytheTaskDescriptor` are engine-local, and that is precisely
what creates F9.

---

## 2. §2 "Still genuinely missing" — largely FALSE

### 2.1 `charon_wmma.rs` — ALREADY EXISTS as `kernels/charon.rs` (1853 lines)

Shipped `__global__` entries (`charon.rs`):

```
:67   grim_moe_fused_dispatch
:168  grim_moe_fused_grouped
:259  grim_moe_fused_grouped_fp8
:347  grim_moe_fused_grouped_mxfp4
:408  grim_moe_fused_grouped_mxfp8
:481  grim_moe_fused_grouped_q80
:740  grim_moe_fused_grouped_iqk
```

Shipped host plumbing: `RoutingAssignment` `:800`, `SortedRouting` `:867`,
`moe_align_block_size` `:887`, `CharonLaunchPlan` `:966`, `choose_block_dim`
`:979`, `plan_fused_dispatch` `:1001`, `plan_grouped_dispatch` `:1033`,
`validate_grouped_inputs` `:1047`, and a live launcher
`RocDevice::launch_charon_fused_dispatch` (`device/roc_device.rs:4100`) with a
golden GPU test (`tests/golden_charon_moe_gpu.rs`).

**Action required:** Phase 1 must be rewritten as *"extend `charon.rs` with a
ring-dispatch entry point"*, not *"create `charon_wmma.rs`"*. Creating a second
kernel module would fork the quant-variant matrix that
`grim-moe-quant-kernels` already verified on gfx1036.

Note the plan's own `launch_fused_dispatch` signature omits `RoutingAssignment`
/ `SortedRouting` entirely and takes only a descriptor — incompatible with the
existing validated launch path.

### 2.2 `arch.rs` — DUPLICATES three existing modules

| `KernelArch` field | Already exists |
|---|---|
| `gcn_arch_name` | `probe::HostGpuCapabilities.gcn: String` (`device/probe.rs:78`), `quantization::gcn_arch()` `:25`, `GcnArch` enum `:7-22` |
| `wave_size` | `HostGpuCapabilities.wavefront_size` (`probe.rs:80`), queried via `hipDeviceGetAttribute` `:99` |
| `lds_bytes` | `HostGpuCapabilities.lds_size_bytes` (`probe.rs:82`) |
| `has_fp8_wmma` | `accel_features::wmma_supported(arch, QuantMode::Fp8Native)` `:50`, `wmma_dispatch` `:64` |
| `num_cu` | **genuinely missing** — no `multiProcessorCount` query anywhere in the crate |
| `block_dim()` | `charon::choose_block_dim(num_pairs, wave_size)` `:979` already does wave-aligned 4-wave capping |

**Action required:** delete `arch.rs` from the plan. Add exactly one field
(`num_cu`) to `HostGpuCapabilities` plus a `cu_partition()` free function.
`[u8; 16]` for the arch name is also a regression from the existing `String`
and would truncate `gfx1100:xnack-` style names.

`fp8_is_fnuz()` as a `const fn` with no arch parameter is wrong on two counts
(F5, and it's arch-independent by construction).

### 2.3 `fp8_wmma.rs`, `mxfp8_wmma_dequant.rs`, `mxfp_weights.rs` — DUPLICATES

Existing: `kernels/fp8_gemm_rdna4.rs` (111 L), `kernels/wmma_gemm.rs` (408 L,
includes gfx1200 MFMA fp8 path `:311-324`), `kernels/fp8_standalone.rs`
(`grim_dequant_fp8`), `kernels/mxfp_standalone.rs` (`grim_dequant_mxfp4`,
`grim_dequant_mxfp8`), plus in-kernel MXFP8 dequant inside
`grim_moe_fused_grouped_mxfp8` (`charon.rs:408`).

Host codecs: `grim-quant::{dequant_fp8, dequant_mxfp4, dequant_mxfp8,
quant_fp8, f32_to_fp8_e4m3, fp8_e4m3_to_f32}` (`grim-quant/src/lib.rs:1088,
1159, 1223, 1631, 1646, 1331`), with device-side host wrappers
`RocDevice::dequantize_mxfp{4,8}_host` (`roc_device.rs:6865/6869`) and parity
tests (`tests/standalone_dequant_parity.rs:222/234`).

**Genuinely missing:** an MXFP8 **quantize** (f32 → codes+shared-exp) path.
`grim-quant` has `quant_fp8` but no `quant_mxfp8`/`quant_mxfp4`. That is the
only defensible part of `mxfp_weights.rs`.

### 2.4 MoE task descriptor — CONFIRMED missing

`MoETaskDescriptor` / `MoEDispatchArgs` do not exist anywhere (grep: 0 hits
outside `exploded.md`). This is the one Phase-0 item that survives review, with
the caveats in F6/F18/F19.

---

## 3. §3 Implementation plan — item-level findings

### 3.1 Phase 0

- **`arch.rs`** → drop (F2). `from_hip()` is a `todo!()` in a plan document,
  while `probe_host_gpu()` already does the real query.
- **`moe_descriptor.rs`** → keep, but:
  - fix the size doc (60 B fields → 64 B with `align(32)`), and make the gate
    `assert_eq!(size_of::<MoETaskDescriptor>(), 64)` rather than `>= 32`;
  - `quant_mode: u32` with 3 values duplicates the existing `QuantMode` enum
    used by `accel_features` — reuse it via a `#[repr(u32)]` mapping instead of
    a bare integer, otherwise the kernel and the dispatch gate can disagree;
  - the variant matrix in `charon.rs` covers **six** quant paths (bf16, fp8,
    mxfp4, mxfp8, q8_0, iqk); a 3-value `quant_mode` cannot express it.
- **`schedule.rs`/`MoEDispatchArgs`** → keep, but `_pad: u32` after a `u64`
  field leaves the struct at 8-byte alignment with a trailing 4-byte hole and
  no `repr(align)`; field order should put `num_batches`/`_pad` last as a pair,
  and the struct needs a size assertion. Also note `SortedRouting` +
  `moe_align_block_size` (`charon.rs:867/887`) already produce
  `sorted_token_ids` / `sorted_expert_ids` / `sorted_weights` /
  `num_tokens_post_padded` — `MoEDispatchArgs` should be defined as the device
  view **of those exact arrays**, not a new layout.

### 3.2 Phase 1

- `launch_fused_dispatch(..., ring: Option<&ScytheRing>)` — **cyclic dep**
  (F9). Fix: define a `RingEnqueue` trait in `grim-tensor::backend` (where
  `ScythePlacement` already lives) and have `grim-engine` implement it; the
  ROCm crate depends only on the trait.
- The three "corrections carried forward" reference a prior kernel source that
  is not the shipped `charon.rs`. The `__syncthreads()` / `WAVES_PER_BLOCK` /
  `atomicAdd` claims must be re-derived against `charon.rs:67-780`, otherwise
  they are fixes to code that does not exist in this repo.
- **WI-I `minibatch_group_size`**: see F15 (uses `comm_cu` where MoK uses
  compute CUs). Also overlaps `CharonSelector::select` (`charon.rs:1373`) which
  already picks a variant from `WaveCostModel::predict` + `routing_skew`. Two
  competing heuristics will fight.
- Gate "single-expert correctness vs. rocBLAS" — `tests/golden_charon_moe_gpu.rs`
  already exists; extend it rather than adding a parallel CI job.

### 3.3 Phase 2

- `fnuz` gate is unimplementable as stated (F5). The repo's fp8 is OCP e4m3
  (`f32_to_fp8_e4m3`, exponent bias 7, `lib.rs:1660`). Either (a) add an
  explicit fnuz codec and a conversion, and say so, or (b) drop the fnuz claim.
  Silently asserting "bit-identical to on-AMD requantization" against an OCP
  codec will fail.
- Super-Expert guard: the per-batch branch is sound, but the mask source must
  be `MoeFfn::shared_expert` (`grim-nn/src/moe.rs:252`) / `MoeSpec.has_shared_expert`
  (`moe_block.rs:34`). The plan implies the engine's `C2plrController` invents
  it; the controller has no expert-level state today (it decides layer
  placement only).
- "PPL 8.70 → 59.86" is an unsourced external claim carried into an internal
  plan with no repo-side measurement path. Either cite it or cut it.

### 3.4 Phase 3

MoE backward is genuinely absent. But `save_intermediates` does not exist as a
flag anywhere, and `grim-autograd` has no MoE hook. Budget must include the
autograd wiring, not just the kernel. Determinism gate is realistic.

### 3.5 Phase 4

- `hipExtLaunchMultiKernelMultiDevice` is named as the CU-masking mechanism;
  it is not a CU-masking API (it launches on multiple devices). The relevant
  ROCm surfaces are `hipExtLaunchKernelGGL` with CU masks via
  `hipStreamCreateWithCUMask` / `hipExtStreamCreateWithCUMask`. As written the
  step is not actionable.
- Gate "Profiling (MI300X)" — the project's stated target is **gfx1036 /
  RDNA2, 24 GB**. No MI300X in the repo's test matrix. Gate is unrunnable here.
- §3.5.2 "no new `peer_mapping.rs` needed" is right, but for the wrong reason:
  the covering code is `p2p_route.rs` (`RouteLink` `:42`, `HostStagingBuffer`
  `:91`, `copy_route` `:190`, `tests/p2p_route.rs`), not `ScytheLink` alone.
  `RouteLink` and `ScytheLink` are two enums for the same concept and the plan
  should call out the reconciliation (F10).

### 3.6 Phase 5

Blocked on F12: there is no persistent kernel to add opcode 6 to. The
prerequisite work item — "implement the device-resident poll loop and make
`ScytheRing::enqueue` actually write to `slots_device_ptr`" — is missing from
the plan entirely and is the single largest unbudgeted item.

Test-matrix rows "already tested in `scythe2.rs`" are accurate
(`scythe2.rs:693` cache-hit test), but the ring latency test measures an
enqueue that performs no device write (F8) — it is not evidence of a <100 ns
dispatch.

---

## 4. §4 "Deleted work items" — validation

| Deletion | Verdict |
|---|---|
| `schedule.rs` controller logic | CORRECT — `C2plrController` + `PlacementCache` cover it |
| `peer_mapping.rs` / `PeerWorkspace` | CORRECT outcome, wrong justification (F10) |
| `ring_buffer.rs` (WI-H) | **WRONG** — `ScytheRing` is a host-side counter, not a working VRAM ring (F8). Deleting WI-H removes the only work item that would make opcode 6 real. |
| SM partitioning in `arch.rs` | Partially correct; `cu_partition` should land next to `HostGpuCapabilities`, not in a new `arch.rs` |
| Standalone `hipLaunchKernel` demoted to fallback | **WRONG for this repo** — the direct path is the *only* working path today (`roc_device.rs:4100`, JIT via hipRTC `device/util.rs:94`). Demoting it before the ring works inverts the risk order. |

---

## 5. §5 File creation order — revised

Recommended replacement:

```
1. probe.rs            MODIFY  + num_cu (hipDeviceAttributeMultiprocessorCount)
                               + cu_partition()
2. grim-quant          ADD     quant_mxfp8 / quant_mxfp4 (host codec, the only
                               genuinely missing quant direction)
3. moe_descriptor.rs   NEW     MoETaskDescriptor (64 B, size-asserted) +
                               MoEDispatchArgs as the device view of
                               SortedRouting's existing arrays
4. grim-tensor/backend NEW     RingEnqueue trait (breaks the engine↔rocm cycle)
5. grim-engine/scythe2 MODIFY  make enqueue() actually write the descriptor to
                               slots_device_ptr; add opcode 6 constant
6. persistent_kernel   NEW     device poll loop (UNBUDGETED IN PLAN — this is
                               the critical path, not Phase 0)
7. charon.rs           MODIFY  add ring-dispatch entry alongside the existing
                               direct launcher; reuse the 7 existing __global__
                               entries and CharonSelector
8. charon_backward     NEW     the one true greenfield kernel + autograd wiring
```

Deleted from the plan: `arch.rs`, `charon_wmma.rs`, `fp8_wmma.rs`,
`mxfp8_wmma_dequant.rs`, `schedule.rs` (as a file), `mxfp_weights.rs` (reduced
to a `grim-quant` addition).

---

## 6. §6 Timeline — not supportable

The plan's 15–23 days assumes Phases 1–2 are greenfield. They are ~60–70 %
shipped (F1, F3). Conversely Phases 4–5 assume a persistent kernel and a
functioning ring that do not exist (F8, F12) and are not on the schedule at
all. Net effect: the estimate is misallocated rather than merely wrong —
re-baseline with the ring/persistent-kernel work as Phase 1 and the WMMA
kernels as a modification pass.

The closing claim "no duplicated engine work" holds for the *engine*; it does
not hold for the *kernel crate*, which is where the duplication actually is.

---

## 7. Minimum set of edits to make `exploded.md` truthful

1. Replace §2 with a delta table against `charon.rs`, `fp8_gemm_rdna4.rs`,
   `wmma_gemm.rs`, `fp8_standalone.rs`, `mxfp_standalone.rs`, `probe.rs`,
   `accel_features.rs`, `p2p_route.rs`.
2. Delete `arch.rs`; replace with "add `num_cu` + `cu_partition()` to
   `HostGpuCapabilities`".
3. Add a Phase-1 work item: *make `ScytheRing::enqueue` write to device memory*
   and *implement the persistent poll kernel*. Un-delete WI-H.
4. Fix descriptor sizing: state 64 B, assert it.
5. Resolve the fp8 format question (fnuz vs. the shipped OCP e4m3) before any
   Phase-2 gate is written.
6. Break the `grim-engine` ↔ `grim-backend-rocm` cycle via a trait in
   `grim-tensor`.
7. Re-target Phase-4 gates from MI300X to gfx1036, or mark them explicitly
   out-of-scope-for-CI.
8. Reconcile `RouteLink` and `ScytheLink`; state which is canonical.
9. Reconcile `minibatch_group_size` with `CharonSelector` — one heuristic, not
   two.
10. Expand `quant_mode` to cover the six shipped variants or reuse `QuantMode`.
