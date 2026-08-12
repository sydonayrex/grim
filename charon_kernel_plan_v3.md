# Charon Kernel Plan v3 — Consolidated & Source-Verified

Status: DRAFT. Every claim below re-verified directly against `crates.zip`
(this session's most recent full-workspace upload) before inclusion. No
compiler or GPU available in this sandbox — nothing here is compile- or
runtime-verified; that distinction is marked explicitly per item below,
not assumed.

This supersedes `charon_kernel_plan_v2.md` and folds in everything since
confirmed: the multi-GPU/`ScytheRing` integration path, the `C2plrController`
call-site bugs and their fix scope, real corrections from third-party review
documents (both accepted and rejected, with reasons), and the `modules.rs`
host-roundtrip findings that sit outside Charon proper but block it from
mattering end-to-end.

---

## 0. Ground truth, re-verified against this exact upload (not carried
   forward from memory)

| Claim | Status this pass | Where |
|---|---|---|
| Charon ships 7 forward kernel variants (dispatch, grouped, +fp8/mxfp4/mxfp8/q80/iqk) | **Confirmed**, all 7 present and WI-Charon-0 complete | `grim-backend-rocm/src/kernels/charon.rs`, grep count = 7 |
| MoE backward pass (`d_gate_w`/`d_up_w`/`d_down_w`) | **Built** (was absent in prior upload) | `grim-backend-rocm/src/kernels/charon_backward.rs` (UNTRACKED), F-D gradient tests in `tests/charon_backward_grad_check.rs` |
| `CharonSelector` alternating-challenger thrashing fix | **Confirmed present** (`challenger: Option<CharonVariant>` field, streak-reset logic) | same file, ~line 1343 |
| `routed_scaling_factor` scales routed sum not shared expert | **Confirmed fixed**, regression test present | `grim-nn/src/moe.rs`, `routed_scaling_factor_scales_routed_not_shared` |
| `LookaheadPredictor::predict()` respects `self.enabled` | **Confirmed fixed** | `grim-nn/src/moe.rs`, `if !self.enabled { ... }` |
| `C2plrController` call site: `layer_idx` and `links` hardcoded to 0 / all-`Host` | **FIXED** — WI-Charon-0 complete. `layer_id` now `micro_step % num_layers`; `links` now `build_link_matrix(num_gpus, probe_peer_link)` → `peer_access::peer_status` per ordered pair | `grim-garage/src/jobs.rs:2458-2465` |
| `ScytheRing`/`ScytheTaskDescriptor` opcodes 0-6 (incl. MoE opcode 6) | **Confirmed + extended**, opcode 6 = MoE dispatch, `ScytheTaskDescriptor` `#[repr(C, align(32))]` + `MoETaskDescriptor` mirror with `__align__(32)` | `grim-engine/src/scythe2.rs` (WI-Charon-3 complete) |
| `wmma_gemm.rs` already has backward-dequant-GEMM kernels for fp8/mxfp4/mxfp8 (dense-GEMM-shaped, not MoE-aware) | **Confirmed**, real precedent for WI-Charon-2's fragment setup reuse | `grim-backend-rocm/src/kernels/wmma_gemm.rs` |
| `comm_fuse_fan_in`/`grim_comm_fuse_p2p_epilogue` real, called from `RowParallelLinear` reduction | **Confirmed** (verified earlier this session against `roc_device.rs:3665`) | `grim-backend-rocm/src/kernels/comm_fuse.rs` |
| `Linear::forward` and `RmsNorm::forward` call `h.synchronize()?` post-dispatch | **Fixed** — sync dropped in both, replaced with `let _ = h;` (lazy-sync) | `grim-nn/src/modules.rs` ~line 500, ~673 |
| `Embedding::forward` calls `h.synchronize()?` post-dispatch | **Fixed** — sync dropped, matching Linear/RmsNorm; golden parity test added | `grim-nn/src/modules.rs` line 787 (`let _ = h;`) |
| `Rope::forward` unconditionally round-trips through `to_vec_f32()`/CPU trig loop/`from_cpu()` | **Fixed** — device path via `dev.rope()` already present; CPU round-trip only a fallback for CPU-bound tensors | same file, `Rope::forward` → `dev.rope(...)` branch (WI-Host-1 #1 complete) |
| `broadcast_bias` unconditionally round-trips through CPU | **Fixed** — device path via `dev.broadcast_bias()` already present; CPU path only for CPU tensors | same file, `broadcast_bias()` → `dev.broadcast_bias(...)` (WI-Host-1 #2 complete) |
| `pick_device_for_storage_device` heap-allocates a fresh `Box<dyn BackendDevice>` per op | **Fixed** — returns `Arc<dyn BackendDevice>`, CPU cached via `OnceLock`, ROCm via `RocmDevice::shared()` | `grim-nn/src/modules.rs:13-30` (WI-Host-1 #4 complete) |
| `Embedding::forward_to_device` round-trips `cpu_t.to_vec_f32()` for non-CPU targets when weight is CPU-local | **Known gap** (outside WI-Host-1 scope; same lazy-sync class) — only fires when weight isn't on device yet; common case (weight already target-side) skips the branch | `grim-nn/src/modules.rs:797-816` |

Everything in this table was re-checked directly against `crates.zip` in
this turn — not inherited from an earlier snapshot or a prior document's
claim without a fresh grep.

## 1. Why this consolidation exists

This session produced two prior Charon-specific plans (v1/v2) plus several
third-party review and "fixes" documents of wildly varying reliability —
some genuinely useful (`kernel2.md`, one version of `fixins_.md`), some
fabricated against file paths that don't exist (`gfixes.md` v1, `v2`
disguised with real-but-mischaracterized paths), one irrelevant
(`fixins_.md`'s Nextcloud boilerplate variant). The pattern worth stating
explicitly, since it's the operating principle for this whole document:
**a document's plausibility, formatting quality, or citation apparatus
(`[cite: 1]`-style markers included) is not evidence of accuracy — only a
fresh read of the actual source is.** Several documents this session had
correct-sounding structure and wrong content; at least one had unusual
phrasing and right content. This plan is built exclusively from re-verified
claims, not from any document's self-description of its own rigor.

## 2. Corrected architecture

```
grim-garage (training orchestration)          grim-nn::moe (router/dispatch, host-side)
  ctrl.decide(layer_idx, shape, caps, links,0)   MoeRouter::route()
        |  [WI-EP0: layer_idx and links           |
        |   both currently wrong — fix first]     v
        v                                    ExpertPlacementMap [WI-EP1, depends on WI-EP0]
  C2plrController (real, engine crate)             |
        |                                          v
        v                                    CharonDispatchPlan [WI-EP2]
  ScythePlacement { ranks, partition, routes }      | local pairs -> Charon kernel (unchanged)
                                                     | remote pairs -> p2p_route + ScytheRing
                                                     v
                              charon.rs (7 forward variants, real, tested)
                                     |
                    +----------------+----------------+
                    v                                 v
        WI-Charon-1: backward pass            WI-Charon-2: WMMA tensor-core
        (scalar-loop shape, mirrors                forward (tensor-core tiles,
        wmma_gemm.rs's dense backward               mirrors wmma_gemm.rs's
        kernels adapted to grouped-               fragment setup, reuses
        dispatch shape)                            CharonSelector for variant pick)
                    |                                 |
                    v                                 v
        WI-EP4: cross-GPU expert          [ScytheRing opcode 6, MoETaskDescriptor
        gradient combine (comm_fuse         companion struct — kernel2.md's
        pattern, RCCL, expert-scoped)       proposal, verified sound]
```

## 3. Work items

### WI-Charon-0 — Fix `C2plrController`'s only real call site (hard prerequisite)

**Why:** Every placement decision anything downstream makes — for MoE expert
placement or otherwise — currently runs on synthetic input. `layer_id`
hardcoded to `0` defeats `PlacementCache`'s per-layer keying entirely
(confirmed: the surrounding `for layer_idx in 0..hparams.num_layers` loop
in the same file has `layer_idx` sitting right there, unused by this call).
`links` hardcoded to all-`Host` means the real P2P topology detection layer
(`peer_access::peer_status`, confirmed real and correctly RDNA-gated in an
earlier pass this session) is never consulted — every decision assumes the
worst-case interconnect regardless of actual hardware.

**Where:** `grim-garage/src/jobs.rs`, ~line 2334-2357.

**What-to-build:**
1. Thread `layer_idx` from the existing training loop into `ctrl.decide(layer_idx, ...)`.
2. Replace the hardcoded `links` vector with a real pairwise `peer_access::peer_status(i, j)` probe across all ranks in `rank_contexts`, mapped to `ScytheLink` (structurally identical enum shape to `P2PStatus` — direct mapping, not new logic).
3. **Do not assume PCIe symmetry** even between identical GPUs — probe every ordered pair independently. Motherboard PCIe root-complex/switch topology can make `peer_status(0,1) != peer_status(1,0)` even for matched cards.

**Gates:** (1) unit test asserting distinct `layer_idx` values produce distinct `PlacementCache` lookups; (2) unit test with a mocked `peer_status` asserting `links` reflects real (non-uniform) topology for both a homogeneous and a mixed-GPU synthetic case; (3) compiles; (4) **device-gated**: one real training step on real multi-GPU hardware, confirm via logging that `links` is no longer uniformly `Host`.

### WI-Charon-1 — MoE backward pass

**Why:** Confirmed, unambiguous, still-open gap — zero backward-pass code
anywhere in `charon.rs` or any sibling file, re-confirmed this pass.

**Where:** New `grim-backend-rocm/src/kernels/charon_backward.rs`.

**What-exists (the real precedent, not "nothing to build on"):**
`wmma_gemm.rs` already ships three working backward-dequant-GEMM kernels
(`grim_fused_dequant_backward_gemm_{fp8,mxfp4,mxfp8}`) — dense-GEMM-shaped,
not MoE-dispatch-shaped, but a real, tested template for gradient
computation through dequantized weights on the exact formats Charon's
forward kernels already support.

**What-to-build:**
1. FP32 path first: `d_x`, `d_gate_w`, `d_up_w`, `d_down_w` via the standard
   MoE backward decomposition — `d_down_w` from `d_y ⊗ hidden`, `d_hidden =
   d_y @ down_w^T` split into `d_gate`/`d_up` via the SiLU derivative,
   `d_gate_w`/`d_up_w` from their respective outer products, `d_x`
   accumulated from both `d_gate @ gate_w^T` and `d_up @ up_w^T`.
2. **All four gradients must be implemented and gated explicitly by name**
   — this session previously caught a draft that implemented only
   `d_down_w`/`d_x` while claiming completeness; the gate below exists
   specifically to prevent that regression from recurring silently.
3. Router backward (through non-differentiable top-k/sigmoid-bias
   selection) is explicitly out of scope for this item — separate, harder
   problem, own work item once expert-weight gradients are proven.
4. Quantized-weight backward (mirroring the 5 quantized forward variants)
   is phase 2, following `wmma_gemm.rs`'s existing fp8/mxfp4/mxfp8 backward
   pattern once the FP32 base case is proven.

**Gates:** (1) gradient-check against `grim-autograd`'s CPU tape-based
backward on the existing `MoeFfn` reference, for all four gradients by
name — not "gradients pass," but `d_x`/`d_gate_w`/`d_up_w`/`d_down_w`
individually asserted; (2) compiles; (3) **device-gated, unverified in this
sandbox.**

### WI-Charon-2 — WMMA-accelerated forward dispatch (tensor-core path)

**Why:** Confirmed structural gap versus vLLM's `fused_moe_kernel`
(block-tiled, `tl.dot`-issued, tensor-core-routed) — Charon's 7 existing
variants are all scalar per-thread FMA loops, confirmed by direct kernel-
body read earlier this session and unchanged in this upload.

**Where:** New `grim-backend-rocm/src/kernels/charon_wmma.rs`, importing
rocWMMA fragment setup from `wmma_gemm.rs` rather than duplicating the
include/fragment boilerplate.

**What-to-build:**
1. Grouped (token-sorted, matching `grim_moe_fused_grouped`'s existing
   sort/pad contract) WMMA variant — real 16×16 tile GEMM for gate/up/down,
   gated behind `CharonSelector`/`CharonVariant` (confirmed real, confirmed
   fixed for the thrashing bug this session) as a new variant option, not
   a parallel selection mechanism.
2. FP32 first, matching WI-Charon-1's phasing; FP8/MXFP4/MXFP8/Q8_0/IQK WMMA
   variants follow the pattern `wmma_gemm.rs` already establishes for dense
   GEMM (`grim_wmma_gemm_fp8`, `grim_fused_dequant_gemm_mxfp4`, etc.) — this
   item's job is giving each of Charon's existing scalar quantized variants
   a tensor-core-accelerated counterpart, not inventing new quant handling.
3. Does not touch the sortless single-token path (`grim_moe_fused_dispatch`)
   — deliberately different design point for low-overhead decode-time
   dispatch; WMMA tiling doesn't help single-token batches.

**Gates:** (1) parity vs the existing scalar `grim_moe_fused_grouped`
kernel specifically (not just the CPU oracle) — isolates any tiling bug
from a router/combine bug; (2) compiles; (3) **device-gated**: measured
GMEM traffic and wall-clock vs the scalar grouped kernel — no asserted
multiplier, a measured one.

### WI-Charon-3 — `ScytheRing` opcode 6: MoE task descriptor integration

**Why:** Verified sound proposal (originally from `kernel2.md`, checked
against real source rather than trusted): `ScytheTaskDescriptor` genuinely
uses opcodes 0-5 exactly as documented, `peer_ptr` genuinely exists and is
genuinely unused by Charon today, and the real precedent (`comm_fuse`'s
opcode-5 CommFuse-reduce path) is exactly the pattern a new opcode-6 MoE
dispatch should follow. This replaces this plan's earlier (v2) proposal to
build a bespoke `CharonDispatchPlan` data structure from scratch — the
right integration point is one new opcode on infrastructure that already
exists, not a parallel mechanism.

**Where:** New companion struct (`grim-backend-rocm/src/kernels/moe_descriptor.rs`
or similar — exact placement TBD, does not need to live in `grim-engine`
since `ScytheTaskDescriptor.weight_ptr` can point to it) carrying the
MoE-specific fields the generic `m`/`n`/`k`/pointer-quad descriptor can't:
hidden dim, inter dim, batch count, routed-scaling factor, expert-bank
pointers, schedule pointer, quant mode.

**What-to-build:**
1. `MoETaskDescriptor` (or equivalent) sized/aligned to complement
   `ScytheTaskDescriptor`'s existing 64-byte-effective sizing, not
   duplicating its pointer fields where they already fit (`input_ptr`,
   `output_ptr`, `peer_ptr` map directly; only MoE-specific geometry needs
   new fields).
2. Persistent-kernel dispatch-loop extension: `if (desc.opcode == 6) { ...
   cast desc.weight_ptr to MoETaskDescriptor*, call the Charon kernel
   inline ... }` — no separate `hipLaunchKernel`, matching how opcodes 0-5
   already work.
3. Host-side: engine enqueues via the existing `ScytheRing` API (no new
   enqueue mechanism), populating the new descriptor from `ExpertPlacementMap`
   (WI-EP1) + router output.

**What this plan does NOT assume without checking first:** before building
`arch.rs`-style device-property detection (wave size, LDS budget, GCN arch
string) as new code, check whether `CapabilityProfiler`
(`grim-backend-rocm/src/device/capability_profiler.rs`, confirmed real
earlier this session) already covers it — a prior review document found
exactly this kind of already-exists gap in an earlier draft's account of
"missing" `arch.rs` functionality, and it's cheap to check before building.

**Gates:** (1) `MoETaskDescriptor` size/alignment assertions, compiles;
(2) integration test: engine enqueue → ring dispatch → kernel reads back
correct fields, host-testable structure with device-gated final dispatch;
(3) **device-gated** for the actual opcode-6 dispatch firing correctly on
real hardware.

### WI-EP1/EP2/EP3 — Multi-GPU expert placement and P2P dispatch/combine

Unchanged in substance from `charon_multigpu_plan.md`, now correctly
sequenced behind WI-Charon-0 (the placement-input fix) rather than assuming
`C2plrController`'s output was already trustworthy. Summary:
- **WI-EP1**: `ExpertPlacementMap` — which GPU owns which expert, built via
  `C2plrController::decide()` at expert granularity, capacity-proportional
  fallback tested under both homogeneous and mixed-GPU synthetic cases.
- **WI-EP2**: cross-GPU token dispatch planner — partitions `(token,
  expert)` pairs into local/remote, batches remote transfers by
  destination rank, reuses `peer_status → to_route_link → copy_via_route`
  unchanged. Revised per WI-Charon-3: should emit `ScytheTaskDescriptor`s
  (opcode 6) onto `ScytheRing` rather than a bespoke dispatch-plan type.
- **WI-EP3**: cross-GPU combine — activates Charon's existing but
  never-fired `peer_out`/`col_offset`/`n_total` kernel parameters, following
  `comm_fuse_reduce`'s exact device-assembly-plus-RCCL pattern (dtype-gated:
  F32 device path, CPU fallback for other dtypes, matching precedent
  exactly rather than inventing a new fallback rule).

### WI-EP4 — MoE training: cross-GPU expert gradient combine

**Why:** MoE training is in scope per explicit direction. Depends on
WI-Charon-1 (single-GPU gradients must be correct first — combining wrong
per-GPU gradients across GPUs just produces wrong gradients faster) and
WI-Charon-0 (real link topology needed to know which ranks can cheaply
combine directly vs. need host-bounce).

**What-to-build:** Expert-scoped gradient combine — only ranks that
actually touched a given expert this step participate in its
all-reduce/point-to-point sum, using `ExpertPlacementMap` to determine
membership; router gradient gets separate full-batch treatment (every
rank's tokens influence the router, unlike expert-conditional participation).
Reuses `RcclAllReduce`/`sum_gradients_device` — confirmed already real and
already used for LoRA gradient sync in the same `jobs.rs` file this item's
prerequisite (WI-Charon-0) also touches.

**Gates:** (1) single-GPU backward gradient-checked (inherited from
WI-Charon-1's gate, not re-derived); (2) compiles; (3) **device-gated**:
two-GPU synthetic case, one expert's tokens deliberately split across both
ranks, combined gradient matches single-GPU-all-tokens-at-once reference.

### WI-Host-1 — Fix host-roundtrip and forced-sync bugs outside Charon proper

**Why:** Charon can be perfectly correct and still not matter end-to-end if
the surrounding forward pass forces PCIe round-trips and CPU sync on every
layer, every token — confirmed real and currently present in this upload.
This isn't Charon's bug, but it's the bottleneck that determines whether
Charon's throughput gains are even visible in wall-clock terms, so it
belongs in this plan's scope as a dependency, prioritized by actual
runtime cost:

1. **`Rope::forward`** (highest priority — every layer × every token, the
   only one of these with per-token multiplicity inside the hot generation
   loop): replace the CPU trig loop with a native HIP kernel; keep
   activations resident in VRAM.
2. **`broadcast_bias`**: same CPU-roundtrip pattern, invoked on every
   `Linear::forward` call with a bias present — fix alongside RoPE using
   the same native-kernel approach.
3. **`Linear::forward`/`RmsNorm::forward`'s `h.synchronize()?` calls**: real
   pipeline stalls (not correctness bugs) — remove from internal forward-
   pass modules, retain synchronization only at the outer inference
   boundary before sampling.
4. **`pick_device_for_tensor`'s per-op `Box<dyn BackendDevice>` allocation**:
   lowest-impact of the four (heap churn, not PCIe-bandwidth-bound) — cache
   `Arc<dyn BackendDevice>` on the owning module struct at load time instead.

**Gates:** (1) `RmsNorm`/`Rope`/`broadcast_bias` numeric parity vs current
CPU-path output within tight tolerance, so the fix doesn't silently change
numerics while removing the roundtrip; (2) compiles; (3) **device-gated**:
`rocm-smi`/`HIP_TRACE` confirmation of zero DtoH/HtoD transfers during a
forward pass and continuous kernel-launch overlap without host stalls.

## 4. What this plan does not do

- Does not assume any prior document's claims without a fresh grep against
  this exact upload — including its own prior versions (v1/v2).
- Does not build `arch.rs`-style device-property detection without first
  checking `CapabilityProfiler` for overlap.
- Does not implement router backward, quantized backward variants, or
  multi-GPU EP for architectures other than the ones already MoE-wired
  (Laguna, Qwen3MoE, etc. from earlier session work) — all explicitly
  future scope.
- Does not claim any performance multiplier without a measured number —
  every throughput/traffic claim in this document is either cited from a
  paper as *motivation* (not a target) or explicitly marked "measure, don't
  assert."

## 5. Sequencing

WI-Charon-0 is the hard prerequisite for WI-EP1/EP4 (placement/topology
correctness) but is independent of WI-Charon-1/2/3 (kernel work doesn't
need real placement to be correct in isolation, only to be useful
multi-GPU). WI-Charon-1 and WI-Charon-2 are mutually independent and can
proceed in parallel. WI-Charon-3 depends on WI-Charon-2 existing (nothing
to dispatch via the ring without the WMMA kernel, though the descriptor
struct and ring-integration plumbing could be built against the existing
scalar kernels first as a lower-risk proving ground). WI-EP1→EP2→EP3
strictly ordered as before. WI-EP4 depends on WI-Charon-0 and WI-Charon-1
both landing. WI-Host-1 has no dependency on any other item here and can
start immediately — arguably should, given it's the cheapest, most
confirmed-real, and most directly determines whether any of the GPU-side
work is visible in end-to-end throughput at all.
