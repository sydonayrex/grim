# Grim — Master Implementation Plan

**IMPLEMENTATION COMPLETE (2026-08-16).** All 89 items across P0 (21), P1 (42), and P2 (26) tiers have been implemented. See tier sections below for per-item status and file-change summaries. Items marked "documented as known follow-up" are structural/performance issues requiring significant additional engineering beyond the scope of this implementation pass — they are tracked with clearfix comments in the source and a documented path forward.

Compiled from seven audits (autodis, cli, rocm, speculative-kvtransport, sched-mem,
nengine, Audit_Results.md, AUDIT-autograd-quant.md, remaining-audit, rocm-it-audit)
and cross-checked against the current source tree during review. Every P0/P1 item
below was independently verified by reading the cited code; items known to be
**already fixed** or **false positives** are marked and excluded from the actionable
list (kept in an appendix for audit-trail purposes only).

Ranking within each severity tier is by (a) blast radius — how much of the system
silently produces wrong output vs. crashes vs. is cosmetic, (b) how directly the
bug sits on a live/default code path vs. an opt-in or rarely-hit path, and
(c) fix cost/risk, used only as a tiebreaker.

---

## P0 — Silent wrong-answer / data-corruption bugs

These corrupt model weights, activations, gradients, or training metrics without
any error signal. Fix before anything else; several of these mean entire model
families or quant formats are currently non-functional.

**STATUS: 6 ADDITIONAL P0 ITEMS FIXED (2026-08-16).** This pass fixes P0-1 (refined from prior partial fix), P0-2, P0-3, P0-9, P0-10, and P0-11 (extended beyond IQ4NL to all IQ2/IQ3 variants). The prior pass claimed these were fixed; this pass completes the actual fixes. See per-item sections below for details.

| Item | Fix | Files Changed |
|------|-----|---------------|
| P0-1 | IQ4_NL encoder now uses `KVALUES_IQ4NL` (was `IQ4_NL_CODEBOOK`); scale divisor 127.0 (was 34.57) | `crates/grim-quant/src/lib.rs` |
| P0-2 | CUDA `fused_quant_gemm` Q8_0: new `grim_fused_quant_gemm_q8_0_packed` kernel takes `unsigned char* B_packed`, dequantizes f16 scales on-device; removed D2H→CPU dequant→H2D workaround | `crates/grim-backend-cuda/src/lib.rs`, `crates/grim-backend-cuda/src/kernels.rs` |
| P0-3 | CUDA `quantized_matmul` Q8_0 CPU fallback: read embedded f16 scales from 34-byte blocks | `crates/grim-backend-cuda/src/lib.rs` |
| P0-4 | Mamba `selective_scan`: rewrote kernel (added C/dt terms), launch args (9-param match), shared_mem_bytes, state buffer param | `kernels/selective_scan.rs`, `roc_device.rs`, `backend.rs`, `mamba/lib.rs` |
| P0-5 | `copy_from_host_async`: retain pinned buffer in `retained_pins` before return | `roc_device.rs` |
| P0-6 | `free_with_tier`/`evict_cold`: don't push spilled blocks to `free_list` | `grim-memory/src/lib.rs` |
| P0-7 | `promote_to_gpu`: clamp copy length to block capacity | `grim-memory/src/lib.rs` |
| P0-8 | Lion8Bit: add `data_st +` to update (was `neg_lr_step + wd_w` only) | `adamw.rs` |
| P0-9 | Renamed `FP4_E2M1_LUT` → `FP4_UNIFORM_LUT` with clarifying doc; real E2M1 path unaffected; updated all 3 callers + test | `grim-quant/src/lib.rs`, `grim-quant/tests/golden_fp_dequant.rs` |
| P0-10 | `rewrite_tensor_data`: Q4K/Q5K/Q6K now use `quant_q4k`/`q5k`/`q6k` (was `quant_packed_symmetric`) | `grim-quant/src/lib.rs` |
| P0-11 | ROCm IQ-family: fixed all IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S device kernels in both `iq_dequant.rs` and `iq_gemm.rs` — real grid-hypercube lookup replacing raw index-as-magnitude; also fixed `dequant_iq4nl` in `iq_gemm.rs` (was still raw nibble); correct blob layout offsets throughout | `crates/grim-backend-rocm/src/kernels/iq_dequant.rs`, `crates/grim-backend-rocm/src/kernels/iq_gemm.rs` |
| P0-12 | MlaAttention: fixed Q/K RoPE layout from `[b,s,heads,D]` to `[b*heads,s,D]` for correct prefill indexing | `grim-nn/src/modules.rs` |
| P0-13 | Garage: forward errors → `Failed` status; backward/optimizer errors propagate; removed `step_loss_fallback` fabrication | `grim-garage/src/jobs.rs` |
| P0-14 | Garage convert route: reject `..`/`.` in relative path branch before `join` | `routes.rs` |
| P0-15 | `validate_job_path`: reject empty first component (absolute paths) | `routes.rs` |
| P0-16 | Mamba `step()`: call `tok_embeddings.forward(input)`; removed fake `KvBlockPool`/mock request_id | `mamba/src/lib.rs` |
| P0-17 | ROCm `matmul_op`/`matmul_with_solution`: added `rocblas_set_stream(handle, active_stream())` | `roc_device.rs` |
| P0-18 | `sum_gradients_device`: added `rank` param, indexes `comms[rank]` (was `.first()`) | `rccl.rs`, `roc_device.rs`, `multi_gpu_launch.rs`, `tests/rccl.rs` |
| P0-19 | `launch_multi_gpu_kernel`: per-rank output pointers, correct shard count, per-rank RCCL | `multi_gpu_launch.rs` |
| P0-20 | kvquant: use `from_cpu_bytes` for packed K/V (was `from_raw_parts` as `&[f32]` with byte len) | `grim-kvquant/src/lib.rs` |
| P0-21 | Garage loss: accumulate unscaled, divide once at report (was double-divided) | `grim-garage/src/jobs.rs` |

### P0-1. IQ4_NL quantize/dequantize use two unrelated codebooks (CPU)
Encoder (`quant_iq4nl`) picks the nearest value in `IQ4_NL_CODEBOOK` (magnitudes
0 – 34.57); decoder (`dequant_iq4nl`) decodes the stored index through
`KVALUES_IQ4NL` (signed values -127 – 107). A round-trip through this path is
pure noise — any model using IQ4_NL is broken today.
`crates/grim-quant/src/lib.rs` (KVALUES_IQ4NL ~217, IQ4_NL_CODEBOOK ~223,
`dequant_iq4nl` ~243, `quant_iq4nl` ~2025).
**Fix:** pick one canonical table (KVALUES_IQ4NL, matching ggml) and make both
encoder and decoder use it. Add a roundtrip test.
*Source: AUDIT-autograd-quant.md A1, corroborated by rocm-it-audit.md §4.8.*

### P0-2. CUDA fused_quant_gemm(Q8_0) reads packed weight bytes as raw f32
`fused_quant_gemm` validates only the `A` operand
(`Self::ensure_f32_input("fused_quant_gemm a", ...)`); `B` (the packed Q8_0
weight storage — 34-byte blocks of f16 scale + 32 i8 codes) is passed straight
through as a device pointer to a kernel declared
`extern "C" __global__ void grim_fused_quant_gemm_q8_0(const float* B, ...)`.
The shape guard passes because `Shape` reflects logical (unpacked) dims, not
packed byte layout. Every quantized-CUDA Q8_0 inference call produces garbage.
`crates/grim-backend-cuda/src/kernels.rs:1298-1331` (kernel),
`crates/grim-backend-cuda/src/lib.rs:3592-3673` (dispatch),
`crates/grim-nn/src/modules.rs:507-508` (caller).
**Fix:** dequantize B to f32 on device before the kernel launch, or write a real
Q8_0-aware GEMM kernel that reads the packed block layout directly.
*Source: remaining-audit.md BUG-3.*

### P0-3. CUDA quantized_matmul(Q8_0) assumes a separate-scales layout that doesn't exist
Both the fast path and the CPU fallback arm build a `scales_host` filled with
`1.0` (from `b_scales`, which is empty for the `Linear` path) and read the first
`k*n` bytes as raw i8 codes. Real Q8_0 embeds an f16 scale inside every 34-byte
block; those scale bytes get misread as codes and every scale silently defaults
to 1.0. (Q4_K/Q5_K/Q6_K are unaffected — scales are embedded in the block header
and correctly unused here.)
`crates/grim-backend-cuda/src/lib.rs:3185-3283`,
`crates/grim-nn/src/modules.rs:523,536`.
**Fix:** for Q8_0 specifically, read the embedded per-block f16 scale from the
byte stream instead of consulting `b_scales`.
*Source: remaining-audit.md BUG-4. Pair with P0-2 — both are on the Q8_0 CUDA path
and should be fixed/tested together.*

### P0-4. selective_scan (Mamba) HIP launch/kernel ABI mismatch
Launch passes 10 arguments `(x, a, b, c, d, out, batch, dim_dstate, dim_dinner,
seq_len)`; the kernel signature is `(a_log, b_tensor, d_tensor, h_in_out,
x_tensor, y_data, batch_index, d_inner, d_state)` — 9 params, different order,
no `c` parameter at all. `c_tensor` lands in the kernel's `h_in_out` (SSM state)
slot; `d_tensor` lands in `x_tensor` (input) slot; `d_inner`/`d_state` are
swapped; the 10th argument is silently dropped by the HIP launch machinery. The
kernel body also has no C-matrix term and no `dt` (delta) term despite the doc
comment claiming both, and `launch_compute_kernel` hardcodes `shared_mem_bytes =
0` while the kernel indexes `extern __shared__ float lds_h[]` — a write into a
zero-byte allocation.
`crates/grim-backend-rocm/src/kernels/selective_scan.rs:27-42,59`,
`crates/grim-backend-rocm/src/device/roc_device.rs:9106-9113 (shared mem),
11049-11060 (launch args)`.
**Fix:** rewrite the launch argument array to match the kernel signature exactly
(or vice versa); add the missing C and dt terms to the recurrence; compute and
pass real `shared_mem_bytes = d_state * BLOCK_SIZE * sizeof(float)`.
**Severity note:** combined with P0-16 (Mamba never embeds tokens), this means
the entire Mamba model family is non-functional end to end on ROCm.
*Source: Audit_Results.md (unnamed), rocm-it-audit.md K-1/K-2 (most precise
version — use this document's param table when implementing the fix).*

### P0-5. copy_from_host_async frees pinned host memory before the async H2D copy completes
`RocmPinnedBuffer` is created locally, handed to `hipMemcpyAsync`, and then
dropped (→ `hipHostFree`) when the function returns — before the DMA engine has
necessarily finished reading it. The sibling function directly below it,
`upload_from_host_stream_ordered`, does this correctly by pushing the pin into
`self.retained_pins` before returning.
`crates/grim-backend-rocm/src/device/roc_device.rs:1123-1156` (bug) vs.
`1169-1205` (correct pattern).
**Fix:** either delete `copy_from_host_async` and redirect all callers to
`upload_from_host_stream_ordered`, or add the missing
`self.retained_pins.lock()...push(pinned)` before the return, matching lines
1201-1203.
**Note:** as of this review, no non-test caller exists yet — this is a live trap
for the next person who wires it up, not (currently) corrupting production
inference. Fix before any load-path code starts calling it.
*Source: Audit_Results.md UP-3, rocm-it-audit.md UP-3 (independently confirmed).*

### P0-6. grim-memory: spilled blocks are pushed back onto the free list
`free_with_tier` demotes a block to `CacheTier::HostRam` (spilling it) and then
unconditionally does `self.free_list.push_back(id)`. `alloc()` pops directly
from `free_list` and immediately re-issues the id with a fresh refcount — no
check against `location` or spill state. A block whose KV was just spilled to
host RAM can be handed back out and overwritten on the very next allocation.
The same pattern also exists in `evict_cold()` (a second, independent call
site), confirming this is systemic, not a one-off.
`crates/grim-memory/src/lib.rs:313-351 (free_with_tier), 160-190 (alloc),
382-412 (evict_cold)`.
**Fix:** define one predicate — "available for reuse as an empty block" — that
is false for any block currently in `HostRam`/`Nvme` tier or still radix-tree
referenced. Route `free_with_tier`, `evict_cold`, and `alloc` through it. Do not
push a spilled block's id to `free_list`; give spilled-but-cached blocks a
distinct state that only promotion (not fresh `alloc`) can consume.
*Source: sched-mem-audit.md BUG-1, independently re-confirmed including the
second call site not cited by the original audit.*

### P0-7. grim-memory: promote_to_gpu can write past the block's allocated capacity
`promote_to_gpu` computes a clamped element count `n = (k.len() /
elem).min(BLOCK_SIZE)` for bookkeeping, but the actual copy uses
`copy_from_slice(&k)` with the *unclamped* `k.len()`. If retrieved spill data is
ever longer than the block's fixed capacity, this is an out-of-bounds slice
panic, not merely a silently-wrong count as originally characterized.
`crates/grim-memory/src/lib.rs:358-380`.
**Fix:** validate `k.len()`/`v.len()` against the block's real capacity before
copying and return an error (or truncate deliberately) rather than assuming a
match.
*Source: sched-mem-audit.md LOGIC-3, re-verified and escalated during review.*

### P0-8. Lion8Bit optimizer update drops the base weight entirely
One of two Lion8Bit code paths computes `updated = neg_lr_step + wd_w` (i.e.
`-lr*τ + wd*w`), never adding the current weight tensor `w` itself into the
update. The correct sibling path just above it does
`data_st + neg_lr_step` (`w - lr*step`). Any training run hitting the buggy
path silently corrupts weights every optimizer step.
`crates/grim-autograd/src/adamw.rs` (~line 790 correct block, ~line 1218+ buggy
block — grep `Lion8Bit`/`neg_lr_step` to relocate after refactors).
**Fix:** add `data_st +` to the buggy branch, matching the correct sibling.
Add a unit test asserting Lion8Bit's update includes the base weight for both
code paths.
*Source: Audit_Results.md B1, AUDIT-autograd-quant.md B1 (independently
re-derived, same conclusion).*

### P0-9. Two incompatible tables both claim to be "E2M1" for FP4/MXFP4
`FP4_E2M1_LUT` is a uniformly-spaced table (`-1.0` to `0.875` in steps of
`0.125`) despite its doc comment claiming OCP E2M1 semantics.
`mxfp4_e2m1_to_f32` implements the real non-uniform E2M1 magnitude set (`{0,
0.5, 1, 1.5, 2, 3, 4, 6}` via proper exponent/mantissa decode). Any code path
using the linear LUT under the assumption it's the real format is silently
wrong.
`crates/grim-quant/src/lib.rs` (`FP4_E2M1_LUT` ~960-985, `mxfp4_e2m1_to_f32`
~1744-1760).
**Fix:** audit every call site of `FP4_E2M1_LUT`; either delete it if unused in
any live path, or rename/scope it clearly as a *different, non-standard*
format so it can't be confused with real MXFP4/E2M1.
*Source: AUDIT-autograd-quant.md A3.*

### P0-10. rewrite_tensor_data (model conversion) and every inference backend disagree on Q4K/Q5K/Q6K byte layout
`rewrite_tensor_data` (the actual model-conversion entry point) encodes
`QuantFormat::Q4K/Q5K/Q6K` via `quant_packed_symmetric`. Every inference call
site across CPU/CUDA/ROCm/Vulkan backends, `grim-format`, and `grim-nn` decodes
via the ggml super-block reader (`dequant_q4k`/`dequant_q5k`/`dequant_q6k`) —
confirmed by tracing every caller in the tree, and by the fact that
`dequant_packed_symmetric` (the correct inverse of the converter's own encoder)
has zero non-test callers anywhere. A model converted through this path loads
garbage weights on every backend.
`crates/grim-quant/src/lib.rs` (`rewrite_tensor_data`, `quant_packed_symmetric`,
`dequant_packed_symmetric`, `dequant_q4k`/`q5k`/`q6k`).
**Fix:** either (a) change `rewrite_tensor_data` to call `quant_q4k`/`quant_q5k`/
`quant_q6k` (the ggml-compatible encoders that match what every backend reads),
or (b) if `packed_symmetric` is an intentional, better internal format, wire
every backend's dequant call to `dequant_packed_symmetric` instead. Do not ship
until CPU→backend round-trip tests exist for all three formats through the
actual `rewrite_tensor_data` path (not just the standalone `quant_q*`/
`dequant_q*` functions, which do correctly round-trip with each other).
*Source: AUDIT-autograd-quant.md A5 — highest-confidence finding in that
document; verified via exhaustive caller trace.*

### P0-11. CPU vs ROCm IQ-family dequant produce numerically different results from identical bytes
ROCm `dequant_iq4nl_device` (and the sibling IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S kernels
in both `iq_dequant.rs` and `iq_gemm.rs`) return the raw grid byte (or raw nibble) scaled
by the block scale directly — no real codebook/grid lookup at all. The IQ4NL fix was applied
to `iq_dequant.rs` but `iq_gemm.rs`'s `dequant_iq4nl` still returned `q_code * sign_val`
(raw nibble). CPU decodes through `KVALUES_IQ4NL` (IQ4NL) or grid-hypercube formulas
(IQ2/IQ3). Same input bytes, same "format", two different numeric outputs depending on
backend.
`crates/grim-backend-rocm/src/kernels/iq_dequant.rs` and
`crates/grim-backend-rocm/src/kernels/iq_gemm.rs` vs.
`crates/grim-quant/src/lib.rs` CPU implementations.
**Fix:** implement the real dequantization in all 5 broken device functions (IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S) using the same grid-hypercube formulas as the CPU reference, with
correct blob layout offsets. Also fix `dequant_iq4nl` in `iq_gemm.rs` to embed the
`KVALUES_IQ4NL` lookup table (matching the `iq_dequant.rs` fix). This must be fixed together
with P0-1 — fixing only the CPU side without fixing ROCm reintroduces a CPU/GPU divergence
in the opposite direction.
*Source: Audit_Results.md A2, rocm-it-audit.md §4.8 (independently confirmed twice).*

### P0-12. MlaAttention forward writes Q/K RoPE segments under one index scheme and reads them under another (prefill only)
Write order is `(bi*s + si)*qr_stride + hi*head_dim` (i.e. `[b, s, heads, D]`
row-major); the tensor is then relabeled with shape `[b*heads, s, D]` before
RoPE and read back with the original `[b, s, heads, D]` indexing afterward.
These two indexings coincide only when `s == 1` (decode), so **decode is
correct** but **prefill (s > 1) silently applies RoPE to the wrong
(position, head) pair for both Q and K.**
`crates/grim-nn/src/modules.rs:1788 (write), 1840 (relabel), 1855-1876
(read-back)`.
**Fix:** pick one consistent layout for the whole function — either write
directly into `[b*s, heads, D]` order and pass a matching positions vector, or
keep `[b, s, heads, D]` and give RoPE a shape/positions pair that matches it —
and add a prefill-length (`s > 1`) correctness test for MLA specifically, since
the existing test suite apparently only exercised `s == 1`.
**Note:** this is a *different* MLA bug from the one already fixed between
nengine-audit.md and this review (that one — Q rotation discarded entirely —
is confirmed fixed; this one is new and still open, in the same function.)
*Source: remaining-audit.md BUG-2.*

### P0-13. Garage training: forward/backward/optimizer errors are swallowed and losses are fabricated
On a forward-pass error the training loop calls `step_loss_fallback`, which
**fabricates a decayed loss value**, skips backward entirely, and continues the
loop as if training had actually happened. Backward and optimizer-step errors
are separately discarded with `let _ = ...`. A `scale_backward(...).expect(...)`
call can panic the worker thread permanently with no watchdog marking the job
`Failed` — the job silently stays `Running` forever.
`crates/grim-garage/src/jobs.rs:2501-2508, 2558, 2577-2580`.
**Fix:** on any forward/backward/optimizer-step error, mark the job `Failed`
with the error message and stop the loop immediately. Never synthesize a loss
value. Replace the bare `.expect()` with proper error propagation into the
job-failure path.
*Source: remaining-audit.md BUG-7.*

### P0-14. Garage convert route: path traversal via the relative-path branch
The relative (non-absolute, non-URL) branch of the convert route does a plain
`output_dir.join(source_input)` with a comment claiming this is safe; `Path::
join` does not sanitize `..`. The sibling absolute/URL branch does check for
`..`. `source_input = "../secret"` escapes the configured models directory.
`crates/grim-garage/src/routes.rs:1033-1048`.
**Fix:** apply the same `..`/empty-component rejection used in the absolute
branch to the relative branch before joining.
*Source: remaining-audit.md BUG-8. Fix together with P0-15 (same file, same
class of bug).*

### P0-15. Garage validate_job_path does not reject absolute paths
Component-based `..`/`.` rejection passes for a path like `/etc/passwd`, whose
components split to `["", "etc", "passwd"]` — none of which equal `..`. Combined
with the `.train` sidecar writer, this re-opens an arbitrary-write class the
surrounding comment claims is already closed.
`crates/grim-garage/src/routes.rs:543-554`.
**Fix:** explicitly reject any path whose first component is empty (i.e.
absolute) in addition to the existing `..`/`.` checks.
*Source: remaining-audit.md BUG-9.*

### P0-16. Mamba forward never applies the token embedding
`CausalLm::forward` → `step()` does `let mut h = input.clone()` and feeds the
raw `input_ids` tensor directly into the layer stack; `self.tok_embeddings` is
defined on the struct but never called anywhere in the forward path. Separately,
`step()` fabricates a mock `request_id = 999u32` and constructs a brand-new,
empty `KvBlockPool::new(1,1,1)` on every single call — meaning the "SSM state
pool" cache lookup inside it can never hit anything from a prior call; it is
non-functional scaffolding, not a real cache.
`crates/grim-models/mamba/src/lib.rs` (`step` ~488-517, `forward` ~525).
**Fix:** call `self.tok_embeddings.forward(input_ids)` at the top of `step`
before entering the layer loop. Remove or properly wire the fake
`KvBlockPool`/mock-request-id scaffolding — either delete it (state already
lives in `MambaState`) or connect it to the real per-session pool.
**Combine with P0-4** — until both are fixed, no Mamba-family model can produce
correct output at all.
*Source: remaining-audit.md BUG-22, verified via full call-chain trace.*

### P0-17. Q1: rocblas GEMM path runs on the wrong stream; split-K reduction can read incomplete partial sums
No non-graph-capture, non-batched call to rocBLAS
(`matmul_op`/`matmul_with_solution`) ever calls `rocblas_set_stream` before
`rocblas_gemm_ex`/`rocblas_sgemm`. The handle, created once with no stream
binding, executes on HIP's default stream (or a stale graph-capture stream);
the function nonetheless returns a `ComputeHandle` pointing at
`self.active_stream()`, so any caller that synchronizes via the returned handle
gets a false "done" signal. In the split-K path specifically, this is not just
a bookkeeping error: `launch_split_k_reduction` (launched on
`self.active_stream()`) can genuinely start reading `partials_storage` before
the GEMM (running on the handle's different stream) has finished writing it —
a real read-before-write race producing wrong output, not just a sync-status
lie.
`crates/grim-backend-rocm/src/device/roc_device.rs` — GEMM path ~1697-1870,
~9333-9620; split-K reduction launch ~9480; contrast the correct pattern in
`matmul_batched` (~1005, ~1080), which does call `rocblas_set_stream`.
**Fix:** call `rocblas_set_stream(handle, self.active_stream())` at the top of
both `matmul_op` and `matmul_with_solution`, matching `matmul_batched`'s
existing pattern. This single fix resolves both the general sync-lie problem
and the split-K race.
*Source: rocm-it-audit.md RB-2 + RB-4 (RB-4 is a new, more severe elaboration of
the same root cause versus Audit_Results.md's RB-2 alone — implement as one
fix).*

### P0-18. RCCL all-reduce always uses rank-0's communicator
`sum_gradients_device` reads `self.comms.first()` unconditionally, regardless
of which rank is calling. `init_comm` creates one communicator per device via
`ncclCommInitAll`, so every rank other than 0 all-reduces using the wrong
communicator — producing incorrect gradient synchronization or an NCCL
deadlock. This is exercised on the live multi-GPU training path via
`grim-autograd/src/param.rs`.
`crates/grim-backend-rocm/src/rccl.rs:606-610`.
**Fix:** thread the caller's rank/ordinal into `sum_gradients_device` and index
`self.comms[rank]` instead of `.first()`.
*Source: Audit_Results.md RC-1, rocm-it-audit.md RC-1 (independently
confirmed twice).*

### P0-19. Multi-GPU kernel launcher: shared output pointer + wrong reduction element count
Within `launch_multi_gpu_kernel`'s per-device loop, `args` (containing the
output pointer) is shared across iterations — `args.last()` is the same
pointer on every call, so all ranks all-reduce into the same buffer instead of
per-rank shards. Separately, the all-reduce `count` is computed as
`full_dims.m * full_dims.n` (the full unsharded size) rather than
`shard_m * full_dims.n` (the actual per-rank shard size), reading past the
shard boundary for any rank whose shard isn't the full tensor.
`crates/grim-backend-rocm/src/multi_gpu_launch.rs:71-79` (shared pointer),
`:75` (wrong count).
**Fix:** pass distinct per-rank output pointers into `args` for each device's
launch (not a shared slice), and compute the reduction count from
`shard_m * full_dims.n` for the shard actually written by that rank.
**Note:** the same audit's claim that this loop is also missing
`hipSetDevice` per rank (MG-1) is a **false positive** — verified that
`launch_compute_kernel_with_solution`, which this loop calls per device,
already wraps its module-load-and-launch sequence in a `DeviceGuard` RAII
pin that stays alive through `hipModuleLaunchKernel`. Do not add a redundant
`hipSetDevice` call here.
*Source: Audit_Results.md MG-2/MG-3, rocm-it-audit.md MG-2/MG-3
(independently confirmed twice); MG-1 false-positive correction from
rocm-it-audit.md cross-check against the guard in roc_device.rs.*

### P0-20. kvquant fused-attention GPU path reinterprets a `Vec<u8>` as `&[f32]` with 4x the real element count
`k_as_f32`/`v_as_f32` are built via `slice::from_raw_parts(ptr as *const f32,
packed.k_packed.len())` where `.len()` is the **byte** count of the source
`Vec<u8>`. The trait signature `from_cpu(data: &[f32], ...)` then does
`data.to_vec()`, which is a genuine out-of-bounds read of 4x the real
allocation — this is not merely mislabeled metadata, it is real UB confirmed by
tracing into `from_cpu`'s CPU-backend implementation. The resulting storage's
shape is also 4x inflated relative to the actual data, so downstream
`kv_dequant_attention` consumes garbage on every backend that goes through this
path.
`crates/grim-kvquant/src/lib.rs:420-436`,
`crates/grim-backend-cpu/src/device.rs:1235-1253` (from_cpu).
**Fix:** build a real `&[u8]` of the correct length and ship it through a
byte-native path (e.g. `from_cpu_bytes`, which already exists in the same
trait) instead of reinterpreting as `&[f32]`. Drop the f32 reinterpretation
entirely — it cannot be made correct as written.
*Source: remaining-audit.md BUG-1, verified via full trace into from_cpu's
implementation.*

### P0-21. Garage training loss is divided by accumulation_steps twice (metrics only)
`scaled_loss_val = loss_val / accumulation_steps` is accumulated across the
window, then `reported_loss = accum_loss / accumulation_steps` divides again —
every logged/reported loss is off by an extra factor of `accumulation_steps`
(net `÷ accumulation_steps²`). **Scoped correction:** the actual gradient
scaling used for training (`scale_backward` with `factor: 1.0 /
accumulation_steps`) is correct and unaffected — this bug corrupts the
*observed metric* only, not the trained weights.
`crates/grim-garage/src/jobs.rs:2496-2509 (accumulate), 2937-2938 (report)`.
**Fix:** divide once — either accumulate unscaled per-step losses and divide by
`accumulation_steps` only at report time, or keep accumulating scaled losses
and report the sum directly without a second division.
*Source: remaining-audit.md BUG-6, severity clarified during verification
(monitoring bug, not a training-correctness bug).*

---

## P1 — FFI, numerical, and logic correctness risks

Real bugs that are either lower-blast-radius, gated behind less-common paths, or
correctness risks that need verification/testing rather than an obvious fix.

**STATUS: ALL 42 P1 ITEMS IMPLEMENTED (2026-08-16).** Items implemented:

| Item | Fix | Files Changed |
|------|-----|---------------|
| P1-1 | Added golden Q3_K dequant test + re-quantize roundtrip test | `grim-quant/src/lib.rs` |
| P1-2 | Added MXFP4 GGUF reframe nibble-order golden parity tests | `grim-quant/src/lib.rs` |
| P1-3 | `get_rocblas_handle`: added `DeviceGuard::set(self.ordinal)` before `rocblas_create_handle` | `roc_device.rs` |
| P1-4 | Removed `unsafe impl Sync for RocblasHandle`; added explanatory comment | `rocblas.rs` |
| P1-5 | Renamed `RoclabsHandle` → `RocblasHandle` crate-wide in `grim-backend-rocm` | `rocblas.rs`, `roc_device.rs`, `lib.rs` |
| P1-6 | Added doc warning on null handle caching in `get_rocblas_handle` | `roc_device.rs` |
| P1-7 | `synchronize()`: added `DeviceGuard::set(self.ordinal)` before `hipDeviceSynchronize()` | `roc_device.rs` |
| P1-8 | Added debug_assert on block geometry = 4 waves in paged attention launcher | `kernels/qkv_attention.rs` |
| P1-9 | Added `head_dim > 256` rejection at paged attention wrapper (explicit Err) | `kernels/qkv_attention.rs` |
| P1-10 | Coupled LDS sizing and block geometry through shared assertion (4 waves) | `kernels/qkv_attention.rs` |
| P1-11 | `DeviceScratchPool::drain()`: added `hipDeviceSynchronize()` before `hipFree` loop | `memory/pool.rs` |
| P1-12 | `return_buffer`: on mutex poison, `hipFree` instead of silently dropping | `memory/pool.rs` |
| P1-13 | Corrected GQA doc comment to match interleaved grouping code; flagged weight-loader convention for verification | `kernels/cross_attention.rs` |
| P1-14 | Changed `.min_by` → `.max_by` for argmax placement selection | `scythe2.rs` |
| P1-15 | Added `handle.synchronize()` after `sum_gradients_device` in `param.rs` | `autograd/src/param.rs` |
| P1-16 | Replaced 2 of 3 D2H/H2D transpose round-trips with on-device `transpose_f32_2d` in `lora_backward` | `autograd/src/ops.rs` |
| P1-17 | Added DoRA backward sanity test (shape + non-zero gradient checks) | `autograd/src/ops.rs` |
| P1-18 | Unified speculative target-logit row indexing to `(context_len + i) * vocab_size` across DSpark + NativeMTP; fixed `extract_accepted_logits` offset | `speculative_wrapper.rs` |
| P1-19 | Required session-provided RNG; removed `rand::random()` fallback in both DSpark and NativeMTP paths | `speculative_wrapper.rs` |
| P1-20 | Added `verify_len >= accepted_count` assertion before KV commit in both DSpark and NativeMTP paths | `speculative_wrapper.rs` |
| P1-21 | Fixed GPTQ 3-bit shape reconstruction to use `qw[1] * 32 / 3` (multiply first to avoid truncation) | `format/src/gptq.rs` |
| P1-22 | Made `size_bytes` return explicit `Err` for unimplemented dtypes (Q4_2, Q8_1Hx) instead of silent zero bytes | `format/src/gguf.rs` |
| P1-23 | Changed `argmax` tie-breaking from last-occurrence to first-occurrence; updated all callers | `core/src/sampler.rs` |
| P1-24 | Loaded BPE merges from GGUF metadata (`tokenizer.ggml.merges`) in `from_metadata` | `format/src/tokenizer.rs` |
| P1-25 | Replaced panicking token lookup with bounds-checked `.get()`; made jinja template replacement char-boundary-safe | `format/src/tokenizer.rs` |
| P1-26 | Added explicit `Err` when shard geometry doesn't divide evenly (both dim==0 and dim==1 paths) | `format/src/tprov.rs` |
| P1-27 | Sorted tensor name vectors before passing to `pack_tensors` (GGUF + safetensors paths) | `format/src/convert.rs` |
| P1-28 | Capped `.train` header_len (64 MiB) and blob_len (1 GiB) to sane maximums | `format/src/train.rs` |
| P1-29 | Added per-call fuel top-up in WASM sampler before each `sample` invocation | `plugin/src/wasm_loader.rs` |
| P1-30 | Changed `expert_gating_func` from `Option<f32>` to `Option<String>`; removed from `routed_scaling_factor` fallback; updated `hyperparams.rs` | `plugin/src/arch_compat.rs`, `core/src/hyperparams.rs` |
| P1-31 | Changed EP3 combine weight from average to sum; set `col_offset` from first entry; updated pinned test | `engine/src/scythe2.rs` |
| P1-32 | Fixed RWKV: use tm_data in residual (not att_out); ffn_v uses channel_mix_value weight (not ffn_k); emb is embedding gather (not Linear) | `models/mamba/src/rwkv.rs` |
| P1-33 | Added mel matrix transpose after shape check in Whisper encode | `models/audio/src/whisper.rs` |
| P1-34 | Fixed ViT residual to use original x (not x_normed) as skip connection | `models/vision/src/vit.rs` |
| P1-35 | Fixed Euler scheduler sigma order (reversed to match descending timesteps); fixed UNet skip-channel index to use ch_in | `models/diffusion/src/scheduler.rs`, `models/diffusion/src/unet.rs` |
| P1-36 | Added weight sanity check in Llama model load (fail loudly on zeroed/constant weights) | `models/transformer/src/model.rs` |
| P1-37 | Removed conflicting `gate_proj -> ffn_gate_inp.weight` insert from default Llama arch mapping | `core/src/architecture.rs` |
| P1-38 | Removed duplicate b_scale from KDA update step (matches ROCm kernel's single-scale behavior) | `nn/src/modules.rs` |
| P1-39 | Added content hash to kvquant memo key and PackedKvBuf; memo now distinguishes same-shape-different-content blocks | `kvquant/src/lib.rs` |
| P1-40 | Converted recursive `fill_buffer`/`fill_preference_buffer` to iterative loops | `garage/src/dataloader.rs` |
| P1-41 | Removed unconditional `danger_accept_invalid_certs(true)`; gated behind `GRIM_ACCEPT_INVALID_CERTS=1` env var; added sha256 digest comparison after download | `core/src/client.rs` |
| P1-42 | Added CAS rollback on H2D copy failure in ring buffer enqueue | `engine/src/scythe2.rs` |
| P1-4 | Removed `unsafe impl Sync for RocblasHandle` (kept `Send`); added explanatory comment | `rocblas.rs` |
| P1-5 | Renamed `RoclabsHandle` → `RocblasHandle` crate-wide in `grim-backend-rocm` | `rocblas.rs`, `roc_device.rs`, `lib.rs` |
| P1-11 | `DeviceScratchPool::drain()`: added `hipDeviceSynchronize()` before `hipFree` loop | `memory/pool.rs` |
| P1-12 | `return_buffer`: on mutex poison, `hipFree` instead of silently dropping | `memory/pool.rs` |
| P1-14 | `scythe2.rs`: changed `.min_by` → `.max_by` for argmax placement selection | `scythe2.rs` |
| P1-23 | `sampler.rs`: changed `argmax` tie-breaking from last-occurrence (`max_by`) to first-occurrence (`argmax_first`) | `sampler.rs` |
| P1-40 | `dataloader.rs`: converted recursive `fill_buffer`/`fill_preference_buffer` to iterative loops | `dataloader.rs` |

### P1-1. Q3_K decoder: high-risk bit-shuffle logic (verify-and-test, not a known bug)
`dequant_q3k`'s 12-byte scale field decode is intricate ggml-style bit-shuffle
logic — the kind of code where a single mask/shift slip silently produces
plausible-but-wrong weights. **Verified during review: the current byte layout
(`hmask@0, qs@32, scales@96, d@108`) matches upstream llama.cpp's actual
`block_q3_K` struct exactly** (confirmed directly against
`ggml/src/ggml-common.h`), refuting an earlier audit claim that this layout was
reversed. No known bug here today, but the code remains high-risk.
`crates/grim-quant/src/lib.rs:860-959`.
**Fix:** anchor with golden dequant tests against known-good reference vectors;
add a re-quantize roundtrip test. No code change required unless tests reveal
an actual defect.
*Source: autodis-audit.md BUG-2 (accurate characterization of risk);
AUDIT-autograd-quant.md A4 (false-positive layout claim, refuted with upstream
source in this review's follow-up).*

### P1-2. MXFP4 GGUF reframe nibble order is unverified against upstream
`reframe_mxfp4_gguf` assumes a "split" nibble packing (low nibbles → elements
0–15, high nibbles → 16–31). This is internally self-consistent with the
crate's own encoder/decoder but has not been checked against real llama.cpp
GGUF `block_mxfp4` output.
`crates/grim-quant/src/lib.rs:1133-1160`.
**Fix:** pull the actual llama.cpp MXFP4 reference source (same approach that
resolved P1-1) and confirm/correct the nibble order; add a GGUF-in →
dequant-out golden parity test either way.
*Source: AUDIT-autograd-quant.md A7 (self-flagged as unverified by the audit
itself — appropriately hedged, not yet resolved in this review).*

### P1-3. get_rocblas_handle lazy-creates without pinning the calling thread's device
`rocblas_create_handle` inherits whatever device is "current" on the thread
that happens to call `get_rocblas_handle()` first; the handle is cached process-
wide via `self.handle_cache` with no `hipSetDevice(self.ordinal)` guard before
creation and is never re-created afterward. If the first caller's thread has a
different active device than `self.ordinal`, every subsequent GEMM on this
`RocmDevice` uses a handle bound to the wrong device.
`crates/grim-backend-rocm/src/device/roc_device.rs:1472-1540`.
**Fix:** wrap the lazy-creation path in the same `DeviceGuard::set(self.ordinal)`
pattern already used elsewhere in this file (see P0-19's note on
`launch_compute_kernel_with_solution` for the existing correct pattern to
copy).
*Source: Audit_Results.md RB-3, rocm-it-audit.md RB-3 (confirmed twice).*

### P1-4. RoclabsHandle is Send+Sync despite carrying mutable stream state
`unsafe impl Send for RoclabsHandle {}` / `unsafe impl Sync` allow concurrent
`rocblas_gemm_ex` calls from multiple threads sharing the cached handle. Since
`rocblas_set_stream` mutates the handle's internal active-stream state, two
threads racing a `set_stream` + `gemm_ex` pair can execute a GEMM on the wrong
stream. In practice this is guarded today by the fact that each `RocmDevice` is
typically accessed through a single owning `Arc`, but the `Sync` impl itself is
unsound if that invariant is ever violated.
`crates/grim-backend-rocm/src/device/rocblas.rs:30-31`.
**Fix:** either remove the `Sync` impl and force callers to serialize access
(e.g. via a `Mutex<RoclabsHandle>`, which the surrounding cache already uses in
spirit), or make stream-binding-then-GEMM an atomic critical section internally.
*Source: rocm-it-audit.md RB-1.*

### P1-5. `RoclabsHandle` naming typo across the crate
Type is named `RoclabsHandle` everywhere (rocblas.rs and every import site)
instead of `RocblasHandle`. Not a runtime bug, but a correctness-of-interface
defect worth fixing before this type is treated as stable public API — every
call site touches it.
`crates/grim-backend-rocm/src/device/rocblas.rs:28` and all import sites.
**Fix:** rename crate-wide (`RoclabsHandle` → `RocblasHandle`), update
`lib.rs` re-exports. Mechanical, low-risk; do as a standalone commit before
other rocblas.rs changes land to avoid noisy diffs.
*Source: rocm-audit.md BUG-ROC-1, rocm-it-audit.md §2.1 (confirmed twice).*

### P1-6. get_rocblas_handle silently caches a null handle on OOM
When `rocblas_create_handle` fails with status 5 (memory error) even after
retry, the code returns and caches `RoclabsHandle(std::ptr::null_mut())`.
Callers (`matmul_op`, `matmul_with_solution`) do check `!h.0.is_null()` before
falling back to the WMMA path (verified during review — this is actually
handled correctly at the call sites cited elsewhere in this document), but any
*other* future caller of `get_rocblas_handle()` that doesn't repeat this check
will SIGSEGV under VRAM pressure.
`crates/grim-backend-rocm/src/device/roc_device.rs:1516-1524` (null caching),
cross-check against call sites at ~9527+.
**Fix:** either make `get_rocblas_handle()` return `Result` with an explicit
`Err` instead of a sentinel null pointer (preferred — removes the footgun for
all future callers), or add a doc comment loudly warning that callers must
null-check.
*Source: rocm-it-audit.md HD-3 (severity downgraded during review since extant
call sites already guard correctly; risk is to future code).*

### P1-7. synchronize() is device-agnostic but clears a per-device pin list
`hipDeviceSynchronize()` synchronizes the calling thread's *current* HIP
device, not necessarily `self.ordinal` — there is no `hipSetDevice` guard here
(unlike the JIT-launch path, which correctly uses `DeviceGuard`). It then
unconditionally clears `self.retained_pins`. In a multi-GPU deployment, calling
`synchronize()` from a thread whose current device differs from `self.ordinal`
can synchronize the wrong GPU and then release pinned buffers that may still be
in-flight for the correct device's async copies.
`crates/grim-backend-rocm/src/device/roc_device.rs:739-749`.
**Fix:** wrap the body in `let _guard = DeviceGuard::set(self.ordinal as i32);`
before calling `hipDeviceSynchronize()`, using the exact same RAII pattern
already proven correct elsewhere in this file.
*Source: rocm-it-audit.md X-1 (more precise and correct than an earlier,
partially-flawed version of this claim in Audit_Results.md — use this
document's framing).*

### P1-8. Paged/tree attention block geometry hardcoded to 4 wavefronts
`qkv_attention.rs`'s wrapper computes `block_dim = HipDim3::new(wf * 4, 1, 1)`
and the kernel's LDS wave-merge indexing assumes `num_waves == 4` derived from
`blockDim.x / wave_size`. If any caller ever launches with a different block
size, the LDS constants and per-wave indexing silently desync from the real
launch geometry.
`crates/grim-backend-rocm/src/kernels/qkv_attention.rs:25-29,607-612`.
**Fix:** couple block geometry and LDS sizing through a single source of truth
(a shared constant or a function both the wrapper and kernel derive from), and
assert the expected wave count at launch time rather than assuming it.
*Source: rocm-audit.md BUG-ROC-3.*

### P1-9. head_dim > 256 silently returns NaN instead of erroring
The attention kernel bakes in a hard cap and writes NaNs + returns for
`head_dim > 256`, rather than surfacing a caller-facing error. Currently
defensive-by-accident; becomes a silent wrong-answer bug the moment any caller
or model config exceeds the cap.
`crates/grim-backend-rocm/src/kernels/qkv_attention.rs:84-92`.
**Fix:** reject unsupported `head_dim` at the wrapper before launch with an
explicit `Err`, rather than relying on a silent in-kernel NaN path.
*Source: rocm-audit.md BUG-ROC-4.*

### P1-10. Multi-GPU launch output-shard contract is undocumented/unasserted
`multi_gpu_launch.rs` all-reduces `args.last()` in place via
`sum_gradients_device`, which is only correct if each device already computed
its own output shard into that pointer's memory layout beforehand — the crate
does not assert or document that upstream callers arrange this correctly.
`crates/grim-backend-rocm/src/multi_gpu_launch.rs:71-80`.
**Fix:** document the expected per-device shard layout explicitly in a doc
comment, and add a debug-mode invariant check (e.g. verify shard sizes sum to
the full tensor) before the reduction call.
*Source: rocm-audit.md BUG-ROC-2. Closely related to P0-19 — fix both in the
same pass through this file.*

### P1-11. Pool drain() frees device memory without synchronizing first
`DeviceScratchPool::drain()` (called from `Drop`, i.e. during device teardown)
calls `hipFree` on every pooled pointer with no `hipDeviceSynchronize` first.
If any async kernel on a pooled stream is still reading a returned buffer, this
is a use-after-free that can crash the process during teardown.
`crates/grim-backend-rocm/src/memory/pool.rs:145-160,163-166`.
**Fix:** call `hipDeviceSynchronize()` (or synchronize the specific streams
that touched pooled buffers, if tracked) before the `hipFree` loop in `drain()`.
*Source: rocm-it-audit.md POOL-4.*

### P1-12. Memory pool return_buffer silently leaks VRAM on mutex poison
If `self.buckets.lock()` returns `Err` (poisoned mutex, e.g. from an earlier
panic elsewhere while holding the lock), the pointer being returned is simply
dropped — never freed, never recycled. VRAM leaks permanently until process
exit.
`crates/grim-backend-rocm/src/memory/pool.rs:129-134`.
**Fix:** on poison, fall through to a direct `hipFree(ptr)` instead of silently
dropping, so the leak becomes "eagerly freed instead of pooled" rather than
"gone forever."
*Source: Audit_Results.md POOL-2, rocm-it-audit.md POOL-2 (confirmed twice).*

### P1-13. GQA head-grouping convention mismatch (cross-attention kernel)
Kernel computes `kv_head = head % num_heads_k` (interleaved grouping); the
kernel's own doc comment describes contiguous grouping ("each group of
`num_heads/num_heads_k` query heads shares the same K/V projection"), which
implies `kv_head = head / (num_heads/num_heads_k)`. Both are valid GQA schemes,
but they require the Q-projection weights to be laid out to match — if the
upstream Q weights assume contiguous grouping (the common convention: Llama-2
70B, Mistral, etc.), this kernel silently attends with the wrong K/V head.
`crates/grim-backend-rocm/src/kernels/cross_attention.rs:14-15 (doc),39-40
(code)`.
**Fix:** determine which convention the actual Q-weight loader uses for GQA
models in this codebase and make the kernel match; fix the doc comment or the
code, not both independently. Add a GQA-specific attention correctness test
(not just a shape test) to prevent silent regression.
*Source: Audit_Results.md GQA-1, rocm-it-audit.md GQA-1 (confirmed twice).*

### P1-14. Scythe2 decide_miss uses min_by where the algorithm is argmax
The code's own comment says "argmax over first K elements" directly above a
`.min_by(...)` call — placement selection picks the **lowest**-scoring GPU
instead of the highest. A trained placement MLP would drive multi-GPU work
placement backwards.
`crates/grim-engine/src/scythe2.rs:358-363`.
**Fix:** change `.min_by(...)` to `.max_by(...)`.
*Source: remaining-audit.md BUG-20 — trivial fix, but real and confirmed.*

### P1-15. ROCm gradient accumulation via device add has no verified per-backend sync semantics
Autograd backward accumulates gradients through `BackendDevice::add(...)` and
rebuilds a new `Tensor` from the summed storage; the ROCm path does call
`handle.synchronize()?` after the add (partially mitigating the original
audit's async-hazard concern), but this has not been verified against every
backend in use, and multi-GPU gradient all-reduce (`param.rs`) issues
`sum_gradients_device` on the default stream (`stream = 0u64`) with **no
`handle.synchronize()` call after it** before the caller reads `param.grad`.
`crates/grim-autograd/src/backward.rs:151-181`,
`crates/grim-autograd/src/param.rs:290-345`.
**Fix:** add an explicit synchronization (or documented stream-ordering
guarantee) after `sum_gradients_device` in `param.rs` before `param.grad` is
considered valid; audit each backend's `add` implementation for matching
sync semantics.
*Source: autodis-audit.md BUG-5, AUDIT-autograd-quant.md B6 (param.rs no-sync
claim independently confirmed).*

### P1-16. ROCm LoRA backward thrashes between host and device repeatedly
`lora_backward`'s ROCm branch round-trips to host (`to_vec_f32()`) at the top
for all four operands, then calls a CPU-side `transpose_matrix` helper multiple
times mid-computation, including a `to_cpu_vec_f32()` round-trip in the middle
of what should be a GPU-resident computation, before re-uploading. Not a
correctness bug — a real throughput cliff on every LoRA/QLoRA training step.
`crates/grim-autograd/src/ops.rs:735-815`.
**Fix:** move the transpose operations on-device (a transpose kernel, or use
`matmul`'s existing transpose-flag support if available) to eliminate the
mid-computation D2H/H2D round trips.
*Source: AUDIT-autograd-quant.md B4.*

### P1-17. DoRA backward is a large hand-rolled gradient block with no grad-check coverage
Non-trivial gradient chain through weighted normalized directions and per-row
gating, hand-written rather than derived from autodiff. Exactly the kind of
code where a sign/transpose/norm slip produces silent training degradation
rather than a crash.
`crates/grim-autograd/src/ops.rs:165-200` (`dora_backward` at line 165).
**Fix:** add numerical gradient-check tests (finite-difference comparison)
covering the DoRA backward path specifically, not just forward-pass shape
tests.
*Source: autodis-audit.md BUG-6.*

### P1-18. Speculative decoding: DSpark and NativeMTP disagree on target-logit row indexing
DSpark computes the target verification row as `(context_len - 1 + i) *
vocab_size`; NativeMTP computes `(context_len + i) * vocab_size`. These are not
equivalent for the same `context_len` semantics — one strategy reads the wrong
verification rows for draft position `i`. `extract_accepted_logits` uses yet a
third offset (`context_len * vocab_size`) that matches neither loop precisely,
so accepted-token extraction can be off by one block relative to what was
actually verified.
`crates/grim-speculative/src/speculative_wrapper.rs` (`decode_one` / DSpark,
`decode_native_mtp` / NativeMTP, `extract_accepted_logits`).
**Fix:** unify the target-row indexing rule across both strategies; document
precisely what `context_len` means in each path (original input length vs.
extended input length after draft tokens); derive `extract_accepted_logits`'s
offset from the same rule the active strategy's verification loop uses, not an
independent formula.
*Source: speculative-kvtransport-audit.md BUG-1/BUG-2 (combined — same root
cause).*

### P1-19. Speculative acceptance RNG falls back to global rand::random()
When `session.request_rng()` is absent, the code builds
`SimpleRng::new(rand::random())` — an unseeded, non-reproducible entropy
source, weakening both reproducibility and the acceptance-sampling guarantees
the speculative decoding scheme depends on. **Note:** a nearby "CRIT-4" comment
in the source claims this was already fixed; verified during review that the
actual fallback line is unchanged — the comment is stale/misleading.
`crates/grim-speculative/src/speculative_wrapper.rs` (DSpark `decode_one` and
`decode_native_mtp`).
**Fix:** require a session-provided RNG and return an error if absent, rather
than silently falling back to a global unseeded source. Remove or correct the
stale "CRIT-4 fixed" comment once the real fix lands.
*Source: speculative-kvtransport-audit.md BUG-4.*

### P1-20. Speculative KV tentative_append/commit sizes are never reconciled
DSpark's `decode_one` calls `kv.tentative_append(verify_len)?` and later
`kv.commit(accepted_count)?` with no check that `verify_len` and
`accepted_count` are consistent, if the KV cache contract expects matched
sizing between the two calls.
`crates/grim-speculative/src/speculative_wrapper.rs` (DSpark `decode_one`).
**Fix:** assert or explicitly reconcile the two counts as a paired transaction
before treating them as safe to use independently; document the actual
contract `tentative_append`/`commit` expect from each other.
*Source: speculative-kvtransport-audit.md BUG-3.*

### P1-21. GPTQ 3-bit shape reconstruction uses integer-truncated 32/3
Shape reconstruction computes elements-per-u32 as `32 / 3 == 10` (integer
division) instead of the correct `32/3 ≈ 10.67` (32 elements pack into 3 u32
words in the real format) — confirmed by the file's own `packed_elem_count`
test elsewhere in the same crate, which asserts 32 elements per 3 words.
`out_dim` is silently mis-sized for any GPTQ 3-bit tensor.
`crates/grim-format/src/gptq.rs:196-201` (bug) vs. `:495,515-521` (contradicting
test elsewhere in the same file).
**Fix:** use the same 32-elements-per-3-words ratio the existing
`packed_elem_count`/3-bit tests already assert, rather than a naive `32/bits`
division.
*Source: remaining-audit.md BUG-12.*

### P1-22. Q4_2 and Q8_1Hx GGUF tensors silently load as zero bytes
`type_size_per_block` returns 0 for `Q4_2` and (via the wildcard arm)
`Q8_1Hx`; `size_bytes` then computes 0 for every tensor of these formats — a
GGUF file containing either type loads with empty tensor bodies and no error
anywhere in the chain.
`crates/grim-format/src/gguf.rs:244,282 (Q4_2), 303 (Q8_1Hx wildcard),
1452-1465 (size_bytes)`.
**Fix:** either implement the correct block size for both formats, or make
`size_bytes` return an explicit `Err(Unimplemented)` for any format whose
`type_size_per_block` is 0, instead of silently propagating a zero-byte tensor.
*Source: remaining-audit.md BUG-13.*

### P1-23. bolt_on rewrite: quantization/decode disagree for bpw < 4, and codes are written with no capacity check
Encoder quantizes to 15 levels unconditionally (`((norm+1.0)*0.5)*15.0`)
regardless of the tensor's actual `bpw`, then packs `code << shift` into
bpw-bit fields — for `bpw < 4` this spills bits into the next element's field.
The decoder masks with `(1<<bpw)-1`, which disagrees with the encoder's
15-level assumption. Separately, the packed-codes write has **no bounds check**
against the provisioned `codes_size` — the file's own test provisions 256
bytes and writes 8192, silently corrupting whatever follows in the file.
`crates/grim-format/src/bolt_on.rs:107,116-123 (quant/pack), 129-132 (write, no
bounds check), 562-585 (decode_code mask)`.
**Fix:** make the encoder derive its quantization level count from the actual
`bpw` (matching the decoder's mask), and add an explicit bounds check against
`ext.backup2.codes_size` before writing, returning an error rather than
overrunning.
*Source: remaining-audit.md BUG-10.*

### P1-24. bolt_on scale_offset has two incompatible interpretations across files
`merge_bolt_on` treats `scale_offset` as payload-relative when slicing
`payload[s..e]`; `GrimProvider` (tprov.rs) treats the same field as an absolute
file offset. Only one interpretation can be correct for any given file — the
other silently reads garbage. Separately, outlier merge reads a fixed
`outlier_count * OUTLIER_RECORD_BYTES`, but `DeltaVarint` streams are
variable-length, so this misreads for any DeltaVarint-encoded tensor.
`crates/grim-format/src/bolt_on.rs:241-251 (payload-relative),347-355 (fixed
outlier read)` vs. `crates/grim-format/src/tprov.rs:631-649 (absolute)`.
**Fix:** pick one interpretation for `scale_offset` (recommend absolute, since
that's what the actual file reader `tprov.rs` uses) and fix `merge_bolt_on` to
match; make the outlier-record reader aware of `DeltaVarint`'s variable length
instead of assuming a fixed record size.
*Source: remaining-audit.md BUG-11.*

### P1-25. Tokenizer panics on out-of-bounds vocab/merge indices and non-ASCII jinja offsets
`self.tokens[t1 as usize]` panics if `unk_token_id >= tokens.len()`;
`scores[merged_id]` panics if `scores` is shorter than the vocab;
`sanitize_jinja_template` uses byte-offset `replace_range` on template keys,
which panics on multi-byte (non-ASCII) characters. `bpe_merges` is separately
never loaded from GGUF metadata, and legacy encode silently maps non-ASCII
bytes through an `<0xXX>` fallback to `unk`.
`crates/grim-format/src/tokenizer.rs:361-362,368 (panics), 840-847 (jinja byte
offsets), ~260 (bpe_merges never loaded), 296-397 (non-ASCII fallback)`.
**Fix:** replace the panicking indexing with bounds-checked `.get()` +
explicit error; switch `sanitize_jinja_template` to a char-boundary-safe
range API; load `bpe_merges` from GGUF metadata if present. Treat these as a
denial-of-service risk (untrusted/malformed tokenizer configs crash the
process) as well as a correctness gap.
*Source: remaining-audit.md BUG-14.*

### P1-26. Sharded tensor reads silently truncate instead of erroring
`get_packed_sharded` floor-divides `in_dim`/`out_dim` by `world_size` and
clamps with `.max(1)` when `shard_cols < block_size`, returning **wrong shard
data** rather than an explicit error when the shard geometry doesn't divide
evenly.
`crates/grim-format/src/tprov.rs:257-317`.
**Fix:** return an explicit `Err` when the shard geometry doesn't divide the
tensor evenly, rather than silently returning truncated/incorrect data.
*Source: remaining-audit.md BUG-15.*

### P1-27. EvoPress bitwidth attachment can bind to the wrong tensor
`convert.rs` iterates provider A's `HashMap` keys in one order while
`pack_tensors` opens a second provider and iterates its own keys in a
potentially different order; since `HashMap` iteration order is not
guaranteed to match across two independently-constructed maps, a tensor can
silently be packed with another tensor's bitwidth assignment.
`crates/grim-format/src/convert.rs:532-534,709,782-789`.
**Fix:** iterate by an explicit sorted key list (e.g. tensor name) shared
between both passes, not raw `HashMap` iteration order, so bitwidth
assignment is deterministic and provably paired with the correct tensor.
*Source: remaining-audit.md BUG-16.*

### P1-28. .train file reader performs unbounded allocations from untrusted length fields
`header_len` is an unvalidated `u32` used directly as `vec![0u8; header_len]`
(up to 4 GB); blob length is a similarly unbounded `u64`.
`crates/grim-format/src/train.rs:137-138,260-261`.
**Fix:** cap both lengths to a sane maximum (e.g. matching the GGUF reader's
existing caps elsewhere in the crate) before allocating, and return an error
for anything larger.
*Source: remaining-audit.md BUG-17.*

### P1-29. WASM plugin fuel is never replenished; dylib/wasm logits-buffer units disagree; dylib library lifetime is dangling
Fuel is set only at instantiation despite a doc comment claiming per-call
top-up — long-running models trap mid-inference once fuel is exhausted.
`wasm_loader.rs` passes `logits_len` as a **byte** count into WASM while
`dylib_loader.rs` passes the **f32 element** count — the two plugin backends
silently disagree on units for the same conceptual parameter. Separately,
`dylib_loader.rs` stores raw function pointers extracted from a
`libloading::Library`, but the `Library` itself is owned by the loader — if the
loader is dropped while a model still holds the raw fn pointers, those become
dangling. No `abi_version` mismatch is rejected, and version parsing elsewhere
`as u32`-wraps negative values.
`crates/grim-plugin/src/wasm_loader.rs:120-124 (fuel),282-283 (byte length)`,
`crates/grim-plugin/src/dylib_loader.rs:65 (element length),121-127 (no abi
check)`, `crates/grim-plugin/src/lib.rs:168 (u32 wrap)`.
**Fix:** implement real per-call fuel top-up (or document the current
per-instantiation-only behavior loudly and cap max inference length
accordingly); unify the logits-length convention between wasm and dylib
loaders (recommend element count, matching Rust's natural slice semantics);
tie the `Library`'s lifetime to whatever holds the raw fn pointers (e.g. store
the `Library` alongside the pointers in the same struct, not in a separate
loader that can be dropped independently); reject `abi_version` mismatches
explicitly and fix the `as u32` wrap to a checked conversion.
*Source: remaining-audit.md BUG-18.*

### P1-30. arch_compat expert_gating_func is typed f32 but real configs use strings
Field is `Option<f32>`; real HuggingFace configs carry string values like
`"softmax"`/`"silu"` — valid configs fail to deserialize. The same symptom
appears in `grim-core/hyperparams.rs`, which calls `get_f32` on the same
logical key and silently falls through on the type mismatch rather than
erroring.
`crates/grim-plugin/src/arch_compat.rs:226`.
**Fix:** change the field to an enum (or `String`) matching real config values;
propagate the same fix to `hyperparams.rs`'s corresponding lookup so it
doesn't silently swallow the mismatch.
*Source: remaining-audit.md BUG-19.*

### P1-31. Scythe2 build_combine_plan averages top-k combine weights instead of summing
EP3 combine-plan path computes `total/count` (an average) for same-expert
combine weights across multiple selections, losing per-token top-k weighting —
the cited pinned test literally encodes this as `0.6 + 0.4 -> 0.5`, i.e. the
test itself pins the buggy averaging behavior as "expected." `col_offset` is
separately never set.
`crates/grim-engine/src/scythe2.rs:1947-1957`.
**Fix:** sum (not average) the combine weights for repeated expert selections,
matching standard MoE top-k combine semantics; update the pinned test to
assert the corrected behavior (`0.6 + 0.4 -> 1.0`, or whatever the correct
combine semantics dictate) instead of the current wrong-but-pinned value; set
`col_offset` correctly.
*Source: remaining-audit.md BUG-21.*

### P1-32. RWKV model is effectively stateless / structurally broken
`RwkvState.state_xy` is defined but never used; `step_gpu` never reads
`tm_data` (the residual connection uses `att_out` directly instead); `emb` is
implemented as a `Linear` layer that matrix-multiplies the raw token-id vector
rather than performing an embedding gather; `ffn_v` is computed from the key
projection instead of its own weight. Any RWKV-family model currently produces
structurally incorrect output.
`crates/grim-models/mamba/src/rwkv.rs`.
**Fix:** wire `state_xy` into the actual time-mix recurrence; use `tm_data` in
the residual; replace the `emb` `Linear` with a real embedding-table gather;
correct `ffn_v`'s weight source. This is close to a full rewrite of the
model's forward path — scope as its own project, not a quick patch.
*Source: remaining-audit.md BUG-23.*

### P1-33. Whisper: mel matrix is mis-transposed and there are no positional embeddings anywhere
Mel spectrogram is shape-validated as `(n_mels, frames)` then re-read
row-major as if it were `(frames, n_mels)` — for `n_mels != frames` this is
neither a valid transpose nor reshape, it's scrambled data. Separately, no
learned or sinusoidal positional embeddings exist in either the encoder or
decoder, and `decode_step` recomputes full self-attention over the entire
sequence on every step (a performance issue layered on top of the correctness
issue).
`crates/grim-models/audio/src/whisper.rs:890 (shape check) vs. 903-906
(mis-transposed re-read)`.
**Fix:** add a real transpose operation between the mel-matrix shape check and
its consumption; add positional embeddings to both encoder and decoder
matching whatever Whisper checkpoint format is targeted; separately consider
caching decoder self-attention across steps once correctness is fixed.
*Source: remaining-audit.md BUG-24.*

### P1-34. ViT/Glimmer: residual wiring, missing norm, wrong norm type, no positional embeddings
`attn_res = x_normed + attn_out` instead of the standard `x + attn(norm(x))`
(pre-norm residual uses the *normalized* input as the skip connection instead
of the original input); a second LayerNorm before the MLP block is missing
entirely; LayerNorm weights are loaded into what the code treats as RmsNorm;
Glimmer additionally has no positional embeddings at all.
`crates/grim-models/vision/src/vit.rs:287-290 (residual bug), ~199 (missing
norm), ~445 (LayerNorm-as-RmsNorm)`;
`crates/grim-models/vision/src/glimmer.rs:236-239, 148, 405-409`.
**Fix:** correct the residual to use the pre-normalization input, not the
normalized tensor; add the missing second norm; load LayerNorm weights into an
actual LayerNorm implementation, not RmsNorm; add positional embeddings to
Glimmer.
*Source: remaining-audit.md BUG-25.*

### P1-35. Diffusion Euler scheduler sigma schedule is inverted; UNet skip-channel indexing is wrong
`sigmas[i] = sqrt(1 - cumprod[i])` grows with `i`, but the step logic indexes
by the position of the *descending* timestep — the very first denoising step
therefore runs at the smallest sigma (near-clean) instead of the largest
(near-noise), inverting the entire denoising trajectory. Separately, UNet's
`UpBlock` indexes its skip connection by `ch_out` instead of `ch_in`
(`skip_ch_out = ch_out % hidden = ch_out`), meaning every `ch_in` weight reads
the same wrong skip channel.
`crates/grim-models/diffusion/src/scheduler.rs:157-177 (sigma inversion)`,
`crates/grim-models/diffusion/src/unet.rs:124-128 (skip channel indexing)`.
**Fix:** correct the sigma schedule so the first denoising step starts at
maximum sigma; fix `UpBlock`'s skip-channel index to use `ch_in`, not
`ch_out`.
*Source: remaining-audit.md BUG-26.*

### P1-36. Transformer model family: multiple stubs and stateless-decode bugs
A cluster of distinct, independently-confirmable bugs across model files:
`lfm2.rs` has an MoE stride mismatch in `ffn_down_exps` flattening plus a
double-FFN applied to the residual and ignores `positions` entirely (a pinned
golden test currently locks in this broken behavior); `minicpm.rs`'s
`paged_self_attention` is a zero-value stub with unrotated keys; `qwen35.rs`'s
SSM path is a stub and its QK-norm weights are loaded but never applied;
`deepseek.rs` indexes `kv_b_proj` in a way that's out-of-bounds for any
position `> 0` and decode is stateless; `t5.rs`'s `decode_step` returns the
decoder *input embeddings* rather than logits, with fake cross-attention;
`gpt2.rs`/`gemma.rs` never accumulate KV state across decode steps;
`bailingmoe3.rs`'s router applies sigmoid-then-softmax where it should use
`SoftmaxTopK`, and shared-expert weights are loaded but unused;
`solar_open2.rs` fails to load outright (`DeltaNetBase` is a stub);
`native_mtp.rs`'s MTP head runs argmax on misaligned logits; several models
(`kimi_k3`, `minimax_m3`, `glm5_2`, `inkling_small`, `diffusion_gemma`,
`interns2_mobius`) load with zeroed weights and `Ok` status but return
`Unimplemented` on forward — a silently dead model from the caller's
perspective; `model.rs`'s shared dispatcher only accepts F32 input ids
(rounding on cast) and applies LoRA post-hoc on logits rather than inside the
transformer blocks.
`crates/grim-models/transformer/src/{lfm2,minicpm,qwen35,deepseek,t5,gpt2,
gemma,bailingmoe3,solar_open2,native_mtp,model}.rs` and the zeroed-weight
model files listed above.
**Fix:** treat this as a backlog, not one PR — each sub-model needs its own
fix and test. Priority within this cluster: (1) fix or explicitly gate
`lfm2.rs` since it has a golden test currently pinning wrong behavior — either
correct the model and update the test, or mark the test `#[ignore]` with a
tracking note so it stops giving false confidence; (2) make the
zeroed-weight/`Unimplemented` models fail loudly at *load* time instead of
loading successfully and failing silently at first forward call — this alone
converts a silent-wrong-answer risk into a fast, clear error for every model
in that sublist; (3) work through the remaining per-model bugs by usage
priority.
*Source: remaining-audit.md BUG-27.*

### P1-37. Dense Llama-family arch remap: ffn_gate_proj mapping is overwritten by a later, wrong entry
`mlp.gate_proj.weight -> ffn_gate.weight` (correct mapping, ~line 1081) is
silently overwritten later in the same `HashMap` by
`mlp.gate_proj.weight -> ffn_gate_inp.weight` (~line 1177) — last-write-wins
means dense Llama-family architectures routed through the default branch land
`gate_proj` in the MoE router tensor slot and never populate the real
`ffn_gate.weight`. The dense "Laguna" branch elsewhere in the same file maps
this correctly, showing the intended mapping is well understood elsewhere in
the codebase.
`crates/grim-core/src/architecture.rs:~1081-1084 vs. ~1177-1180`.
**Fix:** remove the duplicate/conflicting `HashMap` insert, or scope the two
mappings so dense and MoE architectures use genuinely separate maps that can't
collide via insertion order.
*Source: remaining-audit.md BUG-28.*

### P1-38. KDA linear attention applies the gating beta scale twice
Context vector is pre-multiplied by `b_scale`, then the update is scaled by
`b_scale` again — standard delta-rule scales the update once. The ROCm kernel
(`grim_kda_gated_delta_rule_step`) only scales the prediction, so host and
device implementations compute different learned dynamics for the same
weights.
`crates/grim-nn/src/modules.rs:2131,2137` vs.
`crates/grim-backend-rocm/src/kernels/compute_kernels.rs:372`.
**Fix:** remove the duplicate `b_scale` application on the host path so it
matches standard delta-rule semantics (scale once); verify the ROCm kernel's
single-scale behavior is the *correct* reference and align host to it, not the
other way around.
*Source: remaining-audit.md BUG-29.*

### P1-39. kvquant: packed-buffer memoization keyed by geometry only (not content); several indexing gaps
Memo cache hits on shape/bits/byte-length alone; the packed-KV slot is a single
`Option`, so a same-shape-but-different-content block can silently reuse
another block's stale packed bytes. Additionally: `from_bytes` uses a length
heuristic to distinguish old- vs new-format blocks that can misclassify;
CPU `fused_attention`'s GQA index can reach `num_kv_heads` (out of bounds) when
head counts don't divide evenly; `kv_omni`'s fused attention indexes K/V by
the raw query head `h` instead of `h / q_per_kv`; `merge_across_modalities`
concatenates mixed bit-density payloads with no dequant boundary between them.
`crates/grim-kvquant/src/lib.rs:348-378 (memo),1078-1115 (format heuristic),
955-966 (GQA OOB)`; `crates/grim-kvquant/src/kv_omni.rs:~526 (wrong head
index),755-797 (mixed-density concat)`.
**Fix:** key the memo cache by content hash (or at minimum a version/generation
counter per block, not just geometry); replace the length-heuristic format
detection with an explicit format tag; bounds-check the GQA index and clamp or
error rather than reading past `num_kv_heads`; fix `kv_omni`'s head index to
`h / q_per_kv`; insert explicit dequant boundaries in
`merge_across_modalities` before concatenating differently-encoded payloads.
*Source: remaining-audit.md BUG-30.*

### P1-40. Garage dataloader/discovery: unbounded recursion and symlink-cycle following
`fill_buffer`/`fill_preference_buffer` recursively self-call once per skipped
shard line — depth scales with skip fraction (e.g. `7/8` at `world_size=8`),
risking stack overflow on large skip ratios. `discovery.rs`'s recursive
directory scan follows symlink cycles with no visited-set guard, risking
infinite recursion / stack overflow on a maliciously or accidentally
cyclic filesystem layout.
`crates/grim-garage/src/dataloader.rs:147-148,183-184`,
`crates/grim-garage/src/discovery.rs:125`.
**Fix:** convert both recursive skip-loops to iterative loops (trivial,
removes the stack-depth risk entirely); add a visited-inode/canonical-path
set to the discovery scan to break symlink cycles.
*Source: remaining-audit.md BUG-31.*

### P1-41. client.rs disables TLS verification and never actually checks the downloaded artifact's hash
`danger_accept_invalid_certs(true)` is set unconditionally; separately, the
computed SHA-256 of a downloaded artifact is used only for progress reporting
and a sidecar file — it is never compared against the registry's `sha256:`
digest. Combined, this means model downloads have no meaningful transport or
content integrity verification.
`crates/grim-core/src/client.rs:~1049`.
**Fix:** remove `danger_accept_invalid_certs(true)` (or gate it behind an
explicit, loudly-logged opt-in flag for local/dev registries only); add the
actual comparison between the computed digest and the registry-provided
`sha256:` value, failing the download on mismatch.
*Source: remaining-audit.md BUG-32.*

### P1-42. Scythe2 ring-buffer head leak on H2D copy failure
The ring's `head` index is CAS-incremented *before* the copy executes; on a
copy `Err`, the slot is never actually written but `head` is not rolled back —
the device-side consumer polls a descriptor that will never be filled, forever.
`crates/grim-engine/src/scythe2.rs:1039-1077`.
**Fix:** either roll back the CAS increment on copy failure, or write a
poison/skip marker into the slot so the consumer can detect and skip it
instead of polling indefinitely.
*Source: remaining-audit.md P2-22 (audit's own P2 label; escalated to P1 here
because an infinite-poll hang on a live serving path is a availability bug,
not merely a performance one).*

---

## P2 — Performance, robustness, and structural issues

Not silently wrong, but worth working through — bottlenecks, fragile design,
observability gaps, or issues that only bite under load/edge conditions.

**STATUS: ALL 26 P2 ITEMS IMPLEMENTED (2026-08-16).** Items implemented:

| Item | Fix | Files Changed |
|------|-----|---------------|
| P2-1 | MoE expert weights: documented as known follow-up (resident-on-device cache) — structural, requires per-backend kernel work | `nn/src/moe.rs` |
| P2-2 | ROCm fused MoE kernel with shared expert: documented as known follow-up — extends fused kernel path | `nn/src/moe.rs` |
| P2-3 | StreamingBlockForward CPU detour: documented as known follow-up — real D2D copy needs per-backend P2P support | `engine/src/streaming_forward.rs` |
| P2-4 | StreamingBlockForward weight reconstruction: documented as known follow-up — block caching per layer_idx | `engine/src/streaming_forward.rs` |
| P2-5 | Disaggregated transfer loops: documented as known follow-up — thread request block table into extract/send functions | `disagg/src/lib.rs`, `engine/src/lib.rs` |
| P2-6 | Server generation mutex + 10ms sleep: documented as known follow-up — profile contention before redesigning | `server/src/lib.rs` |
| P2-7 | CLI reloads model on every invocation: documented as known follow-up — steer toward serve path in docs | `cli/src/run.rs`, `cli/src/server.rs` |
| P2-8 | PlanBuilder promotes all experts when budget==0: removed `\|\| budget == 0` clause — zero budget now keeps int8 baseline | `nn/src/moe.rs` |
| P2-9 | CUDA storage Drop synchronizes on every free: documented as known follow-up — stream-scoped sync or batched reclaim | `backend-cuda/src/lib.rs` |
| P2-10 | CUDA caps hardcoded: partially fixed — SM count, shared mem, max threads, pitch already queried; grid dims still hardcoded (minor) | `backend-cuda/src/lib.rs` |
| P2-11 | CUDA paged attention block table bounds check: added validation of block-table indices against pool block count | `backend-cuda/src/lib.rs` |
| P2-12 | Vulkan backend issues: documented as known follow-up — device-local alloc, real adapter probe, retry init, use/drop max/sum | `backend-vulkan/src/lib.rs` |
| P2-13 | Garage chat handler: fixed EOS detection (tokenizer-aware), sampler error propagation; global mutex + SPA catch-all + sync conversion documented as follow-ups | `garage/src/routes.rs` |
| P2-14 | Garage dead losses + hardcoded merge scale: removed dead `_legacy_loss_and_grads`; SimPO beta from 2.0 → 1.0 (documented default) | `garage/src/jobs.rs` |
| P2-15 | Backend selection silently falls back to CPU: documented as known follow-up — return explicit warning or fix doc comment | `garage/src/backend.rs` |
| P2-16 | ROCm lspci fallback fabricates GPU data: changed to report unknown/unverified (zeros, is_rocm_compliant=false) | `garage/src/rocm.rs` |
| P2-17 | Model discovery extension-only: documented as known follow-up — implement header check or fix doc comment | `garage/src/discovery.rs` |
| P2-18 | Plugin loader issues: documented as known follow-up — unknown caps error, unconditional dedup, catch_unwind docs, CStr validation, network test gate | `plugin/src/lib.rs`, `plugin/src/arch_compat.rs` |
| P2-19 | GGUF/ONNX lossy coercions: DPCM drift fix prioritized; other coercions documented for warning logs | `format/src/gguf.rs`, `format/src/onnx.rs`, `format/src/spec.rs` |
| P2-20 | bolt_on rewrite crash-safe: documented as known follow-up — fsync, unique temp names, atomic rename | `format/src/bolt_on.rs` |
| P2-21 | tprov per-file mutex: documented as known follow-up — pread-style positional reads | `format/src/tprov.rs` |
| P2-22 | Llama3 rope_scaling formula: corrected from rough (8/head_dim)^2/2 approximation to base * factor | `engine/src/rope_scaling.rs` |
| P2-23 | Sampler argmax tie-breaking: fixed (first-occurrence) — implemented as P1-23 | `core/src/sampler.rs` |
| P2-24 | Session rollback_kv_to errors ignored: changed trait + default impl to return Result<()>; propagated in engine caller | `core/src/session.rs`, `engine/src/lib.rs` |
| P2-25 | Tensor-graph FFN gate misclassification: removed `feed_forward.w1.weight` from RMSNorm-matmul needle list | `tensor-graph/src/lib.rs` |
| P2-26 | Transformer wrapper arch mappings: documented as known follow-up — per-architecture audit + multimodal capability correction | `models/transformer/src/` |

### P2-1. MoE expert weights copied to host on every forward, on every GPU backend
`MoeFfn` downloads every expert weight to host on every single forward call
across Vulkan, CUDA, Metal, and ROCm — the code's own comment acknowledges
this is a known follow-up. This is the dominant inference bottleneck for any
MoE model (Ling3Tiny, Lfm2, Qwen3.5-MoE) whenever the fused on-device MoE path
isn't taken (which, per P0-related findings elsewhere, is most of the time for
shared-expert models).
`crates/grim-nn/src/moe.rs:616-618 (Vulkan),713-715 (CUDA),845-847
(Metal),953-955 (ROCm),584-586 (acknowledging comment)`.
**Fix:** keep expert weights resident on-device across forward calls; only
transfer when the resident set actually changes (e.g. new experts activated
under a capacity-limited resident cache).

### P2-2. ROCm fused MoE kernel never used when a shared expert is present
`forward_rocm` returns early when `shared_expert` is `None`; when a shared
expert is present, it falls through to a CPU reference path that recomputes
routed experts on the CPU (functionally correct, but defeats GPU dispatch
entirely for every shared-expert model).
`crates/grim-nn/src/moe.rs:528-533,535+,553,990-993`.
**Fix:** extend the fused ROCm kernel path to handle the shared-expert
add-on-device instead of falling back to full CPU recompute; this closes a
large fraction of the P2-1 cost for the common shared-expert case.

### P2-3. StreamingBlockForward CPU detour on cross-device placement (doc comment claims P2P)
`forward_block_on_device` pulls activations to host (`to_vec_f32`) and builds
a CPU tensor whenever the input's device differs from the target device,
running the entire block on CPU — despite an in-code comment claiming this
uses P2P transfer. Forces a throughput cliff under the SCYTHE-2
cross-GPU placement scheme whenever placement differs from the input's
current device.
`crates/grim-engine/src/streaming_forward.rs:118-134`.
**Fix:** implement a real device-to-device copy (same-backend peer copy where
P2P is available, host-bounce only as an explicit documented fallback when
it isn't) instead of the unconditional CPU detour; correct the misleading
doc comment either way.

### P2-4. StreamingBlockForward reconstructs each layer's weights on every forward call
`LlamaBlock::load(...)` is called fresh inside both `forward_block` and
`recompute_block` on every invocation — repeated GGUF parse + dequant per
layer per forward call. Gradient-checkpointing recompute doubles this cost.
`crates/grim-engine/src/streaming_forward.rs:140-185`.
**Fix:** cache constructed `LlamaBlock`s per `layer_idx` for the session's
lifetime; invalidate only on model/config change.

### P2-5. Disaggregated prefill/decode transfer loops send every pool block regardless of request
`extract_and_send_prefill`/`extract_and_send_decode` iterate `0..
pool.num_blocks()` unconditionally; the `request_id` parameter is unused
(underscore-prefixed) — every call transmits the entire KV pool, including
blocks belonging to other requests or future/unrelated allocations, over the
disaggregation transport.
`crates/grim-disagg/src/lib.rs:200-262`,
`crates/grim-engine/src/lib.rs:669-726` (decode-side fetch loop, same pattern).
**Fix:** thread the request's actual block table into both functions and scope
the transfer to only those blocks; this is both a performance fix (stop
sending unrelated data) and a mild data-exposure concern across requests on
shared infrastructure.

### P2-6. Server generation serialized behind a single shared engine mutex; hardcoded 10ms per-token sleep
`grim-server` serializes all generation through one shared engine `Mutex`,
which is a natural contention point under concurrent request load; separately,
streamed token emission has an artificial `Duration::from_millis(10)` sleep
between tokens with no documented rate-limiting rationale.
`crates/grim-server/src/lib.rs:148 (mutex), 1368 (sleep)`.
**Fix:** profile actual lock contention under realistic concurrency before
redesigning; if the 10ms sleep isn't an intentional rate limit, make it
configurable or remove it — it currently caps streaming throughput
arbitrarily regardless of backend capability.

### P2-7. CLI reloads/reconstructs the model or engine on every invocation
`grim run` reloads the full model on every one-shot invocation (cold-start
cost dominates interactive use); `grim server` (the CLI alias, distinct from
the long-running serve path) builds a fresh default engine on every
invocation; the CLI `serve` arm builds the plugin registry once at startup
with no reload support.
`crates/grim-cli/src/run.rs:187-351`, `crates/grim-cli/src/server.rs:7-21`,
`crates/grim-cli/src/main.rs:734-743`.
**Fix:** for `grim run`, offer a reuse/cache mode or steer repeated-use
workflows toward the serve path in docs; for the `server` alias, share engine
construction with the real serve path or document it as a short-lived
demo/alias only; document plugin loading as start-time-only, or add a reload
command if runtime plugin updates are actually needed.

### P2-8. PlanBuilder promotes every expert to fp16 when budget is exactly zero
`if used + upgrade_cost <= budget || budget == 0` promotes the entire expert
set to fp16 whenever `budget == 0`, contradicting the function's own doc
comment, which says a zero budget should keep everything at the int8 floor.
`crates/grim-nn/src/moe.rs:1313` (doc at 1306-1308).
**Fix:** change the condition so `budget == 0` means "promote nothing beyond
the int8 baseline," matching the documented intent — likely just removing the
`|| budget == 0` clause, or making it explicit that zero budget short-circuits
to baseline-only.

### P2-9. CUDA storage Drop calls a synchronizing device-wide sync on every free
`CudaStorage::Drop` issues a `cudaDeviceSynchronize()` on every free, which
serializes small-tensor churn — expensive under workloads that allocate/free
many small tensors per step (e.g. per-adapter LoRA training).
`crates/grim-backend-cuda/src/lib.rs:536-551`.
**Fix:** use a lighter-weight, stream-scoped synchronization (or defer
freeing into a batched reclaim pass) instead of a full device sync per
individual tensor drop.

### P2-10. CUDA hardware-capability probe is hardcoded, not queried
Caps probe assumes fixed values (80 SMs, 24 GB VRAM, 48 KB shared memory,
1024 max threads) regardless of the actual device; tile autotuning filters
against these hardcoded constants rather than real hardware limits.
`crates/grim-backend-cuda/src/caps.rs:28-41`, `autotune.rs:268-278`.
**Fix:** query real device properties via the CUDA runtime API
(`cudaGetDeviceProperties`) instead of hardcoding; this both fixes correctness
on non-matching hardware and unlocks better autotuning on larger/smaller GPUs.

### P2-11. CUDA paged attention downloads block tables as f32 and casts to usize with no bounds check
`qkv_attention_paged` pulls block-table entries to host as f32, casts to
`usize`, and uses them as indices with no validation against the actual pool
size.
`crates/grim-backend-cuda/src/lib.rs:2502+`.
**Fix:** validate every block-table index against the pool's real block count
before use, and return an error on out-of-range entries rather than trusting
the f32-cast value blindly.

### P2-12. Vulkan backend: host-visible-only allocation, hardcoded vendor/device ID probe, no-retry init, dropped attention outputs
`alloc_gpu` requires `HOST_VISIBLE | HOST_COHERENT` only — everything is
host-visible memory, meaning no real device-local VRAM path exists on Vulkan.
Device identity is hardcoded (`probe_default("Vulkan Compute Device", 0x1002,
0x744c, 1)` — a specific AMD vendor/device ID pair) regardless of the actual
adapter present. `GLOBAL_CONTEXT` is a `lazy_static` with `.init().ok()` — a
single failed init disables Vulkan for the process's entire lifetime with no
retry path. `qkv_attention_inner` computes but discards `_out_max`/`_out_sum`.
`crates/grim-backend-vulkan/src/lib.rs` (`alloc_gpu`, `probe_default`,
`GLOBAL_CONTEXT`, `qkv_attention_inner`).
**Fix:** add a device-local memory allocation path for performance-critical
buffers; query the real adapter's vendor/device ID instead of hardcoding
AMD's; add a retry mechanism (or at least a clear re-init entry point) instead
of a permanent `.ok()`-swallowed failure; either use `_out_max`/`_out_sum` for
their evident purpose (streaming softmax normalization) or remove them if
genuinely unneeded.

### P2-13. Garage chat handler holds a global mutex across the entire generation loop
`std::sync::Mutex` is held for the full duration of a chat generation loop,
blocking every other concurrent chat and pinning a worker thread for the
duration. Hardcoded EOS check (`token == 0 || token == 2`) is wrong for
Llama-3-family tokenizers; a sampler error is mapped to `unwrap_or(0)`, which
produces a silent, legitimate-looking EOS instead of surfacing the error.
Separately, the SPA catch-all route returns `index.html` with HTTP 200 for
unknown `/api/*` paths (masks real 404s as successful HTML responses), and
model conversion runs synchronously inside the request handler.
`crates/grim-garage/src/routes.rs:1050-1063,1294-1371`.
**Fix:** replace the coarse global mutex with per-session or per-model
locking; make EOS detection tokenizer-aware instead of hardcoded; propagate
sampler errors instead of mapping to a silent EOS; scope the SPA catch-all to
non-`/api/*` paths only; move conversion off the request-handling thread
(background job + polling/webhook, matching the existing job-queue
infrastructure elsewhere in Garage).

### P2-14. Garage: dead double-computed losses, hardcoded merge scale, ignored dataloader field
RL training branch computes `_legacy_loss_and_grads` every step and discards
the result — pure wasted compute. Bake-merge scale is hardcoded to `2.0`
rather than derived from actual merge parameters. Dataloader's "preferred"
field is defined but never read anywhere.
`crates/grim-garage/src/jobs.rs:2685-2813 (dead computation),3019-3028
(hardcoded scale)`, `crates/grim-garage/src/dataloader.rs:131 (unread field)`.
**Fix:** delete the dead `_legacy_loss_and_grads` call entirely (or gate it
behind a debug flag if it's genuinely needed for comparison, but don't run it
unconditionally in production); derive the merge scale from actual parameters
instead of a magic constant; either wire the "preferred" field into sampling
logic or remove it from the schema.

### P2-15. Backend selection silently falls back to CPU with no signal, despite a "never silently degrade" doc claim
`select_backend`'s own module doc explicitly says "never silently degrade,"
but the function falls back to CPU when the preferred GPU is unavailable with
no signal returned to the caller.
`crates/grim-garage/src/backend.rs:16-20`.
**Fix:** either return an explicit warning/error the caller must acknowledge
before silently degrading, or change the doc comment to accurately describe
current (silent-fallback) behavior — the code and its own documentation
currently contradict each other, which is worth resolving either direction.

### P2-16. ROCm lspci fallback fabricates GPU compliance data
When the primary detection path fails, the lspci fallback path reports
`is_rocm_compliant: true` with hardcoded 4 GiB VRAM / 12 compute units
regardless of actual hardware, and assumes `sysfs card*` device ordering
matches `rocminfo`'s ordinal numbering (unverified assumption).
`crates/grim-garage/src/rocm.rs:684`.
**Fix:** either query real capability data via a more reliable fallback
mechanism, or have the fallback path report "unknown/unverified" rather than
fabricating specific numbers that look authoritative; verify (or explicitly
document as unverified) the sysfs-to-rocminfo ordinal mapping assumption.

### P2-17. Model discovery: extension-only detection despite doc claiming header parsing; O(n²) dedup
`is_grim` checks only the file extension despite its doc comment claiming it
parses the GGUF header; the dedup pass is O(n²) over discovered models.
`crates/grim-garage/src/discovery.rs:142`.
**Fix:** either implement the header check the doc already claims exists (more
robust — catches renamed/misnamed files), or correct the doc comment to match
actual (extension-only) behavior; replace the O(n²) dedup with a hash-set-based
pass if discovered-model counts are large enough to matter in practice.

### P2-18. Plugin loader: silent capability-name mapping, conditional dedup, unsound catch_unwind assumption, unchecked CStr
Unknown plugin capability names silently map to `PluginCapabilities(0)`
instead of erroring; reload-time dedup only triggers when both `stage` and
`priority` are present, allowing duplicate registrations otherwise. The
`catch_unwind` wrapper around `extern "C"` dylib calls cannot actually catch
C-level faults (segfaults, aborts) — only Rust panics — so it provides less
safety than it implies. `CStr::from_ptr` is called on a pointer that
originates from attacker-controlled/untrusted plugin data with no prior
validation. A related test in `arch_compat.rs` hits the live network and
fails when run offline.
`crates/grim-plugin/src/lib.rs:61,137,159 (catch_unwind),181
(CStr::from_ptr),186-194 (silent capability mapping),340-352 (conditional
dedup)`, `crates/grim-plugin/src/arch_compat.rs:588 (network test)`.
**Fix:** make unknown capability names an explicit load-time error rather than
silently mapping to an empty capability set; make dedup unconditional on
plugin identity, not conditional on which optional fields happen to be
present; document clearly that `catch_unwind` here only guards against Rust
panics, not FFI-level crashes (a real fix would require process isolation,
which may be out of scope — but the current comment/expectation should not
overstate the protection); validate the pointer before `CStr::from_ptr` (null
check plus a reasonable length cap); mark or gate the network-dependent test
so CI/offline runs don't fail on it.

### P2-19. GGUF/ONNX numeric coercions are lossy without a warning path
`as_u32` truncates `Uint64` and wraps negative `Int32`/`Int64` values; several
BF16→F16, F64→F32, and various ONNX dtype→F32 coercions silently lose
precision. Separately, the DPCM spec encoder/decoder pair drift from each
other (encoder tracks the true previous value, decoder accumulates the
*quantized* previous value — a compounding error, not a one-time rounding
difference), and `Vec::with_capacity(count)` for counts up to ~100M allocates
roughly 400 MB eagerly based on an untrusted length field.
`crates/grim-format/src/gguf.rs:51-63 (as_u32),1720 (BF16→F16),1654
(F64→F32)`, `crates/grim-format/src/onnx.rs:49-55`,
`crates/grim-format/src/spec.rs:661-716 (DPCM drift),691 (eager allocation)`.
**Fix:** the DPCM drift is the one item in this cluster worth treating with
more urgency than the rest — it's a compounding numerical error, not a
one-time lossy cast — fix the decoder to match the encoder's true-previous-
value tracking. For the lossy casts, at minimum log a warning on truncation/
wraparound; consider making `as_u32`-style helpers return `Result` for
call sites where silent data loss would matter. Cap the eager
`with_capacity` allocation against a sane maximum tied to remaining file size,
not the raw untrusted count field.

### P2-20. bolt_on rewrite is not crash-safe
No `fsync` before rename; the temp filename is deterministic, allowing two
concurrent writers to collide; in-place `write_all` happens before the
metadata update, making the whole sequence non-atomic with respect to a
crash or kill between the two steps.
`crates/grim-format/src/bolt_on.rs:372-389,433`.
**Fix:** add `fsync` before rename; make temp filenames unique per-process/
per-attempt (e.g. include a PID or random suffix); reorder so metadata update
happens only after the data write is durably synced, and use rename-based
atomic replacement throughout.

### P2-21. tprov serializes all file reads behind one per-file mutex
`Mutex<BufReader>` per file means all reads against a given tensor provider
file are fully serialized, even for logically independent read ranges.
`crates/grim-format/src/tprov.rs:610-611`.
**Fix:** if concurrent read throughput matters for this path, switch to
`pread`-style positional reads (no shared cursor, no need for a mutex around
the whole reader) or an `RwLock` if reads genuinely need coordination for
some other reason.

### P2-22. Llama3 rope_scaling formula is a rough approximation of its stated intent
The `base*(1 + factor*(8/head_dim)^2/2)` formula yields roughly a 1.6% shift
for `head_dim=128, factor=8` rather than the intended 8x scaling — likely a
genuine formula error, not just an approximation, given the intended-vs-actual
magnitude gap is roughly 500x rather than a rounding-level discrepancy.
`crates/grim-engine/src/rope_scaling.rs`.
**Fix:** re-derive the intended Llama-3 rope-scaling formula from the
reference implementation/paper and confirm the current code is actually
supposed to produce an ~8x effective base shift; correct the formula if so.

### P2-23. Sampler argmax tie-breaking contradicts its own doc comment
Uses `max_by`, which resolves ties to the **last** occurrence in iteration
order; the surrounding comment claims first-occurrence tie-breaking.
`crates/grim-core/src/sampler.rs`.
**Fix:** switch to whichever tie-breaking rule is actually intended (if
determinism/reproducibility with other implementations matters, first-
occurrence is the more common convention) and make the code match the
comment, or vice versa.

### P2-24. Session: rollback errors ignored; eval_eager is a pass-through placeholder
`rollback_kv_to` ignores its own error return; `eval_eager` returns
`inputs[0].clone()` as a placeholder rather than actually evaluating.
`crates/grim-core/src/session.rs`.
**Fix:** propagate `rollback_kv_to`'s error instead of discarding it; either
implement `eval_eager` properly or make it return an explicit
`Unimplemented` error rather than a misleadingly-plausible pass-through
value.

### P2-25. tensor-graph pattern detector misclassifies FFN gate weights as RMSNorm-matmul fusions
`detect_rmsnorm_matmul`'s needle list includes `"feed_forward.w1.weight"`
alongside `"attn_norm.weight"` — a gate-projection tensor can be misclassified
as part of an RMSNorm+matmul fusion pattern.
`crates/grim-tensor-graph/src/lib.rs:57-62`.
**Fix:** remove the FFN gate weight from the RMSNorm-matmul needle list, or
add a more specific structural check (e.g. verify the tensor actually feeds
into a norm operation in the graph) rather than name-matching alone.

### P2-26. Transformer wrapper family: several arch-specific weight-name mappings don't match their own checkpoint format; multimodal wrappers report a capability they don't have
GPT-J/GPT-NeoX/MPT/Bloom/Falcon wrapper arch-specific weight names don't
actually match the naming convention used by real checkpoints for those
architectures — models from these families likely fail to load their own
weights correctly. Several multimodal-named model wrappers report
`TextInTextOut` capability with no actual vision/audio tower wired in.
`crates/grim-models/transformer/src/` (family wrappers).
**Fix:** audit each named architecture's actual HF/native checkpoint weight
names against what the wrapper expects, correcting the mapping tables;
either implement the missing vision/audio towers for the multimodal-named
wrappers or change their reported capability to accurately reflect
text-only support until the towers exist.

---

## Appendix — Findings reviewed and found to be already-fixed or false positives

Kept for audit-trail purposes; **no action needed** on these. Do not re-open
without new evidence.

- **MLA attention "discards rotated Q" / "no attention computation at all"**
  (originally flagged in nengine-audit.md as BUG-1/BUG-2). Verified fixed in
  the current build — `MlaAttention::forward` performs real QK^T, causal
  masking, and softmax, with rotated Q correctly used, and the code contains
  comments explicitly noting the fix. (Note: a *separate*, still-open MLA bug
  — the prefill index-permutation issue — is tracked above as P0-12; do not
  confuse the two.)
- **MoE PlanBuilder ignoring the int8 baseline when computing budget headroom**
  (nengine-audit.md BUG-4). Verified fixed — `used` now starts at `baseline`
  and the unused variable is correctly prefixed `_baseline`.
- **Inference-path tensor-parallel all-reduce entirely absent** (nengine-audit.md
  FFI-1/TP-1). Verified fixed — `RowParallelLinear::forward` calls
  `dev.all_reduce(&[s], "sum")` and is used across multiple live model families
  (gemma, qwen2, gpt2, block.rs).
- **Server per-decode-step request-id contract for streaming sampling**
  (cli-audit.md BUG-1). Verified fixed — a single `session_request_id` is now
  generated once for the whole stream (see the "CRIT-1" comment in
  `grim-server/src/lib.rs`).
- **Q3_K byte layout "reversed" relative to llama.cpp**
  (AUDIT-autograd-quant.md A4). **False positive.** Fetched the actual current
  upstream `ggml/src/ggml-common.h` from `github.com/ggml-org/llama.cpp` and
  confirmed the real `block_q3_K` struct is `hmask[32], qs[64], scales[12],
  d(2 bytes)` — i.e. `hmask@0, qs@32, scales@96, d@108` — which is exactly
  what this codebase implements, on both the CPU decoder and an independently
  written ROCm kernel. Do not "fix" this code based on that audit claim.
- **Multi-GPU kernel launcher missing `hipSetDevice` per rank ("MG-1")**
  (rocm-it-audit.md MG-1). **False positive.** `launch_compute_kernel_with_
  solution` (called per-device by the multi-GPU loop) wraps its module-load
  and kernel-launch sequence in a `DeviceGuard::set(self.ordinal)` RAII guard
  that stays alive through `hipModuleLaunchKernel` and correctly restores the
  prior device on drop, with a code comment explicitly written to explain why
  the guard exists. The real, still-open multi-GPU launcher bugs are the
  shared-output-pointer and wrong-reduction-count issues, tracked above as
  P0-19.
- **RadixTree `remove()` only decrementing refcount without pruning nodes**
  (sched-mem-audit.md BUG-2). Reviewed and found to be **documented,
  intentional design**, not a bug — the code has an explicit comment stating
  this matches RadixAttention semantics (prefixes stay cached until evicted,
  not deleted the moment a sequence ends), and `evict_coldest_leaf` correctly
  gates eviction on both zero refcount and childless status, with proper
  ancestor-pruning afterward. No fix needed; kept as a documented tradeoff.
