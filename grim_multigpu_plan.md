# Implementation Plan: Multi-GPU (Symmetric + Asymmetric) Training in `grim-garage`

## Findings confirmed against source (baseline for this plan)

All claims below were verified directly against the uploaded `grim` source tree before this plan was written.

- `run_training_worker` (`crates/grim-garage/src/jobs.rs`) calls `select_backend()` exactly once, producing a single `Device`. Model weights (`WeightSource::root`) load onto that one device regardless of `job.num_gpus`. **There is no per-GPU rank today — one device runs the entire job.**
- `RcclAllReduce::new(num_gpus)` (`crates/grim-backend-rocm/src/rccl.rs`) builds `devlist: Vec<i32> = (0..ndev).collect()` with no call to `hipGetDeviceCount`/`enumerate_devices()` first — ordinals `0..num_gpus` are assumed to exist, never checked against what's actually installed.
- `ncclCommInitAll` failure degrades to `comm: None` behind a `log::warn!`, and the training loop's `all_reduce_grads(...)` call result is discarded via `let _ =` at `jobs.rs:1116` — a failed or degraded collective is invisible to the operator.
- `C2plrController`'s partition math is hardcoded equal-share: `vec![1.0 / num_gpus as f32; num_gpus]` — no capability signal factors in. This is the concrete site of the "no asymmetric scheduling" gap.
- Real, unwired building blocks already exist and should be reused rather than rebuilt:
  - `peer_access::enumerate_devices()` — wraps `hipGetDeviceCount`.
  - `hipGetDeviceProperties` / `gcnArchName` bucketing (`GcnArch`) — architecture identification per device.
  - VRAM probing (`hipMemGetInfo`).
  - A P2P/PCIe/Host link-verdict probe purpose-built for "what kind of link do I have between these GPUs" (`peer_access.rs`, doc-commented as "the apparatus for asking... before I try a P2P memcpy / RCCL collective").
  
  This is meaningfully more infrastructure than "scaffolding" — it's real, tested primitives that are simply never called from the training path.

## Phase A — Enumeration and validation (prerequisite for everything else)

**Goal:** `run_training_worker` knows, before doing any work, exactly which GPUs exist and what they can do — and refuses to proceed on a bad configuration instead of silently degrading.

1. Add `fn enumerate_training_gpus() -> Result<Vec<GpuInfo>>` (new, in `grim-backend-rocm` alongside `peer_access.rs`) that wraps `enumerate_devices()` plus per-device `hipGetDeviceProperties` to produce a small struct: `{ ordinal, gcn_arch, vram_bytes, compute_units }` (confirm exact fields cheaply available from `hipDeviceProp_t` before finalizing the struct).
2. In `run_training_worker`, **before** constructing `RcclAllReduce`, call this and validate `job.num_gpus <= enumerated.len()`. If not, fail the job immediately (`JobStatus::Failed`) with an explicit message — replacing today's silent "ordinals 0..num_gpus assumed to exist" behavior.
3. Reject `job.num_gpus > 1` requests that don't request/aren't backed by ROCm. Today only `backend.label == "rocm"` gates the RCCL handle construction, but nothing stops a CUDA/Vulkan job with `num_gpus: 2` from silently running single-GPU with no error at all. Make that combination fail loudly too, consistent with the "fail fast" pattern already established for CPU + `ResidualPacked` elsewhere in the project's planning.
4. Surface the RCCL failure path: change the `let _ =` discard on `all_reduce_grads` at `jobs.rs:1116` to a real match — on error, either fail the job or (if a documented "degrade to local-only" mode is intentionally supported) log at `warn` *and* surface it in job status/metrics so it's visible to the operator, not just the process log.

## Phase B — Per-rank replica construction (the actual missing architecture)

**Goal:** one model replica per selected GPU, not one shared device.

1. Refactor the single `backend`/`backend.device` construction in `run_training_worker` into a `Vec<Backend>` (or a new `Rank` struct wrapping `{ device, ordinal, weight_share: f32 }`), one entry per enumerated-and-validated GPU from Phase A.
2. Load the base model (`GgufProvider::open`, `WeightSource::root`, the `sft_base` tuple construction) once per rank, onto that rank's device. This is the most invasive change in the plan — today `sft_base` is a single tuple; it needs to become `Vec<sft_base>` or equivalent, with the same GGUF file reopened/reloaded per device (or read once host-side and uploaded N times, if cheaper — worth profiling given large models, but correctness first per the project's own gate ordering: correctness → compile → architecture-cleanliness → performance).
3. Same for the `AutogradRegistry`/LoRA injection registry — one instance per rank, each independently tracking its own adapter weights, which must be **kept identical across ranks** after each all-reduce + optimizer step (standard data-parallel semantics — see open question below).

## Phase C — Batch sharding across ranks

1. Extend the `JsonlBatchIterator`/dataloader wiring (`dataloader.next_batch()`) to shard: rank `i` of `N` reads every `N`th example (or a pre-partitioned index range) so each replica trains on disjoint data per step. Standard data-parallel batch sharding; not currently present since there's only one dataloader instance today.
2. For the **asymmetric** case (e.g. a 9070 + 9060 pairing), shard proportional to `weight_share` from Phase B's `Rank` struct rather than equal `1/N` splits — a faster card gets a larger micro-batch or more gradient-accumulation steps per synchronization round. This directly replaces the hardcoded `vec![1.0 / num_gpus as f32; num_gpus]` partition math with a capability-weighted one.

## Phase D — Per-rank forward/backward + gradient sync

1. Run the existing forward/backward/loss computation (already real — GGUF-loaded weights, real dataset batches, real DPO/SFT loss) independently on each rank's device, in parallel (likely via `tokio::spawn` per rank or a thread-per-GPU model, given HIP streams are already per-device in this codebase per `peer_access.rs`'s "one stream per device" skill attribution).
2. After each rank computes local gradients, call `all_reduce_grads`. This part of the pipe is real and mostly correct already — `ncclAllReduce`/`hipMemcpyPeerAsync` are genuinely invoked, not stubs — but it needs to run across the **actual N constructed ranks** from Phase B, not as a single call from a lone-device worker pretending N ranks exist.
3. For asymmetric capability weighting on the *gradient* side (not just batch sharding): confirm whether a naive `ncclAllReduce` sum is still correct when local batch counts differ per rank — it usually needs a weighted average, not a plain sum, when batch sizes diverge across ranks. This is a real numerical-correctness question, not just a scheduling one, and should get its own explicit answer/test before landing.

## Phase E — Validation and observability

1. Add the "does selected device count agree with RCCL communicator" check explicitly — after `RcclAllReduce::new`, confirm `handle.num_gpus` (or an equivalent post-init query) matches the number of ranks actually constructed in Phase B, and fail if not. This covers driver-level partial failures where `ncclCommInitAll` might succeed but bind fewer ranks than requested (confirm NCCL's actual failure semantics here before assuming this check is reachable in practice).
2. Add integration tests (hardware-gated, per the project's `TODO(gpu-verify)` convention) for:
   - (a) 2x symmetric same-model cards producing loss curves that converge equivalently to the single-GPU baseline.
   - (b) An asymmetric 2-card configuration completing a job without erroring and producing a reasonable relative step-time split.
   - (c) `num_gpus` exceeding available hardware failing fast with a clear message.
   - (d) An injected RCCL init failure surfacing as a job failure or a visibly-logged degraded-mode status, not a silent pass.

## Suggested sequencing

Phase A is small, standalone, and should land first regardless of anything else — it's the fail-fast layer and doesn't require the bigger replica refactor.

**Phase A → Phase B → Phase C (can partly parallelize with D) → Phase D → Phase E.**

Phase B is the largest, most invasive change (turning a single-device worker into a multi-replica one) and is worth landing as its own reviewable unit with symmetric-only support first, before layering Phase C's capability-weighted sharding on top for the asymmetric case. Don't build both in one pass.

## Resolved: weight synchronization across ranks is NOT structurally guaranteed

This was flagged as an open question and has since been checked directly against `AutogradRegistry`/`TrainableParams` internals (`crates/grim-autograd/src/param.rs`). Confirmed findings:

1. **`TrainableParams::all_reduce_grads` operates on a single registry's `HashMap<ParamId, TrainableParam>`.** Under Phase B (one `AutogradRegistry` instance per rank), this function has no way to reach into any other rank's registry. As written, it can only act on gradients already visible to the one device pointer it holds — it is not a multi-rank-aware function today, despite living on a struct that Phase B intends to instantiate once per GPU.
2. **It only touches `param.grad`, never a weight/value field.** Even when the RCCL fast path correctly all-reduces a given tensor's gradient (confirmed real: `ncclAllReduce` via `sum_gradients_device`, then an on-device `mul_scalar` to convert sum→mean), nothing in this file applies the resulting optimizer step identically across ranks or verifies ranks remain in agreement afterward. Replica consistency is presumed to be an emergent property of "same averaged gradient + same optimizer + same starting weights," not something the code enforces or checks.
3. **The CPU/no-RCCL fallback branch does not reduce at all.** At `param.rs:247-253` ("Fallback: CPU-only accumulate"), each rank would call `param.accumulate_grad(&grad_tensor)` using its own **unreduced local gradient** — there is no cross-rank communication in this branch whatsoever. If this path is ever hit during genuine multi-rank training (RCCL unavailable, feature disabled, or `rccl_handle` is `None`), every rank silently trains on its own local gradient with zero synchronization, and replica weights diverge starting at step one. This is a second, more severe bug than the outer `let _ =` result-discard already noted in Phase A — that discard at least reflects a *real* (if unchecked) reduction attempt; this fallback branch doesn't attempt reduction at all.

### Phase B.4 (new) — Per-rank weight identity guarantee

1. Restructure gradient reduction to be genuinely multi-rank-aware. Two options:
   - **(a)** Change `all_reduce_grads` to take `&mut [AutogradRegistry]` (or an equivalent multi-rank param collection) so one call reduces across every rank's gradients for a given `ParamId` in a single pass.
   - **(b)** Keep it per-registry, but make the Phase D per-rank loop responsible for having *every* rank call `all_reduce_grads` so each participates in the *same* NCCL collective — one call per rank per collective op, each contributing its own local buffer. This matches NCCL's actual expected usage pattern (collectives are inherently "each rank calls in", not "one rank acts on behalf of all"), and the current single-call, single-`ptr` design (`sum_gradients_device(ptr, ptr, count, stream)`) suggests option **(b)** is almost certainly the correct model — the function was likely written assuming a topology it doesn't actually have yet.
2. Fix the CPU fallback branch: either implement genuine cross-rank reduction (host-side gather + average, matching what `rccl_handle.scale_gradients` does on the fast path) or explicitly refuse multi-rank training when this path is reached. Silently proceeding with unsynced local gradients is worse than failing the job.
3. Add an explicit post-optimizer-step consistency check — at minimum in tests, ideally periodically in production — that compares a checksum/hash of adapter weights across ranks and fails/warns on divergence. This class of bug compounds silently over hundreds of steps before producing visibly bad output, and is exactly what the project's own "implementations that compile ≠ implementations that work" principle argues for catching explicitly rather than trusting to be correct by construction.

This sub-phase should land as part of Phase B, before Phase D's per-rank forward/backward loop is wired up — Phase D assumes a working, multi-rank-correct `all_reduce_grads` to call, and today that assumption does not hold.

## Host-backed VRAM overflow and layer residency

The ROCm path now has an opt-in overflow tier for machines whose model or
training working set exceeds device VRAM. `RocmStorage` can allocate HIP
managed memory, which is addressable by HIP kernels while HIP migrates pages
between VRAM and system RAM. The policy is global across weights, activations,
gradients, and temporary outputs:

- `GRIM_ROCM_MANAGED_ALLOCATIONS=always` forces managed allocations.
- `GRIM_ROCM_MANAGED_ALLOCATIONS=auto` selects managed memory when the live
  free-memory watermark or `GRIM_ROCM_VRAM_BUDGET_BYTES` would be exceeded.
- `GRIM_ROCM_MANAGED_WEIGHTS=always|auto` applies the same choice only to
  F32 weights materialized by `WeightSource`.

Managed storage exposes `BackendStorage::prefetch_to_device()`. The streaming
forward, recomputation, and autograd-forward paths invoke it for every block's
weights before execution, allowing layer-wise promotion without changing
kernel call sites. This is a residency policy and prefetch seam, not a promise
that all pages remain in VRAM; actual migration and throughput require ROCm
hardware validation.
