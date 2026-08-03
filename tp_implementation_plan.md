# WI-TP: Tensor Parallelism for Inference — Implementation Plan

## Status quo (verified against source)

- `ColumnParallelLinear` / `RowParallelLinear` / `TensorParallelConfig` exist and are correct
  in `grim-nn/src/modules.rs`. `RowParallelLinear::forward` reduces through
  `BackendDevice::all_reduce`, which is genuinely implemented on ROCm
  (`roc_device.rs::all_reduce` — branches cross-GPU RCCL / on-device intra-process
  accumulation / CPU fallback). This part is real and does not need rework.
- Nothing constructs these types. `grep` across the workspace outside `grim-nn` turns up
  only doc-comments referencing them, not call sites.
- `WeightSource::get` (`grim-nn/src/varbuilder.rs`) always calls
  `TensorProvider::get_packed(name)`, which always materializes the **full** tensor. The
  `TensorProvider` trait (`grim-tensor/src/provider.rs`) has no range/shard-aware read
  method at all.
- Every model block constructor (`LlamaBlock::load` in `grim-models/transformer/src/block.rs`,
  and the equivalent in gemma.rs, deepseek.rs, gpt2.rs, t5.rs, lfm2.rs) builds plain
  `Linear` via `Linear::load(ws, in_dim, out_dim, has_bias)`, with no `TensorParallelConfig`
  parameter and no per-projection sharding decision.
- `grim-engine::Engine` holds one `Device` per loaded model
  (`self.models: HashMap<String, ModelEntry>` with a single `device` field per entry, per
  `model_loader.rs`) and has no concept of a model spanning multiple devices or multiple
  cooperating `Engine`/process instances.

## Goal

Make it possible to load and serve a single model sharded column/row-parallel across N
ROCm GPUs, with `world_size` and `rank` config driving weight loading and forward dispatch,
without silently degrading to "load full model on every GPU."

## Non-goals for this plan

- Pipeline parallelism (layer-sharding across devices) — separate work item, different
  scheduling model, no code overlap worth forcing together.
- CUDA/Vulkan/Metal TP — ROCm/RCCL is the only backend with a real `all_reduce` and RCCL
  comm today; other backends' `all_reduce` impls need their own correctness pass first
  (out of scope here, flagged as a dependency below).
- Multi-node — this is single-box, multi-GPU only. RCCL comm init assumes local devices.

---

## WI-TP-1 — Sharded weight reads at the provider layer

**Why:** Loading the full weight matrix on every rank and slicing in Rust after the fact
defeats the point of TP (you'd need N× the VRAM of a single GPU to shard across N GPUs,
which is backwards). The shard decision has to happen before the bytes leave disk/mmap.

**Where:** `grim-tensor/src/provider.rs` (trait), plus every `TensorProvider` impl:
GGUF reader, safetensors reader, `.grim` reader (find via `grep -rln "impl TensorProvider"`
before starting — do not assume the file list from memory).

**What already exists (read first):** `RawTensor { bytes, shape, dtype, provenance }` is a
flat byte buffer with a shape descriptor. `get_packed` bypasses eager dequant for quantized
formats — the sharded read needs to compose with that, not bypass it, since most real
checkpoints will be Q4_K/Q5_K/etc, not F32.

**What to build:**
- Add `TensorProvider::get_packed_sharded(&self, name: &str, dim: usize, rank: usize, world_size: usize) -> Result<RawTensor>` with a default impl that falls back to
  `get_packed` + in-Rust slice (so every existing provider compiles immediately; only the
  formats worth optimizing get a real override).
- Row-major layout means sharding `dim=0` (output features, for column-parallel) is a
  contiguous byte-range slice — cheap, no format-specific work needed even for quantized
  block formats **as long as the shard boundary is quant-block-aligned** (e.g. Q4_K's
  256-weight superblocks — a shard split must land on a superblock boundary or you corrupt
  the block's shared scale/min encoding). Add a helper
  `fn shard_boundary_valid(out_dim: usize, world_size: usize, block_size: usize) -> bool`
  and fail loudly (not silently pad/truncate) when `out_dim / world_size` doesn't divide
  evenly by the quant format's block size.
- Sharding `dim=1` (input features, for row-parallel) is **not** contiguous for row-major
  `[out_dim, in_dim]` storage — it's a strided read (every row needs its middle slice). This
  needs an actual per-format reader change, not a generic byte-range slice. Scope this as
  its own gate: land dim=0 (column-parallel) first, since QKV/gate/up sharding alone
  already gets most of the memory win; dim=1 (row-parallel, for `wo`/`w_down`) is the
  harder half.

**Left/right limits:** This crate must not know about `TensorParallelConfig` (that's an
`grim-nn` concept) — pass `(dim, rank, world_size)` as plain integers, keep `grim-tensor`
backend-agnostic per existing scope fences.

**Gates:** correctness (round-trip test: shard N ways, reassemble, compare byte-for-byte
against unsharded read) → compile → quant-block-alignment fuzz test across Q4_K/Q5_K/Q6_K/Q8_0
→ perf (non-blocking): confirm peak RSS during load actually drops with `world_size`.

---

## WI-TP-2 — `WeightSource` TP-awareness

**Why:** Model constructors call `ws.get(...)`; that's the only surface they should need to
touch. Threading raw rank/world_size ints through every model file is worse than giving
`WeightSource` a mode.

**Where:** `grim-nn/src/varbuilder.rs`.

**What to build:**
- Add `tp_config: Option<TensorParallelConfig>` field to `WeightSource`, propagated through
  `pp()` (already clones prefix + fields, so this is additive, not a rewrite).
- Add `WeightSource::get_column_sharded(shape, leaf) -> Result<Tensor>` and
  `get_row_sharded(shape, leaf) -> Result<Tensor>`, both calling
  `TensorProvider::get_packed_sharded` with `self.tp_config` when set, falling through to
  plain `get()` when `tp_config` is `None` or `world_size == 1` — this keeps every existing
  single-GPU call site (`Linear::load` for embeddings, norms, lm_head, anything not
  TP-sharded) working unmodified.
- `Linear::load` gets a sibling constructor, `Linear::load_column_parallel(ws, in_dim,
  out_dim, has_bias, tp_config)` / `Linear::load_row_parallel(...)`, that calls the sharded
  getters and returns the existing `Linear` struct sized to the *local* shard
  (`out_dim / world_size` or `in_dim / world_size`) — **not** a new `ColumnParallelLinear`
  wrapper at load time. Reasoning below.

**Design decision worth stating explicitly:** `ColumnParallelLinear::forward` today does its
sharding *after* a full-size forward pass, at the activation level. Once weight loading is
shard-aware, that's wrong — the forward pass should already be operating on a correctly
undersized local weight, needing zero runtime slicing. `ColumnParallelLinear`/
`RowParallelLinear` should be redefined so `forward` on `ColumnParallelLinear` is just
"call inner `Linear::forward`" (no slicing) and `RowParallelLinear::forward` is "call inner
`Linear::forward`, then `all_reduce`" (keep the reduce, drop the input-slicing — the weight
was already loaded row-sharded, so the *input* passed to `RowParallelLinear::forward` needs
to already be the correct local-width tensor coming out of the previous column-parallel
layer, not sliced here). This is a **breaking change to the current `ColumnParallelLinear`/
`RowParallelLinear::forward` bodies** — flag it as such in the PR, don't silently leave the
old CPU-round-trip slicing as dead code alongside the new path.

**Gates:** correctness (shard N ways, run forward, compare fully-reduced output against
`world_size=1` reference within float tolerance) → compile → verify no behavior change when
`tp_config` is `None` (regression-test every existing model-load test).

---

## WI-TP-3 — Model architecture wiring

**Why:** This is where the actual "call `ColumnParallelLinear`" happens.

**Where:** `grim-models/transformer/src/block.rs` first (covers Llama-family: whichever of
Qwen/Gemma/etc. reuse `LlamaBlock` — check `configs.rs` and each model's own file before
assuming reuse, some may have bespoke blocks). Repeat per-architecture for any model file
that doesn't go through `LlamaBlock`.

**What to build:**
- `LlamaBlock::load` gains a `tp_config: TensorParallelConfig` parameter (default
  `TensorParallelConfig::default()` = single-GPU, so callers not opting into TP pass the
  default and get identical behavior to today).
- Per-projection assignment, following standard Megatron-LM TP convention (matches the
  doc-comments already in `modules.rs` — §4.1 references suggest this was the intended
  design even before wiring):
  - `wq`, `wk`, `wv` → column-parallel (each rank gets a slice of attention heads)
  - `wo` → row-parallel (each rank contributes a partial sum, all-reduced)
  - `w_gate`, `w_up` → column-parallel
  - `w_down` → row-parallel
- **GQA head-count correctness is the sharp edge here.** `num_kv_heads` may already be
  smaller than `num_heads` (grouped-query attention) before TP is even involved. Sharding
  `wk`/`wv` by `world_size` requires `num_kv_heads % world_size == 0` — when it isn't (e.g.
  8 KV heads over 6 GPUs), the standard approach is replicating KV heads onto ranks that
  would otherwise get a fractional shard, not silently flooring. Add an explicit
  `fn plan_kv_head_sharding(num_kv_heads: usize, world_size: usize) -> Result<KvShardPlan>`
  that errors clearly (not panics, not silent truncation) on configurations that need
  replication logic not yet implemented, so unsupported topologies fail at model-load time
  with a clear message instead of producing silently wrong attention output.
  `LlamaBlock::forward`'s internal head-count math (reshape into `[batch, heads, seq,
  head_dim]`) must use the **local** (per-rank) head count after this point, not the config's
  global count — audit every place `cfg.num_heads`/`cfg.num_kv_heads` is read inside
  `forward`/`forward_with_kv`, not just at construction.
- Embedding table, final norm, and `lm_head` stay unsharded (replicated per rank) initially
  — vocab-parallel embedding is a real optimization but a separate, smaller work item; don't
  couple it to this one.

**Left/right limits:** `grim-models/*` crates must not construct RCCL communicators or know
about device topology directly — `TensorParallelConfig` is a plain `{rank, world_size}`
struct, communicator setup is `grim-engine`'s job (WI-TP-4). Model crates only decide *which*
projections shard which way.

**Gates:** correctness (per-architecture: single-rank TP config must byte-match non-TP
forward output) → compile → repeat for each model architecture that has its own block
(don't assume `LlamaBlock` coverage is universal — check gemma.rs/deepseek.rs/gpt2.rs/t5.rs/
lfm2.rs each have their own attention/MLP layer construction before claiming architecture
parity) → correctness at world_size=2/4 against a CPU or single-GPU reference implementation
(needs real multi-GPU hardware — flag as `TODO(gpu-verify)` per existing project convention
until validated).

---

## WI-TP-4 — `grim-engine` orchestration and process/communicator model

**Why:** This is the structurally open-ended piece flagged earlier. Needs a decision before
code, not during.

**Where:** `grim-engine/src/lib.rs`, `model_loader.rs`, and whatever currently owns
multi-GPU device inventory / RCCL comm init for training (`grim-garage`'s
`select_backend`/rank-admission logic in `jobs.rs` is the closest existing analog — read it
first, this plan should reuse that pattern rather than inventing a second one).

**Decision to make before writing code — two real options, pick one, document why:**

1. **Multi-process, one `Engine` per rank.** Each rank is a separate OS process running its
   own `Engine` bound to one device, holding only its local weight shards, communicating via
   RCCL. This matches how vLLM/production servers do it, composes cleanly with the existing
   single-device `Engine` design (minimal changes to `Engine` itself), but needs new
   process-launch/supervision code (who spawns rank 0..N, how does an HTTP request on rank 0
   get sharded work dispatched to ranks 1..N, how do you handle a rank crashing mid-request)
   that doesn't exist anywhere in the codebase today.
2. **Single-process, `Engine` drives N devices internally.** `Engine` gains a
   `Vec<RankContext>` and `drive_forward` loops over ranks, dispatching to each device and
   collecting results. Simpler to get running for a first cut (no IPC, no process
   supervision), but doesn't parallelize *host-side* work across ranks (Rust code
   orchestrating N GPUs sequentially from one thread, even if the GPU kernels themselves run
   concurrently) and works against `grim_scheduler`'s existing per-request admission model,
   which was written assuming one device per model.

Given the project's existing patterns (`run_training_worker`'s `rank_contexts` — plural,
built once, fail-closed on validated device inventory — already exists for **training** in
`grim-garage/src/jobs.rs`), option 2's "Engine owns N `RankContext`s" is the closer fit to
reuse existing code shape, and avoids inventing IPC/process-supervision from scratch. Start
there; option 1 is the natural follow-up once inference-side TP correctness is established
and you're optimizing for the host-side dispatch bottleneck.

**What to build (assuming option 2):**
- `EngineConfig` gains `tp_world_size: usize` (default 1).
- `Engine::new`/model-load path builds one `RankContext { device, model: LlamaModel, ... }`
  per rank when `tp_world_size > 1`, reusing the admission-and-fail-closed pattern from
  `grim-garage::run_training_worker` (validate live ROCm device inventory before
  transitioning to serving-ready state — do not silently serve on fewer devices than
  requested).
- `drive_forward` loops rank contexts, calls each rank's local forward with its shard, lets
  `RowParallelLinear`'s internal `all_reduce` handle cross-rank sync (no engine-level
  reduce step needed — it's already inside the layer).
- RCCL communicator init: reuse whatever `grim-garage` already does for training rank setup
  (`select_backend` + rank admission) rather than duplicating comm-init code in `grim-engine`
  — check whether that logic can be extracted to a shared crate (`grim-backend-rocm`?) both
  `grim-garage` and `grim-engine` depend on, instead of copy-pasting.

**Left/right limits:** `grim-scheduler`'s admission/batching logic should not need to know
whether a model is TP-sharded — that's an `Engine`-internal detail. If scheduler changes
turn out to be required, that's a sign the abstraction boundary is wrong and needs
rethinking before proceeding, not a green light to leak TP concerns into the scheduler.

**Gates:** correctness (2/4-GPU end-to-end serving smoke test, output matches single-GPU
reference) → compile → architecture-cleanliness (confirm `grim-scheduler` unchanged) →
performance (non-blocking, `TODO(gpu-verify)`: confirm actual VRAM reduction and throughput
scaling, since host-side sequential dispatch in option 2 may not scale linearly — measure
before claiming a number).

---

## Sequencing and dependencies

```
WI-TP-1 (provider sharded reads)
   │  column-parallel (dim=0) lands first — cheaper, contiguous
   ▼
WI-TP-2 (WeightSource / Linear sharded constructors)
   │  redefine ColumnParallelLinear/RowParallelLinear::forward
   │  as load-time-sharded, not runtime-sliced
   ▼
WI-TP-3 (model block wiring — LlamaBlock first, then per-architecture)
   │  GQA head-count audit is the correctness-critical step here
   ▼
WI-TP-4 (Engine orchestration decision + RankContext plumbing)
   │  reuses grim-garage's existing rank-admission pattern
   ▼
   hardware validation pass (2-GPU, then 4-GPU) — everything above this
   line is checkable on a single GPU by setting world_size=1 and
   confirming zero behavior change; TP correctness itself needs real
   multi-GPU hardware and gets TODO(gpu-verify) tags until validated
```

Row-parallel (dim=1, strided) sharded reads inside WI-TP-1 can slip behind WI-TP-2/3 landing
for column-parallel only — `wq`/`wk`/`wv`/`w_gate`/`w_up` sharding alone captures most of the
per-GPU memory win even with `wo`/`w_down` still full-size-then-sliced as an interim state,
if you want a smaller first PR. Flag that explicitly as a known interim gap rather than
silently shipping partial TP as if it were complete.

## Explicit exclusions (for future reviewers — don't re-derive)

- Vocab-parallel embedding/lm_head: real optimization, deliberately deferred, not coupled to
  this plan.
- CUDA/Vulkan/Metal backends: `all_reduce` correctness on those backends wasn't part of this
  trace and shouldn't be assumed equivalent to the ROCm implementation without separately
  verifying each.
- Pipeline parallelism: different sharding axis (layers, not weights within a layer),
  different scheduling implications (pipeline bubbles, micro-batching) — treat as an
  unrelated work item, not a TP follow-on.
- Speculative decoding + TP interaction: not addressed here. The speculative KV rollback
  fix (token/block unit correctness) was verified single-GPU; whether it needs changes
  under TP wasn't traced and should get its own pass before assuming compatibility.
