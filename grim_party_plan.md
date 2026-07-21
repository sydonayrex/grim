# grim: Unsloth-Parity Training Plan

**Status:** proposed — no training loop exists in the codebase yet (verified by grep: no `backward`, `Adam`/`SGD`, or gradient-step code anywhere in the tree as of this plan).
**Author context:** written for an AI coding agent implementing against the current grim workspace. Read the "What already exists" section of each work item before writing new code — several of these are extensions of existing modules, not new subsystems.

---

## 0. Architectural thesis

Unsloth's core trick is: never materialize the full unquantized model in VRAM. Frozen base weights stay quantized; only LoRA adapters + optimizer state are kept in full precision; dequantization happens fused, just-in-time, per-op, and is thrown away immediately after use. Gradient checkpointing trades recompute for activation memory. Custom fused kernels (cross-entropy, RoPE, RMSNorm) cut both memory and kernel-launch overhead.

grim already has half of this by accident of its inference design:

- `grim-format/src/tprov.rs` reads tensors **lazily, per-tensor, from disk** (`BufReader<File>` + lazy normals/outliers stream reads) rather than eagerly loading a full model into memory.
- `grim-backend-rocm/src/kernels/fused_dequant_gemm.rs` and `kv_dequant_attention.rs` already fuse dequantization into the compute op instead of materializing a dequantized tensor.
- `grim-format/src/train.rs` already specifies a sidecar format (`model.grim.train`, magic `GRIMTRN\x01`) for adapter weights, optimizer moments, and low-rank error matrices, decoupled from the inference-only `.grim` (V1, `GRIM\x01`) file.
- `grim-memory`'s demote-before-drop tiering (host RAM / GPU / spill tiers) is a working pattern for "don't keep everything resident" that generalizes past the KV cache it was built for.
- `grim-models/transformer/src/lora.rs` already applies adapters at inference time — it just has no backward counterpart.

**The idea in this plan ("the .grim file is also the training file") is: don't build a second, separate training-time model representation.** Training reads the same quantized `.grim` file through the same lazy tensor provider used for inference, dequantizes each block just-in-time via the same fused-kernel family already built for inference, computes forward + backward for that block, discards the dequantized copy, and persists only the adapter/optimizer state to the existing `.grim.train` sidecar. The base file never gets a training-specific decompressed twin sitting in memory or on disk.

This is architecturally QLoRA-shaped (frozen quantized base, trainable low-rank adapters), but the "quantized base lives in one file read lazily" part is stronger than what Unsloth/bitsandbytes do — they still page a quantized copy of the whole model into VRAM up front. grim's version can, at least in principle, stream layer-by-layer with a bounded VRAM footprint independent of model size, at the cost of more disk/host-RAM traffic per step. That tradeoff (and whether it's actually faster than paging the whole quantized model into VRAM once, which fits on most consumer cards anyway) is an open empirical question — see §5.

## 1. Scope decision

`grim-garage/src/view_model/training_panel.rs` already lists six training modes in its UI: LoRA, QLoRA, Bf16-Full, GRPO, DPO, ORPO. These are not equal-effort:

| Mode | Needs gradient w.r.t. base weights? | Scope vs. LoRA |
|---|---|---|
| LoRA | No — base frozen | baseline |
| QLoRA | No — base frozen, quantized | baseline + streamed dequant |
| DPO / ORPO | No (if applied on top of LoRA) | baseline + reference-model forward + pairwise loss |
| GRPO | No (if applied on top of LoRA) | baseline + rollout/sampling loop + group-relative reward |
| Bf16-Full | **Yes — every parameter** | full autodiff through the whole model; the "file is the training file" streaming story gets much harder because gradients w.r.t. base weights must accumulate somewhere |

**This plan scopes WI-T1 through WI-T6 to LoRA/QLoRA only.** That is the mode where "base weights are frozen and read-only" holds, which is exactly what makes file-backed streaming without a full in-memory model tractable — you never need to write gradients back into the base tensor stream, only read it. DPO/ORPO/GRPO (WI-T7) are scoped as SFT-infra extensions once WI-T1–T6 land, since they change the loss function and add a second forward pass, not the memory model. **Bf16-Full (WI-T8) is flagged as a stretch/likely-separate-spec item** — it breaks the central memory-saving assumption this plan is built on and probably wants a different design (e.g., ZeRO-style optimizer state sharding) rather than an extension of this one. Recommend explicitly deferring a decision on Bf16-Full until T1–T6 are gated and real.

## 2. Work items

Dependency order — each item lists blocking predecessors.

```
WI-T1 (scoped autograd) ──┬──> WI-T3 (fused backward kernels) ──┐
                          │                                      ├──> WI-T5 (training loop) ──> WI-T7 (DPO/ORPO/GRPO)
WI-T2 (streaming fwd) ────┘                                      │                │
                                                                   │                │
WI-T4 (optimizer + sidecar wiring) ───────────────────────────────┘                ├──> WI-T8 (backup2 bolt-on fusion) ──┐
                                                                                     │                                     ├──> WI-T10 (garage attach/detach UI)
WI-T6 (quant-aware backward correctness) — parallel to T3, gates into T5 ───────────┘                                     │
                                                                                                                            │
WI-T9 (drop CVKG, browser frontend) — fully independent of T1–T8, runs on its own track ───────────────────────────────────┘
```

WI-T8's kernel-wiring half (backup2 dequant application) doesn't depend on T1–T7 at all — it can be built and tested with a synthetic delta the moment there's agreement on conversion-time capacity provisioning. Its attach/detach half needs a real trained adapter from T4/T5 to be meaningful end-to-end. WI-T9 depends on nothing in this plan and can start immediately. WI-T10 needs both T8 (mechanism) and T9 (a frontend to put the panel in).

---

### WI-T1: Scoped autograd for adapter-only backward

**Why:** There is currently no backward pass anywhere in grim. Building a general-purpose autodiff engine (à la PyTorch) is a large, risky undertaking and is unnecessary here — with the base model frozen, the only tensors that need gradients are the LoRA `A`/`B` matrices (and biases/norm-scale if those are ever unfrozen later). A small, purpose-built reverse-mode graph over just the trainable path is enough, and it's much easier to make correct and fast on ROCm.

**Where:** New module, e.g. `grim-nn/src/autograd.rs` or a new `grim-autograd` crate if `grim-nn` shouldn't own it (check left/right limits below — `grim-nn` currently has no backward-shaped code at all, so this is a judgment call on whether it fits `grim-nn`'s existing scope or needs its own crate).

**What already exists:**
- `grim-tensor-graph/src/ir.rs` — a **forward-only, inference-oriented fusion IR** (`ComputationGraph`, `GraphNode`, `OpType`). It exists to detect fusable op sequences for the ROCm backend, not to track gradients. Do not try to bolt backward edges onto this — it's the wrong shape (no edge/parent tracking, nodes reference tensors by name string not by graph position). Treat it as prior art for how grim represents ops, not as a base to extend.
- `grim-models/transformer/src/lora.rs::apply_adapters_to_logits` — the forward LoRA application (`y += α/r · (x @ A) @ B`). This is the function whose backward you need: given `dL/dy`, produce `dL/dA` and `dL/dB` (and optionally `dL/dx` if adapters are applied at intermediate layers rather than only at the output projection — check whether QLoRA-parity requires adapters on `q_proj`/`k_proj`/`v_proj`/`o_proj`/MLP as Unsloth does, not just the output logits; the current CPU implementation only patches logits, which is **not sufficient for real QLoRA parity** and is item 1 of what needs to change here, not just "add backward").
- `grim_core::model::AdapterHandle` — existing adapter representation; extend rather than replace.

**What to build:**
1. Extend LoRA application to standard QLoRA injection points (attention Q/K/V/O projections, MLP up/down/gate projections) — currently it's logits-only. This is a forward-side prerequisite, not backward, but it belongs in this work item since the backward graph needs to match wherever forward adapters get applied.
2. A minimal tape/graph: record just the ops touching adapter parameters during forward (matmuls, the scale-by-`α/r`, elementwise add into the frozen-base output), keep references (not copies) to whatever intermediate activations backward needs.
3. Backward implementations for exactly this op set: linear/matmul backward, elementwise-add backward (trivial, just routes gradient), scale backward.
4. Gradient accumulation buffers for `A`/`B` per adapter, per layer.

**Left-right limits:**
- Do not implement autodiff for the frozen base weights. If a future change needs that, it's WI-T8's problem.
- Do not reimplement `grim-tensor-graph`'s fusion IR or try to unify the two graphs. They serve different purposes (inference fusion planning vs. training gradient bookkeeping) and forcing a shared abstraction now is premature.
- Do not let this crate reach into `grim-backend-rocm` kernel internals directly — go through `BackendDevice` the way existing forward code does.

**Gates:** correctness (numerical gradient check against finite differences on CPU) → compiles across CPU backend at minimum → architecture cleanliness (no backend-specific code leaking into the autograd module) → perf (non-blocking initially).

---

### WI-T2: Streaming forward with gradient checkpointing, backed by the lazy tensor provider

**Why:** This is the "file is the training file" piece. Forward pass during training must read each transformer block's frozen weights from the `.grim` file the same way inference does — lazily, per-tensor — rather than through a preloaded in-memory model object.

**Where:** `grim-engine/src/model_loader.rs` (currently only has `load_model_from_grim` / `load_model_from_gguf` — full-model load functions, no streaming/step-wise variant) and `grim-format/src/tprov.rs`.

**What already exists:**
- `tprov.rs` already does lazy per-tensor reads (`BufReader<File>`, described in its own doc comment as "lazily reads normals + outliers streams"). This is the exact mechanism needed, but it was built for a **single sequential load per inference session**, not for **repeated re-reads across many training steps/epochs**. Re-check its reopen behavior (`"Reopen for lazy reads — the BufReader above was consumed by the parse"`) for whether it supports being called into repeatedly and cheaply, or whether it currently assumes one-shot use.
- `grim-memory`'s tiered demote-before-drop pool (host RAM / GPU / spill) is a working pattern for bounding what's resident; it currently drives KV cache eviction. Whether to reuse this pool for weight-block residency during training, or build a separate simpler LRU for weight blocks, is an open design question — the KV cache pool's eviction policy is tuned for attention cache access patterns, not for the very different access pattern of "read block, use once per step, discard."

**What to build:**
1. A step-wise forward driver that, per transformer block: (a) requests the block's quantized weight tensors from the tensor provider, (b) runs fused-dequant forward (WI-T3 provides the kernels), (c) either keeps activations for backward or discards them and marks the block for recompute (gradient checkpointing — see below), (d) releases the block's weight buffer back to a bounded pool before moving to the next block.
2. Gradient checkpointing: by default, do **not** keep every block's activations resident. Recompute forward for a block during backward instead of storing its activations. This is what actually bounds peak memory — without it, streaming weights but keeping all activations resident defeats the purpose for long sequences. Decide the checkpoint granularity (per-block is the natural unit given grim's existing per-layer structure).
3. A bounded prefetch buffer (likely 1-2 blocks ahead) so weight-block I/O overlaps with compute on the previous block, rather than stalling per-step. This is where reuse of `grim-memory`'s tiering machinery is most likely to pay off — it already understands host/GPU tier movement.

**Left-right limits:**
- This work item does not change the on-disk `.grim` V1 format. Per existing project convention (wire version stays `GRIM\x01`), streaming reads must work against the *existing* layout — normals/outliers/backup streams as already defined in `format.rs`. If profiling later shows the layout itself is bad for repeated re-reads (e.g. poor locality), that's a new finding to bring back, not a license to bump the format version speculatively.
- Do not make streaming mandatory for inference — this must be additive, following the existing feature-flag-default-off discipline (`QkvAttentionFusionConfig::enabled` pattern). Training-mode loading is a new code path, not a change to `load_model_from_grim`'s existing behavior.

**Gates:** correctness (checkpointed-recompute forward produces bit-identical activations to non-checkpointed forward on CPU, modulo float non-associativity tolerance) → compiles → peak-memory measurement actually shows boundedness w.r.t. model size (this is the whole point of the work item, so treat it as a gate, not just a nice-to-have metric) → perf.

---

### WI-T3: Fused dequant+backward kernels (ROCm)

**Why:** Mirrors WI-T2's memory goal on the compute side. If forward reads a quantized block and dequantizes fused-into-the-matmul (already true for inference via `fused_dequant_gemm.rs`), backward needs the equivalent: gradient-w.r.t.-adapter through a frozen *quantized* weight, without ever materializing the dequantized weight matrix in VRAM.

**Where:** `grim-backend-rocm/src/kernels/fused_dequant_gemm.rs`, `grim-backend-rocm/src/kernels/kv_dequant_attention.rs` (patterns to extend, not reuse verbatim — attention backward and GEMM backward are different kernels), `grim-backend-rocm/src/gptq_kernel.rs`.

**What already exists:** Forward-only fused dequant GEMM and fused dequant attention. No backward kernel of any kind exists in the ROCm backend (confirmed by grep). The GEMM tiling work in the ROCm consumer-perf plan (SplitK-style tiling, asymmetric row/column sizing, LDS double-buffering) targets **decode-phase** shapes specifically (batch=1, single-token). Training backward GEMM shapes are different — larger batch (however you define "batch" for a training minibatch of sequences), and specifically a transpose pattern (`dL/dx = dL/dy @ W^T`, `dL/dW = x^T @ dL/dy`) that decode-phase tuning wasn't built for. **Do not assume the existing GEMM tuning work carries over to training shapes without re-profiling** — this is a distinct tiling problem even though it shares the "fused dequant" idea.

**What to build:**
1. Fused dequant + backward-GEMM for the adapter-touching linear layers: given `dL/dy` and the quantized frozen weight, produce `dL/dx` without materializing dequantized `W`.
2. Since the base `W` is frozen, `dL/dW` is never needed — only `dL/dx` (to keep propagating the graph backward toward the adapters at earlier layers) and the adapter gradients (handled in WI-T1, but the *matmul* backward through the base path feeds them). This is a meaningful simplification vs. general dequant-GEMM-backward and should be exploited, not built generically and then unused.
3. If adapters are injected at attention Q/K/V (per WI-T1's scope change), attention backward through the frozen dequantized K/V path is also needed. This is the highest-risk kernel in this plan — attention backward is inherently more complex than GEMM backward (softmax Jacobian, etc.) — budget accordingly and consider whether a correct-but-unfused CPU or naive-GPU fallback should ship first, with the fused version as a follow-up gated on the fallback being correct.

**Left-right limits:**
- Do not extend this to computing gradients w.r.t. the quantized base weights themselves (see WI-T1's scope note). This kernel family only ever produces `dL/dx`, never `dL/dW_base`.
- Do not fold this into the existing `qkv_attention.rs` forward kernel file if it makes that file harder to reason about — prefer new files under `kernels/` (e.g. `qkv_attention_backward.rs`) so the forward-path parallelism fix already in flight (75%-idle-wavefront issue) isn't put at risk by concurrent edits to the same file.

**Gates:** correctness (compare against CPU reference backward, not against the forward kernel's own dequant path — that would just check self-consistency) → compile → architecture cleanliness → perf (this one is *not* non-blocking in the usual sense — if the backward kernel materializes the dequantized weight anyway, the whole work item has failed its purpose regardless of speed, so add an explicit VRAM-footprint check as a correctness-adjacent gate, not just a perf number).

---

### WI-T4: Optimizer + `.grim.train` sidecar wiring

**Why:** `train.rs` already defines the sidecar file format (adapter blobs, optimizer moment blobs, `TrainFpFormat`) but nothing writes or reads it at runtime yet — it's a format spec with no producer or consumer.

**Where:** `grim-format/src/train.rs` (extend), new module for the optimizer itself (likely `grim-nn` or alongside WI-T1's autograd module — optimizer state is naturally adjacent to gradient computation).

**What already exists:** The binary layout (`GRIMTRN\x01` magic, header JSON + per-blob records), `TrainBlob`, `TrainFpFormat` enum (FP16/FP32/FP8 E4M3/E5M2/FP4 — note this already anticipates low-precision optimizer state, which is good, but AdamW moments in FP8 is numerically aggressive; start with FP16/FP32 moments and treat FP8 optimizer state as a later optimization once step-to-step training is verified correct, not a day-one requirement).

**What to build:**
1. AdamW (the minimum needed for parity with Unsloth's default recipe). Only needs state for adapter parameters — per WI-T1's scope, this is small (LoRA rank × hidden_size × num_layers × num_injection_points), which is exactly why keeping optimizer state fully resident in VRAM is fine even on constrained consumer cards — no need for CPU-offloaded/paged optimizer state the way bitsandbytes needs it for full fine-tunes.
2. Runtime read/write of `.grim.train` — save on checkpoint interval, load on resume. Must round-trip bit-for-bit per the format doc's stated goal ("a resumed fine-tune can reconstruct step-N state bit-for-bit").
3. Checkpoint/resume integration with WI-T5's training loop.

**Left-right limits:** Do not add a paged/CPU-offloaded optimizer path in this iteration — flagged above as unnecessary given adapter-only state is small. If GRPO (WI-T7) later needs a larger trainable footprint (e.g. a value head), revisit.

**Gates:** correctness (bit-identical resume) → compile → perf non-blocking.

---

### WI-T5: Training loop, dataset handling, SFT loss

**Why:** Everything above is infrastructure; this is where it becomes runnable.

**Where:** Likely a new crate or a module in `grim-cli` (there's already a `run.rs`, `service.rs`, `doctor.rs` pattern in `grim-cli/src` — a `grim-cli/src/train.rs` following that pattern is the natural fit) plus wiring into `grim-garage`'s existing `training_panel.rs` view-model and `jobs.rs` (job tracking already exists in garage — reuse it rather than building a second job-tracking mechanism).

**What already exists:** `grim-garage/src/jobs.rs`, `grim-garage/src/view_model/training_panel.rs` (UI already lists LoRA/QLoRA mode + quant-format picker — the UI is ahead of the backend here, which is somewhat unusual and worth being aware of: this plan is catching the backend up to a UI that was apparently built assuming this training capability would exist), `grim-garage/src/view_model/hyperparam.rs` (hyperparameter form — check what it already assumes about learning rate/batch size/epochs schema and match it rather than inventing a new config shape).

**What to build:**
1. Dataset loading (need a decision on supported formats — Alpaca/ShareGPT-style JSON is the common baseline; not scoping the specific format choice here, flag as a decision point).
2. Cross-entropy loss + its backward (fits into WI-T1's scoped autograd as one more op).
3. The actual step loop: forward (WI-T2, streamed) → loss → backward (WI-T1/T3) → optimizer step (WI-T4) → checkpoint interval → repeat.
4. Wire into `jobs.rs` so training runs show up the same way other garage jobs do.

**Left-right limits:** Do not build a distributed/multi-GPU training loop in this pass — everything above (streaming from one file, adapter-only gradients) is single-device by construction; multi-GPU is a different problem (would need to reconcile with `grim-backend-rocm/src/rccl.rs` and `p2p_route.rs`, which exist for inference-time tensor/pipeline parallelism, not training gradient sync — out of scope here).

**Gates:** correctness (loss decreases on a toy overfit-a-tiny-dataset smoke test — this is the standard sanity check for "is the training loop actually wired correctly" and should be a named test, not just eyeballed) → compile → architecture cleanliness → perf.

---

### WI-T6: Quantization-aware backward correctness (parallel track)

**Why:** grim's quant formats are its own K-quant-style formats (`Q4_K`/`Q5_K`/`Q8_0` per the training panel's quant-format picker), not bitsandbytes' NF4. The "quantized base, backward through it" story needs to be verified specifically against grim's own quant math (`grim-quant/src/lib.rs`, `grim-format/src/gptq.rs`), not assumed to work the same way NF4-based QLoRA does just because the high-level idea (frozen quantized base + adapters) matches.

**Why this is its own item and not folded into WI-T3:** dequantization *numerics* (does the fused kernel's dequant math match the reference dequant exactly enough that gradients aren't corrupted by quantization-induced discontinuities) is a correctness question distinct from *kernel* correctness. Keep it separate so it gets its own explicit gate rather than being assumed as a side effect of WI-T3 passing.

**What to build:** A numerical audit — for each supported quant format, confirm gradient flow through the fused dequant path matches gradient flow through an unfused reference dequant-then-matmul, within acceptable tolerance. The `train.rs` docstring already flags SERQ (saliency-based low-rank error correction for 4-bit GEMM) as a relevant technique from the research corpus — per the standing instruction not to dismiss enterprise-derived techniques just because grim targets consumer hardware, this is worth evaluating here as a way to recover accuracy lost to 4-bit quantization during training specifically (it's less necessary for inference-only serving, more relevant when quantization error compounds over many gradient steps).

**Gates:** correctness only, blocking — this gates WI-T5's step loop being trusted, not just a nice-to-have.

---

### WI-T7: DPO / ORPO / GRPO (deferred until T1–T6 gate)

Loss-function-level extensions on top of the SFT infra above. DPO/ORPO need a second (reference) model forward pass — worth checking whether that reference forward can reuse the same streamed-from-file path as the policy model (likely yes, since the reference model is also frozen — no adapters, no gradients, just forward). GRPO additionally needs a rollout/sampling loop, which overlaps with existing inference/generation code paths (`grim-speculative`, `grim-engine`) more than with anything new in this plan. Not detailed further here — write this as its own spec once T1–T6 are gated, since "what does the reference model forward path share with the policy model's streamed forward" is worth answering with working code in hand rather than speculating now.

### WI-T8: Bolt-on adapter fusion via the `backup2` residual slot

**Why:** Rather than a one-way destructive merge, treat the trained QLoRA adapter as a **toggleable, reversible correction** written into the base `.grim` file's already-spec'd-but-unused `backup2` residual slot. This only works cleanly if capacity is reserved once and never resized — see the design constraint below, which is the reason this has to be its own work item rather than a simple "write bytes" task.

**Correction to the previous version of this plan:** I previously implied the backup-stream mechanism was already wired at dequant time generically. Checking the actual kernels shows that's only half true:
- `grim-backend-cpu/src/dequant_gemm.rs` and the ROCm fused kernel (`grim-backend-rocm/src/kernels/fused_dequant_gemm.rs`, `device/roc_device.rs`) **do** already apply `backup1` as an additive correction during dequant — but the CPU path gates it on `ext.backup1.is_present() && ext.gptq_ordered > 0`. `backup1` is semantically claimed as the **GPTQ weight-reordering residual**, not a free general-purpose slot. Repurposing it for a LoRA bolt-on would collide with any model that went through GPTQ conversion.
- `backup2` — the second reserved slot in `GrimTensorExt` — is **not referenced by either dequant kernel at all**. It's spec'd, round-trip-tested at the I/O level (`format.rs`'s `write_backup`/`read_backup` tests), but nothing applies it during a real forward pass.

**So this work item has two real parts, not one:**
1. **Wire `backup2` into both dequant paths** (mirror `backup1`'s existing pattern in `dequant_gemm.rs` and the ROCm kernel), gated on `ext.backup2.is_present()` alone — no `gptq_ordered`-style qualifier, since this slot is dedicated to bolt-on adapters and should stay orthogonal to the GPTQ-residual use of `backup1`.
2. **The attach/detach mechanism**, which is genuinely simple *given* (1) and given that capacity is provisioned once at conversion time: `codes_offset`/`scale_offset`/`bpw` for `backup2` never change after that point, so:
   - **Attach:** quantize `ΔW = (α/r)·B@A` (reusing `grim-quant`'s existing routines) to `backup2`'s already-reserved `bpw`, write those bytes into the existing `codes_offset`/`scale_offset` region.
   - **Detach:** zero the same byte region. A zero correction changes nothing at dequant time — no flag needed, the bytes are the state.
   - Neither operation resizes the JSON metadata blob, the fixed-width `GrimTensorEntry` registry, or shifts any subsequent tensor's payload offset. This is what actually makes it a bolt-on instead of the one-way merge from the earlier draft of this plan.

**Left-right limits:**
- Do not repurpose `backup1` for this. It's already spoken for by GPTQ-ordered models; keep the two residual concerns cleanly separated rather than trying to squeeze two simultaneous bolt-ons out of both slots.
- This gives exactly **one** bolt-on slot per tensor. If simultaneous multiple adapters are ever needed, that's either a future 3rd-slot format extension or a reason to fall back to WI-T1's live-adapter-stacking path for that specific use case — not a reason to scope-creep this item.
- Conversion-time provisioning of `backup2` capacity is still the open question flagged in the previous plan (§5) — this work item assumes it's been decided to reserve capacity by default (or opt-in per conversion), not that it happens automatically today.
- Do not let this motivate a `.grim` wire-version bump, per standing convention — `backup2` is already-spec'd surface, this is filling in unused capability, not adding new format.

**Gates:** correctness (numerical parity vs. live-adapter forward when attached; bit-identical-to-original-base output when detached — that second check matters as much as the first, since "detach" claiming to be a no-op needs to actually be verified as one) → compile → architecture cleanliness (kernel changes mirror the existing `backup1` pattern rather than diverging in style) → perf non-blocking (attach/detach are infrequent operations; the dequant-path kernel changes do sit in the hot loop though, so confirm they don't regress `backup1`-absent throughput on models that don't use bolt-ons at all).

---

### WI-T9: Drop CVKG, serve grim-garage as a browser-rendered frontend over the existing axum API

**Why:** Decision made explicitly: stop building on the CVKG native-widget framework, render in a standard web browser instead, over `grim-garage`'s existing axum server. This also resolves a gap flagged in the previous review of the current state — the CVKG renderer host runs headless with no display target ever wired up (`main.rs`'s own comment: *"production would forward it to a winit window or another consumer via a channel"*, which was never implemented). That gap disappears entirely under this approach — there's no native window to forward to; the browser is the display target, and it already has one (the dev-server route people were already hitting at `http://localhost:8741/`).

**Where:** `grim-garage/Cargo.toml` (remove `cvkg`, `cvkg-components`, `cvkg-themes`, `cvkg-core`, `cvkg-runic-text`, `cvkg-webkit-server`), `grim-garage/src/ui/` (`panels.rs`, `dashboard.rs`, `view_kind.rs` — delete, CVKG-specific widget-tree construction), `grim-garage/src/renderer_host/` (delete — no headless native renderer needed), `grim-garage/src/theme.rs` (port, not discard — see below), `grim-garage/src/view_model/` (keep as-is — this is the layer that survives), `grim-garage/src/routes.rs` (extend to serve the frontend).

**What already exists and survives the swap unchanged:**
- `ViewModel` and its five constituent structs (`HyperparamFormV1`, `JobCardV1`, `AppShellLayout`, `RocmTogglesV1`, `TrainingPanelV1`) — already `Serialize`/`Deserialize`, already decoupled from the renderer by design (per its own doc comment), already unit-tested (`tests/view_model.rs`). None of this needs to change.
- The `/api/models`, `/api/datasets`, `/api/rocm/devices`, `/api/train/jobs`, `/api/train/start`, `/api/train/status/{id}`, `/api/train/cancel/{id}` JSON endpoints — these are renderer-agnostic already.
- `/sse/metrics/{id}` — already a standard SSE endpoint, which browsers consume natively via `EventSource` with zero server-side change needed.
- The OKLCH color decision in `theme.rs` doesn't need to be redesigned, just re-expressed: CSS has a native `oklch()` color function now, so the existing seed color and derived tokens translate close to 1:1 into CSS custom properties instead of `cvkg-themes::ThemeBuilder` calls.

**What to build:**
1. Remove the CVKG dependencies and delete `ui/panels.rs`, `ui/dashboard.rs`, `ui/view_kind.rs`, `renderer_host/`.
2. A route serving the actual frontend. Two shapes to choose between, worth deciding explicitly rather than defaulting into one (see open questions): server-rendered HTML generated from `ViewModel` on each request, vs. a static HTML/CSS/vanilla-JS bundle that fetches `ViewModel` (or the existing per-resource endpoints) client-side and renders in the browser. Given the API surface is already shaped for client-driven polling/fetch (that's what `GarageClient`'s GET methods and the SSE endpoint already assume), the static-plus-fetch shape is probably the smaller lift and keeps the existing endpoints exactly as they are — but this is a real decision, not an assumption to bake in silently.
3. If not already fully covered by the existing per-resource endpoints, a single `GET` endpoint that returns the whole `ViewModel` as JSON, so the frontend doesn't have to stitch together five separate fetches to render one dashboard.
4. Port `theme.rs`'s OKLCH tokens to CSS custom properties.
5. Real interactivity: wire the "Start Training" button, ROCm toggles, mode picker, etc. to actually call the existing POST endpoints. This was already broken under CVKG too (every widget callback in the old `panels.rs` was a no-op closure) — it's the same fix either way, just a `fetch()` call in JS now instead of a Rust closure that was never filled in.

**Left-right limits:**
- Don't touch the `/api/*` JSON surface or the SSE endpoint as part of this item — they're correct and renderer-agnostic; this work item is scoped to what serves/renders the frontend, not the data layer underneath it.
- Don't reach for a heavy JS build toolchain (React/Vite/bundlers) by default. CVKG's own pitch was explicitly "no React/Vite/Tailwind" — dropping CVKG doesn't have to mean abandoning that instinct, just the native-widget part of it specifically. Plain HTML/CSS/vanilla JS (or a light progressive-enhancement approach) keeps the same spirit. Worth confirming this is actually the intent rather than assuming it — flagged as an open question below.
- No partial migration — don't leave some panels on CVKG and others on HTML. Full removal in one pass; a half-migrated UI is worse than either endpoint state.

**Gates:** correctness (`tests/view_model.rs` and the API-surface tests should be unaffected and keep passing as-is; `tests/ui_compositors.rs` and `tests/renderer_host.rs` specifically test CVKG tree-building and need to be deleted or replaced with equivalent coverage of the new rendering path, not just left broken) → compile → architecture cleanliness (`ViewModel` stays the single source of truth per its own existing doc comment; the frontend is a thin consumer of it, not a place where view logic creeps back in) → perf non-blocking.

---

### WI-T10: grim-garage bolt-on management (attach/detach/list)

**Why:** This is the actual user-facing requirement — training an adapter (WI-T5) and having WI-T8's mechanism exist in the format layer isn't useful until someone can attach/detach a bolt-on without hand-editing file bytes.

**Where:** `grim-garage/src/routes.rs` (existing `build_router` — has `/api/train/*`, `/api/models`, `/api/datasets` already; no adapter-related routes yet), `grim-garage/src/jobs.rs` (existing job-tracking, reused rather than duplicated), `grim-garage/src/view_model/training_panel.rs` (the view-model layer, not the CVKG rendering of it — extend with a sibling `bolt_on_panel.rs` following the same `V1` struct naming convention; this work item assumes WI-T9 has already landed, so "the panel" here means a section of the browser-rendered frontend, not a CVKG widget tree).

**What to build:**
1. New routes: something like `GET /api/models/{id}/bolt-ons` (list what's currently attached, reading `backup2.is_present()` and whatever byte-nonzero check indicates "active" vs. "provisioned but empty"), `POST /api/models/{id}/bolt-ons` (attach — takes a reference to a trained adapter, presumably a `.grim.train` sidecar path or job id from WI-T5), `DELETE /api/models/{id}/bolt-ons/{slot}` (detach).
2. A view-model + panel for picking a model + adapter and attaching/detaching, following `training_panel.rs`'s existing `V1` struct pattern.
3. **The integration point most likely to get missed:** if `grim-server` already has a model loaded for serving (it exposes `/v1/models/load` / `/v1/models/unload`), attaching or detaching a bolt-on by editing the file on disk doesn't retroactively change what's already resident in a running server's memory. Attach/detach in garage should trigger (or at minimum clearly surface a need for) a reload through `grim-server`'s existing load/unload endpoints — otherwise "detach" can silently appear to succeed while the live-served model keeps answering with the adapter still baked into whatever was loaded before the edit.

**Left-right limits:**
- Attach/detach should not require restarting `grim-server` process-wide — model-level reload via the existing `/v1/models/load`/`/v1/models/unload` pair should be sufficient; if it isn't, that's a finding to bring back, not a reason to build a heavier mechanism speculatively.
- Do not build multi-model batch attach/detach in this pass — one model, one bolt-on slot, one operation at a time, matching WI-T8's one-slot scope.

**Gates:** correctness (attach via garage produces output identical to attach via direct file manipulation; detach-then-reload genuinely restores base-model output) → compile → architecture cleanliness (garage stays a thin client over the format-layer mechanism in WI-T8, doesn't reimplement quantization logic itself) → perf non-blocking.

---

### WI-T11: Full fine-tune / Bf16-Full (explicitly out of scope for this plan)

Flagged in §1. Needs gradients w.r.t. every frozen-in-this-plan base weight, which invalidates the "read-only streamed base" assumption this entire plan is built on. Recommend treating as a separate spec if/when it's prioritized, likely requiring a different memory strategy (optimizer state sharding, activation offload to host RAM at a much larger scale) than adapter-only training needs.

## 3. Suggested sequencing

1. WI-T1 + WI-T2 in parallel (different people/sessions could plausibly own these — T1 is autograd bookkeeping, T2 is I/O/memory streaming; they meet at the interface of "what does forward need to keep around for backward").
2. WI-T3 once T1's op set is stable enough to know exactly which backward kernels are needed (don't build backward kernels speculatively for ops T1 doesn't end up using).
3. WI-T4 in parallel with T3 — it's independent (optimizer math doesn't depend on how backward is computed, only on gradients existing).
4. WI-T6 as a correctness audit running alongside T3, not after it — cheaper to catch numeric issues before the training loop is built on top of a shaky kernel.
5. WI-T5 last, since it's the integration point for everything above.
6. WI-T7 only after T5 gates.
7. WI-T8's kernel half (backup2 dequant wiring) can start as soon as conversion-time capacity provisioning is decided — it doesn't need T5's training loop, just a synthetic delta to test against. Its attach/detach half needs a real trained adapter, so realistically trails T5.
8. WI-T9 (drop CVKG, browser-rendered frontend) is independent of the entire training-backend stack (T1–T8) and can run in parallel with any of it, on a different track entirely — it's blocked on nothing above and blocks only WI-T10, which needs a real frontend to attach the bolt-on panel to.
9. WI-T10 last of the garage-facing items, after both T8 (mechanism exists) and T9 (frontend exists).

## 4. Left-right limits for the plan as a whole

- No `.grim` V1 format version bump. Training rides the existing wire format via the lazy tensor provider and the already-separate `.grim.train` sidecar, per standing project convention.
- No general-purpose autodiff engine. If a future need (e.g. WI-T8) demands one, that's a deliberate follow-on decision, not a natural extension of this plan's scoped autograd.
- No multi-GPU training in this pass.
- No CPU-offloaded/paged optimizer state in this pass (adapter-only state doesn't need it).
- Remove tests/ui_compositors.rs and tests/renderer_host.rs after implementing the web rendering solution from grim-garage.

## 5. Open questions to resolve before/during implementation

- **Does per-step file re-reading actually beat "quantize once, page the whole quantized model into VRAM" for the model sizes grim targets on consumer cards?** The streaming approach bounds peak memory independent of model size, which matters most for models that don't fit even quantized. For models that *do* fit quantized in VRAM (the common case on a 24GB card with a 7-13B model), a single quantized load might just be faster with no memory downside — worth benchmarking both paths rather than assuming streaming is strictly better. This may argue for streaming being a *mode*, not the only path — a `training_mode: streamed | resident-quantized` toggle following the existing feature-flag-default-off discipline.
- **Where exactly should adapters be injected** (logits-only today, needs to become Q/K/V/O + MLP for real QLoRA parity) — this is a forward-path decision inside WI-T1 that affects which backward kernels WI-T3 needs to build, so nail it down early.
- **Which crate owns autograd** — `grim-nn` extension vs. new `grim-autograd` crate. Given the existing left-right-limits discipline about backend crates not reaching into other workspace crates, and that autograd needs to call into `grim-backend-rocm` kernels, a new crate sitting at the same layer as `grim-nn` (rather than inside it) may keep dependency direction cleaner — worth a short design note before WI-T1 starts, not a default assumption baked into this plan.
- **Should `convert.rs` start provisioning `backup2` capacity by default on every base-model conversion, to enable WI-T8's bolt-on mechanism?** This costs file size on every conversion, including models that never get fine-tuned, in exchange for making attach/detach cheap and reversible later with no format resize. Reasonable to default off (current behavior — no backup capacity reserved for either slot) and treat as an opt-in conversion flag for people planning to fine-tune, rather than a blanket default; worth deciding explicitly. Note this is specifically about `backup2` — `backup1` is already semantically committed to GPTQ-reordering residuals in the existing dequant kernels and shouldn't be repurposed.
- **Server-rendered HTML vs. static HTML/JS fetching the existing JSON API (WI-T9)?** The existing endpoints (per-resource GETs, SSE metrics) are already shaped for the fetch-driven model, which argues for that side of the fork, but it's worth deciding explicitly rather than defaulting into it partway through implementation.
- **How much frontend tooling is actually wanted (WI-T9)?** Dropping CVKG's native widgets doesn't by itself answer whether the replacement should be zero-build-step plain JS, something like htmx for progressive enhancement, or a light bundler. Given the project avoided React/Vite/Tailwind for the native UI specifically, confirm whether that preference was about avoiding native-widget lock-in or about avoiding frontend build tooling generally — the two paths this item could take look quite different depending on which.
