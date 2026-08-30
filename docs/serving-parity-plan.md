# Serving-layer parity plan: closing the vLLM/SGLang gaps

Status: in progress (2026-08-29). Verification first, implementation second —
each phase lists the exact files and the acceptance gate that closes it.

## Verification summary (what was actually true on 2026-08-29)

The external audit ("grim is missing TP/PP/EP/DP; multi-LoRA is name-only;
disagg is single-transport") was **partially outdated and partially correct**:

| Claim | Verdict | Evidence |
|---|---|---|
| No tensor parallelism | **Wrong** | Multi-process TP (Design A) is real: `GRIM_TP_SIZE`/`GRIM_TP_RANK`/`GRIM_GPUS` bootstrap in `grim-engine/src/lib.rs` (`Engine::new`), sharded loading via `load_tp` for ~10 archs in `model_loader.rs`, device-side RCCL `ncclAllReduce` (F32/F16/BF16) in `RocmDevice::all_reduce` (`roc_device.rs`), wired through `BackendDevice::all_reduce` into `RowParallelLinear::forward`. Unsupported archs fail loudly via `require_single_device`. |
| No pipeline parallelism | **Correct** | `tp_layers.rs`/`pipeline_engine.rs` (untracked WIP) are inert scaffolding: partition math + send/recv helpers, zero production consumers, no stage-scheduled execution loop. |
| No expert parallelism | **Correct** | `eplb.rs` is a planner (greedy LPT expert→rank packing + replication) consumed only by its own test. No token dispatch/combine. |
| No data parallelism | **Mostly correct** | Scythe farm replicas (`{base}#scythe{r}` full weight copies + C2PLR request pinning) are functionally request-level DP, but there is no vLLM-style DP scheduler and no per-rank process launcher. |
| Multi-LoRA not wired | **Correct** | `adapter_batches` was write-only outside its unit test. The uncommitted WIP added `lora_segments`, `Engine::execute_fused_batched_lora`, and `kernels/batched_lora.rs` — with zero production callers. Production path remains per-request `apply_adapters_to_logits` (`model.rs`, `muse_glimmer.rs`). Latent WIP bug: scheduler segments count *sequences* while the engine function treats them as packed *token rows* — only equivalent at 1 token/sequence. |
| Disagg is single-transport | **Correct** | 1,958 lines. Wire path is TCP-only (V3 protocol). `TransportProtocol::{RdmaRoce, UcxDirect, SharedMemP2p}` are metadata: `with_protocol` writes `kv_client.protocol`, which no send/fetch path reads. |

Scope decision: TP and request-level routing already work; this plan closes the
genuinely missing pieces — batched multi-LoRA (P0), real transport selection
(P1), pipeline parallelism (P2), EP audit (P3), and TP launch ergonomics (P4).

---

## P0 — Wire batched multi-LoRA end-to-end (S-LoRA/Punica-style)

Architecture constraint discovered during verification: grim decode is
per-sequence (`decode_one` per session; adapters applied *inside* the model
forward), so "one GPU pass for many adapters" must happen at the engine's
grouped decode step (`step_batch`, WI-X1) by splitting each tick into a base
pass and a batched LoRA-apply pass over stacked logits rows.

1. **Row-range segments.** Segments must be computed over the *actual batch
   rows the engine forwards*, not scheduler sequence counts. Add a row-based
   planner (`LoraSegment::plan_for_rows`) in grim-scheduler; the engine builds
   segments from its final item order per model group. The scheduler's
   sequence-level `lora_segments` stays for observability only.
2. **Two-phase `step_batch`.**
   - Phase A: for each item, drive a **base** decode (empty adapter list) and
     record the request's primary adapter id. Only `Strategy::Plain` items
     with ≤1 adapter take this path; speculative strategies and multi-adapter
     requests keep the legacy per-request path (adapters applied inside
     `decode_one`) — documented boundary, no numerics change for them.
   - Phase B: group rows by model id (vocab must match within a group);
     stable-sort rows by adapter id into contiguous segments; apply all
     segments to the stacked `[n, vocab]` logits in one batched call;
     slice rows back into each `StepOutcome`. Zero-adapter rows are
     untouched (base passthrough).
3. **Batched apply primitive.** Rewrite `Engine::execute_fused_batched_lora`
   into `apply_batched_lora(stacked, segments, in_dim, out_dim)` with a CPU
   reference path (`batched_lora_accumulate_cpu`) and a ROCm device path.
   The same in_dim==vocab surrogate contract as `apply_adapters_to_logits`
   applies (documented, MED-5).
4. **GPU kernel.** Complete `batched_lora.rs`: shrink kernel (X_seg·Aᵀ) plus
   the existing expand/scatter kernel (atomicAdd into the base output).
   Compile via `jit_compile_hsaco` + the hipModule launch pattern used by
   `gptq_kernel.rs`; persistent disk cache applies automatically. Device
   path activates only when the logits are ROCm-resident and a device is
   available; otherwise the CPU reference runs (behavior-portable by design,
   §4.5).
5. **Tests.**
   - Kernel parity: CPU reference vs GPU kernel on random data (GPU-gated).
   - Engine parity: mixed-adapter batch through grouped `step_batch` must
     match legacy per-request `step_one` outcomes on identical engines.
   - Segment planner unit tests (contiguity, ordering, base passthrough).
   - Fix the overstated `adapter_batches` doc comment to state the actual
     contract (groups are advisory; execution consumes row segments).

Files: `grim-scheduler/src/lib.rs`, `grim-engine/src/lib.rs`,
`grim-backend-rocm/src/kernels/batched_lora.rs`.

## P1 — Make disagg transport selection real

1. Extract a `KvWireTransport` trait (`send_block` / `fetch_block`) in
   grim-kvtransport; today's TCP V3 wire code becomes the first impl.
2. Implement `SharedMemP2p` for same-host endpoints (loopback addresses):
   shared/host-pinned ring buffer handoff with graceful TCP fallback on any
   miss. `RdmaRoce`/`UcxDirect` return explicit `Unsupported` errors until
   hardware-backed impls land (no silent TCP downgrade for explicitly
   requested RDMA).
3. Replace the placeholder protocol-selection test with per-protocol loopback
   data-integrity tests (payload, num_tokens, layer_idx round-trip).

Files: `grim-kvtransport/src/lib.rs`, `grim-disagg/src/lib.rs` (+ tests).

## P2 — Pipeline parallelism

1. Config: `GRIM_PP_SIZE` / `EngineConfig.pp_size`; `PipelineStageConfig::
   partition_layers` (exists) produces a `PipelinePlan` (layer → device).
2. Execution: in-process stage split for the Llama family — layers of stage 0
   on its device, remaining layers on the next stage's device, activations
   moved at the boundary via the existing cross-device copy path, logits on
   the last stage. No collectives needed (PP has no all-reduce).
3. Parity test: PP-2 vs single-device logits on a tiny model (CPU-plumbed in
   CI; real device split verified when ≥2 ROCm devices are visible).

Files: `grim-engine/src/pipeline_engine.rs`, `grim-engine/src/lib.rs`,
`grim-models/transformer/src/` (stage-aware execution entry point).

## P3 — Expert parallelism audit (document, don't build yet)

Audit how MoE experts behave under TP today for Qwen3-MoE/Qwen35-MoE
(replicated vs sharded), and record findings here. EPLB stays a planner until
EP execution (token all-to-all + combine, `GRIM_EP_SIZE`) is scheduled as its
own work item — it is the largest lift and must not land half-wired.

## P4 — TP launch ergonomics

`grim serve` gains a launcher that spawns one process per rank with
`GRIM_TP_SIZE`/`GRIM_TP_RANK`/`GRIM_GPUS` stamped per child (vLLM `--tp 2`
UX), rank 0 serving HTTP. This automates the manual operator procedure the
TP design comment describes; it does not change the TP execution design.

## Acceptance gates

- `cargo test -p grim-scheduler -p grim-engine -p grim-backend-rocm
  -p grim-kvtransport -p grim-disagg` green (plus `-p grim-cli` for P4).
- New tests above pass; zero behavior change for single-adapter,
  zero-adapter, multi-adapter, and speculative decode paths (parity tests
  prove it).
- This file updated with per-phase status as phases close.
