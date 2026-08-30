# Serving-layer parity plan: closing the vLLM/SGLang gaps

Status: **P0–P4 landed and tested** (2026-08-29). P2 block-level execution and P3
EP execution remain explicit follow-ups, documented below, not silently shipped.

Verification first, implementation second — each phase lists the exact files
and the acceptance gate that closes it.

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

## P2 — Pipeline parallelism — **config + planner + gate landed; block execution pending**

1. ✅ Config: `GRIM_PP_SIZE` / `EngineConfig.pp_size` (env read in
   `EngineConfig::default`).
2. ✅ Planner: `pipeline_engine::PipelinePlan` consumes the existing
   `PipelineStageConfig::partition_layers` math and is validated at engine
   startup.
3. ✅ Gate: `Engine::new` hard-fails loudly when `pp_size > 1` (rather than
   loading PP-shaped weights and silently running single-device execution),
   pointing at this doc. Covered by `test_engine_rejects_pipeline_parallel_size`.
4. ⏳ Block-level execution is NOT wired and is the deliberate follow-up:
   a paged KV pool is single-device (`KvBlockPool`), so PP first needs
   per-stage KV pools plus cross-stage activation transfer — a prerequisite
   this gate forces to be solved before anyone can flip the switch. Parity
   test deferred until then.

Files: `grim-engine/src/pipeline_engine.rs`, `grim-engine/src/lib.rs`
(+ `tests/disagg_engine_loopback.rs` initializer).

## P3 — Expert parallelism audit — **documented**

Finding (verified against the actual upstream model at
`Qwen/Qwen3.8-Flash-Next`, which is `Qwen4ExpForConditionalGeneration`): MoE
experts are **replicated across TP ranks**, not sharded. `MoeBlock::load`
stores the `tp_config` but `ExpertBank::load` (3D `[num_experts, hidden, inter]`
GGUF layout) loads the full expert bank on every rank; only the attention
heads/KV/output projection are sharded (via `plan_kv_head_sharding`). So TP
for MoE today = full expert replication (memory-expensive, compute-correct),
and the EPLB planner in `eplb.rs` is unused. EP execution (token all-to-all +
combine, a `GRIM_EP_SIZE` work item) is the largest lift and is intentionally
not started — it must not land half-wired. Bonus fix landed: the Qwen38 loader
in `grim-engine/src/model_loader.rs` referenced a removed
`gated_residual_branches` field and was missing six fields added in a recent
config refactor (verified against `config.json`); both initializer sites now
match the struct and the HuggingFace-published config.

## P4 — TP launch ergonomics — **landed**

`grim serve --tp-size N` spawns one OS process per rank (Design A): this
process is rank 0 (serving HTTP on the requested port); ranks `1..N` are
child processes with `GRIM_TP_SIZE`/`GRIM_TP_RANK` stamped and their HTTP
port offset by rank. `--address` is refused under `--tp-size` (ports must be
derivable). A `TpChildGuard` kills peers on rank 0 exit; a fail-stop monitor
takes rank 0 down if any peer dies (a missing peer deadlocks the survivors'
collectives on the next forward — better to exit than hang). This automates
the manual operator procedure the TP design comment describes; it does not
change the TP execution design. Verified: `--help` shows the flag, and
`--tp-size 2 --address ...` exits 2 with the expected message.

Files: `grim-cli/src/main.rs`.

## Acceptance gates — all green (2026-08-29)

- `cargo test -p grim-scheduler(35) -p grim-engine(130) -p grim-backend-rocm(372)
  -p grim-kvtransport(31) -p grim-disagg(22+10) -p grim-cli(42+67)` — **0 failures**.
- New tests pass; zero behavior change for single-adapter, zero-adapter,
  multi-adapter, and speculative decode paths (P0 parity tests prove the
  grouped-decode path matches the legacy per-request path exactly).
- GPU kernel parity: the JIT shrink+expand LoRA kernel pair matches the CPU
  reference on the local device (gfx).
