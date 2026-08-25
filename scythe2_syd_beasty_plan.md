# Plan: SCYTHE-2 inference — execution plan for syd-beasty

Extends the WI-INF1..INF5 work from `scythe2_inference_plan.md`. Everything
here is scoped so that **every gate runs on syd-beasty** — the RX 9070 XT /
RX 9060 XT asymmetric pair is the only machine where the findings this plan
exists to produce can be validated. Host-only checks (CPU unit gates, compile,
clippy) are listed per-WI as pre-flight, but a WI is not "done" until its
hardware gate row is checked on the box.

State at time of writing (2026-08-22): WI-INF1/INF2/INF3 implemented and
gated (Engine owns profiler + controller; streaming-forward routing with
byte-parity gate; farm-mode serving integration over per-GPU weight replicas
with load-balanced pinning); WI-INF4's decide_miss sweep passes host-side
(~2 µs/layer vs ~10 claimed) but end-to-end TTFT A/B awaits this box.
Note: that plan file went missing from the working tree during a concurrent
cleanup of scratch markdown files — its validation record is preserved here;
restore it from git history or session notes if needed.

Status vocabulary:
- `[host]` — verifiable off-box, before shipping code to syd-beasty.
- `[sb]` — must run ON syd-beasty; produces a number or a verdict.
- Size: S ≤ half day · M ≈ 1–2 days · L = multi-day WI with its own doc.

---

## Machine setup checklist (once, before any WI)

1. Toolchain pin. The repo builds under `nightly-2026-04-11`
   (`rustc 1.96.0-nightly (02c7f9bec)`). Record it in the results file header;
   do not mix compilers in one `target/`.
2. Capture ground truth per card, saved to
   `docs/benchmarks/syd_beasy_topology.json`:
   - `rocm-smi --showproductname --showmeminfo vram --showclocks`
   - `hipGetDeviceProperties` dump per ordinal (`gcnArchName`, clocks, CU count,
     LDS, totalMemory) via a tiny `examples/gpu_props.rs` (added in WI-SB3).
3. Ordinal-order matrix. Every hardware experiment below runs TWICE:
   - Order F-first: `GRIM_GPUS=<fast_ordinal>,<slow_ordinal>`
   - Order S-first: `GRIM_GPUS=<slow_ordinal>,<fast_ordinal>`
   Rank-0-sticky regressions only reproduce in one order; both must pass.
4. Results protocol: every `[sb]` run appends one JSON line to
   `docs/benchmarks/scythe2_syd_beasty_results.jsonl`
   `{ "wi": "SBx", "order": "F-first|S-first", "metric": ..., "value": ...,
      "commit": ..., "ts": ... }`. Verdicts flip checkboxes here, not memory.
5. Rollback invariant (re-assert after each WI): unset
   `GRIM_SCYTHE_INFERENCE` ⇒ engine behavior byte-identical to pre-plan main
   (farm degrades to plain registration; profiler stays `None` on single-GPU).

---

## WI-SB0 — Capability differentiation audit  `[sb]` · size S · GATES EVERYTHING

**Problem found during validation:** `arch_tflops_table()`
(`grim-backend-rocm/src/device/capability_profiler.rs`) is family-keyed —
every `gfx12*` returns `(80.0, 160.0, 960.0)`. Both cards on syd-beasty are
RDNA4 ⇒ identical caps ⇒ WaveTune argmin always ties ⇒ placement collapses to
sticky rank 0 regardless of which card that is. Also the profiler docstring
claims "HIP attributes + micro-GEMM" but `probe_host_gpu` reads attributes
only — no measurement anywhere. The controller cannot place what it cannot
tell apart.

**Changes**
- Extend `arch_tflops_table` to match full arch strings first
  (`gfx1201`/`gfx1200`/…), falling back to family rows. Seed values from step 2
  below, marked `TODO(gpu-verify)` until measured.
- Add boot-time calibration: one small FP16 GEMM per device at profiler
  construction (~ms, once per process, cached by `(gcnArchName, clock_mhz)`)
  producing *measured* effective TFLOPS + bandwidth; static table becomes the
  fallback when calibration errors. Throttle correction already applied
  downstream stays as-is.
- Fix the docstring to describe what actually runs.

**Gates**
- `[host]` unit: table returns distinct values for `gfx1201` vs `gfx1200`.
- `[host]` unit: calibration path falls back cleanly when HIP absent.
- `[sb]` profiler over the pair reports `tflops_fp16(fast) > tflops_fp16(slow)`
  AND distinct `hbm_bandwidth_gbps`, in BOTH ordinal orders. Ratio recorded.
- `[sb]` `decide()` over the pair's real caps picks the fast card unloaded
  (extends `test_untrained_mlp_prefers_faster_gpu` to real caps).

**Exit:** without this WI passing, no downstream placement finding is
meaningful. Do not proceed on a tie.

## WI-SB1 — CLI farm wiring  · size S

Server arms farm mode today; the CLI still registers plainly, which blocks
`grim serve`-style runs on the box from exercising the farm.

**Changes**
- `Engine::load_and_register_scythe_farm_speculative(id, base_path, draft,
  lookahead)` — farm-aware variant of the existing speculative loader: rank-0
  replica carries the draft/EAGLE3 attachments; replicas ≥ 1 register plain.
  Comment states the limitation: drafter lives on rank 0's device.
- `grim-cli/src/main.rs` sites :993 and :1150 swap to the new call.

**Gates**
- `[host]` unarmed engine + farm loader ⇒ identical registration surface to
  today (extends `test_scythe_farm_degrades_to_plain_registration`).
- `[sb]` `grim serve` boots with `GRIM_SCYTHE_INFERENCE=1`, log shows
  `farm armed: N replica(s)`, one smoke completion per rank pin
  (`resolved_model_id` visible in status output).

## WI-SB2 — Admission-time VRAM guard  · size S

Asymmetric VRAM (16 GB vs 8 GB-class) currently enters only as a soft input.
An oversized prompt can be pinned onto a card that cannot hold its KV.

**Changes**
- In `scythe_pick_rank`: compute request footprint before consulting the
  controller:
  `kv_bytes = 2·(prompt_tokens + max_new_tokens)·num_kv_heads·head_dim·layers·4B`
  (+ working-set floor `2·seq·hidden·layers·4B`); drop ranks whose
  `vram_free_bytes < footprint + watermark` from the cap vector fed to
  `decide()`; none survive ⇒ return to scheduler queue via the existing
  `AdmissionController` rather than pinning blind.
- `max_new_tokens` plumbed from the request; watermark constant documented.

**Gates**
- `[host]` synthetic-caps unit: 8 GB-card excluded for a 100k-token prompt,
  included for a 1k prompt; all-excluded ⇒ queued, never pinned.
- `[sb]` long-context prompt (near slow-card capacity) pins fast card in both
  ordinal orders; `rocm-smi` shows no OOM event on the slow card.

## WI-SB3 — TTFT/ITL A/B harness  · size S→M · produces the WI-INF4 verdict

The decide_miss sweep (`benches/scythe2_decide_miss.rs`) covers host-side cost
only. What WI-INF4 actually demands is end-to-end prefill TTFT and decode ITL,
controller on vs off, on the pair.

**Changes**
- New `examples/scythe_ttft_ab.rs`:
  - args: `--model <path> --arm on|off --prompts <file> --iters N`;
    prompts fixed in-repo (`examples/prompts_scythe_ab.txt`: mix of ~200,
    ~2k, ~8k tokens).
  - drives the real engine loop (`enqueue_request_with_kv` → `tick()`),
    samples `last_ttft_ms` / `last_itl_ms` / `tokens_per_sec_ema`, appends JSON
    lines per §setup-4.
  - prints the A/B table and applies the verdict rule.
- Verdict rule (WI-INF4 gate): mean TTFT overhead ≤ 5 % AND p95 ITL overhead
  ≤ 2 % across the prompt mix and BOTH ordinal orders ⇒ eligible to flip the
  flag default; otherwise stays opt-in and the cost model gets retuned with
  the measured numbers.
- Plus `examples/gpu_props.rs` (topology capture for §setup-2).

**Gates**
- `[host]` harness compiles + runs against CPU mock model (deterministic
  smoke); JSON schema validated.
- `[sb]` full matrix: {on, off} × {F-first, S-first} × prompt mix, N ≥ 30
  iters. Verdict recorded in this file's checkbox + results jsonl.
- `[sb]` placement sanity: with arm=on, pin distribution logged; fast card
  receives ≥ half of concurrent-session placements under load (validates
  WI-SB0 differentiation end-to-end).

## WI-SB4 — Contiguous layer pipeline (per-layer straddle, step 1)  · size M

Closes "one request can't straddle GPUs" for the case that pays first: prefill.

**Design**
- `Llama` gains `layer_devices: Vec<Device>` (default: single entry = own
  device — zero behavior change). Built by the farm registrar: controller
  `decide()`s per layer index over real caps, then contiguous same-rank runs
  are merged so transfers happen only at boundaries.
- `decode_paged`: transfer `h` at each boundary via existing
  `transfer_to_device` (P2P when ROCm↔ROCm); final norm + head stay on the
  last segment.
- **KV constraint, stated honestly:** the paged session cache is
  single-device. Sub-items:
  - **SB4a (this WI):** classic per-layer-cache path (caches follow segment
    devices naturally) — targets prefill TTFT, weight-streaming-style serving.
  - **SB4b (follow-up):** per-segment `PagedKvCache` pools in grim-memory +
    layer-range block tables. Filed, gated on SB4a showing pipeline TTFT wins
    worth the memory-pool surgery.

**Gates**
- `[host]` parity: split across two fake segments mapped to CPU ⇒ logits
  byte-identical to unsplit (same pattern as
  `scythe_route_parity_and_untrained_fallback_gate`).
- `[host]` transfer-count unit: grouping emits ≤ #segments−1 hops.
- `[sb]` prefill TTFT: 8k-token prompt, split vs unsplit, both orders. Pass =
  measurable TTFT improvement on the pair OR honest retirement of the idea
  with numbers (either outcome updates WI-INF5's ledger).

## WI-SB5 — Scythe2Linear real cross-GPU GEMM  *(filed as WI, design now)* · size L

Validation pass found `forward_col_parallel` accumulates shards as
`Vec<Vec<f32>>` — host emulation, not device execution. Before tensor-split
placement can serve: per-rank device GEMM shards (rocBLAS handles bound per
rank/stream), CommFuse P2P fan-in for row-parallel partials, ring descriptors
(`ScytheTaskDescriptor` opcodes 1/2) carrying shard pointers instead of host
slices. Gates mirror rocm-hip-kernels checklist (wave32 sizing, rocprof-compute
occupancy read-out, numerics parity vs monolithic GEMM within fp tolerance —
not byte-parity: split-K accumulation order differs; assert max-abs-diff bound
instead). Blocked on SB4 numbers justifying the work.

## WI-SB6 — Persistent-ring dispatch  *(filed)* · size L

Host side of WI-7 exists (`ScytheRing`, MoE dispatch planner WI-EP2). Device-
resident polling kernel + descriptor-driven dense-layer execution is the last
mile of scythe2.md §3. Design doc only in this plan; no code until SB4/SB5
results justify it.

---

## Sequencing

```
SB0 ──► SB3 ──► WI-INF4 verdict (flag decision)
  │      │
  │      └──► SB4a ──► (SB4b? / SB5 / SB6 — gated on measured wins)
  └──► SB1 ──► SB2          (independent, any time after SB0)
```

Rationale: SB0 makes placement decisions meaningful — everything downstream
consumes its numbers. SB3 turns the box into a one-command experiment rig that
every later WI reuses for verdicts. SB1/SB2 are small correctness/QoI pieces
that ride along. SB4 starts only once baseline A/B numbers exist so its win/lose
call is grounded.

## Checkbox ledger

| WI | host gates | sb gates | done |
|----|-----------|----------|------|
| SB0 differentiation | ☑ distinct gfx1201/gfx1200 rows + tests | ☑ calibration live (WI-SB0 rocBLAS f16/f32-ex micro-GEMM + DtoD bandwidth sweep, cached per `(arch, 500 MHz clock bucket)`): gfx1201 ≈ 4.4 TFLOPS / ~311 GB/s vs gfx1200 ≈ 2.3 TFLOPS / ~245 GB/s (debug-build numbers, ordering is what placement consumes); `decide()` on real caps picks fast card in BOTH orders (`test_real_measured_caps_prefer_faster_gpu_both_orders`). En-route: HIP attribute constants were CUDA-numbered — "throttle" attr read shared-memory size → throttle_pct clamped to 1.0 → **effective TFLOPS zeroed on every GPU**; fixed + honest 0.0 | ☑ |
| SB1 CLI farm | ☑ loader + call sites :993/:1150; degradation test | ☑ farm arms exactly 2 replicas `[Rocm(0), Rocm(1)]` in release after event-seam pin fix; smoke completions served with pins logged, status lists `#scythe1` replica telemetry; both physical cards served across F-first/S-first boots. Load-spreading implemented post-verification (`decide_forced` + pin cooldown + external busy weight 2.0) but clamped to rank 0 by default behind `GRIM_SCYTHE_SPREAD=1` until the rank-1 sampler crash is fixed (validation log 2026-08-23e) | ☑ |
| SB2 VRAM guard | ☑ footprint formula w/ hidden hint; synthetic-caps gates (8 GB excl. @100k, incl. @1k, all-excluded ⇒ waitlist); queue-not-blind-pin wired via `scythe_admission_decision` + VRAM waitlist | ◑ pins verified via farm legs (both orders); near-capacity exclusion not reproducible on this box (16 GB cards, 230 M model ⇒ max footprint ≪ VRAM) — synthetic gates stand in | ☐ |
| SB3 A/B harness | ☑ `scythe_ab` module (§setup-4 JSONL, throttle_pct, verdict rule ≤5%/≤2%), real prompt mix (~200/2k/8k), order detection; harness fixed (per-sample trace clear + decode drain) and measured in RELEASE | ☑ campaign run {on,off}×{F,S} interleaved rounds, 30 samples/arm per order — see results jsonl | ☑ |
| WI-INF4 verdict | — | ☑ **STAYS OPT-IN**: mean TTFT overhead −0.09%/−0.00% (F/S, PASS ≤5%); p95 ITL overhead −18.56% (F, PASS) / **+2.43% (S, FAIL ≤2%)**. Flag default does not flip; cost model to be retuned with these numbers (§SB3 rule). Known measurement defect: `parse_samples` has no time filter so the in-harness cumulative report mixes stale fault-era rows — ts-filtered computation is authoritative | ☑ |
| SB4a layer pipeline | ☑ `decode_paged` boundary transfers + `segment_devices`; planner `absorb_short_runs`; parity gate (fp-tolerance cross-backend fake segments) + hop-bound gates | ☐ TTFT split-vs-unsplit both orders (blocked) | ☐ |
| SB4b paged-per-segment | — | gated on SB4a numbers | ☐ |
| SB5 real GEMM shards | **COMPLETE**: per-rank device GEMM shards; shard-residency cache (transposed operands pinned per rank device keyed layer/ordinal/slice); zero-copy activations; split_counts() remainder fix; device-side fan-in (cross-ordinal routed scratch + pairwise device adds) + device column gather (per-row bounce copies, RCCL-independent); per-rank handles satisfied by per-ordinal caches under pins; descriptor-linked fan-in proven via opcode-1 GEMMs + opcode-7 ADD through ScytheRingExec (gate `ring_row_parallel_descriptor_fanin_parity`, max-abs-diff 2.98e-7 at decode shapes). Parity on gfx1201+gfx1200: col 4.9e-7 / row 3.6e-7 / cached-reuse 4.9e-7. Occupancy read-out substitute recorded via rocprofv3 (rocprof-compute not installed) | — | ☑ |
| SB6 persistent ring | **RESIDENT-WAVE FIXED**: bounded s_sleep backoff in empty-queue spin (root cause: unthrottled atomic busy-poll starved the single workgroup after idle gaps on RDNA4+ROCm7.2; NOT JIT flake, NOT head race — both disproven). Verified clean-boot: phase0=2/A=4/B=6 across idle gaps, parity 2.384e-7, shutdown clean; 3/3 ring suite + opcode-6 gate green. Eternal-kernel coexistence rules documented (no blocking H2D/DtoH/pinned-alloc against live wave) | **COMPLETE** (2026-08-25): production F32 matmul routing behind `GRIM_SCYTHE_RING=1` (`device::scythe_route` bounded-wave channel; parity gate EXACT vs rocBLAS), ring-vs-direct decode benchmark run (`ring_vs_direct_decode` example) — verdict **no win at current kernel quality** (169×–91000× slower; single-workgroup reference GEMM + per-op publish sync), gate stays opt-in. Bonus Tier-0 find: split-K F32 corruption fixed (see AUDIT ledger row) | ☑ |

### Validation log

- 2026-08-22 (this box = syd-beasty): topology captured to
  `docs/benchmarks/syd_beasy_topology.json` (gpu1201 + gfx1200 + gfx1036 APU;
  APU excluded from farm ranks). Engine full suites green in BOTH
  `GRIM_GPUS=0,1` and `GRIM_GPUS=1,0` orders.
- **Blocker:** every GGUF model (LFM2.5-VL-3B, LFM2.5-230M, Mellum2 MoE)
  faults the GPU ("Memory access fault … Page not present") during the first
  prefill — arm=off too, so unrelated to SCYTHE-2. Raw kernels are healthy on
  this box (WMMA GEMM gate passes; persistent-dispatch launches pass), which
  bounds the fault to the GGUF dequant→device-upload path. Until fixed,
  end-to-end TTFT/ITL legs cannot run and WI-INF4 stays undecided.
- 2026-08-23 fault hunt (LFM2.5-230M, arm=off, caches purged each run):
  bracketed the fault to immediately after `grim_embedding` completes and
  before `grim_rms_norm`'s launcher entry, on the first prefill (204-token
  prompt; load itself now completes). Systematically ELIMINATED: stale HSACO
  (toolchain-versioned cache keys + purge), DeviceGuard context leak (impl
  verified save/restore), allocator pool reuse (`GRIM_ALLOC_NO_POOL=1` still
  faults), rocwmma module poisoning (`GRIM_DISABLE_ROCWMA_KERNELS=1` still
  faults), XNACK retry mode (`HSA_XNACK=1` no change), and standalone bugs in
  transpose/embedding/rms_norm at engine shapes (all pass isolated). Also
  found & fixed en route: a genuine WMMA-kernel correctness bug in
  grim_qkv_attention (vector-as-matrix fragments → zero accumulator;
  replaced with a 16Q×16K scalar-tile kernel, see qkv_attention.rs) and
  confirmed rocwmma f32 fragments return an empty accumulator under
  hipRTC/gfx1201. Diagnostic hooks kept behind env gates for the next
  session: GRIM_ALLOC_TRACE (upload/jit/launch trace + post-launch done
  marker), GRIM_ALLOC_NO_POOL, GRIM_DISABLE_ROCWMA_KERNELS,
  GRIM_Q4K_REPRO (minimal fused-q4k launch probe).
- **2026-08-23b — BLOCKER RESOLVED.** The "GGUF forward fault" was TWO bugs,
  both fixed and verified on-box this session:
  1. *HIP context drift* (see `gguf_multigpu_context_plan.md`, M1–M4 all
     closed): kernels launched under a foreign device context after raw
     seams let the thread park elsewhere. All seams pinned; trace rerun
     shows 0 `ctx_dev≠self_dev` lines and 0 page faults across
     `{ROCR=0}/{ROCR=0,1}/all-visible`.
  2. *Dequant scale-byte corruption*: Q4_K m-line read `scales[s-4]`
     instead of upstream ggml's `scales[s]` (commit 9801d50), corrupting
     sub-block minima; q2k/q3k fused-GEMM sources were additionally gated
     out of default builds so default GGUF loads crashed with
     hipModuleGetFunction error 500.
  Residual latent defects surfaced by the newly-runnable end-to-end paths:
  HIP attribute constants used CUDA numbering (throttle attr actually read
  shared-memory bytes ⇒ every GPU's effective TFLOPS was zeroed — explains
  the historical `throttle_pct: 1.0` in this file's jsonl); LFM2 fused-KV
  scratch capped at 4096 positions against a 128k-context model (decode
  past 4096 panicked); farm replica load died on an unpinned cached event
  (`hipEventRecord 400`). All fixed; details in
  `gguf_multigpu_context_plan.md` validation log.
- 2026-08-23c — WI-INF4 A/B verdict (release build, interleaved rounds, 30
  samples per arm per order; full rows in the sibling jsonl). The fault-hunt
  NEXT LEADS from the previous entry are all retired: the HSA-level dump was
  superseded by the M2 named-frame trace (flips attributed), and the
  embed→attn_norm window audit is moot — the fault was context drift plus
  dequant corruption, both fixed (see 2026-08-23b).

  | order | mean TTFT overhead | p95 ITL overhead |
  |-------|--------------------|------------------|
  | F-first | −0.09 % (PASS ≤ 5 %) | −18.56 % (PASS ≤ 2 %) |
  | S-first | −0.00 % (PASS ≤ 5 %) | **+2.43 % (FAIL > 2 %)** |

  ⇒ **STAYS OPT-IN** per the §SB3 rule: the S-first ITL tail exceeds budget,
  so `GRIM_SCYTHE_INFERENCE` keeps its opt-in default and the controller's
  cost model gets retuned with these numbers. Absolute latencies are
  prefill-dominated (~35 s mean TTFT on the 8k-heavy mix — the classic KV
  path is PCIe-download-bound), so the controller's per-decision cost is
  invisible in TTFT; the ITL delta concentrates in one ordinal order's
  decode tail. Measurement defect to fix in the harness:
  `scythe_ab::parse_samples` applies no time/commit filter, so its
  cumulative report mixed stale fault-era rows (it reported ITL Δ=28.5%);
  the ts-filtered computation above is authoritative.
- 2026-08-23d — rank-1 idle finding + fix. Farm admissions pinned every
  request to rank 0 on the serve surface (and a desktop game maxing GPU 0
  did not steer traffic either): the shape-keyed PlacementCache is
  load-blind, pins released instantly (EOS within ~2 tokens), and external
  GPU utilization was never consulted. Fixed by `C2plrController::
  decide_forced` (cache bypass whenever any rank carries load), a 1 s pin
  cooldown after `finish_request`, and rsmi busy-% folded into the admission
  load vector at weight 2.0. Unit gates green (`test_external_busy_flips_
  placement_to_idle_rank`, `test_scythe_effective_loads_weights_and_expiry`,
  `test_finished_pin_enters_cooldown_window`); serve-surface confirmation
  pending rebuild.
- 2026-08-23e — rank-1 serving crash + spread gate. With the steering live,
  the first-ever rank-1 pin page-faulted node-1 (`Page not present`) inside
  the server's `sample_on_device` path. Evidence: the entire forward ran
  clean on the replica (every launch `self_dev=1 ctx_dev=1`; WI-M1/M2
  discipline holding), and `GRIM_DEBUG_PROMPT=1` shows step-0 logits with
  correct shape (`len=65536 width=65536`) but **all-zero values** — the same
  signature as the known first-JIT-launch zeroing flake recorded in
  `gguf_multigpu_context_plan.md` (gfx1036 mxfp4 parity), now reproduced on
  gfx1200 where every launch in a fresh process is first-JIT. Next-session
  lead: root-cause first-launch zeroing (rocprofiler trace of a failing run
  per M4's toolchain note), then re-enable spreading. RESOLVED 2026-08-23f: root cause was the unpinned rocBLAS dispatch in
`matmul_op`/`matmul_with_solution` (rocBLAS runs on the calling thread's
CURRENT device; context-neutral try_new leaves it on ordinal 0). Full
P1-3 guard sweep across 26 functions + p2p_route HostBounce per-leg pins +
copy_slice_into cross-ordinal fail-loud; enforcement lint added to
hip_context_contract (mutation-checked). Verification: minimal repro 3/3
correct on ordinal 1; replica1 logits bit-identical to control0; drift
gates green both orders; live serve under game-loaded GPU0 spread
18/17 ranks with GPU1 sampled at 88–93 %. GRIM_SCYTHE_SPREAD now defaults
ON (opt-out =0).

## Risks

- **Tie-collapse regression risk lives in SB0**: if measured per-card ratios
  are closer than the controller's noise floor, placement gains vanish — the
  A/B harness will show it as on≈off, which is itself the finding; record and
  stop rather than tuning toward a predetermined answer.
- Thermal drift between A/B legs skews TTFT: interleave on/off iterations
  rather than batching them; profiler throttle correction helps but is not
  proof — the harness records throttle_pct alongside every sample.
- All changes stay behind `GRIM_SCYTHE_INFERENCE`; rollback invariant
  (§setup-5) re-verified per WI keeps default-off credible until the verdict
  says otherwise.

### Validation log addendum — 2026-08-25 (SB6 closeout + audit fixes)

- **SB6 COMPLETE.** Production layer routing: `GRIM_SCYTHE_RING=1` reroutes
  F32 `matmul_op` GEMMs through a per-ordinal bounded-wave ScytheRing channel
  (`grim-backend-rocm/src/device/scythe_route.rs`). Parity gate
  `production_ring_routing_matmul_parity`: EXACT match vs direct rocBLAS at
  4×64×96 (0.0 diff). Ring-vs-direct decode benchmark
  (`examples/ring_vs_direct_decode.rs`, release, gfx1201, 30 iters/shape):
  ring is 169× (1×576×576) to 91048× (1×12288×4096) slower than direct —
  the single-workgroup reference GEMM arm plus the per-op head-publish
  stream sync dominate. **Verdict: no measured win; the gate stays opt-in.**
  Competitive routing needs a multi-CU wave fan-out and/or resident mode to
  amortize the publish sync.
- **Audit fixes landed the same pass** (scythe-audit-fix-plan.md): F0
  re-verified (5/5 ring suite incl. resident wave after kernel edits),
  F8/F10 kvtransport pull-mode deadlock fixed (trait read methods + server
  FETCH branch + client timeouts + disagg loopback gate), F9 scheduler
  chunk accumulation fixed, F2 attention head_dim guard + error-path tail
  advance, F3/F4 MoE descriptor device upload + three-pointer schedule
  (e2e public-API gate at 2.3e-10), F6 dead softmax removed, F7 per-layer
  bucket cache, F5 published_head removed.
- **Bonus Tier-0 find**: `grim_split_k_reduction` was hard-typed `_Float16*`
  — every F32 split-K matmul (m>1 or k>8192) silently corrupted its output
  (~1e3-scale errors; e.g. 4×4096×4096, 1×4096×12288 were garbage on main).
  Found by the new benchmark comparing ring vs direct against expectations.
  Fixed with dtype-dispatched f32/bf16 reduction kernels; regression gate
  `tests/split_k_matmul_parity.rs` (≤2.5e-7 vs CPU at all trigger shapes).
