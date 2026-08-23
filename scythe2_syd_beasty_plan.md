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
| SB0 differentiation | ☐ | ☐ both orders | ☐ |
| SB1 CLI farm | ☐ | ☐ | ☐ |
| SB2 VRAM guard | ☐ | ☐ both orders | ☐ |
| SB3 A/B harness | ☐ | ☐ full matrix | ☐ |
| WI-INF4 verdict | — | ☐ flag decision recorded | ☐ |
| SB4a layer pipeline | ☐ | ☐ TTFT both orders | ☐ |
| SB4b paged-per-segment | — | gated on SB4a | ☐ |
| SB5 real GEMM shards | — | gated on SB4 | ☐ |
| SB6 persistent ring | — | gated on SB5 | ☐ |

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
